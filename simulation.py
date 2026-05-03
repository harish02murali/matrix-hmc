"""HMC simulation runner — the main public entry point for library use."""

from __future__ import annotations

import cProfile
import datetime
import importlib
import importlib.util
import json
import os
import pstats
import shutil
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from matrix_hmc import config as _config
from matrix_hmc.hmc import HMCParams, thermalize, update
from matrix_hmc.models.base import MatrixModel

_BUILTIN_MODEL_DIR = Path(__file__).resolve().parent / "models"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model_module(model_name_or_path: str):
    """Return a model module by built-in name or file path.

    If the argument contains a path separator or ends in .py it is loaded
    directly from disk; otherwise it is looked up in the built-in models dir.
    """
    is_path = "/" in model_name_or_path or os.sep in model_name_or_path or model_name_or_path.endswith(".py")

    if is_path:
        p = Path(model_name_or_path)
        if not p.exists():
            raise ValueError(f"Model file not found: {p}")
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    if (_BUILTIN_MODEL_DIR / f"{model_name_or_path}.py").exists():
        return importlib.import_module(f"matrix_hmc.models.{model_name_or_path}")

    known = sorted(p.stem for p in _BUILTIN_MODEL_DIR.glob("*.py") if not p.stem.startswith("_"))
    raise ValueError(
        f"Unknown model '{model_name_or_path}'. Built-in models: {known}. "
        "To use a custom model, pass the path: --model ./my_model.py"
    )


# ---------------------------------------------------------------------------
# I/O helpers (intentionally private — details that callers shouldn't see)
# ---------------------------------------------------------------------------

def _ensure_output_slots(paths: Iterable[str], *, force: bool, allow_existing: bool) -> None:
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not (force or allow_existing):
        raise FileExistsError(
            "Output files already exist:\n" + "\n".join(existing) +
            "\nUse force=True to overwrite or resume=True to append."
        )
    if force:
        for path in existing:
            os.remove(path)


def _prepare_matrix_snapshot_dir(run_dir: str, *, force: bool, allow_existing: bool) -> tuple[str, int]:
    target = os.path.join(run_dir, "all_mats")
    if os.path.exists(target):
        if force:
            shutil.rmtree(target)
        elif not allow_existing:
            raise FileExistsError(
                f"Matrix snapshot directory {target} already exists. "
                "Use force=True or resume=True."
            )
    os.makedirs(target, exist_ok=True)
    max_end = 0
    if allow_existing:
        for name in os.listdir(target):
            if not name.endswith(".npy") or "_" not in name[:-4]:
                continue
            start_str, end_str = name[:-4].split("_", 1)
            if start_str.isdigit() and end_str.isdigit():
                max_end = max(max_end, int(end_str))
    return target, max_end


def _append_npz(path: str, values: np.ndarray, *, key: str, dtype: np.dtype) -> None:
    if values.size == 0:
        return
    new_values = np.asarray(values, dtype=dtype)
    if os.path.exists(path):
        with np.load(path) as existing:
            new_values = np.concatenate((existing[key], new_values), axis=0)
    np.savez(path, **{key: new_values})


def _flush_buffers(ev_buf: list, corr_buf: list, paths: dict[str, str]) -> None:
    if ev_buf:
        _append_npz(paths["eigs"], np.stack(ev_buf).astype(np.complex128),
                    key="values", dtype=np.complex128)
        ev_buf.clear()
    if corr_buf:
        _append_npz(paths["corrs"], np.stack(corr_buf).astype(np.complex128),
                    key="values", dtype=np.complex128)
        corr_buf.clear()


def _create_snapshot_chunk(
    directory: str, *, start: int, count: int, dtype: np.dtype, shape_tail: tuple
) -> np.memmap:
    end = start + count - 1
    filename = os.path.join(directory, f"{start:08d}_{end:08d}.npy")
    return np.lib.format.open_memmap(filename, mode="w+", dtype=dtype, shape=(count, *shape_tail))


def _write_metadata(path: str, model: MatrixModel, **run_kwargs) -> None:
    summary = {
        "model": model.run_metadata(),
        "run": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                for k, v in run_kwargs.items()},
        "runtime": {
            "device": str(_config.device),
            "dtype": str(_config.dtype),
            "num_threads": _config.CPU_NUM_THREADS,
            "num_interop_threads": _config.CPU_NUM_INTEROP_THREADS,
            "timestamp": datetime.datetime.now().isoformat(),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _maybe_profile(enabled: bool) -> cProfile.Profile | None:
    if not enabled:
        return None
    p = cProfile.Profile()
    p.enable()
    return p


def _stop_profile(profiler: cProfile.Profile | None) -> None:
    if profiler is None:
        return
    profiler.disable()
    ps = pstats.Stats(profiler)
    ps.strip_dirs().sort_stats(pstats.SortKey.TIME)
    ps.print_stats(10)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    model: MatrixModel,
    *,
    niters: int = 100,
    step_size: float = 0.5,
    nsteps: int = 50,
    output: str | Path = "data",
    name: str = "run",
    save_every: int = 10,
    save_checkpoints: bool = True,
    save_matrices: bool = False,
    resume: bool = False,
    force: bool = False,
    seed: int | None = None,
    profile: bool = False,
    dry_run: bool = False,
) -> MatrixModel:
    """Run HMC trajectories for *model* and write observables to *output/name_.../*.

    Parameters
    ----------
    model:            A MatrixModel instance (already constructed with all params).
    niters:           Number of HMC trajectories.
    step_size:        Total leapfrog trajectory length.
    nsteps:           Leapfrog steps per trajectory (dt = step_size / nsteps).
    output:           Root directory for output files.
    name:             Prefix for the run subdirectory.
    save_every:       Flush observables (and checkpoint if save_checkpoints) every K steps.
    save_checkpoints: Write a checkpoint .pt file every save_every steps.
    save_matrices:    Also write raw matrix snapshots.
    resume:           Append to existing output files and load checkpoint if present.
    force:            Overwrite existing output files without error.
    seed:             RNG seed for reproducibility.
    profile:          Enable cProfile and print top-10 hotspots at the end.
    dry_run:          Print configuration and return without running.
    """
    hmc_params = HMCParams(dt=step_size / nsteps, nsteps=nsteps)

    paths = model.build_paths(name, str(output))
    os.makedirs(paths["dir"], exist_ok=True)

    _ensure_output_slots([paths["eigs"], paths["corrs"]], force=force, allow_existing=resume)
    _write_metadata(paths["meta"], model, niters=niters, step_size=step_size,
                    nsteps=nsteps, output=str(output), name=name)

    print("\n" + "=" * 52)
    print("  Matrix Model HMC — run configuration")
    print("=" * 52)
    print(f"  Model          {model.model_name}")
    print(f"  Matrix size N  {model.ncol}")
    for line in model.extra_config_lines():
        print(line)
    print(f"  Step size      {step_size}  ({nsteps} steps, dt = {step_size/nsteps:.4g})")
    print(f"  Device         {_config.device}  [{_config.dtype}]")
    print(f"  CPU threads    {_config.CPU_NUM_THREADS}/{_config.CPU_NUM_INTEROP_THREADS} (intra/inter-op)")
    print(f"  Checkpoint     {'every ' + str(save_every) + ' steps' if save_checkpoints else 'disabled'}")
    print(f"  Output dir     {paths['dir']}")
    print("=" * 52 + "\n")

    if dry_run:
        return model

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    profiler = _maybe_profile(profile)
    resumed = model.initialize_configuration(paths["ckpt"], resume=resume)

    if not resumed:
        _ensure_output_slots([paths["eigs"], paths["corrs"]], force=True, allow_existing=False)
        thermalize(model, hmc_params, steps=10)

    acc_count = 0
    ev_buf: list[np.ndarray] = []
    corr_buf: list[np.ndarray] = []
    snapshot_dir = None
    snapshot_offset = 0
    chunk: np.memmap | None = None

    if save_matrices:
        snapshot_dir, snapshot_offset = _prepare_matrix_snapshot_dir(
            paths["dir"], force=force, allow_existing=resume
        )
        state_shape = model.get_state().shape
        state_dtype = np.dtype(model.get_state().detach().cpu().numpy().dtype)

    for i in range(1, niters + 1):
        acc_count = update(acc_count, hmc_params, model)

        if snapshot_dir is not None:
            global_i = snapshot_offset + i
            if chunk is None or (global_i - 1) % save_every == 0:
                if chunk is not None:
                    chunk.flush()
                remaining = niters - i + 1
                chunk = _create_snapshot_chunk(
                    snapshot_dir, start=global_i,
                    count=min(save_every, remaining),
                    dtype=state_dtype, shape_tail=state_shape,
                )
            state = model.get_state().detach()
            if state.device.type != "cpu":
                state = state.to("cpu")
            chunk[global_i - (snapshot_offset + (i - 1) // save_every * save_every) - 1] = state.numpy()

        eigs, corrs = model.measure_observables()
        ev_buf.append(np.stack(eigs))
        if corrs is not None:
            corr_buf.append(corrs)

        if i % save_every == 0:
            _flush_buffers(ev_buf, corr_buf, paths)
            if chunk is not None:
                chunk.flush()
            print(f"Iteration {i}, acceptance = {acc_count/i:.3f}, " + model.status_string())
            if save_checkpoints:
                model.save_state(paths["ckpt"])

    _flush_buffers(ev_buf, corr_buf, paths)
    if chunk is not None:
        chunk.flush()

    if acc_count / max(niters, 1) < 0.5:
        print("WARNING: Acceptance rate is below 50%")

    _stop_profile(profiler)
    return model


__all__ = ["run", "_load_model_module"]
