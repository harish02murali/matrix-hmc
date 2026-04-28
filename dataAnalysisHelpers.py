import json
import os
from pathlib import Path

import numpy as np
import torch

def _default_data_path() -> Path:
    return Path(os.environ.get("MATRIX_HMC_DATA") or Path.cwd())


DATA_PATH = _default_data_path()

class RunRecord:
    """Lightweight container for a single simulation run."""

    def __init__(self, run: str | Path, *, base_path: Path | str | None = None, load_checkpoint: bool = True) -> None:
        base = Path(base_path) if base_path is not None else _default_data_path()
        path = Path(run)
        if not path.is_dir():
            path = base / run
        if not path.exists():
            raise FileNotFoundError(f"Run directory '{run}' not found under {base}")
        self.path = path
        self.metadata = self._load_json(path / "metadata.json")
        self.evals = self._load_npz(path / "evals.npz", empty=np.empty((0, 0, 0), dtype=np.complex128))
        self.corrs = self._load_npz(path / "corrs.npz", empty=np.empty((0, 0), dtype=np.complex128))
        self.mats = self._load_chunks(path / "all_mats")
        model_info = self.metadata.get("model", {})
        self.model_name = model_info.get("model_name")
        self.nmat = model_info.get("nmat")
        self.ncol = model_info.get("ncol") or (self.evals.shape[-1] if self.evals.ndim == 3 else None)
        self.couplings = model_info.get("couplings", [])
        self.g = self.couplings[0] if self.couplings else None
        self.omega = self.couplings[1] if len(self.couplings) > 1 else None
        self.X: np.ndarray | None = None
        if load_checkpoint:
            self.load_checkpoint()

    @staticmethod
    def _load_npz(path: Path, *, empty: np.ndarray) -> np.ndarray:
        if not path.exists():
            return empty
        with np.load(path) as data:
            return data["values"]

    @staticmethod
    def _load_chunks(path: Path):
        if not path.exists():
            return None
        files = sorted(f for f in path.glob("*.npy") if f.is_file())
        if not files:
            return None
        return [np.load(f, mmap_mode="r", allow_pickle=False) for f in files]

    @staticmethod
    def _load_json(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_checkpoint(self) -> None:
        ckpt_path = self.path / "checkpoint.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.X = state["X"].to(dtype=torch.complex128).cpu().numpy()

    def ensure_checkpoint(self) -> None:
        if self.X is None:
            self.load_checkpoint()

    @property
    def n_measurements(self) -> int:
        return self.evals.shape[0] if self.evals.ndim else 0

    def iter_mats(self):
        if self.mats is None:
            return
        for chunk in self.mats:
            for mat in chunk:
                yield mat

    def get_mat(self, idx: int):
        if self.mats is None:
            return None
        for chunk in self.mats:
            if idx < chunk.shape[0]:
                return chunk[idx]
            idx -= chunk.shape[0]
        raise IndexError("idx out of range")


def jackknife_error(values, window_size: int = 1):
    """Return mean and jackknife 1-sigma error for 1D samples with optional binning.

    For complex input the returned error is complex: its real and imaginary parts
    are the jackknife errors on the real and imaginary components respectively.
    """
    raw = np.asarray(values).ravel()
    is_complex = np.iscomplexobj(raw)
    data = raw.astype(np.complex128 if is_complex else np.float64)

    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    if window_size > 1:
        n_bins = data.size // window_size
        if n_bins < 2:
            raise ValueError("Not enough data for the requested window_size.")
        data = data[: n_bins * window_size].reshape(n_bins, window_size).mean(axis=1)

    n = data.size
    if n < 2:
        raise ValueError("Jackknife requires at least two samples after windowing.")

    stat = data.mean()
    jk_samples = (stat * n - data) / (n - 1)
    delta = jk_samples - jk_samples.mean()

    if is_complex:
        error = (
            np.sqrt((n - 1) * np.mean(delta.real ** 2))
            + 1j * np.sqrt((n - 1) * np.mean(delta.imag ** 2))
        )
    else:
        error = np.sqrt((n - 1) * np.mean(delta ** 2))

    return stat, error

def standardize(values):
    data = np.asarray(np.real(values), dtype=np.float64).ravel()
    mean = data.mean()
    std = data.std()
    if std == 0:
        return data - mean
    return (data - mean) / std
