"""Shared configuration: device selection, dtypes, and host runtime knobs."""

from __future__ import annotations

import os
import torch

# Ensure Apple GPUs fall back cleanly to CPU for unsupported ops.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_positive_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


ALLOW_TF32 = _parse_bool_env("IKKT_ALLOW_TF32", True)
ENABLE_TORCH_COMPILE = _parse_bool_env("IKKT_COMPILE", False)
CPU_NUM_THREADS = _parse_positive_int_env("IKKT_NUM_THREADS")
CPU_NUM_INTEROP_THREADS = _parse_positive_int_env("IKKT_NUM_INTEROP_THREADS")

def _real_dtype_for(complex_dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if complex_dtype == torch.complex64 else torch.float64


# Default to double-precision complex unless overridden via CLI.
dtype = torch.complex128
real_dtype = _real_dtype_for(dtype)


def _enable_tf32() -> None:
    if not ALLOW_TF32:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")


def configure_torch_compile(enabled: bool | None = None) -> bool:
    """Set the global torch.compile toggle used by model constructors."""
    global ENABLE_TORCH_COMPILE
    if enabled is None:
        return ENABLE_TORCH_COMPILE
    ENABLE_TORCH_COMPILE = bool(enabled)
    return ENABLE_TORCH_COMPILE


def configure_threads(
    num_threads: int | None = None,
    num_interop_threads: int | None = None,
) -> tuple[int, int]:
    """Apply explicit CPU threading policy for PyTorch."""
    global CPU_NUM_THREADS, CPU_NUM_INTEROP_THREADS

    resolved_threads = CPU_NUM_THREADS if num_threads is None else num_threads
    resolved_interop = CPU_NUM_INTEROP_THREADS if num_interop_threads is None else num_interop_threads

    if resolved_threads is not None:
        if resolved_threads < 1:
            raise ValueError("num_threads must be positive")
        torch.set_num_threads(resolved_threads)
    else:
        resolved_threads = torch.get_num_threads()

    if resolved_interop is not None:
        if resolved_interop < 1:
            raise ValueError("num_interop_threads must be positive")
        try:
            torch.set_num_interop_threads(resolved_interop)
        except RuntimeError:
            resolved_interop = torch.get_num_interop_threads()
    else:
        resolved_interop = torch.get_num_interop_threads()

    CPU_NUM_THREADS = resolved_threads
    CPU_NUM_INTEROP_THREADS = resolved_interop
    return resolved_threads, resolved_interop


def configure_device(noGPUFlag: bool | None) -> torch.device:
    """Select device from CLI preference."""
    
    requested = "cpu" if noGPUFlag else "auto"
    dev: torch.device
    if requested == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            _enable_tf32()
            print(f"Using CUDA device: {torch.cuda.get_device_name(dev)}")
        else:
            dev = torch.device("cpu")
            print("GPU not available, using CPU.")
    else:
        dev = torch.device("cpu")
        print("Using CPU.")

    global device
    device = dev
    return dev


def configure_dtype(use_complex64: bool) -> torch.dtype:
    """Set the global complex dtype used for matrices."""
    global dtype, real_dtype
    dtype = torch.complex64 if use_complex64 else torch.complex128
    real_dtype = _real_dtype_for(dtype)
    return dtype


# Default to CPU until configure_device is called.
device = torch.device("cpu")


__all__ = [
    "ALLOW_TF32",
    "CPU_NUM_INTEROP_THREADS",
    "CPU_NUM_THREADS",
    "ENABLE_TORCH_COMPILE",
    "configure_threads",
    "configure_torch_compile",
    "configure_device",
    "configure_dtype",
    "device",
    "dtype",
    "real_dtype",
]
