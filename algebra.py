"""Algebra utilities: Hermitian projections, commutators, and cached maps."""

from __future__ import annotations

import torch
from typing import Optional
import numpy as np

try:
    from MatrixModelHMC_pytorch import config
except ImportError:  # pragma: no cover
    import config  # type: ignore

# Caches keyed by (size, device, dtype) to avoid repeated allocations.
_traceless_cache: dict[tuple[int, str, Optional[int], torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
_eye_cache: dict[tuple[int, str, Optional[int], torch.dtype], torch.Tensor] = {}


def dagger(a: torch.Tensor) -> torch.Tensor:
    """Hermitian conjugate."""
    return a.transpose(-1, -2).conj()


def comm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix commutator [A, B]."""
    return A @ B - B @ A


def random_hermitian(n: int, *, traceless: bool = True) -> torch.Tensor:
    """
    Draw a random Hermitian n x n matrix.

    Set ``traceless=False`` to retain the identity mode.
    """
    re = torch.randn(n, n, device=config.device, dtype=config.real_dtype)
    im = torch.randn(n, n, device=config.device, dtype=config.real_dtype)

    mat = torch.zeros(n, n, dtype=config.dtype, device=config.device)

    iu, ju = torch.triu_indices(n, n, offset=1)
    vals = (re[iu, ju] + 1j * im[iu, ju]) / (2.0**0.5)
    mat[iu, ju] = vals
    mat[ju, iu] = vals.conj()

    diag_re = torch.randn(n, device=config.device, dtype=config.real_dtype)
    idx = torch.arange(n, device=config.device)
    mat[idx, idx] = diag_re.to(config.dtype)

    if traceless:
        mat = mat - (torch.trace(mat) / n) * torch.eye(n, dtype=config.dtype, device=config.device)
    return mat


def kron_2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Kronecker product of 2D tensors A (m*n) and B (p*q):

        kron(A, B) has shape (m*p, n*q)
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"kron_2d expects 2D tensors, got {A.shape}, {B.shape}")

    return torch.kron(A, B)


def _kron_batched(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batched Kronecker product supporting broadcasting over leading dims."""
    kron = torch.einsum("...ij,...kl->...ikjl", A, B)
    *batch, m, k, n, l = kron.shape
    return kron.reshape(*batch, m * k, n * l)


def make_traceless_maps(N: int, device=None, dtype=None):
    """
    Build linear maps Q and S for the traceless subspace:

      vec(A) = Q v,   A traceless, v ∈ C^{N^2-1}
      v      = S vec(A), for traceless A.
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.complex64

    N2 = N * N
    Dtr = N2 - 1

    Q = torch.zeros(N2, Dtr, device=device, dtype=dtype)
    S = torch.zeros(Dtr, N2, device=device, dtype=dtype)

    def idx(i, j):
        return i + j * N

    coords = []
    for j in range(N):
        for i in range(N):
            if not (i == N - 1 and j == N - 1):
                coords.append((i, j))
    last_diag_row = idx(N - 1, N - 1)

    for col, (i, j) in enumerate(coords):
        row = idx(i, j)

        Q[row, col] = 1.0
        S[col, row] = 1.0

        if i == j and i < N - 1:
            Q[last_diag_row, col] -= 1.0

    return Q, S


def get_traceless_maps_cached(N: int, device: torch.device, dtype: torch.dtype):
    """Cache Q,S per (N, device, dtype) to avoid rebuilding every call."""
    key = (N, device.type, device.index, dtype)
    if key not in _traceless_cache:
        _traceless_cache[key] = make_traceless_maps(N, device=device, dtype=dtype)
    return _traceless_cache[key]


_projector_cache: dict[tuple[int, str, Optional[int], torch.dtype], torch.Tensor] = {}
_trace_diag_indices_cache: dict[tuple[int, str, Optional[int]], torch.Tensor] = {}


def get_trace_diag_indices_cached(N: int, device: torch.device) -> torch.Tensor:
    """Cache flattened diagonal indices for vec(I) in column-major ordering."""
    key = (N, device.type, device.index)
    diag_indices = _trace_diag_indices_cache.get(key)
    if diag_indices is None:
        diag_indices = torch.arange(0, N * N, N + 1, device=device, dtype=torch.long)
        _trace_diag_indices_cache[key] = diag_indices
    return diag_indices


def add_trace_projector_inplace(block: torch.Tensor, N: int) -> None:
    """
    Add P = (1/N) |I><I| to an (N^2 x N^2) block without materializing dense P.

    This lifts the trace-mode zero mode while touching only the N x N trace sub-block.
    """
    diag_indices = get_trace_diag_indices_cached(N, block.device)
    block[diag_indices.unsqueeze(-1), diag_indices] += block.new_tensor(1.0 / N)


def get_trace_projector_cached(N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Get cached projector onto the trace mode (identity matrix direction).
    Returns P such that P @ vec(I) = vec(I) and P @ vec(A) = 0 for traceless A.
    """
    key = (N, device.type, device.index, dtype)
    if key not in _projector_cache:
        P = torch.zeros((N * N, N * N), device=device, dtype=dtype)
        # Indices of diagonal elements in flattened array: 0, N+1, 2N+2, ...
        diag_indices = torch.arange(0, N * N, N + 1, device=device)
        
        # P = (1/N) * |I><I|
        # |I> has 1s at diag_indices.
        # P[i, j] = 1/N if i,j in diag_indices
        val = torch.tensor(1.0 / N, device=device, dtype=dtype)
        P[diag_indices.unsqueeze(-1), diag_indices] = val
        _projector_cache[key] = P
    return _projector_cache[key]


def get_eye_cached(n: int, device: torch.device, dtype: torch.dtype):
    """Cache identity matrices per (n, device, dtype)."""
    key = (n, device.type, device.index, dtype)
    eye = _eye_cache.get(key)
    if eye is None:
        eye = torch.eye(n, device=device, dtype=dtype)
        _eye_cache[key] = eye
    return eye


def ad_matrix(X: torch.Tensor) -> torch.Tensor:
    """
    Adjoint action ad_X on the full matrix space M_N(C).

    X: (..., N, N) with optional batch dimension.
    Returns:
        (..., N^2, N^2) such that
        v' = ad_X v corresponds to [X, A] in the vectorized basis.
    """
    if X.ndim not in (2, 3):
        raise ValueError(f"ad_matrix expects X with shape (N,N) or (B,N,N), got {X.shape}")

    single = X.ndim == 2
    if single:
        X = X.unsqueeze(0)

    B, N, _ = X.shape
    dev = X.device
    dtp = X.dtype

    I = get_eye_cached(N, device=dev, dtype=dtp)
    Xt = X.transpose(-1, -2)

    # ad_X = I ⊗ X - X^T ⊗ I
    kron_eye = _kron_batched(I.view(1, N, N), X)
    kron_xt = _kron_batched(Xt, I.view(1, N, N))
    result = kron_eye - kron_xt

    if single:
        result = result.squeeze(0)
    return result


def makeH(mat: torch.Tensor) -> torch.Tensor:
    """Project a matrix (or batch of matrices) to its Hermitian part."""
    return 0.5 * (mat + dagger(mat))


def spinJMatrices(j_val: float):
    """Generate spin-j angular momentum matrices Jx, Jy, Jz on CPU with NumPy."""
    dim = int(round(2 * j_val + 1))

    Jp = np.zeros((dim, dim), dtype=np.complex128)

    # Physical m-values in descending order: j, j-1, ..., -j
    m_vals = np.arange(j_val, -j_val - 1, -1, dtype=np.float64)

    # Ladder operator: J+ |m> = sqrt(j(j+1) - m(m+1)) |m+1>
    # In descending order, raising moves one index up (row = col-1).
    for col in range(1, dim):
        m = m_vals[col]
        Jp[col - 1, col] = np.sqrt(j_val * (j_val + 1) - m * (m + 1))

    Jm = Jp.conj().T

    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5j * (Jp - Jm)
    Jz = np.diag(m_vals)

    assert np.allclose(Jx @ Jy - Jy @ Jx, 1j * Jz, atol=1e-7)
    assert np.allclose(Jy @ Jz - Jz @ Jy, 1j * Jx, atol=1e-7)
    assert np.allclose(Jz @ Jx - Jx @ Jz, 1j * Jy, atol=1e-7)

    return np.stack([Jx, Jy, Jz], axis=0)


__all__ = [
    "add_trace_projector_inplace",
    "ad_matrix",
    "comm",
    "get_eye_cached",
    "get_trace_diag_indices_cached",
    "get_trace_projector_cached",
    "kron_2d",
    "makeH",
    "random_hermitian",
    "spinJMatrices",
]
