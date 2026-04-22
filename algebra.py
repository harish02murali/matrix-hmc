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
_hermitian_basis_index_cache: dict[tuple[int, str, Optional[int]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_hermitian_diag_basis_change_cache: dict[tuple[int, str, Optional[int], torch.dtype], torch.Tensor] = {}


def dagger(a: torch.Tensor) -> torch.Tensor:
    """Hermitian conjugate."""
    return a.transpose(-1, -2).conj()


def comm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix commutator [A, B]."""
    return A @ B - B @ A


def random_hermitian(
    n: int,
    *,
    traceless: bool = True,
    batchsize: int | None = None,
) -> torch.Tensor:
    """
    Draw a random Hermitian matrix.

    Set ``traceless=False`` to retain the identity mode.
    If ``batchsize`` is provided, return a batch of ``batchsize`` matrices.
    """
    shape = (batchsize, n, n) if batchsize is not None else (n, n)
    re = torch.randn(*shape, device=config.device, dtype=config.real_dtype)
    im = torch.randn(*shape, device=config.device, dtype=config.real_dtype)

    mat = torch.zeros(*shape, dtype=config.dtype, device=config.device)

    iu, ju = torch.triu_indices(n, n, offset=1)
    if batchsize is None:
        vals = (re[iu, ju] + 1j * im[iu, ju]) / (2.0**0.5)
        mat[iu, ju] = vals
        mat[ju, iu] = vals.conj()
    else:
        vals = (re[:, iu, ju] + 1j * im[:, iu, ju]) / (2.0**0.5)
        mat[:, iu, ju] = vals
        mat[:, ju, iu] = vals.conj()

    diag_re = torch.randn(*((batchsize, n) if batchsize is not None else (n,)), device=config.device, dtype=config.real_dtype)
    idx = torch.arange(n, device=config.device)
    if batchsize is None:
        mat[idx, idx] = diag_re.to(config.dtype)
    else:
        mat[:, idx, idx] = diag_re.to(config.dtype)

    if traceless:
        trace = torch.diagonal(mat, dim1=-2, dim2=-1).sum(-1)
        eye = torch.eye(n, dtype=config.dtype, device=config.device)
        mat = mat - (trace / n).unsqueeze(-1).unsqueeze(-1) * eye
    return mat


def kron_2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Kronecker product of 2D tensors A (m*n) and B (p*q):

        kron(A, B) has shape (m*p, n*q)
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"kron_2d expects 2D tensors, got {A.shape}, {B.shape}")

    return torch.kron(A, B)



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


def get_hermitian_basis_indices_cached(N: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache flat indices for diagonal and paired off-diagonal Hermitian basis elements."""
    key = (N, device.type, device.index)
    cached = _hermitian_basis_index_cache.get(key)
    if cached is None:
        diag = get_trace_diag_indices_cached(N, device)
        row_idx, col_idx = torch.triu_indices(N, N, offset=1, device=device)
        upper = row_idx + col_idx * N
        lower = col_idx + row_idx * N
        cached = (diag, upper.to(torch.long), lower.to(torch.long))
        _hermitian_basis_index_cache[key] = cached
    return cached


def get_hermitian_diag_basis_change_cached(N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Cache the orthogonal change from diagonal units to identity-plus-Cartan generators."""
    key = (N, device.type, device.index, dtype)
    change = _hermitian_diag_basis_change_cache.get(key)
    if change is None:
        change = torch.zeros((N, N), device=device, dtype=dtype)
        change[:, 0] = change.new_tensor(1.0 / (N ** 0.5))

        for a in range(N - 1):
            denom = ((a + 1) * (a + 2)) ** 0.5
            value = change.new_tensor(1.0 / denom)
            change[: a + 1, a + 1] = value
            change[a + 1, a + 1] = change.new_tensor(-(a + 1) / denom)

        _hermitian_diag_basis_change_cache[key] = change
    return change


def ad_matrix(X: torch.Tensor) -> torch.Tensor:
    """
    Adjoint action ad_X on the full matrix space M_N(C).

    X: (..., N, N) with optional batch dimension.
    Returns:
        (..., N^2, N^2) such that
        v' = ad_X v corresponds to [X, A] in the vectorized basis.

    Computed as T - σ(T) where T = I⊗X and σ is the cyclic index permutation
    (0,4,1,2,3) that maps T to X^T⊗I in the 4-tensor representation.  This
    makes the anti-symmetric structure manifest while requiring only one
    Kronecker-product evaluation.
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

    # T[b,i,k,j,l] = δ_{ij} X[b,k,l]  (I⊗X in 4-tensor form)
    # T.permute(0,4,1,2,3)[b,i,k,j,l] = δ_{kl} X[b,j,i]  (X^T⊗I)
    # ad_X = I⊗X - X^T⊗I
    T = torch.einsum("ij,...kl->...ikjl", I, X)
    result = (T - T.permute(0, 4, 1, 2, 3)).reshape(B, N * N, N * N)

    if single:
        result = result.squeeze(0)
    return result


def ad_matrix_real_antisymmetric(X: torch.Tensor, *, traceless: bool = False) -> torch.Tensor:
    """
    Represent ``i ad_X`` on the real vector space of Hermitian matrices.

    Basis order:
      1. ``I / sqrt(N)``
      2. the ``N-1`` diagonal Cartan generators
      3. ``(E_ij + E_ji) / sqrt(2)`` for ``i < j``
      4. ``i (E_ij - E_ji) / sqrt(2)`` for ``i < j``

    For Hermitian ``X``, the returned matrix is real and anti-symmetric.  The
    identity direction is the first basis vector, so ``traceless=True`` drops
    that row and column and returns the ``su(N)`` block directly.

    The implementation keeps the same asymptotic cost as ``ad_matrix``: it
    reuses the fast Kronecker-form construction for ``i ad_X`` and applies only
    sparse pair-combinations plus a dense ``N x N`` rotation in the diagonal
    sector, rather than a dense ``N^2 x N^2`` similarity transform.
    """
    if X.ndim not in (2, 3):
        raise ValueError(
            f"ad_matrix_real_antisymmetric expects X with shape (N,N) or (B,N,N), got {X.shape}"
        )

    single = X.ndim == 2
    if single:
        X = X.unsqueeze(0)

    _, N, _ = X.shape
    real_dtype = X.real.dtype if X.is_complex() else X.dtype

    # ``i ad_X`` preserves Hermitian matrices for Hermitian X.
    superop = 1j * ad_matrix(X)

    diag_idx, upper_idx, lower_idx = get_hermitian_basis_indices_cached(N, X.device)
    diag_rows = superop.index_select(-2, diag_idx)
    upper_rows = superop.index_select(-2, upper_idx)
    lower_rows = superop.index_select(-2, lower_idx)

    dd = diag_rows.index_select(-1, diag_idx)
    du = diag_rows.index_select(-1, upper_idx)
    dl = diag_rows.index_select(-1, lower_idx)
    ud = upper_rows.index_select(-1, diag_idx)
    uu = upper_rows.index_select(-1, upper_idx)
    ul = upper_rows.index_select(-1, lower_idx)
    ld = lower_rows.index_select(-1, diag_idx)
    lu = lower_rows.index_select(-1, upper_idx)
    ll = lower_rows.index_select(-1, lower_idx)

    inv_sqrt2 = superop.new_tensor(2.0 ** -0.5)
    half = superop.new_tensor(0.5)

    ds = inv_sqrt2 * (du + dl)
    da = 1j * inv_sqrt2 * (du - dl)
    sd = inv_sqrt2 * (ud + ld)
    ad = -1j * inv_sqrt2 * (ud - ld)
    ss = half * (uu + ul + lu + ll)
    sa = 0.5j * (uu - ul + lu - ll)
    as_block = 0.5j * (-uu - ul + lu + ll)
    aa = half * (uu - ul - lu + ll)

    diag_change = get_hermitian_diag_basis_change_cached(N, X.device, real_dtype).to(dtype=superop.dtype)
    diag_change_t = diag_change.transpose(0, 1)

    dd = torch.matmul(diag_change_t, torch.matmul(dd, diag_change))
    ds = torch.matmul(diag_change_t, ds)
    da = torch.matmul(diag_change_t, da)
    sd = torch.matmul(sd, diag_change)
    ad = torch.matmul(ad, diag_change)

    top = torch.cat((dd, ds, da), dim=-1)
    middle = torch.cat((sd, ss, sa), dim=-1)
    bottom = torch.cat((ad, as_block, aa), dim=-1)
    result = torch.cat((top, middle, bottom), dim=-2)

    # Enforce the target real skew structure explicitly so the result can be
    # passed straight to Pfaffian routines without extra cleanup.
    result = 0.5 * (result.real.to(dtype=real_dtype) - result.real.to(dtype=real_dtype).transpose(-1, -2))

    if traceless:
        result = result[..., 1:, 1:]
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
    "ad_matrix_real_antisymmetric",
    "comm",
    "get_eye_cached",
    "get_hermitian_basis_indices_cached",
    "get_trace_diag_indices_cached",
    "get_trace_projector_cached",
    "kron_2d",
    "makeH",
    "random_hermitian",
    "spinJMatrices",
]
