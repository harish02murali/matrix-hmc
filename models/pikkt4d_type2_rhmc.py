"""Type II polarized IKKT model with RHMC-style pseudofermions."""

from __future__ import annotations

import os

import numpy as np
import torch

from MatrixModelHMC_pytorch import config
from MatrixModelHMC_pytorch.algebra import (
    add_trace_projector_inplace,
    ad_matrix,
    get_eye_cached,
    makeH,
    random_hermitian,
    spinJMatrices,
)
from MatrixModelHMC_pytorch.models.base import MatrixModel
from MatrixModelHMC_pytorch.models.utils import _commutator_action_sum

model_name = "pikkt4d_type2_rhmc"


def _adjoint_grad_from_matrix(M: torch.Tensor, ncol: int) -> torch.Tensor:
    """Gradient of Re(sum conj(M) * ad_X) w.r.t X in column-major vec ordering."""
    M4 = M.reshape(ncol, ncol, ncol, ncol).permute(1, 0, 3, 2)
    diag_jl = M4.diagonal(dim1=1, dim2=3)
    grad_left = diag_jl.sum(dim=-1)
    diag_ik = M4.diagonal(dim1=0, dim2=2)
    grad_right = diag_ik.sum(dim=-1).transpose(0, 1)
    return (grad_left - grad_right).conj()


def _adjoint_grad_from_outer_sum(
    U: torch.Tensor,
    V: torch.Tensor,
    coeff: torch.Tensor,
) -> torch.Tensor:
    """
    Batched low-rank form of _adjoint_grad_from_matrix for column-major vec(U) vec(V)^dagger.

    U, V have shape (S, N, N) and coeff has shape (S,).
    """
    coeff_c = coeff.to(dtype=U.dtype)
    term1 = torch.einsum("s,sij,skj->ik", coeff_c, U, V.conj())
    term2 = torch.einsum("s,sji,sjk->ik", coeff_c, V.conj(), U)
    return term1 - term2


def build_model(args):
    return PIKKTTypeIIRHMCModel(
        ncol=args.ncol,
        couplings=args.coupling,
        source=args.source,
        bosonic=getattr(args, "bosonic", False),
        lorentzian=getattr(args, "lorentzian", False),
        rhmc_order=getattr(args, "rhmc_order", 20),
        rhmc_lmin=getattr(args, "rhmc_lmin", None),
        rhmc_lmax=getattr(args, "rhmc_lmax", None),
        rhmc_cg_tol=getattr(args, "rhmc_cg_tol", 1e-8),
        rhmc_cg_maxiter=getattr(args, "rhmc_cg_maxiter", 400),
    )


def _fit_partial_fraction_power(
    power: float,
    order: int,
    lmin: float,
    lmax: float,
) -> tuple[np.float64, np.ndarray, np.ndarray, float]:
    """
    Fit x^power ~= c0 + sum_j alpha_j / (x + beta_j) on [lmin, lmax].

    Uses fixed log-spaced positive shifts beta_j and linear least squares for
    c0, alpha_j. This is not optimal-Zolotarev, but is robust and simple.
    """
    if order < 1:
        raise ValueError("rhmc order must be at least 1")
    if lmin <= 0 or lmax <= lmin:
        raise ValueError("rhmc spectrum bounds must satisfy 0 < lmin < lmax")

    betas = np.logspace(np.log10(lmin), np.log10(lmax), num=order, dtype=np.float64)
    npts = max(8 * order, 256)
    xs = np.logspace(np.log10(lmin), np.log10(lmax), num=npts, dtype=np.float64)
    target = xs**power

    A = np.empty((npts, order + 1), dtype=np.float64)
    A[:, 0] = 1.0
    for j in range(order):
        A[:, j + 1] = 1.0 / (xs + betas[j])

    # Fit in weighted least squares with weight ~ 1/|target|
    # to control relative error across decades.
    w = 1.0 / np.maximum(np.abs(target), 1e-30)
    Aw = A * w[:, None]
    tw = target * w

    coeffs, _, _, _ = np.linalg.lstsq(Aw, tw, rcond=None)
    approx = A @ coeffs
    rel_err = float(np.max(np.abs((approx - target) / target)))

    c0 = np.float64(coeffs[0])
    alphas = coeffs[1:].astype(np.float64)
    return c0, alphas, betas, rel_err


def _complex_normal(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Complex standard normal vector with E[|z|^2]=1."""
    re = torch.randn(size, device=device, dtype=config.real_dtype)
    im = torch.randn(size, device=device, dtype=config.real_dtype)
    return ((re + 1j * im) / np.sqrt(2.0)).to(dtype=dtype)


def _direct_multi_shift_solve(
    mat: torch.Tensor,
    b: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Solve (mat + shift_s I) x_s = b for all shifts via batched linalg.solve.

    Replaces the iterative CG loop for small systems where GPU kernel-launch
    overhead dominates.  ``mat`` must be square and (nvec x nvec).
    """
    nshifts = int(shifts.numel())
    nvec = int(b.numel())
    if nshifts == 0:
        return torch.zeros((0, nvec), dtype=b.dtype, device=b.device)
    eye = torch.eye(nvec, dtype=mat.dtype, device=mat.device)
    # Build (nshifts, nvec, nvec) shifted systems
    M = mat.unsqueeze(0) + shifts.to(dtype=mat.real.dtype).view(-1, 1, 1).to(dtype=mat.dtype) * eye
    rhs = b.unsqueeze(0).expand(nshifts, -1)  # (nshifts, nvec)
    return torch.linalg.solve(M, rhs)          # (nshifts, nvec)


def _multi_shift_cg_solve(
    matvec,
    b: torch.Tensor,
    shifts: torch.Tensor,
    tol: float,
    maxiter: int,
    finite_check_every: int = 1,
) -> torch.Tensor:
    """
    Solve (A + shift_s I)x_s = b for all shifts using one Krylov sequence.

    A is represented by ``matvec`` and assumed Hermitian positive semidefinite.
    Returns shape (nshifts, nvec).
    """
    if b.ndim != 1:
        raise ValueError(f"multi-shift CG expects vector rhs, got shape {b.shape}")
    finite_check_every = max(int(finite_check_every), 0)
    # Convergence polling less frequently avoids host-device sync every iteration.
    convergence_check_every = 32 if b.device.type == "cuda" else 1

    shifts = shifts.to(device=b.device, dtype=config.real_dtype)
    nshifts = int(shifts.numel())
    nvec = int(b.numel())
    if nshifts == 0:
        return torch.zeros((0, nvec), dtype=b.dtype, device=b.device)

    tiny = torch.tensor(1e-30, dtype=config.real_dtype, device=b.device)
    one = torch.tensor(1.0, dtype=config.real_dtype, device=b.device)

    def _safe_nonzero(x: torch.Tensor) -> torch.Tensor:
        sign = torch.where(x >= 0, one, -one)
        return torch.where(x.abs() > tiny, x, sign * tiny)

    x_shift = torch.zeros((nshifts, nvec), dtype=b.dtype, device=b.device)
    r = b.clone()
    p = r.clone()
    p_shift = r.unsqueeze(0).expand(nshifts, -1).clone()

    rr = torch.vdot(r, r).real
    if not torch.isfinite(rr):
        return x_shift
    b_norm_sq = rr.clamp_min(tiny)
    stop_sq = (tol * tol) * b_norm_sq
    if rr <= stop_sq:
        return x_shift

    zeta_prev = torch.ones(nshifts, dtype=config.real_dtype, device=b.device)
    zeta = torch.ones(nshifts, dtype=config.real_dtype, device=b.device)
    alpha_prev = one.clone()
    beta_prev = -one.clone()
    iter_idx = torch.zeros((), dtype=torch.int64, device=b.device)

    for it in range(maxiter):
        Ap = matvec(p)
        pAp = torch.vdot(p, Ap).real
        pAp = _safe_nonzero(pAp)

        beta = -rr / pAp
        beta = torch.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
        beta_c = beta.to(dtype=b.dtype)
        r = r + beta_c * Ap
        rr_new = torch.vdot(r, r).real
        alpha = rr_new / rr.clamp_min(tiny)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

        hat_alpha_prev = one + alpha_prev * beta / _safe_nonzero(beta_prev)
        hat_alpha = torch.where(iter_idx > 0, hat_alpha_prev, one)

        denom = (hat_alpha - shifts * beta) / zeta + (one - hat_alpha) / zeta_prev
        denom = _safe_nonzero(denom)
        zeta_next = one / denom
        ratio = zeta_next / zeta
        zeta_next = torch.nan_to_num(zeta_next, nan=0.0, posinf=0.0, neginf=0.0)
        ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

        beta_shift = ratio * beta
        # alpha_shift must scale as ratio^2 (not ratio).
        alpha_shift = (ratio * ratio) * alpha

        beta_shift_c = beta_shift[:, None].to(dtype=b.dtype)
        alpha_shift_c = alpha_shift[:, None].to(dtype=b.dtype)
        zeta_next_c = zeta_next[:, None].to(dtype=b.dtype)
        x_shift = x_shift - beta_shift_c * p_shift
        p_shift = (
            zeta_next_c * r[None, :]
            + alpha_shift_c * p_shift
        )
        p = r + alpha.to(dtype=b.dtype) * p
        rr = rr_new
        alpha_prev = alpha
        beta_prev = beta
        zeta_prev = zeta
        zeta = zeta_next
        iter_idx = iter_idx + 1

        if finite_check_every > 0 and ((it + 1) % finite_check_every == 0):
            finite_scalar = (
                torch.isfinite(pAp)
                & torch.isfinite(beta)
                & torch.isfinite(alpha)
                & torch.isfinite(rr)
                & torch.isfinite(zeta).all()
            )
            if not bool(finite_scalar.item()):
                break

        if convergence_check_every > 0 and ((it + 1) % convergence_check_every == 0):
            if bool((rr <= stop_sq).item()):
                break

    return x_shift


def _apply_rational_to_vec(
    matvec,
    b: torch.Tensor,
    c0: torch.Tensor,
    alphas: torch.Tensor,
    betas: torch.Tensor,
    tol: float,
    maxiter: int,
    finite_check_every: int = 1,
) -> torch.Tensor:
    """Apply c0 I + sum_j alpha_j (A + beta_j I)^-1 to vector b."""
    y = c0.to(dtype=b.dtype) * b
    if betas.numel() == 0:
        return y
    x_shift = _multi_shift_cg_solve(
        matvec,
        b,
        betas,
        tol=tol,
        maxiter=maxiter,
        finite_check_every=finite_check_every,
    )
    return y + torch.einsum("s,sn->n", alphas.to(dtype=b.dtype), x_shift)


class PIKKTTypeIIRHMCModel(MatrixModel):
    """Type II polarized IKKT model with RHMC pseudofermion action."""

    model_name = model_name

    def __init__(
        self,
        ncol: int,
        couplings: list,
        source: np.ndarray | None = None,
        bosonic: bool = False,
        lorentzian: bool = False,
        rhmc_order: int = 20,
        rhmc_lmin: float | None = None,
        rhmc_lmax: float | None = None,
        rhmc_cg_tol: float = 1e-8,
        rhmc_cg_maxiter: int = 400,
    ) -> None:
        super().__init__(nmat=4, ncol=ncol)
        self.couplings = couplings
        self.g = self.couplings[0]
        self.omega = self.couplings[1]
        self.bosonic = bosonic
        self.lorentzian = lorentzian
        self.source = (
            torch.diag(torch.tensor(source, device=config.device, dtype=config.dtype))
            if source is not None
            else None
        )
        self.is_hermitian = True
        self.is_traceless = True

        self.rhmc_order = int(rhmc_order)
        self._rhmc_manual_window = (rhmc_lmin is not None and rhmc_lmax is not None)
        self.rhmc_lmin = float(rhmc_lmin) if rhmc_lmin is not None else None
        self.rhmc_lmax = float(rhmc_lmax) if rhmc_lmax is not None else None
        self.rhmc_cg_tol = float(rhmc_cg_tol)
        self.rhmc_cg_maxiter = int(rhmc_cg_maxiter)
        self._rhmc_cg_finite_check_every = 32 if config.device.type == "cuda" else 1
        # Use direct dense solve (linalg.solve) instead of iterative CG when
        # nvec = 2*N^2 is small enough that GPU kernel-launch overhead dominates.
        # At N=10 (nvec=200) or N=16 (nvec=512) this is orders of magnitude faster.
        self._rhmc_direct_solve_threshold = 512
        self._rhmc_window_pad = 5.0
        self._rhmc_probe_min_eval_raw: float | None = None
        self._rhmc_probe_max_eval_raw: float | None = None
        self._rhmc_probe_cond_raw: float | None = None

        dim_tr = self.ncol * self.ncol
        self._eye23 = (2.0 / 3.0) * get_eye_cached(
            2 * dim_tr, device=config.device, dtype=config.dtype
        )
        if config.ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
            self._force_impl = torch.compile(self._force_impl, dynamic=False, backend=config.TORCH_COMPILE_BACKEND)
            # Compile the hot CG matvec kernels individually: they are called
            # ~400 times per force step and contain Python control flow that
            # prevents them from being traced inside _force_impl's graph.
            self._apply_K_vec = torch.compile(self._apply_K_vec, dynamic=False, backend=config.TORCH_COMPILE_BACKEND)
            self._apply_K_dag_vec = torch.compile(self._apply_K_dag_vec, dynamic=False, backend=config.TORCH_COMPILE_BACKEND)

        coeffs = torch.full(
            (self.nmat,), self.omega / 3.0, dtype=config.real_dtype, device=config.device
        )
        coeffs[: min(3, self.nmat)] += torch.tensor(2.0 / 9.0, dtype=config.real_dtype, device=config.device)
        self._mass_coeffs = coeffs

        self._inv_c0: torch.Tensor | None = None
        self._inv_alphas: torch.Tensor | None = None
        self._inv_betas: torch.Tensor | None = None
        self._hb_c0: torch.Tensor | None = None
        self._hb_alphas: torch.Tensor | None = None
        self._hb_betas: torch.Tensor | None = None
        self._rhmc_fit_err_inv: float | None = None
        self._rhmc_fit_err_hb: float | None = None

        self._phi: torch.Tensor | None = None
        self._obs_eig_fallback_count = 0
        if self._rhmc_manual_window:
            self._setup_rhmc_coefficients()

    def _safe_eigvalsh_numpy(self, mat: torch.Tensor, *, label: str) -> np.ndarray:
        """
        Robust Hermitian spectrum extraction for diagnostics.

        Falls back to CPU + diagonal regularization and ultimately to sorted
        real diagonal entries if eigensolvers fail.
        """
        herm = 0.5 * (mat + mat.conj().transpose(-1, -2))
        herm_cpu = herm.to("cpu")
        herm_cpu = torch.nan_to_num(herm_cpu, nan=0.0, posinf=1e100, neginf=-1e100)
        n = herm_cpu.shape[-1]
        if n == 0:
            return np.empty((0,), dtype=np.float64)
        eye = torch.eye(n, dtype=herm_cpu.dtype, device=herm_cpu.device)

        norm = float(torch.linalg.norm(herm_cpu).real.item())
        if not np.isfinite(norm) or norm <= 0.0:
            norm = 1.0
        scaled = herm_cpu / norm

        def _try_torch_eigvalsh(candidate: torch.Tensor) -> np.ndarray | None:
            try:
                vals = torch.linalg.eigvalsh(candidate)
                if torch.isfinite(vals).all():
                    return (vals * norm).numpy()
            except Exception:
                return None
            return None

        def _try_numpy_eigvalsh(candidate: torch.Tensor) -> np.ndarray | None:
            try:
                arr = np.asarray(candidate.numpy(), dtype=np.complex128)
                vals = np.linalg.eigvalsh(arr)
                vals = np.real(vals) * norm
                if np.all(np.isfinite(vals)):
                    return vals.astype(np.float64, copy=False)
            except Exception:
                return None
            return None

        vals_np = _try_torch_eigvalsh(scaled)
        if vals_np is not None:
            return vals_np

        vals_np = _try_numpy_eigvalsh(scaled)
        if vals_np is not None:
            if self._obs_eig_fallback_count < 5:
                print(f"[RHMC] eigvalsh fallback ({label}): used numpy eigvalsh")
            self._obs_eig_fallback_count += 1
            return vals_np

        base = 1e-14
        for mult in (1.0, 10.0, 100.0, 1000.0, 1e5, 1e6):
            jittered = scaled + (base * mult) * eye
            vals_np = _try_torch_eigvalsh(jittered)
            if vals_np is None:
                vals_np = _try_numpy_eigvalsh(jittered)
            if vals_np is not None:
                if self._obs_eig_fallback_count < 5:
                    print(
                        f"[RHMC] eigvalsh fallback ({label}): "
                        f"used diagonal jitter {base * mult:.3e}"
                    )
                self._obs_eig_fallback_count += 1
                return vals_np

        # Last-resort fallback: sorted real diagonal entries.
        vals = torch.diagonal(herm_cpu, dim1=-2, dim2=-1).real.numpy()
        vals = np.sort(vals.astype(np.float64, copy=False))
        if self._obs_eig_fallback_count < 5:
            print(f"[RHMC] eigvalsh fallback ({label}): used real(diag)")
        self._obs_eig_fallback_count += 1
        return vals

    def _safe_eigvals_numpy(self, mat: torch.Tensor, *, label: str) -> np.ndarray:
        """
        Robust non-Hermitian spectrum extraction for diagnostics.

        Tries numpy eigvals first, then diagonal-jitter retries, and finally
        returns diagonal entries as a last-resort approximation.
        """
        mat_cpu = mat.to("cpu")
        mat_cpu = torch.nan_to_num(mat_cpu, nan=0.0, posinf=1e100, neginf=-1e100)
        n = mat_cpu.shape[-1]
        if n == 0:
            return np.empty((0,), dtype=np.complex128)
        eye = torch.eye(n, dtype=mat_cpu.dtype, device=mat_cpu.device)

        norm = float(torch.linalg.norm(mat_cpu).real.item())
        if not np.isfinite(norm) or norm <= 0.0:
            norm = 1.0
        scaled = mat_cpu / norm

        def _try_numpy_eigvals(candidate: torch.Tensor) -> np.ndarray | None:
            try:
                arr = np.asarray(candidate.numpy(), dtype=np.complex128)
                vals = np.linalg.eigvals(arr) * norm
                if np.all(np.isfinite(vals.real)) and np.all(np.isfinite(vals.imag)):
                    return vals.astype(np.complex128, copy=False)
            except Exception:
                return None
            return None

        vals_np = _try_numpy_eigvals(scaled)
        if vals_np is not None:
            return vals_np

        base = 1e-14
        for mult in (1.0, 10.0, 100.0, 1000.0, 1e5, 1e6):
            vals_np = _try_numpy_eigvals(scaled + (base * mult) * eye)
            if vals_np is not None:
                if self._obs_eig_fallback_count < 5:
                    print(
                        f"[RHMC] eigvals fallback ({label}): "
                        f"used diagonal jitter {base * mult:.3e}"
                    )
                self._obs_eig_fallback_count += 1
                return vals_np

        vals = np.diag(np.asarray(mat_cpu.numpy(), dtype=np.complex128))
        if self._obs_eig_fallback_count < 5:
            print(f"[RHMC] eigvals fallback ({label}): used diag entries")
        self._obs_eig_fallback_count += 1
        return vals

    def _setup_rhmc_coefficients(self) -> None:
        if self.rhmc_lmin is None or self.rhmc_lmax is None:
            raise RuntimeError("RHMC spectrum window is unset")
        inv_c0, inv_a, inv_b, inv_err = _fit_partial_fraction_power(
            power=-0.5,
            order=self.rhmc_order,
            lmin=self.rhmc_lmin,
            lmax=self.rhmc_lmax,
        )
        hb_c0, hb_a, hb_b, hb_err = _fit_partial_fraction_power(
            power=0.25,
            order=self.rhmc_order,
            lmin=self.rhmc_lmin,
            lmax=self.rhmc_lmax,
        )
        self._inv_c0 = torch.tensor(inv_c0, dtype=config.real_dtype, device=config.device)
        self._inv_alphas = torch.tensor(inv_a, dtype=config.real_dtype, device=config.device)
        self._inv_betas = torch.tensor(inv_b, dtype=config.real_dtype, device=config.device)
        self._hb_c0 = torch.tensor(hb_c0, dtype=config.real_dtype, device=config.device)
        self._hb_alphas = torch.tensor(hb_a, dtype=config.real_dtype, device=config.device)
        self._hb_betas = torch.tensor(hb_b, dtype=config.real_dtype, device=config.device)
        self._rhmc_fit_err_inv = float(inv_err)
        self._rhmc_fit_err_hb = float(hb_err)

    def _auto_probe_window_from_X(self, X_eff: torch.Tensor) -> None:
        """Estimate spectral bounds of K†K via matrix-free power/inverse iteration.

        Avoids materialising the full (2N²×2N²) K matrix so this is O(N³) in
        memory and O(n_iter × N³) in time — safe at N=100 on H200.
        """
        matvec = self._build_kdagk_matvec(X_eff)
        nvec = 2 * self.ncol * self.ncol

        with torch.no_grad():
            # --- λ_max via power iteration ---
            v = torch.randn(nvec, dtype=X_eff.dtype, device=X_eff.device)
            v = v / v.norm()
            for _ in range(80):
                v = matvec(v)
                nrm = v.norm()
                if nrm < 1e-30:
                    break
                v = v / nrm
            lmax = float(torch.real(torch.vdot(v, matvec(v))).item())
            lmax = max(lmax, 1e-12)

            # --- λ_min via shift-and-invert: largest eval of (lmax·I − K†K) is (lmax − λ_min) ---
            w = torch.randn(nvec, dtype=X_eff.dtype, device=X_eff.device)
            w = w / w.norm()
            for _ in range(80):
                w = lmax * w - matvec(w)
                nrm = w.norm()
                if nrm < 1e-30:
                    break
                w = w / nrm
            # Rayleigh quotient of K†K at the converged vector gives λ_min
            lmin = float(torch.real(torch.vdot(w, matvec(w))).item())
            lmin = max(lmin, 1e-12)
            if lmin >= lmax:
                lmin = lmax * 1e-6

        self._rhmc_probe_min_eval_raw = lmin
        self._rhmc_probe_max_eval_raw = lmax
        self._rhmc_probe_cond_raw = lmax / lmin

        self.rhmc_lmin = lmin / self._rhmc_window_pad
        self.rhmc_lmax = lmax * self._rhmc_window_pad

    def _ensure_rhmc_ready(self, X_eff: torch.Tensor) -> None:
        if self._inv_alphas is not None:
            return
        if self.rhmc_lmin is None or self.rhmc_lmax is None:
            self._auto_probe_window_from_X(X_eff)
            print(
                "[RHMC] auto window from K^dag K: "
                f"raw [{self._rhmc_probe_min_eval_raw:.3e}, {self._rhmc_probe_max_eval_raw:.3e}], "
                f"cond={self._rhmc_probe_cond_raw:.3e}, "
                f"using [{self.rhmc_lmin:.3e}, {self.rhmc_lmax:.3e}]"
            )
        self._setup_rhmc_coefficients()
        if not self._rhmc_manual_window:
            print(
                "[RHMC] rational fit errors: "
                f"x^-1/2={self._rhmc_fit_err_inv:.2e}, x^1/4={self._rhmc_fit_err_hb:.2e}"
            )

    def _effective_X(self, X: torch.Tensor) -> torch.Tensor:
        if not self.lorentzian:
            return X
        X_eff = X.clone()
        X_eff[3] = 1j * X_eff[3]
        return X_eff

    def load_fresh(self, args):
        mats = [random_hermitian(self.ncol) for _ in range(self.nmat)]
        X = torch.stack(mats, dim=0).to(dtype=config.dtype, device=config.device)

        if args.spin is not None:
            J_matrices = torch.from_numpy(spinJMatrices(args.spin)).to(
                dtype=config.dtype, device=config.device
            )
            ntimes = self.ncol // J_matrices.shape[1]
            eye_nt = torch.eye(ntimes, dtype=config.dtype, device=config.device)
            for i in range(3):
                X[i][: ntimes * J_matrices.shape[1], : ntimes * J_matrices.shape[1]] = (
                    (2 / 3 + self.omega) * torch.kron(eye_nt, J_matrices[i])
                )
            X[3] = torch.zeros_like(X[3])

        self.set_state(X)

    def _set_phi_none(self) -> None:
        self._phi = None

    def _vec_to_mat_col(self, vec: torch.Tensor) -> torch.Tensor:
        return vec.reshape(*vec.shape[:-1], self.ncol, self.ncol).transpose(-2, -1)

    def _mat_to_vec_col(self, mat: torch.Tensor) -> torch.Tensor:
        return mat.transpose(-2, -1).reshape(*mat.shape[:-2], self.ncol * self.ncol)

    def _trace_project_matrix(self, mat: torch.Tensor) -> torch.Tensor:
        eye = get_eye_cached(self.ncol, device=mat.device, dtype=mat.dtype)
        tr = mat.diagonal(dim1=-2, dim2=-1).sum(-1) / self.ncol
        return tr[..., None, None] * eye

    @staticmethod
    def _ad_action(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return X @ A - A @ X

    @staticmethod
    def _ad_action_batch(X_ops: torch.Tensor, A_batch: torch.Tensor) -> torch.Tensor:
        """
        Batched commutators [X_k, A_s] for k in operators and s in RHS batch.
        """
        left = torch.matmul(X_ops[:, None, :, :], A_batch[None, :, :, :])
        right = torch.matmul(A_batch[None, :, :, :], X_ops[:, None, :, :])
        return left - right

    def _apply_K_vec(self, X_eff: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """Apply fermion operator K(X_eff) to vector(s) in column-major basis."""
        single = psi.ndim == 1
        if single:
            psi = psi.unsqueeze(0)

        n2 = self.ncol * self.ncol
        psi_u = psi[:, :n2]
        psi_l = psi[:, n2:]
        A_u = self._vec_to_mat_col(psi_u).contiguous()
        A_l = self._vec_to_mat_col(psi_l).contiguous()

        ad_u = self._ad_action_batch(X_eff[:4], A_u)
        ad_l = self._ad_action_batch(X_eff[:4], A_l)

        # K blocks in terms of raw ad_X operators (adX in fermionMat has extra i factor).
        out_u = (-1j * ad_u[3] - ad_u[2]) + (1j * ad_l[1] - ad_l[0])
        out_l = (-1j * ad_u[1] - ad_u[0]) + (-1j * ad_l[3] + ad_l[2])

        eye = get_eye_cached(self.ncol, device=A_u.device, dtype=A_u.dtype)
        out_u = out_u - (2.0 / 3.0) * A_u + (A_u.diagonal(dim1=-2, dim2=-1).sum(-1) / self.ncol)[
            ..., None, None
        ] * eye
        out_l = out_l - (2.0 / 3.0) * A_l + (A_l.diagonal(dim1=-2, dim2=-1).sum(-1) / self.ncol)[
            ..., None, None
        ] * eye

        out = torch.cat((self._mat_to_vec_col(out_u), self._mat_to_vec_col(out_l)), dim=-1)
        return out[0] if single else out

    def _apply_K_dag_vec(self, X_eff: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """Apply K(X_eff)^dagger to vector(s) in column-major basis."""
        single = psi.ndim == 1
        if single:
            psi = psi.unsqueeze(0)

        n2 = self.ncol * self.ncol
        psi_u = psi[:, :n2]
        psi_l = psi[:, n2:]
        A_u = self._vec_to_mat_col(psi_u).contiguous()
        A_l = self._vec_to_mat_col(psi_l).contiguous()

        X_ops_d = X_eff[:4].conj().transpose(-1, -2).contiguous()
        ad_u = self._ad_action_batch(X_ops_d, A_u)
        ad_l = self._ad_action_batch(X_ops_d, A_l)

        out_u = (1j * ad_u[3] - ad_u[2]) + (-ad_l[0] + 1j * ad_l[1])
        out_l = (-ad_u[0] - 1j * ad_u[1]) + (ad_l[2] + 1j * ad_l[3])

        eye = get_eye_cached(self.ncol, device=A_u.device, dtype=A_u.dtype)
        out_u = out_u - (2.0 / 3.0) * A_u + (A_u.diagonal(dim1=-2, dim2=-1).sum(-1) / self.ncol)[
            ..., None, None
        ] * eye
        out_l = out_l - (2.0 / 3.0) * A_l + (A_l.diagonal(dim1=-2, dim2=-1).sum(-1) / self.ncol)[
            ..., None, None
        ] * eye

        out = torch.cat((self._mat_to_vec_col(out_u), self._mat_to_vec_col(out_l)), dim=-1)
        return out[0] if single else out

    def _build_kdagk_matvec(self, X_eff: torch.Tensor):
        return lambda v: self._apply_K_dag_vec(X_eff, self._apply_K_vec(X_eff, v))

    def _build_kdagk_matrix(self, X_eff: torch.Tensor) -> torch.Tensor:
        """Materialize K†K as a dense (nvec x nvec) matrix.

        Uses a single batched _apply_K_vec call (passing the identity) rather
        than nvec separate matvec applications.  Only practical for small N
        (nvec = 2*N^2 ≤ ~512).
        """
        nvec = 2 * self.ncol * self.ncol
        I = torch.eye(nvec, dtype=X_eff.dtype, device=X_eff.device)
        # _apply_K_vec(I): row i → K applied to e_i → i-th column of K
        # so the result is K.T, i.e. K = result.mT
        K_out = self._apply_K_vec(X_eff, I)       # (nvec, nvec), = K.T
        # K† = K_out.conj()  (derived: K†[i,j] = conj(K_out[i,j]))
        # K†K = K† @ K = K_out.conj() @ K_out.mT
        return K_out.conj() @ K_out.mT

    def begin_trajectory(
        self,
        X: torch.Tensor | None = None,
        *,
        already_effective: bool = False,
    ) -> None:
        """Refresh pseudofermion field once per trajectory."""
        if self.bosonic:
            self._set_phi_none()
            return

        X = self._resolve_X(X)
        X_eff = X if already_effective else self._effective_X(X)
        self._ensure_rhmc_ready(X_eff)
        nvec = 2 * self.ncol * self.ncol

        with torch.no_grad():
            if self._hb_c0 is None or self._hb_alphas is None or self._hb_betas is None:
                raise RuntimeError("RHMC heatbath coefficients are not initialized")
            eta = _complex_normal(nvec, device=X_eff.device, dtype=X_eff.dtype)
            if nvec <= self._rhmc_direct_solve_threshold:
                KdagK = self._build_kdagk_matrix(X_eff)
                hb_c0 = self._hb_c0.to(device=X_eff.device)
                hb_alphas = self._hb_alphas.to(device=X_eff.device)
                hb_betas = self._hb_betas.to(device=X_eff.device)
                x_shift = _direct_multi_shift_solve(KdagK, eta, hb_betas)
                phi = hb_c0.to(dtype=eta.dtype) * eta + torch.einsum(
                    "s,sn->n", hb_alphas.to(dtype=eta.dtype), x_shift
                )
            else:
                matvec = self._build_kdagk_matvec(X_eff)
                phi = _apply_rational_to_vec(
                    matvec,
                    eta,
                    c0=self._hb_c0.to(device=X_eff.device),
                    alphas=self._hb_alphas.to(device=X_eff.device),
                    betas=self._hb_betas.to(device=X_eff.device),
                    tol=self.rhmc_cg_tol,
                    maxiter=self.rhmc_cg_maxiter,
                    finite_check_every=self._rhmc_cg_finite_check_every,
                )
            self._phi = phi.detach()

    def end_trajectory(self, accepted: bool) -> None:
        del accepted
        # Pseudofermions are refreshed at the start of each trajectory.
        return

    def fermionMat(self, X: torch.Tensor) -> torch.Tensor:
        adX = 1j * ad_matrix(X[:4])
        adX1, adX2, adX3, adX4 = adX
        i = 1j

        upper_left = -adX4 + i * adX3
        upper_right = adX2 + i * adX1
        lower_left = -adX2 + i * adX1
        lower_right = -adX4 - i * adX3

        top = torch.cat((upper_left, upper_right), dim=1)
        bottom = torch.cat((lower_left, lower_right), dim=1)
        K = torch.cat((top, bottom), dim=0)

        K = K - self._eye23.to(dtype=K.dtype, device=K.device)

        N = X.shape[-1]
        dim = N * N
        add_trace_projector_inplace(K[:dim, :dim], N)
        add_trace_projector_inplace(K[dim:, dim:], N)
        return K

    def _require_phi(self, X_eff: torch.Tensor) -> torch.Tensor:
        self._ensure_rhmc_ready(X_eff)
        if self._phi is None:
            self.begin_trajectory(X_eff, already_effective=True)
        if self._phi is None:
            raise RuntimeError("Pseudofermion field is not initialized")
        if self._phi.device != X_eff.device or self._phi.dtype != X_eff.dtype:
            self._phi = self._phi.to(device=X_eff.device, dtype=X_eff.dtype)
        return self._phi

    def _fermion_force(self, X_eff: torch.Tensor) -> torch.Tensor:
        phi = self._require_phi(X_eff)

        if self._inv_alphas is None or self._inv_betas is None:
            raise RuntimeError("RHMC force coefficients are not initialized")
        inv_alphas = self._inv_alphas.to(device=X_eff.device)
        inv_betas = self._inv_betas.to(device=X_eff.device)

        nvec = 2 * self.ncol * self.ncol
        if nvec <= self._rhmc_direct_solve_threshold:
            KdagK = self._build_kdagk_matrix(X_eff)
            chis = _direct_multi_shift_solve(KdagK, phi, inv_betas)
        else:
            matvec = self._build_kdagk_matvec(X_eff)
            chis = _multi_shift_cg_solve(
                matvec,
                phi,
                inv_betas,
                tol=self.rhmc_cg_tol,
                maxiter=self.rhmc_cg_maxiter,
                finite_check_every=self._rhmc_cg_finite_check_every,
            )
        Kchis = self._apply_K_vec(X_eff, chis)
        coeff = (-2.0 * inv_alphas).to(dtype=X_eff.dtype)

        if self.lorentzian:
            # Keep the robust autograd pullback for the Lorentzian branch.
            B_total = torch.einsum("s,si,sj->ij", coeff, Kchis, chis.conj())
            X_var = X_eff.detach().clone().requires_grad_(True)
            K_var = self.fermionMat(X_var)
            lin = torch.real(torch.sum(torch.conj(B_total.detach()) * K_var))
            grad = torch.autograd.grad(lin, X_var, create_graph=False, retain_graph=False)[0]
            return grad

        n2 = self.ncol * self.ncol
        chi_u = self._vec_to_mat_col(chis[:, :n2]).contiguous()
        chi_l = self._vec_to_mat_col(chis[:, n2:]).contiguous()
        kchi_u = self._vec_to_mat_col(Kchis[:, :n2]).contiguous()
        kchi_l = self._vec_to_mat_col(Kchis[:, n2:]).contiguous()

        g11 = _adjoint_grad_from_outer_sum(kchi_u, chi_u, coeff)
        g12 = _adjoint_grad_from_outer_sum(kchi_u, chi_l, coeff)
        g21 = _adjoint_grad_from_outer_sum(kchi_l, chi_u, coeff)
        g22 = _adjoint_grad_from_outer_sum(kchi_l, chi_l, coeff)

        # For K blocks written in terms of raw ad_X operators:
        #   UR = i ad2 - ad1, LL = -i ad2 - ad1, UL = -i ad4 - ad3, LR = -i ad4 + ad3
        g1 = -(g12 + g21)
        g2 = 1j * (g21 - g12)
        g3 = g22 - g11
        g4 = 1j * (g11 + g22)
        return torch.stack([g1, g2, g3, g4], dim=0)

    def _force_impl(self, X: torch.Tensor) -> torch.Tensor:
        X = self._resolve_X(X)
        X_eff = self._effective_X(X)

        # Vectorised double commutator: grad[i] = -sum_{j≠i} [X_j, [X_i, X_j]]
        # All pairwise products X_i @ X_j, shape (D, D, N, N)
        XiXj = torch.matmul(X_eff.unsqueeze(1), X_eff.unsqueeze(0))
        # [X_i, X_j] for all i, j; comm_all[i,i] = 0 so diagonal terms are free
        comm_all = XiXj - XiXj.transpose(0, 1)
        # sum_j X_j @ [X_i, X_j]  and  sum_j [X_i, X_j] @ X_j
        left  = torch.einsum("jkl,ijlm->ikm", X_eff, comm_all)
        right = torch.einsum("ijkl,jlm->ikm", comm_all, X_eff)
        grad = -(left - right)

        coeff = 2j * (1 + self.omega)
        grad[0] = grad[0] + coeff * (X_eff[1] @ X_eff[2] - X_eff[2] @ X_eff[1])
        grad[1] = grad[1] + coeff * (X_eff[2] @ X_eff[0] - X_eff[0] @ X_eff[2])
        grad[2] = grad[2] + coeff * (X_eff[0] @ X_eff[1] - X_eff[1] @ X_eff[0])

        grad = grad + 2 * self._mass_coeffs.to(device=X.device)[:, None, None] * X_eff
        grad = grad * (self.ncol / self.g)

        if not self.bosonic:
            grad = grad + self._fermion_force(X_eff)

        if self.source is not None:
            grad[0] += -(self.ncol / np.sqrt(self.g)) * self.source

        if self.lorentzian:
            grad[3] = 1j * grad[3]

        return grad

    def force(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        grad = self._force_impl(X)
        if self.is_hermitian:
            grad = makeH(grad)
        if self.is_traceless:
            trs = torch.diagonal(grad, dim1=-2, dim2=-1).sum(-1).real / self.ncol
            eye = get_eye_cached(self.ncol, device=grad.device, dtype=grad.dtype)
            grad = grad - trs[..., None, None] * eye
        return grad

    def bosonic_potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        X_eff = self._effective_X(X)

        bos = -0.5 * _commutator_action_sum(X_eff)
        bos = bos + 2j * (1 + self.omega) * (
            torch.trace(X_eff[0] @ X_eff[1] @ X_eff[2])
            - torch.trace(X_eff[0] @ X_eff[2] @ X_eff[1])
        )
        trace_sq = torch.einsum("bij,bji->b", X_eff, X_eff).real
        bos = bos + torch.dot(self._mass_coeffs.to(device=X.device), trace_sq)
        return bos.real * (self.ncol / self.g)

    def ferm_potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        X_eff = self._effective_X(X)
        phi = self._require_phi(X_eff)

        matvec = self._build_kdagk_matvec(X_eff)
        if self._inv_c0 is None or self._inv_alphas is None or self._inv_betas is None:
            raise RuntimeError("RHMC potential coefficients are not initialized")
        y = _apply_rational_to_vec(
            matvec,
            phi,
            c0=self._inv_c0.to(device=X_eff.device),
            alphas=self._inv_alphas.to(device=X_eff.device),
            betas=self._inv_betas.to(device=X_eff.device),
            tol=self.rhmc_cg_tol,
            maxiter=self.rhmc_cg_maxiter,
            finite_check_every=self._rhmc_cg_finite_check_every,
        )
        return torch.real(torch.vdot(phi, y))

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        src = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        if self.source is not None:
            src = -(self.ncol / np.sqrt(self.g)) * torch.trace(self.source @ X[0])
        if self.bosonic:
            return self.bosonic_potential(X) + src.real
        return self.bosonic_potential(X) + self.ferm_potential(X) + src.real

    def measure_observables(self, X: torch.Tensor | None = None):
        with torch.no_grad():
            X = self._resolve_X(X)
            eigs = []
            for i in range(self.nmat):
                e = self._safe_eigvalsh_numpy(X[i], label=f"X[{i}]")
                eigs.append(e)

            eigs.append(self._safe_eigvals_numpy(X[0] + 1j * X[1], label="X[0]+iX[1]"))
            eigs.append(self._safe_eigvals_numpy(X[2] + 1j * X[3], label="X[2]+iX[3]"))

            eigs.append(self._safe_eigvalsh_numpy(X[0] @ X[0] + X[1] @ X[1] + X[2] @ X[2], label="casimir"))

            C = X[0] @ X[1] - X[1] @ X[0]
            A = X[0] @ X[1] + X[1] @ X[0]
            c1 = torch.trace(C @ C).real
            c2 = torch.trace(A @ A).real
            C = X[2] @ X[3] - X[3] @ X[2]
            A = X[2] @ X[3] + X[3] @ X[2]
            c3 = torch.trace(C @ C).real
            c4 = torch.trace(A @ A).real

            corrs = torch.stack([c1, c2, c3, c4]).cpu().numpy()

        return eigs, corrs

    def build_paths(self, name_prefix: str, data_path: str) -> dict[str, str]:
        run_dir = os.path.join(
            data_path,
            f"{name_prefix}_{self.model_name}_g{round(self.g, 4)}_omega{round(self.omega, 4)}_N{self.ncol}",
        )
        return {
            "dir": run_dir,
            "eigs": os.path.join(run_dir, "evals.npz"),
            "corrs": os.path.join(run_dir, "corrs.npz"),
            "meta": os.path.join(run_dir, "metadata.json"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }

    def extra_config_lines(self) -> list[str]:
        lines = [
            f"  Coupling g               = {self.g}",
            f"  Coupling Omega2/Omega1   = {self.omega}",
            f"  RHMC order               = {self.rhmc_order}",
            f"  RHMC CG tol / maxiter    = {self.rhmc_cg_tol:g} / {self.rhmc_cg_maxiter}",
            f"  RHMC CG finite-check     = every {self._rhmc_cg_finite_check_every} iter(s)",
        ]
        if self.rhmc_lmin is None or self.rhmc_lmax is None:
            lines.append("  RHMC spectrum window     = auto (probe K^\\dagger K at runtime)")
        else:
            src = "manual" if self._rhmc_manual_window else "auto-probed"
            lines.append(f"  RHMC spectrum window     = [{self.rhmc_lmin:g}, {self.rhmc_lmax:g}] ({src})")
        if self._rhmc_fit_err_inv is not None and self._rhmc_fit_err_hb is not None:
            lines.append(f"  RHMC fit err x^-1/2      = {self._rhmc_fit_err_inv:.2e}")
            lines.append(f"  RHMC fit err x^1/4       = {self._rhmc_fit_err_hb:.2e}")
        if self._rhmc_probe_cond_raw is not None:
            lines.append(f"  RHMC probe cond(K^dag K) = {self._rhmc_probe_cond_raw:.2e}")
        if self.bosonic:
            lines.append("  Fermion determinant      = disabled")
        if self.lorentzian:
            lines.append("  Lorentzian X4            = enabled")
        return lines

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        casimir = (
            torch.trace(X[0] @ X[0] + X[1] @ X[1] + X[2] @ X[2]) / self.ncol
        ).item().real
        trX34 = (torch.trace(X[2] @ X[2] - X[3] @ X[3]) / self.ncol).item().real
        return f"casimir = {casimir:.5f}, mom34 = {trX34:.5f}. "

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "has_source": self.source is not None,
                "model_variant": "type2_rhmc",
                "bosonic_only": self.bosonic,
                "lorentzian_x4": self.lorentzian,
                "rhmc_order": self.rhmc_order,
                "rhmc_manual_window": self._rhmc_manual_window,
                "rhmc_lmin": self.rhmc_lmin,
                "rhmc_lmax": self.rhmc_lmax,
                "rhmc_cg_tol": self.rhmc_cg_tol,
                "rhmc_cg_maxiter": self.rhmc_cg_maxiter,
                "rhmc_cg_finite_check_every": self._rhmc_cg_finite_check_every,
                "rhmc_fit_error_inv_sqrt": self._rhmc_fit_err_inv,
                "rhmc_fit_error_heatbath_q1_4": self._rhmc_fit_err_hb,
                "rhmc_probe_min_eval_raw": self._rhmc_probe_min_eval_raw,
                "rhmc_probe_max_eval_raw": self._rhmc_probe_max_eval_raw,
                "rhmc_probe_cond_raw": self._rhmc_probe_cond_raw,
            }
        )
        return meta
