"""5D Euclidean QCD-like matrix model with 5 adjoint bosons and 1 Dirac fermion."""

from __future__ import annotations

import os

import numpy as np
import torch

from matrix_hmc import config
from matrix_hmc.algebra import (
    ad_matrix,
    add_trace_projector_inplace,
    get_eye_cached,
    random_hermitian,
)
from matrix_hmc.models.base import MatrixModel
from matrix_hmc.models.utils import _commutator_action_sum, parse_source, _anticommutator_action_sum

model_name = "qcd_5d"


def _gamma_euclidean_5d(device, dtype) -> torch.Tensor:
    """Return Γ^1,...,Γ^5 for the 5D Euclidean Clifford algebra.

    Γ^1,...,Γ^4 are the same 4×4 matrices as in the 4D model.
    Γ^5 = Γ^1 Γ^2 Γ^3 Γ^4, which satisfies:
      - Hermitian
      - (Γ^5)^2 = I
      - {Γ^5, Γ^μ} = 0  for μ = 1,...,4

    Together {Γ^I, Γ^J} = 2δ^{IJ} for I,J = 1,...,5, giving the
    irreducible 4×4 representation of the d=5 Euclidean Clifford algebra.

    Returns:
        Tensor of shape (5, 4, 4).
    """
    s1 = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device)
    s2 = torch.tensor([[0.0, -1j], [1j, 0.0]], dtype=dtype, device=device)
    s3 = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device)
    I2 = torch.eye(2, dtype=dtype, device=device)

    g1 = torch.kron(s3, I2)
    g2 = torch.kron(s1, s3)
    g3 = torch.kron(s1, s1)
    g4 = -torch.kron(s2, I2)
    g5 = g1 @ g2 @ g3 @ g4

    return torch.stack([g1, g2, g3, g4, g5], dim=0)  # (5, 4, 4)


def _qcd5d_logdet(
    X: torch.Tensor,
    gammas: torch.Tensor,
    massless: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (sign, log|det K|) for the 5D QCD fermion matrix.

    Massive:  K = I_{4N²} + Σ_{μ=1}^5 Γ^μ ⊗ ad_{X_μ}
    Massless: K = Σ_{μ=1}^5 Γ^μ ⊗ ad_{X_μ}  with trace mode lifted.

    D = Σ_μ Γ^μ ⊗ ad_{X_μ} is anti-Hermitian (each Γ^μ Hermitian, each
    ad_{X_μ} anti-Hermitian), so eigenvalues of K are 1+iλ with λ real,
    giving |eigenvalue|² = 1+λ² > 0.  The fermionic determinant is positive
    for all configurations (verified numerically).
    """
    N = X.shape[-1]
    N2 = N * N
    dtype = X.dtype
    device = X.device

    ad_ops = [ad_matrix(X[mu]) for mu in range(5)]
    D = sum(torch.kron(gammas[mu], ad_ops[mu]) for mu in range(5))

    if massless:
        # 4 trace-direction zero modes (one per spinor component), same as 4D.
        K = D
        for alpha in range(4):
            add_trace_projector_inplace(
                K[alpha * N2:(alpha + 1) * N2, alpha * N2:(alpha + 1) * N2], N
            )
    else:
        K = get_eye_cached(4 * N2, device, dtype) + D

    result = torch.linalg.slogdet(K)
    if result[0].real.item() < 0.0:
        import warnings
        warnings.warn(f"qcd_5d: det(K) sign = {result[0].item():.6f} (expected +1, sign problem?)", RuntimeWarning, stacklevel=2)
    return result


class QCD5DModel(MatrixModel):
    """5D Euclidean QCD-like matrix model.

    Action (before fermion integration):

    .. math::

        S = \\frac{N}{g} \\left[ -\\frac{1}{4} \\sum_{\\mu<\\nu}
            \\mathrm{Tr}([X_\\mu, X_\\nu])^2
            + m_b \\sum_\\mu \\mathrm{Tr}(X_\\mu^2)
            + \\bar{\\psi}(1 + i \\Gamma^\\mu \\,[X_\\mu, .]) \\psi
            \\right]

    with μ running over 1,...,5.  After integrating out the Dirac fermion
    the partition function acquires ``det K`` with
    ``K = I + Σ_{μ=1}^5 Γ^μ ⊗ ad_{X_μ}``.

    Args:
        ncol: Matrix size N.
        couplings: ``[g]`` — the 't Hooft coupling.
        source: Optional external source.
        boson_mass: Coefficient of Tr(X_μ²). Default 1.0.
        massless: If True, drop Tr(X_μ²) and the identity in K (lift trace
            modes). Default False.
    """

    model_name = model_name

    def __init__(
        self,
        ncol: int,
        couplings: list,
        source: np.ndarray | None = None,
        boson_mass: float = 1.0,
        massless: bool = False,
    ) -> None:
        super().__init__(nmat=5, ncol=ncol)
        self.couplings = couplings
        self.g = couplings[0]
        self.source = parse_source(source, self.nmat, config.device, config.dtype)
        self.boson_mass = float(boson_mass)
        self.massless = massless
        self.is_hermitian = True
        self.is_traceless = True

        self._gammas = _gamma_euclidean_5d(config.device, config.dtype)

    def load_fresh(self):
        X = 0.01 * random_hermitian(self.ncol, batchsize=self.nmat)
        self.set_state(X)

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        comm_sq = -0.5 * _commutator_action_sum(X)
        trace_sq = (
            0.0
            if self.massless
            else self.boson_mass * torch.einsum("bij,bji->", X, X)
        )
        bos = (comm_sq + trace_sq) * (self.ncol / self.g)

        _, log_abs_det = _qcd5d_logdet(X, self._gammas, massless=self.massless)
        ferm = -log_abs_det.real

        src = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        if self.source is not None:
            src = -(self.ncol / self.g**0.5) * torch.einsum(
                "iab,iba->", self.source, X
            )

        return bos.real + ferm + src.real

    def measure_observables(self, X: torch.Tensor | None = None):
        with torch.no_grad():
            X = self._resolve_X(X)
            eigs = [torch.linalg.eigvalsh(mat).cpu().numpy() for mat in X]
            eigs.append(torch.linalg.eigvals(X[0] + 1j * X[1]).cpu().numpy())

            comm_raw = _commutator_action_sum(X).real.item() / self.nmat / (self.nmat - 1) / self.ncol
            anticomm_raw = _anticommutator_action_sum(X).real.item() / self.nmat / (self.nmat - 1) / self.ncol
            moments = torch.einsum("aij,bji->ab", X, X).real

            parts = [
                np.array([anticomm_raw, comm_raw], dtype=np.float64),
                moments.detach().cpu().numpy().astype(np.float64).reshape(-1),
            ]

            corrs = np.concatenate(parts)

        return eigs, corrs

    def build_paths(self, name_prefix: str, data_path: str) -> dict[str, str]:
        bm = f"_bm{round(self.boson_mass, 4)}" if self.boson_mass != 1.0 else ""
        run_dir = os.path.join(
            data_path,
            f"{name_prefix}_{self.model_name}_g{round(self.g, 4)}{bm}_N{self.ncol}",
        )
        return {
            "dir": run_dir,
            "eigs": os.path.join(run_dir, "evals.npz"),
            "corrs": os.path.join(run_dir, "corrs.npz"),
            "meta": os.path.join(run_dir, "metadata.json"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        avg = (torch.einsum("bij,bji->", X, X).real / (self.nmat * self.ncol)).item()
        return f"<trX^2>/N = {avg:.5f}. "

    def extra_config_lines(self) -> list[str]:
        return [
            f"  Coupling g               = {self.g}",
            f"  Boson mass               = {self.boson_mass}",
            f"  Massless                 = {self.massless}",
        ]

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "boson_mass": self.boson_mass,
                "massless": self.massless,
            }
        )
        return meta


def build_model(args):
    return QCD5DModel(
        ncol=args.ncol,
        couplings=args.coupling,
        source=args.source,
        boson_mass=getattr(args, "boson_mass", 1.0),
        massless=getattr(args, "massless", False),
    )
