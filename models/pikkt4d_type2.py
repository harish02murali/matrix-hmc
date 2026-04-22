"""Type II polarized IKKT model."""

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
from MatrixModelHMC_pytorch.models.utils import _commutator_action_sum, parse_source, source_grad_inplace

model_name = "pikkt4d_type2"


def build_model(args):
    return PIKKTTypeIIModel(
        ncol=args.ncol,
        couplings=args.coupling,
        source=args.source,
        bosonic=getattr(args, "bosonic", False),
        lorentzian=getattr(args, "lorentzian", False),
    )


def _adjoint_grad_from_matrix(M: torch.Tensor, ncol: int) -> torch.Tensor:
    """Return grad of Tr(M^T ad_X) with column-major vec ordering."""
    M4 = M.reshape(ncol, ncol, ncol, ncol).permute(1, 0, 3, 2)
    diag_jl = M4.diagonal(dim1=1, dim2=3)
    grad_left = diag_jl.sum(dim=-1)
    diag_ik = M4.diagonal(dim1=0, dim2=2)
    grad_right = diag_ik.sum(dim=-1).transpose(0, 1)
    return (grad_left - grad_right).conj()


class PIKKTTypeIIModel(MatrixModel):
    """Type II polarized IKKT model definition."""

    model_name = model_name

    def __init__(
        self,
        ncol: int,
        couplings: list,
        source: np.ndarray | None = None,
        bosonic: bool = False,
        lorentzian: bool = False,
    ) -> None:
        super().__init__(nmat=4, ncol=ncol)
        self.couplings = couplings
        self.g = self.couplings[0]
        self.omega = self.couplings[1]
        self.bosonic = bosonic
        self.lorentzian = lorentzian
        self.source = parse_source(source, self.nmat, config.device, config.dtype)
        self.is_hermitian = True
        self.is_traceless = True

        dim_tr = self.ncol * self.ncol
        self._eye23 = (2 / 3) * get_eye_cached(
            2 * dim_tr, device=config.device, dtype=config.dtype
        )
    def _effective_X(self, X: torch.Tensor) -> torch.Tensor:
        if not self.lorentzian:
            return X
        X_eff = X.clone()
        X_eff[3] = 1j * X_eff[3]
        return X_eff

    def load_fresh(self, args):
        X = random_hermitian(self.ncol, batchsize=self.nmat)

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

        K = K - self._eye23.to(dtype=K.dtype)

        N = X.shape[-1]
        dim = N * N
        add_trace_projector_inplace(K[:dim, :dim], N)
        add_trace_projector_inplace(K[dim:, dim:], N)
        return K

    def _fermion_force(self, X: torch.Tensor) -> torch.Tensor:
        K = self.fermionMat(X)
        dim = self.ncol * self.ncol
        eye = get_eye_cached(2 * dim, device=K.device, dtype=K.dtype)
        K_inv = torch.linalg.solve(K, eye)

        G11 = K_inv[:dim, :dim].t()
        G12 = K_inv[:dim, dim:].t()
        G21 = K_inv[dim:, :dim].t()
        G22 = K_inv[dim:, dim:].t()

        M1 = -(G21 + G12)
        M2 = 1j * (G21 - G12)
        M3 = -G11 + G22
        M4 = -1j * (G11 + G22)

        grads = []
        for M in (M1, M2, M3, M4):
            grads.append(-_adjoint_grad_from_matrix(M, self.ncol))
        return torch.stack(grads, dim=0)

    def _force_impl(self, X: torch.Tensor) -> torch.Tensor:
        X = self._resolve_X(X)
        X_eff = self._effective_X(X)
        grad = torch.zeros_like(X_eff)

        for i in range(self.nmat):
            acc = torch.zeros_like(X_eff[i])
            for j in range(self.nmat):
                if i == j:
                    continue
                comm = X_eff[i] @ X_eff[j] - X_eff[j] @ X_eff[i]
                acc = acc + (X_eff[j] @ comm - comm @ X_eff[j])
            grad[i] = -acc

        coeff = 2j * (1 + self.omega)
        grad[0] += coeff * (X_eff[1] @ X_eff[2] - X_eff[2] @ X_eff[1])
        grad[1] += coeff * (X_eff[2] @ X_eff[0] - X_eff[0] @ X_eff[2])
        grad[2] += coeff * (X_eff[0] @ X_eff[1] - X_eff[1] @ X_eff[0])

        coeffs = torch.full((self.nmat,), self.omega / 3, dtype=config.real_dtype, device=X.device)
        extra = torch.tensor(2 / 9, dtype=config.real_dtype, device=X.device)
        upto = min(3, self.nmat)
        coeffs[:upto] = coeffs[:upto] + extra
        grad = grad + 2 * coeffs[:, None, None] * X_eff

        grad = grad * (self.ncol / self.g)
        if not self.bosonic:
            grad = grad + self._fermion_force(X_eff)

        if self.source is not None:
            source_grad_inplace(self.source, grad, self.ncol, self.g)

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
        coeffs = torch.full((self.nmat,), self.omega / 3, dtype=config.real_dtype, device=X.device)
        extra = torch.tensor(2 / 9, dtype=config.real_dtype, device=X.device)
        upto = min(3, self.nmat)
        coeffs[:upto] = coeffs[:upto] + extra
        bos = bos + torch.dot(coeffs, trace_sq)
        return (bos.real * (self.ncol / self.g))
    
    def ferm_potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        X_eff = self._effective_X(X)
        K = self.fermionMat(X_eff)
        det = -torch.slogdet(K)[1].real
        return det

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)

        src = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        if self.source is not None:
            src = -(self.ncol / self.g ** 0.5) * torch.einsum("iab,iba->", self.source, X)
        if self.bosonic:
            return self.bosonic_potential(X) + src.real
        return self.bosonic_potential(X) + self.ferm_potential(X) + src.real

    def measure_observables(self, X: torch.Tensor | None = None):
        with torch.no_grad():
            X = self._resolve_X(X)
            eigs = []
            for i in range(self.nmat):
                e = torch.linalg.eigvalsh(X[i]).cpu().numpy()
                eigs.append(e)

            e_complex = torch.linalg.eigvals((X[0] + 1j * X[1])).cpu().numpy()
            eigs.append(e_complex)
            e_complex = torch.linalg.eigvals((X[2] + 1j * X[3])).cpu().numpy()
            eigs.append(e_complex)

            eigs.append(
                torch.linalg.eigvalsh(X[0] @ X[0] + X[1] @ X[1] + X[2] @ X[2])
                .cpu()
                .numpy()
            )

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
        ]
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
                "model_variant": "type2",
                "bosonic_only": self.bosonic,
                "lorentzian_x4": self.lorentzian,
            }
        )
        return meta
