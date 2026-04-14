"""10D polarized IKKT model with bosonic terms and placeholder fermions."""

from __future__ import annotations

import os

import numpy as np
import torch

from MatrixModelHMC_pytorch import config
from MatrixModelHMC_pytorch.algebra import random_hermitian, spinJMatrices
from MatrixModelHMC_pytorch.models.base import MatrixModel
from MatrixModelHMC_pytorch.models.utils import _commutator_action_sum, parse_source

model_name = "pikkt10d"


def build_model(args):
    return PIKKT10DModel(
        ncol=args.ncol,
        couplings=args.coupling,
        source=args.source,
    )


class PIKKT10DModel(MatrixModel):
    """10D polarized IKKT model with Omega fixed to 1."""

    model_name = model_name

    def __init__(self, ncol: int, couplings: list, source: np.ndarray | None = None) -> None:
        super().__init__(nmat=10, ncol=ncol)
        self.couplings = couplings
        self.g = self.couplings[0]
        self.omega = 1.0
        self.source = parse_source(source, self.nmat, config.device, config.dtype)
        self.is_hermitian = True
        self.is_traceless = True

        coeffs = torch.full((self.nmat,), 1.0 / 64.0, dtype=config.real_dtype, device=config.device)
        coeffs[:3] = 3.0 / 64.0
        self._mass_coeffs = coeffs

    def load_fresh(self, args):
        scale = float(np.sqrt(self.g / self.ncol))
        mats = [scale * random_hermitian(self.ncol) for _ in range(self.nmat)]
        X = torch.stack(mats, dim=0).to(dtype=config.dtype, device=config.device)

        if args.spin is not None:
            J_matrices = torch.from_numpy(spinJMatrices(args.spin)).to(
                dtype=config.dtype, device=config.device
            )
            ntimes = self.ncol // J_matrices.shape[1]
            eye_nt = torch.eye(ntimes, dtype=config.dtype, device=config.device)
            dim = ntimes * J_matrices.shape[1]

            X = torch.zeros_like(X)
            for i in range(3):
                X[i][:dim, :dim] = 3/8 * torch.kron(eye_nt, J_matrices[i])

        self.set_state(X)

    def fermion_determinant(self, X: torch.Tensor | None = None) -> torch.Tensor:
        """Placeholder for fermionic term; currently disabled."""
        X = self._resolve_X(X)
        return torch.tensor(0.0, dtype=config.real_dtype, device=X.device)

    def bosonic_potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)

        bos = -0.5 * _commutator_action_sum(X)
        trace_sq = torch.einsum("bij,bji->b", X, X).real
        bos = bos + torch.dot(self._mass_coeffs.to(device=X.device), trace_sq)

        # (i/3) * eps^{ijk} Tr(X_i X_j X_k) with i,j,k in {1,2,3}
        myers = 1j * (torch.trace(X[0] @ X[1] @ X[2]) - torch.trace(X[0] @ X[2] @ X[1]))
        bos = bos + myers

        return (self.ncol / self.g) * bos.real

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        src = torch.tensor(0.0, dtype=config.real_dtype, device=X.device)
        if self.source is not None:
            src = (-(self.ncol / self.g ** 0.5) * torch.einsum("iab,iba->", self.source, X)).real
        return self.bosonic_potential(X) + self.fermion_determinant(X) + src

    def measure_observables(self, X: torch.Tensor | None = None):
        with torch.no_grad():
            X = self._resolve_X(X)
            eigs = [torch.linalg.eigvalsh(mat).cpu().numpy() for mat in X]
            eigs.append(
                torch.linalg.eigvalsh(X[0] @ X[0] + X[1] @ X[1] + X[2] @ X[2])
                .cpu()
                .numpy()
            )

            trace_sq = torch.einsum("bij,bji->b", X, X).real
            tr_i = trace_sq[:3].sum() / (3 * self.ncol)
            tr_p = trace_sq[3:].sum() / (7 * self.ncol)
            comm = _commutator_action_sum(X).real / self.ncol
            corrs = torch.stack([tr_i, tr_p, comm]).cpu().numpy()

        return eigs, corrs

    def build_paths(self, name_prefix: str, data_path: str) -> dict[str, str]:
        run_dir = os.path.join(
            data_path,
            f"{name_prefix}_{self.model_name}_g{round(self.g, 4)}_N{self.ncol}",
        )
        return {
            "dir": run_dir,
            "eigs": os.path.join(run_dir, "evals.npz"),
            "corrs": os.path.join(run_dir, "corrs.npz"),
            "meta": os.path.join(run_dir, "metadata.json"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }

    def extra_config_lines(self) -> list[str]:
        return [
            f"  Coupling g               = {self.g}",
            "  Omega                    = 1 (fixed)",
            "  Fermion determinant      = placeholder (0)",
        ]

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        trace_sq = torch.einsum("bij,bji->b", X, X).real
        tr_i = (trace_sq[:3].sum() / (3 * self.ncol)).item()
        tr_p = (trace_sq[3:].sum() / (7 * self.ncol)).item()
        return f"<tr Xi^2> = {tr_i:.5f}, <tr Xp^2> = {tr_p:.5f}. "

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "has_source": self.source is not None,
                "model_variant": "pikkt10d",
                "omega_fixed": 1.0,
                "fermion_determinant": "disabled_placeholder",
            }
        )
        return meta
