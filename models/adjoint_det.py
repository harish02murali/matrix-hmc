"""Adjoint determinant matrix model."""

from __future__ import annotations

import os

import numpy as np
import torch

from MatrixModelHMC_pytorch import config
from MatrixModelHMC_pytorch.models.base import MatrixModel
from MatrixModelHMC_pytorch.models.utils import (
    _anticommutator_action_sum,
    _commutator_action_sum,
    _fermion_det_log_identity_plus_sum_adX,
    parse_source,
)
model_name = "adjoint_det"


def build_model(args):
    if args.nmat is None:
        raise ValueError("--nmat must be provided for adjoint_det model")
    return AdjointDetModel(
        dim=args.nmat,
        ncol=args.ncol,
        couplings=args.coupling,
        source=args.source,
    )


class AdjointDetModel(MatrixModel):
    """Matrix model with product fermion determinant det(1 + \sum_i ad X_i)."""

    model_name = model_name

    def __init__(self, dim: int, ncol: int, couplings: list, source: np.ndarray | None = None) -> None:
        super().__init__(nmat=dim, ncol=ncol)
        self.source = parse_source(source, dim, config.device, config.dtype)
        self.couplings = couplings
        self.g = self.couplings[0]
        self.is_hermitian = True
        self.is_traceless = True

        def base_fn(X: torch.Tensor, *, model=self) -> torch.Tensor:
            return _fermion_det_log_identity_plus_sum_adX(X)

        if config.ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
            self._log_det_fn = torch.compile(base_fn, dynamic=False, backend=config.TORCH_COMPILE_BACKEND)
        else:
            self._log_det_fn = base_fn

    def load_fresh(self, args):
        X = torch.zeros((self.nmat, self.ncol, self.ncol), dtype=config.dtype, device=config.device)
        if self.source is not None:
            X = self.source / 2
        self.set_state(X)

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        trace_sq = torch.einsum("bij,bji->", X, X).real
        comm_term = -0.5 * _commutator_action_sum(X).real
        bos = trace_sq + comm_term

        det_coeff = torch.tensor((self.nmat - 2), dtype=config.real_dtype, device=X.device)
        det = -det_coeff * _fermion_det_log_identity_plus_sum_adX(X)

        src = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        if self.source is not None:
            src = -(self.ncol / self.g ** 0.5) * torch.einsum("iab,iba->", self.source, X)

        return (self.ncol / self.g) * bos + det + src.real

    def measure_observables(self, X: torch.Tensor | None = None):
        with torch.no_grad():
            X = self._resolve_X(X)
            eigs = [torch.linalg.eigvalsh(mat).cpu().numpy() for mat in X]
            eigs.append(torch.linalg.eigvals(X[0] + 1j * X[1]).cpu().numpy())
            comm_raw = _commutator_action_sum(X).real.item() / self.nmat / (self.nmat - 1) / self.ncol
            anticomm_raw = _anticommutator_action_sum(X).real.item() / self.nmat / (self.nmat - 1) / self.ncol

            # Moments tr(X_i X_j) to diagnose emergent rotational symmetry.
            moments = torch.einsum("aij,bji->ab", X, X).real
            corrs = np.concatenate(
                (
                    np.array([anticomm_raw, comm_raw], dtype=np.float64),
                    moments.detach().cpu().numpy().astype(np.float64).reshape(-1),
                )
            )
        return eigs, corrs

    def build_paths(self, name_prefix: str, data_path: str) -> dict[str, str]:
        run_dir = os.path.join(
            data_path,
            f"{name_prefix}_{self.model_name}_D{self.nmat}_g{round(self.g, 4)}_N{self.ncol}",
        )
        return {
            "dir": run_dir,
            "eigs": os.path.join(run_dir, "evals.npz"),
            "corrs": os.path.join(run_dir, "corrs.npz"),
            "meta": os.path.join(run_dir, "metadata.json"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }

    def extra_config_lines(self) -> list[str]:
        return [f"  Coupling g               = {self.g}", f"  Dimension D             = {self.nmat}"]

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        # return f"tr X_1^2 = {torch.trace(X[0] @ X[0]).real.item() / self.ncol:.5f} , tr X_{self.nmat}^2 = {torch.trace(X[-1] @ X[-1]).real.item() / self.ncol:.5f}"
        return 'tr X_i^2 = ' + ','.join([f"{torch.trace(X[i] @ X[i]).real.item() / self.ncol:.2f}" for i in range(self.nmat)])

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "has_source": self.source is not None,
                "model_variant": "adjoint_det",
            }
        )
        return meta
