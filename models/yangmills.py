"""D-dimensional Yang-Mills matrix model."""

from __future__ import annotations

import os

import numpy as np
import torch

from matrix_hmc import config
from matrix_hmc.models.base import MatrixModel
from matrix_hmc.models.utils import _commutator_action_sum, _anticommutator_action_sum, parse_source

model_name = "yangmills"


def build_model(args):
    if args.nmat is None:
        raise ValueError("--nmat must be provided for yangmills model")
    return YangMillsModel(
        dim=args.nmat,
        ncol=args.ncol,
        couplings=args.coupling,
        source=args.source,
        mass=getattr(args, "mass", 1.0),
    )


class YangMillsModel(MatrixModel):
    """D-dimensional Yang-Mills matrix model."""

    model_name = model_name

    def __init__(self, dim: int, ncol: int, couplings: list, source: np.ndarray | None = None, mass: float = 1.0) -> None:
        super().__init__(nmat=dim, ncol=ncol)
        self.source = parse_source(source, self.nmat, config.device, config.dtype)
        self.couplings = couplings
        self.is_hermitian = True
        self.is_traceless = True
        self.g = self.couplings[0]
        self.mass = mass

    def load_fresh(self, args):
        X = torch.zeros((self.nmat, self.ncol, self.ncol), dtype=config.dtype, device=config.device)
        self.set_state(X)

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        trace_sq = torch.einsum("bij,bji->", X, X).real
        mass_term = self.mass * trace_sq
        comm_term = -0.5 * _commutator_action_sum(X).real
        src = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        if self.source is not None:
            src = -(self.ncol / self.g ** 0.5) * torch.einsum("iab,iba->", self.source, X)
        return (self.ncol / self.g) * (mass_term + comm_term) + src.real

    def measure_observables(self, X: torch.Tensor | None = None):
        X = self._resolve_X(X)
        eigs = [torch.linalg.eigvalsh(mat).cpu().numpy() for mat in X]
        eigs = eigs + [torch.linalg.eigvals(X[0] + 1j * X[1]).cpu().numpy()]
        eigs = eigs + [torch.linalg.eigvals(X[0] @ X[1] - X[1] @ X[0]).cpu().numpy()]
        comm_raw = _commutator_action_sum(X).real.item() / self.nmat / (self.nmat - 1) / self.ncol
        anticomm_raw = _anticommutator_action_sum(X).real.item() / self.nmat / (self.nmat - 1) / self.ncol
        corrs = np.array([anticomm_raw, comm_raw], dtype=np.float64)
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

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        avg_tr = (torch.einsum("bij,bji->", X, X).real / (self.nmat * self.ncol)).item()
        return f"trX_i^2 = {avg_tr:.5f}. "

    def extra_config_lines(self) -> list[str]:
        return [f"Coupling g               = {self.g}", f"Dimension D             = {self.nmat}"]

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "has_source": self.source is not None,
                "model_variant": "yangmills",
            }
        )
        return meta
