"""Adjoint determinant matrix model."""

from __future__ import annotations

import os

import numpy as np
import torch

from matrix_hmc import config
from matrix_hmc.models.base import MatrixModel
from matrix_hmc.models.utils import (
    _anticommutator_action_sum,
    _commutator_action_sum,
    _fermion_det_log_identity_plus_sum_adX,
    parse_source,
)
from matrix_hmc.algebra import random_hermitian
model_name = "adjoint_det"

def build_model(args):
    if args.nmat is None:
        raise ValueError("--nmat must be provided for adjoint_det model")
    num_fermions = args.num_fermions if args.num_fermions is not None else 2 * (args.nmat - 2)
    return AdjointDetModel(
        dim=args.nmat,
        ncol=args.ncol,
        couplings=args.coupling,
        source=args.source,
        num_fermions=num_fermions,
        massless=args.massless,
    )


class AdjointDetModel(MatrixModel):
    r"""Matrix model whose fermion sector is ``det(1 + i \sum_i \mathrm{ad}_{\sqrt{X^2}})``.

    The action combines a Yang-Mills bosonic term with a simplified fermionic
    determinant that avoids the full Pfaffian computation:

    .. math::

        S = \frac{N}{g} \left[ \mathrm{Tr}({\sum_i X_i^2})
            - \frac{1}{2} \sum_{i<j} \mathrm{Tr}([X_i, X_j]^2) \right]
            - \frac{N_f}{2} \sum_{a<b} \log\!\left(1 + (\mu_a - \mu_b)^2\right)

    With ``--massless`` the mass term ``\mathrm{Tr}(\sum_i X_i^2)`` is dropped,
    the log-det becomes ``\log(\mu_a - \mu_b)^2``, and ``g`` is fixed to 1.

    Args:
        dim: Number of matrices ``D``.
        ncol: Matrix size ``N``.
        couplings: List of couplings; ``couplings[0]`` is ``g``.
        source: Optional external source (see
            :func:`~matrix_hmc.models.utils.parse_source`).
        num_fermions: Number of adjoint fermions ``N_f``.
            Default ``2*(dim-2)``.
        massless: Drop ``Tr(X²)`` mass term and replace ``log(1+Δ²)`` with
            ``log(Δ²)`` in the fermion determinant. Forces ``g=1``.
    """

    model_name = model_name

    def __init__(self, dim: int, ncol: int, couplings: list, source: np.ndarray | None = None, num_fermions: int | None = None, massless: bool = False) -> None:
        super().__init__(nmat=dim, ncol=ncol)
        self.source = parse_source(source, dim, config.device, config.dtype)
        self.couplings = couplings
        self.massless = massless
        self.g = 1.0 if massless else self.couplings[0]
        self.num_fermions = num_fermions if num_fermions is not None else 2 * (dim - 2)
        self.is_hermitian = True
        self.is_traceless = True

    def load_fresh(self):
        # X = torch.zeros((self.nmat, self.ncol, self.ncol), dtype=config.dtype, device=config.device)
        X = 0.01 * random_hermitian(self.ncol, traceless=self.is_traceless, batchsize=self.nmat)
        if self.source is not None:
            X = self.source / 2
        self.set_state(X)

    def _fermion_det(self, X: torch.Tensor) -> torch.Tensor:
        sum_X2 = (X @ X).sum(dim=0)
        eigvals = torch.sqrt(torch.linalg.eigvalsh(sum_X2).real.to(dtype=config.real_dtype))
        N = eigvals.shape[0]
        i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=eigvals.device)
        delta = eigvals[j_idx] - eigvals[i_idx]   # >= 0 since eigvalsh returns sorted ascending
        if self.massless:
            return torch.log(delta * delta).sum()
        return torch.log(1 + delta * delta).sum()

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        sum_X2 = (X @ X).sum(dim=0)
        comm_term = -0.5 * _commutator_action_sum(X).real
        if self.massless:
            bos = comm_term
        else:
            trace_sq = torch.einsum("ii->", sum_X2).real
            bos = trace_sq + comm_term

        det = -self.num_fermions * (0.5 * self._fermion_det(X))

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
        return 'tr X_i^2 = ' + ','.join([f"{torch.trace(X[i] @ X[i]).real.item() / self.ncol / np.sqrt(self.g):.2f}" for i in range(self.nmat)])

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "has_source": self.source is not None,
                "model_variant": "adjoint_det",
                "num_fermions": self.num_fermions,
                "massless": self.massless,
            }
        )
        return meta
