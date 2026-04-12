"""Type I polarized IKKT model."""

from __future__ import annotations

import os

import numpy as np
import torch

from MatrixModelHMC_pytorch import config
from MatrixModelHMC_pytorch.algebra import (
    add_trace_projector_inplace,
    ad_matrix,
    get_eye_cached,
)
from MatrixModelHMC_pytorch.models.base import MatrixModel
from MatrixModelHMC_pytorch.models.utils import _commutator_action_sum
model_name = "pikkt4d_type1"


def build_model(args):
    return PIKKTTypeIModel(
        ncol=args.ncol,
        couplings=args.coupling,
        source=args.source,
        no_myers=getattr(args, "no_myers", False),
        eta=getattr(args, "eta", 1.0),
    )


def _type1_logdet_impl(X: torch.Tensor, A: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
    """Original implementation — kept for benchmarking against _type1_logdet_impl."""
    # eta rescales the constant fermion block: A -> eta*A and A^{-1} -> (1/eta)*A^{-1}.
    eta_t = torch.tensor(eta, dtype=X.dtype, device=X.device)
    adX = 1j * ad_matrix(X[:4])
    adX1, adX2, adX3, adX4 = adX
    i = torch.tensor(1j, dtype=X.dtype, device=X.device)
    twoi = torch.tensor(2j, dtype=X.dtype, device=X.device)

    upper_left = -(adX3 + i * adX4)
    upper_right = -(adX1 - i * adX2)
    lower_left = -(adX1 + i * adX2)
    lower_right = adX3 - i * adX4

    C = torch.cat(
        (torch.cat((upper_left, lower_left), dim=1), torch.cat((upper_right, lower_right), dim=1)),
        dim=0,
    )

    AB = torch.cat(
        (
            torch.cat((twoi * lower_left, twoi * lower_right), dim=1),
            torch.cat((-twoi * upper_left, -twoi * upper_right), dim=1),
        ),
        dim=0,
    )

    K = -eta_t * A - (0.25 / eta_t) * (C @ AB)

    # Lift zero modes from the trace sector (identity direction)
    # This adds a constant mass term to the trace mode, ensuring invertibility
    # without affecting the physics (since it's a constant factor in det).
    N = X.shape[-1]
    dim = N * N
    add_trace_projector_inplace(K[:dim, :dim], N)
    add_trace_projector_inplace(K[dim:, dim:], N)

    det = torch.slogdet(K)
    return det


def _type1_logdet_impl_changed(X: torch.Tensor, A: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
    """
    Optimized Dirac operator log-det.  See _type1_logdet_impl_ref for the original.

    Key changes vs the reference:

    1.  N×N pre-combination instead of N²×N² post-combination.
        The four Dirac blocks satisfy UL = ad(W2), LL = ad(W1), UR = ad(V1),
        LR = ad(V2) where W1,W2,V1,V2 are simple complex linear combinations of
        X1…X4 at the N×N level.  We form those first, then call ad_matrix once
        on the batch — skipping the four N²×N² linear-combination tensors that
        the reference builds after ad_matrix.

    2.  No C or AB intermediates.
        The reference materialises two (2N²)×(2N²) matrices (C and AB) and
        multiplies them.  For N=50 in complex64 that is two 200 MB allocations
        plus a 200 MB output.  Here we derive K's four N²×N² blocks directly:

            K₀₀ = c · (UL·LL − LL·UL),       c = −0.5i/η
            K₀₁ = −2iη·I  +  c · (UL·LR − LL·UR)
            K₁₀ = +2iη·I  +  c · (UR·LL − LR·UL)
            K₁₁ = c · (UR·LR − LR·UR)

        requiring eight N²×N² matmuls (same flop count as one (2N²)×(2N²) matmul
        but better cache behaviour and ~400 MB lower peak allocation at N=50).

    3.  Diagonal shift via .diagonal() view — no D×D identity tensor allocated.
    """
    N = X.shape[-1]
    D = N * N
    X1, X2, X3, X4 = X[0], X[1], X[2], X[3]

    # --- Step 1: N×N complex combinations ---
    # Derivation: each block equals (I⊗W − W^T⊗I) = ad_matrix(W) for these W.
    #   upper_left  = -(adX3 + i·adX4)  →  W2 = X4 − i·X3
    #   lower_left  = -(adX1 + i·adX2)  →  W1 = X2 − i·X1
    #   upper_right = -(adX1 − i·adX2)  →  V1 = −i·X1 − X2
    #   lower_right =  (adX3 − i·adX4)  →  V2 =  i·X3 + X4
    W2 = X4 - 1j * X3
    W1 = X2 - 1j * X1
    V1 = -1j * X1 - X2
    V2 =  1j * X3 + X4

    # --- Step 2: one batched ad_matrix call for all four D×D blocks ---
    adW = ad_matrix(torch.stack([W2, W1, V1, V2]))   # (4, D, D)
    UL, LL, UR, LR = adW.unbind(0)

    # --- Step 3: assemble K's blocks with eight D×D matmuls ---
    c = complex(0.0, -0.5 / eta)   # = −0.5i/η  (Python complex, no tensor alloc)

    K00 = torch.mm(UL, LL).sub_(torch.mm(LL, UL)).mul_(c)
    K11 = torch.mm(UR, LR).sub_(torch.mm(LR, UR)).mul_(c)
    K01 = torch.mm(UL, LR).sub_(torch.mm(LL, UR)).mul_(c)
    K10 = torch.mm(UR, LL).sub_(torch.mm(LR, UL)).mul_(c)

    # Add ±2iη on the diagonals of the off-diagonal blocks (no full identity tensor)
    K01.diagonal().add_(complex(0.0, -2.0 * eta))
    K10.diagonal().add_(complex(0.0,  2.0 * eta))

    # --- Step 4: assemble full K and lift trace zero-modes ---
    K = torch.empty(2 * D, 2 * D, dtype=X.dtype, device=X.device)
    K[:D, :D] = K00
    K[:D, D:] = K01
    K[D:, :D] = K10
    K[D:, D:] = K11

    add_trace_projector_inplace(K[:D, :D], N)
    add_trace_projector_inplace(K[D:, D:], N)

    return torch.slogdet(K)


class PIKKTTypeIModel(MatrixModel):
    """Type I polarized IKKT model definition."""

    model_name = model_name

    def __init__(self, ncol: int, couplings: list, source: np.ndarray | None = None, no_myers: bool = False, eta: float = 1.0) -> None:
        super().__init__(nmat=4, ncol=ncol)
        self.couplings = couplings
        self.g = self.couplings[0]
        self.source = torch.diag(torch.tensor(source, device=config.device, dtype=config.dtype)) if source is not None else None
        self.no_myers = no_myers
        self.eta = float(eta)
        self.is_hermitian = True
        self.is_traceless = True

        dim_tr = self.ncol * self.ncol
        eye_tr = get_eye_cached(dim_tr, device=config.device, dtype=config.dtype)
        i = 1j
        two_i_I = (2 * i) * eye_tr
        A = torch.zeros((2 * dim_tr, 2 * dim_tr), device=config.device, dtype=config.dtype)
        A[:dim_tr, dim_tr:] = two_i_I
        A[dim_tr:, :dim_tr] = -two_i_I
        self._type1_A = A.clone()

        def base_fn(X: torch.Tensor, *, model=self) -> torch.Tensor:
            return _type1_logdet_impl(X, model._type1_A, eta=model.eta)

        if config.ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
            self._log_det_fn = torch.compile(base_fn, dynamic=False, backend=config.TORCH_COMPILE_BACKEND)
        else:
            self._log_det_fn = base_fn

    def load_fresh(self, args):
        X = torch.zeros((self.nmat, self.ncol, self.ncol), dtype=config.dtype, device=config.device)
        self.set_state(X)

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        bos = -0.5 * _commutator_action_sum(X)
        trace_sq = torch.einsum("bij,bji->", X, X)
        bos = bos + trace_sq

        det = -0.5 * self._log_det_fn(X)[1].real
        src = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        if self.source is not None:
            src = -(self.ncol / np.sqrt(self.g)) * torch.trace(self.source @ X[0])

        return (bos.real * (self.ncol / self.g)) + det + src.real

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

            eigs.append(torch.linalg.eigvalsh(X[0] @ X[0] + X[1] @ X[1] + X[2] @ X[2]).cpu().numpy())

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
        eta_suffix = f"_eta{round(self.eta, 4)}" if self.eta != 1.0 else ""
        run_dir = os.path.join(
            data_path,
            f"{name_prefix}_{self.model_name}_g{round(self.g, 4)}{eta_suffix}_N{self.ncol}",
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
        trX1 = (torch.trace(X[0] @ X[0]) / self.ncol).item().real
        trX4 = (torch.trace(X[3] @ X[3]) / self.ncol).item().real
        return f"trX_1^2 = {trX1:.5f}, trX_4^2 = {trX4:.5f}. "

    def extra_config_lines(self) -> list[str]:
        return [
            f"  Coupling g               = {self.g}",
            f"  SUSY breaking parameter eta = {self.eta}",
        ]

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "has_source": self.source is not None,
                "model_variant": "type1",
                "eta": self.eta,
            }
        )
        return meta
