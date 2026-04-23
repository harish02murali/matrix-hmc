"""Base class for matrix models used with the HMC driver."""

from __future__ import annotations

import os
from argparse import Namespace

import torch

from matrix_hmc import config
from matrix_hmc.algebra import get_eye_cached, makeH


class MatrixModel:
    """Base class for matrix models used with the HMC driver."""

    def __init__(self, nmat: int, ncol: int) -> None:
        self.nmat = nmat
        self.ncol = ncol
        self.couplings = None
        self.is_hermitian = None
        self.is_traceless = None
        self._X: torch.Tensor | None = None

    def _resolve_X(self, X: torch.Tensor | None = None) -> torch.Tensor:
        if X is not None:
            return X
        if self._X is None:
            raise ValueError("Model configuration has not been initialized")
        return self._X

    def set_state(self, X: torch.Tensor) -> None:
        self._X = X

    def get_state(self) -> torch.Tensor:
        return self._resolve_X()

    def load_fresh(self, args: Namespace) -> None:
        """Load a fresh configuration (zero matrices)."""
        X = torch.zeros((self.nmat, self.ncol, self.ncol), dtype=config.dtype, device=config.device)
        self.set_state(X)

    def initialize_configuration(self, args: Namespace, ckpt_path: str) -> bool:
        if args.resume:
            if os.path.isfile(ckpt_path):
                print("Reading old configuration file:", ckpt_path)
                ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=True)
                self.set_state(ckpt["X"].to(dtype=config.dtype, device=config.device))
                return True
            else:
                print("Configuration not found, loading fresh")
        else:
            print("Loading fresh configuration")

        self.load_fresh(args)

        return False

    def save_state(self, ckpt_path: str) -> None:
        torch.save({"X": self.get_state()}, ckpt_path)

    def force(self, X: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the force dV/dX, assuming X stays Hermitian when configured."""
        X = self._resolve_X(X)
        Y = X.detach().requires_grad_(True)
        pot = self.potential(Y)
        pot.backward()
        res = Y.grad
        if self.is_hermitian:
            res = makeH(res)
        if self.is_traceless:
            trs = torch.diagonal(res, dim1=-2, dim2=-1).sum(-1).real / self.ncol
            eye = get_eye_cached(self.ncol, device=res.device, dtype=res.dtype)
            res = res - trs[..., None, None] * eye
        return res

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        avg_tr = (torch.einsum("bij,bji->", X, X).real / (self.nmat * self.ncol)).item()
        return f"trX_i^2 = {avg_tr:.5f}. "
    
    def build_paths(self, name_prefix: str, data_path: str) -> dict[str, str]:
        model_name = getattr(self, "model_name", self.__class__.__name__.lower())
        run_dir = os.path.join(
            data_path,
            f"{name_prefix}_{model_name}_D{self.nmat}_N{self.ncol}",
        )
        return {
            "dir": run_dir,
            "eigs": os.path.join(run_dir, "evals.npz"),
            "corrs": os.path.join(run_dir, "corrs.npz"),
            "meta": os.path.join(run_dir, "metadata.json"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }
    
    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def measure_observables(self, X: torch.Tensor | None = None) -> tuple:
        raise NotImplementedError

    def extra_config_lines(self) -> list[str]:
        """Optional human-readable configuration lines for logging."""
        return []

    def run_metadata(self) -> dict[str, object]:
        return {
            "model_name": getattr(self, "model_name", self.__class__.__name__.lower()),
            "nmat": self.nmat,
            "ncol": self.ncol,
            "couplings": self.couplings,
            "dtype": str(config.dtype),
        }
