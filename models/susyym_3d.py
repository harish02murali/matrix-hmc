"""Massive Yang-Mills in 3D with adjoint fermions, with SUSY-breaking mass deformation."""

from __future__ import annotations

import os

import numpy as np
import torch

from MatrixModelHMC_pytorch import config
from MatrixModelHMC_pytorch.algebra import (
	add_trace_projector_inplace,
	ad_matrix,
	get_eye_cached,
	random_hermitian,
)
from MatrixModelHMC_pytorch.models.base import MatrixModel
from MatrixModelHMC_pytorch.models.utils import (
	_anticommutator_action_sum,
	_commutator_action_sum,
	parse_source,
)

model_name = "susyym_3d"

# The 3 matrix model action is S = N/g (-0.5 * _commutator_action_sum(X) + tr X^2 - 0.5 * log|det(M)|
# where M = [[1/2 (-i adX[0] + adX[2]), identity + 1/2 i adX[1]],
#            [-identity + 1/2 i adX[1], 1/2 (i adX[0] + adX[2])]].


def build_model(args):
	return SUSYYM3DModel(
		ncol=args.ncol,
		couplings=args.coupling,
		source=args.source,
		fermion_mass=getattr(args, "fermion_mass", 1.0),
		boson_mass=getattr(args, "boson_mass", 1.0),
	)


class SUSYYM3DModel(MatrixModel):
	"""3D supersymmetric Yang-Mills-inspired matrix model with massive adjoint fermions."""

	model_name = model_name

	def __init__(
		self,
		ncol: int,
		couplings: list,
		source: np.ndarray | None = None,
		fermion_mass: float = 1.0,
		boson_mass: float = 1.0,
	) -> None:
		super().__init__(nmat=3, ncol=ncol)
		self.couplings = couplings
		self.g = self.couplings[0]
		self.fermion_mass = float(fermion_mass)
		self.boson_mass = float(boson_mass)
		self.source = parse_source(source, self.nmat, config.device, config.dtype)
		self.is_hermitian = True
		self.is_traceless = True

	def load_fresh(self, args):
		mats = [0.01 * random_hermitian(self.ncol, traceless=self.is_traceless) for _ in range(self.nmat)]
		X = torch.stack(mats, dim=0).to(dtype=config.dtype, device=config.device)
		if self.source is not None:
			X = self.source / 2
		self.set_state(X)

	def fermion_matrix(self, X: torch.Tensor) -> torch.Tensor:
		adX0, adX1, adX2 = ad_matrix(X[:3])
		dim = self.ncol * self.ncol
		eye = get_eye_cached(dim, device=X.device, dtype=X.dtype)

		if self.fermion_mass == 0.0: # krauth-nicolai-staudacher basis for gamama matrices
			upper_left = (adX2 + 1j * adX1)
			upper_right = 1j * adX0
			lower_left = 1j * adX0
			lower_right = (adX2 - 1j * adX1)
		else:
			upper_left = 0.5 * (-1j * adX0 + adX2)
			upper_right = self.fermion_mass * eye + 0.5j * adX1
			lower_left = -self.fermion_mass * eye + 0.5j * adX1
			lower_right = 0.5 * (1j * adX0 + adX2)

		add_trace_projector_inplace(upper_left, self.ncol)
		add_trace_projector_inplace(lower_right, self.ncol)

		return torch.cat((torch.cat((upper_left, upper_right), dim=1), 
						torch.cat((lower_left, lower_right), dim=1)), dim=0)

	def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
		X = self._resolve_X(X)
		bos = -0.5 * _commutator_action_sum(X)
		if self.boson_mass != 0.0:
			bos += self.boson_mass * torch.einsum("bij,bji->", X, X)
		fermion = -0.5 * torch.slogdet(self.fermion_matrix(X))[1].real

		src = torch.tensor(0.0, dtype=X.dtype, device=X.device)
		if self.source is not None:
			src = -(self.ncol / self.g ** 0.5) * torch.einsum("iab,iba->", self.source, X)

		return (self.ncol / self.g) * bos.real + fermion + src.real

	def measure_observables(self, X: torch.Tensor | None = None):
		with torch.no_grad():
			X = self._resolve_X(X)
			eigs = [torch.linalg.eigvalsh(mat).cpu().numpy() for mat in X]
			eigs.append(torch.linalg.eigvals(X[0] + 1j * X[1]).cpu().numpy())

			comm_raw = _commutator_action_sum(X).real.item() / self.nmat / (self.nmat - 1) / self.ncol
			anticomm_raw = _anticommutator_action_sum(X).real.item() / self.nmat / (self.nmat - 1) / self.ncol
			moments = torch.einsum("aij,bji->ab", X, X).real

			corrs = np.concatenate(
				(
					np.array([anticomm_raw, comm_raw], dtype=np.float64),
					moments.detach().cpu().numpy().astype(np.float64).reshape(-1),
				)
			)

		return eigs, corrs

	def build_paths(self, name_prefix: str, data_path: str) -> dict[str, str]:
		mf_suffix = f"_mf{round(self.fermion_mass, 4)}" if self.fermion_mass != 1.0 else ""
		run_dir = os.path.join(
			data_path,
			f"{name_prefix}_{self.model_name}_g{round(self.g, 4)}{mf_suffix}_N{self.ncol}",
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
		return "tr X_i^2 = " + ",".join(
			[f"{torch.trace(X[i] @ X[i]).real.item() / self.ncol / np.sqrt(self.g):.2f}" for i in range(self.nmat)]
		)

	def extra_config_lines(self) -> list[str]:
		return [
			f"  Coupling g               = {self.g}",
			f"  Fermion mass deformation = {self.fermion_mass}",
			f"  Boson mass deformation   = {self.boson_mass}",
		]

	def run_metadata(self) -> dict[str, object]:
		meta = super().run_metadata()
		meta.update(
			{
				"has_source": self.source is not None,
				"model_variant": "susyym_3d",
				"fermion_mass": self.fermion_mass,
				"boson_mass": self.boson_mass,
			}
		)
		return meta
