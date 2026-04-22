"""Model-agnostic Hybrid Monte Carlo kernels: leapfrog and Metropolis step."""

import math
import numpy as np
import random
from dataclasses import dataclass, replace
from typing import Any
import torch

try:
    from MatrixModelHMC_pytorch.algebra import random_hermitian
except ImportError:  # pragma: no cover
    from algebra import random_hermitian  # type: ignore

@dataclass
class HMCParams:
    """Integrator controls for an HMC trajectory."""
    dt: float
    nsteps: int
    

def hamil(X: torch.Tensor, mom_X: torch.Tensor, model: Any) -> float:
    """Total Hamiltonian = potential(X) + kinetic(momentum)."""
    ham = model.potential(X).item()
    kin = 0.0
    for j in range(model.nmat):
        Pj = mom_X[j]
        kin = kin + 0.5 * torch.trace(Pj @ Pj).real
    return ham + kin.item()


def leapfrog(X: torch.Tensor, hmc_params: HMCParams, model: Any) -> tuple[torch.Tensor, float, float]:
    """Symplectic leapfrog integrator returning the proposal and initial/final energies."""
    dt_local = hmc_params.dt
    begin_traj = getattr(model, "begin_trajectory", None)
    if callable(begin_traj):
        begin_traj(X)

    mom_X = random_hermitian(
        model.ncol,
        traceless=bool(model.is_traceless),
        batchsize=model.nmat,
    )
    ham_init = hamil(X, mom_X, model)

    X = X + 0.5 * dt_local * mom_X

    for _ in range(1, hmc_params.nsteps):
        f_X = model.force(X)
        mom_X = mom_X - dt_local * f_X
        X = X + dt_local * mom_X

    f_X = model.force(X)
    mom_X = mom_X - dt_local * f_X
    X = X + 0.5 * dt_local * mom_X

    ham_final = hamil(X, mom_X, model)
    return X, ham_init, ham_final


def update(acc_count: int, hmc_params: HMCParams, model: Any, reject_prob: float = 1.0):
    """Run one HMC trajectory and Metropolis accept/reject step, mutating model.X."""
    X = model.get_state()
    X_bak = X.clone()
    X_new, H0, H1 = leapfrog(X, hmc_params, model)
    dH = H1 - H0
    finite_h0 = np.isfinite(H0)
    finite_h1 = np.isfinite(H1)
    finite_dh = np.isfinite(dH)

    accept = bool(finite_h0 and finite_h1 and finite_dh)
    if accept and reject_prob > 0.0:
        r = random.uniform(0.0, reject_prob)
        if dH > 0.0:
            accept = (-dH) > math.log(r)

    if accept:
        model.set_state(X_new)
        acc_count += 1
        print(f"ACCEPT: dH={dH: 8.3f}, expDH={np.exp(-dH): 8.3f}, H0={H0: 8.4f}, ", model.status_string())
    else:
        model.set_state(X_bak)
        if finite_h0 and finite_h1 and finite_dh:
            print(f"REJECT: dH={dH: 8.3f}, expDH={np.exp(-dH): 8.3f}, H0={H0: 8.4f}, ", model.status_string())
        else:
            print(
                "REJECT: non-finite Hamiltonian encountered "
                f"(H0={H0}, H1={H1}, dH={dH}), ",
                model.status_string(),
            )

    end_traj = getattr(model, "end_trajectory", None)
    if callable(end_traj):
        end_traj(accept)

    return acc_count


def thermalize(model: Any, hmc_params: HMCParams, steps: int = 10) -> None:
    """Run short, mostly-accepting trajectories to move the system toward equilibrium."""
    print("Thermalization steps, accept most jumps")
    therm_params = replace(hmc_params, nsteps=int(hmc_params.nsteps * 2), dt=hmc_params.dt / 20.0)
    acc_count = 0
    for _ in range(steps):
        acc_count = update(acc_count, therm_params, model)
    print("End of thermalization ", model.status_string())

__all__ = [
    "HMCParams",
    "hamil",
    "leapfrog",
    "update",
    "thermalize",
]
