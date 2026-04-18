"""Command Line Interface parsing helpers for the D=4 pIKKT HMC driver."""

import argparse
import math
import os
from typing import Sequence

import numpy as np

root_path = ("/mnt/beegfs/hmurali/ML" if os.path.isdir('/mnt/beegfs/hmurali/ML') else "../")
DEFAULT_DATA_PATH = os.path.join(root_path, "data")
DEFAULT_PROFILE = False


def _parse_source(expr: str):
    """Evaluate a numpy-based expression like np.linspace(-1,1,20) for --source."""
    try:
        return eval(expr, {"np": np}, {})  # noqa: S307 - controlled namespace for numpy expressions
    except Exception as exc:  # pragma: no cover - arg parsing error path
        raise argparse.ArgumentTypeError(f"Invalid --source expression {expr!r}: {exc}") from exc


def _default_nmat_for_model(model_lower: str) -> int | None:
    if model_lower in {"pikkt4d_type1", "pikkt4d_type2", "pikkt4d_type2_rhmc"}:
        return 4
    if model_lower == "pikkt10d":
        return 10
    if model_lower == "susyym_3d":
        return 3
    return None


def _validate_source_shape(source: np.ndarray, ncol: int, expected_nmat: int | None) -> None:
    s = np.asarray(source)
    if s.ndim == 1:
        if s.shape != (ncol,):
            raise ValueError(f"--source 1D expression must evaluate to shape ({ncol},), got {s.shape}")
        return

    if s.ndim != 3:
        raise ValueError(
            f"--source expression must evaluate to shape ({ncol},) or (nmat,{ncol},{ncol}), got {s.shape}"
        )

    if s.shape[1:] != (ncol, ncol):
        raise ValueError(
            f"--source 3D expression must evaluate to shape (nmat,{ncol},{ncol}), got {s.shape}"
        )
    if expected_nmat is not None and s.shape[0] != expected_nmat:
        raise ValueError(
            f"--source 3D expression must have first dimension nmat={expected_nmat}, got {s.shape[0]}"
        )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """User-facing CLI with flags."""
    parser = argparse.ArgumentParser(
        description="Choosing the matrix model and simulation parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Matrix model name registered in the models package (e.g., 1mm, "
            "pikkt4d_type1, pikkt4d_type2, pikkt4d_type2_rhmc, pikkt10d, yangmills, susyym_3d)"
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Load a checkpoint if present")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoints and start from zero fields")
    parser.add_argument("--save", action="store_true", help="Save configurations every --save-every trajectories")
    parser.add_argument("--saveAllMats", action="store_true", help="Store raw matrix configurations every --save-every trajectories")
    parser.add_argument("--ncol", type=int, default=4, help="Matrix size N")
    parser.add_argument("--nmat", type=int, default=None, help="Number of matrices")
    parser.add_argument("--niters", type=int, default=300, help="Number of trajectories to run")
    parser.add_argument("--coupling", type=float, nargs="+", default=[100.0], help="Coupling g (can specify multiple, depending on model)")
    parser.add_argument("--no-gpu", action="store_true", dest="noGPU", default=False, help="Disable CUDA GPU even if available")
    parser.add_argument("--complex64", action="store_true", help="Use complex64/float32 precision instead of complex128/float64")
    parser.add_argument("--threads", type=int, default=None, help="Set torch intra-op CPU thread count")
    parser.add_argument("--interop-threads", type=int, default=None, help="Set torch inter-op CPU thread count")
    parser.add_argument("--name", type=str, default="run", help="Prefix for outputs")
    parser.add_argument("--step-size", type=float, dest="step_size", default=2, help="Leapfrog step size Δt")
    parser.add_argument("--nsteps", type=int, default=180, help="Leapfrog steps per trajectory")
    parser.add_argument("--save-every", type=int, dest="save_every", default=10, help="Write observables every K trajectories")
    parser.add_argument("--data-path", type=str, dest="data_path", default=DEFAULT_DATA_PATH, help="Directory for outputs and checkpoints")
    parser.add_argument("--profile", dest="profile", action="store_true", default=DEFAULT_PROFILE, help="Enable cProfile sampling")
    parser.add_argument("--no-profile", dest="profile", action="store_false", help="Disable cProfile sampling")
    parser.add_argument("--seed", type=int, default=None, help="Set RNG seed for reproducibility")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files and checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved configuration and exit")
    parser.add_argument("--source", type=_parse_source, default=None, help="Numpy expression for source, e.g., np.linspace(-1,1,20)")
    ym_group = parser.add_argument_group(
        "Yang-Mills options", "Relevant for --model yangmills"
    )
    ym_group.add_argument(
        "--mass",
        type=float,
        default=1.0,
        help="Coefficient of Tr(X_i^2) mass term (mass=1 is standard Yang-Mills)",
    )
    susy3d_group = parser.add_argument_group(
        "SUSY YM 3D options", "Relevant for --model susyym_3d"
    )
    susy3d_group.add_argument(
        "--fermion-mass",
        type=float,
        default=1.0,
        dest="fermion_mass",
        help="Adjoint fermion mass deformation in the 3D SUSY model",
    )
    susy3d_group.add_argument(
        "--boson-mass",
        type=float,
        default=1.0,
        dest="boson_mass",
        help="Coefficient of Tr(X_i^2) boson mass term in the 3D SUSY model",
    )
    type1_group = parser.add_argument_group(
        "Type I options", "Relevant for --model pikkt4d_type1"
    )
    type1_group.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="Type I fermion deformation parameter eta (eta=1 is undeformed SUSY)",
    )
    type1_group.add_argument(
        "--massless",
        action="store_true",
        help="Set Omega=0 in the Type I fermion matrix (massless limit)",
    )
    type2_group = parser.add_argument_group(
        "Type II / 10D options", "Relevant for --model pikkt4d_type2 and --model pikkt10d"
    )
    type2_group.add_argument(
        "--spin",
        type=float,
        default=None,
        help="Spin for the fuzzy-sphere background in X1,X2,X3",
    )
    type2_group.add_argument("--bosonic", action="store_true", help="Disable fermionic determinant term")
    type2_group.add_argument(
        "--lorentzian",
        action="store_true",
        help="Replace X4 -> i X4 in the potential (and corresponding force)",
    )
    rhmc_group = parser.add_argument_group(
        "RHMC options", "Relevant for --model pikkt4d_type2_rhmc"
    )
    rhmc_group.add_argument("--rhmc-order", type=int, default=20, help="Number of partial-fraction shifts")
    rhmc_group.add_argument(
        "--rhmc-lmin",
        type=float,
        default=None,
        help="Lower spectral bound for RHMC rational fit (auto-probed if omitted)",
    )
    rhmc_group.add_argument(
        "--rhmc-lmax",
        type=float,
        default=None,
        help="Upper spectral bound for RHMC rational fit (auto-probed if omitted)",
    )
    rhmc_group.add_argument("--rhmc-cg-tol", type=float, default=1e-8, help="Shifted CG relative tolerance")
    rhmc_group.add_argument("--rhmc-cg-maxiter", type=int, default=400, help="Shifted CG maximum iterations")
    adjdet2_group = parser.add_argument_group(
        "adjoint_det2 options", "Relevant for --model adjoint_det2"
    )
    adjdet2_group.add_argument(
        "--det-coeff",
        type=float,
        default=0.5,
        dest="det_coeff",
        help="Coefficient in the fermion determinant term: det = -det_coeff * (nmat-2) * log|det(...)|",
    )
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Clamp obviously invalid inputs early to avoid cryptic runtime failures."""
    if args.ncol < 1:
        raise ValueError("--ncol must be positive")
    if args.niters < 1:
        raise ValueError("--niters must be positive")
    if len(args.coupling) == 0:
        raise ValueError("--coupling requires at least one value")
    model_lower = args.model.lower()
    if model_lower == "1mm":
        if len(args.coupling) < 1:
            raise ValueError("1mm model requires at least one coupling via --coupling t1 [t2 ...]")
    if model_lower == "pikkt4d_type1":
        if len(args.coupling) != 1:
            raise ValueError("pIKKT Type I requires exactly one coupling g via --coupling g")
        if not math.isfinite(args.eta) or args.eta == 0.0:
            raise ValueError("--eta must be finite and non-zero for pIKKT Type I")
    if model_lower == "pikkt4d_type2" and len(args.coupling) != 2:
        raise ValueError("pIKKT Type II requires exactly two couplings via --coupling g omega")
    if model_lower == "pikkt4d_type2_rhmc":
        if len(args.coupling) != 2:
            raise ValueError("pIKKT Type II RHMC requires exactly two couplings via --coupling g omega")
        if args.rhmc_order < 1:
            raise ValueError("--rhmc-order must be positive")
        if (args.rhmc_lmin is None) ^ (args.rhmc_lmax is None):
            raise ValueError("Specify both --rhmc-lmin and --rhmc-lmax, or neither for auto-probe")
        if args.rhmc_lmin is not None and args.rhmc_lmin <= 0:
            raise ValueError("--rhmc-lmin must be > 0")
        if args.rhmc_lmax is not None and args.rhmc_lmax <= args.rhmc_lmin:
            raise ValueError("--rhmc-lmax must be > --rhmc-lmin")
        if args.rhmc_cg_tol <= 0:
            raise ValueError("--rhmc-cg-tol must be > 0")
        if args.rhmc_cg_maxiter < 1:
            raise ValueError("--rhmc-cg-maxiter must be positive")
    if model_lower == "pikkt10d":
        if len(args.coupling) != 1:
            raise ValueError("pikkt10d requires exactly one coupling g via --coupling g")
        if args.nmat is not None and args.nmat != 10:
            raise ValueError("pikkt10d has fixed dimension D=10; omit --nmat or set --nmat 10")
    if model_lower == "yangmills":
        if len(args.coupling) != 1:
            raise ValueError("Yang-Mills model requires a single coupling g via --coupling g")
        if args.nmat < 2:
            raise ValueError("--nmat must be atleast 2 for Yang-Mills model")
        if not math.isfinite(args.mass):
            raise ValueError("--mass must be finite")
    if model_lower == "susyym_3d":
        if len(args.coupling) != 1:
            raise ValueError("susyym_3d requires exactly one coupling g via --coupling g")
        if args.nmat is not None and args.nmat != 3:
            raise ValueError("susyym_3d has fixed dimension D=3; omit --nmat or set --nmat 3")
        if not math.isfinite(args.fermion_mass):
            raise ValueError("--fermion-mass must be finite")
        if not math.isfinite(args.boson_mass):
            raise ValueError("--boson-mass must be finite")
    if model_lower == "adjoint_det":
        if len(args.coupling) != 1:
            raise ValueError("adjoint_det model requires a single coupling g via --coupling g")
        if args.nmat is None or args.nmat < 1:
            raise ValueError("--nmat must be provided and positive for adjoint_det model")
    if args.nsteps < 1:
        raise ValueError("--nsteps must be positive")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    if args.save_every < 1:
        raise ValueError("--save-every must be positive")
    if args.threads is not None and args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.interop_threads is not None and args.interop_threads < 1:
        raise ValueError("--interop-threads must be positive")
    if args.source is not None:
        expected_nmat = args.nmat if args.nmat is not None else _default_nmat_for_model(model_lower)
        _validate_source_shape(args.source, args.ncol, expected_nmat)


__all__ = ["parse_args", "DEFAULT_DATA_PATH", "DEFAULT_PROFILE"]
