import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.optimize import curve_fit

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (5, 2.5)
# %matplotlib widget

DATA_PATH = Path("/mnt/beegfs/hmurali/ML/data")

class RunRecord:
    """Lightweight container for a single simulation run."""

    def __init__(self, run: str | Path, *, base_path: Path | str = DATA_PATH, load_checkpoint: bool = True) -> None:
        base = Path(base_path)
        path = Path(run)
        if not path.is_dir():
            path = base / run
        if not path.exists():
            raise FileNotFoundError(f"Run directory '{run}' not found under {base}")
        self.path = path
        self.metadata = self._load_json(path / "metadata.json")
        self.evals = self._load_npz(path / "evals.npz", empty=np.empty((0, 0, 0), dtype=np.complex128))
        self.corrs = self._load_npz(path / "corrs.npz", empty=np.empty((0, 0), dtype=np.complex128))
        self.mats = self._load_chunks(path / "all_mats")
        model_info = self.metadata.get("model", {})
        self.model_name = model_info.get("model_name")
        self.nmat = model_info.get("nmat")
        self.ncol = model_info.get("ncol") or (self.evals.shape[-1] if self.evals.ndim == 3 else None)
        self.couplings = model_info.get("couplings", [])
        self.g = self.couplings[0] if self.couplings else None
        self.omega = self.couplings[1] if len(self.couplings) > 1 else None
        self.block_names = self._infer_block_names()
        self.X: np.ndarray | None = None
        if load_checkpoint:
            self.load_checkpoint()

    @staticmethod
    def _load_npz(path: Path, *, empty: np.ndarray) -> np.ndarray:
        if not path.exists():
            return empty
        with np.load(path) as data:
            return data["values"]

    @staticmethod
    def _load_chunks(path: Path):
        if not path.exists():
            return None
        files = sorted(f for f in path.glob("*.npy") if f.is_file())
        if not files:
            return None
        return [np.load(f, mmap_mode="r", allow_pickle=False) for f in files]

    @staticmethod
    def _load_json(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_checkpoint(self) -> None:
        ckpt_path = self.path / "checkpoint.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.X = state["X"].to(dtype=torch.complex128).cpu().numpy()

    def ensure_checkpoint(self) -> None:
        if self.X is None:
            self.load_checkpoint()

    def _infer_block_names(self) -> list[str]:
        if self.evals.ndim < 2:
            return []
        count = self.evals.shape[1]
        key = (self.model_name or "").lower()
        if key in {"pikkt4d_type1", "pikkt4d_type2"}:
            defaults = ["X1", "X2", "X3", "X4", "X1+iX2", "X3+iX4", "Casimir"]
            return defaults[:count]
        if key == "yangmills" and self.nmat:
            return [f"X{i + 1}" for i in range(min(count, self.nmat))]
        if key == "adjoint_det" and self.nmat:
            labels = [f"X{i + 1}" for i in range(min(count, self.nmat))]
            if count > len(labels):
                labels.extend(f"block_{i}" for i in range(len(labels), count))
            return labels
        return [f"block_{i}" for i in range(count)]

    @property
    def n_measurements(self) -> int:
        return self.evals.shape[0] if self.evals.ndim else 0

    def _resolve_block_index(self, block: int | str) -> int:
        if isinstance(block, int):
            return block
        if block in self.block_names:
            return self.block_names.index(block)
        raise KeyError(f"Unknown block '{block}'")

    def plot_eval_hist(self, block: int | str = 0, *, idxs: list = [0,-1], burn_in: int = None, bins: int = 60, ax=None) -> None:
        if self.evals.size == 0:
            print("No eigenvalues saved for this run.")
            return
        if burn_in is not None:
            idxs = [burn_in, -1]
        idx = self._resolve_block_index(block)
        data = self.evals[idxs[0]:idxs[1], idx, :].real.reshape(-1)
        if data.size == 0:
            print("No samples after burn-in.")
            return
        if ax is None:
            ax = plt.gca()
        sns.histplot(data.real, bins=bins, stat="density", ax=ax)
        label = self.block_names[idx] if idx < len(self.block_names) else str(block)
        ax.set_title(f"{label} eigenvalue density")
        ax.set_xlabel("eigenvalue")

    def plot_trX2(self, block: int | str = 0, *, idxs: list = [0,-1], burn_in: int = None, ax=None) -> None:
        if self.evals.size == 0:
            print("No eigenvalues saved for this run.")
            return
        if burn_in is not None:
            idxs = [burn_in, -1]
        idx = self._resolve_block_index(block)
        data = self.evals[idxs[0]:idxs[1], idx, :].real
        series = (data ** 2).sum(axis=1)
        x = np.arange(idxs[0], idxs[0] + len(series))
        if ax is None:
            ax = plt.gca()
        ax.plot(x, series)
        label = self.block_names[idx] if idx < len(self.block_names) else str(block)
        ax.set_title(f"Tr(X^2) for {label}")
        ax.set_xlabel("measurement")
        ax.set_ylabel("Tr(X^2)")

        # from IPython.display import display
        # import ipywidgets as widgets

        # x_min = int(x[0])
        # x_max = int(x[-1])
        # line_color = "tab:red"
        # line_lo = ax.axvline(x_min, color=line_color, lw=1.5)
        # line_hi = ax.axvline(x_max, color=line_color, lw=1.5)
        # fig_hist, ax_hist = plt.subplots()

        # slider_lo = widgets.IntSlider(
        #     value=x_min,
        #     min=x_min,
        #     max=x_max,
        #     step=1,
        #     description="start",
        #     continuous_update=True,
        # )
        # slider_hi = widgets.IntSlider(
        #     value=x_max,
        #     min=x_min,
        #     max=x_max,
        #     step=1,
        #     description="end",
        #     continuous_update=True,
        # )

        # def _update_hist(*_args) -> None:
        #     lo = int(slider_lo.value)
        #     hi = int(slider_hi.value)
        #     if hi < lo:
        #         lo, hi = hi, lo
        #     line_lo.set_xdata([lo, lo])
        #     line_hi.set_xdata([hi, hi])
        #     ax_hist.clear()
        #     self.plot_eval_hist(6, idxs=[lo, hi + 1], ax=ax_hist)
        #     for j in np.arange(0, 2.5, 0.5):
        #         ax_hist.axvline((2/3 + 1)**2 * j * (j + 1), color='red', linestyle='--', alpha=0.7)
        #     fig.canvas.draw_idle()
        #     fig_hist.canvas.draw_idle()

        # slider_lo.observe(_update_hist, names="value")
        # slider_hi.observe(_update_hist, names="value")
        # display(widgets.VBox([slider_lo, slider_hi]))
        # _update_hist()

    def _rotated(self, basis: str) -> tuple[np.ndarray, np.ndarray]:
        self.ensure_checkpoint()
        assert self.X is not None
        key = basis.lower()
        if key == "x1":
            ref = self.X[0]
        elif key == "casimir":
            upto = min(3, self.X.shape[0])
            ref = sum(self.X[i] @ self.X[i] for i in range(upto))
        else:
            raise ValueError("basis must be 'X1' or 'casimir'")
        eigvals, eigvecs = np.linalg.eigh(ref)
        U = eigvecs.conj().T
        rotated = np.array([U @ Xi @ U.conj().T for Xi in self.X])
        return rotated, eigvals

    def iter_mats(self):
        if self.mats is None:
            return
        for chunk in self.mats:
            for mat in chunk:
                yield mat

    def get_mat(self, idx: int):
        if self.mats is None:
            return None
        for chunk in self.mats:
            if idx < chunk.shape[0]:
                return chunk[idx]
            idx -= chunk.shape[0]
        raise IndexError("idx out of range")

    def plot_rotated_matrices(
        self,
        matrices: tuple[int, ...] = (0, 1, 2, 3),
        bases: tuple[str, ...] = ("X1", "casimir"),
        cmap: str = "RdBu_r",
    ) -> None:
        matrices = tuple(matrices)
        bases = tuple(bases)
        rotated_cache = {}
        vmax = 0.0
        for basis in bases:
            rotated, _ = self._rotated(basis)
            rotated_cache[basis] = rotated
            if matrices:
                vmax = max(vmax, np.max(np.abs(rotated[list(matrices)])))
            else:
                vmax = max(vmax, np.max(np.abs(rotated)))
        fig, axes = plt.subplots(len(bases), len(matrices), figsize=(2 * len(matrices), 2 * len(bases)))
        if len(bases) == 1:
            axes = np.expand_dims(axes, axis=0)
        for row, basis in enumerate(bases):
            rotated = rotated_cache[basis]
            for col, idx in enumerate(matrices):
                ax = axes[row][col]
                mat = rotated[idx].real
                ax.matshow(np.abs(mat))
                # ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax)
                ax.set_title(f"{basis}: X_{idx+1}")
                ax.set_xticks([])
                ax.set_yticks([])
            axes[row][0].set_ylabel(f"{basis} basis")
        plt.tight_layout()
        plt.show()


def jackknife_error(values, window_size: int = 1):
    """Return mean and jackknife 1\sigma error for 1D samples with optional binning."""
    data = np.asarray(values, dtype=np.float64).ravel()
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    if window_size > 1:
        n_bins = data.size // window_size
        if n_bins < 2:
            raise ValueError("Not enough data for the requested window_size.")
        data = data[: n_bins * window_size].reshape(n_bins, window_size).mean(axis=1)

    n = data.size
    if n < 2:
        raise ValueError("Jackknife requires at least two samples after windowing.")

    stat = data.mean()
    jackknife_samples = (stat * n - data) / (n - 1)
    error = np.sqrt((n - 1) * np.mean((jackknife_samples - jackknife_samples.mean()) ** 2))
    return stat, error

def standardize(values):
    data = np.asarray(values.real, dtype=np.float64).ravel()
    mean = data.mean()
    std = data.std()
    if std == 0:
        return data - mean
    return (data - mean) / std

def count_spin_irreps(evals, omega=1.0, j_values=None, roundOut=False):
    if j_values is None:
        j_values = np.arange(0, 2.5, 0.5)
    j_values = np.asarray(j_values.real, dtype=float)
    targets = ((2/3 + omega) ** 2) * j_values * (j_values + 1)
    evals = np.asarray(evals.real, dtype=float).ravel()
    idx = np.abs(evals[:, None] - targets[None, :]).argmin(axis=1)
    counts = {float(j): (np.sum(idx == k) / (2 * j + 1)) for k, j in enumerate(j_values)}
    if roundOut:
        counts = {k: round(v) for k, v in counts.items()}
    return counts

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root
import matplotlib.pyplot as plt

def midpoint_density(x_sorted):
    n = len(x_sorted)
    dx = x_sorted[1:] - x_sorted[:-1]
    rho = 1.0 / (n * dx)
    xm = (x_sorted[1:] + x_sorted[:-1]) / 2.0
    return xm, rho

def residual_fast(z, ns, dims, lam, omega):
    xs = []
    k = 0
    for n in ns:
        xs.append(z[k:k+n])
        k += n

    Ntot = float(sum(n * d for n, d in zip(ns, dims)))

    res_blocks = []
    for s, (x_s, n_s, dim_s) in enumerate(zip(xs, ns, dims)):
        r = -omega * x_s.copy()

        for t, (x_t, n_t, dim_t) in enumerate(zip(xs, ns, dims)):
            diff = x_s[:, None] - x_t[None, :]          # (n_s, n_t)
            diff2 = diff * diff

            den = ((dim_s - dim_t) ** 2 + diff2) * ((dim_s + dim_t) ** 2 + diff2)

            # Safe divide: kernel = diff/den, but force kernel=0 where diff==0 (avoids 0/0 when a==0)
            ker = np.divide(diff, den, out=np.zeros_like(diff), where=(diff != 0.0))

            r += (8.0 * lam * n_t * dim_t) / Ntot * np.sum(ker, axis=1) / n_t

        res_blocks.append(r)

    out = np.concatenate(res_blocks)

    # optional sanity check (helps catch issues early)
    if not np.isfinite(out).all():
        raise FloatingPointError("residual produced NaN/Inf (try smaller initial L or continuation).")

    return out

def solve_system(ns, lam, omega, spins, seed=0, method="krylov", z0=None):
    rng = np.random.default_rng(seed)
    dims = [2*s + 1 for s in spins]
    if ns is None:
        ns = [100 for _ in spins]

    L = 3.0 * (lam ** 0.5)
    if z0 is None:
        z0 = np.concatenate([np.linspace(-L, L, n) + 0.02 * rng.standard_normal(n) for n in ns])

    fun = lambda z: residual_fast(z, ns, dims, lam, omega)
    if method == "hybr":
        opt = {}
    else:
        opt = {"maxiter": 200}
    sol = root(fun, z0, method=method, tol=1e-10,
               options=opt)

    xs = []
    k = 0
    for n in ns:
        xs.append(np.sort(sol.x[k:k+n]))
        k += n

    return xs, dims, ns, sol


def solve_recursive(ns, lam, omega=1.0, spins=(0, 1), seed=0, method="krylov", max_depth=6, quiet=False):
    """Recursively solve SPE by halving ns and quartering lambda on failure.
    Returns xs, dims, ns, sol for the target (ns, lam).
    """
    def _try_solve(ns_i, lam_i, z0=None):
        return solve_system(ns=ns_i, lam=lam_i, omega=omega, spins=spins,
                            seed=seed, method=method, z0=z0)

    # attempt direct solve
    xs, dims, ns, sol = _try_solve(ns, lam)
    if sol.success:
        if not quiet:
            print(f"Direct solve succeeded for ns={ns}, lam={lam}.")
        return xs, dims, ns, sol

    # backoff ladder
    ladder = [(ns, lam)]
    for _ in range(max_depth):
        ns = [max(4, int(n // 2)) for n in ns]
        lam = lam / 4.0
        ladder.append((ns, lam))
        xs, dims, ns, sol = _try_solve(ns, lam)
        if sol.success:
            if not quiet:
                print(f"Backoff solve succeeded for ns={ns}, lam={lam}.")
            break
    else:
        return xs, dims, ns, sol

    # forward continuation back to target using interpolation to expand z0
    for (ns_next, lam_next) in reversed(ladder[:-1]):
        # build an initial guess by interpolating each block to the new size
        z0_blocks = []
        for s, x in enumerate(xs):
            u_old = np.linspace(0.0, 1.0, len(x))
            u_new = np.linspace(0.0, 1.0, ns_next[s])
            z0_blocks.append(np.interp(u_new, u_old, x))
        z0 = np.concatenate(z0_blocks)
        xs, dims, ns, sol = _try_solve(ns_next, lam_next, z0=z0)
        if not quiet:
            print(f"Continuation solve for ns={ns_next}, lam={lam_next} "
                  f"{'succeeded' if sol.success else 'failed'} "
                    f"max|res|={np.max(np.abs(sol.fun)):.2e}")
        if not sol.success:
            break

    return xs, dims, ns, sol
# HMC convention helpers

def loc_params_from_hmc(N, g, omega_hmc):
    Omega1 = (N / g)**0.25
    Omega2 = omega_hmc * Omega1
    Omega = (2.0 / 3.0) * Omega1 + Omega2
    omega_loc = Omega2 / Omega
    lam = N / Omega**4
    return omega_loc, lam, Omega1, Omega2, Omega

def moments34(ns, spins, g, omega_hmc=1.0, moments=(0, 2), seed=0, method="krylov", max_depth=6, quiet=False):
    """Compute moments of X_3 + i X_4 in HMC conventions.

    Returns a dict mapping each n in `moments` to Tr (X_3 + i X_4)^n.
    """
    # drop zero elements in ns
    ns, spins = zip(*[(n, s) for n, s in zip(ns, spins) if n > 0])
    
    N = np.sum([n * (2 * s + 1) for n, s in zip(ns, spins)])
    omega, lam, Omega1, Omega2, Omega = loc_params_from_hmc(N, g, omega_hmc)
    xs, dims, ns, sol = solve_recursive(
        ns=ns, lam=lam, omega=omega, spins=spins, seed=seed, method=method, max_depth=max_depth, quiet=quiet
    )

    if not quiet:
        print("success:", sol.success, "| max|res|:", np.max(np.abs(sol.fun)))

    scale = Omega / Omega1
    moments_out = {}
    for n in moments:
        total = 0.0 + 0.0j
        for idx, s in enumerate(spins):
            M = np.asarray(xs[idx])
            m_vals = np.arange(-s, s + 1)
            term = (m_vals[None, :] + 1j * M[:, None]) ** n
            total += term.sum()
        moments_out[n] = (scale ** n) * total / N

    return moments_out
