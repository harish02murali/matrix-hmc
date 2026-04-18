# MatrixModelHMC_pytorch

PyTorch implementation of Hybrid Monte Carlo (HMC) for several matrix models, including polarized IKKT variants.  
The code is model-driven: each file in `models/` defines one model via `build_model(args)`, while `hmc.py` and `main.py` provide shared integrator + I/O logic.

## Requirements

- Python `>=3.9`
- `torch`
- `numpy`

Install in editable mode:

```bash
pip install -e .
```

## Available Models
**Four dimensional Models**

Type I, $SO(4)$ invariant, with coupling constant $g$ and optional fermion deformation parameter $\eta$:

$$
S_{D=4,\text{type I}}=\frac{1}{g}\text{Tr}\Bigl[-\frac14 [X_I,X_J]^2 -\frac{i}{2}\bar\psi \Gamma^I [X_I,\psi] + X_I^2 + \eta\,\bar\psi\psi \Bigr]
$$

Type II, $SO(3)$ invariant, two coupling constants $g$ and $\omega$:

$$
\begin{aligned}
S_{D=4,\text{type II}}=\frac{1}{g}\text{Tr}\Bigl[&-\frac14 [X_I,X_J]^2 -\frac{i}{2}\bar\psi \Gamma^I [X_I,\psi] + i \frac{2}{3}(1+\omega) \epsilon_{ijk} X_i X_j X_k \\
&+ \frac{1}{3}\left(\omega + \frac{2}{3}\right) X_i X_i + \frac{\omega}{3} X_4^2 - \frac{1}{3} \bar\psi \Gamma^{123} \psi \Bigr]
\end{aligned}
$$

In addition to these two, the repository includes a one-matrix model with polynomial potential, 10D bosonic pIKKT model, a variable-dimension Yang-Mills model, and an adjoint determinant model. Each has its own coupling structure and optional flags as described below.

Model selection is dynamic: `--model <name>` loads `models/<name>.py`.

- `pikkt4d_type1`
  - Fixed `D=4`
  - Couplings: `--coupling g`
  - Extra flag: `--eta` (default `1.0`)
- `pikkt4d_type2`
  - Fixed `D=4`
  - Couplings: `--coupling g omega`
  - Extra flags: `--spin`, `--bosonic`, `--lorentzian`
- `pikkt10d`
  - Fixed `D=10`
  - Couplings: `--coupling g`
  - Current implementation uses the bosonic sector with `Omega=1`; fermion determinant is a placeholder returning `0`
- `yangmills`
  - Variable dimension
  - Requires `--nmat D`
  - Couplings: `--coupling g`
- `adjoint_det`
  - Variable dimension
  - Requires `--nmat D`
  - Couplings: `--coupling g`
- `susyym_3d`
  - Fixed `D=3`
  - Couplings: `--coupling g`
  - Extra flag: `--fermion-mass` (default `1.0`)
- `1mm`
  - Single-matrix polynomial model
  - Couplings: `--coupling t1 [t2 ...]`

## Running

Basic Type I run:

```bash
python main.py --model pikkt4d_type1 --ncol 10 --niters 300 --coupling 100.0 --eta 1.0 --name runA --data-path outputs
```

Type II run:

```bash
python main.py --model pikkt4d_type2 --ncol 10 --niters 300 --coupling 100.0 1.0 --name runB --data-path outputs
```

10D pIKKT run:

```bash
python main.py --model pikkt10d --ncol 10 --niters 300 --coupling 0.1 --step-size 0.1 --nsteps 300 --name run10d --data-path outputs
```

Yang-Mills in `D=6`:

```bash
python main.py --model yangmills --nmat 6 --ncol 12 --niters 200 --coupling 50.0 --name ymRun --data-path outputs
```

Adjoint determinant model:

```bash
python main.py --model adjoint_det --nmat 6 --ncol 12 --niters 200 --coupling 50.0 --name adRun --data-path outputs
```

3D SUSY Yang-Mills-inspired model:

```bash
python main.py --model susyym_3d --ncol 12 --niters 300 --coupling 60.0 --fermion-mass 1.0 --name susy3d --data-path outputs
```

Resume a run (same model/name/path configuration):

```bash
python main.py --model pikkt4d_type2 --coupling 100.0 1.0 --resume --name runB --data-path outputs
```

Dry-run config check (no trajectories):

```bash
python main.py --model pikkt10d --ncol 10 --coupling 0.1 --dry-run --data-path outputs
```

## CLI Notes

Important flags:

- `--model`: required model name.
- `--ncol`: matrix size `N`.
- `--coupling`: model-dependent couplings.
- `--niters`: number of trajectories.
- `--step-size`, `--nsteps`: leapfrog controls (`dt = step-size / nsteps`).
- `--no-gpu`: force CPU even if CUDA is available.
- `--complex64`: switch from complex128/float64 to complex64/float32.
- `--compile`, `--no-compile`: control `torch.compile` explicitly (default is off unless `IKKT_COMPILE=1` is set).
- `--threads`, `--interop-threads`: set PyTorch CPU intra-op and inter-op thread counts.
- `--resume`: load checkpoint if present.
- `--fresh`: ignore checkpoint and initialize a fresh configuration.
- `--save`: write checkpoint every `--save-every` iterations.
- `--saveAllMats`: dump raw matrix snapshots in chunked `.npy` files.
- `--force`: overwrite existing observable files.
- `--source`: optional source vector expression, e.g. `--source "np.linspace(-1,1,10)"`.
- `--eta`: Type I fermion deformation parameter (`--model pikkt4d_type1` only, default `1.0`).

Environment overrides:

- `IKKT_COMPILE=1`: enable `torch.compile` by default.
- `IKKT_NUM_THREADS=<n>`: default CPU intra-op threads.
- `IKKT_NUM_INTEROP_THREADS=<n>`: default CPU inter-op threads.

## Outputs

Each run writes into a model-specific directory under `--data-path`, with:

- `evals.npz`: accumulated eigenvalue measurements
- `corrs.npz`: accumulated correlator measurements
- `metadata.json`: run + model metadata
- `checkpoint.pt`: last saved configuration (only if `--save` is used)
- `all_mats/` (optional): chunked raw matrix snapshots when `--saveAllMats` is enabled

## Repository Layout

- `main.py`: orchestration, model loading, trajectory loop, output handling
- `cli.py`: argument parsing + validation
- `hmc.py`: leapfrog + Metropolis accept/reject
- `algebra.py`: matrix algebra helpers (Hermitian draws/projections, adjoint maps)
- `models/`: one file per model (`build_model(args)` entry point)
