"""Tests for the qcd_4d model."""

import tempfile
import unittest

import numpy as np
import torch

import matrix_hmc as hmc
from matrix_hmc.algebra import ad_matrix, random_hermitian
from matrix_hmc.models.qcd_4d import QCD4DModel, _gamma_euclidean_4d, _qcd4d_logdet


class TestGammaMatrices(unittest.TestCase):
    """Verify the 4D Euclidean Clifford algebra {Γ^I, Γ^J} = 2δ^{IJ}."""

    def setUp(self):
        hmc.configure(device="cpu", precision="complex128")
        self.gammas = _gamma_euclidean_4d(torch.device("cpu"), torch.complex128)

    def test_anticommutator_diagonal(self):
        I4 = torch.eye(4, dtype=torch.complex128)
        for mu in range(4):
            G = self.gammas[mu]
            anticomm = G @ G + G @ G  # {Γ,Γ} = 2Γ²
            self.assertTrue(
                torch.allclose(G @ G, I4, atol=1e-12),
                msg=f"Γ^{mu+1} is not an involution",
            )

    def test_anticommutator_offdiagonal(self):
        zero = torch.zeros(4, 4, dtype=torch.complex128)
        for mu in range(4):
            for nu in range(mu + 1, 4):
                anticomm = self.gammas[mu] @ self.gammas[nu] + self.gammas[nu] @ self.gammas[mu]
                self.assertTrue(
                    torch.allclose(anticomm, zero, atol=1e-12),
                    msg=f"{{Γ^{mu+1}, Γ^{nu+1}}} ≠ 0",
                )

    def test_hermitian(self):
        for mu in range(4):
            G = self.gammas[mu]
            self.assertTrue(
                torch.allclose(G, G.conj().T, atol=1e-12),
                msg=f"Γ^{mu+1} is not Hermitian",
            )

    def test_gamma5_anticommutes(self):
        """Γ_5 = Γ^1Γ^2Γ^3Γ^4 should anticommute with each Γ^μ."""
        g = self.gammas
        gamma5 = g[0] @ g[1] @ g[2] @ g[3]
        zero = torch.zeros(4, 4, dtype=torch.complex128)
        for mu in range(4):
            anticomm = gamma5 @ g[mu] + g[mu] @ gamma5
            self.assertTrue(
                torch.allclose(anticomm, zero, atol=1e-12),
                msg=f"Γ_5 does not anticommute with Γ^{mu+1}",
            )


class TestNoSignProblem(unittest.TestCase):
    """det(K) is always real and positive.

    K = I + Σ_μ Γ^μ ⊗ ad_{X_μ}.  The adjoint action ad_{X_μ} = i[X_μ,.] is
    anti-Hermitian, so K has eigenvalues 1 + iλ (λ ∈ ℝ).  The γ_5 symmetry
    pairs them as (1+iλ, 1−iλ), giving det K = Π(1+λ²) > 0 for all X.
    """

    def setUp(self):
        hmc.configure(device="cpu", precision="complex128")
        torch.manual_seed(42)
        self.gammas = _gamma_euclidean_4d(torch.device("cpu"), torch.complex128)

    def _random_X(self, N, scale=1.0):
        return scale * random_hermitian(N, batchsize=4).to(torch.complex128)

    # ------------------------------------------------------------------
    # Core eigenvalue-structure test
    # ------------------------------------------------------------------

    def test_eigenvalue_pairing(self):
        """Eigenvalues of K are 1+iλ: real part = 1 and imag parts ±-paired.

        Proof sketch:
          D = Σ Γ^μ ⊗ ad_{X_μ} is anti-Hermitian  (D† = −D).
          Write D = iA with A = −iD Hermitian; eigenvalues of A are real λ.
          K = I + iA  →  eigenvalue 1 + iλ for each λ of A.
          γ_5 anticommutes with each Γ^μ, so {γ_5⊗I, D} = 0, meaning
          eigenvalues of A come in ±λ pairs.
          Therefore eigenvalues of K pair as (1+iλ, 1−iλ), and
          det K = Π_pairs (1+iλ)(1−iλ) = Π(1+λ²) > 0.
        """
        for N in (3, 4, 5):
            X = self._random_X(N, scale=1.5)
            N2 = N * N

            # Build K directly (mirrors _qcd4d_logdet internals)
            D = sum(
                torch.kron(self.gammas[mu], ad_matrix(X[mu])) for mu in range(4)
            )
            K = torch.eye(4 * N2, dtype=torch.complex128) + D

            eigs = torch.linalg.eigvals(K).numpy()

            # 1. All real parts must equal 1
            np.testing.assert_allclose(
                eigs.real,
                np.ones(len(eigs)),
                atol=1e-8,
                err_msg=f"N={N}: Re(eigenvalue) ≠ 1 — D is not anti-Hermitian",
            )

            # 2. Imaginary parts come in ±λ pairs (γ_5 pairing)
            imag = np.sort(eigs.imag)
            np.testing.assert_allclose(
                imag,
                -imag[::-1],
                atol=1e-8,
                err_msg=f"N={N}: imaginary parts are not ±-paired",
            )

            # 3. det(K) = Π(1+λ²) > 0  (sign = +1)
            sign, _ = torch.linalg.slogdet(K)
            self.assertAlmostEqual(
                sign.real.item(), 1.0, places=8,
                msg=f"N={N}: det(K) sign is not +1",
            )
            self.assertAlmostEqual(
                sign.imag.item(), 0.0, places=8,
                msg=f"N={N}: det(K) is not real",
            )

    def test_det_positive_large_x(self):
        """det(K) > 0 even for large X (structural guarantee, not just small X)."""
        for N in (3, 4):
            for scale in (0.5, 2.0, 5.0):
                X = self._random_X(N, scale=scale)
                sign, _ = _qcd4d_logdet(X, self.gammas, massless=False)
                self.assertAlmostEqual(
                    sign.real.item(), 1.0, places=8,
                    msg=f"N={N} scale={scale}: sign ≠ +1",
                )

    def test_det_positive_massless(self):
        for N in (3, 4):
            X = self._random_X(N, scale=1.5)
            sign, _ = _qcd4d_logdet(X, self.gammas, massless=True)
            self.assertAlmostEqual(
                sign.real.item(), 1.0, places=8,
                msg=f"N={N} massless: sign ≠ +1",
            )


class TestQCD4DDryRun(unittest.TestCase):
    def setUp(self):
        hmc.configure(device="cpu", precision="complex64")

    def _run(self, model, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            return hmc.run(
                model,
                step_size=kwargs.pop("step_size", 0.1),
                nsteps=kwargs.pop("nsteps", 50),
                niters=kwargs.pop("niters", 20),
                output=tmp,
                name="test",
                dry_run=True,
                **kwargs,
            )

    def test_dry_run_massive(self):
        model = QCD4DModel(ncol=8, couplings=[1.0])
        self.assertIs(self._run(model), model)

    def test_dry_run_massless(self):
        model = QCD4DModel(ncol=8, couplings=[1.0], massless=True)
        self.assertIs(self._run(model), model)

    def test_dry_run_boson_mass_varied(self):
        model = QCD4DModel(ncol=8, couplings=[10.0], boson_mass=0.5)
        self.assertIs(self._run(model), model)

    def test_gradient_flows(self):
        """force() should return a finite tensor of the right shape."""
        hmc.configure(device="cpu", precision="complex64")
        model = QCD4DModel(ncol=6, couplings=[1.0])
        model.load_fresh()
        force = model.force()
        self.assertEqual(force.shape, model.get_state().shape)
        self.assertFalse(torch.any(torch.isnan(force)))

    def test_gradient_flows_massless(self):
        hmc.configure(device="cpu", precision="complex64")
        model = QCD4DModel(ncol=6, couplings=[1.0], massless=True)
        model.load_fresh()
        force = model.force()
        self.assertEqual(force.shape, model.get_state().shape)
        self.assertFalse(torch.any(torch.isnan(force)))


if __name__ == "__main__":
    unittest.main()
