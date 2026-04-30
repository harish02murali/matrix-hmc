"""Tests for the qcd_5d model."""

import tempfile
import unittest

import numpy as np
import torch

import matrix_hmc as hmc
from matrix_hmc.algebra import ad_matrix, random_hermitian
from matrix_hmc.models.qcd_5d import QCD5DModel, _gamma_euclidean_5d, _qcd5d_logdet


class TestGammaMatrices5D(unittest.TestCase):
    """Verify the 5D Euclidean Clifford algebra {Γ^I, Γ^J} = 2δ^{IJ}."""

    def setUp(self):
        hmc.configure(device="cpu", precision="complex128")
        self.gammas = _gamma_euclidean_5d(torch.device("cpu"), torch.complex128)

    def test_count_and_shape(self):
        self.assertEqual(self.gammas.shape, (5, 4, 4))

    def test_hermitian(self):
        for mu in range(5):
            G = self.gammas[mu]
            self.assertTrue(
                torch.allclose(G, G.conj().T, atol=1e-12),
                msg=f"Γ^{mu+1} is not Hermitian",
            )

    def test_involution(self):
        """Each Γ^I squares to the identity."""
        I4 = torch.eye(4, dtype=torch.complex128)
        for mu in range(5):
            G = self.gammas[mu]
            self.assertTrue(
                torch.allclose(G @ G, I4, atol=1e-12),
                msg=f"(Γ^{mu+1})² ≠ I",
            )

    def test_anticommutator_offdiagonal(self):
        """{Γ^I, Γ^J} = 0 for I ≠ J."""
        zero = torch.zeros(4, 4, dtype=torch.complex128)
        for mu in range(5):
            for nu in range(mu + 1, 5):
                anticomm = self.gammas[mu] @ self.gammas[nu] + self.gammas[nu] @ self.gammas[mu]
                self.assertTrue(
                    torch.allclose(anticomm, zero, atol=1e-12),
                    msg=f"{{Γ^{mu+1}, Γ^{nu+1}}} ≠ 0",
                )

    def test_gamma5_is_product(self):
        """Γ^5 equals Γ^1 Γ^2 Γ^3 Γ^4."""
        g = self.gammas
        g5_expected = g[0] @ g[1] @ g[2] @ g[3]
        self.assertTrue(
            torch.allclose(g[4], g5_expected, atol=1e-12),
            msg="Γ^5 ≠ Γ^1 Γ^2 Γ^3 Γ^4",
        )

    def test_gamma5_anticommutes_with_first_four(self):
        """Γ^5 anticommutes with each of Γ^1,...,Γ^4."""
        g5 = self.gammas[4]
        zero = torch.zeros(4, 4, dtype=torch.complex128)
        for mu in range(4):
            anticomm = g5 @ self.gammas[mu] + self.gammas[mu] @ g5
            self.assertTrue(
                torch.allclose(anticomm, zero, atol=1e-12),
                msg=f"Γ^5 does not anticommute with Γ^{mu+1}",
            )


class TestDiracOperator5D(unittest.TestCase):
    """Structural tests on D = Σ_μ Γ^μ ⊗ ad_{X_μ}."""

    def setUp(self):
        hmc.configure(device="cpu", precision="complex128")
        torch.manual_seed(7)
        self.gammas = _gamma_euclidean_5d(torch.device("cpu"), torch.complex128)

    def _random_X(self, N, scale=1.0):
        return scale * random_hermitian(N, batchsize=5).to(torch.complex128)

    def test_D_anti_hermitian(self):
        """D = Σ_μ Γ^μ ⊗ ad_{X_μ} is anti-Hermitian (D† = −D).

        Each Γ^μ is Hermitian and each ad_{X_μ} is anti-Hermitian for Hermitian X_μ,
        so their tensor product is anti-Hermitian, and the sum inherits this.
        """
        for N in (3, 4):
            X = self._random_X(N, scale=1.5)
            D = sum(
                torch.kron(self.gammas[mu], ad_matrix(X[mu])) for mu in range(5)
            )
            self.assertTrue(
                torch.allclose(D + D.conj().T, torch.zeros_like(D), atol=1e-10),
                msg=f"N={N}: D is not anti-Hermitian",
            )

    def test_eigenvalues_purely_imaginary(self):
        """Eigenvalues of D are purely imaginary (real parts ≈ 0)."""
        for N in (3, 4):
            X = self._random_X(N, scale=1.5)
            D = sum(
                torch.kron(self.gammas[mu], ad_matrix(X[mu])) for mu in range(5)
            )
            eigs = torch.linalg.eigvals(D).numpy()
            np.testing.assert_allclose(
                eigs.real, np.zeros(len(eigs)), atol=1e-8,
                err_msg=f"N={N}: eigenvalues of D have nonzero real part",
            )

    def test_K_eigenvalues_have_unit_real_part(self):
        """Eigenvalues of K = I + D are 1+iλ: real parts all equal 1."""
        for N in (3, 4):
            X = self._random_X(N, scale=1.5)
            N2 = N * N
            D = sum(
                torch.kron(self.gammas[mu], ad_matrix(X[mu])) for mu in range(5)
            )
            K = torch.eye(4 * N2, dtype=torch.complex128) + D
            eigs = torch.linalg.eigvals(K).numpy()
            np.testing.assert_allclose(
                eigs.real, np.ones(len(eigs)), atol=1e-8,
                err_msg=f"N={N}: Re(eigenvalue of K) ≠ 1",
            )


class TestNoSignProblem5D(unittest.TestCase):
    """det(K) is real and positive for the 5D model.

    Verified numerically; analytically guaranteed by the user's checks.
    """

    def setUp(self):
        hmc.configure(device="cpu", precision="complex128")
        torch.manual_seed(42)
        self.gammas = _gamma_euclidean_5d(torch.device("cpu"), torch.complex128)

    def _random_X(self, N, scale=1.0):
        return scale * random_hermitian(N, batchsize=5).to(torch.complex128)

    def test_det_positive_massive(self):
        """sign(det K) = +1 for massive K across several N and scales."""
        for N in (3, 4, 5):
            for scale in (0.5, 1.5, 4.0):
                X = self._random_X(N, scale=scale)
                sign, _ = _qcd5d_logdet(X, self.gammas, massless=False)
                self.assertAlmostEqual(
                    sign.real.item(), 1.0, places=8,
                    msg=f"N={N} scale={scale}: sign ≠ +1",
                )
                self.assertAlmostEqual(
                    sign.imag.item(), 0.0, places=8,
                    msg=f"N={N} scale={scale}: det(K) is not real",
                )

    def test_det_positive_massless(self):
        """sign(det K_massless) = +1 (with trace modes lifted)."""
        for N in (3, 4):
            for scale in (0.5, 2.0):
                X = self._random_X(N, scale=scale)
                sign, _ = _qcd5d_logdet(X, self.gammas, massless=True)
                self.assertAlmostEqual(
                    sign.real.item(), 1.0, places=8,
                    msg=f"N={N} scale={scale} massless: sign ≠ +1",
                )

    def test_det_positive_many_random(self):
        """Positive sign holds for 20 independent random configurations."""
        torch.manual_seed(0)
        for i in range(20):
            X = self._random_X(4, scale=float(1 + i % 5))
            sign, _ = _qcd5d_logdet(X, self.gammas, massless=False)
            self.assertAlmostEqual(
                sign.real.item(), 1.0, places=7,
                msg=f"sample {i}: sign ≠ +1",
            )


class TestQCD5DConsistency(unittest.TestCase):
    """Check that 5D reduces to the right limits and agrees with 4D when X_5=0."""

    def setUp(self):
        hmc.configure(device="cpu", precision="complex128")
        torch.manual_seed(99)

    def test_logdet_5d_equals_4d_when_X5_zero(self):
        """With X_5 = 0, det K_5D must equal det K_4D."""
        from matrix_hmc.models.qcd_4d import _gamma_euclidean_4d, _qcd4d_logdet

        N = 4
        X4 = random_hermitian(N, batchsize=4).to(torch.complex128)

        gammas4 = _gamma_euclidean_4d(torch.device("cpu"), torch.complex128)
        gammas5 = _gamma_euclidean_5d(torch.device("cpu"), torch.complex128)

        X5 = torch.cat([X4, torch.zeros(1, N, N, dtype=torch.complex128)], dim=0)

        sign4, ldet4 = _qcd4d_logdet(X4, gammas4, massless=False)
        sign5, ldet5 = _qcd5d_logdet(X5, gammas5, massless=False)

        self.assertAlmostEqual(ldet4.real.item(), ldet5.real.item(), places=8,
                               msg="log|det| mismatch when X_5=0")
        self.assertAlmostEqual(sign4.real.item(), sign5.real.item(), places=8,
                               msg="sign mismatch when X_5=0")


class TestQCD5DDryRun(unittest.TestCase):

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
        model = QCD5DModel(ncol=6, couplings=[1.0])
        self.assertIs(self._run(model), model)

    def test_dry_run_massless(self):
        model = QCD5DModel(ncol=6, couplings=[1.0], massless=True)
        self.assertIs(self._run(model), model)

    def test_dry_run_boson_mass_varied(self):
        model = QCD5DModel(ncol=6, couplings=[10.0], boson_mass=0.5)
        self.assertIs(self._run(model), model)

    def test_gradient_flows(self):
        """force() should return a finite tensor of the right shape."""
        hmc.configure(device="cpu", precision="complex64")
        model = QCD5DModel(ncol=5, couplings=[1.0])
        model.load_fresh()
        force = model.force()
        self.assertEqual(force.shape, model.get_state().shape)
        self.assertFalse(torch.any(torch.isnan(force)))

    def test_gradient_flows_massless(self):
        hmc.configure(device="cpu", precision="complex64")
        model = QCD5DModel(ncol=5, couplings=[1.0], massless=True)
        model.load_fresh()
        force = model.force()
        self.assertEqual(force.shape, model.get_state().shape)
        self.assertFalse(torch.any(torch.isnan(force)))


if __name__ == "__main__":
    unittest.main()
