"""Reference 11: residual energy, not a general unbiased noise estimator.

First compare the diagnostic with exact Gaussian linear-regression posterior
moments at known observation variance. Then check the exact sampling means of
OLS residual SD and its one-step correction using the chi-square law. The
second calculation is an analytic limiting illustration, not validation of
noise recovery under the default Gamma prior.
"""
import numpy as np
from scipy.special import gammaln

from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit, observation_noise_correction


def main():
    rng = np.random.default_rng(42)
    worst_energy_error = 0.0
    worst_z = 0.0
    for T, p in [(25, 5), (50, 3), (200, 5)]:
        X = np.column_stack([np.ones(T), rng.normal(size=(T, p - 1))])
        y = X @ np.linspace(-0.3, 0.8, p) + rng.normal(size=T)
        prior_variance = 4.0
        model = StateModel(
            observation=lambda x, phi, u: u @ phi,
            family="gaussian",
            priors=Priors(observation=GaussianPrior(np.zeros(p), np.full(p, prior_variance))),
            observation_covariance=1.0,
        )
        fit = individual_fit(y, model, u=X, config=Config(num_init=1, verbose=False))
        out = observation_noise_correction(fit)
        covariance = np.linalg.inv(X.T @ X + np.eye(p) / prior_variance)
        mean = covariance @ X.T @ y
        sse = np.sum((y - X @ mean) ** 2)
        uncertainty = np.trace(X @ covariance @ X.T)
        expected = (sse + uncertainty) / T
        error = abs(out.corrected_sd[0] ** 2 - expected)
        worst_energy_error = max(worst_energy_error, error)
        np.testing.assert_allclose(out.uncertainty_trace, uncertainty, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(out.corrected_sd[0] ** 2, expected, rtol=1e-5, atol=1e-6)

        # Exact OLS distribution at true sigma=1, full-rank X, p mean parameters.
        nu = T - p
        mean_sd = np.sqrt(2 / T) * np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2))
        variance_sd = nu / T - mean_sd**2
        factor = np.sqrt(1 + p / T)
        n = 200_000
        residual_sd = np.sqrt(rng.chisquare(nu, n) / T)
        corrected_sd = residual_sd * factor
        residual_z = abs(residual_sd.mean() - mean_sd) / np.sqrt(variance_sd / n)
        corrected_z = abs(corrected_sd.mean() - mean_sd * factor) / np.sqrt(variance_sd * factor**2 / n)
        worst_z = max(worst_z, residual_z, corrected_z)
        assert max(residual_z, corrected_z) < 5
        # One-step variance remains biased; this is deliberately not an
        # assertion that corrected SD has mean 1.
        print(f"T={T}, p={p}: energy error={error:.3g}; "
              f"exact E[residual SD]={mean_sd:.6f}; E[one-step SD]={mean_sd * factor:.6f}; "
              f"E[one-step variance]={1 - (p/T)**2:.6f}; MC z={max(residual_z, corrected_z):.3f}")
    print(f"PASS: maximum posterior residual-energy error={worst_energy_error:.3g}; MC z={worst_z:.3f}")


if __name__ == "__main__":
    main()
