"""Hierarchical refitting: stopping rule, prior alignment and empty blocks."""

import numpy as np

from gbmtoolbox import Config, GaussianPrior, HierarchicalFitConfig, Priors, StateModel, hierarchical_fit


def binary_model():
    return StateModel(observation=lambda x, p, u: p[0], family="bernoulli", priors=Priors(observation=GaussianPrior([0], [4])))


def test_hbi_covariance_changes_prevent_false_convergence_and_priors_align():
    data = [{"y": np.tile([0, 1], 5), "u": None} for _ in range(3)]
    model = binary_model()
    out = hierarchical_fit(data, [model], fit_config=Config(num_init=1, verbose=False), hbi_config=HierarchicalFitConfig(maxiter=3, tol=1e-6, verbose=False))
    assert out.n_iter == 3 and not out.converged
    assert out.convergence_history[0]["covariance_change"] > 3
    assert out.convergence_history[0]["mean_change"] < 1e-8
    np.testing.assert_array_equal(out.fits[0].input.prior_covariance, out.group_priors[0].covariance)
    np.testing.assert_array_equal(out.fits[0].input.prior_mean, out.group_priors[0].mean)


def test_hbi_empty_scientific_parameter_blocks_are_supported():
    model = StateModel(observation=lambda x, p, u: 0.0, priors=Priors(observation=GaussianPrior.empty()))
    out = hierarchical_fit(
        [{"y": np.array([0.1, -0.2, 0.4]), "u": None}], [model], fit_config=Config(num_init=1, verbose=False), hbi_config=HierarchicalFitConfig(maxiter=2, verbose=False)
    )
    assert out.converged
    assert out.group_priors[0].dim == 0
