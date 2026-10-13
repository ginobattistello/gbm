"""Seeding: independent optimizer/latent streams, HBI seeds and seed validation."""

from dataclasses import replace

import numpy as np
import pytest

from gbmtoolbox import Config, GaussianPrior, HBIConfig, Priors, StateModel, hbi_main, individual_fit


def binary_model():
    return StateModel(observation=lambda x, p, u: p[0], family="bernoulli", priors=Priors(observation=GaussianPrior([0], [4])))


def test_optimizer_and_latent_rng_streams_are_independent_and_reproducible():
    model = binary_model()
    data = [{"y": np.array([0, 1, 1, 0, 1, 1]), "u": None}, {"y": np.array([1, 1, 0, 1, 0, 1]), "u": None}]
    cfg = Config(num_init=3, random_state=73, verbose=False)
    base = individual_fit(data, model, config=cfg)
    propagated = individual_fit(data, model, config=replace(cfg, latent_uncertainty="propagated", latent_samples=12, latent_interval=0.95))
    repeat = individual_fit(data, model, config=cfg)
    for a, b, c in zip(base.math.diagnostics, propagated.math.diagnostics, repeat.math.diagnostics):
        for s, t, u in zip(a.starts, b.starts, c.starts):
            np.testing.assert_array_equal(s.initial_parameters, t.initial_parameters)
            np.testing.assert_array_equal(s.initial_parameters, u.initial_parameters)
            np.testing.assert_array_equal(s.final_parameters, t.final_parameters)
            assert s.n_evaluations == t.n_evaluations == u.n_evaluations
    np.testing.assert_array_equal(base.output.parameters, repeat.output.parameters)


def test_hbi_seed_is_used_and_inherits_config_seed_when_not_supplied():
    data = [{"y": np.tile([0, 1], 3), "u": None}]
    fit_cfg = Config(num_init=2, random_state=101, verbose=False)
    config = HBIConfig(maxiter=1, random_state=73, verbose=False)
    a = hbi_main(data, [binary_model()], fit_config=fit_cfg, hbi_config=config)
    b = hbi_main(data, [binary_model()], fit_config=fit_cfg, hbi_config=config)
    c = hbi_main(data, [binary_model()], fit_config=fit_cfg, hbi_config=replace(config, random_state=74))
    assert a.model_seeds == b.model_seeds and a.model_seeds != c.model_seeds
    for s, t in zip(a.fits[0].math.diagnostics[0].starts, b.fits[0].math.diagnostics[0].starts):
        np.testing.assert_array_equal(s.initial_parameters, t.initial_parameters)
    expected = tuple(int(s.generate_state(1)[0]) for s in np.random.SeedSequence(101).spawn(1))
    # Use an inherited HBIConfig with a short loop via the default constructor.
    inherited = hbi_main(data, [binary_model()], fit_config=fit_cfg)
    assert inherited.model_seeds == expected


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_seeds_fail_early(seed):
    with pytest.raises(ValueError, match="random_state"):
        Config(random_state=seed)
    with pytest.raises(ValueError, match="random_state"):
        HBIConfig(random_state=seed)
