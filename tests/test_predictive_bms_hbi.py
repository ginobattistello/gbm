import numpy as np

from gbmtoolbox import Config, HBIConfig, bms, hbi_main, individual_fit, posterior_predictive, prior_predictive


def test_predictive_checks(binary_model, binary_data):
    prior = prior_predictive(binary_model, binary_data[0], n_samples=4, random_state=42)
    assert len(prior.replicated_data) == 4
    fit = individual_fit(binary_data, binary_model, config=Config(num_init=2, verbose=False))
    post = posterior_predictive(fit, n_samples=4, random_state=42)
    assert post.parameters.shape == (4, 2)
    assert set(np.unique(post.replicated_data[0]["y"])).issubset({0, 1})


def test_bms_prefers_high_evidence_model():
    lme = np.array([[5.0, 0.0], [4.0, 0.0], [6.0, 0.0], [5.0, 0.0]])
    out = bms(lme, n_samples=5000, random_state=42)
    assert out.model_frequency[0] > out.model_frequency[1]
    assert np.isclose(out.protected_exceedance_prob.sum(), 1)
    assert 0 <= out.bor <= 1


def test_hbi_runs_single_model(binary_model):
    rng = np.random.default_rng(2)
    data = []
    for _ in range(3):
        data.append({"y": rng.integers(0, 2, 12), "u": {"reward": rng.integers(0, 2, 12).astype(float)}})
    out = hbi_main(
        data, [binary_model], fit_config=Config(num_init=1, verbose=False, random_state=42), hbi_config=HBIConfig(maxiter=2, tol=1e-2, verbose=False)
    )
    assert out.responsibilities.shape == (3, 1)
    np.testing.assert_allclose(out.model_frequency, [1.0])
