import numpy as np

from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit


def test_deterministic_bernoulli_fit_uses_ad_hessian():
    def evolution(x, theta, u_t, y_t):
        return x

    def observation(x, phi, u_t):
        return phi[0]

    y = np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 1], dtype=int)
    model = StateModel(
        evolution=evolution,
        observation=observation,
        family="bernoulli",
        priors=Priors(GaussianPrior([0.0], [0.0], names=["dummy"]), GaussianPrior([0.0], [4.0], names=["logit"])),
        initial_state=[0.0],
    )
    fit = individual_fit([{"y": y, "u": None}], model, config=Config(num_init=1, verbose=False, hessian_method="autodiff"))
    assert fit.math.diagnostics[0].hess_method == "autodiff"
    assert fit.math.diagnostics[0].laplace_valid
    assert np.isfinite(fit.output.log_evidence[0])
