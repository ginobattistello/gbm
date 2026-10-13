"""Reference check 10: default observation-noise MAP against the exact Gamma conditional posterior.

Reference: sigma_hat = sqrt((1 + SSE/2)/(1 + T/2)), the exponentiated
log-SD MAP. This is not the mode or mean of SD in its own coordinate.
"""

import numpy as np

from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit


def main():
    mu = 1.25
    model = StateModel(observation=lambda x, phi, u: mu, priors=Priors(observation=GaussianPrior([0.0], [0.0])))
    print("Exact Gamma(1,1) precision prior; fitting coordinate: log SD")
    worst = 0.0
    for sd, T, seed in [(0.3, 200, 1), (0.7, 400, 2), (1.5, 800, 3), (2.5, 200, 4)]:
        y = mu + np.random.default_rng(seed).normal(scale=sd, size=T)
        S = np.sum((y - mu) ** 2)
        reference = np.sqrt((1 + S / 2) / (1 + T / 2))
        fit = individual_fit([{"y": y, "u": None}], model, config=Config(num_init=3, verbose=False))
        fitted = fit.output.observation_noise_sd[0, 0]
        err = abs(fitted / reference - 1)
        worst = max(worst, err)
        print(f"T={T:4d}, true SD={sd:.3f}, fitted={fitted:.8f}, reference={reference:.8f}, rel error={err:.3e}")
    assert worst < 1e-3
    print(f"Worst relative error: {worst:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
