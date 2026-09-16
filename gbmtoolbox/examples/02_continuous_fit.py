"""Tutorial 02 — Gaussian outcomes with internally estimated diagonal R."""

import matplotlib.pyplot as plt
import numpy as np
from gbmtoolbox.examples.tutorial_models import continuous_model, simulate_continuous_subject
from gbmtoolbox import Config, individual_fit

rng = np.random.default_rng(42)
n_subjects, n_trials = 30, 80
true_gain = rng.uniform(0.40, 0.90, size=n_subjects)
true_offset = rng.uniform(-0.50, 0.50, size=n_subjects)
true_sigma = rng.uniform(0.20, 1.00, size=n_subjects)
data = [simulate_continuous_subject(rng, n_trials, g, o, s) for g, o, s in zip(true_gain, true_offset, true_sigma)]
model = continuous_model()
print("resolved parameters:", model.parameter_layout.names)
fit = individual_fit(data, model, config=Config(num_init=5, random_state=42, display=False))
print(fit.summary(subject=0))
estimated_gain = fit.output.evolution_parameters[:, 0]
estimated_offset = fit.output.observation_parameters[:, 0]
estimated_sigma = fit.output.observation_noise_sd[:, 0]
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
for ax, (label, true, est) in zip(
    axes, [("gain", true_gain, estimated_gain), ("offset", true_offset, estimated_offset), ("observation SD", true_sigma, estimated_sigma)]
):
    ax.scatter(true, est)
    lo, hi = min(true.min(), est.min()), max(true.max(), est.max())
    ax.plot([lo, hi], [lo, hi], "--k")
    ax.set_xlabel("True")
    ax.set_ylabel("Estimated")
    ax.set_title(label)
fig.tight_layout()
plt.show()
