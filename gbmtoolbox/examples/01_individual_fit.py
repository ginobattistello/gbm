"""Tutorial 01 — individual fit and propagated latent uncertainty."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from gbmtoolbox import Config, individual_fit
from gbmtoolbox.examples.tutorial_models import binary_learning_model, simulate_binary_subject

rng = np.random.default_rng(42)
n_subjects = 20
true_alpha = rng.uniform(0.10, 0.60, size=n_subjects)
true_beta = rng.uniform(1.0, 5.0, size=n_subjects)
data = [simulate_binary_subject(rng, n_trials=150, alpha=a, beta=b) for a, b in zip(true_alpha, true_beta)]
fit = individual_fit(
    data, binary_learning_model(), config=Config(num_init=5, random_state=42, latent_uncertainty="propagated", latent_samples=500, display=False)
)
print(fit.summary(subject=0))
fit.plot(subject=0)
estimated_alpha = expit(fit.output.evolution_parameters[:, 0])
estimated_beta = np.exp(fit.output.observation_parameters[:, 0])
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, (label, true, estimated) in zip(axes, [("Learning rate", true_alpha, estimated_alpha), ("Inverse temperature", true_beta, estimated_beta)]):
    r = np.corrcoef(true, estimated)[0, 1]
    rmse = np.sqrt(np.mean((estimated - true) ** 2))
    ax.scatter(true, estimated)
    lo, hi = min(true.min(), estimated.min()), max(true.max(), estimated.max())
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
    ax.set_xlabel("True")
    ax.set_ylabel("Estimated")
    ax.set_title(label)
    ax.text(0.05, 0.95, f"r={r:.2f}\nRMSE={rmse:.2f}", transform=ax.transAxes, va="top")
fig.tight_layout()
plt.show()
