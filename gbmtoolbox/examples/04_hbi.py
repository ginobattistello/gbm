"""Tutorial 04 — hierarchical theta/phi refitting; Q/R priors remain model-level."""

import numpy as np

from gbmtoolbox import Config, HBIConfig, hbi_main
from gbmtoolbox.examples.tutorial_models import binary_learning_model, simulate_binary_subject

rng = np.random.default_rng(42)
data = [simulate_binary_subject(rng, n_trials=50) for _ in range(6)]
result = hbi_main(
    data, [binary_learning_model()], fit_config=Config(num_init=2, random_state=42, verbose=False), hbi_config=HBIConfig(maxiter=5, tol=1e-2, verbose=True)
)
print("group evolution mean:", result.group_priors[0].evolution.mean)
print("group observation mean:", result.group_priors[0].observation.mean)
print("model frequency:", result.model_frequency)
