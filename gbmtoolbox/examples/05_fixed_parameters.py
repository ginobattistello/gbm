"""Tutorial 05 — fixing static parameters with zero prior variance."""

import numpy as np
from gbmtoolbox import Config, individual_fit
from gbmtoolbox.examples.tutorial_models import binary_learning_model, simulate_binary_subject

rng = np.random.default_rng(42)
data = [simulate_binary_subject(rng, n_trials=60)]
fit = individual_fit(data, binary_learning_model(fixed_alpha=True), config=Config(num_init=3, random_state=42, verbose=False))
print(fit.summary())
print("full posterior covariance:\n", fit.math.covariance[0])
