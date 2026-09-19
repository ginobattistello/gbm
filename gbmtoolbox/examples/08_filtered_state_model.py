"""Tutorial 08 — unified filtering with internally estimated diagonal Q and R."""

import numpy as np

from gbmtoolbox import Config, individual_fit
from gbmtoolbox.examples.tutorial_models import filtered_continuous_model

rng = np.random.default_rng(42)
T = 100
x = 0.0
y = np.zeros(T)
for t in range(T):
    y[t] = x + rng.normal(0, 0.5)
    x = 0.8 * x + rng.normal(0, np.sqrt(0.10))
data = [{"y": y, "u": None}]
fit = individual_fit(data, filtered_continuous_model(), config=Config(num_init=4, random_state=42, latent_uncertainty="filtered", display=False))
print(fit.summary())
print("estimated process SD:", fit.output.process_noise_sd[0])
print("estimated observation SD:", fit.output.observation_noise_sd[0])
fit.plot(subject=0)
