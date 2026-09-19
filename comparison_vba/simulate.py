"""Simulation of the three observation-only datasets used in the comparison.

All three models share the same structure: a single trial-wise regressor `s`
drives the outcome through a linear predictor, with no hidden state and no
evolution parameters ("null evolution"). Only the outcome distribution and
the link function change.

    continuous   y_t ~ Normal(b0 + b1*s_t, sigma^2)
    binary       y_t ~ Bernoulli(sigmoid(b0 + b1*s_t))
    categorical  y_t ~ Categorical(softmax([0, a1 + b1*s_t, a2 + b2*s_t]))

For the categorical model class 0 is the reference: its logit is fixed to 0,
which is what makes the remaining parameters identifiable.
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 3


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_continuous(rng, n_subjects, n_trials):
    """Gaussian outcomes. Returns (data, true_params).

    true_params columns: intercept, slope, noise SD.
    """
    true = np.column_stack([
        rng.uniform(-1.0, 1.0, n_subjects),   # intercept
        rng.uniform(0.5, 2.0, n_subjects),    # slope
        rng.uniform(0.3, 1.0, n_subjects),    # noise SD
    ])
    y = np.zeros((n_subjects, n_trials))
    u = rng.normal(size=(n_subjects, n_trials))
    for i, (b0, b1, sd) in enumerate(true):
        y[i] = b0 + b1 * u[i] + rng.normal(0.0, sd, n_trials)
    return {"y": y, "u": u}, true


def simulate_binary(rng, n_subjects, n_trials):
    """Bernoulli outcomes. true_params columns: intercept, slope."""
    true = np.column_stack([
        rng.uniform(-1.0, 1.0, n_subjects),
        rng.uniform(0.5, 2.5, n_subjects),
    ])
    y = np.zeros((n_subjects, n_trials), dtype=int)
    u = rng.normal(size=(n_subjects, n_trials))
    for i, (b0, b1) in enumerate(true):
        p = _sigmoid(b0 + b1 * u[i])
        y[i] = (rng.random(n_trials) < p).astype(int)
    return {"y": y, "u": u}, true


def simulate_categorical(rng, n_subjects, n_trials):
    """3-class categorical outcomes.

    true_params columns: a1, b1, a2, b2 (intercept/slope for classes 1 and 2;
    class 0 is the fixed reference).
    """
    true = np.column_stack([
        rng.uniform(-1.0, 1.0, n_subjects),
        rng.uniform(0.5, 2.0, n_subjects),
        rng.uniform(-1.0, 1.0, n_subjects),
        rng.uniform(-2.0, -0.5, n_subjects),
    ])
    y = np.zeros((n_subjects, n_trials), dtype=int)
    u = rng.normal(size=(n_subjects, n_trials))
    for i, (a1, b1, a2, b2) in enumerate(true):
        logits = np.stack([
            np.zeros(n_trials),
            a1 + b1 * u[i],
            a2 + b2 * u[i],
        ], axis=1)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        y[i] = np.array([rng.choice(N_CLASSES, p=row) for row in p])
    return {"y": y, "u": u}, true


def one_hot(y, n_classes=N_CLASSES):
    """Integer labels (n_subjects, n_trials) -> one-hot (n_subjects, K, n_trials).

    VBA codes categorical outcomes one-hot as K x T; GBM Toolbox uses the
    integer labels directly.
    """
    n_subjects, n_trials = y.shape
    out = np.zeros((n_subjects, n_classes, n_trials))
    for i in range(n_subjects):
        out[i, y[i], np.arange(n_trials)] = 1.0
    return out


def simulate_all(seed=0, n_subjects=20, n_trials=200):
    """Simulate all three datasets from one seed."""
    rng = np.random.default_rng(seed)
    cont, true_cont = simulate_continuous(rng, n_subjects, n_trials)
    binr, true_bin = simulate_binary(rng, n_subjects, n_trials)
    cat, true_cat = simulate_categorical(rng, n_subjects, n_trials)
    return {
        "continuous": (cont, true_cont),
        "binary": (binr, true_bin),
        "categorical": (cat, true_cat),
    }
