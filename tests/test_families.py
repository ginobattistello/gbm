import jax.numpy as jnp
import numpy as np
from gbmtoolbox.families import get_family


def test_bernoulli_score_and_fisher():
    fam = get_family("bernoulli")
    eta = jnp.array([0.3])
    p = 1 / (1 + np.exp(-0.3))
    np.testing.assert_allclose(fam.mean(eta), [p])
    np.testing.assert_allclose(fam.score(1, eta), [1-p])
    np.testing.assert_allclose(fam.fisher(eta), [[p*(1-p)]])


def test_categorical_score_sums_to_zero():
    fam = get_family("categorical")
    eta = jnp.array([0.2, -0.1, 0.4])
    score = np.asarray(fam.score(1, eta))
    assert abs(score.sum()) < 1e-12
