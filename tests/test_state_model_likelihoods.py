import jax.numpy as jnp
import numpy as np
from gbmtoolbox import GaussianPrior, Priors, StateModel


def test_bernoulli_likelihood_matches_formula():
    # The observation returns a logit; families.py applies the sigmoid.
    logit = float(np.log(0.8 / 0.2))
    model = StateModel(
        evolution=lambda x, th, u, y: x,
        observation=lambda x, ph, u: logit,
        family="bernoulli",
        priors=Priors(GaussianPrior([0], [0]), GaussianPrior([0], [0])),
        initial_state=[0],
    )
    out = model.evaluate([0, 0], {"y": np.array([1, 0]), "u": None})
    np.testing.assert_allclose(out["loglik"], [np.log(.8), np.log(.2)], rtol=1e-10)


def test_categorical_likelihood_matches_formula():
    # Logits: softmax(log p) recovers p up to an additive constant.
    logits = jnp.asarray(np.log([.2, .3, .5]))
    model = StateModel(
        evolution=lambda x, th, u, y: x,
        observation=lambda x, ph, u: logits,
        family="categorical",
        priors=Priors(GaussianPrior([0], [0]), GaussianPrior([0], [0])),
        initial_state=[0],
    )
    out = model.evaluate([0, 0], {"y": np.array([2, 1]), "u": None})
    np.testing.assert_allclose(out["loglik"], [np.log(.5), np.log(.3)], rtol=1e-10)


def test_gaussian_likelihood_matches_formula():
    model = StateModel(
        evolution=lambda x, th, u, y: x,
        observation=lambda x, ph, u: jnp.asarray([ph[0]]),
        family="gaussian",
        priors=Priors(GaussianPrior([0], [0]), GaussianPrior([0], [1])),
        initial_state=[0],
        observation_covariance=4.0,
    )
    y = np.array([2.0])
    out = model.evaluate([0, 0], {"y": y, "u": None})
    expected = -0.5 * (np.log(2 * np.pi * 4) + 1.0)
    np.testing.assert_allclose(out["loglik"][0], expected, rtol=1e-10)


def test_generic_u_dictionary_is_sliced_without_reserved_names():
    """Trial inputs are sliced per trial and carry no reserved field names.

    The evolution reads u["abc"] into the state, so the recorded state trace
    shows which slice each trial received. The JAX backend requires numeric or
    bool fields, so a categorical condition is a numeric code, not a string.
    """
    def evo(x, th, u, y):
        # Carry the current trial's input forward so it is observable in states.
        return jnp.asarray([u["abc"] + 0.0 * u["condition"]])

    model = StateModel(
        evo, lambda x, ph, u: 0.0, "bernoulli",
        Priors(GaussianPrior([0], [0]), GaussianPrior([0], [0])), [0]
    )
    out = model.evaluate([0, 0], {"y": np.array([0, 1]), "u": {"abc": np.array([3.0, 4.0]), "condition": np.array([1.0, 1.0])}})
    # states[t] is the state entering trial t: initial_state, then the value
    # evolution built from trial 0's slice of u["abc"].
    np.testing.assert_allclose(np.asarray(out["states"]).reshape(-1), [0.0, 3.0])
