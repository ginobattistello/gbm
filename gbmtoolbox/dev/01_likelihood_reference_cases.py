"""Reference check 01: family likelihoods against direct formulas.

Question
--------
Does StateModel generate the same per-trial log likelihood as the elementary
Gaussian, Bernoulli, and categorical formulas?

Failure criterion
-----------------
Absolute error above 1e-12 in these deterministic reference cases.
"""
import jax.numpy as jnp
import numpy as np
from gbmtoolbox import GaussianPrior, Priors, StateModel

fixed = Priors(GaussianPrior([0], [0]), GaussianPrior([0], [0]))

# Observations return logits; families.py applies the sigmoid/softmax.
bern = StateModel(lambda x,t,u,y:x, lambda x,p,u:float(np.log(.8/.2)), "bernoulli", fixed, [0])
out = bern.evaluate([0,0], {"y":np.array([1,0]), "u":None})["loglik"]
ref = np.log([.8,.2])
print("Bernoulli max error:", np.max(np.abs(out-ref)))
assert np.allclose(out, ref, atol=1e-12)

cat = StateModel(lambda x,t,u,y:x, lambda x,p,u:jnp.asarray(np.log([.2,.3,.5])), "categorical", fixed, [0])
out = cat.evaluate([0,0], {"y":np.array([2,1]), "u":None})["loglik"]
ref = np.log([.5,.3])
print("Categorical max error:", np.max(np.abs(out-ref)))
assert np.allclose(out, ref, atol=1e-12)

gau = StateModel(lambda x,t,u,y:x, lambda x,p,u:jnp.asarray([0.]), "gaussian", fixed, [0], observation_covariance=4.)
out = gau.evaluate([0,0], {"y":np.array([2.]), "u":None})["loglik"][0]
ref = -.5*(np.log(2*np.pi*4)+1.)
print("Gaussian error:", abs(out-ref))
assert abs(out-ref) < 1e-12
print("PASS")
