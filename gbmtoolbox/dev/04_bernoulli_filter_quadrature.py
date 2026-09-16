"""Reference check 04: Bernoulli Laplace-Gaussian filter versus quadrature.

The exact one-step predictive probability is a one-dimensional integral. This
script measures the approximation error rather than claiming exact filtering.
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from gbmtoolbox import GaussianPrior, Priors, StateModel

model=StateModel(
    lambda x,th,u,y:x,
    lambda x,ph,u:x[0],  # logit; families.py applies the sigmoid
    "bernoulli",
    Priors(GaussianPrior([0],[0]),GaussianPrior([0],[0])),
    [.4],initial_state_covariance=[.5]
)
out=model.evaluate([0,0], {"y":np.array([1]),"u":None})
approx=float(np.exp(out['loglik'][0]))
mu=.4; var=.5
normal=lambda x: np.exp(-.5*(x-mu)**2/var)/np.sqrt(2*np.pi*var)
exact=quad(lambda x: expit(x)*normal(x),-np.inf,np.inf,epsabs=1e-12)[0]
print(f"exact predictive P(y=1) = {exact:.8f}")
print(f"Laplace-Gaussian approximation = {approx:.8f}")
print(f"absolute error = {abs(approx-exact):.3e}")
assert abs(approx-exact) < .04
print("PASS within documented approximation tolerance")
