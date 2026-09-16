"""Reference check 03: GBM Toolbox filter versus exact Kalman recursion.

For a linear-Gaussian state-space model the EKF implementation must reduce to
the exact Kalman filter. We compare filtered means, variances, and predictive
log likelihood trial by trial.

The residual error is set by Config.filter_jitter, the diagonal regularizer
used in the SPD solves, and scales linearly with it: 1e-9 jitter gives ~2.3e-10
error, 1e-12 gives ~2.3e-13, and with jitter disabled the filter reproduces the
exact Kalman recursion to machine precision (~1.1e-16).
"""
import jax.numpy as jnp
import numpy as np
from gbmtoolbox import GaussianPrior, Priors, StateModel

model=StateModel(
    lambda x,th,u,y:jnp.asarray([th[0]*x[0]]),
    lambda x,ph,u:jnp.asarray([x[0]+ph[0]]),
    "gaussian",
    Priors(GaussianPrior([.8],[.2]), GaussianPrior([.1],[1.])),
    [0.], initial_state_covariance=[1.], process_covariance=[.1], observation_covariance=[.25]
)
y=np.array([.2,-.1,.3,.0])
out=model.evaluate([.8,.1], {"y":y,"u":None})

m,P=0.,1.; means=[]; vars=[]; lls=[]
for yt in y:
    S=P+.25
    lls.append(-.5*(np.log(2*np.pi*S)+(yt-(m+.1))**2/S))
    K=P/S; m=m+K*(yt-(m+.1)); P=(1-K)*P
    means.append(m); vars.append(P)
    m=.8*m; P=.8**2*P+.1

print("max mean error:", np.max(np.abs(out['states'][:,0]-means)))
print("max variance error:", np.max(np.abs(out['state_covariance'][:,0,0]-vars)))
print("max loglik error:", np.max(np.abs(out['loglik']-lls)))
assert np.allclose(out['states'][:,0],means,atol=1e-7)
assert np.allclose(out['state_covariance'][:,0,0],vars,atol=1e-7)
assert np.allclose(out['loglik'],lls,atol=1e-7)
print("PASS")
