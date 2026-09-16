"""Reference check 06: posterior sampling versus analytic linear propagation.

If z_t = a_t * theta and theta ~ N(mu,var), then Var(z_t)=a_t^2*var exactly.
The Monte-Carlo propagation used by GBM Toolbox should converge to that result.
"""
import numpy as np

rng=np.random.default_rng(42)
mu=.3; var=.2; a=np.array([1.,2.,-1.5,4.])
draws=rng.normal(mu,np.sqrt(var),size=200_000)
z=draws[:,None]*a[None,:]
emp=np.var(z,axis=0,ddof=1)
ref=a*a*var
print("analytic variance:",ref)
print("sampling variance:",emp)
print("relative error:",np.abs(emp-ref)/ref)
assert np.allclose(emp,ref,rtol=.015)
print("PASS")
