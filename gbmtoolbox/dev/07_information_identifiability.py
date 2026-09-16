"""Reference check 07: numerical local identifiability on a known ridge.

The Bernoulli probability depends only on phi0+phi1, so the two-parameter
observable Jacobian has rank one. GBM Toolbox should report one null direction.
"""
import numpy as np
from gbmtoolbox import GaussianPrior,Priors,StateModel,Config,individual_fit,numerical_local_identifiability

model=StateModel(
    lambda x,th,u,y:x,
    lambda x,ph,u:ph[0]+ph[1],  # logit depends only on the sum
    "bernoulli",
    Priors(GaussianPrior([0],[0]),GaussianPrior([0,0],[1,1],names=['phi0','phi1'])),[0]
)
data=[{"y":np.array([0,1,1,0,1,0,1,0]),"u":None}]
fit=individual_fit(data,model,config=Config(num_init=2,verbose=False))
diag=numerical_local_identifiability(fit)
print("singular values:",diag.singular_values)
print("rank/nullity:",diag.rank,diag.nullity)
print("weak direction (last right singular vector):",diag.parameter_directions[-1])
assert diag.rank==1 and diag.nullity==1
print("PASS")
