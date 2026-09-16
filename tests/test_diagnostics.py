import numpy as np
from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit
from gbmtoolbox import convergence_diagnostics, posterior_hessian_diagnostics, prior_preconditioned_information, numerical_local_identifiability


def test_diagnostics_run(binary_model,binary_data):
    fit=individual_fit(binary_data,binary_model,config=Config(num_init=2,verbose=False))
    c=convergence_diagnostics(fit)
    assert c.n_starts==2
    h=posterior_hessian_diagnostics(fit)
    assert h.eigenvalues.size==2
    info=prior_preconditioned_information(fit)
    assert info.eigenvalues.size==2
    ident=numerical_local_identifiability(fit)
    assert ident.rank <= 2


def test_local_identifiability_detects_redundant_parameters():
    # logit = phi0 + phi1: only the sum is observable, so the information
    # matrix is rank 1 in a two-parameter model.
    model=StateModel(
        lambda x,th,u,y:x,
        lambda x,ph,u:ph[0]+ph[1],
        "bernoulli",
        Priors(GaussianPrior([0],[0]),GaussianPrior([0,0],[1,1],names=["a","b"])),[0]
    )
    data=[{"y":np.array([0,1,1,0,1,0]),"u":None}]
    fit=individual_fit(data,model,config=Config(num_init=2,verbose=False))
    ident=numerical_local_identifiability(fit)
    assert ident.rank==1
    assert ident.nullity==1
