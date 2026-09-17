import numpy as np
import pytest

from gbmtoolbox import (
    Config, GaussianPrior, Priors, StateModel, individual_fit,
    prior_sensitivity, bms,
)
from gbmtoolbox.display import _trace_styles
from gbmtoolbox.parameters import DEFAULT_OBSERVATION_NOISE_PRIOR_VARIANCE
from gbmtoolbox.validation import validate_fit_spec


def _fixed_priors():
    return Priors(GaussianPrior([0],[0]), GaussianPrior([0],[0]))


def test_initial_covariance_scalar_vector_matrix():
    for spec, ref in [
        (.2, .2*np.eye(2)),
        ([.2,.3], np.diag([.2,.3])),
        ([[.2,.05],[.05,.3]], np.array([[.2,.05],[.05,.3]])),
    ]:
        m=StateModel(lambda x,t,u,y:x,lambda x,p,u:.5,"bernoulli",_fixed_priors(),[0,0],initial_state_covariance=spec)
        np.testing.assert_allclose(m.initial_covariance_matrix(),ref)


def test_process_covariance_dimension_follows_state_not_theta():
    m=StateModel(
        lambda x,t,u,y:x,lambda x,p,u:.5,"bernoulli",
        Priors(GaussianPrior([0,0,0],[1,1,1]),GaussianPrior([0],[0])),
        [0,0],process_covariance=[.1,.2]
    )
    assert m.process_covariance_matrix(np.zeros(3),None).shape==(2,2)


def test_gaussian_defaults_to_estimated_observation_covariance():
    m=StateModel(lambda x,t,u,y:x,lambda x,p,u:[0.],"gaussian",_fixed_priors(),[0])
    assert m.observation_covariance_mode=="diagonal"
    assert "log_observation_sd" in m.parameter_layout.names
    var=np.diag(m.parameter_layout.covariance)[m.parameter_layout.observation_noise_slice]
    assert var==pytest.approx([DEFAULT_OBSERVATION_NOISE_PRIOR_VARIANCE])


def test_explicit_observation_covariance_overrides_default():
    m=StateModel(lambda x,t,u,y:x,lambda x,p,u:[0.],"gaussian",_fixed_priors(),[0],observation_covariance=0.25)
    assert m.observation_covariance_mode=="fixed"
    assert "log_observation_sd" not in m.parameter_layout.names
    assert float(m.observation_covariance_matrix(np.zeros(1),None,1)[0,0])==pytest.approx(0.25)


def test_discrete_rejects_observation_covariance():
    with pytest.raises(ValueError, match="only used for Gaussian"):
        StateModel(lambda x,t,u,y:x,lambda x,p,u:.5,"bernoulli",_fixed_priors(),[0],observation_covariance=1.)


def test_parameter_names_unique_across_blocks():
    priors=Priors(GaussianPrior([0],[1],names=["same"]),GaussianPrior([0],[1],names=["same"]))
    with pytest.raises(ValueError, match="unique across"):
        _=priors.names


def test_categorical_validation_requires_integer_y():
    model=StateModel(
        lambda x,t,u,y:x,
        lambda x,p,u:np.array([.4,.6]),
        "categorical",_fixed_priors(),[0]
    )
    with pytest.raises(ValueError,match="integer labels"):
        validate_fit_spec([{"y":np.array([0.,.5]),"u":None}],model,Config())


def test_trace_style_order():
    s=_trace_styles(6)
    assert s[0]==("0.000","-")
    assert s[1]==("0.000","--")
    assert s[2][1]=="-" and s[3][1]=="--"
    assert float(s[2][0]) > 0


def test_seed_reproducibility_for_map(binary_model,binary_data):
    cfg=Config(num_init=3,verbose=False,random_state=42)
    a=individual_fit(binary_data,binary_model,config=cfg)
    b=individual_fit(binary_data,binary_model,config=cfg)
    np.testing.assert_allclose(a.output.parameters,b.output.parameters)
    np.testing.assert_allclose(a.output.log_evidence,b.output.log_evidence)


def test_seed_reproducibility_for_bms():
    L=np.array([[1.,0.],[2.,0.],[1.5,0.]])
    a=bms(L,n_samples=5000,random_state=42)
    b=bms(L,n_samples=5000,random_state=42)
    np.testing.assert_allclose(a.exceedance_prob,b.exceedance_prob)


def test_prior_sensitivity_runs(binary_model,binary_data):
    p1=binary_model.priors
    p2=Priors(GaussianPrior([0],[2],names=["alpha_raw"]),GaussianPrior([1],[2],names=["log_beta"]))
    out=prior_sensitivity(binary_data,binary_model,[p1,p2],config=Config(num_init=1,verbose=False))
    assert len(out.posterior_means)==2
    assert out.posterior_means[0].shape==(1,2)


def test_callable_observation_covariance_uses_phi():
    model=StateModel(
        lambda x,t,u,y:x,
        lambda x,p,u:np.array([x[0]]),
        "gaussian",
        Priors(GaussianPrior([0],[0]),GaussianPrior([0],[1])),
        [0], observation_covariance=lambda phi,u: np.exp(2*phi[0])
    )
    R=model.observation_covariance_matrix(np.array([np.log(2.)]),None,1)
    np.testing.assert_allclose(R,[[4.]])
