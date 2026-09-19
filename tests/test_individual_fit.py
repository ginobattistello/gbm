import numpy as np
from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit


def test_binary_individual_fit_and_propagated_uncertainty(binary_model, binary_data):
    fit = individual_fit(binary_data, binary_model, config=Config(num_init=2, verbose=False, latent_uncertainty="propagated", latent_samples=80))
    assert fit.output.parameters.shape == (1,2)
    assert np.isfinite(fit.output.log_evidence[0])
    est=fit.output.latent[0]["state"]
    assert est["uncertainty_type"]=="propagated"
    assert est["covariance"].shape==(30,2,2)
    assert est["interval_low"].shape==(30,2)


def test_filtered_latent_output(gaussian_filter_model):
    y=np.array([.2,-.1,.3,.0])
    fit=individual_fit([{"y":y,"u":None}], gaussian_filter_model, config=Config(num_init=2,verbose=False,latent_uncertainty="filtered"))
    est=fit.output.latent[0]["state"]
    assert est["uncertainty_type"]=="filtered"
    assert est["interval_low"] is None
    assert est["covariance"].shape==(4,1,1)


def test_all_parameters_fixed_skips_optimizer(binary_data):
    model = StateModel(
        lambda x,th,u,y:x,
        lambda x,ph,u:.5,
        "bernoulli",
        Priors(GaussianPrior([0],[0]), GaussianPrior([0],[0])),
        [0],
    )
    fit=individual_fit(binary_data,model,config=Config(verbose=False))
    assert fit.math.hessian[0].shape==(0,0)
    assert fit.math.diagnostics[0].laplace_valid


def test_fixed_parameter_has_zero_posterior_variance(binary_data):
    def obs(x,ph,u): return ph[0]  # Bernoulli logit
    model=StateModel(
        lambda x,th,u,y:x, obs,"bernoulli",
        Priors(GaussianPrior([0],[0]), GaussianPrior([0],[1])), [0]
    )
    fit=individual_fit(binary_data,model,config=Config(num_init=2,verbose=False))
    cov=fit.math.covariance[0]
    assert cov[0,0]==0


def test_display_false_fit_remains_plottable(binary_model,binary_data):
    import matplotlib
    matplotlib.use("Agg")
    fit=individual_fit(binary_data,binary_model,config=Config(num_init=1,verbose=False,display=False))
    fig=fit.plot(subject=0,display=False)
    assert len(fig.axes)==6
