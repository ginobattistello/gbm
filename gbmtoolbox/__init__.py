"""GBM Toolbox: Bayesian Generative Brain/Behavior Modelling."""

import jax

# Inference relies on Hessians, covariance inversions and log determinants.
jax.config.update("jax_enable_x64", True)

from .diagnostics import (
    ObservationNoiseCorrection,
    convergence_diagnostics,
    noise_parameter_correlations,
    numerical_local_identifiability,
    observation_noise_correction,
    posterior_hessian_diagnostics,
    prior_preconditioned_information,
)
from .hbi import HBIConfig, HBIResult, hbi_main
from .individual_fit import FitResult, individual_fit
from .model_selection import BMSResult, bms
from .optimization import Config
from .parameters import ParameterLayout, observation_noise_prior_from_scale, robust_scale
from .predictive import posterior_predictive, prior_predictive, simulate_subject
from .priors import GaussianPrior, Priors
from .sensitivity import prior_sensitivity
from .state_model import StateModel

__all__ = [
    "BMSResult",
    "Config",
    "FitResult",
    "GaussianPrior",
    "HBIConfig",
    "HBIResult",
    "ParameterLayout",
    "Priors",
    "StateModel",
    "bms",
    "convergence_diagnostics",
    "hbi_main",
    "individual_fit",
    "noise_parameter_correlations",
    "ObservationNoiseCorrection",
    "numerical_local_identifiability",
    "observation_noise_correction",
    "observation_noise_prior_from_scale",
    "posterior_hessian_diagnostics",
    "posterior_predictive",
    "prior_preconditioned_information",
    "prior_predictive",
    "prior_sensitivity",
    "robust_scale",
    "simulate_subject",
]

__version__ = "0.1.0"
