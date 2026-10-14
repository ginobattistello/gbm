"""GBM Toolbox: Bayesian Generative Brain/Behavior Modelling."""

import jax

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
from .hierarchical_fit import HierarchicalFitConfig, HierarchicalFitResult, hierarchical_fit
from .individual_fit import FitResult, individual_fit
from .model_selection import BMSResult, ConditionBMSResult, FamilyBMSResult, GroupBMSResult, bms, bms_conditions, bms_groups
from .optimization import Config
from .parameters import ParameterLayout, observation_noise_prior_from_scale, robust_scale
from .predictive import posterior_predictive, prior_predictive, simulate_subject
from .priors import GammaPrecisionPrior, GaussianPrior, Priors
from .sensitivity import prior_sensitivity
from .state_model import StateModel

__all__ = [
    "BMSResult",
    "ConditionBMSResult",
    "Config",
    "FamilyBMSResult",
    "FitResult",
    "GammaPrecisionPrior",
    "GaussianPrior",
    "GroupBMSResult",
    "HierarchicalFitConfig",
    "HierarchicalFitResult",
    "ObservationNoiseCorrection",
    "ParameterLayout",
    "Priors",
    "StateModel",
    "bms",
    "bms_conditions",
    "bms_groups",
    "convergence_diagnostics",
    "hierarchical_fit",
    "individual_fit",
    "noise_parameter_correlations",
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

__version__ = "0.2.0"
