"""Reproducible observation-only cold/warm fit timings (no time assertions).

Run from the repository root: python gbmtoolbox/dev/13_fit_performance.py
For before/after measurements, run this same file with each checkout on
PYTHONPATH in separate processes. Compilation disk caching is disabled here.
"""

import argparse
import json
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--unscaled", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.repeats < 1:
        parser.error("trials and repeats must be positive")
    jax.config.update("jax_enable_compilation_cache", False)
    rng = np.random.default_rng(17)
    u = rng.normal(size=args.trials)
    records = []
    for family in ("gaussian", "bernoulli", "categorical"):
        if family == "gaussian":
            y = 0.4 + 0.7*u + rng.normal(0, 0.6, len(u))
        elif family == "bernoulli":
            y = (rng.random(len(u)) < 1/(1+np.exp(-(0.4+0.7*u)))).astype(int)
        else:
            y = rng.integers(0, 3, len(u))

        def observation(x, phi, stimulus):
            eta = phi[0]+phi[1]*stimulus
            return jnp.array([0.0, eta, -0.3*eta]) if family == "categorical" else eta

        model = StateModel(observation=observation, family=family,
                           priors=Priors(observation=GaussianPrior([0, 0], [1, 1])))
        cfg_args = dict(num_init=1, verbose=False, display=False)
        if "parameter_scaling" in Config.__dataclass_fields__:
            cfg_args["parameter_scaling"] = not args.unscaled
        cfg = Config(**cfg_args)
        durations = []
        for _ in range(args.repeats+1):
            start = perf_counter()
            fit = individual_fit([{"y": y, "u": u}], model, config=cfg)
            durations.append(perf_counter()-start)
        diag = fit.math.diagnostics[0]
        records.append(dict(family=family, cold_seconds=durations[0], warm_median_seconds=float(np.median(durations[1:])),
                            warm_seconds=durations[1:], parameters=fit.output.parameters[0].tolist(),
                            log_likelihood=float(fit.math.log_likelihood[0]), log_evidence=float(fit.output.log_evidence[0]),
                            iterations=diag.starts[0].n_iter, evaluations=getattr(diag, "n_evaluations", None),
                            profile=getattr(fit.profile, "timings", {}), subject_timings=getattr(diag, "timings", {})))
    print(json.dumps(dict(jax_version=jax.__version__, numpy_version=np.__version__, backend=jax.default_backend(),
                         trials=args.trials, repeats=args.repeats,
                         parameter_scaling=(not args.unscaled) if "parameter_scaling" in Config.__dataclass_fields__ else False,
                         results=records), indent=2))


if __name__ == "__main__":
    main()
