"""Compact text reporting."""
from __future__ import annotations


def fit_summary(result, subject: int = 0) -> str:
    p = result.output.parameters[subject]
    names = result.input.parameter_names
    diag = result.math.diagnostics[subject]
    rows = [f"GBM Toolbox fit · {result.input.model_name} · subject {subject}"]
    rows.append(
        f"family={result.input.family}  loglik={result.math.log_likelihood[subject]:.3f}  "
        f"log-evidence={result.output.log_evidence[subject]:.3f}"
    )
    rows.append("parameters: " + ", ".join(f"{n}={v:.4g}" for n, v in zip(names, p)))
    if result.output.process_noise_sd.shape[1]:
        rows.append("process SD: " + ", ".join(f"{x:.4g}" for x in result.output.process_noise_sd[subject]))
    if result.output.observation_noise_sd.shape[1]:
        rows.append("observation SD: " + ", ".join(f"{x:.4g}" for x in result.output.observation_noise_sd[subject]))
    rows.append(f"Laplace valid={diag.laplace_valid}  fragile={diag.laplace_fragile}  kappa={diag.hess_condition_number:.3g}")
    return "\n".join(rows)
