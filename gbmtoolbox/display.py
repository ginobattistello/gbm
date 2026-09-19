"""Diagnostic plotting for GBM Toolbox individual fits."""

from __future__ import annotations

import numpy as np


def _mpl():
    """Import matplotlib lazily, so the toolbox imports without a display."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plotting requires matplotlib; install gbmtoolbox[display]") from exc
    return plt


def _trace_styles(n: int):
    """Black solid, black dashed, then progressively lighter gray pairs."""
    if n <= 0:
        return []
    n_shades = (n + 1) // 2
    if n_shades == 1:
        shades = ["0.0"]
    else:
        shades = [f"{x:.3f}" for x in np.linspace(0.0, 0.68, n_shades)]
    out = []
    for shade in shades:
        out.append((shade, "-"))
        if len(out) < n:
            out.append((shade, "--"))
    return out[:n]


def _prediction_panel(ax, family, y, pred):
    """Observed outcomes against the model's trial-wise prediction."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    if family == "bernoulli":
        ax.plot(np.arange(len(y)) + 1, pred, lw=1.0, color="0.0")
        ax.scatter(np.arange(len(y)) + 1, y, s=7, color="0.55", alpha=0.7)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(y=1) / y")
        ax.set_title("A  observation prediction")
    elif family == "categorical":
        for j, (color, ls) in enumerate(_trace_styles(pred.shape[1])):
            ax.plot(np.arange(len(y)) + 1, pred[:, j], color=color, ls=ls, lw=1.0, label=f"P({j})")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("probability")
        ax.set_title("A  observation probabilities")
        ax.legend(frameon=False, ncol=min(4, pred.shape[1]), fontsize=7)
    else:
        yy = np.asarray(y, dtype=float).reshape(len(y), -1)
        pp = np.asarray(pred, dtype=float).reshape(len(y), -1)
        for j, (color, ls) in enumerate(_trace_styles(pp.shape[1])):
            ax.plot(np.arange(len(y)) + 1, pp[:, j], color=color, ls=ls, lw=1.0)
            ax.scatter(np.arange(len(y)) + 1, yy[:, j], s=7, color=color, alpha=0.35)
        ax.set_ylabel("observed / predicted")
        ax.set_title("A  continuous prediction")
    ax.set_xlabel("trial")
    ax.grid(alpha=0.25)


def _parameter_path_panel(ax, result, subject):
    """Optimiser trajectory for each free parameter."""
    diag = result.math.diagnostics[subject]
    path = diag.search_path
    if path is None or len(path) == 0:
        ax.text(0.5, 0.5, "no path retained", transform=ax.transAxes, ha="center")
    else:
        styles = _trace_styles(path.shape[1])
        for j, (color, ls) in enumerate(styles):
            ax.plot(np.arange(path.shape[0]), path[:, j], color=color, ls=ls, lw=1.0, label=result.input.parameter_names[j])
        ax.legend(frameon=False, fontsize=7, ncol=min(3, path.shape[1]))
    ax.set_xlabel("accepted optimization step")
    ax.set_ylabel("parameter value")
    ax.set_title("B  parameter path")
    ax.grid(alpha=0.25)


def _objective_panel(ax, result, subject):
    """Log joint along the winning optimiser path."""
    diag = result.math.diagnostics[subject]
    f = diag.search_log_joint
    if f is None or len(f) == 0:
        ax.text(0.5, 0.5, "no objective path retained", transform=ax.transAxes, ha="center")
    else:
        ax.plot(np.arange(len(f)), f, color="0.0", lw=1.2)
    ax.set_xlabel("accepted optimization step")
    ax.set_ylabel("log joint")
    ax.set_title("C  optimization")
    ax.grid(alpha=0.25)


def _parameter_estimates_panel(ax, result, subject):
    """MAP estimates with their Laplace standard errors."""
    p = result.output.parameters[subject]
    cov = result.math.covariance[subject]
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0)) if cov is not None else np.full(len(p), np.nan)
    yy = np.arange(len(p))
    ax.errorbar(p, yy, xerr=1.96 * sd, fmt="o", color="0.0", ecolor="0.35", capsize=2)
    ax.set_yticks(yy)
    ax.set_yticklabels(result.input.parameter_names)
    ax.invert_yaxis()
    ax.set_xlabel("MAP ± 1.96 posterior SD")
    ax.set_title("D  fitted parameters")
    ax.grid(axis="x", alpha=0.25)


def _latent_panel(ax, result, subject):
    """Latent-state trajectory, with an uncertainty band when available."""
    est = result.output.latent[subject]["state"]
    mean = np.asarray(est["mean"], dtype=float)
    low = est.get("interval_low")
    high = est.get("interval_high")
    sd = est.get("sd")
    styles = _trace_styles(mean.shape[1])
    x = np.arange(mean.shape[0]) + 1
    for j, (color, ls) in enumerate(styles):
        label = est.get("state_names", result.input.state_names)[j]
        ax.plot(x, mean[:, j], color=color, ls=ls, lw=1.1, label=label)
        if low is not None and high is not None:
            lo = np.asarray(low)[:, j]
            hi = np.asarray(high)[:, j]
            ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0)
        elif sd is not None and est.get("uncertainty_type") == "filtered":
            s = np.asarray(sd)[:, j]
            ax.fill_between(x, mean[:, j] - 1.96 * s, mean[:, j] + 1.96 * s, color=color, alpha=0.14, linewidth=0)
    ax.set_xlabel("trial")
    ax.set_ylabel("latent state")
    title = "E  latent states"
    if est.get("uncertainty_type") == "propagated":
        title += " · propagated credible shadow"
    elif est.get("uncertainty_type") == "filtered":
        title += " · filtered Gaussian approximation"
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7, ncol=min(4, mean.shape[1]))
    ax.grid(alpha=0.25)


def _status_panel(ax, result, subject):
    """Text panel repeating the fit's key numbers and warnings."""
    diag = result.math.diagnostics[subject]
    ax.axis("off")
    starts = diag.starts
    best = max((s.log_joint for s in starts), default=result.math.log_joint[subject])
    agree = sum(abs(s.log_joint - best) <= 1e-4 * (1 + abs(best)) for s in starts)
    lines = [
        f"L-BFGS-B: {'converged' if diag.lbfgsb_success else 'WARNING'} · starts agreeing in objective: {agree}/{max(len(starts), 1)}",
        f"|gradient|: {diag.abs_grad:.2e} · invalid evaluations: {diag.n_invalid_evaluations}",
        f"observed Hessian: {diag.hess_method} · min eig: {diag.hess_raw_min_eig:.2e} · kappa: {diag.hess_condition_number:.2e}",
        f"Laplace valid: {diag.laplace_valid} · fragile: {diag.laplace_fragile} · log evidence: {result.output.log_evidence[subject]:.3f}",
    ]
    ax.text(0.0, 1.0, "F  status", va="top", fontweight="bold", transform=ax.transAxes)
    for i, line in enumerate(lines):
        ax.text(0.0, 0.80 - 0.18 * i, line, va="top", family="monospace", fontsize=8, transform=ax.transAxes)


def plot_subject(result, subject: int = 0, *, figsize=(10, 9), display=True, save=None, return_axes=False):
    """Create the standard six-panel diagnostic figure for one subject."""
    if not 0 <= subject < result.output.parameters.shape[0]:
        raise IndexError("subject index out of range")
    plt = _mpl()
    with plt.rc_context({"figure.dpi": 110, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.75], hspace=0.6, wspace=0.35)
        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[0, 1])
        axC = fig.add_subplot(gs[1, 0])
        axD = fig.add_subplot(gs[1, 1])
        axE = fig.add_subplot(gs[2, :])
        axF = fig.add_subplot(gs[3, :])
        _prediction_panel(axA, result.input.family, result.data[subject]["y"], result.output.prediction[subject])
        _parameter_path_panel(axB, result, subject)
        _objective_panel(axC, result, subject)
        _parameter_estimates_panel(axD, result, subject)
        _latent_panel(axE, result, subject)
        _status_panel(axF, result, subject)
        fig.suptitle(f"{result.input.model_name} · subject {subject}", y=0.995)
        if save is not None:
            fig.savefig(save, bbox_inches="tight")
        if display:
            backend = str(plt.get_backend()).lower()
            if "agg" not in backend:
                plt.show()
    if return_axes:
        return fig, {"prediction": axA, "path": axB, "objective": axC, "parameters": axD, "latent": axE, "status": axF}
    return fig
