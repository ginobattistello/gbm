"""Random-effects BMS, family inference, and condition/group comparisons.

GBM inputs use subjects first and zero-based model/condition indices.
The RFX model has a Dirichlet frequency prior and categorical subject labels.
Model reports use mean-field VB. Two-family marginal reports and two-category
group hypothesis evidences instead integrate the frequency by quadrature.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, betaln, expit, gammaln, logsumexp, psi, roots_jacobi


@dataclass
class BMSResult:
    """Random-effects model posterior for one subjects-by-models evidence matrix.

    ``bor`` compares the variable model-frequency profile with fixed uniform
    model frequencies. ``families.bor`` uses the explicitly specified family
    null instead. ``log_evidence`` is the RFX variational bound, not a sum of
    subject evidences or an exact RFX integral. Exactly uninformative subject
    labels are marginalized before variational inference.
    """

    posterior_parameters: np.ndarray
    model_frequency: np.ndarray
    exceedance_prob: np.ndarray
    bor: float
    protected_exceedance_prob: np.ndarray
    responsibilities: np.ndarray
    alpha0: np.ndarray | None = None
    log_evidence: float = np.nan
    null_log_evidence: float = np.nan
    n_iter: int = 0
    families: FamilyBMSResult | None = None
    _centered_log_evidence: float = field(default=np.nan, repr=False)
    n_informative_subjects: int = 0


@dataclass
class FamilyBMSResult:
    """Family inference using marginal evidence (default) or aggregation.

    The null fixes family mass at 1/F and model mass within each family at
    1/size. Within-family fits are separate conditional analyses. For two
    marginal families the frequency posterior is integrated and is generally
    not Dirichlet: posterior_parameters is None, with moments and expected
    counts reported separately. Aggregate reports retain Dirichlet parameters.
    """

    names: tuple[str, ...]
    indices: tuple[np.ndarray, ...]
    posterior_parameters: np.ndarray | None
    model_frequency: np.ndarray
    exceedance_prob: np.ndarray
    bor: float
    protected_exceedance_prob: np.ndarray
    responsibilities: np.ndarray
    alpha0: np.ndarray
    null_log_evidence: float
    within: tuple[BMSResult, ...] = ()
    log_evidence: float = np.nan
    method: str = "aggregate"
    n_informative_subjects: int = 0
    expected_counts: np.ndarray | None = None
    frequency_variance: np.ndarray | None = None


@dataclass
class TupleSelection:
    """The most frequent condition tuple under one equal/not-equal comparison."""

    tuple_idx: int
    models: np.ndarray
    families: np.ndarray
    family_names: tuple[str, ...] | None
    is_equal: bool


@dataclass
class ConditionBMSResult:
    """Paired condition stability, or stability along each factorial axis.

    Each entry of equal_* concerns the prevalence of equal versus unequal
    subject tuples. It is not a posterior probability that all subjects use
    the same model. Single-factor ``xp``/``pxp`` convenience properties are
    scalars; factorial results expose one entry per non-singleton axis.
    """

    equal_exceedance_prob: np.ndarray
    equal_protected_exceedance_prob: np.ndarray
    bor: np.ndarray
    tuples: np.ndarray
    equal_mask: np.ndarray
    between: tuple[BMSResult, ...]
    per_condition: tuple[BMSResult, ...]
    best: tuple[TupleSelection, ...]
    factors: np.ndarray | None
    factor_axes: tuple[int, ...]

    @property
    def xp(self):
        """Equal-tuple exceedance probability: a scalar for one axis, else one per axis."""
        return float(self.equal_exceedance_prob[0]) if len(self.equal_exceedance_prob) == 1 else self.equal_exceedance_prob

    @property
    def pxp(self):
        """Equal-tuple protected exceedance probability, shaped like ``xp``."""
        return float(self.equal_protected_exceedance_prob[0]) if len(self.equal_protected_exceedance_prob) == 1 else self.equal_protected_exceedance_prob

    @property
    def n_tuples(self):
        """Number of model (or family) tuples across conditions."""
        return len(self.tuples)

    @property
    def n_equal(self):
        """Number of equal tuples, per comparison axis."""
        return self.equal_mask.sum(axis=1)

    @property
    def n_not_equal(self):
        """Number of unequal tuples, per comparison axis."""
        return self.n_tuples - self.n_equal


@dataclass
class GroupBMSResult:
    """Posterior odds of shared versus separate population frequencies.

    Unlike condition exceedance probabilities, probability_equal is a
    posterior probability for a hypothesis. With two comparison categories
    it uses integrated RFX evidence; larger spaces use variational bounds.
    log_bayes_factor is log evidence for DIFFERENT minus that for EQUAL.
    """

    probability_equal: float
    probability_different: float
    log_bayes_factor: float
    log_evidence_equal: float
    log_evidence_different: float
    prior_equal: float
    comparison_level: str
    pooled: BMSResult
    per_group: tuple[BMSResult, ...]
    comparison_pooled: BMSResult
    comparison_per_group: tuple[BMSResult, ...]
    group_sizes: np.ndarray
    group_names: tuple[str, ...]
    evidence_method: str = "variational"
    variational_log_evidence_equal: float = np.nan
    variational_log_evidence_different: float = np.nan


def binary_frequency_posterior(lme, alpha0):
    """Integrate a two-category RFX frequency instead of factorizing its labels.

    The posterior is proportional to Beta(a,b) times a product of N linear
    mixtures. It is generally NOT Beta/Dirichlet. Gauss-Jacobi quadrature
    integrates that polynomial and its moments; weighted adaptive quadrature
    integrates the half-interval exceedance probabilities. No sampling occurs.
    """
    L = _matrix(lme)
    if L.shape[1] != 2:
        raise ValueError("binary frequency integration requires two columns")
    prior = _prior(alpha0, 2, None)
    a, b = prior
    offset = L.max(axis=1)
    centered = L - offset[:, None]
    informative = np.any(centered != 0, axis=1)
    active = centered[informative]
    if not len(active):
        mean = a / (a + b)
        variance = a * b / ((a + b) ** 2 * (a + b + 1))
        return dict(
            log_evidence=float(offset.sum()),
            centered_log_evidence=0.0,
            frequency=np.array([mean, 1 - mean]),
            variance=np.array([variance, variance]),
            exceedance=dirichlet_exceedance(prior),
            responsibilities=np.tile([mean, 1 - mean], (len(L), 1)),
            expected_counts=prior.copy(),
            n_informative=0,
        )
    nodes = max(48, (len(active) + 4) // 2)
    # SciPy evaluates a removable 0/0 in an unused recurrence branch when
    # a+b=1; the returned nodes/weights remain finite and are checked below.
    with np.errstate(invalid="ignore", divide="ignore"):
        x, weights = roots_jacobi(nodes, b - 1, a - 1)
    if not np.all(np.isfinite(weights)) or not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise FloatingPointError("binary quadrature weights overflowed; use less extreme prior counts")
    r = (x + 1) / 2
    weights = weights / weights.sum()
    log_r, log_one_r = np.log(r), np.log1p(-r)
    loglik = np.zeros(nodes)
    # Bound the temporary grid-by-subject array rather than allocating N^2.
    block = max(1, min(256, 2000000 // nodes))
    for start in range(0, len(active), block):
        rows = active[start : start + block]
        mixture = np.logaddexp(log_r[:, None] + rows[None, :, 0], log_one_r[:, None] + rows[None, :, 1])
        loglik += mixture.sum(axis=1)
    with np.errstate(divide="ignore"):
        logweights = np.log(weights) + loglik
    logZ = float(logsumexp(logweights))
    posterior_weights = np.exp(logweights - logZ)
    mean = float(posterior_weights @ r)
    variance = max(0.0, float(posterior_weights @ (r * r)) - mean * mean)
    responsibilities = np.tile([mean, 1 - mean], (len(L), 1))
    probs = np.zeros((len(active), 2))
    for start in range(0, len(active), block):
        rows = active[start : start + block]
        first = log_r[:, None] + rows[None, :, 0]
        mixture = np.logaddexp(first, log_one_r[:, None] + rows[None, :, 1])
        probs[start : start + len(rows), 0] = posterior_weights @ np.exp(first - mixture)
    probs[:, 1] = 1 - probs[:, 0]
    responsibilities[informative] = probs
    expected_counts = prior + probs.sum(axis=0)
    scale = float(loglik.max())

    def likelihood(f):
        """Log likelihood of the informative subjects at first-column frequency ``f``."""
        if f <= 0:
            return float(active[:, 1].sum())
        if f >= 1:
            return float(active[:, 0].sum())
        return float(np.logaddexp(np.log(f) + active[:, 0], np.log1p(-f) + active[:, 1]).sum())

    left, _ = quad(lambda f: (1 - f) ** (b - 1) * np.exp(likelihood(f) - scale), 0, 0.5, weight="alg", wvar=(a - 1, 0), epsabs=1e-11, epsrel=1e-10, limit=200)
    right, _ = quad(lambda f: f ** (a - 1) * np.exp(likelihood(f) - scale), 0.5, 1, weight="alg", wvar=(0, b - 1), epsabs=1e-11, epsrel=1e-10, limit=200)
    if not np.isfinite(left + right) or left + right <= 0:
        raise FloatingPointError("binary frequency tail integration failed")
    # Check the independent tail quadrature against the polynomial integral.
    tail_logZ = np.log(left + right) + scale - betaln(a, b)
    if abs(tail_logZ - logZ) > 1e-7 * (1 + abs(logZ)):
        raise FloatingPointError("binary frequency quadratures disagree; inspect evidence/prior scale")
    ep = float(right / (left + right))
    return dict(
        log_evidence=logZ + float(offset.sum()),
        centered_log_evidence=logZ,
        frequency=np.array([mean, 1 - mean]),
        variance=np.array([variance, variance]),
        exceedance=np.array([ep, 1 - ep]),
        responsibilities=responsibilities,
        expected_counts=expected_counts,
        n_informative=int(informative.sum()),
    )


def _positive_integer(value, name):
    """Return ``value`` as an int, rejecting bools and values below one."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _matrix(lme):
    """Validate a non-empty finite subjects-by-models log-evidence matrix."""
    L = np.asarray(lme, dtype=float)
    if L.ndim != 2 or min(L.shape, default=0) < 1 or not np.all(np.isfinite(L)):
        raise ValueError("lme must be a non-empty finite subjects x models matrix")
    return L


def _partition(families, K, family_names=None):
    """Validate a family partition of ``K`` models and resolve family names."""
    if families is None:
        if family_names is not None:
            raise ValueError("family_names requires families")
        return None, None
    try:
        parts = tuple(np.asarray(indices) for indices in families)
    except (TypeError, ValueError) as exc:
        raise ValueError("families must be sequences of integer model indices") from exc
    if not parts or any(p.ndim != 1 or p.size == 0 or p.dtype.kind not in "iu" for p in parts):
        raise ValueError("families must contain non-empty one-dimensional integer index lists")
    if any(np.any(p < 0) or np.any(p >= K) for p in parts):
        raise ValueError("family model indices are out of range")
    indices = np.concatenate(parts)
    if len(indices) != K or not np.array_equal(np.sort(indices), np.arange(K)):
        raise ValueError("families must be disjoint and cover every model exactly once")
    names = tuple(f"f_{i}" for i in range(len(parts))) if family_names is None else tuple(str(v) for v in family_names)
    if len(names) != len(parts) or len(set(names)) != len(names):
        raise ValueError("family_names must contain one unique name per family")
    return tuple(p.astype(int, copy=True) for p in parts), names


def _prior(alpha0, K, parts):
    """Dirichlet prior counts: explicit, one per model, or family balanced."""
    if alpha0 is None:
        if parts is None:
            return np.ones(K)
        prior = np.empty(K)
        for idx in parts:
            prior[idx] = 1.0 / (len(parts) * len(idx))
        return prior
    prior = np.asarray(alpha0, dtype=float).reshape(-1)
    if prior.shape != (K,) or not np.all(np.isfinite(prior)) or np.any(prior <= 0):
        raise ValueError("alpha0 must contain one finite positive value per model")
    return prior.copy()


def dirichlet_exceedance(alpha, n_samples=100000, random_state=42):
    """Dirichlet exceedance: analytic for one/two entries, chunked MC otherwise."""
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    if not alpha.size or not np.all(np.isfinite(alpha)) or np.any(alpha <= 0):
        raise ValueError("Dirichlet parameters must be finite and positive")
    n_samples = _positive_integer(n_samples, "n_samples")
    if len(alpha) == 1:
        return np.ones(1)
    if len(alpha) == 2:
        # r[0] ~ Beta(alpha[0], alpha[1]); P(r[0] > 1/2).
        p = float(betainc(alpha[1], alpha[0], 0.5))
        return np.array([p, 1 - p])
    rng = np.random.default_rng(random_state)
    wins = np.zeros(len(alpha), dtype=np.int64)
    # At most about 16 MB of draws per block, including large tuple spaces.
    block = min(100000, max(1, 2000000 // len(alpha)))
    remaining = n_samples
    while remaining:
        n = min(block, remaining)
        draws = rng.dirichlet(alpha, size=n)
        wins += np.bincount(np.argmax(draws, axis=1), minlength=len(alpha))
        remaining -= n
    return wins / n_samples


def _compute_fe(L, alpha, r, alpha0):
    """RFX mean-field variational bound; L and r are models x subjects."""
    Elogr = psi(alpha) - psi(np.sum(alpha))
    Sqf = np.sum(gammaln(alpha)) - gammaln(np.sum(alpha)) - np.sum((alpha - 1) * Elogr)
    Sqm = -np.sum(r * np.log(np.clip(r, np.finfo(float).tiny, 1.0)))
    ELJ = gammaln(np.sum(alpha0)) - np.sum(gammaln(alpha0)) + np.sum((alpha0 - 1) * Elogr)
    ELJ += np.sum(r * (Elogr[:, None] + L))
    return float(ELJ + Sqf + Sqm)


def _null_fe(L, probabilities=None):
    """Exact evidence for a fixed frequency vector; L is models x subjects."""
    K = L.shape[0]
    p = np.ones(K) / K if probabilities is None else np.asarray(probabilities)
    return float(np.sum(logsumexp(L + np.log(p)[:, None], axis=0)))


def _bms_matrix(
    lme,
    *,
    alpha0=None,
    families=None,
    family_names=None,
    within_families=True,
    n_samples=100000,
    random_state=42,
    tol=1e-6,
    maxiter=10000,
    family_method="marginal",
):
    """Random-effects BMS on one subjects-by-models matrix, with optional families."""
    L = _matrix(lme)
    N, K = L.shape
    _positive_integer(n_samples, "n_samples")
    maxiter = _positive_integer(maxiter, "maxiter")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if family_method not in {"marginal", "aggregate"}:
        raise ValueError("family_method must be 'marginal' or 'aggregate'")
    parts, names = _partition(families, K, family_names)
    alpha0 = _prior(alpha0, K, parts)
    offset = np.max(L, axis=1)
    centered = L - offset[:, None]
    # Exactly constant rows have p(y_s|frequency)=exp(offset_s), independent
    # of frequency. Marginalize their labels instead of imposing a mean-field
    # label/count update that invents information from missing model preference.
    informative = np.any(centered != 0.0, axis=1)
    informative_L = centered[informative]
    alpha = alpha0.copy()
    for it in range(1, maxiter + 1):
        log_r = informative_L + (psi(alpha) - psi(np.sum(alpha)))[None, :]
        r = np.exp(log_r - logsumexp(log_r, axis=1, keepdims=True))
        new_alpha = alpha0 + np.sum(r, axis=0)
        if np.linalg.norm(new_alpha - alpha) <= tol:
            alpha = new_alpha
            break
        alpha = new_alpha
    else:
        raise RuntimeError("BMS variational update did not converge")
    freq = alpha / np.sum(alpha)
    xp = dirichlet_exceedance(alpha, n_samples=n_samples, random_state=random_state)
    F1 = _compute_fe(informative_L.T, alpha, r.T, alpha0) if np.any(informative) else 0.0
    full_r = np.broadcast_to(freq, (N, K)).copy()
    full_r[informative] = r
    r = full_r
    F0 = _null_fe(centered.T)
    bor = float(expit(F0 - F1)) if K > 1 else 1.0
    evidence_offset = float(offset.sum())
    out = BMSResult(
        alpha,
        freq,
        xp,
        bor,
        (1 - bor) * xp + bor / K,
        r,
        alpha0,
        F1 + evidence_offset,
        F0 + evidence_offset,
        it,
        _centered_log_evidence=F1,
        n_informative_subjects=int(informative.sum()),
    )
    if parts is not None:
        a_f = np.array([alpha[idx].sum() for idx in parts])
        a0_f = np.array([alpha0[idx].sum() for idx in parts])
        r_f = np.column_stack([r[:, idx].sum(axis=1) for idx in parts])
        xp_f = dirichlet_exceedance(a_f, n_samples=n_samples, random_state=random_state)
        p0 = np.empty(K)
        for idx in parts:
            p0[idx] = 1.0 / (len(parts) * len(idx))
        F0_f = _null_fe(centered.T, p0)
        bor_f = float(expit(F0_f - F1)) if len(parts) > 1 else 1.0
        within = ()
        if within_families:
            within = tuple(
                _bms_matrix(L[:, idx], alpha0=np.ones(len(idx)) / len(idx), n_samples=n_samples, random_state=random_state, tol=tol, maxiter=maxiter)
                for idx in parts
            )
        out.families = FamilyBMSResult(
            names,
            parts,
            a_f,
            a_f / a_f.sum(),
            xp_f,
            bor_f,
            (1 - bor_f) * xp_f + bor_f / len(parts),
            r_f,
            a0_f,
            F0_f + evidence_offset,
            within,
            out.log_evidence,
            "aggregate",
            out.n_informative_subjects,
        )
        if family_method == "marginal":
            family_L = np.column_stack([logsumexp(centered[:, idx], axis=1) - np.log(len(idx)) for idx in parts])
            family_L += offset[:, None]
            family_fit = _bms_matrix(family_L, alpha0=a0_f, n_samples=n_samples, random_state=random_state, tol=tol, maxiter=maxiter)
            out.families = FamilyBMSResult(
                names,
                parts,
                family_fit.posterior_parameters,
                family_fit.model_frequency,
                family_fit.exceedance_prob,
                family_fit.bor,
                family_fit.protected_exceedance_prob,
                family_fit.responsibilities,
                a0_f,
                family_fit.null_log_evidence,
                within,
                family_fit.log_evidence,
                "marginal",
                family_fit.n_informative_subjects,
            )
            if len(parts) == 2:
                exact = binary_frequency_posterior(family_L, a0_f)
                F0_exact = family_fit.null_log_evidence
                # Compare centered values to cancel potentially large offsets.
                family_centered = family_L - family_L.max(axis=1, keepdims=True)
                bor_exact = float(expit(_null_fe(family_centered.T) - exact["centered_log_evidence"]))
                out.families = FamilyBMSResult(
                    names,
                    parts,
                    None,
                    exact["frequency"],
                    exact["exceedance"],
                    bor_exact,
                    (1 - bor_exact) * exact["exceedance"] + bor_exact / 2,
                    exact["responsibilities"],
                    a0_f,
                    F0_exact,
                    within,
                    exact["log_evidence"],
                    "marginal_binary_quadrature",
                    exact["n_informative"],
                    exact["expected_counts"],
                    exact["variance"],
                )
    return out


def bms(
    lme,
    *,
    alpha0=None,
    n_samples=100000,
    random_state=42,
    tol=1e-6,
    maxiter=10000,
    families=None,
    family_names=None,
    within_families=True,
    factors=None,
    max_tuples=100000,
    group_level="auto",
    prior_equal=0.5,
    group_names=None,
    family_method="marginal",
):
    """Dispatch BMS by data shape, keeping the existing 2D model API.

    * 2D: (subjects, models), optionally partitioned into families.
    * 3D ndarray: (subjects, models, paired conditions).
    * list/tuple of 2D arrays: independent groups, possibly unequal sizes.

    With no families the original alpha0=ones(K) default is preserved. With
    families the default is 1/(F*family_size), one prior count in total.
    All family and factorial condition indices are zero based.
    """
    common = dict(
        alpha0=alpha0,
        families=families,
        family_names=family_names,
        within_families=within_families,
        n_samples=n_samples,
        random_state=random_state,
        tol=tol,
        maxiter=maxiter,
        family_method=family_method,
    )
    if isinstance(lme, (list, tuple)) and len(lme) and all(np.asarray(v).ndim == 2 for v in lme):
        if factors is not None:
            raise ValueError("factors applies only to paired-condition comparisons")
        return bms_groups(lme, group_level=group_level, prior_equal=prior_equal, group_names=group_names, **common)
    arr = np.asarray(lme, dtype=float)
    if arr.ndim == 3:
        return bms_conditions(arr, factors=factors, max_tuples=max_tuples, **common)
    if factors is not None:
        raise ValueError("factors requires a subjects x models x conditions array")
    return _bms_matrix(arr, **common)


def _condition_masks(labels, factors, C):
    """Equal-tuple masks: across all conditions, or along each factorial axis."""
    if factors is None:
        return np.all(labels == labels[:, :1], axis=1)[None, :], None, (0,)
    grid = np.asarray(factors)
    if grid.ndim < 1 or grid.size != C or grid.dtype.kind not in "iu" or not np.array_equal(np.sort(grid.ravel()), np.arange(C)):
        raise ValueError("factors must contain every zero-based condition index exactly once")
    axes = tuple(axis for axis, size in enumerate(grid.shape) if size > 1)
    if not axes:
        raise ValueError("factors must have a dimension with at least two levels")
    masks = []
    for axis in axes:
        # Rows fix the other factor levels; columns vary this factor's level.
        lines = np.moveaxis(grid, axis, -1).reshape(-1, grid.shape[axis])
        chosen = labels[:, lines]
        masks.append(np.all(chosen == chosen[:, :, :1], axis=(1, 2)))
    return np.asarray(masks), grid.copy(), axes


def bms_conditions(
    lme,
    *,
    families=None,
    family_names=None,
    alpha0=None,
    within_families=True,
    factors=None,
    max_tuples=100000,
    n_samples=100000,
    random_state=42,
    tol=1e-6,
    maxiter=10000,
    family_method="marginal",
):
    """VBA-style paired-condition tuple/family inference.

    Tuple evidence is a sum of condition evidences, requiring conditional
    independence and separately integrated condition-specific parameters.
    alpha0 specifies the per-condition model prior. The between-condition
    equal/not_equal tuple families always have balanced prior mass, total 1.
    """
    L = np.asarray(lme, dtype=float)
    if L.ndim != 3 or min(L.shape) < 1 or not np.all(np.isfinite(L)):
        raise ValueError("lme must be a non-empty finite subjects x models x conditions array")
    N, K, C = L.shape
    parts, names = _partition(families, K, family_names)
    common = dict(
        families=parts,
        family_names=names,
        alpha0=alpha0,
        within_families=within_families,
        n_samples=n_samples,
        random_state=random_state,
        tol=tol,
        maxiter=maxiter,
        family_method=family_method,
    )
    if C == 1:
        if factors is not None:
            raise ValueError("factorial comparison requires at least two conditions")
        warnings.warn("Only one condition; returning standard BMS", UserWarning, stacklevel=2)
        return _bms_matrix(L[:, :, 0], **common)
    limit = _positive_integer(max_tuples, "max_tuples")
    nt = K**C
    if nt > limit:
        raise ValueError(f"{K}^{C} = {nt} tuples exceeds max_tuples={limit}; reduce the model/condition space or raise the limit explicitly")
    if nt > 10000:
        warnings.warn(f"{nt} tuples: variational inference and model exceedance sampling may be expensive", RuntimeWarning, stacklevel=2)
    tuples = np.array(list(product(range(K), repeat=C)), dtype=int)
    model_family = np.arange(K)
    if parts is not None:
        model_family = np.empty(K, dtype=int)
        for f, idx in enumerate(parts):
            model_family[idx] = f
    mask, grid, axes = _condition_masks(model_family[tuples], factors, C)
    # Removing per-subject/condition constants preserves tuple posteriors and
    # protects the summation from extreme but irrelevant evidence offsets.
    centered = L - np.max(L, axis=1, keepdims=True)
    Lt = np.zeros((N, nt))
    for c in range(C):
        Lt += centered[:, tuples[:, c], c]
    offset = float(np.max(L, axis=1).sum())
    between, best = [], []
    xp, pxp, bor = [], [], []
    for equal in mask:
        eq, neq = np.flatnonzero(equal), np.flatnonzero(~equal)
        tuple_parts = [eq, neq] if len(neq) else [eq]
        tuple_names = ["equal", "not_equal"] if len(neq) else ["equal"]
        out = _bms_matrix(
            Lt,
            families=tuple_parts,
            family_names=tuple_names,
            within_families=False,
            n_samples=n_samples,
            random_state=random_state,
            tol=tol,
            maxiter=maxiter,
            family_method=family_method,
        )
        out.log_evidence += offset
        out.null_log_evidence += offset
        out.families.null_log_evidence += offset
        out.families.log_evidence += offset
        between.append(out)
        xp.append(out.families.exceedance_prob[0])
        pxp.append(out.families.protected_exceedance_prob[0])
        bor.append(out.families.bor)
        t = int(np.argmax(out.model_frequency))
        chosen_families = model_family[tuples[t]]
        best.append(
            TupleSelection(t, tuples[t].copy(), chosen_families.copy(), tuple(names[f] for f in chosen_families) if names is not None else None, bool(equal[t]))
        )
    per_cond = tuple(_bms_matrix(L[:, :, c], **common) for c in range(C))
    return ConditionBMSResult(np.asarray(xp), np.asarray(pxp), np.asarray(bor), tuples, mask, tuple(between), per_cond, tuple(best), grid, axes)


def bms_groups(
    lmes,
    *,
    families=None,
    family_names=None,
    alpha0=None,
    group_level="auto",
    prior_equal=0.5,
    group_names=None,
    within_families=True,
    n_samples=100000,
    random_state=42,
    tol=1e-6,
    maxiter=10000,
    family_method="marginal",
):
    """Shared versus separate frequency profiles for independent groups.

    Without families, compare model frequencies. With families, auto compares
    family frequencies after uniform within-family evidence marginalization:
    log p(y|family) = logsumexp(log p(y|model)) - log(family size).
    group_level='models' instead retains the full model-frequency comparison,
    with family-balanced model priors (the VBA-style group calculation).
    """
    if not isinstance(lmes, (list, tuple)) or len(lmes) < 2:
        raise ValueError("between-group BMS requires a list/tuple of at least two groups")
    groups = tuple(_matrix(L) for L in lmes)
    K = groups[0].shape[1]
    if any(L.shape[1] != K for L in groups):
        raise ValueError("all groups must have the same model columns in the same order")
    parts, names = _partition(families, K, family_names)
    if group_level not in {"auto", "models", "families"}:
        raise ValueError("group_level must be 'auto', 'models', or 'families'")
    level = ("families" if parts is not None else "models") if group_level == "auto" else group_level
    if level == "families" and parts is None:
        raise ValueError("group_level='families' requires families")
    if not np.isfinite(prior_equal) or not 0 < prior_equal < 1:
        raise ValueError("prior_equal must lie strictly between zero and one")
    gnames = tuple(f"group_{g}" for g in range(len(groups))) if group_names is None else tuple(str(v) for v in group_names)
    if len(gnames) != len(groups) or len(set(gnames)) != len(gnames):
        raise ValueError("group_names must contain one unique name per group")
    common = dict(
        alpha0=alpha0,
        families=parts,
        family_names=names,
        within_families=within_families,
        n_samples=n_samples,
        random_state=random_state,
        tol=tol,
        maxiter=maxiter,
        family_method=family_method,
    )
    per_group = tuple(_bms_matrix(L, **common) for L in groups)
    pooled = _bms_matrix(np.vstack(groups), **common)
    compare_group, compare_pool = per_group, pooled
    if level == "families":
        collapsed = tuple(np.column_stack([logsumexp(L[:, idx], axis=1) - np.log(len(idx)) for idx in parts]) for L in groups)
        model_prior = _prior(alpha0, K, parts)
        family_prior = np.array([model_prior[idx].sum() for idx in parts])
        args = dict(alpha0=family_prior, n_samples=n_samples, random_state=random_state, tol=tol, maxiter=maxiter)
        compare_group = tuple(_bms_matrix(L, **args) for L in collapsed)
        compare_pool = _bms_matrix(np.vstack(collapsed), **args)
    F_equal = compare_pool.log_evidence
    F_different = float(sum(r.log_evidence for r in compare_group))
    variational_equal, variational_different = F_equal, F_different
    # Same subject offsets occur under both hypotheses; subtract centered bounds.
    log_bf = float(sum(r._centered_log_evidence for r in compare_group) - compare_pool._centered_log_evidence)
    evidence_method = "variational"
    comparison_data = collapsed if level == "families" else groups
    comparison_prior = family_prior if level == "families" else _prior(alpha0, K, parts)
    if len(comparison_prior) == 2:
        exact_groups = tuple(binary_frequency_posterior(L, comparison_prior) for L in comparison_data)
        exact_pool = binary_frequency_posterior(np.vstack(comparison_data), comparison_prior)
        F_equal = exact_pool["log_evidence"]
        F_different = float(sum(r["log_evidence"] for r in exact_groups))
        log_bf = float(sum(r["centered_log_evidence"] for r in exact_groups) - exact_pool["centered_log_evidence"])
        evidence_method = "binary_quadrature"
    p = float(expit(-log_bf + np.log(prior_equal) - np.log1p(-prior_equal)))
    return GroupBMSResult(
        p,
        1 - p,
        log_bf,
        F_equal,
        F_different,
        float(prior_equal),
        level,
        pooled,
        per_group,
        compare_pool,
        compare_group,
        np.array([len(L) for L in groups]),
        gnames,
        evidence_method,
        variational_equal,
        variational_different,
    )
