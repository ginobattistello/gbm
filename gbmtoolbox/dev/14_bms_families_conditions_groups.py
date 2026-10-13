"""Independent references for the extended BMS methods.

RFX evidence with effectively observed model labels has an analytic Dirichlet
integral. Condition evidence is checked against direct products, binary EP
against the Beta tail, and family null evidence against a finite mixture.
"""

import numpy as np
from scipy.special import expit, gammaln
from scipy.stats import beta

from gbmtoolbox import bms, bms_conditions, bms_groups
from gbmtoolbox.model_selection import dirichlet_exceedance


def labels_lme(labels, K=2):
    L = np.full((len(labels), K), -10000.0)
    L[np.arange(len(labels)), labels] = 0
    return L


def log_integral(labels, prior):
    counts = np.bincount(labels, minlength=len(prior))
    return float(gammaln(sum(prior))-gammaln(sum(prior)+len(labels))
                 + np.sum(gammaln(prior+counts)-gammaln(prior)))


def main():
    labels = [np.array([0]*9+[1]), np.array([0]*2+[1]*7)]
    prior = np.array([0.4, 0.6])
    result = bms_groups([labels_lme(l) for l in labels], alpha0=prior, n_samples=1)
    Fe = log_integral(np.concatenate(labels), prior)
    Fd = sum(log_integral(l, prior) for l in labels)
    errors = {"shared_frequency_evidence": abs(result.log_evidence_equal-Fe),
              "separate_frequency_evidence": abs(result.log_evidence_different-Fd),
              "group_probability": abs(result.probability_equal-expit(Fe-Fd))}
    errors["binary_exceedance"] = abs(dirichlet_exceedance([2, 7])[0]-beta.sf(0.5, 2, 7))

    L = np.array([[2., 0., -1.], [-1., 1., 0.]])
    family = bms(L, families=[[0], [1, 2]], n_samples=1000).families
    exact_null = np.log(0.5*np.exp(L[:, 0])+0.25*np.exp(L[:, 1])+0.25*np.exp(L[:, 2])).sum()
    errors["family_null_mixture"] = abs(family.null_log_evidence-exact_null)

    paired = np.stack([labels_lme([0]*8), labels_lme([1]*8)], axis=2)
    condition = bms_conditions(paired, n_samples=1000)
    # Known tuple (0,1) belongs to not_equal: prior counts 1/4 each.
    expected = log_integral(np.full(8, 1, dtype=int), np.full(4, 0.25))
    errors["condition_tuple_evidence"] = abs(condition.between[0].log_evidence-expected)
    for name, value in errors.items():
        print(f"{name}: {value:.3e}")
    if max(errors.values()) > 1e-10:
        raise AssertionError("BMS reference error exceeds 1e-10")
    if condition.xp >= 0.01:
        raise AssertionError("known unequal tuples should favor not_equal")
    print("PASS")


if __name__ == "__main__":
    main()
