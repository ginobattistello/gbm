"""Family inference, paired conditions, and independent group comparisons.

Run from the repository root:
    python gbmtoolbox/examples/11_family_condition_group_bms.py
All evidence arrays are subjects first; family indices are zero based.
"""

import numpy as np

from gbmtoolbox import bms, bms_conditions, bms_groups

rng = np.random.default_rng(42)
families = [[0, 1], [2, 3]]
names = ["learning", "no_learning"]
L = rng.normal(-20, 1, (30, 4))
L[:, :2] += 3

standard = bms(L, families=families, family_names=names, n_samples=10000)
print("family frequencies:", standard.families.model_frequency)
print("family PXP:", standard.families.protected_exceedance_prob)

# The same subjects in two independently fitted conditions.
L_paired = np.stack([L, L + rng.normal(0, 0.3, L.shape)], axis=2)
conditions = bms_conditions(L_paired, families=families, family_names=names, n_samples=10000)
print("equal-family condition EP:", conditions.xp)
print("equal-family condition PXP:", conditions.pxp)
print("best tuple:", conditions.best[0].models)

# Independent groups can have different numbers of subjects.
L_control = rng.normal(-20, 1, (24, 4))
L_control[:, 2:] += 3
groups = bms_groups([L, L_control], families=families, family_names=names, group_names=["patients", "controls"], n_samples=10000)
print("group comparison level:", groups.comparison_level)
print("P(shared family frequencies):", groups.probability_equal)
print("P(separate family frequencies):", groups.probability_different)

# bms() dispatches on the input shape: a 3D array runs the paired-condition
# analysis and a list of 2D arrays the group comparison.
paired = bms(L_paired, families=families, family_names=names, n_samples=10000)
print("dispatched condition PXP:", paired.pxp)

# Without families the native prior is one count per model. VBA instead uses
# one count in total; pass it explicitly to reproduce that analysis.
grouped = bms([L, L_control], alpha0=np.full(4, 1 / 4), n_samples=10000)
print("model-level P(equal), VBA prior:", grouped.probability_equal)
