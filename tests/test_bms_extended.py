"""Mathematical references for family, paired-condition and group BMS."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import betaln, expit, gammaln, logsumexp
from scipy.stats import beta

from gbmtoolbox import BMSResult, ConditionBMSResult, GroupBMSResult, bms, bms_conditions, bms_groups
from gbmtoolbox.model_selection import binary_frequency_posterior, dirichlet_exceedance


def known_assignments(labels, K=2):
    """Finite evidences forcing known labels to floating-point precision."""
    L = np.full((len(labels), K), -10000.0)
    L[np.arange(len(labels)), labels] = 0.0
    return L


def exact_label_evidence(labels, alpha0):
    counts = np.bincount(labels, minlength=len(alpha0))
    posterior = np.asarray(alpha0) + counts
    return gammaln(alpha0.sum()) - gammaln(posterior.sum()) + np.sum(gammaln(posterior) - gammaln(alpha0))


def test_rfx_free_energy_matches_dirichlet_integral_for_known_labels():
    labels = np.array([0, 0, 1, 0, 2, 1, 0])
    a0 = np.array([0.2, 0.4, 0.8])
    result = bms(known_assignments(labels, 3), alpha0=a0, n_samples=5000)
    np.testing.assert_allclose(result.posterior_parameters, a0 + np.bincount(labels, minlength=3), atol=1e-12)
    assert result.log_evidence == pytest.approx(exact_label_evidence(labels, a0), abs=1e-12)
    assert result.null_log_evidence == pytest.approx(-len(labels) * np.log(3), abs=1e-12)
    assert result.bor == pytest.approx(expit(result.null_log_evidence - result.log_evidence), abs=1e-12)


@pytest.mark.parametrize("alpha", [[1, 1], [2, 7], [30, 2], [0.1, 0.3]])
def test_binary_exceedance_is_analytic_beta_tail(alpha):
    result = dirichlet_exceedance(alpha, n_samples=1)
    assert result[0] == pytest.approx(beta.sf(0.5, *alpha), abs=1e-14)
    assert result.sum() == pytest.approx(1, abs=1e-15)


def test_family_aggregation_null_and_unequal_family_sizes():
    L = np.array([[2, 0, -1], [-1, 1, 0], [1, -2, 3], [0, 0, 0]], dtype=float)
    result = bms(L, families=[[0], [1, 2]], family_names=["a", "b"], n_samples=2000, family_method="aggregate")
    np.testing.assert_allclose(result.alpha0, [0.5, 0.25, 0.25])
    fam = result.families
    expected_a = np.array([result.posterior_parameters[0], result.posterior_parameters[1:].sum()])
    np.testing.assert_allclose(fam.posterior_parameters, expected_a)
    np.testing.assert_allclose(fam.model_frequency, expected_a / expected_a.sum())
    np.testing.assert_allclose(fam.responsibilities, np.column_stack([result.responsibilities[:, 0], result.responsibilities[:, 1:].sum(axis=1)]))
    null = np.log(0.5 * np.exp(L[:, 0]) + 0.25 * np.exp(L[:, 1]) + 0.25 * np.exp(L[:, 2])).sum()
    assert fam.null_log_evidence == pytest.approx(null, abs=1e-12)
    assert fam.bor == pytest.approx(expit(null - result.log_evidence), abs=1e-12)
    np.testing.assert_allclose(fam.protected_exceedance_prob, (1 - fam.bor) * fam.exceedance_prob + fam.bor / 2)
    assert len(fam.within) == 2
    assert fam.within[0].model_frequency.tolist() == [1]
    np.testing.assert_allclose(fam.within[1].alpha0, [0.5, 0.5])


def test_family_dirichlet_aggregation_matches_sample_distribution():
    result = bms(known_assignments([0, 0, 1, 2, 2, 3], 4), families=[[0, 1], [2, 3]], n_samples=1000, family_method="aggregate")
    samples = np.random.default_rng(9).dirichlet(result.posterior_parameters, size=50000)
    summed = samples[:, :2].sum(axis=1)
    expected = result.families
    assert summed.mean() == pytest.approx(expected.model_frequency[0], abs=0.006)
    assert np.mean(summed > 0.5) == pytest.approx(expected.exceedance_prob[0], abs=0.009)


def test_existing_2d_default_prior_and_list_matrix_dispatch():
    L = [[2.0, 0.0], [3.0, 0.0]]
    result = bms(L, n_samples=1)
    assert isinstance(result, BMSResult)
    np.testing.assert_array_equal(result.alpha0, [1, 1])


def test_condition_tuple_evidence_matches_direct_product():
    L = np.array([[[1, 2], [3, -1]], [[0, 1], [-2, 4]]], dtype=float)
    out = bms_conditions(L, n_samples=2000)
    assert isinstance(out, ConditionBMSResult)
    np.testing.assert_array_equal(out.tuples, [[0, 0], [0, 1], [1, 0], [1, 1]])
    direct = np.column_stack([L[:, i, 0] + L[:, j, 1] for i, j in out.tuples])
    expected = bms(direct, families=[[0, 3], [1, 2]], family_names=["equal", "not_equal"], within_families=False, n_samples=2000)
    np.testing.assert_allclose(out.between[0].posterior_parameters, expected.posterior_parameters, atol=1e-12)
    assert out.between[0].log_evidence == pytest.approx(expected.log_evidence, abs=1e-12)
    assert out.xp == pytest.approx(expected.families.exceedance_prob[0], abs=1e-14)
    assert out.pxp == pytest.approx(expected.families.protected_exceedance_prob[0], abs=1e-14)
    assert out.bor[0] == expected.families.bor
    np.testing.assert_array_equal(out.n_equal, [2])
    np.testing.assert_allclose(out.between[0].families.alpha0, [0.5, 0.5])


def test_condition_equal_family_can_use_different_models():
    L = np.stack([known_assignments([0] * 12, 4), known_assignments([1] * 12, 4)], axis=2)
    family = bms(L, families=[[0, 1], [2, 3]], family_names=["a", "b"], n_samples=2000)
    models = bms(L, n_samples=2000)
    assert family.xp > 0.99
    assert models.xp < 0.01
    assert family.best[0].is_equal
    np.testing.assert_array_equal(family.best[0].models, [0, 1])
    assert family.best[0].family_names == ("a", "a")
    assert family.bor[0] == family.between[0].families.bor


def test_factorial_axis_masks_and_selected_labels():
    L = np.stack([known_assignments([m] * 10) for m in [0, 0, 1, 1]], axis=2)
    out = bms_conditions(L, factors=np.array([[0, 1], [2, 3]]), n_samples=3000)
    assert out.factor_axes == (0, 1)
    np.testing.assert_array_equal(out.n_equal, [4, 4])
    assert out.xp[0] < 0.01 and out.xp[1] > 0.99
    for t, models in enumerate(out.tuples):
        assert out.equal_mask[0, t] == (models[0] == models[2] and models[1] == models[3])
        assert out.equal_mask[1, t] == (models[0] == models[1] and models[2] == models[3])


def test_one_condition_fallback_and_vacuous_single_family():
    L = np.stack([known_assignments([0, 1])], axis=2)
    with pytest.warns(UserWarning, match="one condition"):
        result = bms(L, n_samples=10)
    assert isinstance(result, BMSResult)
    out = bms(np.concatenate([L, L], axis=2), families=[[0, 1]], n_samples=100)
    assert out.xp == out.pxp == 1
    assert out.n_not_equal[0] == 0


def test_between_groups_matches_exact_shared_separate_frequency_integrals():
    labels = [np.array([0] * 9 + [1] * 1), np.array([0] * 2 + [1] * 7)]
    a0 = np.array([0.4, 0.6])
    result = bms_groups([known_assignments(l) for l in labels], alpha0=a0, n_samples=1)
    expected_equal = exact_label_evidence(np.concatenate(labels), a0)
    expected_different = sum(exact_label_evidence(l, a0) for l in labels)
    assert result.log_evidence_equal == pytest.approx(expected_equal, abs=1e-12)
    assert result.log_evidence_different == pytest.approx(expected_different, abs=1e-12)
    assert result.log_bayes_factor == pytest.approx(expected_different - expected_equal, abs=1e-12)
    assert result.probability_equal == pytest.approx(expit(expected_equal - expected_different), abs=1e-12)
    assert isinstance(bms([known_assignments(l) for l in labels], n_samples=1), GroupBMSResult)
    np.testing.assert_array_equal(result.group_sizes, [10, 9])


def test_same_winning_model_does_not_imply_same_group_frequencies():
    groups = [known_assignments([0] * 95 + [1] * 5), known_assignments([0] * 55 + [1] * 45)]
    out = bms_groups(groups, n_samples=1)
    assert all(np.argmax(r.model_frequency) == 0 for r in out.per_group)
    assert out.probability_different > 0.99


def test_family_group_comparison_normalizes_unequal_family_sizes():
    groups = [np.array([[5, -1, -2], [4, -2, -2], [-2, 4, 3]], dtype=float), np.array([[4, -2, -1], [-2, 5, 5]], dtype=float)]
    parts = [[0], [1, 2]]
    out = bms_groups(groups, families=parts, n_samples=1000)
    assert out.comparison_level == "families"
    collapsed = [np.column_stack([L[:, 0], logsumexp(L[:, 1:], axis=1) - np.log(2)]) for L in groups]
    direct = bms_groups(collapsed, alpha0=np.array([0.5, 0.5]), n_samples=1)
    assert out.probability_equal == pytest.approx(direct.probability_equal, abs=1e-12)
    assert out.log_bayes_factor == pytest.approx(direct.log_bayes_factor, abs=1e-12)
    full_models = bms_groups(groups, families=parts, group_level="models", n_samples=1000)
    assert full_models.log_evidence_equal == full_models.pooled.log_evidence


def test_between_groups_same_family_different_internal_model():
    groups = [known_assignments([0] * 25, 4), known_assignments([1] * 25, 4)]
    family = bms_groups(groups, families=[[0, 1], [2, 3]], n_samples=1000)
    models = bms_groups(groups, families=[[0, 1], [2, 3]], group_level="models", n_samples=1000)
    assert family.probability_equal > 0.8
    assert models.probability_equal < 0.01


def test_group_prior_odds_and_three_groups():
    groups = [known_assignments([0] * n) for n in [4, 7, 3]]
    out = bms_groups(groups, prior_equal=0.8, n_samples=1)
    assert out.probability_equal == pytest.approx(expit(-out.log_bayes_factor + np.log(4)), abs=1e-14)
    assert out.probability_equal + out.probability_different == 1
    assert len(out.per_group) == 3
    assert not hasattr(out, "exceedance_prob")


def test_offset_invariance_for_family_condition_and_group_probabilities():
    L = np.array([[1, -2, 0], [-1, 3, 2], [4, 0, 1]], dtype=float)
    parts = [[0], [1, 2]]
    shifts = np.array([10000.0, -5000.0, 5000.0])
    a = bms(L, families=parts, n_samples=1000)
    b = bms(L + shifts[:, None], families=parts, n_samples=1000)
    np.testing.assert_allclose(a.model_frequency, b.model_frequency, atol=1e-12)
    assert a.families.bor == pytest.approx(b.families.bor, abs=1e-12)
    C = np.stack([L, L[:, ::-1]], axis=2)
    a = bms(C, families=parts, n_samples=1000)
    b = bms(C + shifts[:, None, None], families=parts, n_samples=1000)
    np.testing.assert_allclose(a.bor, b.bor, atol=1e-12)
    a = bms_groups([L, L[:2]], families=parts, n_samples=1000)
    b = bms_groups([L + shifts[:, None], L[:2] + shifts[:2, None]], families=parts, n_samples=1000)
    assert a.probability_equal == pytest.approx(b.probability_equal, abs=1e-11)


@pytest.mark.parametrize("parts", [[], [[0], []], [[0, 1], [1, 2]], [[0], [1]], [[0], [1, 3]], [[0.0], [1.0, 2.0]], [[True], [1, 2]]])
def test_invalid_family_partitions(parts):
    with pytest.raises(ValueError, match="famil"):
        bms(np.zeros((2, 3)), families=parts, n_samples=1)


@pytest.mark.parametrize("L", [np.zeros((0, 2)), np.zeros((2, 0)), [[0, np.nan]], [[0, np.inf]], np.zeros(2)])
def test_invalid_evidence_matrices(L):
    with pytest.raises(ValueError, match="matrix"):
        bms(L, n_samples=1)


def test_invalid_controls_and_tuple_guard_before_allocation():
    for kwargs in [
        {"n_samples": 0},
        {"n_samples": 1.5},
        {"tol": 0},
        {"tol": np.nan},
        {"maxiter": 0},
        {"alpha0": [np.nan, 1]},
        {"alpha0": [0, 1]},
        {"family_names": ["x"]},
    ]:
        with pytest.raises(ValueError):
            bms(np.zeros((2, 2)), **kwargs)
    with pytest.raises(ValueError, match="max_tuples"):
        bms(np.zeros((1, 3, 12)), max_tuples=10, n_samples=1)
    with pytest.raises(ValueError, match="factors"):
        bms(np.zeros((2, 2, 4)), factors=[[0, 1], [1, 3]], n_samples=1)
    with pytest.raises(ValueError, match="at least two"):
        bms_groups([np.zeros((2, 2))], n_samples=1)
    with pytest.raises(ValueError, match="same model"):
        bms_groups([np.zeros((2, 2)), np.zeros((2, 3))], n_samples=1)
    with pytest.raises(ValueError, match="prior_equal"):
        bms_groups([np.zeros((2, 2))] * 2, prior_equal=1, n_samples=1)
    with pytest.raises(ValueError, match="requires families"):
        bms_groups([np.zeros((2, 2))] * 2, group_level="families", n_samples=1)


def test_seeded_chunked_sampling_is_reproducible():
    a = dirichlet_exceedance([1.0, 2.0, 3.0], n_samples=5000, random_state=9)
    b = dirichlet_exceedance([1.0, 2.0, 3.0], n_samples=5000, random_state=9)
    np.testing.assert_array_equal(a, b)
    assert a.sum() == pytest.approx(1)


def test_uninformative_rows_do_not_invent_counts_or_group_evidence():
    L = np.zeros((20, 3))
    result = bms(L, families=[[0], [1, 2]], n_samples=1000)
    np.testing.assert_allclose(result.posterior_parameters, result.alpha0, atol=1e-15)
    np.testing.assert_allclose(result.families.model_frequency, [0.5, 0.5], atol=1e-15)
    np.testing.assert_allclose(result.families.exceedance_prob, [0.5, 0.5], atol=1e-15)
    np.testing.assert_allclose(result.families.protected_exceedance_prob, [0.5, 0.5], atol=1e-15)
    assert result.n_informative_subjects == result.families.n_informative_subjects == 0
    groups = bms_groups([np.zeros((10, 2)), np.zeros((10, 2))], prior_equal=0.7, n_samples=1)
    assert groups.probability_equal == pytest.approx(0.7, abs=1e-15)
    assert groups.log_bayes_factor == pytest.approx(0, abs=1e-15)
    varied = np.array([[1.0, 0.0], [0.0, 2.0]])
    a = bms(varied, n_samples=1)
    b = bms(np.vstack([varied, np.ones((5, 2)) * 10]), n_samples=1)
    np.testing.assert_allclose(a.posterior_parameters, b.posterior_parameters, atol=1e-14)
    assert b.log_evidence == pytest.approx(a.log_evidence + 50, abs=1e-12)


def test_binary_frequency_integration_matches_single_subject_beta_mixture():
    a, b = 0.5, 0.5
    out = binary_frequency_posterior(np.log([[4.0, 1.0]]), [a, b])
    assert out["log_evidence"] == pytest.approx(np.log(2.5), abs=1e-12)
    assert out["frequency"][0] == pytest.approx(0.65, abs=1e-12)
    assert out["variance"][0] == pytest.approx(0.1025, abs=1e-12)
    expected_ep = (beta.sf(0.5, a, b) + 3 * a / (a + b) * beta.sf(0.5, a + 1, b)) / 2.5
    assert out["exceedance"][0] == pytest.approx(expected_ep, abs=1e-10)
    np.testing.assert_allclose(out["responsibilities"], [[0.8, 0.2]], atol=1e-12)


def test_weak_evidence_unequal_families_remains_close_to_the_prior():
    L = np.tile([0.001, 0.0, 0.0], (20, 1))
    out = bms(L, families=[[0], [1, 2]], n_samples=1000)
    assert abs(out.families.model_frequency[0] - 0.5) < 0.01
    assert abs(out.families.exceedance_prob[0] - 0.5) < 0.01
    assert out.families.frequency_variance[0] > 0.12
    assert out.families.posterior_parameters is None
    assert out.families.expected_counts is not None
    assert out.families.method == "marginal_binary_quadrature"


def test_weak_group_evidence_matches_independent_beta_weighted_integrals():
    groups = [np.array([[0.01, 0.0], [0.0, 0.02], [0.015, 0.0]]), np.array([[0.0, 0.01], [0.02, 0.0]])]
    prior = np.array([0.5, 0.5])

    def integral(L):
        def likelihood(f):
            return np.prod(f * np.exp(L[:, 0]) + (1 - f) * np.exp(L[:, 1]))

        z = quad(likelihood, 0, 1, weight="alg", wvar=(prior[0] - 1, prior[1] - 1), epsabs=1e-12)[0]
        return np.log(z) - betaln(*prior)

    Fe = integral(np.vstack(groups))
    Fd = sum(integral(L) for L in groups)
    out = bms_groups(groups, alpha0=prior, n_samples=1)
    assert out.evidence_method == "binary_quadrature"
    assert out.log_bayes_factor == pytest.approx(Fd - Fe, abs=1e-11)
    assert out.probability_equal == pytest.approx(expit(Fe - Fd), abs=1e-11)


def test_exact_family_inference_is_invariant_to_model_duplication():
    first = np.array([[0.01, 0.0], [-0.02, 0.0], [0.03, 0.0]])
    repeated = np.column_stack([first[:, 0], first[:, 1], first[:, 1]])
    a = bms(first, families=[[0], [1]], n_samples=1)
    b = bms(repeated, families=[[0], [1, 2]], n_samples=1000)
    np.testing.assert_allclose(a.families.model_frequency, b.families.model_frequency, atol=1e-13)
    np.testing.assert_allclose(a.families.exceedance_prob, b.families.exceedance_prob, atol=1e-13)
