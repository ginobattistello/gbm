import numpy as np
import pytest

from gbmtoolbox import GaussianPrior, Priors


def test_scalar_vector_matrix_covariance():
    p1 = GaussianPrior([0, 0], 2.0)
    np.testing.assert_allclose(p1.covariance, 2 * np.eye(2))
    p2 = GaussianPrior([0, 0], [1, 2])
    np.testing.assert_allclose(p2.covariance, np.diag([1, 2]))
    p3 = GaussianPrior([0, 0], [[1, 0.2], [0.2, 2]])
    assert p3.covariance.shape == (2, 2)


def test_zero_variance_fixes_parameter():
    p = GaussianPrior([1, 2], [0, 3])
    assert p.fixed_mask.tolist() == [True, False]


def test_fixed_parameter_cannot_have_cross_covariance():
    with pytest.raises(ValueError, match="zero cross-covariance"):
        GaussianPrior([0, 0], [[0, 0.1], [0.1, 1]])


def test_names_must_match_and_be_unique():
    with pytest.raises(ValueError):
        GaussianPrior([0, 0], [1, 1], names=["a"])
    with pytest.raises(ValueError):
        GaussianPrior([0, 0], [1, 1], names=["a", "a"])


def test_priors_concatenate_blocks():
    p = Priors(GaussianPrior([1], [2], names=["theta"]), GaussianPrior([3, 4], [5, 6], names=["phi1", "phi2"]))
    np.testing.assert_allclose(p.mean, [1, 3, 4])
    assert p.names == ("theta", "phi1", "phi2")
    assert p.theta_slice == slice(0, 1)
    assert p.phi_slice == slice(1, 3)
