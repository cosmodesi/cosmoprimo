"""Tests for Space.

Written against measurements rather than invented cases: several assert the properties whose
absence produced real bugs (silent clipping outside the box, unreachable nodes discovered only
after a full sampling campaign, whitening a diagonal covariance and expecting a gain).
"""

import numpy as np
import pytest

from cosmoprimo.emulators.tools.space import Space


def correlated_space():
    """A CMB-like posterior: loose external constraints plus one tight constraint on a linear
    combination, which is what makes the covariance strongly correlated."""
    params = ['a', 'b', 'c']
    sigma = np.array([0.1, 0.2, 0.5])
    jac = np.array([1., -2., 0.5])          # the tightly constrained direction
    precision = np.diag(1. / sigma**2) + np.outer(jac, jac) / 1e-3**2
    return Space(mean=np.array([1., 2., 3.]), covariance=np.linalg.inv(precision), params=params)


def test_space_from_limits():
    space = Space(limits={'a': (0., 1.), 'b': (-1., 1.)}, levels={'b': 3})
    assert space.params == ['a', 'b']
    assert space.limits['a'] == (0., 1.)
    assert space.levels == {'a': 2, 'b': 3}
    assert space.center == {'a': 0.5, 'b': 0.}
    # limits alone give no correlation, so whitening cannot help
    assert not space.is_correlated()


def test_space_from_covariance_and_samples_agree():
    space = correlated_space()
    assert space.is_correlated()
    draws = space.draw(size=20000, seed=7)
    from_samples = Space(samples={name: np.array([draw[name] for draw in draws])
                                  for name in space.params})
    assert np.allclose(from_samples.mean, space.mean, atol=0.02)
    assert np.allclose(from_samples.covariance, space.covariance, rtol=0.1, atol=1e-8)


def test_levels_override_and_unknown_name_raises():
    space = Space(limits={'a': (0., 1.), 'b': (0., 1.)}, levels={'b': 4})
    assert space.levels == {'a': 2, 'b': 4}
    with pytest.raises(ValueError):
        Space(limits={'a': (0., 1.)}, levels={'nope': 3})


def test_empty_range_raises():
    with pytest.raises(ValueError):
        Space(limits={'a': (1., 0.)})


def test_covariance_requires_params():
    with pytest.raises(ValueError):
        Space(mean=[0.], covariance=[[1.]])


def test_covariance_and_marginal():
    space = correlated_space()
    marginal = space.marginal(['a', 'c'])
    # marginalising is taking the sub-block; conditioning (a Schur complement) would describe the
    # region at fixed values of the dropped parameters and shrink the box wrongly
    assert np.allclose(marginal.covariance, space.covariance[np.ix_([0, 2], [0, 2])])
    conditional = np.linalg.inv(np.linalg.inv(space.covariance)[np.ix_([0, 2], [0, 2])])
    assert not np.allclose(marginal.covariance, conditional)


def test_limits_override_a_covariance_without_touching_its_correlations():
    """A hard bound on one parameter -- a physical positivity, a prior edge -- must not throw away
    what the chain knows about the others."""
    space = correlated_space()
    bounded = Space(mean=space.mean, covariance=space.covariance, params=space.params,
                    limits={'b': (1.9, 2.1)})
    assert bounded.limits['b'] == (1.9, 2.1)
    assert bounded.limits['a'] == space.limits['a']
    assert np.allclose(bounded.covariance, space.covariance)
    assert bounded.is_correlated()
