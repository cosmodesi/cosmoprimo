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


def test_space_from_bounds():
    space = Space(bounds={'a': (0., 1.), 'b': (-1., 1.)}, levels={'b': 3})
    assert space.params == ['a', 'b']
    assert space.limits['a'] == (0., 1.)
    assert space.levels == {'a': 2, 'b': 3}
    assert space.center == {'a': 0.5, 'b': 0.}
    # bounds alone give no correlation, so whitening cannot help
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
    space = Space(bounds={'a': (0., 1.), 'b': (0., 1.)}, levels={'b': 4})
    assert space.levels == {'a': 2, 'b': 4}
    with pytest.raises(ValueError):
        Space(bounds={'a': (0., 1.)}, levels={'nope': 3})


def test_empty_range_raises():
    with pytest.raises(ValueError):
        Space(bounds={'a': (1., 0.)})


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


def test_bounds_override_a_covariance_without_touching_its_correlations():
    """A hard bound on one parameter -- a physical positivity, a prior edge -- must not throw away
    what the chain knows about the others."""
    space = correlated_space()
    bounded = Space(mean=space.mean, covariance=space.covariance, params=space.params,
                    bounds={'b': (1.9, 2.1)})
    assert bounded.limits['b'] == (1.9, 2.1)
    assert bounded.limits['a'] == space.limits['a']
    assert np.allclose(bounded.covariance, space.covariance)
    assert bounded.is_correlated()


def test_extent_widens_where_a_bound_would_have_cut():
    """A measured reach is not a bound. `map` records where the image of a chain lands, and a
    non-linear map puts that outside `mean +- nsigma sigma` in the very directions it curves --
    intersecting it, the way a bound is intersected, would discard exactly that."""
    space = correlated_space()
    sigma = np.sqrt(np.diag(space.covariance))
    far = {'a': (space.mean[0] - 9. * sigma[0], space.mean[0] + 9. * sigma[0])}
    widened = Space(mean=space.mean, covariance=space.covariance, params=space.params, extent=far)
    assert widened.limits['a'] == far['a']
    assert widened.limits['b'] == space.limits['b']
    # and it is not a bound, so nothing may shrink for it
    assert widened.bounds == {}
    cut = Space(mean=space.mean, covariance=space.covariance, params=space.params, bounds=far)
    assert cut.limits['a'] == space.limits['a']         # a bound only ever tightens


def test_map_records_a_bounding_box_as_extent_not_as_a_bound():
    """The failure this exists for: the image of a railed chain stops short of
    `mean +- nsigma sigma` on the railed axis, and read as a bound that pulled nsigma 3.75 -> 1.27
    in all eight directions of a CMB w0waCDM box."""
    rng = np.random.default_rng(42)
    draws = rng.multivariate_normal(np.array([1., 2., 3.]), correlated_space().covariance,
                                    size=5000)
    space = Space(samples={name: draws[:, index] for index, name in enumerate('abc')})
    mapped = space.map(lambda point: {'a': point['a'], 'bc': point['b'] * point['c']})
    assert mapped.bounds == {}
    for name in mapped.params:
        assert mapped.limits[name][0] <= mapped.samples[:, mapped.params.index(name)].min()
        assert mapped.limits[name][1] >= mapped.samples[:, mapped.params.index(name)].max()


def test_marginal_keeps_bounds_bounds_and_derived_limits_derived():
    space = correlated_space()
    bounded = Space(mean=space.mean, covariance=space.covariance, params=space.params,
                    bounds={'b': (1.9, 2.1)})
    marginal = bounded.marginal(['a', 'b'])
    assert marginal.bounds == {'b': (1.9, 2.1)}         # the real bound survives
    assert 'a' not in marginal.bounds                   # the derived one does not become one
    assert marginal.limits['a'] == bounded.limits['a']


def test_marginal_does_not_transform_an_already_transformed_limit():
    """Everything a Space holds is in the expansion variable, so a sub-space must not re-apply
    the transform its parent already applied."""
    space = Space(bounds={'m': (0.04, 0.16), 'a': (0., 1.)}, transforms={'m': 'sqrt'})
    assert np.allclose(space.limits['m'], (0.2, 0.4))
    marginal = space.marginal(['m'])
    assert np.allclose(marginal.limits['m'], (0.2, 0.4))
    assert marginal.transforms['m'] == 'sqrt'


def test_engine_shrinks_for_a_bound_and_not_for_a_measured_extent():
    """`_shrink_to_limits` narrows every axis at once, so what it is allowed to fire on decides
    the whole box. A declared bound must still narrow it; a bounding box measured on a chain must
    not, however far short of `mean +- nsigma sigma` a finite sample stops."""
    from cosmoprimo.emulators.tools.engines import ChebyshevEngine

    space = correlated_space()
    sigma = np.sqrt(np.diag(space.covariance))
    short = {name: (space.mean[index] - 3. * sigma[index],
                    space.mean[index] + 3. * sigma[index])
             for index, name in enumerate(space.params)}      # 3 sigma against nsigma 3.75

    def engine(source):
        return ChebyshevEngine(budget=2, **source.geometry())

    free = Space(mean=space.mean, covariance=space.covariance, params=space.params, nsigma=3.75)
    assert engine(free).nsigma == pytest.approx(3.75)

    measured = Space(mean=space.mean, covariance=space.covariance, params=space.params,
                     nsigma=3.75, extent=short)
    assert engine(measured).nsigma == pytest.approx(3.75)

    declared = Space(mean=space.mean, covariance=space.covariance, params=space.params,
                     nsigma=3.75, bounds=short)
    assert engine(declared).nsigma < 3.75


def weighted_chain(size=20000, seed=11):
    """A chain whose multiplicities are correlated with position -- as a real MCMC chain's are,
    the tails being where a walker lingers least."""
    space = correlated_space()
    draws = np.array([[draw[name] for name in space.params]
                      for draw in space.draw(size=size, seed=seed)])
    sigma = np.sqrt(np.diag(space.covariance))
    z = (draws - space.mean) / sigma
    weights = np.exp(-0.5 * (z**2).sum(axis=1) / 4.)     # down-weight the tails
    return space, {name: draws[:, index] for index, name in enumerate(space.params)}, weights


def test_weights_change_the_moments_the_box_is_built_from():
    space, samples, weights = weighted_chain()
    unweighted = Space(samples=samples)
    weighted = Space(samples=samples, weights=weights)
    assert np.allclose(weighted.mean, np.average(unweighted.samples, axis=0, weights=weights))
    assert not np.allclose(weighted.mean, unweighted.mean)
    # down-weighting the tails narrows every axis, so the box is tighter, not merely shifted
    assert np.all(np.diag(weighted.covariance) < np.diag(unweighted.covariance))


def test_weights_are_taken_from_the_chain_when_not_passed():
    """getdist spells it `weights`, desilike `weight`; dropping either is silent and gives a
    different posterior."""
    _, samples, weights = weighted_chain()

    class Chain(dict):
        pass

    for attr in ('weights', 'weight'):
        chain = Chain(samples)
        setattr(chain, attr, weights)
        assert np.allclose(Space(samples=chain).mean,
                           Space(samples=samples, weights=weights).mean)


def test_map_and_marginal_carry_the_weights():
    _, samples, weights = weighted_chain(size=4000)
    space = Space(samples=samples, weights=weights)
    mapped = space.map(lambda point: {'a': point['a'], 'bc': point['b'] * point['c']})
    assert mapped.weights is not None and len(mapped.weights) == len(mapped.samples)
    assert np.allclose(mapped.mean, np.average(mapped.samples, axis=0, weights=mapped.weights))
    marginal = space.marginal(['a', 'b'])
    assert marginal.weights is not None and len(marginal.weights) == len(marginal.samples)


def test_weights_survive_a_state_round_trip():
    _, samples, weights = weighted_chain(size=500)
    space = Space(samples=samples, weights=weights)
    other = Space.__new__(Space)
    other.__setstate__(space.__getstate__())
    assert np.allclose(other.weights, space.weights)
    assert np.allclose(other.mean, space.mean)
    # a state written before weights were carried reads back unweighted, as it behaved
    state = space.__getstate__()
    del state['weights']
    older = Space.__new__(Space)
    older.__setstate__(state)
    assert older.weights is None


def test_bad_weights_raise():
    _, samples, weights = weighted_chain(size=100)
    with pytest.raises(ValueError):
        Space(samples=samples, weights=weights[:50])
    with pytest.raises(ValueError):
        Space(samples=samples, weights=np.zeros(len(weights)))


def test_a_bound_cuts_an_extent_even_with_no_covariance():
    """The order -- `mean +- nsigma sigma`, widened to `extent`, cut by `bounds` -- has to hold
    whether or not there is a covariance underneath it. With none, the box used to start from the
    bounds and then be widened by the extent, which is their union: a declared bound simply was
    not enforced."""
    space = Space(bounds={'x': (0., 1.)}, extent={'x': (-1., 2.), 'y': (0., 3.)})
    assert space.limits['x'] == (0., 1.)         # the bound cuts, and the extent does not undo it
    assert space.limits['y'] == (0., 3.)         # an extent alone still defines its axis
    assert space.bounds == {'x': (0., 1.)}
