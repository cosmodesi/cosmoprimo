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
    widened = Space(mean=space.mean, covariance=space.covariance, params=space.params).widen(**far)
    assert widened.limits['a'] == far['a']
    assert widened.limits['b'] == space.limits['b']
    # and it is not a bound, so nothing may shrink for it
    assert widened.bounds == {}
    cut = Space(mean=space.mean, covariance=space.covariance, params=space.params, bounds=far)
    assert cut.limits['a'] == space.limits['a']         # a bound only ever tightens


def test_map_boxes_an_introduced_parameter_by_its_image():
    """An introduced parameter's box is the image's own bounding box, never `mean +- nsigma
    sigma` of it -- three sigma about the mean of a skewed image reaches outside it, and measured
    on `Omega_cdm = omega_cdm / h^2` that put a node at -0.053, where CLASS is non-finite.

    A pass-through parameter keeps the source's box instead, so nothing re-measures an axis the
    mapping never touched."""
    rng = np.random.default_rng(42)
    draws = rng.multivariate_normal(np.array([1., 2., 3.]), correlated_space().covariance,
                                    size=5000)
    space = Space(samples={name: draws[:, index] for index, name in enumerate('abc')})
    mapped = space.map(lambda point: {'a': point['a'], 'bc': point['b'] * point['c']})
    # 'a' passes through, so it keeps the source's box and is not re-measured
    assert mapped.limits['a'] == space.limits['a']
    assert 'a' not in mapped.bounds
    # 'bc' is introduced, so its box is exactly the image's bounding box
    column = mapped.samples[:, mapped.params.index('bc')]
    assert mapped.limits['bc'] == pytest.approx((column.min(), column.max()))
    assert mapped.bounds['bc'] == mapped.limits['bc']


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
                     nsigma=3.75).clone(limits=short)
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


def test_bound_cuts_and_widen_does_not_undo_it():
    """`bound` and `widen` are opposite verbs and must stay so: bounding then widening leaves the
    bound recorded, and widening is not a way to escape one. With no covariance underneath, the
    box used to start from the bounds and then be unioned with the extent, which silently made a
    declared bound unenforced."""
    space = Space(bounds={'x': (0., 1.), 'y': (0., 3.)})
    assert space.limits['x'] == (0., 1.) and space.bounds['x'] == (0., 1.)
    widened = space.widen(x=(-1., 2.))
    assert widened.limits['x'] == (-1., 2.)      # widen unions
    assert widened.bounds['x'] == (0., 1.)       # and does not touch what was declared hard
    assert widened.bound(x=(0., 1.)).limits['x'] == (0., 1.)   # a bound cuts it back


# ── the derivation verbs, and the invariant they exist to keep ────────────────
#
# `__init__` is the only method that maps a range from the user's parameter into the expansion
# variable. Everything below derives a new Space from an existing one's state, so a range that is
# already expanded is never expanded twice -- `sqrt` twice, or the logit of a logit, which is nan.

def test_uncorrelated_keeps_the_pool_and_drops_the_rotation():
    """The wish `BackgroundEmulator` used to hand-roll in twenty lines: `Omega_b` and `Omega_cdm`
    are both `omega / h^2`, so their correlation is the basis change talking, not the posterior.
    A whitened grid follows that band and then refuses a point moving `omega_cdm` at fixed `h`.
    The samples must survive, because `measure='samples'` draws its candidates from them."""
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(4000, 3))
    raw[:, 1] += 0.9 * raw[:, 0]                      # a strong, deliberate correlation
    space = Space(samples={name: raw[:, index] for index, name in enumerate('abc')})
    assert space.is_correlated()

    plain = space.uncorrelated()
    assert not plain.is_correlated()
    assert plain.samples is not None and plain.samples.shape == space.samples.shape
    assert plain.limits == space.limits                # the box is untouched
    assert 'mean' not in plain.geometry() and 'covariance' not in plain.geometry()
    assert 'mean' in space.geometry()                  # the engine would have whitened before


def test_bound_and_widen_reject_unknown_names():
    space = Space(bounds={'a': (0., 1.)})
    for call in (lambda: space.bound(zz=(0., 1.)),
                 lambda: space.widen(zz=(0., 1.))):
        with pytest.raises(ValueError, match='unknown parameters'):
            call()


def test_bound_to_an_empty_range_is_an_error_not_an_empty_box():
    space = Space(bounds={'a': (0., 1.)})
    with pytest.raises(ValueError, match='empty range'):
        space.bound(a=(2., 3.))


def test_inverse_round_trips_forward():
    """`forward` and `inverse` are the two halves of the one boundary between the user's
    parameters and the expansion variable. Callers used to reach into `TRANSFORMS[spec][1]` by
    hand for want of the second."""
    space = Space(bounds={'m': (0.02, 0.4), 'a': (0., 1.)}, transforms={'m': 'sqrt'})
    point = {'m': 0.09, 'a': 0.5}
    expanded = space.forward(point)
    assert expanded['m'] == pytest.approx(0.3)         # sqrt applied
    assert expanded['a'] == pytest.approx(0.5)         # untransformed passes through
    back = space.inverse(expanded)
    assert back['m'] == pytest.approx(point['m'])
    assert back['a'] == pytest.approx(point['a'])


# ── map ───────────────────────────────────────────────────────────────────────

def test_map_carries_the_source_transform_without_applying_it_twice():
    """A pass-through arrives from the mapping already in the source's expansion variable, since
    the points fed to it are the stored (transformed) samples. Re-applying its transform is the
    'logit of a logit is nan' failure; dropping the declaration is worse still, leaving the engine
    to read a transformed value as a raw one and place nodes accordingly."""
    rng = np.random.default_rng(3)
    space = Space(samples={'m': rng.uniform(0.05, 0.3, size=2000),
                           'x': rng.normal(size=2000)}, transforms={'m': 'sqrt'})
    mapped = space.map(lambda point: {'m': point['m'], 'y': 2. * point['x']})
    assert mapped.transforms['m'] == 'sqrt'                     # carried
    assert mapped.transforms['y'] is None
    # still sqrt-space values, not sqrt(sqrt(...))
    index = mapped.params.index('m')
    assert mapped.samples[:, index].min() == pytest.approx(space.samples[:, space.params.index('m')].min())
    assert np.all(np.isfinite(list(mapped.limits['m'])))


def test_map_refuses_a_transform_on_a_pass_through_parameter():
    """A pass-through keeps the source's limits, which are already expanded; declaring a different
    transform for it would need them re-expressed, and applying one twice is silent."""
    space = Space(bounds={'a': (0., 1.), 'b': (0., 1.)})
    with pytest.raises(ValueError, match='pass-through'):
        space.map(lambda point: {'a': point['a'], 'c': point['b']}, transforms={'a': 'sqrt'})


def test_map_drops_points_outside_a_declared_transform_domain():
    """One map that both renames and derives, with a logit on the derived name.

    A uniform draw from a rectangle in (w0, wa) reaches w0 + wa >= 0, where `logit_w0pwa` is
    undefined. Keeping those puts nan in the limits, which is why the caller used to need two
    separate maps; they are outside the region the transform asserts, so they are dropped exactly
    as `contains` drops a point outside the box."""
    space = Space(bounds={'w0_fld': (-1.2, -0.8), 'wa_fld': (-0.6, 1.5), 'h': (0.6, 0.7)})
    mapped = space.map(
        lambda point: {'h': point['h'], 'w0_fld': point['w0_fld'],
                       'w0pwa': point['w0_fld'] + point['wa_fld']},
        transforms={'w0pwa': 'logit_w0pwa'})
    assert np.all(np.isfinite(list(mapped.limits['w0pwa'])))
    # every surviving point is inside the transform's domain, so the box is too
    assert mapped.inverse({'w0pwa': mapped.limits['w0pwa'][1]})['w0pwa'] < 0.
    assert mapped.limits['h'] == (0.6, 0.7)          # pass-through keeps the declared box
    assert mapped.samples.shape[0] < 100000          # some were dropped


def test_map_raises_when_no_point_is_in_the_transform_domain():
    space = Space(bounds={'w0_fld': (0.5, 1.0), 'wa_fld': (0.5, 1.0)})
    with pytest.raises(ValueError, match='outside the domain'):
        space.map(lambda point: {'w0pwa': point['w0_fld'] + point['wa_fld']},
                  transforms={'w0pwa': 'logit_w0pwa'})


def test_map_does_not_reinflate_a_pass_through_axis():
    """The bug `map_space` existed to patch: re-measuring a bounds-defined axis as
    `mean +- nsigma sigma` of a uniform image widens it by 3/sqrt(12) ~ 1.7x, on axes the mapping
    never touched. Measured in production, a `wa_fld` box of +-0.9 came back +-1.56 and training
    died on a node at w0 + wa = 0.56."""
    space = Space(bounds={'a': (-0.9, 0.9), 'b': (0.5, 1.5)})
    mapped = space.map(lambda point: {'a': point['a'], 'ab': point['a'] * point['b']})
    assert mapped.limits['a'] == (-0.9, 0.9)
