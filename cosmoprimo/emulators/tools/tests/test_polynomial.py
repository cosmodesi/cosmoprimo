"""The polynomial regression engine: a declared basis, chosen points, an overdetermined solve.

Two things separate it from the sparse grid, and both are tested here rather than assumed. It
gives up exactness at the nodes -- so the only exactness left is on a target the basis can
represent, and that is what pins the basis, the scaling and the solve together. And it gains
tolerance to a node set with holes in it -- so a box containing a region the calculator refuses
is trained over here and fails there, which is the reason the engine exists.
"""

import numpy as np
import pytest

from cosmoprimo.emulators.tools import Emulator, Space
from cosmoprimo.emulators.tools.polynomial import (PolynomialEngine, fekete_selection,
                                                   FEKETE_BELOW)
from cosmoprimo.emulators.tools.utils import multi_index_set, tensor_basis


K = np.linspace(0.01, 0.3, 20)

POINT = {'amplitude': 2.2, 'tilt': 0.3}


def target(params):
    return {'pk': params['amplitude'] * K ** (-1.5 + 0.2 * params['tilt'])}


def space():
    return Space(bounds={'amplitude': (1., 3.), 'tilt': (-1., 1.)})


def trained(**options):
    emulator = Emulator(target, space(), engine='polynomial')
    emulator.train(**options)
    return emulator


# ── the index set ─────────────────────────────────────────────────────────────

def test_the_hyperbolic_cross_is_what_makes_the_basis_payable():
    """The claim the engine is sold on, at the size it is sold at: 6 parameters, degree 3. The
    tensor product is unaffordable, total degree is borderline, the cross is tens of terms."""
    assert len(multi_index_set([3] * 6, budget=3, interaction='tensor')) == 4 ** 6
    assert len(multi_index_set([3] * 6, budget=3, interaction='total')) == 84
    assert len(multi_index_set([3] * 6, budget=3, interaction='hyperbolic')) == 34


def test_the_pruned_enumeration_agrees_with_filtering_the_whole_product():
    """The enumeration prunes on a partial cost rather than filtering a full tensor product, which
    is only valid because the cost is monotone in every entry. Checked against the definition on a
    case small enough to build both ways -- get this wrong and terms go missing silently."""
    import itertools
    orders = [3, 2, 4]
    for interaction, cost in [('total', sum),
                              ('hyperbolic', lambda a: np.prod([v + 1 for v in a]) - 1)]:
        for budget in range(6):
            brute = sorted(alpha for alpha in itertools.product(*[range(o + 1) for o in orders])
                           if cost(alpha) <= budget)
            got = multi_index_set(orders, budget=budget, interaction=interaction)
            assert [tuple(row) for row in got] == brute


def test_an_anisotropic_order_drops_the_terms_that_axis_alone_needed():
    """The cheapest knob: an axis the output barely depends on is given a low degree and every
    higher term along it disappears."""
    powers = multi_index_set([3, 1], interaction='tensor')
    assert powers[:, 1].max() == 1 and powers[:, 0].max() == 3


# ── what the fit is exact on ──────────────────────────────────────────────────

def test_a_polynomial_inside_the_basis_is_reproduced_exactly():
    """A least-squares fit is exact nowhere in general, with one exception: when the target lies
    in the span of the basis, the residual is zero and the fit recovers it everywhere. That is the
    single test that catches a mis-scaled domain, a wrong basis or a botched solve, since every
    one of those breaks it and nothing else would show them apart from a slightly worse error."""
    def polynomial(params):
        x, y = params['x'], params['y']
        return {'f': np.array([1. + 2. * x - 3. * y + x * y + 0.5 * x**2 - y**3])}

    emulator = Emulator(polynomial, Space(bounds={'x': (-1., 1.), 'y': (-1., 1.)}),
                        engine='polynomial')
    emulator.train(order=3, interaction='total')
    for point in [{'x': 0.7, 'y': -0.4}, {'x': -0.9, 'y': 0.9}, {'x': 0., 'y': 0.}]:
        assert np.allclose(emulator.predict(**point)['f'], polynomial(point)['f'],
                           rtol=1e-9, atol=1e-9)


def test_the_error_is_spread_over_the_box_rather_than_anchored_at_a_point():
    """What separates this from the Taylor engine at a comparable cost. A Taylor expansion is
    exact at its centre and worst at the corners; a fit over the whole box has no such ordering,
    and its centre carries the same error as anywhere else."""
    emulator = trained(order=4)
    def error(point):
        return np.max(np.abs(emulator.predict(**point)['pk'] / target(point)['pk'] - 1.))
    at_center = error({'amplitude': 2., 'tilt': 0.})
    at_edge = error({'amplitude': 2.9, 'tilt': 0.9})
    assert at_center > 1e-8                       # not exact at the centre: it is not an expansion
    assert at_edge < 30. * at_center              # and not orders of magnitude worse at the edge


def test_raising_the_degree_reduces_the_error():
    """The knob has to do what it says, and on a smooth non-polynomial target the improvement is
    the only evidence the basis is being used rather than merely being large."""
    errors = []
    for order in (2, 4, 6):
        emulator = trained(order=order, interaction='total')
        point = {'amplitude': 2.4, 'tilt': -0.55}
        errors.append(np.max(np.abs(emulator.predict(**point)['pk'] / target(point)['pk'] - 1.)))
    assert errors[0] > errors[1] > errors[2]


# ── the reason the engine exists: holes ───────────────────────────────────────

def hole_target(params):
    """A box with a corner the calculator refuses -- the shape of ``w0 + wa > 0`` in a w0waCDM
    prior, where CLASS has no answer rather than a wrong one."""
    if params['x'] + params['y'] > 0.6:
        return {'f': np.full(3, np.nan)}
    return {'f': np.array([1., 2., 3.]) * np.exp(0.3 * params['x'] - 0.4 * params['y'])}


HOLE_SPACE = dict(bounds={'x': (-1., 1.), 'y': (-1., 1.)})


def test_the_grid_cannot_be_laid_over_a_hole_and_the_regression_can():
    """Stated as one test because it is one claim, and the contrast is the claim. The Smolyak grid
    is unisolvent, so a refused node is not a smaller problem but an unsolvable one; the
    regression loses that node's row and keeps the rest."""
    from cosmoprimo.emulators.tools.training import NodeEvaluationError

    grid = Emulator(hole_target, Space(**HOLE_SPACE), engine='chebyshev')
    with pytest.raises(NodeEvaluationError):
        grid.train(budget=3)

    fit = Emulator(hole_target, Space(**HOLE_SPACE), engine='polynomial')
    fit.train(order=5, interaction='total', oversampling=3., max_non_finite=0.5)
    point = {'x': -0.5, 'y': 0.3}
    assert np.allclose(fit.predict(**point)['f'], hole_target(point)['f'], rtol=1e-5)


def test_a_valid_predicate_spends_no_evaluation_in_the_hole():
    """The point of filtering the candidate pool rather than discovering the refusal afterwards:
    the rejected region costs nothing at all, instead of costing every node that landed in it.
    Ten thousand candidates, and only the survivors are ever handed to the calculator."""
    def valid(x, y):
        return x + y <= 0.6

    engine = PolynomialEngine(['x', 'y'], {'x': (-1., 1.), 'y': (-1., 1.)},
                              order=3, interaction='total', valid=valid)
    nodes = engine.nodes()
    assert len(nodes) > 0
    assert np.all(nodes[:, 0] + nodes[:, 1] <= 0.6)

    # and the fit over them is still determined: the selection found a conditioned subset inside
    # the region rather than merely whatever was left of a lattice.
    outputs = np.array([hole_target(dict(zip(['x', 'y'], row)))['f'] for row in nodes])
    engine.fit(nodes, outputs)
    assert engine.ndropped == 0
    assert engine.residual < 1e-3


def test_dropping_a_share_of_the_samples_costs_only_those_samples():
    """The premise of ``oversampling`` spelled out: with more points than terms, losing some
    leaves the fit determined, and the error degrades smoothly rather than the solve failing."""
    engine = PolynomialEngine(['amplitude', 'tilt'], {'amplitude': (1., 3.), 'tilt': (-1., 1.)},
                              order=4, interaction='total', oversampling=3.)
    nodes = engine.nodes()
    outputs = np.array([target(dict(zip(engine.params, row)))['pk'] for row in nodes])
    keep = np.ones(len(nodes), dtype='?')
    keep[np.random.default_rng(0).choice(len(nodes), len(nodes) // 4, replace=False)] = False
    full = PolynomialEngine(['amplitude', 'tilt'], {'amplitude': (1., 3.), 'tilt': (-1., 1.)},
                            order=4, interaction='total').fit(nodes, outputs)
    partial = engine.fit(nodes[keep], outputs[keep])
    point = np.array([2.2, 0.3])
    reference = target(dict(amplitude=2.2, tilt=0.3))['pk']
    for fitted in (full, partial):
        assert np.max(np.abs(np.asarray(fitted.predict(point)) / reference - 1.)) < 1e-3


def test_an_underdetermined_basis_is_refused_rather_than_guessed():
    """Fewer usable samples than terms leaves directions nothing measured, and a minimum-norm
    solve would fill them in with zeros dressed up as an answer. The message has to name the knob
    that fixes it, because at that point the evaluations have already been paid for."""
    engine = PolynomialEngine(['x', 'y'], {'x': (-1., 1.), 'y': (-1., 1.)},
                              order=4, interaction='tensor')
    nodes = engine.nodes()[:5]
    outputs = np.ones((5, 2))
    with pytest.raises(ValueError, match='underdetermined'):
        engine.fit(nodes, outputs)


def test_ridge_damps_a_direction_the_samples_do_not_span():
    """A degenerate design -- every sample on one line -- leaves the basis rank deficient. Without
    a ridge that is refused; with one it is damped towards zero, which is a choice the caller
    makes rather than one the solve makes for them."""
    limits = {'x': (-1., 1.), 'y': (-1., 1.)}
    nodes = np.stack([np.linspace(-0.9, 0.9, 12)] * 2, axis=1)     # y == x: no independent y term
    outputs = (1. + nodes[:, :1]) * np.array([[1., 2.]])
    with pytest.raises(ValueError, match='unconstrained'):
        PolynomialEngine(['x', 'y'], limits, order=2, interaction='tensor').fit(nodes, outputs)
    fitted = PolynomialEngine(['x', 'y'], limits, order=2, interaction='tensor',
                              ridge=1e-8).fit(nodes, outputs)
    assert np.all(np.isfinite(fitted.coefficients))


# ── point selection ───────────────────────────────────────────────────────────

def test_pivoted_qr_selection_conditions_the_basis_better_than_taking_the_pool_in_order():
    """The reason the pool is scored rather than truncated. Same candidates, same count, same
    basis: choosing by maximum volume is what makes the design matrix invertible in practice, and
    the gap is what a well-placed point is worth."""
    powers = multi_index_set([4, 4], budget=4, interaction='total')
    domains = np.array([[-1., 1.], [-1., 1.]])
    rng = np.random.default_rng(1)
    pool = rng.uniform(-1., 1., size=(2000, 2))
    design = np.asarray(tensor_basis(pool.T, powers, domains)).T
    nsamples = 2 * len(powers)
    chosen = fekete_selection(design, nsamples)
    selected = np.linalg.cond(design[chosen])
    arbitrary = np.linalg.cond(design[:nsamples])
    assert selected < arbitrary


def test_the_selection_returns_distinct_points_beyond_one_round():
    """Oversampling runs the selection in rounds; a bug there quietly returns the same point
    several times, which reads as a well-conditioned fit on far fewer samples than were paid for."""
    powers = multi_index_set([3, 3], budget=3, interaction='total')
    domains = np.array([[-1., 1.], [-1., 1.]])
    pool = np.random.default_rng(2).uniform(-1., 1., size=(500, 2))
    design = np.asarray(tensor_basis(pool.T, powers, domains)).T
    chosen = fekete_selection(design, 3 * len(powers))
    assert len(chosen) == 3 * len(powers) == len(set(chosen.tolist()))


def test_the_selection_switches_at_the_measured_crossover():
    """``auto`` is a rule with a number in it, and the number is what is worth pinning: pivoted QR
    while the fit is near-interpolatory, the pool once it is not. Checked against the two explicit
    settings, so a change to the rule shows up as a changed node set rather than as a slightly
    different error somewhere downstream."""
    def engine(**options):
        return PolynomialEngine(['x', 'y', 'z'], {name: (-1., 1.) for name in 'xyz'},
                                order=3, budget=3, **options)

    scarce, ample = 1.05, 2.
    assert scarce < FEKETE_BELOW <= ample
    for oversampling, expected in [(scarce, 'fekete'), (ample, 'pool')]:
        auto = engine(oversampling=oversampling, selection='auto').nodes()
        explicit = engine(oversampling=oversampling, selection=expected).nodes()
        assert np.allclose(auto, explicit)
    # ...and the two are genuinely different node sets, or the check above proves nothing.
    assert not np.allclose(engine(oversampling=ample, selection='fekete').nodes(),
                           engine(oversampling=ample, selection='pool').nodes())


def test_fekete_selection_conditions_better_than_the_pool_order():
    """Why the knob exists at all. The gain is in the condition number, and it is what recommends
    Fekete where the design is barely determined; the error trade that runs the other way at
    higher oversampling is measured in `fekete_selection`, not here."""
    conditions = {}
    for selection in ('fekete', 'pool'):
        engine = PolynomialEngine(['x', 'y', 'z'], {name: (-1., 1.) for name in 'xyz'},
                                  order=3, budget=3, interaction='total',
                                  oversampling=1.05, selection=selection)
        nodes = engine.nodes()
        internal = np.array([engine._internal(row) for row in nodes])
        powers = multi_index_set([engine.order[name] for name in engine.params],
                                 budget=engine.budget, interaction=engine.interaction)
        box = np.array([engine._domain(name) for name in engine.params])
        design = np.asarray(tensor_basis(internal.T, powers, box)).T
        conditions[selection] = np.linalg.cond(design)
    assert conditions['fekete'] < conditions['pool']


def test_the_node_count_follows_from_the_basis_and_the_oversampling():
    """The whole cost model of the engine, and the number a training is sized on."""
    engine = PolynomialEngine(['x', 'y', 'z'], {name: (-1., 1.) for name in 'xyz'},
                              order=3, budget=3, interaction='hyperbolic', oversampling=2.)
    assert len(engine.nodes()) == 2 * engine.nterms
    assert len(PolynomialEngine(['x', 'y', 'z'], {name: (-1., 1.) for name in 'xyz'},
                                order=3, nsamples=17).nodes()) == 17


# ── where the nodes are drawn from ────────────────────────────────────────────

def test_samples_drawn_nodes_land_on_the_chain_not_across_the_box():
    """The whole point of ``measure='samples'``: in more than about four dimensions almost all of
    a box is corner, and filling it spends the evaluations where the posterior is not. Checked
    geometrically -- the nodes have to carry the chain's own spread, not the box's."""
    rng = np.random.default_rng(0)
    mean = np.array([0.67, 0.12])
    covariance = np.array([[1., 0.9], [0.9, 1.]]) * np.outer([0.01, 0.002], [0.01, 0.002])
    chain = rng.multivariate_normal(mean, covariance, size=4000)
    common = dict(params=['h', 'omega_cdm'], limits={'h': (0.6, 0.74), 'omega_cdm': (0.1, 0.14)},
                  order=3, mean=mean, covariance=covariance, nsigma=3.,
                  shrink_to_limits=False)
    on_chain = PolynomialEngine(**common, measure='samples', samples=chain).nodes()
    on_box = PolynomialEngine(**common, measure='uniform').nodes()
    # the chain-drawn nodes sit inside the chain's own hull; the box-drawn ones spread wider
    assert np.all(on_chain.min(axis=0) >= chain.min(axis=0))
    assert np.all(on_chain.max(axis=0) <= chain.max(axis=0))
    assert np.all(on_box.std(axis=0) > 1.3 * on_chain.std(axis=0))


def test_samples_are_read_in_the_expansion_variable():
    """A sample is transformed but NOT whitened, so reaching the calculator is the inverse
    transform alone. Running one through `_physical` -- which also unwhitens -- silently lands
    somewhere else entirely, and the fit is then garbage rather than merely worse."""
    mean, covariance = np.array([0.5]), np.array([[0.04]])
    chain = np.random.default_rng(1).normal(0.5, 0.2, size=(500, 1))
    engine = PolynomialEngine(['x'], {'x': (0.01, 4.)}, order=2, transform={'x': 'sqrt'},
                              mean=mean, covariance=covariance, nsigma=2.,
                              shrink_to_limits=False, measure='samples', samples=chain)
    nodes = engine.nodes()[:, 0]
    # `sqrt` is the expansion variable, so every physical node is the square of a chain value --
    # the inverse transform and nothing else. Unwhitening as well would put it somewhere with no
    # chain value behind it at all.
    assert np.all(nodes >= 0.)
    squared = chain[:, 0] ** 2
    assert all(np.min(np.abs(squared - value)) < 1e-9 for value in nodes)


def test_the_measure_needs_the_samples_it_names():
    """Asking to draw from a chain without one is a configuration error, and it has to be raised
    at construction: the alternative is a silent fall back to the box, which is the arrangement
    this option exists to avoid."""
    with pytest.raises(ValueError, match='samples'):
        PolynomialEngine(['x'], {'x': (-1., 1.)}, order=2, measure='samples')


def test_an_emulator_hands_its_chain_to_the_engine():
    """`geometry()` deliberately passes plain arrays and not the chain behind them, so the engine
    asks for the samples by name and the Emulator supplies them. Without that wiring
    ``measure='samples'`` would be unusable through the normal entry point."""
    chain = np.random.default_rng(2).multivariate_normal(
        [2., 0.], [[0.09, 0.02], [0.02, 0.09]], size=3000)
    space_ = Space(samples={'amplitude': chain[:, 0], 'tilt': chain[:, 1]})
    emulator = Emulator(target, space_, engine='polynomial')
    nodes = emulator.nodes(order=3, measure='samples')
    assert len(nodes) and np.all(nodes[:, 0] >= chain[:, 0].min())

    emulator.train(order=3, measure='samples')
    point = {'amplitude': 2.0, 'tilt': 0.0}
    assert np.max(np.abs(emulator.predict(**point)['pk'] / target(point)['pk'] - 1.)) < 1e-2


# ── the basis families ────────────────────────────────────────────────────────

def test_legendre_over_a_uniform_measure_is_as_accurate_as_chebyshev_over_an_arcsine_one():
    """The pairing is a conditioning choice, not an accuracy one. Compared against the truth
    rather than against each other, because they are two different fits and agreeing to better
    than their own error would be a coincidence, not the property being claimed."""
    points = np.random.default_rng(4).uniform([1., -1.], [3., 1.], size=(200, 2))
    errors = {}
    for basis in ('chebyshev', 'legendre'):
        emulator = Emulator(target, space(), engine='polynomial')
        emulator.train(order=5, interaction='total', basis=basis)
        errors[basis] = max(
            np.max(np.abs(emulator.predict(amplitude=point[0], tilt=point[1])['pk']
                          / target(dict(amplitude=point[0], tilt=point[1]))['pk'] - 1.))
            for point in points)
    # Over the box rather than at one point: which of two fits happens to be better at a single
    # place is noise, and it is the worst case over the box that either one is chosen for.
    assert max(errors.values()) < 5e-3
    assert 0.1 < errors['legendre'] / errors['chebyshev'] < 10.


# ── plumbing ──────────────────────────────────────────────────────────────────

def test_a_saved_emulator_predicts_what_it_predicted(tmp_path):
    """The `valid` predicate is not serialisable and is deliberately dropped, so this also checks
    that dropping it changes nothing about a prediction -- it only ever shaped the node set."""
    emulator = trained(order=4)
    loaded = Emulator.read(emulator.write(str(tmp_path / 'polynomial.h5')))
    assert np.allclose(loaded.predict(**POINT)['pk'], emulator.predict(**POINT)['pk'],
                       rtol=1e-12, atol=0.)


def test_a_valid_predicate_belongs_on_train_not_on_the_constructor(tmp_path):
    """Constructor options are kept and written into the saved state, and a callable has no HDF5
    representation. Given to `train` it shapes the node set and is then gone, which is all it was
    ever for -- so the emulator writes. The other way round it does not, and this pins the
    difference so nobody has to rediscover it after paying for a training."""
    space_ = Space(bounds={'amplitude': (1., 3.), 'tilt': (-1., 1.)})
    predicate = lambda amplitude, tilt: amplitude + tilt < 3.5     # noqa: E731

    on_train = Emulator(target, space_, engine='polynomial')
    on_train.train(order=3, valid=predicate)
    assert on_train.write(str(tmp_path / 'on_train.h5'))

    on_constructor = Emulator(target, space_, engine='polynomial', valid=predicate)
    on_constructor.train(order=3)
    with pytest.raises(TypeError, match='HDF5'):
        on_constructor.write(str(tmp_path / 'on_constructor.h5'))


def test_the_prediction_jits_and_differentiates():
    """The reason for a polynomial rather than a Gaussian process: evaluation is one contraction
    and the derivative is exact, so a sampler pays essentially nothing per point."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    emulator = trained(order=3)
    predict = jax.jit(lambda amplitude, tilt: emulator.predict(amplitude=amplitude, tilt=tilt)['pk'])
    assert np.allclose(predict(2.2, 0.3), emulator.predict(**POINT)['pk'])
    # Checked against a finite difference of the emulator itself, not of the target: what has to
    # be exact is the derivative *of the fit*. The fit is a polynomial in the amplitude but not a
    # linear one, so the target's own dP/dA is a different quantity and would only be reached to
    # within the fit error -- which would test the fit, not the differentiation.
    step = 1e-5
    reference = (emulator.predict(amplitude=2.2 + step, tilt=0.3)['pk']
                 - emulator.predict(amplitude=2.2 - step, tilt=0.3)['pk']) / (2. * step)
    gradient = jax.jacobian(predict, argnums=0)(2.2, 0.3)
    assert np.allclose(gradient, reference, rtol=1e-6)


def test_contracting_the_output_is_exact():
    """Left-multiplying by a fixed matrix commutes with the fit, so it can be done once on the
    coefficients instead of at every prediction."""
    engine = PolynomialEngine(['amplitude', 'tilt'], {'amplitude': (1., 3.), 'tilt': (-1., 1.)},
                              order=3)
    nodes = engine.nodes()
    outputs = np.array([target(dict(zip(engine.params, row)))['pk'] for row in nodes])
    engine.fit(nodes, outputs)
    point = np.array([2.2, 0.3])
    matrix = np.random.default_rng(3).normal(size=(4, len(K)))
    before = matrix @ np.asarray(engine.predict(point))
    assert np.allclose(engine.contract(matrix).predict(point), before, rtol=1e-10)
