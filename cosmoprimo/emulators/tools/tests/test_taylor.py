"""The Taylor engine: derivatives at a point, not values across a box.

The two engines are checked against each other where they must agree (a polynomial both can
represent exactly) and where they must differ (a Taylor expansion is exact at its centre and
degrades away from it; an interpolant is neither).
"""

import numpy as np
import pytest

from cosmoprimo.emulators.tools import Emulator, Space
from cosmoprimo.emulators.tools.taylor import TaylorEngine


K = np.linspace(0.01, 0.3, 30)

POINT = {'amplitude': 2.2, 'tilt': 0.3}


def target(params):
    return {'pk': params['amplitude'] * K**(-1.5 + 0.2 * params['tilt'])}


def space():
    return Space(bounds={'amplitude': (1., 3.), 'tilt': (-1., 1.)})


def trained(**options):
    emulator = Emulator(target, space(), engine='taylor')
    emulator.train(**options)
    return emulator


# ── the expansion itself ──────────────────────────────────────────────────────

def test_a_polynomial_within_the_order_is_reproduced_exactly():
    """A Taylor expansion of degree n is exact on polynomials of degree <= n, everywhere -- not
    just near the centre. This is the one case where truncation error is zero, so it is the test
    that catches a wrong factorial or a mismatched stencil."""
    def polynomial(params):
        x, y = params['x'], params['y']
        return {'f': np.array([1. + 2. * x - 3. * y + x * y + 0.5 * x**2 - y**3])}

    emulator = Emulator(polynomial, Space(bounds={'x': (-1., 1.), 'y': (-1., 1.)}),
                        engine='taylor')
    emulator.train(order=3)
    for point in [{'x': 0.7, 'y': -0.4}, {'x': -0.9, 'y': 0.9}, {'x': 0., 'y': 0.}]:
        assert np.allclose(emulator.predict(**point)['f'], polynomial(point)['f'],
                           rtol=1e-8, atol=1e-8)


def test_the_expansion_is_exact_at_its_centre_and_degrades_away_from_it():
    """The defining property, and what separates this from the interpolant: error zero at the
    centre, growing with distance. An interpolant is exact at its nodes instead, which are spread
    over the box, so it has no such ordering."""
    emulator = trained(order=2)
    center, edge = {'amplitude': 2., 'tilt': 0.}, {'amplitude': 2.9, 'tilt': 0.9}
    at_center = np.max(np.abs(emulator.predict(**center)['pk'] / target(center)['pk'] - 1.))
    at_edge = np.max(np.abs(emulator.predict(**edge)['pk'] / target(edge)['pk'] - 1.))
    assert at_center < 1e-12
    assert at_edge > 100. * max(at_center, 1e-14)


def test_raising_the_order_reduces_the_error_away_from_the_centre():
    """Accuracy is bought with the order -- the truncation -- and this is what that means.

    Over a small box, which is the regime a local expansion is for: measured over the wide box
    the sequence is not monotone at low order, because a truncated series is only ordered once
    it is in its asymptotic regime, and the tilt dependence at the corner is not.
    """
    small = Space(bounds={'amplitude': (1.8, 2.2), 'tilt': (-0.2, 0.2)})
    edge = {'amplitude': 2.1, 'tilt': 0.18}
    errors = []
    for order in (1, 2, 3):
        emulator = Emulator(target, small, engine='taylor')
        emulator.train(order=order)
        errors.append(np.max(np.abs(emulator.predict(**edge)['pk'] / target(edge)['pk'] - 1.)))
    assert errors[0] > errors[1] > errors[2]


def test_the_derivatives_are_the_real_derivatives():
    """The coefficients are not fitting parameters: the term at multi-index p is the p-th partial
    over prod(p_i!), and a Fisher forecast reads them straight off."""
    def quadratic(params):
        return {'f': np.array([3. * params['x']**2 * params['y']])}

    emulator = Emulator(quadratic, Space(bounds={'x': (-1., 1.), 'y': (-1., 1.)}),
                        engine='taylor')
    emulator.train(order=3)
    engine = emulator._engines['f'][0]
    coefficients = {tuple(power): float(value[0])
                    for power, value in zip(engine.powers, engine.derivatives)}
    assert np.isclose(coefficients[(2, 1)], 3., atol=1e-8)      # d3f/dx2dy / (2! 1!) = 3
    assert np.isclose(coefficients[(0, 0)], 0., atol=1e-8)
    assert np.isclose(coefficients[(1, 0)], 0., atol=1e-8)


# ── the knobs ─────────────────────────────────────────────────────────────────

def test_order_and_accuracy_are_different_knobs():
    """`accuracy` buys a better estimate of each derivative, not more terms, so it changes the
    node count while leaving the number of kept terms alone."""
    box = space()
    coarse = TaylorEngine(box.params, box.limits, order=2, accuracy=2)
    fine = TaylorEngine(box.params, box.limits, order=2, accuracy=4)
    assert len(fine.nodes()) > len(coarse.nodes())
    assert len(fine.powers) == len(coarse.powers)


def test_order_may_be_per_parameter():
    box = space()
    engine = TaylorEngine(box.params, box.limits, order={'amplitude': 3, '*': 1})
    assert engine.order == {'amplitude': 3, 'tilt': 1}
    assert engine.powers[:, 1].max() == 1
    with pytest.raises(ValueError, match='unknown parameters'):
        TaylorEngine(box.params, box.limits, order={'nope': 2, '*': 1})
    with pytest.raises(ValueError, match='missing'):
        TaylorEngine(box.params, box.limits, order={'amplitude': 2})


def test_budget_drops_cross_terms_and_the_nodes_only_they_needed():
    box = space()
    full = TaylorEngine(box.params, box.limits, order=2)
    axes_only = TaylorEngine(box.params, box.limits, order=2, budget=1)
    assert (1, 1) in [tuple(power) for power in full.powers]
    assert (1, 1) not in [tuple(power) for power in axes_only.powers]
    assert len(axes_only.nodes()) < len(full.nodes())


def test_an_odd_accuracy_is_refused():
    box = space()
    with pytest.raises(ValueError, match='EVEN'):
        TaylorEngine(box.params, box.limits, order=2, accuracy=3)


def test_the_widest_stencil_spans_the_box():
    """The step is not a free knob: the stencil reaches the edge of the box, which is the
    convention the previous implementation used."""
    box = space()
    nodes = TaylorEngine(box.params, box.limits, order=3, accuracy=2).nodes()
    for index, name in enumerate(box.params):
        low, high = box.limits[name]
        assert np.isclose(nodes[:, index].min(), low)
        assert np.isclose(nodes[:, index].max(), high)


# ── it is an engine like the others ───────────────────────────────────────────

def test_the_hooks_work_the_same_way():
    """Switching engines is a one-word change: what `select_params` takes off the stencil is
    handled by `transform`, exactly as with the grid."""
    class Exact(Emulator):
        def select_params(self, names):
            return [name for name in names if name != 'amplitude']

        def transform(self, values, params):
            return {name: value / params['amplitude'] for name, value in values.items()}

        def inverse_transform(self, values, params):
            return {name: value * params['amplitude'] for name, value in values.items()}

    emulator = Exact(target, space(), engine='taylor')
    emulator.train(order=3)
    assert emulator.params == ['tilt'] and emulator.exact_params == ['amplitude']
    assert np.allclose(emulator.predict(amplitude=99., tilt=0.)['pk'],
                       target({'amplitude': 99., 'tilt': 0.})['pk'], rtol=1e-8)


def test_whitening_puts_the_stencil_on_the_principal_axes():
    """A correlated space is expanded along its own axes, like every other engine, and the
    parameter names stay physical."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=10000)
    samples = {'a': 0.3 + 0.01 * x, 'b': 0.7 + 0.02 * x + 0.002 * rng.normal(size=10000)}
    emulator = Emulator(lambda params: {'f': np.array([params['a'] * params['b']])},
                        Space(samples=samples), engine='taylor')
    emulator.train(order=2)
    engine = emulator._engines['f'][0]
    assert engine.whitened
    assert engine.params == ['a', 'b']
    point = {'a': 0.305, 'b': 0.71}
    assert np.isclose(float(emulator.predict(**point)['f'][0]), point['a'] * point['b'],
                      rtol=1e-8)


def test_contract_is_exact():
    emulator = trained(order=2)
    before = np.asarray(emulator.predict(**POINT)['pk'])
    matrix = np.random.default_rng(0).normal(size=(4, len(K)))
    emulator.contract('pk', matrix)
    after = np.asarray(emulator.predict(**POINT)['pk'])
    assert after.shape == (4,)
    assert np.allclose(after, matrix @ before, rtol=1e-10, atol=1e-12)


def test_it_round_trips_through_a_file(tmp_path):
    emulator = trained(order=3)
    loaded = Emulator.read(emulator.write(str(tmp_path / 'taylor.h5')))
    assert loaded._engines['pk'][0].name == 'taylor'
    assert np.allclose(loaded.predict(**POINT)['pk'], emulator.predict(**POINT)['pk'],
                       rtol=1e-12, atol=0.)


def test_predict_is_jittable():
    import jax

    emulator = trained(order=2)
    predict = jax.jit(lambda amplitude, tilt: emulator.predict(amplitude=amplitude, tilt=tilt)['pk'])
    assert np.allclose(predict(2.2, 0.3), emulator.predict(**POINT)['pk'], rtol=1e-10)
    # and differentiable, which is the point of keeping the centre out of a bare 0**0
    grad = jax.grad(lambda amplitude: emulator.predict(amplitude=amplitude, tilt=0.)['pk'].sum())
    assert np.isfinite(float(grad(2.0)))
