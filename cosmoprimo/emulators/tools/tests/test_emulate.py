"""The four steps: build, inspect, train, predict."""

import numpy as np
import pytest

from cosmoprimo.emulators.tools import Emulator, Space, NotTrained, CoverageError

K = np.linspace(0.01, 0.3, 30)

#: Points inside the box. Hoisted because a point that drifts between tests is a coverage
#: failure waiting to happen.
POINT = {'amplitude': 2.2, 'tilt': 0.3}
OTHER = {'amplitude': 2.9, 'tilt': 0.2}

#: Enough training for the network to be worth asking, and no more: these run in the suite.
MLP = dict(engine='mlp', nsamples=256, epochs=100, patience=30)


def target(params):
    """A plain callable, the whole contract: params in, named arrays out."""
    return {'pk': params['amplitude'] * K**(-1.5 + 0.2 * params['tilt'])}


class Exact(Emulator):
    """`pk` is exactly linear in `amplitude`, and the subclass says so.

    Three overrides, each independent: what to expand, what to divide out, how to put it back.
    Nothing is asked of the target, which stays the same plain function.
    """
    def select_params(self, names):
        return [name for name in names if name != 'amplitude']

    def transform(self, values, params):
        return {name: value / params['amplitude'] for name, value in values.items()}

    def inverse_transform(self, values, params):
        return {name: value * params['amplitude'] for name, value in values.items()}


def space():
    return Space(limits={'amplitude': (1., 3.), 'tilt': (-1., 1.)})


def trained(cls=Emulator, **options):
    """A fitted emulator; ``engine='mlp'`` in *options* switches the engine."""
    emulator = cls(target, space())
    emulator.train(**{'budget': 3, **options})
    return emulator


# ── build, size, train ────────────────────────────────────────────────────────

def test_emulate_returns_an_untrained_emulator():
    """Training is hours of Boltzmann calls, so it is a deliberate separate step."""
    emu = Emulator(target, space())
    assert not emu.trained
    assert emu.params == ['amplitude', 'tilt']
    with pytest.raises(NotTrained):
        emu.predict(amplitude=2., tilt=0.)


def test_train_then_predict():
    emu = trained()
    assert emu.trained
    assert np.allclose(emu.predict(**POINT)['pk'], target(POINT)['pk'], rtol=1e-3)


def test_nodes_can_be_sized_before_paying_for_them():
    emu = Emulator(target, space())
    assert len(emu.nodes(budget=2)) < len(emu.nodes(budget=3))


def test_budget_may_be_given_at_construction_or_at_training():
    """`Emulator(..., budget=2)` keeps it in the engine options and `train` passes its own; both
    reaching the engine is a TypeError, which is how the FOLPSD example first failed."""
    emu = Emulator(target, space(), budget=2)
    assert len(emu.nodes()) == len(Emulator(target, space()).nodes(budget=2))
    emu.train()                      # construction-time budget alone
    assert emu.trained
    # an explicit budget at training wins over the constructor's
    other = Emulator(target, space(), budget=0)
    assert len(other.nodes(budget=3)) > len(other.nodes())


def test_the_target_is_only_ever_a_callable():
    """No protocol on the target: a bare lambda with no attributes at all must work."""
    emu = Emulator(lambda params: {'pk': np.full(3, params['amplitude'])}, space())
    emu.train(budget=2)
    assert np.allclose(emu.predict(amplitude=2.2, tilt=0.)['pk'], 2.2, rtol=1e-8)
    with pytest.raises(TypeError, match='callable'):
        Emulator(object(), space())


def test_outside_the_box_raises():
    with pytest.raises(CoverageError, match='outside the trained box'):
        trained(budget=2).predict(amplitude=99., tilt=0.)


def test_validate_defaults_to_the_target_itself():
    report = trained().validate(
        npoints=10, metric=lambda p, t: float(np.max(np.abs(p['pk'] / t['pk'] - 1.))))
    assert report.sigma < 1e-2 and report.coverage_failures == 0


def test_an_unknown_engine_names_the_ones_that_exist():
    with pytest.raises(ValueError, match='mlp'):
        Emulator(target, space()).train(engine='nonesuch')


# ── the hooks ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('options', [dict(budget=3), MLP], ids=['chebyshev', 'mlp'])
def test_exact_params_leave_the_grid_and_stay_exact(options):
    """The engine is orthogonal to the subclass: what `select_params` takes off the grid is
    handled by `transform` either way, so switching engines is a one-word change."""
    emu = trained(cls=Exact, **options)
    assert emu.params == ['tilt'] and emu.exact_params == ['amplitude']
    # exact, therefore unbounded: far outside the trained range and still exact
    outside = emu.predict(amplitude=99., tilt=0.2)
    assert np.allclose(outside['pk'], 99. / 2.9 * emu.predict(**OTHER)['pk'], rtol=1e-10)


def test_exact_params_cost_no_nodes():
    assert len(Exact(target, space()).nodes(budget=3)) \
        < len(Emulator(target, space()).nodes(budget=3))


def test_transform_is_applied_after_collection_not_before():
    """The checkpoint must hold physical values, so changing what is divided out costs a refit,
    not another run of the expensive calculator."""
    seen = []

    class Recording(Exact):
        def transform(self, values, params):
            seen.append(dict(params))
            return super().transform(values, params)

    emu = trained(cls=Recording, budget=2)
    # every node was transformed with the amplitude held at the space centre, after the fact
    assert seen and all(np.isclose(params['amplitude'], 2.) for params in seen)
    assert len(seen) == len(emu.nodes(budget=2))


# ── the mlp engine ────────────────────────────────────────────────────────────

def test_mlp_trains_and_predicts():
    """Not exact the way the grid is -- a network is a stochastic fit -- so the assertion is
    percent-level, which is what this engine is for: many parameters, approximate answer."""
    emu = trained(**{**MLP, 'nsamples': 512, 'epochs': 300, 'patience': 60})
    assert np.max(np.abs(emu.predict(**POINT)['pk'] / target(POINT)['pk'] - 1.)) < 0.05


def test_mlp_nodes_are_quasi_random_samples_of_the_box():
    """A pile of Sobol samples, not a grid -- and every one inside the box, or the calculator is
    being asked for a cosmology the Space never claimed."""
    from cosmoprimo.emulators.tools.mlp import MLPEngine

    box = space()
    nodes = MLPEngine(box.params, box.limits, nsamples=128).nodes()
    assert nodes.shape == (128, 2)
    for index, name in enumerate(box.params):
        low, high = box.limits[name]
        assert nodes[:, index].min() >= low and nodes[:, index].max() <= high
    # ... and they actually cover it, rather than clustering
    assert nodes[:, 0].min() < 1.2 and nodes[:, 0].max() > 2.8


# ── saving ────────────────────────────────────────────────────────────────────

def test_mlp_round_trips_through_a_file(tmp_path):
    emu = trained(**MLP)
    loaded = Emulator.read(emu.write(str(tmp_path / 'mlp.h5')))
    assert np.allclose(loaded.predict(**POINT)['pk'], emu.predict(**POINT)['pk'],
                       rtol=1e-12, atol=0.)


def test_a_state_from_another_version_refuses_to_load(tmp_path):
    """A saved emulator outlives the code that wrote it. Refusing loudly is the whole point: a
    silently misread emulator predicts confidently and is wrong everywhere."""
    from cosmoprimo.emulators.tools.emulate import StateVersionError
    from cosmoprimo.emulators.tools.io import write_state, read_state

    path = trained(budget=2).write(str(tmp_path / 'versioned.h5'))
    state = read_state(path)
    assert state['version'] == Emulator.version and 'cosmoprimo_version' in state

    state['version'] = Emulator.version + 1
    write_state(path, state)
    with pytest.raises(StateVersionError, match='version'):
        Emulator.read(path)


# ── contracting an output ─────────────────────────────────────────────────────

@pytest.mark.parametrize('options, rtol', [(dict(budget=3), 1e-10), (MLP, 1e-8)],
                         ids=['chebyshev', 'mlp'])
def test_contract_is_exact(options, rtol):
    """Folding a fixed matrix into the coefficients is an identity, not an approximation: every
    engine is linear in what it contracts -- the network is not, but everything after its last
    layer is affine, so the matrix folds in there with the output standardisation.

    The motivating case is a window matrix: emulate on the fine theory grid, then reduce to the
    data bins once instead of on every evaluation.
    """
    emu = trained(**options)
    before = np.asarray(emu.predict(**POINT)['pk'])
    matrix = np.random.default_rng(0).normal(size=(4, len(K)))

    emu.contract('pk', matrix)
    after = np.asarray(emu.predict(**POINT)['pk'])
    assert after.shape == (4,)
    assert np.allclose(after, matrix @ before, rtol=rtol, atol=1e-10)


def test_contract_shrinks_the_emulator_rather_than_hiding_the_grid():
    emu = trained()
    emu.contract('pk', np.random.default_rng(0).normal(size=(4, len(K))))
    assert emu._engines['pk'][0].coefficients.shape[1] == 4


def test_contract_checks_the_shape_it_is_given():
    emu = trained(budget=2)
    with pytest.raises(ValueError, match='acts on'):
        emu.contract('pk', np.zeros((3, len(K) + 1)))
    with pytest.raises(ValueError, match='no output'):
        emu.contract('nope', np.zeros((3, len(K))))
