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
    return Space(bounds={'amplitude': (1., 3.), 'tilt': (-1., 1.)})


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


def test_predict_in_box_clips_and_reports_the_distance():
    """Outside the box, the prediction at the clipped point and the distance in box widths; inside,
    the plain prediction and zero -- under jit too."""
    import jax
    emu = trained(budget=2)
    assert [constraint.name for constraint in emu.constraints()] == ['box']   # nodes fill the box: no node-cloud constraint
    out, violations = emu.predict_in_box(**POINT)
    assert all(float(value) == 0. for value in violations.values())
    assert np.allclose(out['pk'], emu.predict(**POINT)['pk'], rtol=1e-12)
    # amplitude 0.5 above (1, 3): a quarter of its width; tilt 0.5 below (-1, 1): a quarter too
    out, violations = emu.predict_in_box(amplitude=3.5, tilt=-1.5)
    assert np.allclose(float(violations['box']), 0.5)
    assert np.allclose(out['pk'], emu.predict(amplitude=3., tilt=-1.)['pk'], rtol=1e-12)
    jitted = jax.jit(lambda amplitude: emu.predict_in_box(amplitude=amplitude, tilt=0.3))
    out, violations = jitted(3.5)
    assert np.allclose(float(violations['box']), 0.25)
    assert np.allclose(out['pk'], emu.predict(amplitude=3., tilt=0.3)['pk'], rtol=1e-10)


def test_linear_constraints_parse_from_text():
    from cosmoprimo.emulators.tools import LinearConstraint
    constraint = LinearConstraint.from_string('w0 + wa < -0.5', name='w0_plus_wa', aliases={'w0': 'w0_fld', 'wa': 'wa_fld'})
    assert constraint.coefficients == {'w0_fld': 1., 'wa_fld': 1.} and constraint.upper == -0.5 and constraint.lower is None
    constraint = LinearConstraint.from_string('2 omega_b - omega_cdm >= 0')
    assert constraint.coefficients == {'omega_b': 2., 'omega_cdm': -1.} and constraint.lower == 0.
    constraint = LinearConstraint.from_string('-1 < w0 <= 0', name='w0_range')
    assert (constraint.lower, constraint.upper) == (-1., 0.)
    constraint = LinearConstraint.from_string('0.5 > tilt')
    assert constraint.upper == 0.5
    with pytest.raises(ValueError):
        LinearConstraint.from_string('amplitude * tilt < 1')


def test_a_linear_constraint_is_enforced_like_the_box():
    """A declarative constraint joins the built-in ones: raised in eager calls, NaN when traced,
    reported (not clipped) by predict_in_box."""
    import jax
    emu = Emulator(target, space(), constraints=['amplitude + tilt < 3'])
    emu.train(budget=2)
    assert [constraint.name for constraint in emu.constraints()][-1] == 'amplitude_tilt_upper'
    inside, outside = {'amplitude': 2., 'tilt': 0.5}, {'amplitude': 2.8, 'tilt': 0.5}     # 3.3 > 3, inside the box
    assert np.isfinite(emu.predict(**inside)['pk']).all()
    with pytest.raises(CoverageError, match='amplitude_tilt_upper'):
        emu.predict(**outside)
    assert np.isnan(jax.jit(lambda amplitude: emu.predict(amplitude=amplitude, tilt=0.5)['pk'])(2.8)).all()
    out, violations = emu.predict_in_box(**outside)
    assert np.allclose(float(violations['amplitude_tilt_upper']), 0.3) and float(violations['box']) == 0.
    emu.violation = 'ignore'
    assert np.allclose(out['pk'], emu.predict(**outside)['pk'], rtol=1e-12)   # inside the box: not moved, only reported
    assert np.allclose(emu.violations(**outside)['amplitude_tilt_upper'], 0.3)
    emu.violation = 'clip'
    assert np.isfinite(emu.predict(**outside)['pk']).all()


def test_the_node_cloud_has_a_distance_and_a_clip():
    """Whitened along a strong correlation, the box's corners are off the node cloud: a distance
    there, and clipping moves the point back onto it."""
    covariance = np.array([[0.04, -0.0285], [-0.0285, 0.0225]])    # correlation -0.95
    whitened = Space(mean=[2., 0.], covariance=covariance, params=['amplitude', 'tilt'])
    emu = Emulator(target, whitened)
    emu.train(budget=2)
    low_a, high_a = emu.training.limits['amplitude']
    low_t, high_t = emu.training.limits['tilt']
    corner = {'amplitude': high_a * 0.999 + low_a * 0.001, 'tilt': high_t * 0.999 + low_t * 0.001}   # same-sign corner
    assert [constraint.name for constraint in emu.constraints()] == ['box', 'nodes']
    violations = emu.violations(**corner)
    assert float(violations['box']) == 0. and float(violations['nodes']) > 0.
    out, reported = emu.predict_in_box(**corner)
    assert np.isfinite(out['pk']).all() and float(reported['nodes']) > 0.
    with pytest.raises(CoverageError, match='off the node cloud'):
        emu.predict(**corner)


def test_constraints_and_violation_round_trip(tmp_path):
    from cosmoprimo.emulators.tools import LinearConstraint
    emu = Emulator(target, space(), constraints=[LinearConstraint('sum', {'amplitude': 1., 'tilt': 1.}, upper=3.)],
                   violation='nan')
    emu.train(budget=2)
    state = emu.__getstate__()
    again = Emulator.__new__(Emulator)
    again.__setstate__(state)
    assert again.violation == 'nan' and again._constraints == emu._constraints
    # a file written with `coverage`, before `violation` and `constraints` existed
    state = dict(state)
    for key in ('violation', 'constraints'):
        state.pop(key)
    state['coverage'] = 'warn'
    old = Emulator.__new__(Emulator)
    old.__setstate__(state)
    assert old.violation == 'warn' and old._constraints == []


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


def test_mlp_valid_predicate_keeps_every_node_out_of_the_hole():
    """A predicate on the physical parameters filters the Sobol pool before any evaluation: the
    node set is a sample of the valid region, still `nsamples` strong, and an engine built
    without a predicate is untouched."""
    from cosmoprimo.emulators.tools.mlp import MLPEngine

    box = space()
    plain = MLPEngine(box.params, box.limits, nsamples=128).nodes()
    # vectorised predicate: half the box (tilt > 0), off the pool's own low-discrepancy order
    nodes = MLPEngine(box.params, box.limits, nsamples=128,
                      valid=lambda amplitude, tilt: tilt > 0.).nodes()
    assert nodes.shape == (128, 2) and np.all(nodes[:, 1] > 0.)
    # the survivors are the pool's first valid rows, so the first nodes of the two sets coincide
    first_valid = plain[plain[:, 1] > 0.]
    assert np.allclose(nodes[:len(first_valid)], first_valid)
    # a scalar predicate works too (row loop fallback)
    scalar = MLPEngine(box.params, box.limits, nsamples=64,
                       valid=lambda amplitude, tilt: bool(tilt > 0.)).nodes()
    assert scalar.shape == (64, 2) and np.all(scalar[:, 1] > 0.)
    # a predicate keeping a sliver of the box: the pool doubles until enough survive ...
    sliver = MLPEngine(box.params, box.limits, nsamples=64, candidates=64,
                       valid=lambda amplitude, tilt: tilt > 0.95).nodes()
    assert sliver.shape == (64, 2) and np.all(sliver[:, 1] > 0.95)
    # ... up to the cap, where it is loud rather than a silently smaller set
    engine = MLPEngine(box.params, box.limits, nsamples=128, candidates=128,
                       valid=lambda amplitude, tilt: tilt > 0.999)
    engine.MAX_CANDIDATES = 1024
    with pytest.raises(ValueError, match='cap'):
        engine.nodes()


def test_mlp_trains_through_a_valid_predicate():
    """End to end: `valid` reaches the engine through the emulator's options, the target is never
    called in the hole, and the fit is still usable on the valid side."""
    calls = []

    def guarded(params):
        calls.append(params['tilt'])
        assert params['tilt'] > 0., 'the target was called inside the hole'
        return target(params)

    emu = Emulator(guarded, space())
    emu.train(**{**MLP, 'valid': lambda amplitude, tilt: tilt > 0.})
    assert len(calls) == MLP['nsamples'] and min(calls) > 0.
    assert np.max(np.abs(emu.predict(**POINT)['pk'] / target(POINT)['pk'] - 1.)) < 0.1


def test_mlp_lr_decay_round_trips_and_fits():
    """A decaying rate is part of the fit's recipe, so it travels with the state. (Whether it
    helps depends on the budget: over the suite's 100 epochs a 100-fold decay ends too soon to
    improve on the constant rate, so only usability is asserted here.)"""
    from cosmoprimo.emulators.tools.mlp import MLPEngine

    box = space()
    with pytest.raises(ValueError, match='lr_decay'):
        MLPEngine(box.params, box.limits, lr_decay=0.)
    emu = trained(**{**MLP, 'lr_decay': 1e-2})
    assert emu._engines['pk'][0].lr_decay == 1e-2
    state = emu._engines['pk'][0].__getstate__()
    assert MLPEngine.from_state(state).lr_decay == 1e-2
    assert np.max(np.abs(emu.predict(**POINT)['pk'] / target(POINT)['pk'] - 1.)) < 0.3


def test_train_can_stop_after_the_evaluations(tmp_path):
    """`fit=False`: every node evaluated and checkpointed, no engine fitted -- the CPU half of a
    build whose network fit is then a second job on a GPU. That second call, with the same
    checkpoint, evaluates nothing and fits."""
    calls = []

    def counting(params):
        calls.append(1)
        return target(params)

    checkpoint = str(tmp_path / 'nodes.ckpt.npz')
    emu = Emulator(counting, space())
    emu.train(**{**MLP, 'checkpoint': checkpoint, 'fit': False})
    assert not emu.trained and len(calls) == MLP['nsamples']
    assert np.load(checkpoint)['nodes'].shape == (MLP['nsamples'], 2)
    emu.train(**{**MLP, 'checkpoint': checkpoint})
    assert emu.trained and len(calls) == MLP['nsamples']


class Sampled(Exact):
    """`amplitude` exact as in `Exact`, but sampled at the nodes rather than pinned at the centre."""
    def select_node_params(self, names):
        return ['amplitude']


def test_a_sampled_exact_param_reuses_the_generic_checkpoint(tmp_path):
    """`select_node_params`: a node set drawn while `amplitude` was still expanded -- and its
    checkpoint -- is reused as it stands once `amplitude` is exact-but-sampled. Same coordinates
    in the same column order, so nothing is evaluated again; the fit is over `tilt` alone, on the
    nodes projected onto it; and `amplitude` is exact and unbounded as with the pinned version."""
    calls = []

    def counting(params):
        calls.append(1)
        return target(params)

    checkpoint = str(tmp_path / 'nodes.ckpt.npz')
    generic = Emulator(counting, space(), engine='mlp')
    generic.train(**{**MLP, 'checkpoint': checkpoint, 'fit': False})
    evaluated = len(calls)
    emu = Sampled(counting, space(), engine='mlp')
    # expanded: tilt; exact: amplitude; node coordinates: both, in the TRAINING order, which is
    # the column order the generic checkpoint was written in
    assert emu.params == ['tilt'] and emu.exact_params == ['amplitude']
    assert emu.node_params == ['amplitude', 'tilt']
    draw = {name: value for name, value in MLP.items() if name != 'engine'}
    assert np.allclose(emu.nodes(**draw), generic.nodes(**draw))
    emu.train(**{**MLP, 'checkpoint': checkpoint})
    assert emu.trained and len(calls) == evaluated
    # the fitted engines are over `tilt` alone, and amplitude is exact far outside its range
    assert all(engine.params == ['tilt'] for engine, *_ in emu._engines.values())
    outside = emu.predict(amplitude=99., tilt=0.2)
    assert np.allclose(outside['pk'], 99. / 2.9 * emu.predict(**OTHER)['pk'], rtol=1e-10)
    # the projected nodes are a perfectly good 1-d set: the fit is as good as the pinned one's
    pinned = trained(cls=Exact, **MLP)
    truth = target(OTHER)['pk']
    error = lambda emulator: np.max(np.abs(emulator.predict(**OTHER)['pk'] / truth - 1.))
    assert error(emu) < 5. * max(error(pinned), 1e-3)
    # and it survives a file
    emu.write(str(tmp_path / 'sampled.h5'))
    again = Emulator.read(str(tmp_path / 'sampled.h5'))
    assert again.node_params == ['amplitude', 'tilt']
    assert np.allclose(again.predict(**OTHER)['pk'], emu.predict(**OTHER)['pk'])


def test_a_sampled_param_must_be_a_training_parameter():
    class Wrong(Exact):
        def select_node_params(self, names):
            return ['nothing']

    with pytest.raises(ValueError, match='select_node_params'):
        Wrong(target, space())


def test_outlier_cuts_can_look_at_raw_or_transformed_values(tmp_path, caplog):
    """`outlier_on`: the cuts see what the engines are fitted to by default; 'raw' sees the
    calculator's outputs; 'both' requires a node to pass both. A transform that removes the
    amplitude also removes the amplitude-driven outliers, so the two node sets differ."""
    import logging

    def spiky(params):
        # one output that is exactly exp(amplitude) times a shape: e^10 above the median at the
        # top of the box, e^10 below it at the bottom
        return {'pk': np.exp(params['amplitude']) * K**(-1.5 + 0.2 * params['tilt'])}

    class Scaled(Sampled):
        def transform(self, values, params):
            return {name: value / np.exp(params['amplitude']) for name, value in values.items()}

        def inverse_transform(self, values, params):
            return {name: value * np.exp(params['amplitude']) for name, value in values.items()}

    wide = Space(bounds={'amplitude': (0., 20.), 'tilt': (-1., 1.)})

    def dropped(cls, outlier_on):
        caplog.clear()
        emu = cls(spiky, wide, engine='mlp')
        with caplog.at_level(logging.INFO, logger='Emulator'):
            emu.train(**{**MLP, 'outlier_factor': 10., 'outlier_on': outlier_on})
        lines = [record.getMessage() for record in caplog.records if 'nodes (' in record.getMessage()]
        return int(lines[-1].split()[1].split('/')[0]) if lines else 0

    # generic emulator, transform = identity: raw and transformed are the same values
    assert dropped(Emulator, 'raw') == dropped(Emulator, 'transformed') > 0
    # amplitude divided out: nothing is an outlier in the transformed values, while the raw cut
    # still removes the large-amplitude nodes, and 'both' removes at least as many as either
    assert dropped(Scaled, 'transformed') == 0
    assert dropped(Scaled, 'raw') == dropped(Emulator, 'raw') > 0
    assert dropped(Scaled, 'both') >= dropped(Scaled, 'raw')
    with pytest.raises(ValueError, match='outlier_on'):
        Scaled(spiky, wide, engine='mlp').train(**{**MLP, 'outlier_factor': 10., 'outlier_on': 'sideways'})


def test_mlp_huber_loss_fits_and_round_trips(tmp_path):
    """`loss='huber'`: fits about as well as the mean square on clean data, and its options
    survive a file. `fit_seed` reseeds the fit alone: same nodes, a different network."""
    emu = trained(**{**MLP, 'loss': 'huber', 'huber_delta': 0.5, 'fit_seed': 7})
    truth = target(OTHER)['pk']
    plain = trained(**MLP)
    error = lambda emulator: np.max(np.abs(emulator.predict(**OTHER)['pk'] / truth - 1.))
    assert error(emu) < 3. * max(error(plain), 0.03)
    engine = next(iter(emu._engines.values()))[0]
    assert engine.loss == 'huber' and engine.huber_delta == 0.5 and engine.fit_seed == 7
    emu.write(str(tmp_path / 'huber.h5'))
    again = Emulator.read(str(tmp_path / 'huber.h5'))
    engine = next(iter(again._engines.values()))[0]
    assert engine.loss == 'huber' and engine.huber_delta == 0.5 and engine.fit_seed == 7
    assert np.allclose(again.predict(**OTHER)['pk'], emu.predict(**OTHER)['pk'])
    # the same nodes, another seed: a different network, the same function to within its error
    other = trained(**{**MLP, 'fit_seed': 8})
    assert np.allclose(other.nodes(**{k: v for k, v in MLP.items() if k != 'engine'}),
                       emu.nodes(**{k: v for k, v in MLP.items() if k != 'engine'}))
    with pytest.raises(ValueError, match='loss'):
        trained(**{**MLP, 'loss': 'l1'})


def test_train_can_augment_the_node_set(tmp_path):
    """`augment`: extra nodes of the engine's own kind over a sub-box, through the same `valid`
    predicate, appended AFTER the base draw -- so a checkpoint of the un-augmented training is
    a prefix of the augmented one and resumes with the extra nodes alone."""
    calls = []

    def counting(params):
        calls.append(params['tilt'])
        return target(params)

    def valid(amplitude, tilt):
        return tilt > 0.

    checkpoint = str(tmp_path / 'nodes.ckpt.npz')
    emu = Emulator(counting, space())
    emu.train(**{**MLP, 'checkpoint': checkpoint, 'fit': False, 'valid': valid})
    base = len(calls)
    assert base == MLP['nsamples']
    augment = {'bounds': {'tilt': (0.5, 0.7)}, 'nsamples': 32}
    emu.train(**{**MLP, 'checkpoint': checkpoint, 'valid': valid, 'augment': augment})
    # only the extra nodes were evaluated, every one inside the sub-box (and the gate)
    assert len(calls) == base + 32
    assert all(0.5 <= tilt <= 0.7 for tilt in calls[base:])
    stored = np.load(checkpoint)['nodes']
    assert stored.shape == (base + 32, 2)
    assert np.all((stored[base:, 1] >= 0.5) & (stored[base:, 1] <= 0.7))
    assert emu.trained
    # a list of specifications draws each with its own seed; the sub-box draw is not the base
    # draw rescaled
    two = emu._augmented_nodes([augment, {'bounds': {'amplitude': (2.5, 3.)}, 'nsamples': 8}],
                               valid=valid, nsamples=MLP['nsamples'])
    assert two.shape == (40, 2) and np.all(two[32:, 0] >= 2.5) and np.all(two[:, 1] > 0.)
    assert not np.allclose(two[:32, 0], stored[:32, 0])
    # loud on a name outside the box, on bounds that miss it, and on a whitened space
    with pytest.raises(ValueError, match='not among'):
        emu._augmented_nodes({'bounds': {'slope': (0., 1.)}, 'nsamples': 4})
    with pytest.raises(ValueError, match='overlap'):
        emu._augmented_nodes({'bounds': {'tilt': (5., 6.)}, 'nsamples': 4})
    rng = np.random.default_rng(0)
    chain = {'amplitude': rng.normal(2., 0.2, 500), 'tilt': rng.normal(0.3, 0.1, 500)}
    chain['tilt'] += 0.5 * (chain['amplitude'] - 2.)
    whitened = Emulator(target, Space(samples=chain))
    with pytest.raises(ValueError, match='whitened'):
        whitened._augmented_nodes({'bounds': {'tilt': (0.2, 0.4)}, 'nsamples': 4})


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
