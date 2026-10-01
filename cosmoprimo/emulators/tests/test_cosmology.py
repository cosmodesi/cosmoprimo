"""Tests for the cosmoprimo interface: cosmology in, cosmology out.

Kept cheap on purpose (ellmax_cl = 500, a handful of nodes) so it runs with the rest of the suite;
the accuracy numbers that matter were measured elsewhere, at production ellmax.
"""

import numpy as np
import pytest

from cosmoprimo import Cosmology
from cosmoprimo.emulators import Emulator, emulate, read, Space, CoverageError


def fiducial(**kwargs):
    return Cosmology(engine='camb', lensing=True, ellmax_cl=500, **kwargs)


SMALL = dict(h=(0.66, 0.70), omega_cdm=(0.115, 0.125))
AMPLITUDE = dict(h=(0.66, 0.70), logA=(3.00, 3.10), tau_reio=(0.04, 0.07))
WITH_LOGA = dict(h=(0.66, 0.70), omega_cdm=(0.115, 0.125), logA=(3.0, 3.1))
OMEGAS = dict(Omega_m=(0.30, 0.32), Omega_b=(0.048, 0.050), h=(0.66, 0.69))

#: A point inside SMALL, and one inside OMEGAS. Hoisted because a point that drifts between
#: tests is a coverage failure waiting to happen.
POINT = {'h': 0.673, 'omega_cdm': 0.1201}
NODE = {'h': 0.68, 'omega_cdm': 0.12}
OMEGA_POINT = {'Omega_m': 0.311, 'Omega_b': 0.0492, 'h': 0.673}
LOGA_POINT = {'h': 0.673, 'omega_cdm': 0.1201, 'logA': 3.04}

Z20 = np.linspace(0., 2., 20)
Z5 = np.linspace(0., 2., 5)
KGRID = np.logspace(-3., 0., 20)


def small_space():
    return Space(bounds=SMALL)


def error_vs_truth(guess, point, spectrum='tt', ellmin=30):
    """max |guess / CAMB - 1| above ``ellmin``, the comparison half these tests make."""
    truth = fiducial().clone(**point).get_harmonic().lensed_cl()
    good = truth['ell'] >= ellmin
    return np.max(np.abs(np.asarray(guess)[good] / np.asarray(truth[spectrum])[good] - 1.))


def count_clones(emulator, params):
    """``(n_clones, outputs)`` for one ``compute`` -- the Boltzmann call is the entire cost, so
    several sections must share one."""
    calls, original = [], emulator.cosmo.clone

    def counting(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    emulator.cosmo.clone = counting
    try:
        values = emulator.compute(params)      # compute first: a tuple would count before it ran
    finally:
        emulator.cosmo.clone = original
    return len(calls), values


# ── what leaves the grid, and what is only flattened ──────────────────────────

def test_amplitude_leaves_the_grid_only_when_nothing_is_lensed():
    # lensing is not linear in the amplitude -- the deflection power is itself ~A_s -- so for
    # lensed spectra dividing by A_s flattens the dependence but must not remove the parameter
    lensed = Emulator(fiducial(), Space(bounds=AMPLITUDE), of=('lensed_cl',))
    assert lensed.params == ['h', 'logA', 'tau_reio'] and lensed.exact_params == []
    unlensed = Emulator(fiducial(), Space(bounds=AMPLITUDE), of=('unlensed_cl',))
    assert unlensed.params == ['h', 'tau_reio'] and unlensed.exact_params == ['logA']


def test_optical_depth_screens_the_right_number_of_legs():
    emu = Emulator(fiducial(), Space(bounds=AMPLITUDE), of=('unlensed_cl', 'lens_potential_cl'))
    tau = 0.06
    factors = emu.scaling({'tau_reio': tau, 'A_s': 2e-9})
    # 'tt' has two screened legs, 'tp' one, 'pp' none; getting this wrong is a silent exp(tau)
    assert np.isclose(factors['unlensed_cl.tt'], 2e-9 * np.exp(-2 * tau))
    assert np.isclose(factors['lens_potential_cl.tp'], 2e-9 * np.exp(-tau))
    assert np.isclose(factors['lens_potential_cl.pp'], 2e-9)


def test_the_scaling_cancels_exactly():
    """Whatever `transform` divides out, `inverse_transform` must put back -- to machine
    precision, or the emulator is fitting one thing and reporting another."""
    emu = Emulator(fiducial(), Space(bounds=AMPLITUDE), of=('lensed_cl',))
    params = {'h': 0.68, 'tau_reio': 0.055, 'logA': 3.04}
    truth = emu.compute(params)
    restored = emu.inverse_transform(emu.transform(truth, params), params)
    for name, value in truth.items():
        assert np.allclose(restored[name], value, rtol=1e-12, atol=0.)


def test_amplitude_is_exact_for_unlensed_spectra():
    """The claim that lets A_s leave the grid: at fixed everything else, unlensed Cl are linear in
    the amplitude, so dividing by it gives back the same array.

    It holds to 2.0e-4 peak, not to machine precision, and that is CAMB's own accuracy floor
    rather than physics: the same 2e-4 appears whether the amplitude is given as A_s or logA, and
    is unchanged by `lensing` or `non_linear`. The tolerance is therefore stated at the measured
    value -- tightening it would only make the test fail for a reason it cannot fix."""
    emu = Emulator(fiducial(), Space(bounds=AMPLITUDE), of=('unlensed_cl',))

    def scaled(logA):
        params = {'logA': logA, 'tau_reio': 0.055}
        return emu.transform(emu.compute(params), params)['unlensed_cl.tt'][2:]

    assert np.max(np.abs(scaled(3.10) / scaled(3.00) - 1.)) < 5e-4


# ── the harmonic emulator, end to end ─────────────────────────────────────────

@pytest.fixture(scope='module')
def trained():
    return Emulator(fiducial(), small_space(), section='harmonic').train(budget=1)


def test_end_to_end_prediction(trained):
    validation = trained.validate(npoints=5, seed=3)
    assert validation.coverage_failures == 0
    assert validation.median < 1e-3


def test_to_cosmology_agrees_with_predict(trained):
    table = trained.to_cosmology().clone(**POINT).get_harmonic().lensed_cl()
    # the engine must query the emulator with its own cosmology's parameters, not the fiducial's
    assert np.allclose(table['tt'], trained.predict(**POINT)['lensed_cl.tt'], rtol=1e-10, atol=0.)
    assert table['ell'][-1] == 500
    assert error_vs_truth(table['tt'], POINT) < 2e-3


def test_a_training_that_evaluates_no_node_still_knows_its_multipoles(tmp_path):
    """A resumed training evaluates nothing, and the harmonic section learns its l grid by
    evaluating. It used to write `ell: None` and fail at deployment, after the nodes were paid for.

    The same happens to any rank that was given no node, which is how it turned up: an MPI training
    whose writer had nothing to evaluate. The stored spectra carry the grid either way.
    """
    emu = Emulator(fiducial(), small_space(), section='harmonic', of=('lensed_cl',))
    checkpoint = str(tmp_path / 'ckpt.npz')
    emu.train(budget=1, checkpoint=checkpoint)
    reference = emu.to_cosmology().clone(**POINT).get_harmonic().lensed_cl()

    # a second training over the same checkpoint: every node is already there, so `extract` --
    # where `ell` is captured -- is never called
    again = Emulator(fiducial(), small_space(), section='harmonic', of=('lensed_cl',))
    assert again.ell is None
    again.train(budget=1, checkpoint=checkpoint)
    table = read(again.write(str(tmp_path / 'again.h5'))).to_cosmology().clone(**POINT) \
        .get_harmonic().lensed_cl()
    assert np.allclose(table['tt'], reference['tt'], rtol=1e-12, atol=0.)
    assert table['ell'][-1] == reference['ell'][-1]


def test_a_degenerate_hierarchy_is_served_by_its_total():
    """base_mnu splits a sampled total over three species, so the cosmology holds `[m / 3] * 3`
    where the space holds one number. Reading the per-species list as the space parameter stacked
    a (3,) against scalars ("All input arrays must have the same shape"); the total is both what
    the space means and what training cloned the fiducial from.
    """
    fid = fiducial(neutrino_hierarchy='degenerate', m_ncdm=0.12)
    emu = Emulator(fid, Space(bounds=dict(h=(0.66, 0.70), m_ncdm=(0.06, 0.30))),
                   section='harmonic', of=('lensed_cl',)).train(budget=1)
    point = {'h': 0.673, 'm_ncdm': 0.15}
    cosmo = emu.to_cosmology().clone(**point)
    assert len(cosmo['m_ncdm']) == 3 and np.allclose(cosmo['m_ncdm'], 0.05)
    assert np.allclose(cosmo.get_harmonic().lensed_cl()['tt'],
                       emu.predict(**point)['lensed_cl.tt'], rtol=1e-10, atol=0.)


def test_outside_the_trained_box_raises(trained):
    with pytest.raises(CoverageError):
        trained.to_cosmology().clone(h=0.5).get_harmonic().lensed_cl()


def test_clip_reports_the_violations_on_the_engine(trained):
    """``violation='clip'``: no error outside the box, the clipped prediction, and the
    violations recorded on the engine instance; the default engine still raises."""
    from cosmoprimo.emulators import emulated_engine
    engine = emulated_engine(trained, violation='clip')
    assert engine.violation == 'clip' and engine.emulator is trained
    base = trained.to_cosmology().clone(engine=engine)
    inside = base.clone(**POINT)
    assert np.allclose(inside.get_harmonic().lensed_cl()['tt'], trained.predict(**POINT)['lensed_cl.tt'], rtol=1e-12)
    assert all(float(value) == 0. for value in inside._engine.violations.values())
    outside = base.clone(h=0.72, omega_cdm=0.1201)                    # h above its (0.66, 0.70) range
    cl = outside.get_harmonic().lensed_cl()['tt']
    expected, violations = trained.predict_in_box(h=0.72, omega_cdm=0.1201)
    assert np.allclose(cl, expected['lensed_cl.tt'], rtol=1e-12)
    assert float(outside._engine.violations['box']) > 0.
    assert np.allclose(outside._engine.violations['box'], violations['box'])
    with pytest.raises(CoverageError):
        trained.to_cosmology().clone(h=0.72, omega_cdm=0.1201).get_harmonic().lensed_cl()


def test_asking_for_a_spectrum_that_was_not_emulated_raises(trained):
    # NOTE both parameters: cosmoprimo's default input basis is Omega_cdm, so clone(h=...) alone
    # moves omega_cdm too -- straight out of the trained box
    with pytest.raises(ValueError):
        trained.to_cosmology().clone(**POINT).get_harmonic().lens_potential_cl()


def test_emulating_a_non_cosmology_says_where_to_go_instead():
    with pytest.raises(TypeError, match='tools'):
        Emulator(lambda params: {'y': np.zeros(3)}, small_space())


def test_a_subclass_only_has_to_override_what_it_changes():
    """The point of the template: an emulator that flattens nothing is a two-line class, and one
    that does is only as long as the physics it knows."""
    # the template, not `cosmoprimo.emulators.Emulator`, which is the cosmology entry point
    from cosmoprimo.emulators.tools import Emulator as Template

    class Plain(Template):
        pass

    emu = Plain(lambda params: {'y': np.array([params['a'], params['a']**2])},
                Space(bounds={'a': (0., 1.)}))
    emu.train()
    assert np.allclose(emu.predict(a=0.3)['y'], [0.3, 0.09], atol=1e-8)
    # `predict` is all this layer offers. Turning a trained emulator back into the thing the
    # user started with -- `to_cosmology` here, `to_calculator` in desilike -- is the subclass's
    # business: it is a statement about a world the cosmology-free template has no notion of.
    assert not hasattr(emu, 'to_cosmology') and not hasattr(emu, 'to_calculator')


def test_emulate_builds_and_trains_in_one_call():
    """`Emulator` builds; `emulate` also pays. Routing is by name: `budget` reaches the engine
    through `train`, `of` reaches the section."""
    emu = emulate(fiducial(), small_space(), of=('lensed_cl',), budget=1)
    assert emu.trained
    assert error_vs_truth(emu.predict(**POINT)['lensed_cl.tt'], POINT) < 5e-3
    # what comes back is the emulator, not the cosmology: it can still be saved and validated
    assert emu.to_cosmology().get_harmonic() is not None


def test_emulator_alone_does_not_train():
    """Training is hours of Boltzmann calls, so building must never start it by accident."""
    emu = Emulator(fiducial(), small_space())
    assert not emu.trained
    assert len(emu.nodes(budget=1)) > 0        # sized without paying for it


# ── saving ────────────────────────────────────────────────────────────────────

def test_write_and_read_round_trip(trained, tmp_path):
    loaded = read(trained.write(str(tmp_path / 'cl.h5')))
    assert type(loaded) is type(trained)
    assert np.allclose(loaded.predict(**POINT)['lensed_cl.tt'],
                       trained.predict(**POINT)['lensed_cl.tt'], rtol=1e-12, atol=0.)
    # the box travels with it: a loaded emulator must not silently extrapolate either
    with pytest.raises(CoverageError):
        loaded.predict(h=0.5, omega_cdm=0.1201)


def test_cosmology_takes_the_saved_emulator_as_an_engine(trained, tmp_path):
    """The point of saving: `Cosmology(engine='cl.h5')` behaves like any other engine."""
    cosmo = Cosmology(engine=trained.write(str(tmp_path / 'cl.h5')))
    assert np.allclose(cosmo.clone(**POINT).get_harmonic().lensed_cl()['tt'],
                       trained.predict(**POINT)['lensed_cl.tt'], rtol=1e-12)


def test_hdf5_is_the_default_and_pickle_still_works(trained, tmp_path):
    """HDF5 by default because a trained emulator outlives the session: it is readable by anything
    and does not execute code when opened. `.npy` remains available for anything HDF5 cannot
    represent."""
    bare = trained.write(str(tmp_path / 'noextension'))
    assert bare.endswith('.h5')
    reference = trained.predict(**POINT)['lensed_cl.tt']
    for path in (bare, trained.write(str(tmp_path / 'cl.npy'))):
        assert np.allclose(read(path).predict(**POINT)['lensed_cl.tt'],
                           reference, rtol=1e-12, atol=0.)


def test_the_hdf5_file_mirrors_the_state_rather_than_hiding_it(trained, tmp_path):
    """A browsable file is the reason to prefer HDF5: `h5ls -r` must show the parameter names and
    the output names, not one opaque blob."""
    import h5py

    with h5py.File(trained.write(str(tmp_path / 'cl.h5')), 'r') as file:
        assert set(file['emulator']) >= {'cls', 'space', 'params', 'engines'}
        assert set(file['emulator/space/limits']) == {'h', 'omega_cdm'}
        assert 'lensed_cl.tt' in file['emulator/engines']
        assert file['emulator/space/limits/h'].attrs['type'] == 'tuple'


def test_an_untrained_emulator_refuses_to_be_saved(tmp_path):
    from cosmoprimo.emulators.tools import NotTrained

    with pytest.raises(NotTrained):
        Emulator(fiducial(), small_space()).write(str(tmp_path / 'nothing.h5'))


# ── several sections at once ──────────────────────────────────────────────────

@pytest.fixture(scope='module')
def multi():
    emu = Emulator(fiducial(), small_space(),
                   section={'harmonic': dict(of=('lensed_cl',)),
                            'background': dict(z=Z20, of=('efunc',))})
    return emu.train(budget=1)


def test_lensed_bb_carries_the_amplitude_twice(trained):
    """The rescaling a cosmology given sigma8 rather than A_s is served by is not one factor for
    every spectrum: without tensors the lensed bb is generated entirely by the lensing, so it is
    quadratic in the amplitude where tt, ee and te are linear.

    Measured against CAMB over 2 <= l <= 2500, fitting C(r^2 A_s) = r^(2p) C(A_s) for r^2 = 0.94
    and 1.06: p = 2.06 for lensed bb, 1.000 for every unlensed spectrum, for lensed tt and te and
    for the lensing potential, 0.996 for lensed ee. Scaling bb linearly leaves 6.7e-2 of its own
    peak, quadratically 7.7e-4. The non-linear matter power changes none of that by more than 30%:
    what the linear rescaling misses is the lensing, not halofit.
    """
    harmonic = trained.to_cosmology().clone(**POINT).get_harmonic()
    one = harmonic.lensed_cl()
    assert harmonic._rsigma8 == 1.       # nothing to rescale, the space is written in the amplitude
    ratio = 1.05
    harmonic._rsigma8 = ratio
    other = harmonic.lensed_cl()
    for name in ['tt', 'ee', 'te']:
        assert np.allclose(other[name], ratio**2 * one[name], rtol=1e-12, atol=0.), name
    assert np.allclose(other['bb'], ratio**4 * one['bb'], rtol=1e-12, atol=0.)
    # and the two statements differ by 10% at this amplitude, so the test has teeth
    assert not np.allclose(other['bb'], ratio**2 * one['bb'], rtol=1e-2, atol=0.)


def test_a_section_may_be_fitted_in_its_own_basis(tmp_path):
    """The sections share one node set -- that is what makes the extra ones cheap -- but not
    necessarily the coordinates they are fitted in.

    What a shared basis costs is what desilike avoids by giving each sector an emulator of its
    own: every output is expanded in the union of what any of them needs. Here the Fourier
    spectrum, which routes the amplitude exactly and wants `h` itself, would otherwise be fitted
    in `theta_MC_100` with `logA` on its grid because a lensed Cl needs both. Measured over a
    Planck-like box at budget 1, worst of 8 CAMB points: the spectrum goes from 2.7e-2 to 1.4e-2
    and the Cl are untouched, being fitted in the same theta basis either way.
    """
    theta = ['omega_cdm', 'omega_b', 'theta_MC_100']
    box = dict(h=(0.66, 0.70), omega_b=(0.0220, 0.0228), omega_cdm=(0.115, 0.125),
               logA=(3.0, 3.1))
    point = {'h': 0.673, 'omega_b': 0.02237, 'omega_cdm': 0.1201, 'logA': 3.04}
    sections = {'harmonic': dict(of=('lensed_cl',)), 'fourier': dict(k=KGRID, z=Z5)}
    emu = Emulator(fiducial(), Space(bounds=box),
                   section=sections, basis={None: theta, 'harmonic': theta, 'fourier': None})
    emu.train(budget=1)

    fitted = {name: params for name, (engine, shape, params) in emu._engines.items()}
    # the harmonic section shares the composite's basis, so it declares no coordinates of its own
    assert fitted['harmonic.lensed_cl.tt'] is None
    # and the Fourier one is expanded in `h`, without the amplitude it handles exactly
    assert fitted['fourier.pk.delta_m'] == ['h', 'omega_b', 'omega_cdm']

    predicted = emu.predict(**point)
    reloaded = read(emu.write(str(tmp_path / 'basis.h5')))
    for name, value in reloaded.predict(**point).items():
        assert np.allclose(value, predicted[name], rtol=1e-12, atol=0.), name


def test_a_basis_for_a_section_that_is_not_there_raises():
    with pytest.raises(ValueError, match='not sections'):
        Emulator(fiducial(), small_space(), section={'harmonic': {}},
                 basis={'fourier': 'theta'})


@pytest.mark.parametrize('sections, expected', [
    ({'harmonic': dict(of=('lensed_cl',)), 'background': dict(z=Z20, of=('efunc',))},
     {'harmonic.lensed_cl.tt', 'background.efunc'}),
    ({'harmonic': {}, 'background': dict(z=np.linspace(0., 2., 10)),
      'fourier': dict(k=KGRID, z=Z5), 'thermodynamics': dict(of=('rs_drag',))},
     {'harmonic.lensed_cl.tt', 'background.efunc', 'fourier.pk.delta_m',
      'thermodynamics.rs_drag'}),
])
def test_sections_share_one_boltzmann_call(sections, expected):
    """One clone per node however many sections ride along -- the arrangement the composite exists
    for. (The analytic divisors clone separately and cheaply, outside `compute`.)"""
    emu = Emulator(fiducial(), small_space(), section=sections)
    clones, values = count_clones(emu, NODE)
    assert clones == 1
    # outputs are prefixed by section, so two sections cannot collide on a name
    assert expected <= set(values)


def test_a_section_only_scales_its_own_outputs(multi):
    """Each section divides by its own factors: the harmonic amplitude must never reach
    `background.efunc`, which has no amplitude in it, nor the analytic efunc reach a Cl."""
    scaling = multi.scaling(NODE)
    assert all(name.startswith(('harmonic.', 'background.')) for name in scaling)
    # the background factor is the analytic efunc, not anything from the harmonic section
    analytic = multi.sections['background'].analytic_background(NODE)
    assert np.allclose(scaling['background.efunc'],
                       np.asarray(analytic.efunc(multi.sections['background'].z)))

    values = multi.compute(NODE)
    restored = multi.inverse_transform(multi.transform(values, NODE), NODE)
    for name, value in values.items():
        assert np.allclose(restored[name], value, rtol=1e-12, atol=0.)


def test_a_parameter_leaves_the_grid_only_if_every_section_is_exact():
    """The sections share the node set, so one section that needs a parameter expanded settles it
    for all of them."""
    space = Space(bounds={'h': (0.66, 0.70), 'logA': (3.00, 3.10)})
    # fourier alone: P(k) is exactly linear in A_s, so the amplitude leaves the grid
    assert Emulator(fiducial(), space, section='fourier').exact_params == ['logA']
    # with a lensed harmonic section, which is not linear in the amplitude, it cannot
    together = Emulator(fiducial(), space, section=['fourier', 'harmonic'])
    assert together.exact_params == [] and together.params == ['h', 'logA']


def test_multi_section_to_cosmology_serves_every_section(multi):
    cosmo = multi.to_cosmology().clone(**POINT)
    predicted = multi.predict(**POINT)
    assert np.allclose(cosmo.get_harmonic().lensed_cl()['tt'],
                       predicted['harmonic.lensed_cl.tt'], rtol=1e-10)
    assert np.allclose(cosmo.get_background().efunc(np.array([0.5, 1.5])),
                       np.interp([0.5, 1.5], Z20, predicted['background.efunc']), rtol=1e-3)
    truth = fiducial().clone(**POINT).get_background()
    assert np.allclose(cosmo.get_background().efunc(1.0), truth.efunc(1.0), rtol=1e-3)


def test_multi_section_round_trips_through_a_file(multi, tmp_path):
    loaded = read(multi.write(str(tmp_path / 'multi.h5')))
    assert sorted(loaded.sections) == ['background', 'harmonic']
    for name, value in multi.predict(**POINT).items():
        assert np.allclose(loaded.predict(**POINT)[name], value, rtol=1e-12, atol=0.)


# ── fourier ───────────────────────────────────────────────────────────────────

def test_fourier_round_trips_through_the_interpolator():
    """The k-z orientation is the easy thing to get silently backwards: a transposed grid still
    interpolates, it just returns the wrong spectrum."""
    k = np.logspace(-3., 0., 40)
    z = np.array([0., 0.25, 0.5, 1., 1.5])
    emu = Emulator(fiducial(), small_space(), section='fourier', k=k, z=z)
    emu.train(budget=1)

    truth = fiducial().clone(**POINT).get_fourier().pk_interpolator(of='delta_m')
    guess = emu.to_cosmology().clone(**POINT).get_fourier().pk_interpolator(of='delta_m')
    for redshift in z:
        assert np.allclose(guess(k, redshift), truth(k, redshift), rtol=5e-3)
    # a spectrum falls off steeply in k and grows in z; a transpose would break both
    assert guess(k[0], 0.) > guess(k[-1], 0.)
    assert guess(k[0], 0.) > guess(k[0], 1.)


def test_fourier_refuses_what_it_did_not_emulate():
    emu = Emulator(fiducial(), small_space(), section='fourier', k=KGRID, z=Z5)
    emu.train(budget=0)
    fourier = emu.to_cosmology().get_fourier()
    with pytest.raises(ValueError, match='non_linear'):
        fourier.pk_interpolator(of='delta_m', non_linear=True)
    with pytest.raises(ValueError, match='delta_cb'):
        fourier.pk_interpolator(of='delta_cb')


# ── the analytic divisors ─────────────────────────────────────────────────────

def test_the_analytic_background_matches_the_boltzmann_code():
    """The measurement the `analytic=True` default rests on. If this ever loosens, dividing by
    the analytic result stops being nearly-free accuracy and becomes just another approximation."""
    from cosmoprimo.cosmology import BaseEngine, DefaultBackground

    z = np.linspace(0.1, 3., 20)
    truth = fiducial().clone(**POINT).get_background()
    analytic = DefaultBackground(BaseEngine(fiducial().clone(**POINT)))
    for name, tolerance in [('efunc', 1e-10), ('growth_factor', 1e-10), ('growth_rate', 1e-10),
                            ('comoving_radial_distance', 1e-3)]:
        ratio = np.asarray(getattr(analytic, name)(z)) / np.asarray(getattr(truth, name)(z))
        assert np.max(np.abs(ratio - 1.)) < tolerance, name


def test_dividing_by_the_analytic_background_leaves_almost_nothing_to_fit():
    """The point of `analytic=True`: what the interpolant sees is a ratio of order 1, flat."""
    emu = Emulator(fiducial(), small_space(), section='background', z=Z20,
                   of=('efunc', 'growth_factor'))
    for name, values in emu.transform(emu.compute(POINT), POINT).items():
        assert np.allclose(values, 1., atol=1e-9), name


@pytest.mark.parametrize('analytic', [True, False])
def test_analytic_can_be_switched_off_and_the_round_trip_still_closes(analytic):
    emu = Emulator(fiducial(), small_space(), section='background', z=Z20, analytic=analytic)
    values = emu.compute(POINT)
    transformed = emu.transform(values, POINT)
    for name, value in emu.inverse_transform(transformed, POINT).items():
        assert np.allclose(value, values[name], rtol=1e-12, atol=0.), (analytic, name)
    # z = 0 is in the grid, where comoving_radial_distance is exactly zero: the guard must keep
    # a NaN out of the training data rather than let 0/0 through
    assert np.all(np.isfinite(transformed['comoving_radial_distance']))


def test_the_analytic_growth_flattens_the_redshift_axis_of_pk():
    """A linear P(k, z) divided by D(z)^2 is the same k-shape at every redshift."""
    def spread(analytic):
        emu = Emulator(fiducial(), small_space(), section='fourier',
                       k=np.logspace(-3., 0., 30), z=Z5, analytic=analytic)
        pk = emu.transform(emu.compute(LOGA_POINT), LOGA_POINT)['pk.delta_m']
        return np.max(pk.max(axis=1) / pk.min(axis=1) - 1.)

    assert spread(True) < 1e-3
    assert spread(False) > 1.        # without it, the full growth range


# ── thermodynamics ────────────────────────────────────────────────────────────

def test_eisenstein_hu_is_a_usable_divisor_even_where_its_own_engine_refuses():
    """The fiducial has massive neutrinos, which EisensteinHuEngine rejects outright. As a
    divisor the formula is still fine: it only has to be smooth and roughly right."""
    from cosmoprimo.emulators.cosmology import _eisenstein_hu_scales

    cosmo = fiducial().clone(**POINT)
    scales, truth = _eisenstein_hu_scales(cosmo), cosmo.get_thermodynamics()
    assert 0.9 < scales['rs_drag'] / truth.rs_drag < 1.1
    assert 0.9 < scales['z_drag'] / truth.z_drag < 1.1


def test_thermodynamics_end_to_end():
    emu = Emulator(fiducial(), small_space(), section='thermodynamics',
                   of=('rs_drag', 'z_drag', 'theta_star'))
    emu.train(budget=1)
    truth = fiducial().clone(**POINT).get_thermodynamics()
    served = emu.to_cosmology().clone(**POINT).get_thermodynamics()
    for name in ('rs_drag', 'z_drag', 'theta_star'):
        assert np.isclose(getattr(served, name), getattr(truth, name), rtol=1e-4), name


# ── training in a different basis from the one the space is written in ────────

def test_cosmology_converts_between_bases():
    """`Cosmology._get_params` derives names through the parameter compilation, with no engine:
    the conversion belongs to the cosmology, because nothing else holds the fiducial values of
    what the user did not vary."""
    converted = Cosmology._get_params({'Omega_m': 0.31, 'Omega_b': 0.049, 'h': 0.68},
                                      ['omega_cdm', 'omega_b', 'h'])
    assert set(converted) == {'omega_cdm', 'omega_b', 'h'}
    assert np.isclose(converted['h'], 0.68)
    back = Cosmology._get_params(converted, ['Omega_m', 'Omega_b', 'h'])
    assert np.isclose(back['Omega_m'], 0.31, rtol=1e-8)
    assert np.isclose(back['Omega_b'], 0.049, rtol=1e-8)


def test_the_base_supplies_what_was_not_varied_and_conflicts_are_resolved():
    """`Omega_m` against a fiducial holding `Omega_cdm` must replace it, not clash -- and the
    fiducial's neutrino content must still be the one used, since Omega_cdm depends on it."""
    heavy = Cosmology._get_params({'Omega_m': 0.31, 'h': 0.68}, ['omega_cdm', 'm_ncdm'],
                                  base=fiducial().clone(m_ncdm=0.12)._input_params)
    light = Cosmology._get_params({'Omega_m': 0.31, 'h': 0.68}, ['omega_cdm'],
                                  base=fiducial().clone(m_ncdm=0.06)._input_params)
    assert np.isclose(np.sum(heavy['m_ncdm']), 0.12)
    # at fixed Omega_m, heavier neutrinos take their density out of the cdm
    assert heavy['omega_cdm'] < light['omega_cdm']


def test_the_basis_change_is_not_a_rescaling():
    """Why whitening cannot absorb it: at fixed Omega_m, omega_cdm still moves with h."""
    values = [Cosmology._get_params({'Omega_m': 0.31, 'h': h}, ['omega_cdm'])['omega_cdm']
              for h in (0.64, 0.72)]
    assert values[-1] / values[0] > 1.2


def test_mapping_a_space_keeps_every_point_it_accepted():
    """The property whose absence broke a perfectly valid prediction: a point inside the source
    box must land inside the mapped box. It is not automatic -- the image of an ellipsoid under a
    non-linear map is not an ellipsoid, so mean +- nsigma of the image cuts corners off."""
    cosmo, physical = fiducial(), ['omega_cdm', 'omega_b', 'h']
    names = ['Omega_m', 'Omega_b', 'h']
    mean, sigma = np.array([0.31, 0.049, 0.6766]), np.array([0.0073, 0.0009, 0.0054])
    corr = np.array([[1., 0.35, -0.92], [0.35, 1., -0.45], [-0.92, -0.45, 1.]])
    draws = np.random.default_rng(11).multivariate_normal(
        mean, corr * np.outer(sigma, sigma), size=5000)
    space = Space(samples={name: draws[:, index] for index, name in enumerate(names)})

    convert = lambda point: Cosmology._get_params(point, physical, base=cosmo._input_params)
    mapped = space.map(convert)
    assert mapped.params == physical
    for point in space.draw(size=300, seed=5):
        if space.contains(point):
            assert mapped.contains(convert(point)), point


@pytest.fixture(scope='module')
def basis_trained():
    emu = Emulator(fiducial(), Space(bounds=OMEGAS), section='harmonic', basis='physical')
    return emu.train(budget=1)


def test_emulating_in_a_physical_basis_from_a_space_written_in_omegas(basis_trained):
    """The user's space stays in Omega_m; the interpolant expands omega_cdm; predict takes
    Omega_m and converts."""
    assert basis_trained.space.params == ['Omega_m', 'Omega_b', 'h']
    assert basis_trained.params == ['omega_cdm', 'omega_b', 'h']
    guess = basis_trained.predict(**OMEGA_POINT)['lensed_cl.tt']
    assert error_vs_truth(guess, OMEGA_POINT) < 5e-3
    # and the calculator route: the engine reads Omega_m off the cosmology, predict converts
    served = basis_trained.to_cosmology().clone(**OMEGA_POINT).get_harmonic().lensed_cl()
    assert np.allclose(served['tt'], guess, rtol=1e-10)


def test_a_basis_emulator_round_trips_through_a_file(basis_trained, tmp_path):
    loaded = read(basis_trained.write(str(tmp_path / 'basis.h5')))
    assert loaded.basis == ['omega_cdm', 'omega_b', 'h']
    assert loaded.params == basis_trained.params
    assert loaded.space.params == ['Omega_m', 'Omega_b', 'h']
    assert np.allclose(loaded.predict(**OMEGA_POINT)['lensed_cl.tt'],
                       basis_trained.predict(**OMEGA_POINT)['lensed_cl.tt'], rtol=1e-12, atol=0.)


def test_a_coverage_failure_in_the_training_basis_says_what_was_given(basis_trained):
    """An error in omega_cdm is useless if the user only ever typed Omega_m."""
    with pytest.raises(CoverageError, match='you gave'):
        basis_trained.predict(Omega_m=0.5, Omega_b=0.049, h=0.673)


def test_a_basis_cannot_add_a_direction_the_space_does_not_have():
    """Three physical densities out of a two-parameter space leaves one a deterministic function
    of the others. Saying so here beats the whitening quietly dividing by nothing and the failure
    surfacing later as an unfindable node."""
    space = Space(bounds={'Omega_m': (0.30, 0.32), 'h': (0.66, 0.69)})
    with pytest.raises(ValueError, match='reparametrisation'):
        Emulator(fiducial(), space, section='harmonic', basis='physical')


# ── jax ───────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def jax_trained():
    return Emulator(fiducial(), Space(bounds=WITH_LOGA),
                    section='harmonic').train(budget=1)


def test_predict_jits_and_differentiates(jax_trained):
    """The point of an emulator: a likelihood wraps it in `jit` and asks for gradients. Both
    were broken until `predict` stopped calling np.array() on its inputs."""
    import jax

    def scalar(h, omega_cdm, logA):
        return jax_trained.predict(h=h, omega_cdm=omega_cdm, logA=logA)['lensed_cl.tt'][100]

    point = (0.673, 0.1201, 3.04)
    assert np.isclose(jax.jit(scalar)(*point), scalar(*point), rtol=1e-12)

    gradient = jax.grad(scalar, argnums=0)(*point)
    assert np.isfinite(gradient) and gradient != 0.
    # against a finite difference, which is what the gradient is for
    step = 1e-4
    finite = (scalar(0.673 + step, 0.1201, 3.04) - scalar(0.673 - step, 0.1201, 3.04)) / (2 * step)
    assert np.isclose(gradient, finite, rtol=1e-3)


def test_the_amplitude_scaling_traces_too(jax_trained):
    """`inverse_transform` runs at every prediction, so an np.exp there would break the trace
    just as surely as one in the engine."""
    import jax

    assert np.isfinite(jax.grad(lambda logA: jax_trained.predict(
        h=0.673, omega_cdm=0.1201, logA=logA)['lensed_cl.tt'][100])(3.04))


def test_the_whole_cosmology_route_traces(jax_trained):
    """clone -> get_harmonic -> read a spectrum, all inside a jit. A numpy structured array
    could not hold tracers, so the emulated section returns a lookalike instead."""
    import jax

    fast = jax_trained.to_cosmology()

    def through_cosmology(h):
        return fast.clone(h=h, omega_cdm=0.1201, logA=3.04).get_harmonic().lensed_cl()['tt'][100]

    assert np.isclose(jax.jit(through_cosmology)(0.673), through_cosmology(0.673), rtol=1e-12)
    assert np.isfinite(jax.grad(through_cosmology)(0.673))


def test_the_emulated_table_behaves_like_a_structured_array(jax_trained):
    table = jax_trained.to_cosmology().clone(**LOGA_POINT).get_harmonic().lensed_cl()
    assert isinstance(table.dtype, np.dtype)          # a real structured dtype, not a stand-in
    assert set(table.dtype.names) == {'ell', 'tt', 'ee', 'bb', 'te'}
    assert table.dtype['tt'] == np.float64 and table.dtype.fields is not None
    # the native section's spectra are named the same way, which is the point of the lookalike
    native = fiducial().clone(**LOGA_POINT).get_harmonic().lensed_cl()
    assert set(native.dtype.names) == set(table.dtype.names)
    assert len(table) == len(table['ell']) and table['ell'][-1] == 500
    masked = table[np.asarray(table['ell']) >= 30]        # a mask applies to every column
    assert len(masked['tt']) == len(masked['ell']) == len(table) - 30


@pytest.mark.parametrize('section, options', [
    ('background', dict(z=np.linspace(0., 2., 10), of=('efunc',))),
    ('thermodynamics', dict(of=('rs_drag',))),
    ('fourier', dict(k=KGRID, z=Z5)),
])
def test_every_sections_scaling_traces(section, options):
    """The analytic divisors run at every prediction too, so a numpy cast in one of them breaks
    the trace exactly as an np.array() in `predict` did."""
    emu = Emulator(fiducial(), Space(bounds=WITH_LOGA), section=section, **options)
    for name, factor in emu.scaling(LOGA_POINT).items():
        assert np.all(np.isfinite(np.asarray(factor))), (section, name)
        assert np.all(np.asarray(factor) != 0.), (section, name)


# ── the theta basis ───────────────────────────────────────────────────────────
#
# `basis='theta'` replaces h with the acoustic scale, which is what a CMB box wants and the one
# basis needing a real inverse: theta_MC_100 is not something `clone` accepts.

def _theta_space(size=200, seed=42):
    """A chain-shaped space in the physical densities, small enough to map quickly."""
    rng = np.random.RandomState(seed)
    return Space(samples={'omega_cdm': rng.uniform(0.115, 0.125, size),
                          'omega_b': rng.uniform(0.0220, 0.0226, size),
                          'h': rng.uniform(0.655, 0.695, size)})


def _theta_emulator(**options):
    from cosmoprimo.fiducial import DESI

    return Emulator(DESI(engine='eisenstein_hu'), _theta_space(), section='background',
                    of=('efunc', 'comoving_radial_distance'), basis='theta', **options)


def test_theta_basis_replaces_h_and_nothing_else():
    emulator = _theta_emulator()
    assert 'theta_MC_100' in emulator.training.params and 'h' not in emulator.training.params
    assert set(emulator.training.params) == {'omega_cdm', 'omega_b', 'theta_MC_100'}
    # a reparametrisation, so the dimension is unchanged and the image is a real theta range
    low, high = emulator.training.limits['theta_MC_100']
    assert 1.0 < low < high < 1.1


def test_theta_round_trip_is_exact():
    """The same closed form in both directions, so the composition is the identity -- which is
    what lets the formula's offset from the engine's own theta_MC_100 cancel. An inverse through
    `Cosmology.solve` would not: the node would be fitted at one theta and recorded at another."""
    emulator = _theta_emulator()
    for h in (0.66, 0.68, 0.694):
        point = {'omega_cdm': 0.12, 'omega_b': 0.0223, 'h': h}
        back = emulator.from_training(emulator.to_training(point))
        assert np.isclose(float(back['h']), h, rtol=0., atol=1e-9)
        assert 'theta_MC_100' not in back


def test_theta_basis_takes_a_traced_total_neutrino_mass():
    """base_mnu varies the total mass, and the basis is inverted inside a jit: building the mass
    array with numpy killed the trace. The species are what theta needs -- three of 0.05 and one
    of 0.15 are different radiation contents at the same `N_ur` -- so the fiducial's own split is
    scaled to the sampled total.
    """
    import jax
    from cosmoprimo.fiducial import DESI

    fid = DESI(engine='eisenstein_hu').clone(neutrino_hierarchy='degenerate', m_ncdm=0.12)
    rng = np.random.RandomState(42)
    space = Space(samples={'omega_cdm': rng.uniform(0.115, 0.125, 200),
                           'omega_b': rng.uniform(0.0220, 0.0226, 200),
                           'h': rng.uniform(0.655, 0.695, 200),
                           'm_ncdm': rng.uniform(0.06, 0.30, 200)})
    emulator = Emulator(fid, space, section='background', of=('efunc',), basis='theta')
    point = {'omega_cdm': 0.12, 'omega_b': 0.0223, 'h': 0.67, 'm_ncdm': 0.15}
    back = jax.jit(emulator.from_training)(emulator.to_training(point))
    assert np.isclose(float(back['h']), point['h'], rtol=0., atol=1e-9)
    masses = emulator._theta_kwargs(point)['m_ncdm']
    assert len(masses) == 3 and np.allclose(masses, 0.05)
    # a cosmology writes the same mass as its species, and `to_training` is where the two
    # conventions meet: the basis must not care which way it was handed the neutrinos
    assert np.isclose(emulator.to_training({**point, 'm_ncdm': [0.05, 0.05, 0.05]})['theta_MC_100'],
                      emulator.to_training(point)['theta_MC_100'], rtol=0., atol=1e-12)


def test_theta_basis_trains_and_predicts():
    """End to end: the nodes are laid out in theta, evaluated by inverting to h, and a prediction
    entered in the user's own h reproduces the cosmology it was trained on."""
    from cosmoprimo.fiducial import DESI

    emulator = _theta_emulator().train(budget=1)
    point = {'omega_cdm': 0.121, 'omega_b': 0.0223, 'h': 0.674}
    predicted = emulator.predict(**point)
    exact = DESI(engine='eisenstein_hu').clone(**point).get_background()
    z = emulator.sections['background'].z
    for name in ('efunc', 'comoving_radial_distance'):
        # 3e-4 rather than 1e-4 for one node out of 256: the lowest-z distance, where the number
        # itself is small (73 Mpc/h) and a budget-1 fit leaves 1.1e-4 of it
        np.testing.assert_allclose(predicted[name], getattr(exact, name)(z), rtol=3e-4)


def test_theta_basis_survives_a_write(tmp_path):
    emulator = _theta_emulator().train(budget=1)
    point = {'omega_cdm': 0.121, 'omega_b': 0.0223, 'h': 0.674}
    reloaded = read(emulator.write(str(tmp_path / 'theta.h5')))
    assert reloaded.basis == list(emulator.basis)
    before, after = emulator.predict(**point), reloaded.predict(**point)
    for name in before:
        np.testing.assert_allclose(after[name], before[name], rtol=1e-12, atol=0.)


# ── the tilt ──────────────────────────────────────────────────────────────────

def _fourier_emulator(**options):
    from cosmoprimo.fiducial import DESI

    return Emulator(DESI(engine='eisenstein_hu'),
                    Space(bounds={'omega_cdm': (0.11, 0.13), 'n_s': (0.94, 0.99),
                                  'logA': (2.9, 3.2)}),
                    section='fourier', k=np.geomspace(1e-3, 1., 80), z=np.array([0., 1.]),
                    **options)


def test_the_tilt_takes_n_s_off_the_grid():
    """A whole dimension, not a flattening: the tilt enters a linear spectrum through the
    primordial one alone."""
    emulator = _fourier_emulator()
    assert emulator.params == ['omega_cdm'] and set(emulator.exact_params) == {'n_s', 'logA'}
    assert _fourier_emulator(tilt=False).params == ['omega_cdm', 'n_s']
    # and it is what the node count is spent on: 3 against 5 at budget 1
    assert len(emulator.nodes(budget=1)) < len(_fourier_emulator(tilt=False).nodes(budget=1))


def test_the_tilt_is_exact():
    """The ratio emulated / exact is the same number at three tilts, to machine precision."""
    emulator = _fourier_emulator().train(budget=1)
    ratios = [np.asarray(emulator.predict(**point)['pk.delta_m'])
              / np.asarray(emulator.compute(point)['pk.delta_m'])
              for point in [{'omega_cdm': 0.12, 'n_s': n_s, 'logA': 3.04}
                            for n_s in (0.94, 0.965, 0.99)]]
    for ratio in ratios:
        np.testing.assert_allclose(ratio, ratios[1], rtol=0., atol=1e-12)


def test_the_tilt_is_off_for_a_non_linear_spectrum():
    """Halofit mixes scales, so the factorisation fails and n_s goes back on the grid."""
    emulator = _fourier_emulator(non_linear=True)
    assert emulator.tilt is False and 'n_s' in emulator.params


# ── the dilation ──────────────────────────────────────────────────────────────

def _dilated_emulator(**options):
    from cosmoprimo.fiducial import DESI

    return Emulator(DESI(engine='eisenstein_hu'),
                    Space(bounds={'omega_cdm': (0.11, 0.13), 'h': (0.62, 0.72)}),
                    section='fourier', k=np.geomspace(1e-4, 10., 400), z=np.array([0., 1.]),
                    **options)


def test_the_dilation_takes_h_off_a_physical_grid():
    emulator = _dilated_emulator(dilate=True)
    assert emulator.params == ['omega_cdm'] and 'h' in emulator.exact_params
    assert 'h' in _dilated_emulator(dilate=False).params


def test_the_dilation_keeps_h_on_a_grid_it_cannot_hold_fixed():
    """The dilation holds the physical densities fixed, which a space in Omega_m does not: taking
    h off the grid there would interpolate in Omega_m at an implied omega_cdm moving with the h
    the dilation is meanwhile handling."""
    from cosmoprimo.fiducial import DESI

    emulator = Emulator(DESI(engine='eisenstein_hu'),
                        Space(bounds={'Omega_m': (0.29, 0.33), 'h': (0.62, 0.72)}),
                        section='fourier', k=np.geomspace(1e-3, 1., 60), z=np.array([0.]),
                        dilate=True)
    assert 'h' in emulator.params


def test_the_dilation_reproduces_the_spectrum():
    """Its accuracy is the resampling of its own k grid, so this is checked on a dense one."""
    emulator = _dilated_emulator(dilate=True).train(budget=2)
    k = emulator.sections['fourier'].k
    inside = (k > 1e-3) & (k < 1.)
    for h in (0.63, 0.71):
        point = {'omega_cdm': 0.12, 'h': h}
        ratio = (np.asarray(emulator.predict(**point)['pk.delta_m'])
                 / np.asarray(emulator.compute(point)['pk.delta_m']))
        assert np.max(np.abs(ratio[inside] - 1.)) < 5e-3, h


def test_a_file_written_before_the_growth_convention_changed_is_refused():
    """The divisor changed, so an old file would predict confidently and wrongly."""
    from cosmoprimo.emulators.tools import StateVersionError

    emulator = _fourier_emulator().train(budget=1)
    state = emulator.__getstate__()
    state['version'] = 1
    with pytest.raises(StateVersionError):
        type(emulator).from_state(state)
