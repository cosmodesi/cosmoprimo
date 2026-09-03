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


def test_outside_the_trained_box_raises(trained):
    with pytest.raises(CoverageError):
        trained.to_cosmology().clone(h=0.5).get_harmonic().lensed_cl()


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
    factors = multi._factors(NODE)
    assert all(name.startswith(('harmonic.', 'background.')) for name in factors)
    # the background factor is the analytic efunc, not anything from the harmonic section
    analytic = multi.sections['background'].analytic_background(NODE)
    assert np.allclose(factors['background.efunc'],
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
