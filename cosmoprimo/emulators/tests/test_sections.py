"""What each section knows about itself, and what a served section may be asked to do.

The companion to ``test_cosmology.py``, which tests the emulator machinery: this one is about the
physics written into the sections -- which growth divides a velocity spectrum, what a fixed k/z
grid serves back, whether the amplitude may be spelled sigma8 -- and about the one property the
served sections have to have and nothing else checks: they are read inside a ``jit``.

Cheap on purpose: eisenstein_hu wherever the shape is all that is being checked, and the accuracy
statements quote measurements made against CLASS elsewhere rather than re-running them.
"""

import numpy as np
import pytest

from cosmoprimo import Cosmology
from cosmoprimo.cosmology import BaseEngine, DefaultBackground
from cosmoprimo.emulators import (emulate, Space, FourierEmulator, BackgroundEmulator,
                                  with_harmonic_precision)
from cosmoprimo.emulators.cosmology import _growth_factor_sq, _of_legs, _of_name


def fiducial(engine='eisenstein_hu', **kwargs):
    return Cosmology(engine=engine, **kwargs)


BOX = dict(h=(0.66, 0.70), omega_cdm=(0.115, 0.125))
POINT = {'h': 0.673, 'omega_cdm': 0.1201}
Z = np.linspace(0., 2., 5)
K = np.logspace(-3., 0., 40)


def fourier(of=('delta_cb',), **kwargs):
    return FourierEmulator(fiducial(), Space(bounds=BOX), k=K, z=Z, of=of, **kwargs)


# ── which spectrum is which ───────────────────────────────────────────────────

@pytest.mark.parametrize('of,legs,name', [
    ('delta_cb', ('delta_cb', 'delta_cb'), 'delta_cb'),
    (('delta_cb',), ('delta_cb', 'delta_cb'), 'delta_cb'),
    (('delta_cb', 'delta_cb'), ('delta_cb', 'delta_cb'), 'delta_cb'),
    (('delta_cb', 'theta_cb'), ('delta_cb', 'theta_cb'), 'delta_cb_theta_cb'),
])
def test_a_spectrum_has_one_name_however_it_is_spelled(of, legs, name):
    assert _of_legs(of) == legs
    assert _of_name(of) == name


def test_a_cross_spectrum_is_named_by_both_legs():
    emulator = fourier(of=('delta_cb', ('delta_cb', 'theta_cb')))
    assert set(emulator.names) == {'pk.delta_cb', 'pk.delta_cb_theta_cb'}
    values = emulator.compute(POINT)
    assert set(values) == {'pk.delta_cb', 'pk.delta_cb_theta_cb'}
    # and it really is the cross spectrum, not the density one under another name
    assert not np.allclose(values['pk.delta_cb'], values['pk.delta_cb_theta_cb'])


# ── the growth divisor counts the velocity legs ───────────────────────────────

@pytest.mark.parametrize('of,nrate', [('delta_cb', 0), (('delta_cb', 'theta_cb'), 1),
                                      ('theta_cb', 2)])
def test_the_growth_divisor_carries_one_f_per_velocity_leg(of, nrate):
    """:math:`P_{\\delta\\delta} \\propto D^2`, :math:`P_{\\delta\\theta} \\propto f D^2`,
    :math:`P_{\\theta\\theta} \\propto f^2 D^2`."""
    emulator = fourier(of=(of,))
    background = DefaultBackground(BaseEngine(fiducial().clone(**POINT)))
    expected = np.asarray(background.growth_factor(Z, znorm=0.))**2 \
        * np.asarray(background.growth_rate(Z))**nrate
    divisor = _growth_factor_sq(emulator.analytic_background(POINT), of)
    assert np.allclose(divisor(Z), expected, rtol=1e-10)


def test_dividing_a_velocity_spectrum_by_the_density_growth_leaves_f_squared():
    """The bug this guards: with :math:`D^2` for every spectrum, a theta-theta one keeps the whole
    of :math:`f^2` -- which is most of the z dependence the divisor exists to remove."""
    emulator = fourier(of=('theta_cb',))
    values = emulator.compute(POINT)
    right = emulator.transform(values, POINT)['pk.theta_cb']
    density_growth = _growth_factor_sq(emulator.analytic_background(POINT), 'delta_cb')
    wrong = values['pk.theta_cb'] / density_growth(Z)[None, :]

    def spread(array):     # how much of the z axis is left for the interpolant
        return np.max(np.abs(array / array[:, :1] - 1.))

    assert spread(right) < 0.05 * spread(wrong)


@pytest.mark.parametrize('of', ['delta_cb', 'theta_cb', ('delta_cb', 'theta_cb')])
def test_the_scaling_round_trip_closes_for_every_kind_of_spectrum(of):
    emulator = fourier(of=(of,))
    values = emulator.compute(POINT)
    back = emulator.inverse_transform(emulator.transform(values, POINT), POINT)
    for name, value in values.items():
        assert np.allclose(back[name], value, rtol=1e-12)


# ── the amplitude, in either spelling ─────────────────────────────────────────

def test_sigma8_is_the_amplitude_under_another_name():
    r"""At fixed shape :math:`P \propto A_s \propto \sigma_8^2`, so a space written in
    :math:`\sigma_8` divides out :math:`\sigma_8^2` and the parameter leaves the grid."""
    space = Space(bounds=dict(BOX, sigma8=(0.79, 0.83)))
    emulator = FourierEmulator(fiducial(sigma8=0.81), space, k=K, z=Z)
    assert 'sigma8' not in emulator.select_params(space.params)
    point = dict(POINT, sigma8=0.805)
    factor = emulator.scaling(point)['pk.delta_m']
    growth = _growth_factor_sq(emulator.analytic_background(point), 'delta_m')(Z)
    expected = 0.805**2 * growth[None, :]
    assert np.allclose(factor / factor.max(), expected / expected.max())


def test_a_sigma8_space_predicts_the_asked_for_amplitude():
    space = Space(bounds=dict(BOX, sigma8=(0.79, 0.83)))
    emulator = emulate(fiducial(sigma8=0.81), space, section='fourier', k=K, z=Z, budget=1)
    point = dict(POINT, sigma8=0.805)
    predicted = emulator.predict(**point)['pk.delta_m']
    truth = fiducial(sigma8=0.805).clone(**POINT).get_fourier().pk_interpolator()(K, Z)
    assert np.allclose(predicted, truth, rtol=2e-3)


def test_an_A_s_emulator_serves_a_sigma8_cosmology():
    """cosmoprimo's own convention: run at a first-guess amplitude, then rescale so sigma8 comes
    out right. Exact here, because the spectrum is exactly linear in the amplitude."""
    space = Space(bounds=dict(BOX, logA=(3.0, 3.1)))
    emulator = emulate(fiducial(), space, section='fourier', k=K, z=Z, budget=1)
    cosmo = emulator.to_cosmology().clone(sigma8=0.79, **POINT)
    assert np.isclose(cosmo.get_fourier().sigma8_m, 0.79, rtol=1e-6)


# ── the fixed grid, and what the interpolator is handed ───────────────────────

def test_a_training_redshift_comes_back_exactly():
    """The z grid is short because the growth is handed to the interpolator rather than splined
    (`growth_factor_sq`); the price of that would be paying for it at the nodes too, and it is
    not: the same array on the same nodes divides and multiplies back.

    An A_s-parameterised fiducial, so that nothing else touches the amplitude -- see
    :func:`test_a_sigma8_cosmology_is_renormalised_to_its_sigma8`.
    """
    emulator = emulate(fiducial(logA=3.04), Space(bounds=BOX), section='fourier', k=K, z=Z,
                       budget=1)
    cosmo = emulator.to_cosmology().clone(**POINT)
    interpolator = cosmo.get_fourier().pk_interpolator()
    predicted = emulator.predict(**POINT)['pk.delta_m']
    assert np.allclose(interpolator(K, Z), predicted, rtol=1e-9)


def test_a_sigma8_cosmology_is_renormalised_to_its_sigma8():
    """A sigma8-parameterised cosmology -- which cosmoprimo's default is -- has its emulated
    spectrum rescaled so that sigma8 comes out exactly right, exactly as a native engine's is.

    So the served spectrum is not the raw prediction there but the prediction with its own
    sigma8 error divided out, which is a small renormalisation and a real one.
    """
    emulator = emulate(fiducial(sigma8=0.81), Space(bounds=BOX), section='fourier', k=K, z=Z,
                       budget=1)
    cosmo = emulator.to_cosmology().clone(**POINT)
    served = cosmo.get_fourier().pk_interpolator()(K, Z)
    predicted = emulator.predict(**POINT)['pk.delta_m']
    ratio = served / predicted
    assert np.allclose(ratio, np.asarray(ratio).flatten()[0], rtol=1e-12)   # one number, not a shape
    assert np.isclose(cosmo.get_fourier().sigma8_m, 0.81, rtol=1e-6)


def test_the_growth_goes_to_the_interpolator_not_into_the_array():
    """A served spectrum between the nodes follows the analytic growth, so a 5-node grid is
    accurate where splining P itself over 5 nodes is not."""
    emulator = emulate(fiducial(), Space(bounds=BOX), section='fourier', k=K, z=Z, budget=1)
    cosmo = emulator.to_cosmology().clone(**POINT)
    interpolator = cosmo.get_fourier().pk_interpolator()
    assert interpolator.growth_factor_sq is not None
    z_between = 0.5 * (Z[:-1] + Z[1:])
    truth = fiducial().clone(**POINT).get_fourier().pk_interpolator()(K, z_between)
    assert np.allclose(interpolator(K, z_between), truth, rtol=5e-3)


def test_pk_now_is_filtered_from_the_emulated_spectrum():
    emulator = emulate(fiducial(), Space(bounds=BOX), section='fourier',
                       k=np.logspace(-4., 1., 200), z=Z, budget=1)
    fourier_section = emulator.to_cosmology().clone(**POINT).get_fourier()
    pk = fourier_section.pk_interpolator()(K, z=0.)
    pknow = fourier_section.pk_now_interpolator(cosmo=None)(K, z=0.)
    assert np.allclose(pknow, pk, rtol=0.2)      # same spectrum, wiggles removed
    assert np.max(np.abs(pknow / pk - 1.)) > 1e-4  # and they are not the same array


# ── the background: the density fractions, and the divisor that is not one ────

def test_the_density_fractions_can_be_emulated_and_are_conditioned():
    """They are exact for the standard expansion and are not for a modified one, which is when
    they are worth a node -- so they are available, and conditioned on the analytic background."""
    of = ('efunc', 'Omega_cdm', 'Omega_b', 'Omega_de')
    emulator = BackgroundEmulator(fiducial(), Space(bounds=BOX), z=Z, of=of)
    values = emulator.compute(POINT)
    assert set(values) == set(of)
    scaling = emulator.scaling(POINT)
    assert set(scaling) == set(of)
    for name in of:                     # the ratio is what is fitted, and it is ~1
        assert np.allclose(values[name] / scaling[name], 1., atol=1e-3)


def test_age_is_served_as_a_scalar():
    emulator = emulate(fiducial(), Space(bounds=BOX), section='background', z=Z,
                       of=('efunc', 'age'), budget=1)
    background = emulator.to_cosmology().clone(**POINT).get_background()
    assert np.ndim(background.age) == 0
    assert np.isclose(background.age, fiducial().clone(**POINT).get_background().age, rtol=1e-3)


def test_a_divisor_that_is_nan_is_not_divided_by():
    r"""``DefaultBackground.growth_factor`` is solved on a grid that stops at
    ``_GROWTH_ZMAX``, so it is nan beyond. The default background grid stops there too, but a
    caller may ask for more, and dividing by a nan puts one in the training values, where it
    poisons every coefficient of the fit rather than failing."""
    from cosmoprimo.cosmology import _GROWTH_ZMAX

    z = np.concatenate([BackgroundEmulator(fiducial(), Space(bounds=BOX)).z, [2. * _GROWTH_ZMAX]])
    emulator = BackgroundEmulator(fiducial(), Space(bounds=BOX), z=z)
    assert emulator.z.max() > _GROWTH_ZMAX
    analytic = np.asarray(emulator.analytic_background(POINT).growth_factor(emulator.z, znorm=0.))
    assert not np.isfinite(analytic).all()                              # the trap is real
    scaling = emulator.scaling(POINT)
    for name, value in scaling.items():
        assert np.isfinite(value).all(), name
        assert (np.asarray(value) != 0.).all(), name
    values = emulator.compute(POINT)
    for name, value in emulator.transform(values, POINT).items():
        assert np.isfinite(value).all(), name


def test_a_space_may_vary_a_parameter_the_cosmology_holds_as_a_list():
    """`m_ncdm` is one value per massive species on a cosmology and one number in a space.

    The emulator is trained on the number; the deployed engine reads `[m]` off the cosmology it
    serves, and `predict` stacks its training parameters -- so the two shapes met there and it
    refused with "All input arrays must have the same shape". That is what `base_mnu` does.
    """
    space = Space(bounds=dict(BOX, m_ncdm=(0.05, 0.25)))
    emulator = emulate(fiducial(), space, budget=1,
                       section={'background': dict(z=Z, of=('efunc', 'age')),
                                'thermodynamics': dict(of=('rs_drag',))})
    point = dict(POINT, m_ncdm=0.18)
    served, exact = emulator.to_cosmology().clone(**point), fiducial().clone(**point)
    assert np.isclose(served.get_background().efunc(1.), exact.get_background().efunc(1.),
                      rtol=1e-4)
    assert np.isclose(served.get_thermodynamics().rs_drag, exact.get_thermodynamics().rs_drag,
                      rtol=1e-4)
    # and the total is what it means: the fiducial's hierarchy splits 0.18 over its species, and
    # the emulator is queried with the number it was trained on rather than that list
    assert np.isclose(sum(np.atleast_1d(served['m_ncdm'])), point['m_ncdm'])


# ── a served section is read inside a jit ─────────────────────────────────────

@pytest.fixture(scope='module')
def served():
    """One trained emulator over every section, and the cosmology it serves."""
    emulator = emulate(fiducial(), Space(bounds=BOX), budget=1,
                       section={'fourier': dict(k=K, z=Z),
                                'background': dict(z=Z, of=('efunc', 'luminosity_distance')),
                                'thermodynamics': dict(of=('rs_drag', 'z_drag'))})
    return emulator


def test_the_served_sections_run_inside_a_jit(served):
    """The whole point of an emulator is to be called inside a compiled likelihood, and three
    things used to stop that: `float()` on a thermodynamics scalar, `np.asarray` on a redshift
    that may be a tracer, and a structured array for the Cl."""
    jax = pytest.importorskip('jax')

    def compute(h, z):
        cosmo = served.to_cosmology().clone(h=h, omega_cdm=POINT['omega_cdm'])
        background, thermodynamics = cosmo.get_background(), cosmo.get_thermodynamics()
        fourier = cosmo.get_fourier()
        return (background.luminosity_distance(z) / thermodynamics.rs_drag
                + fourier.pk_interpolator()(0.1, z=0.).sum())

    reference = compute(0.673, 0.5)
    assert np.isfinite(reference)
    assert np.allclose(jax.jit(compute)(0.673, 0.5), reference, rtol=1e-10)


def test_the_served_sections_are_differentiable(served):
    """jit is not the whole ask: a gradient-based sampler differentiates through the emulator, and
    that goes through the analytic divisor too -- the growth ODE and the E&H formulae are applied
    at prediction, not only at training."""
    jax = pytest.importorskip('jax')

    def distance(h):
        cosmo = served.to_cosmology().clone(h=h, omega_cdm=POINT['omega_cdm'])
        return cosmo.get_background().luminosity_distance(0.5)

    gradient = jax.grad(distance)(0.673)
    assert np.isfinite(gradient)
    # D_L ~ 1/h at fixed physical densities, so the derivative is large and negative
    step = 1e-4
    assert np.isclose(gradient, (distance(0.673 + step) - distance(0.673 - step)) / (2 * step),
                      rtol=1e-3)


def test_a_thermodynamics_scalar_is_not_a_python_float(served):
    """`float()` on a tracer is a ConcretizationTypeError; a 0-d array behaves like a scalar
    everywhere a float would, eagerly included."""
    thermodynamics = served.to_cosmology().clone(**POINT).get_thermodynamics()
    assert not isinstance(thermodynamics.rs_drag, float)
    assert np.ndim(thermodynamics.rs_drag) == 0
    assert 90. < float(thermodynamics.rs_drag) < 180.


# ── the dark-energy basis ─────────────────────────────────────────────────────

def w0wa_space():
    return Space(bounds=dict(BOX, omega_b=(0.0219, 0.0228), w0_fld=(-1.1, -0.8), wa_fld=(-0.6, 0.2)))


def test_the_sum_of_the_dark_energy_pair_is_a_basis():
    emulator = FourierEmulator(fiducial(), w0wa_space(), k=K, z=Z, basis=['omega_cdm', 'omega_b', 'theta_MC_100', 'w0pwa'])
    assert emulator.w0pwa
    training = emulator.to_training(dict(POINT, w0_fld=-0.9, wa_fld=-0.3))
    assert 'wa_fld' not in training and np.isclose(training['w0pwa'], -1.2)
    back = emulator.from_training(training)
    assert np.isclose(back['wa_fld'], -0.3) and np.isclose(back['w0_fld'], -0.9)
    assert np.isclose(back['h'], POINT['h'], rtol=1e-6)      # theta round trip, same formula


def test_the_sum_is_bounded_by_a_logit_rather_than_by_an_edge():
    emulator = FourierEmulator(fiducial(), w0wa_space(), k=K, z=Z, basis=['w0pwa'])
    assert emulator.transforms() == {'w0pwa': 'logit_w0pwa'}
    space = emulator.training_space()
    assert 'w0pwa' in space.params
    assert space.transforms['w0pwa'] == 'logit_w0pwa'


def test_a_density_fraction_basis_is_expanded_in_the_log():
    """A fraction the basis introduces, from a space written in the physical densities: it is
    strictly positive on a box that is a plain rectangle, so a wide one crosses zero, and a
    negative density is a non-finite background rather than a slightly wrong one."""
    space = Space(bounds=dict(BOX, omega_b=(0.0219, 0.0228)))
    emulator = FourierEmulator(fiducial(), space, k=K, z=Z,
                               basis=['Omega_cdm', 'Omega_b', 'h'])
    assert emulator.transforms() == {'Omega_cdm': 'log', 'Omega_b': 'log'}


def test_a_transform_is_not_declared_for_a_pass_through_parameter():
    """A space already written in the fractions introduces nothing, and re-declaring a transform
    there would apply it twice -- the logit of a logit is nan."""
    space = Space(bounds=dict(Omega_cdm=(0.24, 0.28), Omega_b=(0.046, 0.052), h=(0.66, 0.70)))
    emulator = FourierEmulator(fiducial(), space, k=K, z=Z,
                               basis=['Omega_cdm', 'Omega_b', 'h'])
    assert emulator.transforms() == {}


def test_the_sum_needs_both_of_its_parameters():
    space = Space(bounds=dict(BOX, w0_fld=(-1.1, -0.8)))
    with pytest.raises(ValueError, match='reparametrisation'):
        FourierEmulator(fiducial(), space, k=K, z=Z, basis=['w0pwa'])


# ── the precision a Cl is worth emulating at ──────────────────────────────────

def test_harmonic_precision_turns_lensing_on_and_only_raises_ellmax():
    cosmo = Cosmology(engine='camb', ellmax_cl=3000)
    tuned = with_harmonic_precision(cosmo, of=('lensed_cl',), ellmax=2000)
    assert tuned['lensing'] is True
    assert tuned['non_linear'] == 'mead2016'
    assert tuned['ellmax_cl'] == 3000          # the request is smaller: not undercut
    assert with_harmonic_precision(cosmo, of=('lensed_cl',), ellmax=4000)['ellmax_cl'] == 4000


def test_the_lensing_potential_gets_the_reconstruction_boost():
    cosmo = Cosmology(engine='camb', ellmax_cl=500)
    tuned = with_harmonic_precision(cosmo, of=('lensed_cl', 'lens_potential_cl'), ellmax=2000)
    assert tuned._engine._extra_params['lens_potential_accuracy'] == 4
    # CAMB needs the reach for `lens_margin` to have room to work with
    assert tuned['ellmax_cl'] >= 4000


def test_harmonic_precision_leaves_an_unlensed_request_alone():
    cosmo = Cosmology(engine='camb', ellmax_cl=500)
    tuned = with_harmonic_precision(cosmo, of=('unlensed_cl',), ellmax=400)
    assert not tuned['lensing']
    assert tuned['ellmax_cl'] == 500
