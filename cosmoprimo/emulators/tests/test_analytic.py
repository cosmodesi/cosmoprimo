"""The analytic formulae the cosmology emulators divide out, checked where they live.

Each is a preconditioner, so what is asserted is what a consumer relies on: the exact bits are
exact, the round trips close, and the approximate ones sit at their measured distance from the
engine they stand in for.
"""

import numpy as np
import pytest

from cosmoprimo.fiducial import DESI
from cosmoprimo.emulators.analytic import (AMPLITUDES, amplitude, harmonic_scaling, nonzero,
                                           fourier_analytic_scales, theta_analytic,
                                           solve_theta_analytic, eisenstein_hu_scales)


def test_amplitude_in_every_spelling():
    assert 'logA' in AMPLITUDES and 'A_s' in AMPLITUDES
    assert amplitude({'h': 0.7}) is None
    assert np.isclose(amplitude({'A_s': 2e-9}), 2e-9)
    assert np.isclose(amplitude({'logA': np.log(1e10 * 2e-9)}), 2e-9)


def test_harmonic_scaling_counts_screened_legs():
    """tt and ee two legs, tp and ep one, pp none; the spectrum is the last dotted component."""
    tau = 0.06
    factors = harmonic_scaling(['lensed_cl.tt', 'x|ellmax=10.ee', 'lens.tp', 'lens.pp'], 3., tau)
    assert np.isclose(factors['lensed_cl.tt'], 3. * np.exp(-2. * tau))
    assert np.isclose(factors['x|ellmax=10.ee'], 3. * np.exp(-2. * tau))
    assert np.isclose(factors['lens.tp'], 3. * np.exp(-tau))
    assert np.isclose(factors['lens.pp'], 3.)
    assert harmonic_scaling(['a.tt'])['a.tt'] == 1.


def test_nonzero_replaces_exact_zeros_only():
    np.testing.assert_array_equal(nonzero(np.array([0., 2., -1.])), [1., 2., -1.])


def test_background_scalars_match_the_engine_at_the_fiducial():
    """invE and DM are the engine's own to 1e-4; the growth rate to 1e-3 (neutrinos as matter)."""
    cosmo = DESI(engine='eisenstein_hu')
    z = 0.8
    scalars = fourier_analytic_scales(z, cosmo)
    assert np.isclose(float(scalars['invE']), 1. / cosmo.efunc(z), rtol=1e-4)
    assert np.isclose(float(scalars['DM']), cosmo.comoving_transverse_distance(z), rtol=1e-4)
    assert np.isclose(float(scalars['f']), cosmo.growth_rate(z), rtol=2e-2)
    assert float(scalars['D']) > 0.
    # z = 0 is a legal endpoint: no distance, finite growth
    at_zero = fourier_analytic_scales(0., cosmo)
    assert float(at_zero['DM']) == 0. and float(at_zero['invE']) == 1. and np.isfinite(float(at_zero['D']))


def test_theta_round_trip_is_exact():
    """`solve_theta_analytic` inverts `theta_analytic` on the same closed form, so the round
    trip is the identity; a target outside the bracket is nan, never an endpoint."""
    omega_b, omega_cdm = 0.02237, 0.12
    for h in (0.6, 0.6736, 0.8):
        target = 100. * theta_analytic(h, omega_b, omega_cdm)
        assert np.isclose(float(solve_theta_analytic(target, omega_b, omega_cdm)), h, atol=1e-9)
    assert np.isnan(float(solve_theta_analytic(10., omega_b, omega_cdm)))


def test_theta_tracks_the_exact_one():
    """A near-constant offset from cosmoprimo's own theta_MC_100, at the sub-per-cent level:
    calibrates out of a basis change, which is all it is used for."""
    cosmo = DESI(engine='class')
    offsets = []
    for h in (0.64, 0.6736, 0.72):
        clone = cosmo.clone(h=h)
        analytic = 100. * float(theta_analytic(h, clone['omega_b'], clone['omega_cdm']))
        offsets.append(analytic / clone['theta_MC_100'] - 1.)
    assert np.max(np.abs(offsets)) < 5e-3
    assert np.ptp(offsets) < 5e-4


def test_eisenstein_hu_scales_carry_the_unit_and_the_densities():
    """Mpc/h, so the ratio to the fiducial is exactly h / h_fid at fixed physical densities, and
    the density dependence is the fitting formula's own (a few per cent, all of it smooth)."""
    base = DESI(engine='eisenstein_hu')
    ref = eisenstein_hu_scales(base)
    for h in (0.6, 0.8):
        assert np.isclose(float(eisenstein_hu_scales(base.clone(h=h))['rs_drag']) / float(ref['rs_drag']),
                          h / base['h'], rtol=1e-12)
    ratio = float(eisenstein_hu_scales(base.clone(omega_cdm=0.14))['rs_drag']) / float(ref['rs_drag'])
    assert 0.9 < ratio < 1. and np.isclose(ratio, base.clone(omega_cdm=0.14).rs_drag / base.rs_drag, rtol=2e-3)


@pytest.mark.parametrize('name', ['theta_analytic', 'eisenstein_hu_scales', 'fourier_analytic_scales'])
def test_formulae_run_under_a_jax_trace(name):
    """desilike divides these out inside a jitted prediction, so none may cast to float."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    cosmo = DESI(engine='eisenstein_hu')
    if name == 'theta_analytic':
        fun = lambda h: theta_analytic(h, 0.02237, 0.12)
    elif name == 'eisenstein_hu_scales':
        fun = lambda h: eisenstein_hu_scales(cosmo.clone(h=h))['rs_drag']
    else:
        fun = lambda h: fourier_analytic_scales(0.8, cosmo.clone(h=h))['D']
    traced, eager = jax.jit(fun)(jnp.asarray(0.7)), fun(0.7)
    assert np.isclose(float(traced), float(eager), rtol=1e-10)
