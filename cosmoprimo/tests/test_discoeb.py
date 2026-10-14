"""Tests for the DISCO-EB engine.

The DISCO-EB engine wraps a JAX Einstein-Boltzmann solver, so these tests run the *real* solver
by default; they are only skipped when DISCO-EB (or jax/interpax) is not importable.

The solve is stiff and vmapped, and its dominant cost in a test session is the JIT compilation of
the autodiff transform rather than the number of adaptive steps, so the settings below are kept
deliberately small (``nk``, ``num_thermo``) to stay fast on a laptop while remaining accurate at
the ~0.1% level on P(k).
"""

import importlib.util

import numpy as np
import pytest

from cosmoprimo import Cosmology, CosmologyInputError
from cosmoprimo.discoeb import DISCOEBEngine, yhe_bbn


try:
    import jax
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

_HAS_DISCOEB = importlib.util.find_spec('discoeb') is not None
_HAS_INTERPAX = importlib.util.find_spec('interpax') is not None

requires_discoeb = pytest.mark.skipif(
    not (_HAS_JAX and _HAS_DISCOEB and _HAS_INTERPAX),
    reason='the DISCO-EB engine requires jax, interpax and DISCO-EB to be installed')


# Fiducial parameters, close to Planck/DESI (a single massive neutrino family, as DISCO-EB uses).
FID = dict(Omega_cdm=0.25, Omega_b=0.05, h=0.67742, A_s=2.1064e-9, n_s=0.96822,
           m_ncdm=[0.06], N_eff=3.044, T_cmb=2.7255)

#: Lightweight settings shared by the solver tests.  ``num_thermo=256`` is DISCO-EB's own fast
#: default (<0.13% on P(k)).  ``max_steps`` is only a safety cap (a compile-time constant), not the
#: number of steps actually taken, so it is not a knob for test speed: the knobs are ``nk`` and
#: ``num_thermo`` below.
REAL_EXTRA = dict(num_thermo=256)

#: Common perturbation grid for the Fourier tests.  Sharing it (and hence the static solver
#: settings that key the module-level JIT cache) means the solve is compiled once for the whole
#: session.
FOURIER_EXTRA = dict(nk=32, kmin=1e-4, kmax=5., z=[0., 1.], **REAL_EXTRA)


# --------------------------------------------------------------------------------------------
# Tests that do not run DISCO-EB at all (engine wiring).
# --------------------------------------------------------------------------------------------

def test_registered():
    from cosmoprimo.cosmology import get_engine
    for name in ['discoeb', 'disco_eb', 'disco-eb', 'DISCOEB']:
        assert get_engine(name) is DISCOEBEngine


def test_params_mapping():
    cosmo = Cosmology(engine='discoeb', **FID)
    p = cosmo.engine._discoeb_params
    h = FID['h']
    # DISCO-EB's Omegam is cdm + baryon (massive neutrinos are handled through mnu/Nmnu)
    assert np.isclose(p['Omegam'], FID['Omega_cdm'] + FID['Omega_b'])
    assert np.isclose(p['Omegab'], FID['Omega_b'])
    assert np.isclose(p['H0'], 100. * h)
    assert np.isclose(p['Tcmb'], FID['T_cmb'])
    assert np.isclose(p['n_s'], FID['n_s'])
    assert np.isclose(p['A_s'], FID['A_s'])
    assert np.isclose(p['k_p'], cosmo['k_pivot'])
    # neutrinos: one degenerate massive family + the remaining massless species
    assert p['Nmnu'] == 1
    assert np.isclose(p['mnu'], 0.06)
    assert np.isclose(p['Neff'], FID['N_eff'] - 1)
    # helium from BBN consistency
    assert 0.24 < p['YHe'] < 0.26
    # dark energy / curvature
    assert np.isclose(p['w_DE_0'], -1.) and np.isclose(p['w_DE_a'], 0.)
    assert np.isclose(p['Omegak'], 0.)


def test_yhe_bbn():
    # BBN helium mass fraction, ~0.245 for Planck baryon density
    assert np.isclose(yhe_bbn(0.0222, N_eff=3.044), 0.2453, atol=2e-3)
    # increases with the baryon density
    assert yhe_bbn(0.024, N_eff=3.044) > yhe_bbn(0.020, N_eff=3.044)


def test_background():
    cosmo = Cosmology(engine='discoeb', **FID)
    ba = cosmo.get_background()
    assert np.isclose(ba.H0, FID['h'] * 100.)
    assert np.isclose(ba.Omega0_b, FID['Omega_b'])
    # distances, growth, age are all provided by the (well-tested) DefaultBackground
    assert ba.comoving_radial_distance(1.) > 0.
    assert 0. < ba.growth_factor(1.) < 1.
    assert 0. < ba.growth_rate(0.) < 1.
    assert 13. < ba.age < 14.5


def test_primordial():
    cosmo = Cosmology(engine='discoeb', **FID)
    pm = cosmo.get_primordial()
    assert np.isclose(pm.A_s, FID['A_s'])
    assert np.isclose(pm.n_s, FID['n_s'])
    assert pm.alpha_s == 0. and pm.beta_s == 0.     # DISCO-EB has no running
    assert np.isclose(pm.k_pivot, cosmo['k_pivot'] / FID['h'])
    k = np.logspace(-3, 0, 10)
    assert np.allclose(pm.pk_interpolator()(k), pm.pk_k(k), rtol=1e-3)


def test_non_linear_raises():
    with pytest.raises(CosmologyInputError):
        Cosmology(engine='discoeb', non_linear='mead', **{k: v for k, v in FID.items()})


def test_state_and_clone():
    cosmo = Cosmology(engine='discoeb', extra_params={'nk': 128, 'rtol': 1e-4}, **FID)
    state = cosmo.__getstate__()
    assert state['engine']['name'] == 'discoeb'
    assert state['engine']['extra_params']['nk'] == 128
    new = Cosmology.__new__(Cosmology)
    new.__setstate__(state)
    assert new.engine.name == 'discoeb'
    assert new.engine._extra_params['nk'] == 128
    assert np.isclose(new.engine._discoeb_params['Omegam'], FID['Omega_cdm'] + FID['Omega_b'])
    clone = cosmo.clone()
    assert clone.engine.name == 'discoeb' and clone.engine._extra_params['nk'] == 128


# --------------------------------------------------------------------------------------------
# Tests running the real DISCO-EB solver (baseline).
# --------------------------------------------------------------------------------------------

@requires_discoeb
def test_perturbation_plumbing():
    cosmo = Cosmology(engine='discoeb', extra_params=dict(FOURIER_EXTRA), **FID)
    kmodes, y, z = cosmo.engine.get_discoeb_perturbations()
    h = FID['h']
    assert y.shape == (32, 2, 20)
    assert np.allclose(z, [0., 1.])
    assert np.all(np.isfinite(y))
    # the k-grid is log-spaced in 1/Mpc between kmin*h and kmax*h
    assert np.isclose(kmodes[0], 1e-4 * h) and np.isclose(kmodes[-1], 5. * h)


@requires_discoeb
def test_fourier_units_and_spectra():
    cosmo = Cosmology(engine='discoeb', extra_params=dict(FOURIER_EXTRA), **FID)
    h = FID['h']
    fo = cosmo.get_fourier()
    kmodes, y, z = cosmo.engine.get_discoeb_perturbations()

    ka, za, pka = fo.table(of='delta_m')
    assert np.allclose(ka, kmodes / h)                    # k: 1/Mpc -> h/Mpc
    assert np.allclose(za, z)
    assert np.all(np.isfinite(pka)) and np.all(pka > 0.)

    # interpolating the table at its own nodes reproduces it exactly, and the two public entry
    # points agree
    assert np.allclose(fo.pk_interpolator(of='delta_m')(ka, z=0.), pka[:, 0], rtol=1e-6)
    kk = np.logspace(-2, 0, 8)
    assert np.allclose(fo.pk_interpolator(of='delta_m')(kk, z=0.), fo.pk_kz(kk, z=0., of='delta_m'))

    # sigma(r, z) and sigma8 are wired up and in the right ballpark
    assert np.all(np.isfinite(fo.sigma_rz([8.], [0., 1.], of='delta_m')))
    assert np.isfinite(fo.sigma8_m) and 0.5 < fo.sigma8_m < 1.1
    assert np.isfinite(fo.sigma8_cb) and 0.5 < fo.sigma8_cb < 1.1


@requires_discoeb
def test_transfer():
    cosmo = Cosmology(engine='discoeb', extra_params=dict(FOURIER_EXTRA), **FID)
    tr = cosmo.get_transfer()
    k = np.logspace(-3, 0.5, 20)
    t = tr.transfer_k(k)
    assert np.all(np.isfinite(t)) and np.all(t > 0.)
    # normalised to unity on the largest scales
    assert np.isclose(tr.transfer_k([1e-4])[0], 1., rtol=1e-3)


@requires_discoeb
def test_sigma8_rescaling():
    # with sigma8 as input, the (linear-in-A_s) spectrum is rescaled to match it exactly
    params = {k: v for k, v in FID.items() if k != 'A_s'}
    cosmo = Cosmology(engine='discoeb', sigma8=0.81, extra_params=dict(FOURIER_EXTRA), **params)
    fo = cosmo.get_fourier()
    assert np.isclose(fo.sigma8_m, 0.81, rtol=1e-5)
    assert np.isclose(cosmo.get_primordial().A_s / cosmo.engine._A_s, cosmo.engine._rsigma8**2)


@requires_discoeb
def test_batched():
    # Regression test for the DISCO-EB batched solver with more than one output time: the raw
    # ``SaveAt`` output has the time axis first, so a plain reshape mixes the time and mode axes
    # and only happens to be right when there is a single output time.
    params = dict(FID)
    extra = dict(FOURIER_EXTRA)
    c_serial = Cosmology(engine='discoeb', extra_params=dict(extra, batched=False), **params)
    c_batch = Cosmology(engine='discoeb',
                        extra_params=dict(extra, batched=True, batch_size=8), **params)
    k_s, z_s, pk_s = c_serial.get_fourier().table(of='delta_m')
    k_b, z_b, pk_b = c_batch.get_fourier().table(of='delta_m')
    assert pk_s.shape == (32, 2)
    assert np.allclose(k_s, k_b) and np.allclose(z_s, z_b)
    # batched and serial must agree: the batched vmap only changes the adaptive step control
    # (a few % at the smallest scales here), whereas the axis-ordering bug this guards against
    # moved the z=1 column to the z=0 solution (~2.7x).
    assert np.allclose(pk_s, pk_b, rtol=0.05)


@requires_discoeb
def test_thermodynamics():
    cosmo = Cosmology(engine='discoeb', extra_params=dict(REAL_EXTRA), **FID)
    th = cosmo.get_thermodynamics()
    # recombination / drag epochs and sound horizon, against the eisenstein_hu engine
    ceh = Cosmology(engine='eisenstein_hu', **FID)
    theh = ceh.get_thermodynamics()
    assert 1050. < th.z_drag < 1090.
    assert 1060. < th.z_star < 1120.
    assert np.isclose(th.rs_drag, theh.rs_drag, rtol=2e-2)
    assert np.isclose(th.rs_drag / cosmo.get_background().rs(th.z_drag), 1., rtol=2e-2)
    assert 0.008 < th.theta_star < 0.011
    assert 0. < th.x_e(0.) < 1e-3 and th.x_e(1100.) > 0.01


@requires_discoeb
def test_fourier_vs_eisenstein_hu():
    # compare DISCO-EB (a full Boltzmann solve) to the Eisenstein & Hu analytic fit, for a
    # massless-neutrino cosmology so that both engines describe the same model.
    params = {k: v for k, v in FID.items() if k != 'm_ncdm'}
    cosmo = Cosmology(engine='discoeb',
                      extra_params=dict(nk=32, z=[0., 1.], **REAL_EXTRA), **params)
    fo = cosmo.get_fourier()
    fo_eh = Cosmology(engine='eisenstein_hu', **params).get_fourier()
    k = np.logspace(-2.5, 0., 20)
    pk = fo.pk_interpolator(of='delta_m')(k, z=0.)
    pk_eh = fo_eh.pk_interpolator(of='delta_m')(k, z=0.)
    # the analytic EH fit is itself only good to ~10-15% against a Boltzmann code
    assert np.allclose(pk, pk_eh, rtol=0.2)
    assert np.isfinite(fo.sigma8_m) and 0.5 < fo.sigma8_m < 1.1


# --------------------------------------------------------------------------------------------
# Gradients / Jacobians.  DISCO-EB is autodifferentiable and the engine keeps the whole
# params -> P(k) chain traceable.  The important test is the *interpolated* spectrum, which is
# what a user actually evaluates: it goes through cosmoprimo's own interpax-based interpolator on
# top of the solver.  The Cosmology (and hence the interpolator) must be built inside the traced
# function; reusing an interpolator built from concrete parameters would naturally give a
# identically zero derivative.
# --------------------------------------------------------------------------------------------

@requires_discoeb
def test_gradients_real_interpolator():
    extra = dict(nk=32, kmin=1e-3, kmax=1., z=[0.], rtol=1e-4, atol=1e-4, **REAL_EXTRA)
    base = dict(Omega_b=0.05, h=0.67742, A_s=2.1064e-9, n_s=0.96822, N_eff=3.044,
                T_cmb=2.7255)
    kq = np.geomspace(1e-3, 1., 8)

    def pk_interp(Omega_cdm):
        cosmo = Cosmology(engine='discoeb', extra_params=extra, Omega_cdm=Omega_cdm, **base)
        return cosmo.get_fourier().pk_interpolator(of='delta_m')(kq, z=0.)

    x = 0.25
    g_auto = np.asarray(jax.jacfwd(pk_interp)(x))
    # the derivative must be non-trivial: this is exactly what is lost if the interpolator is
    # built once from concrete parameters and reused inside the traced function
    assert np.all(np.isfinite(g_auto))
    assert np.any(g_auto != 0.)

    eps = 1e-3
    g_fd = (np.asarray(pk_interp(x + eps)) - np.asarray(pk_interp(x - eps))) / (2 * eps)
    # autodiff (forward mode, DISCO-EB's DirectAdjoint) should match finite differences
    assert np.allclose(g_auto, g_fd, rtol=0.05, atol=0.)


@requires_discoeb
def test_gradients_real_table():
    # Same comparison for the raw table, i.e. without cosmoprimo's interpolator in between.
    extra = dict(nk=32, kmin=1e-3, kmax=1., z=[0.], rtol=1e-4, atol=1e-4, **REAL_EXTRA)
    base = dict(Omega_b=0.05, h=0.67742, A_s=2.1064e-9, n_s=0.96822, N_eff=3.044,
                T_cmb=2.7255)

    def pk_table(Omega_cdm):
        cosmo = Cosmology(engine='discoeb', extra_params=extra, Omega_cdm=Omega_cdm, **base)
        return cosmo.get_fourier().table(of='delta_m')[2][:, 0]

    x = 0.25
    g_auto = np.asarray(jax.jacfwd(pk_table)(x))
    eps = 1e-3
    g_fd = (np.asarray(pk_table(x + eps)) - np.asarray(pk_table(x - eps))) / (2 * eps)
    assert np.all(np.isfinite(g_auto))
    assert np.allclose(g_auto, g_fd, rtol=0.05, atol=0.)
