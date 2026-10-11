"""Tests for the DISCO-EB engine.

The DISCO-EB engine wraps a JAX Einstein-Boltzmann solver.  Its (stiff, vmapped) perturbation
solve is heavy -- it is developed for GPU and needs several GB just to JIT-compile -- so the tests
are split in two groups:

* tests that exercise the engine wiring (parameter mapping, unit conversions, cross-spectra,
  sigma8 rescaling, transfer function, sections, state) using a *stubbed* DISCO-EB solver, which
  always run (they only need jax for the ``jit`` wrapper);
* tests that run the *real* DISCO-EB, which are opt-in: set the environment variable
  ``COSMOPRIMO_TEST_DISCOEB=1`` (and have DISCO-EB installed with enough memory / a GPU).
"""

import os
import importlib.util

import numpy as np
import pytest

from cosmoprimo import Cosmology, CosmologyInputError
from cosmoprimo.discoeb import DISCOEBEngine, DISCOEB_VARIABLES as IV, yhe_bbn


try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

_HAS_DISCOEB = importlib.util.find_spec('discoeb') is not None
_HAS_INTERPAX = importlib.util.find_spec('interpax') is not None

#: opt-in flag for the (heavy) real-DISCO-EB tests
_RUN_REAL = os.environ.get('COSMOPRIMO_TEST_DISCOEB', '0') == '1'

requires_jax = pytest.mark.skipif(not _HAS_JAX, reason='DISCO-EB engine requires jax')
requires_interpax = pytest.mark.skipif(
    not (_HAS_JAX and _HAS_INTERPAX), reason='cosmoprimo jax interpolator requires jax + interpax')
requires_real_discoeb = pytest.mark.skipif(
    not (_RUN_REAL and _HAS_DISCOEB),
    reason='set COSMOPRIMO_TEST_DISCOEB=1 and install DISCO-EB to run the real solver')


# Fiducial parameters, close to Planck/DESI (a single massive neutrino family, as DISCO-EB uses).
FID = dict(Omega_cdm=0.25, Omega_b=0.05, h=0.67742, A_s=2.1064e-9, n_s=0.96822,
           m_ncdm=[0.06], N_eff=3.044, T_cmb=2.7255)

#: Lightweight settings for the (opt-in, CPU-heavy) real-solver tests.  ``num_thermo=256`` is
#: DISCO-EB's own fast default (<0.13% error on P(k)); the dominant cost of these tests is the
#: JIT compilation of the *autodiff* transform, not the adaptive solver's step count, so there is
#: nothing to gain from touching ``max_steps`` (which is only a safety cap).
REAL_EXTRA = dict(num_thermo=256)


@pytest.fixture
def stub_discoeb(monkeypatch):
    """Replace DISCO-EB's solvers with lightweight, correctly-shaped stubs."""
    import cosmoprimo.discoeb as deb

    def stub_background(*, param, **kwargs):
        return dict(param)

    def stub_perturbations(*, param, aexp_out, kmin, kmax, num_k, **kwargs):
        kmodes = jnp.geomspace(kmin, kmax, num_k)
        nout = aexp_out.shape[0]
        y = jnp.zeros((num_k, nout, 20))
        y = y.at[:, :, IV['delta_m']].set(1.0)
        y = y.at[:, :, IV['delta_cb']].set(2.0)
        y = y.at[:, :, IV['theta_m']].set(-0.5)
        y = y.at[:, :, IV['theta_cb']].set(-0.5)
        return y, kmodes, param

    def stub_perturbations_batched(*, param, aexp_out, kmin, kmax, num_k, batch_size=16, **kwargs):
        y, kmodes, _ = stub_perturbations(param=param, aexp_out=aexp_out, kmin=kmin, kmax=kmax,
                                          num_k=num_k)
        return y, kmodes

    def stub_get_power(*, k, y, idx, param):
        return 2 * jnp.pi**2 * param['A_s'] * (k / param['k_p'])**(param['n_s'] - 1)\
            * k**(-3.) * y[..., idx]**2

    monkeypatch.setattr(deb, '_import_discoeb', lambda: (
        jax, jnp, stub_background, stub_perturbations, stub_perturbations_batched, stub_get_power))


# --------------------------------------------------------------------------------------------
# Tests that do not require running DISCO-EB at all (engine wiring).
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
# Tests using a stubbed DISCO-EB solver (need jax for the jit wrapper, but not the real solver).
# --------------------------------------------------------------------------------------------

@requires_jax
def test_perturbation_plumbing(stub_discoeb):
    cosmo = Cosmology(engine='discoeb',
                      extra_params={'nk': 64, 'kmin': 1e-4, 'kmax': 5., 'z': [0., 1., 2.]}, **FID)
    kmodes, y, z = cosmo.engine.get_discoeb_perturbations()
    h = FID['h']
    assert y.shape == (64, 3, 20)
    assert np.allclose(z, [0., 1., 2.])
    # the k-grid is log-spaced in 1/Mpc between kmin*h and kmax*h
    assert np.isclose(kmodes[0], 1e-4 * h) and np.isclose(kmodes[-1], 5. * h)


@requires_jax
def test_fourier_units_and_spectra(stub_discoeb):
    cosmo = Cosmology(engine='discoeb',
                      extra_params={'nk': 128, 'kmin': 1e-4, 'kmax': 5., 'z': [0., 1., 2.]}, **FID)
    h = FID['h']
    fo = cosmo.get_fourier()
    kmodes, y, z = cosmo.engine.get_discoeb_perturbations()

    ka, za, pka = fo.table(of='delta_m')
    assert np.allclose(ka, kmodes / h)                    # k: 1/Mpc -> h/Mpc
    assert np.allclose(za, z)
    # delta_m = 1 => P = 2 pi^2 A_s (k/k_p)^(n_s-1) k^-3, converted to (Mpc/h)^3 (x h^3)
    A_s, n_s, k_p = (cosmo.engine._discoeb_params[name] for name in ['A_s', 'n_s', 'k_p'])
    prim = 2 * np.pi**2 * A_s * (kmodes / k_p)**(n_s - 1) * kmodes**(-3.)
    assert np.allclose(pka[:, 0], prim * h**3)

    # delta_cb = 2 => 4x the delta_m spectrum; the cross => 2x
    _, _, pk_cb = fo.table(of='delta_cb')
    _, _, pk_x = fo.table(of=('delta_m', 'delta_cb'))
    assert np.allclose(pk_cb[:, 0], 4. * pka[:, 0])
    assert np.allclose(pk_x[:, 0], 2. * pka[:, 0])

    # interpolator, pk_kz, sigma_rz, sigma8 are all wired up
    interp = fo.pk_interpolator(of='delta_m')
    kk = np.logspace(-2, 0, 8)
    assert np.allclose(interp(kk, z=0.), fo.pk_kz(kk, z=0., of='delta_m'))
    assert np.all(np.isfinite(fo.sigma_rz([8.], [0., 1.], of='delta_m')))
    assert np.isfinite(fo.sigma8_m) and np.isfinite(fo.sigma8_cb)
    # theta_m = -0.5 delta_m => P_theta_theta = 0.25 P_delta_delta
    _, _, pk_th = fo.table(of='theta_m')
    assert np.allclose(pk_th[:, 0], 0.25 * pka[:, 0])


@requires_jax
def test_transfer(stub_discoeb):
    cosmo = Cosmology(engine='discoeb',
                      extra_params={'nk': 128, 'kmin': 1e-4, 'kmax': 5., 'z': [0., 1.]}, **FID)
    tr = cosmo.get_transfer()
    k = np.logspace(-3, 0.5, 20)
    t = tr.transfer_k(k)
    assert np.all(np.isfinite(t))
    # normalised to unity on the largest scales
    assert np.isclose(tr.transfer_k([1e-4])[0], 1., rtol=1e-3)


@requires_jax
def test_sigma8_rescaling(stub_discoeb):
    # with sigma8 as input, the (linear-in-A_s) spectrum is rescaled to match it exactly
    params = {k: v for k, v in FID.items() if k != 'A_s'}
    cosmo = Cosmology(engine='discoeb', sigma8=0.81,
                      extra_params={'nk': 128, 'kmin': 1e-4, 'kmax': 5., 'z': [0., 1.]}, **params)
    fo = cosmo.get_fourier()
    assert np.isclose(fo.sigma8_m, 0.81, rtol=1e-5)
    assert np.isclose(cosmo.get_primordial().A_s / cosmo.engine._A_s, cosmo.engine._rsigma8**2)


@requires_jax
def test_batched(stub_discoeb):
    cosmo = Cosmology(engine='discoeb',
                      extra_params={'nk': 32, 'batched': True, 'batch_size': 8, 'z': [0., 1.]},
                      **FID)
    kmodes, y, z = cosmo.engine.get_discoeb_perturbations()
    assert y.shape == (32, 2, 20)


# --------------------------------------------------------------------------------------------
# Gradients / Jacobians.  DISCO-EB is autodifferentiable, and the engine keeps the whole
# params -> P(k) chain traceable, so cosmoprimo's standard jax pattern works:
#   jax.jacfwd(lambda params: Cosmology(**params, engine='discoeb').get_fourier().table(...))
# These tests use a parameter-dependent stub so the derivatives are known analytically.
# --------------------------------------------------------------------------------------------

@pytest.fixture
def stub_discoeb_diff(monkeypatch):
    """Stub solver whose delta_m depends on Omegam, so P(k) ∝ A_s * Omegam^2 * h^3."""
    import cosmoprimo.discoeb as deb

    def stub_background(*, param, **kwargs):
        return dict(param)

    def stub_perturbations(*, param, aexp_out, kmin, kmax, num_k, **kwargs):
        kmodes = jnp.geomspace(kmin, kmax, num_k)
        nout = aexp_out.shape[0]
        dm = param['Omegam'] * jnp.ones((num_k, nout))
        y = jnp.zeros((num_k, nout, 20))
        y = y.at[:, :, IV['delta_m']].set(dm)
        y = y.at[:, :, IV['delta_cb']].set(dm)
        return y, kmodes, param

    def stub_perturbations_batched(*, param, aexp_out, kmin, kmax, num_k, batch_size=16, **kwargs):
        y, kmodes, _ = stub_perturbations(param=param, aexp_out=aexp_out, kmin=kmin, kmax=kmax,
                                          num_k=num_k)
        return y, kmodes

    monkeypatch.setattr(deb, '_import_discoeb', lambda: (
        jax, jnp, stub_background, stub_perturbations, stub_perturbations_batched, None))


@requires_jax
def test_jacobian_analytic(stub_discoeb_diff):
    extra = {'nk': 12, 'kmin': 1e-3, 'kmax': 0.5, 'z': [0.]}
    p0 = dict(Omega_cdm=0.25, Omega_b=0.05, h=0.67742, A_s=2.1064e-9, n_s=0.96822, N_eff=3.044)

    def pk_of(params):
        cosmo = Cosmology(engine='discoeb', extra_params=extra, **params)
        return cosmo.get_fourier().table(of='delta_m')[2][:, 0]

    pk0 = np.asarray(pk_of(p0))
    Omegam = p0['Omega_cdm'] + p0['Omega_b']
    jac = jax.jacfwd(pk_of)(p0)

    # P = 2 pi^2 A_s (k/k_p)^(n_s-1) k^-3 * Omegam^2 * h^3  (delta_m = Omegam in the stub)
    assert np.allclose(np.asarray(jac['Omega_cdm']), 2. * pk0 / Omegam, rtol=1e-6)
    assert np.allclose(np.asarray(jac['Omega_b']), 2. * pk0 / Omegam, rtol=1e-6)
    assert np.allclose(np.asarray(jac['A_s']), pk0 / p0['A_s'], rtol=1e-6)
    # d/dn_s = P * ln(k/k_p); recover the k-grid (in 1/Mpc) from the engine
    kdeb = np.asarray(Cosmology(engine='discoeb', extra_params=extra,
                                **p0).engine.get_discoeb_perturbations()[0])
    assert np.allclose(np.asarray(jac['n_s']), pk0 * np.log(kdeb / p0.get('k_pivot', 0.05)),
                       rtol=1e-5)


@requires_jax
def test_grad_finite_difference_and_h(stub_discoeb_diff):
    extra = {'nk': 12, 'kmin': 1e-3, 'kmax': 0.5, 'z': [0.]}
    p0 = dict(Omega_cdm=0.25, Omega_b=0.05, h=0.67742, A_s=2.1064e-9, n_s=0.96822, N_eff=3.044)

    def pk_of(params):
        cosmo = Cosmology(engine='discoeb', extra_params=extra, **params)
        return cosmo.get_fourier().table(of='delta_m')[2][:, 0]

    jac = jax.jacfwd(pk_of)(p0)
    # h enters both the unit conversion (h^3) and the k-grid (k[1/Mpc] = k[h/Mpc]*h)
    eps = 1e-4
    fd_h = (np.asarray(pk_of(dict(p0, h=p0['h'] + eps)))
            - np.asarray(pk_of(dict(p0, h=p0['h'] - eps)))) / (2 * eps)
    assert np.allclose(np.asarray(jac['h']), fd_h, rtol=1e-3)
    # jax.grad of a scalar summary
    g = jax.grad(lambda p: jnp.sum(pk_of(p)))(p0)
    pk0 = np.asarray(pk_of(p0))
    Omegam = p0['Omega_cdm'] + p0['Omega_b']
    assert np.isclose(float(g['Omega_cdm']), 2. * pk0.sum() / Omegam, rtol=1e-6)


@requires_interpax
def test_jacobian_through_cosmoprimo_interpolator(stub_discoeb_diff):
    # Gradients also flow through cosmoprimo's OWN pk interpolator (jax-friendly via interpax):
    # DISCO-EB's internal splines are just traced as part of the solve and never surface here.
    extra = {'nk': 24, 'kmin': 1e-3, 'kmax': 1.0, 'z': [0., 1.]}
    p0 = dict(Omega_cdm=0.25, Omega_b=0.05, h=0.67742, A_s=2.1064e-9, n_s=0.96822, N_eff=3.044)
    Omegam = p0['Omega_cdm'] + p0['Omega_b']
    kq = np.array([0.05, 0.1, 0.2])

    def pk(params):
        return Cosmology(engine='discoeb', extra_params=extra, **params).get_fourier()\
            .pk_interpolator(of='delta_m')(kq, z=0.)

    val = np.asarray(pk(p0))
    jac = jax.jacfwd(pk)(p0)
    # P ∝ Omegam^2 (stub), so dP/dOmega_cdm = 2P/Omegam -- through the interpax spline
    assert np.allclose(np.asarray(jac['Omega_cdm']), 2. * val / Omegam, rtol=1e-3)



# --------------------------------------------------------------------------------------------
# Tests running the real DISCO-EB (opt-in: COSMOPRIMO_TEST_DISCOEB=1).
# --------------------------------------------------------------------------------------------

@requires_real_discoeb
def test_thermodynamics_real():
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


@requires_real_discoeb
def test_fourier_real_vs_eisenstein_hu():
    # compare DISCO-EB (a full Boltzmann solve) to the Eisenstein & Hu analytic fit, for a
    # massless-neutrino cosmology so that both engines describe the same model.
    params = {k: v for k, v in FID.items() if k != 'm_ncdm'}
    cosmo = Cosmology(engine='discoeb',
                      extra_params=dict(nk=128, z=[0., 1.], **REAL_EXTRA), **params)
    fo = cosmo.get_fourier()
    fo_eh = Cosmology(engine='eisenstein_hu', **params).get_fourier()
    k = np.logspace(-2.5, 0., 20)
    pk = fo.pk_interpolator(of='delta_m')(k, z=0.)
    pk_eh = fo_eh.pk_interpolator(of='delta_m')(k, z=0.)
    # the analytic EH fit is itself only good to ~10-15% against a Boltzmann code
    assert np.allclose(pk, pk_eh, rtol=0.2)
    assert np.isfinite(fo.sigma8_m) and 0.5 < fo.sigma8_m < 1.1
    # cross-spectrum and theta are consistent with delta (P_theta_theta = f^2 P_delta_delta)
    pk_th = fo.pk_interpolator(of='theta_m')(k, z=0.)
    assert np.all(pk_th > 0.) and np.all(pk_th < pk)


@requires_real_discoeb
def test_gradients_real():
    # DISCO-EB's headline feature: autodifferentiable P(k).  Compare jax autodiff against
    # finite differences for the real solver (mirrors DISCO-EB's nb_discoeb_autodiff notebook).
    import jax

    def pk_of(Omega_cdm):
        # Tight solver tolerances: autodiff through an *adaptive* ODE solver inherits the
        # step-size-controller's noise, which at rtol = 1e-3 reaches ~10% on one k (against a
        # ~3e-4 median) and is exactly the effect DISCO-EB's autodiff notebook warns about
        # ("increase accuracy ... to reduce noise on derivatives").
        cosmo = Cosmology(engine='discoeb',
                          extra_params=dict(nk=32, kmin=1e-3, kmax=1., z=[0.],
                                            rtol=1e-4, atol=1e-4, **REAL_EXTRA),
                          Omega_cdm=Omega_cdm, Omega_b=0.05, h=0.67742, A_s=2.1064e-9,
                          n_s=0.96822, N_eff=3.044, T_cmb=2.7255)
        return cosmo.get_fourier().table(of='delta_m')[2][:, 0]

    x = 0.25
    g_auto = np.asarray(jax.jacfwd(pk_of)(x))
    eps = 1e-3
    g_fd = (np.asarray(pk_of(x + eps)) - np.asarray(pk_of(x - eps))) / (2 * eps)
    # autodiff (forward mode, DISCO-EB's DirectAdjoint) should match finite differences
    assert np.allclose(g_auto, g_fd, rtol=0.05, atol=0.)
    assert np.all(np.abs(g_auto) > 0.)   # the derivative is non-trivial


