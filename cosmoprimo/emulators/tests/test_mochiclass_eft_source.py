"""The closed-form quasi-static source (``mochiclass.eft_finite_coefficients`` / ``eft_interpolators``):
agreement with mochi_class' own finite coefficients, the no-slip identification, exact GR, shapes, jax.

The reference values are mochi_class' ``Background._eft_finite_coefficients`` (cosmoprimo's 'mochiclass'
engine on the DESI fiducial with the DESI-DR2-MG pipeline's EFT inputs: Omega_smg = -1, expansion_model
'wowa', w0 = -1, wa = 0, pert_ic_tolerance_smg = -1), produced 2026-11-03 by
``CosmoPolyEmu/scripts/check_eft_source.py``'s setup in the DESI-DR2-MG project; pinned here so that the test
needs no ``pyclass.mochiclass``. The times avoid z < 0.05, where mochi_class' own ``cs2num`` column departs from
its own ingredients (its table edge; 2.6e-4 of its scale at z = 0 on a propto_omega model).
"""

import numpy as np
import pytest

from cosmoprimo.emulators import mochiclass as mcs

ETA = np.array([-3.9, -2.5, -1.2, -0.6, -0.26, -0.05])
#: the background the engine ran with (cosmoprimo's m_ncdm round trip through omega_ncdm included)
BG = dict(omega_b=0.0224, w0=-1., wa=0., m_ncdm=0.059999919304831895, N_ur=2.0328, T_cmb=2.7255, T_ncdm=0.71611)

#: hill_valley no-slip, parameters_smg = [alpha_K, c_M, tau, a_t, r, M2_ini], at h = 0.71, omega_cdm = 0.126
HV = ([0.1, -0.6, 1., 0.5, 2., 1.], dict(BG, h=0.71, omega_cdm=0.126))
HV_TRUTH = {'h1': [9.1416203237e-01, 7.4772635444e-01, 5.6946561800e-01, 5.4952532415e-01, 5.6399229111e-01, 5.8158748813e-01],
            'D0': [2.6608450727e-06, 2.0610089142e-06, 9.9735944527e-07, 4.0569617909e-07, 1.7572764867e-07, 8.9317721815e-08],
            'N3': [3.6871417332e-01, 1.2237153364e+00, 2.0657733516e+00, 1.5047224370e+00, 9.4483091331e-01, 6.3064148652e-01]}

#: propto_omega with free braiding, parameters_smg = [alpha_K, c_b, c_m, c_t, M2_ini], at h = 0.7674, omega_cdm = 0.1365
PO = ([0.1, 1.06, 0.81, 0., 1.], dict(BG, h=0.7674, omega_b=0.02236, omega_cdm=0.1365))
PO_TRUTH = {'h1': [9.9999407687e-01, 9.9960008232e-01, 9.8104076543e-01, 9.0544710662e-01, 8.0497742801e-01, 7.2350403560e-01],
            'D0': [3.5854669363e-10, 5.8883576767e-09, 7.1381853097e-08, 1.5239643293e-07, 1.6131377671e-07, 1.4313481364e-07],
            'N3': [1.1220846390e-04, 7.5248749854e-03, 3.3125402296e-01, 1.2256307842e+00, 1.7020391347e+00, 1.7639792426e+00],
            'N5': [1.1221021285e-04, 7.5327675268e-03, 3.4808723986e-01, 1.5659783180e+00, 2.7974515329e+00, 3.5159517528e+00]}


def _close(value, truth, rtol_scale):
    truth = np.asarray(truth)
    return np.max(np.abs(np.asarray(value) - truth)) <= rtol_scale * np.max(np.abs(truth))


def test_hill_valley_matches_mochiclass():
    c = mcs.eft_finite_coefficients('hill_valley', HV[0], ETA, **HV[1])
    # measured: h1 3.6e-10, D0 1.6e-8, N3 3.6e-8 of their scales
    assert _close(c['h1'], HV_TRUTH['h1'], 1e-8)
    assert _close(c['D0'], HV_TRUTH['D0'], 1e-6)
    assert _close(c['N3'], HV_TRUTH['N3'], 1e-6)
    assert c['noslip'] is True and c['N5'] is c['N3']


def test_propto_omega_matches_mochiclass():
    c = mcs.eft_finite_coefficients('propto_omega', PO[0], ETA, **PO[1])
    # measured: h1 1.8e-7, D0 4.0e-7, N3 2.5e-7, N5 1.3e-7 of their scales (with EFT_NA_PROPTO_OMEGA = 4096)
    for name in ('h1', 'D0', 'N3', 'N5'):
        assert _close(c[name], PO_TRUTH[name], 2e-6), name
    assert c['noslip'] is False and c['N5'] is not c['N3']
    # mu depends on k here
    k2 = np.array([1e-6, 1e-2, 1.])[:, None]
    mu = c['h1'] * (c['D0'] + c['N5'] * k2) / (c['D0'] + c['N3'] * k2)
    assert np.ptp(mu[:, -1]) > 1e-2


def test_noslip_and_exact_gr():
    # propto_omega with c_b = -2 c_m, c_t = 0 is no-slip: alpha_1 = 0 exactly, N5 IS N3
    c = mcs.eft_finite_coefficients('propto_omega', [0.1, -1.2, 0.6, 0., 1.], ETA, h=0.7)
    assert c['noslip'] is True and c['N5'] is c['N3'] and np.all(c['alpha_1'] == 0.)
    # forbidding the identification still gives the same values, bit for bit
    c = mcs.eft_finite_coefficients('propto_omega', [0.1, -1.2, 0.6, 0., 1.], ETA, h=0.7, noslip=False)
    assert c['noslip'] is False and c['N5'] is not c['N3'] and np.array_equal(c['N5'], c['N3'])
    # hill_valley at r = 1 is not no-slip
    assert mcs.eft_finite_coefficients('hill_valley', [0.1, -0.5, 1., 0.5, 1., 1.], ETA, h=0.7)['noslip'] is False
    # exact GR: h1 = 1, D0 = N3 = N5 = 0, mu = h1
    for model, p in (('hill_valley', [0.1, 0., 1., 0.5, 2., 1.]), ('propto_omega', [0.1, 0., 0., 0., 1.])):
        c = mcs.eft_finite_coefficients(model, p, ETA, h=0.7)
        assert np.all(c['h1'] == 1.) and np.all(c['D0'] == 0.) and np.all(c['N3'] == 0.) and np.all(c['N5'] == 0.)


def test_shapes():
    p = np.array([[0.1, -0.6, 1., 0.5, 2., 1.], [0.1, -1., 1., 0.5, 2., 1.]])
    c = mcs.eft_finite_coefficients('hill_valley', p, ETA, h=np.array([0.7, 0.6]))
    assert c['h1'].shape == (2, ETA.size)
    one = mcs.eft_finite_coefficients('hill_valley', p[1], ETA, h=0.6)
    assert np.allclose(c['h1'][1], one['h1'], rtol=0, atol=0) and np.allclose(c['D0'][1], one['D0'], rtol=1e-14, atol=0)
    assert np.shape(mcs.eft_finite_coefficients('hill_valley', p[0], -0.3, h=0.7)['h1']) == ()
    assert mcs.eft_finite_coefficients('hill_valley', p[0], ETA.reshape(2, 3), h=0.7)['N3'].shape == (2, 3)
    with pytest.raises(TypeError):
        mcs.eft_finite_coefficients('hill_valley', p[0], ETA, H0=70.)
    with pytest.raises(ValueError):
        mcs.eft_finite_coefficients('propto_omega', p[0], ETA)


def test_interpolators():
    from scipy.interpolate import CubicSpline
    it = mcs.eft_interpolators('hill_valley', HV[0], **HV[1])
    assert set(it) == set(mcs.EFT_FINITE_KEYS.values())
    assert all(isinstance(spline, CubicSpline) for spline in it.values())
    assert it['eftde_n5_interp'] is it['eftde_n3_interp']
    assert np.allclose(it['eftde_h1_interp'](ETA), HV_TRUTH['h1'], rtol=1e-8, atol=0)
    # the default grid is the engine's, [XNOW - 0.2, 0] with 512 points
    assert np.isclose(it['eftde_h1_interp'].x[0], mcs.XNOW - 0.2) and it['eftde_h1_interp'].x.size == 512
    it = mcs.eft_interpolators('propto_omega', PO[0], **PO[1])
    assert it['eftde_n5_interp'] is not it['eftde_n3_interp']
    with pytest.raises(ValueError):
        mcs.eft_interpolators('hill_valley', np.array([HV[0], HV[0]]), h=0.7)


def test_jax_path_matches_numpy():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)

    for model, (p, bg) in (('hill_valley', HV), ('propto_omega', PO)):
        ref = mcs.eft_finite_coefficients(model, p, ETA, **bg)

        @jax.jit
        def f(p, h, omega_cdm):
            c = mcs.eft_finite_coefficients(model, p, ETA, **dict(bg, h=h, omega_cdm=omega_cdm))
            return {name: c[name] for name in ('h1', 'D0', 'N3', 'N5')}

        c = f(jnp.array(p), bg['h'], bg['omega_cdm'])
        for name in ('h1', 'D0', 'N3', 'N5'):
            assert np.allclose(c[name], ref[name], rtol=1e-12, atol=1e-14 * np.max(np.abs(ref[name]))), (model, name)
    # no-slip under a trace: the flag is unknown, the values are N5 == N3 bit for bit anyway
    c = jax.jit(lambda c_M: mcs.eft_finite_coefficients('hill_valley', jnp.array([0.1, c_M, 1., 0.5, 2., 1.]), ETA, h=0.7)['N5']
                - mcs.eft_finite_coefficients('hill_valley', jnp.array([0.1, c_M, 1., 0.5, 2., 1.]), ETA, h=0.7)['N3'])(-0.6)
    assert np.all(np.asarray(c) == 0.)
    # differentiable in the coefficients
    g = jax.grad(lambda c_M: mcs.eft_finite_coefficients('hill_valley', jnp.array([0.1, c_M, 1., 0.5, 2., 1.]), ETA, h=0.7)['h1'][-1])(-0.6)
    assert np.isfinite(g) and g > 0.
    # the spline wrapper refuses tracers
    with pytest.raises(TypeError):
        jax.jit(lambda h: mcs.eft_interpolators('hill_valley', HV[0], h=h)['eftde_h1_interp'].c)(0.7)
