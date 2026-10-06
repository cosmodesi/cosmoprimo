"""The mochi_class stability gate, both gravity models: numpy / jax agreement, the superhorizon IC
test and the quasi-static pole test.

The expected verdicts and exponents below are mochi_class' own (a default run with
``perturbations_verbose=2`` prints ``x_smg = k^2 tau^n with n=...`` for a rejected model), so
these tests pin the gate to the code it reproduces without needing ``pyclass.mochiclass``.
The hill/valley ground truth was produced by ``Stability/validate_hill_valley_gate.py`` in
the DESI-DR2-MG project (2026-10-04).
"""

import numpy as np
import pytest

from cosmoprimo.emulators import mochiclass as mcs

FID = dict(h=0.6736, omega_b=0.02237, omega_cdm=0.12, m_ncdm=0.06)

#: (c_B, c_M, w0, wa) at c_K = 0.1, c_T = 0, M2_ini = 1 -> (mochi_class accepts with the IC test on,
#: x_growth printed by mochi_class or None when <= 3). All pass the background tests.
IC_TRUTH = [((0.5, 0.3, -1.2, -0.8), False, 4.329005),
            ((0.2, 0.3, -1.1, -0.3), False, 3.641415),
            ((0.5, 0.9, -1.2, -0.8), True, None),
            ((0.2, 0.5, -1.1, -0.3), True, None),
            ((1.0, 1.0, -0.9, 0.36), True, None),
            ((0.05, 0.0, -1.0, 0.0), True, None)]

#: hill/valley: (c_M, tau, a_t, r, M2_ini, alpha_K, w0, wa, h, omega_cdm) -> (mochi_class accepts with the IC
#: test on, the exponent it prints or None when <= 3), measured 2026-10-04 (omega_b = 0.02237, m_ncdm = 0.06).
#: All pass the background tests. With the pipeline's alpha_K = 0.1 the exponent never exceeds 3 for
#: w0 + wa < 0 (253 models, every verdict reproduced); the refused model needs alpha_K = 1e-4 and an early, fast
#: transition, where the hill/valley alphas at z = 1e10 are not negligible against the kineticity.
IC_TRUTH_HV = [((0.071263, 0.304104, 0.037931, 3.71527, 1.48726, 0.0001, -0.545101, -1.30779, 0.77965, 0.101399), False, 7.074011),
               ((0.268557, 0.831013, 0.14016, 2.04306, 1.41975, 0.0001, -0.698123, 0.460586, 0.739072, 0.11679), True, None),
               ((-0.190755, 1.3681, 0.115348, 3.57405, 1.30662, 0.0001, -0.794406, 0.375297, 0.640674, 0.102511), True, None),
               ((-0.112391, 0.634145, 0.346137, 2.8967, 1., 0.1, -0.684111, -0.22606, 0.728096, 0.112688), True, None),
               ((-0.234376, 1.93248, 0.358091, 0.549752, 1., 0.1, -0.783993, 0.069236, 0.686841, 0.101249), True, None)]


def _random_models(n, seed=0):
    rng = np.random.default_rng(seed)
    return dict(c_B=rng.uniform(-2, 2, n), c_M=rng.uniform(-3, 3, n), h=rng.uniform(0.5, 1, n),
                omega_b=rng.uniform(0.020005, 0.024995, n), omega_cdm=rng.uniform(0.08, 0.16, n),
                w0=rng.uniform(-3, 0.5, n), wa=rng.uniform(-3, 2, n), m_ncdm=rng.uniform(1e-6, 1, n),
                N_ur=rng.uniform(2, 4, n) - 1.0132)


def test_ic_test_reproduces_mochiclass():
    for (c_B, c_M, w0, wa), accepted, x_ref in IC_TRUTH:
        assert bool(mcs.stable_propto_omega(c_B, c_M, alpha_K=0.1, w0=w0, wa=wa, **FID)), 'background tests'
        ok = mcs.stable_propto_omega(c_B, c_M, alpha_K=0.1, ic_test=True, w0=w0, wa=wa, **FID)
        assert bool(ok) == accepted, (c_B, c_M, w0, wa)
        x = float(mcs.scan_propto_omega(c_B, c_M, alpha_K=0.1, ic_test=True, w0=w0, wa=wa, **FID)['x_growth'])
        if x_ref is None:
            assert x <= 3. + mcs.IC_TOLERANCE
        else:
            assert abs(x - x_ref) < 1e-4, (x, x_ref)
    # off by default, which is what a mochi_class run with pert_ic_tolerance_smg = -1 accepts
    assert bool(mcs.stable_propto_omega(0.5, 0.3, alpha_K=0.1, w0=-1.2, wa=-0.8, **FID))
    # the rescaled evaluation agrees with the raw one
    x_raw = mcs.scan_propto_omega(0.5, 0.3, alpha_K=0.1, ic_test=True, w0=-1.2, wa=-0.8, **FID)['x_growth']
    x_ref = mcs.scan_propto_omega(0.5, 0.3, alpha_K=0.1, ic_test=True, ic_omega_ref=1e-6, w0=-1.2, wa=-0.8, **FID)['x_growth']
    assert abs(float(x_raw) - float(x_ref)) < 1e-3
    with pytest.raises(ValueError, match='alpha_K'):
        mcs.stable_propto_omega(0.5, 0.3, ic_test=True, w0=-1.2, wa=-0.8, **FID)
    # dispatch by mochi_class' own (gravity_model, parameters_smg) pair
    ok = mcs.stable('propto_omega', np.array([[0.1, 0.5, 0.3, 0., 1.], [0.1, 0.5, 0.9, 0., 1.]]), ic_test=True, w0=-1.2, wa=-0.8, **FID)
    assert ok.tolist() == [False, True]


def test_ic_test_is_a_thin_band_above_the_gradient_boundary():
    P = _random_models(4000)
    cos = {k: P[k] for k in ('h', 'omega_b', 'omega_cdm', 'w0', 'wa', 'm_ncdm', 'N_ur')}
    base = mcs.stable_propto_omega(P['c_B'], P['c_M'], alpha_K=0.1, **cos)
    with_ic = mcs.stable_propto_omega(P['c_B'], P['c_M'], alpha_K=0.1, ic_test=True, **cos)
    assert not np.any(with_ic & ~base), 'the IC test can only reject'
    rejected = np.mean(~with_ic[base])
    assert 0.02 < rejected < 0.2, rejected      # measured 8.7% on mochi_class itself


def test_quasi_static_pole_test():
    """mu^2 > 0 over fkptjax's range: a model mochi_class accepts but whose h3 / h5 have a pole
    (the first training node that stalled an emulator training) is refused, the fiducial-like
    models are kept, and the gate's mu^2 matches the engine's construction."""
    pole = dict(c_B=0.38771, c_M=2.77861, w0=-0.27208, wa=-2.72407, h=0.50126, omega_b=0.02033, omega_cdm=0.1196, m_ncdm=0.57323, N_ur=2.66887 - 1.0132)
    cB, cM = pole.pop('c_B'), pole.pop('c_M')
    assert bool(mcs.stable_propto_omega(cB, cM, alpha_K=0.1, **pole))           # mochi_class runs it
    assert not bool(mcs.stable_propto_omega(cB, cM, alpha_K=0.1, qs_mu2=True, **pole))
    s = mcs.scan_propto_omega(cB, cM, alpha_K=0.1, qs_mu2=True, **pole)
    assert float(s['min_mu2_qs']) < -1e-3
    for c_B, c_M, w0, wa in [(1., 1., -1., 0.), (1., 1., -0.9, 0.36), (0.05, 0.1, -1., 0.)]:
        assert bool(mcs.stable_propto_omega(c_B, c_M, alpha_K=0.1, qs_mu2=True, w0=w0, wa=wa, **FID))


def test_exact_gr_passes_the_pole_test():
    """Exact GR on LambdaCDM (c_B = c_M = c_T = 0, w = -1) makes cs2num and mu^2 exact zeros: there
    is no scalar and no pole, and the verdict must say so -- it is the truth of every LambdaCDM
    mock. Phantom GR still fails the gradient test, and a genuine mu^2 crossing still fails."""
    gr = dict(alpha_K=0.1, qs_mu2=True, w0=-1., wa=0., **FID)
    s = mcs.scan_propto_omega(0., 0., **gr)
    assert float(s['min_cs2num']) == 0. and float(s['min_mu2_qs']) == 0.
    assert bool(mcs.stable_propto_omega(0., 0., **gr))
    assert bool(mcs.stable_propto_omega(0., 1e-3, **gr)) and bool(mcs.stable_propto_omega(1e-3, 0., **gr))
    assert not bool(mcs.stable_propto_omega(0., 0., **dict(gr, w0=-1.1)))          # phantom GR: cs2num < 0
    assert not bool(mcs.stable_propto_omega(0., -0.36, **gr))                       # negative c_M at c_B = 0
    import jax.numpy as jnp
    ok = mcs.stable_propto_omega(jnp.array([0., 0., 0.]), jnp.array([0., 0.5, -0.36]), **gr)
    assert list(np.asarray(ok)) == [True, True, False]


HV = dict(alpha_K=0.1, h=0.6736, omega_b=0.02237, omega_cdm=0.12, m_ncdm=0.06)


def _random_hill_valley(n, seed=0):
    rng = np.random.default_rng(seed)
    return dict(c_M=rng.uniform(-1, 1, n), tau=rng.uniform(0.5, 10, n), a_t=rng.uniform(0.01, 1, n), r=rng.uniform(0, 4, n),
                M2_ini=rng.uniform(0.5, 1.5, n), h=rng.uniform(0.5, 1, n), omega_b=rng.uniform(0.020005, 0.024995, n),
                omega_cdm=rng.uniform(0.08, 0.16, n), w0=rng.uniform(-3, 0.5, n), wa=rng.uniform(-3, 2, n),
                m_ncdm=rng.uniform(1e-6, 1, n), N_ur=rng.uniform(2, 4, n) - 1.0132)


def test_hill_valley_verdicts():
    """The background verdicts measured on mochi_class (Stability/ validation, 2026-09-07; the
    HillValley parameter-space notebook, 2026-10-01): the paper's No Slip Gravity point runs, a
    hill (c_M > 0) never runs on LambdaCDM, GR (c_M = 0) runs only on non-phantom backgrounds,
    and the tau ceiling on LambdaCDM at r = 2 is ~2 for a late transition."""
    assert bool(mcs.stable_hill_valley(-0.05, 1., 0.5, r=2., w0=-0.9, wa=0.36, **HV))
    assert bool(mcs.stable_hill_valley(-0.3, 1., 0.5, r=2., w0=-1., wa=0., **HV))
    assert not bool(mcs.stable_hill_valley(0.3, 1., 0.5, r=2., w0=-1., wa=0., **HV))
    assert not bool(mcs.stable_hill_valley(0.05, 1., 0.5, r=0.5, w0=-1., wa=0., **HV))
    assert bool(mcs.stable_hill_valley(0., 1., 0.5, w0=-1., wa=0., **HV))
    assert not bool(mcs.stable_hill_valley(0., 1., 0.5, w0=-1.1, wa=0., **HV))
    tau = np.geomspace(0.5, 10., 200)
    ok = mcs.stable_hill_valley(-0.3, tau, 0.8, r=2., w0=-1., wa=0., **HV)
    assert 1.9 < tau[ok].max() < 2.1 and np.all(ok[:np.argmax(~ok)])
    # dispatch by mochi_class' own (gravity_model, parameters_smg) pair; the braiding never crosses 2 here
    theta = np.array([[0.1, -0.3, 1., 0.5, 2., 1.], [0.1, 0.3, 1., 0.5, 2., 1.]])
    assert mcs.stable('hill_valley', theta, w0=-1., wa=0., **{k: v for k, v in HV.items() if k != 'alpha_K'}).tolist() == [True, False]
    s = mcs.scan_hill_valley(-1., 1., 0.5, r=4., w0=-1., wa=0.)
    assert -2. < float(s['min_bra']) and float(s['max_bra']) < 2.


def test_hill_valley_pole_test_matches_the_engine():
    """mu^2 (and cs2num) of the gate against the mochiclass engine's own, measured 2026-10-04
    on four hill/valley models (r = 0.7 ... 3.5, three backgrounds): 2e-5 relative or better; the
    minima over a = 0.02 ... 1 quoted here are the engine's."""
    ETA = np.log(np.geomspace(0.02, 1., 300))
    A = np.exp(ETA)
    for (c_M, tau, a_t, r, w0, wa), mu2_min in [((-0.3, 1., 0.5, 2., -1., 0.), 4.246e-1), ((-0.05, 1., 0.5, 2., -0.9, 0.36), 1.497e-1),
                                                 ((-0.6, 1.5, 0.7, 0.7, -0.95, -0.2), 1.053e-1), ((-0.2, 0.8, 0.3, 3.5, -0.8, -0.5), 1.169e0)]:
        bg = mcs.background(A, w0=w0, wa=wa, derivs=True, **{k: v for k, v in HV.items() if k != 'alpha_K'})
        aB, aM, aT, daB, M2 = mcs.alphas_hill_valley(A, c_M, tau, a_t, r, 1.)
        mu2 = mcs._qs_mu2(A, bg, aB, aM, aT, daB, M2)
        assert abs(float(mu2.min()) / mu2_min - 1.) < 1e-3, (float(mu2.min()), mu2_min)
        assert bool(mcs.stable_hill_valley(c_M, tau, a_t, r=r, qs_mu2=True, w0=w0, wa=wa, **HV))
    # the pole test can only reject, and exact GR on LambdaCDM is an exact zero that passes
    P = _random_hill_valley(3000, seed=3)
    cos = {k: P[k] for k in ('h', 'omega_b', 'omega_cdm', 'w0', 'wa', 'm_ncdm', 'N_ur')}
    base = mcs.stable_hill_valley(P['c_M'], P['tau'], P['a_t'], r=P['r'], M2_ini=P['M2_ini'], alpha_K=0.1, **cos)
    with_qs = mcs.stable_hill_valley(P['c_M'], P['tau'], P['a_t'], r=P['r'], M2_ini=P['M2_ini'], alpha_K=0.1, qs_mu2=True, **cos)
    assert not np.any(with_qs & ~base) and 0.01 < np.mean(~with_qs[base]) < 0.3     # measured 8.3% on 2^16 draws
    s = mcs.scan_hill_valley(0., 1., 0.5, qs_mu2=True, w0=-1., wa=0., **{k: v for k, v in HV.items() if k != 'alpha_K'})
    assert float(s['min_cs2num']) == 0. and float(s['min_mu2_qs']) == 0.
    assert bool(mcs.stable_hill_valley(0., 1., 0.5, qs_mu2=True, w0=-1., wa=0., **HV))
    assert bool(mcs.stable_hill_valley(-1e-3, 1., 0.5, qs_mu2=True, w0=-1., wa=0., **HV))


def test_hill_valley_ic_test():
    """The superhorizon IC test for hill/valley: off by default, needs alpha_K, can only reject,
    and reproduces mochi_class where its own test fires (IC_TRUTH_HV, measured 2026-10-04 with
    perturbations_verbose=2 on a default mochi_class run)."""
    with pytest.raises(ValueError, match='alpha_K'):
        mcs.stable_hill_valley(-0.3, 1., 0.5, ic_test=True, w0=-1., wa=0.)
    P = _random_hill_valley(3000, seed=4)
    cos = {k: P[k] for k in ('h', 'omega_b', 'omega_cdm', 'w0', 'wa', 'm_ncdm', 'N_ur')}
    base = mcs.stable_hill_valley(P['c_M'], P['tau'], P['a_t'], r=P['r'], M2_ini=P['M2_ini'], alpha_K=0.1, **cos)
    with_ic = mcs.stable_hill_valley(P['c_M'], P['tau'], P['a_t'], r=P['r'], M2_ini=P['M2_ini'], alpha_K=0.1, ic_test=True, **cos)
    assert not np.any(with_ic & ~base)
    x = mcs.scan_hill_valley(P['c_M'], P['tau'], P['a_t'], r=P['r'], M2_ini=P['M2_ini'], alpha_K=0.1, ic_test=True, **cos)['x_growth']
    assert np.isfinite(x).all()
    for (c_M, tau, a_t, r, M2_ini, alpha_K, w0, wa, h, omega_cdm), accepted, x_ref in IC_TRUTH_HV:
        kw = dict(HV, M2_ini=M2_ini, alpha_K=alpha_K, w0=w0, wa=wa, h=h, omega_cdm=omega_cdm)
        assert bool(mcs.stable_hill_valley(c_M, tau, a_t, r=r, **kw)), 'background tests'
        assert bool(mcs.stable_hill_valley(c_M, tau, a_t, r=r, ic_test=True, **kw)) == accepted, (c_M, tau, a_t, r, w0, wa)
        xg = float(mcs.scan_hill_valley(c_M, tau, a_t, r=r, ic_test=True, **kw)['x_growth'])
        if x_ref is None:
            assert xg <= 3. + mcs.IC_TOLERANCE
        else:
            assert abs(xg - x_ref) < 1e-4, (xg, x_ref)


def test_jax_path_matches_numpy():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    P = _random_models(500, seed=1)
    cos = ('h', 'omega_b', 'omega_cdm', 'w0', 'wa', 'm_ncdm', 'N_ur')
    ok_np = mcs.stable_propto_omega(P['c_B'], P['c_M'], alpha_K=0.1, ic_test=True, **{k: P[k] for k in cos})
    J = {k: jnp.asarray(v) for k, v in P.items()}
    # batched, eager
    ok_j = mcs.stable_propto_omega(J['c_B'], J['c_M'], alpha_K=0.1, ic_test=True, **{k: J[k] for k in cos})
    assert bool(jnp.array_equal(ok_j, ok_np))
    # one point at a time under jit(vmap), as a sampler's log-prior would call it

    def one(c_B, c_M, h, omega_b, omega_cdm, w0, wa, m_ncdm, N_ur):
        return mcs.stable_propto_omega(c_B, c_M, alpha_K=0.1, ic_test=True, h=h, omega_b=omega_b, omega_cdm=omega_cdm,
                                       w0=w0, wa=wa, m_ncdm=m_ncdm, N_ur=N_ur)

    ok_v = jax.jit(jax.vmap(one))(*[J[k] for k in ('c_B', 'c_M') + cos])
    assert bool(jnp.array_equal(ok_v, ok_np))
    # ... and it differentiates

    def margins(x):
        s = mcs.scan_propto_omega(x[0], x[1], alpha_K=0.1, ic_test=True, w0=x[2], wa=x[3], **FID)
        return jnp.stack([s['min_cs2num'], s['x_growth']])

    jac = jax.jacfwd(margins)(jnp.array([0.5, 0.3, -1.2, -0.8]))
    assert jac.shape == (2, 4) and bool(jnp.isfinite(jac).all())


def test_hill_valley_jax_path_matches_numpy():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    P = _random_hill_valley(400, seed=2)
    cos = ('h', 'omega_b', 'omega_cdm', 'w0', 'wa', 'm_ncdm', 'N_ur')
    kw = dict(alpha_K=0.1, ic_test=True, qs_mu2=True)
    ok_np = mcs.stable_hill_valley(P['c_M'], P['tau'], P['a_t'], r=P['r'], M2_ini=P['M2_ini'], **kw, **{k: P[k] for k in cos})
    J = {k: jnp.asarray(v) for k, v in P.items()}
    ok_j = mcs.stable_hill_valley(J['c_M'], J['tau'], J['a_t'], r=J['r'], M2_ini=J['M2_ini'], **kw, **{k: J[k] for k in cos})
    assert bool(jnp.array_equal(ok_j, ok_np))

    def one(c_M, tau, a_t, r, M2_ini, h, omega_b, omega_cdm, w0, wa, m_ncdm, N_ur):
        return mcs.stable_hill_valley(c_M, tau, a_t, r=r, M2_ini=M2_ini, h=h, omega_b=omega_b, omega_cdm=omega_cdm,
                                      w0=w0, wa=wa, m_ncdm=m_ncdm, N_ur=N_ur, **kw)

    ok_v = jax.jit(jax.vmap(one))(*[J[k] for k in ('c_M', 'tau', 'a_t', 'r', 'M2_ini') + cos])
    assert bool(jnp.array_equal(ok_v, ok_np))

    def margins(x):
        s = mcs.scan_hill_valley(x[0], x[1], x[2], r=x[3], w0=x[4], wa=x[5], **kw, **{k: v for k, v in HV.items() if k != 'alpha_K'})
        return jnp.stack([s['min_cs2num'], s['min_mu2_qs'], s['x_growth']])

    x0 = jnp.array([-0.3, 1.0, 0.5, 2.0, -0.9, 0.36])
    jac = jax.jacfwd(margins)(x0)
    assert jac.shape == (3, 6) and bool(jnp.isfinite(jac).all())
    eps = 1e-5
    fd = (margins(x0.at[0].add(eps)) - margins(x0.at[0].add(-eps))) / (2 * eps)
    assert abs(float(jac[1, 0]) - float(fd[1])) < 1e-6 * max(1., abs(float(fd[1])))


if __name__ == '__main__':
    test_ic_test_reproduces_mochiclass()
    test_ic_test_is_a_thin_band_above_the_gradient_boundary()
    test_hill_valley_verdicts()
    test_hill_valley_pole_test_matches_the_engine()
    test_hill_valley_ic_test()
    test_jax_path_matches_numpy()
    test_hill_valley_jax_path_matches_numpy()
