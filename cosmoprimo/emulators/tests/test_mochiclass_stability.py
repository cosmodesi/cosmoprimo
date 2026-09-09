"""The mochi_class stability gate: numpy / jax agreement, and the superhorizon IC test.

The expected verdicts and exponents below are mochi_class' own (a default run with
``perturbations_verbose=2`` prints ``x_smg = k^2 tau^n with n=...`` for a rejected model), so
these tests pin the gate to the code it reproduces without needing ``pyclass.mochiclass``.
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


def test_hill_valley_placeholders():
    assert isinstance(bool(mcs.stable_hill_valley(0.1, 1., 0.5, alpha_K=0.1, w0=-1., wa=0.)), bool)
    with pytest.raises(NotImplementedError, match='hill_valley'):
        mcs.stable_hill_valley(0.1, 1., 0.5, alpha_K=0.1, ic_test=True, w0=-1., wa=0.)


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
    # hill_valley refuses jax inputs rather than failing somewhere inside
    with pytest.raises(NotImplementedError, match='hill_valley'):
        mcs.stable_hill_valley(jnp.array([0.1]), 1., 0.5, w0=-1., wa=0.)


if __name__ == '__main__':
    test_ic_test_reproduces_mochiclass()
    test_ic_test_is_a_thin_band_above_the_gradient_boundary()
    test_hill_valley_placeholders()
    test_jax_path_matches_numpy()
