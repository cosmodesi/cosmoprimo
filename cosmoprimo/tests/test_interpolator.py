import pytest
import numpy as np

from cosmoprimo import (Cosmology, Transfer, Fourier, PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D,
                        CorrelationFunctionInterpolator2D)


def check_shape_1d(interp):
    assert interp(0.1).shape == ()
    assert interp([]).shape == (0, )
    assert interp([[0.1, 0.2]] * 3).shape == (3, 2)
    assert interp(np.array([[0.1, 0.2]] * 3, dtype='f4')).dtype.itemsize == 4
    assert np.allclose(interp([0.2, 0.1]), interp([0.1, 0.2])[::-1], atol=0)


def check_shape_2d(interp, grid=True):
    assert interp(0.1, 0.1).shape == ()
    if grid:
        assert interp(np.array([]), np.array(0.1)).shape == (0, )
        assert interp([], []).shape == (0, 0)
        assert interp(0.1, [0.1, 0.1]).shape == (2, )
        assert interp([[0.1, 0.2]] * 3, 0.1).shape == (3, 2)
        assert interp([[0.1, 0.2]] * 3, [0.1]).shape == (3, 2, 1)
        assert interp([[0.1, 0.2]] * 3, [[0.1, 0.1, 0.2]] * 3).shape == (3, 2, 3, 3)
        assert interp(np.array([[0.1, 0.2]] * 3, dtype='f4'), np.array(0.1, dtype='f4')).dtype.itemsize == 4
        assert np.allclose(interp([0.2, 0.1], [0.1, 0.]), interp([0.1, 0.2], [0., 0.1])[::-1, ::-1], atol=0)
    else:
        assert interp([], [], grid=False).shape == (0, )
        assert interp([0.1, 0.2], [0.1, 0.2], grid=False).shape == (2, )
        assert interp([[0.1, 0.2]] * 3, [[0.1, 0.2]] * 3, grid=False).shape == (3, 2)
        assert np.allclose(interp([0.2, 0.1], [0.1, 0.], grid=False), interp([0.1, 0.2], [0., 0.1], grid=False)[::-1], atol=0)


def test_power_spectrum():

    cosmo = Cosmology()
    tr = Transfer(cosmo, engine='eisenstein_hu')
    k = np.logspace(-3, 1.5, 100)
    pk = tr.transfer_k(k)**2 * k ** cosmo['n_s']

    interp = PowerSpectrumInterpolator1D(k, pk)
    check_shape_1d(interp)
    interp2d = PowerSpectrumInterpolator2D(k, z=[0.], pk=pk[..., None])
    interp2d(k, z=0.)

    interp = PowerSpectrumInterpolator1D(k, pk)
    check_shape_1d(interp)
    interp2 = interp.clone()
    assert np.all(interp2(np.ones((4, 2))) == interp(np.ones((4, 2))))
    check_shape_1d(interp.sigma_r)

    interp = PowerSpectrumInterpolator2D(k, z=0, pk=pk, growth_factor_sq=lambda z: np.ones_like(z))
    assert np.allclose(interp(k, z=np.random.uniform(0., 1., 10)), pk[:, None], atol=0, rtol=1e-5)
    check_shape_2d(interp)
    check_shape_2d(interp, grid=False)
    interp2 = interp.clone()
    assert np.all(interp2(k, z=[0] * 2) == interp(k, z=[0] * 2))

    rng = np.random.RandomState(seed=42)
    z = np.linspace(1., 0., 10)
    interp = PowerSpectrumInterpolator2D(k, z=z, pk=np.array([pk] * len(z)).T)
    check_shape_2d(interp)
    assert np.allclose(interp(k, z=rng.uniform(0., 1., 10)), pk[:, None], atol=0, rtol=1e-5)  # ok as same pk for all z
    check_shape_1d(interp.sigma8_z)
    check_shape_1d(interp.sigma_dz)
    check_shape_2d(interp.sigma_rz)
    interp = PowerSpectrumInterpolator2D(k, z=z, pk=np.array([pk * (iz + 1) / len(z) for iz in range(len(z))]).T)
    check_shape_2d(interp.growth_rate_rz)
    dz = 1e-3
    assert np.allclose(interp.growth_rate_rz(8., dz * 2., dz=dz), interp.growth_rate_rz(8., 0., dz=dz), rtol=1e-2)

    interp = PowerSpectrumInterpolator2D(k, z=z, pk=np.array([pk] * len(z)).T, extrap_kmin=1e-6, extrap_kmax=1e2)
    check_shape_2d(interp)

    cosmo = Cosmology()
    for engine in ['eisenstein_hu', 'eisenstein_hu_nowiggle_variants', 'class']:
        fo = Fourier(cosmo, engine=engine)
        interp = fo.pk_interpolator()
        k = np.logspace(-4, 2, 100)
        z = np.linspace(0, 4, 10)
        check_shape_2d(interp)
        pk = interp(k, z)
        interp2 = interp.clone()
        assert np.allclose(interp2(k, z), pk, rtol=1e-4)
        interp2 = interp.clone(pk=2 * interp.pk)
        assert np.allclose(interp2(k, z), 2 * pk, rtol=1e-4)
        for iz, zz in enumerate(z):
            interp1d = interp.to_1d(z=zz)
            check_shape_1d(interp1d)
            assert np.allclose(interp1d.extrap_kmin, interp.extrap_kmin)
            assert np.allclose(interp1d.extrap_kmax, interp.extrap_kmax)
            assert np.allclose(interp1d(k), pk[:, iz])
            assert np.allclose(interp.sigma8_z(zz), interp.to_1d(zz).sigma8(), rtol=1e-4)
            assert np.allclose(interp.sigma_dz(zz), interp.to_1d(zz).sigma_d(), rtol=1e-4)
            assert np.allclose(interp.sigma_dz(zz, nk=None), interp.to_1d(zz).sigma_d(), rtol=1e-4)

        interp2 = PowerSpectrumInterpolator2D.from_callable(interp.k, interp.z, interp)
        check_shape_2d(interp2)
        check_shape_2d(interp2, grid=False)
        assert np.allclose(interp2(k, z), interp(k, z), rtol=1e-4)
        interp_1d = interp.to_1d()
        check_shape_1d(interp_1d)
        interp_1d2 = interp_1d.from_callable(interp_1d.k, interp_1d)
        check_shape_1d(interp_1d2)
        assert np.allclose(interp_1d2(k), interp_1d(k), rtol=1e-4)


def test_correlation_function():
    cosmo = Cosmology()
    fo = Fourier(cosmo, engine='eisenstein_hu')
    pk_interp = fo.pk_interpolator()
    xi_interp = pk_interp.to_xi()
    s = np.logspace(-2, 2, 100)
    z = np.linspace(0, 4, 10)
    assert np.allclose(xi_interp.clone()(s, z), xi_interp(s, z), rtol=1e-4)
    check_shape_2d(xi_interp)
    xi_interp2 = CorrelationFunctionInterpolator2D.from_callable(xi_interp.s, xi_interp.z, xi_interp)
    check_shape_2d(xi_interp2)
    assert np.allclose(xi_interp2(s, z), xi_interp(s, z), rtol=1e-4)
    xi_interp_1d = xi_interp.to_1d()
    check_shape_1d(xi_interp_1d)
    xi_interp_1d2 = xi_interp_1d.from_callable(xi_interp_1d.s, xi_interp_1d)
    check_shape_1d(xi_interp_1d2)
    assert np.allclose(xi_interp_1d2(s), xi_interp_1d(s), rtol=1e-4)

    pk_interp2 = xi_interp.to_pk()
    k = np.logspace(-4, 1, 100)
    z = np.linspace(0, 4, 10)
    z = 0.5
    assert np.allclose(pk_interp(k, z), pk_interp2(k, z), rtol=1e-2)

    z = np.linspace(0, 4, 10)
    for zz in z:
        pk_interp = fo.pk_interpolator().to_1d(z=zz)
        pk_interp2 = pk_interp.to_xi().to_pk()
        assert np.allclose(pk_interp(k), pk_interp2(k), rtol=1e-2)
        pk_interp2 = pk_interp.to_xi().clone().to_pk()
        assert np.allclose(pk_interp(k), pk_interp2(k), rtol=1e-2)
        assert np.allclose(xi_interp.sigma_dz(zz), pk_interp.sigma_d(), rtol=1e-4)
        assert np.allclose(xi_interp.sigma8_z(zz), pk_interp.sigma8(), rtol=1e-4)
        assert np.allclose(xi_interp.sigma8_z(zz), xi_interp.to_1d(zz).sigma8(), rtol=1e-4)


def test_extrap_1d(plot=True):
    if plot:
        from matplotlib import pyplot as plt
    cosmo = Cosmology()

    fo = Fourier(cosmo, engine='eisenstein_hu')
    k = np.logspace(-4, 2, 1000)
    k_extrap = np.logspace(-6, 3, 1000)
    k_eval = k_extrap[1:-1]  # to avoid error with rounding
    pk_interp_callable = fo.pk_interpolator(k=k_extrap).to_1d()
    pk_interp_tab = PowerSpectrumInterpolator1D(k, pk_interp_callable(k))
    assert np.allclose(pk_interp_tab(k), pk_interp_callable(k), atol=0, rtol=0.1)
    pk_interp_tab = PowerSpectrumInterpolator1D(k, pk_interp_callable(k), extrap_kmin=k_extrap[0], extrap_kmax=k_extrap[-1])
    assert np.allclose(pk_interp_tab(k_eval), pk_interp_callable(k_eval), atol=0, rtol=0.1)
    assert np.allclose(pk_interp_tab.extrap_kmin, pk_interp_callable.k[0])
    assert np.allclose(pk_interp_tab.extrap_kmax, pk_interp_callable.k[-1])
    pk_interp_callable(k_eval / 2.)
    pk_interp_callable(k_eval * 2.)
    pk_interp_tab(k_eval)
    assert np.allclose(pk_interp_tab(k_eval[0] / 2., bounds_error=False), pk_interp_tab(k_extrap[0], bounds_error=False), atol=0.)
    assert np.allclose(pk_interp_tab(k_eval[-1] * 2., bounds_error=False), pk_interp_tab(k_extrap[-1], bounds_error=False), atol=0.)
    with pytest.raises(ValueError):
        pk_interp_tab(k_eval / 2.)
    with pytest.raises(ValueError):
        pk_interp_tab(k_eval * 2.)

    if plot:
        plt.loglog(k_eval, pk_interp_callable(k_eval), label='callable')
        plt.loglog(k_eval, pk_interp_tab(k_eval), label='tab')
        plt.legend()
        plt.show()

    xi_interp_callable = pk_interp_callable.to_xi()
    xi_interp_tab = pk_interp_tab.to_xi()
    s_eval = xi_interp_tab.s
    xi_interp_tab(s_eval)
    assert np.allclose(xi_interp_tab(s_eval[0] / 2., bounds_error=False), xi_interp_tab(s_eval[0], bounds_error=False), atol=0.)
    assert np.allclose(xi_interp_tab(s_eval[-1] * 2., bounds_error=False), xi_interp_tab(s_eval[-1], bounds_error=False), atol=0.)
    with pytest.raises(ValueError):
        xi_interp_tab(s_eval / 2.)
    with pytest.raises(ValueError):
        xi_interp_tab(s_eval * 2.)
    assert np.allclose(xi_interp_tab(s_eval), xi_interp_callable(s_eval), rtol=0.1)
    if plot:
        plt.plot(s_eval, s_eval**2 * xi_interp_callable(s_eval), label='callable')
        plt.plot(s_eval, s_eval**2 * xi_interp_tab(s_eval), label='tab')
        plt.xscale('log')
        plt.legend()
        plt.show()

    pk_interp_tab2 = xi_interp_tab.to_pk()
    if plot:
        plt.loglog(k_eval, pk_interp_callable(k_eval), label='callable')
        plt.loglog(k_eval, pk_interp_tab2(k_eval), label='tab')
        plt.legend()
        plt.show()

    assert np.allclose(pk_interp_tab2(k), pk_interp_callable(k), atol=0, rtol=1e-2)


def test_extrap_2d(plot=False):
    if plot:
        from matplotlib import pyplot as plt
    cosmo = Cosmology()

    fo = Fourier(cosmo, engine='eisenstein_hu')
    k = np.logspace(-4, 2, 1000)
    z = np.linspace(0, 4, 10)
    k_extrap = np.logspace(-6, 3, 1000)
    k_eval = k_extrap[1:-1]  # to avoid error with rounding
    z_eval = z
    pk_interp_callable = fo.pk_interpolator(k=k_extrap, z=z)
    pk_interp_tab = PowerSpectrumInterpolator2D(k, z, pk_interp_callable(k, z))
    assert np.allclose(pk_interp_tab(k, z_eval), pk_interp_callable(k, z_eval), atol=0, rtol=0.1)
    pk_interp_tab = PowerSpectrumInterpolator2D(k, z, pk_interp_callable(k, z), extrap_kmin=k_extrap[0], extrap_kmax=k_extrap[-1])
    assert np.allclose(pk_interp_tab(k_eval, z_eval), pk_interp_callable(k_eval, z_eval), atol=0, rtol=0.1)
    assert np.allclose(pk_interp_tab.extrap_kmin, pk_interp_callable.k[0])
    assert np.allclose(pk_interp_tab.extrap_kmax, pk_interp_callable.k[-1])
    pk_interp_callable(k_eval / 2., z_eval)
    pk_interp_callable(k_eval * 2., z_eval)
    pk_interp_tab(k_eval, z_eval)
    assert np.allclose(pk_interp_tab(k_eval[0] / 2., z_eval, bounds_error=False), pk_interp_tab(k_extrap[0], z_eval, bounds_error=False), atol=0.)
    assert np.allclose(pk_interp_tab(k_eval[-1] * 2., z_eval, bounds_error=False), pk_interp_tab(k_extrap[-1], z_eval, bounds_error=False), atol=0.)
    with pytest.raises(ValueError):
        pk_interp_tab(k_eval / 2., z_eval)
    with pytest.raises(ValueError):
        pk_interp_tab(k_eval * 2., z_eval)
    with pytest.raises(ValueError):
        pk_interp_tab(k_eval, z_eval * 2.)
    assert np.allclose(pk_interp_tab(k_eval, z_eval[-1] * 2., bounds_error=False), pk_interp_tab(k_eval, z_eval[-1], bounds_error=False), atol=0.)
    pk_interp_extrapz = PowerSpectrumInterpolator2D(k, z, pk_interp_callable(k, z), extrap_kmin=k_extrap[0], extrap_kmax=k_extrap[-1], extrap_z=True)
    assert np.allclose(pk_interp_extrapz(k_eval, z_eval[-1] * 2.), pk_interp_tab(k_eval, z_eval[-1], bounds_error=False), atol=0.)

    if plot:
        plt.loglog(k_eval, pk_interp_callable(k_eval, z_eval), linestyle='-')
        plt.loglog(k_eval, pk_interp_tab(k_eval, z_eval), linestyle='--')
        plt.show()

    xi_interp_callable = pk_interp_callable.to_xi()
    xi_interp_tab = pk_interp_tab.to_xi()
    s_eval = xi_interp_tab.s
    xi_interp_tab(s_eval, z_eval)
    assert np.allclose(xi_interp_tab(s_eval[0] / 2., z_eval, bounds_error=False), xi_interp_tab(s_eval[0], z_eval, bounds_error=False), atol=0.)
    assert np.allclose(xi_interp_tab(s_eval[-1] * 2., z_eval, bounds_error=False), xi_interp_tab(s_eval[-1], z_eval, bounds_error=False), atol=0.)
    with pytest.raises(ValueError):
        xi_interp_tab(s_eval / 2., z_eval)
    with pytest.raises(ValueError):
        xi_interp_tab(s_eval * 2., z_eval)
    with pytest.raises(ValueError):
        xi_interp_tab(s_eval, z_eval * 2.)
    assert np.allclose(xi_interp_tab(s_eval, z_eval[-1] * 2., bounds_error=False), xi_interp_tab(s_eval, z_eval[-1], bounds_error=False), atol=0.)
    xi_interp_extrapz = pk_interp_extrapz.to_xi()
    assert np.allclose(xi_interp_extrapz(s_eval, z_eval[-1] * 2.), xi_interp_tab(s_eval, z_eval[-1], bounds_error=False), atol=0.)

    assert np.allclose(xi_interp_tab(s_eval, z_eval), xi_interp_callable(s_eval, z_eval), rtol=0.1)
    if plot:
        plt.plot(s_eval, s_eval[:, None]**2 * xi_interp_callable(s_eval, z_eval), linestyle='-')
        plt.plot(s_eval, s_eval[:, None]**2 * xi_interp_tab(s_eval, z_eval), linestyle='--')
        plt.xscale('log')
        plt.show()

    pk_interp_tab2 = xi_interp_tab.to_pk()
    if plot:
        plt.loglog(k_eval, pk_interp_callable(k_eval, z_eval), linestyle='-')
        plt.loglog(k_eval, pk_interp_tab2(k_eval, z_eval), linestyle='--')
        plt.show()

    assert np.allclose(pk_interp_tab2(k), pk_interp_callable(k), atol=0, rtol=1e-2)


if __name__ == '__main__':

    test_power_spectrum()
    test_correlation_function()
    test_extrap_1d(plot=False)
    test_extrap_2d(plot=False)
