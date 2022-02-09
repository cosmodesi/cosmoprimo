import pytest
import numpy as np

from cosmoprimo import Cosmology, Transfer, Fourier, PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D, \
                        CorrelationFunctionInterpolator1D, CorrelationFunctionInterpolator2D


def test_power_spectrum():

    cosmo = Cosmology()
    tr = Transfer(cosmo,engine='eisenstein_hu')
    k = np.logspace(-3,1.5,100)
    pk = tr.transfer_k(k)**2 * k ** cosmo['n_s']

    interp = PowerSpectrumInterpolator1D(k,pk)
    assert np.allclose(interp(k),pk,atol=0,rtol=1e-5)
    assert np.ndim(interp(0.1)) == 0
    assert interp(np.ones(4)).shape == (4,)
    assert interp(np.ones((4,2))).shape == (4,2)
    interp2 = interp.clone()
    assert np.all(interp2(np.ones((4,2))) == interp(np.ones((4,2))))

    interp = PowerSpectrumInterpolator2D(k,z=0,pk=pk,growth_factor_sq=lambda z: np.ones_like(z))
    assert np.allclose(interp(k,z=np.random.uniform(0.,1.,10)),pk[:,None],atol=0,rtol=1e-5)
    assert interp(k,z=0).shape == (k.size,)
    assert interp(k,z=[0]).shape == (k.size,1)
    assert interp(k,z=[0]*2).shape == (k.size,2)
    assert interp(k[0],z=[0]*2).shape == (2,)
    assert np.ndim(interp(k[0],z=0)) == 0
    assert interp([k]*3,z=0).shape == (3,) + k.shape
    interp2 = interp.clone()
    assert np.all(interp2(k,z=[0]*2) == interp(k,z=[0]*2))

    rng = np.random.RandomState(seed=42)
    z = rng.uniform(0.,1.,10)
    interp = PowerSpectrumInterpolator2D(k,z=z,pk=np.array([pk]*len(z)).T)
    assert np.allclose(interp(k,z=rng.uniform(0.,1.,10)),pk[:,None],atol=0,rtol=1e-5) # ok as same pk for all z

    interp = PowerSpectrumInterpolator2D(k,z=z,pk=np.array([pk]*len(z)).T,extrap_kmin=1e-6,extrap_kmax=1e2)
    kk = np.logspace(-5,1.8,100)

    cosmo = Cosmology()
    for engine in ['eisenstein_hu', 'class']:
        fo = Fourier(cosmo, engine=engine)
        pk_interp = fo.pk_interpolator()
        k = np.logspace(-4,2,100)
        z = np.linspace(0,4,10)
        pk = pk_interp(k,z)
        pk_interp2 = pk_interp.clone()
        assert np.allclose(pk_interp2(k,z),pk,rtol=1e-4)
        pk_interp2 = pk_interp.clone(pk=2*pk_interp.pk)
        assert np.allclose(pk_interp2(k,z),2*pk,rtol=1e-4)
        for iz,z_ in enumerate(z):
            pk_interp1d = pk_interp.to_1d(z=z_)
            assert np.allclose(pk_interp1d.extrap_kmin, pk_interp.extrap_kmin)
            assert np.allclose(pk_interp1d.extrap_kmax, pk_interp.extrap_kmax)
            assert np.allclose(pk_interp1d(k),pk[:,iz])
            assert np.allclose(pk_interp.sigma8_z(z_),pk_interp.to_1d(z_).sigma8(),rtol=1e-4)
            assert np.allclose(pk_interp.sigma_dz(z_),pk_interp.to_1d(z_).sigma_d(),rtol=1e-4)
            assert np.allclose(pk_interp.sigma_dz(z_,nk=None),pk_interp.to_1d(z_).sigma_d(),rtol=1e-4)

        pk_interp2 = PowerSpectrumInterpolator2D.from_callable(pk_interp.k,pk_interp.z,pk_interp)
        assert np.allclose(pk_interp2(k,z),pk_interp(k,z),rtol=1e-4)
        pk_interp_1d = pk_interp.to_1d()
        pk_interp_1d2 = pk_interp_1d.from_callable(pk_interp_1d.k,pk_interp_1d)
        assert np.allclose(pk_interp_1d2(k),pk_interp_1d(k),rtol=1e-4)


def test_correlation_function():
    cosmo = Cosmology()
    fo = Fourier(cosmo,engine='eisenstein_hu')
    pk_interp = fo.pk_interpolator()
    xi_interp = pk_interp.to_xi()
    s = np.logspace(-2,2,100)
    z = np.linspace(0,4,10)
    assert np.allclose(xi_interp.clone()(s,z),xi_interp(s,z),rtol=1e-4)
    xi_interp2 = CorrelationFunctionInterpolator2D.from_callable(xi_interp.s,xi_interp.z,xi_interp)
    assert np.allclose(xi_interp2(s,z),xi_interp(s,z),rtol=1e-4)
    xi_interp_1d = xi_interp.to_1d()
    xi_interp_1d2 = xi_interp_1d.from_callable(xi_interp_1d.s,xi_interp_1d)
    assert np.allclose(xi_interp_1d2(s),xi_interp_1d(s),rtol=1e-4)

    pk_interp2 = xi_interp.to_pk()
    k = np.logspace(-4,1,100)
    z = np.linspace(0,4,10)
    z = 0.5
    assert np.allclose(pk_interp(k,z),pk_interp2(k,z),rtol=1e-2)

    z = np.linspace(0,4,10)
    for z_ in z:
        pk_interp = fo.pk_interpolator().to_1d(z=z_)
        pk_interp2 = pk_interp.to_xi().to_pk()
        assert np.allclose(pk_interp(k),pk_interp2(k),rtol=1e-2)
        pk_interp2 = pk_interp.to_xi().clone().to_pk()
        assert np.allclose(pk_interp(k),pk_interp2(k),rtol=1e-2)
        assert np.allclose(xi_interp.sigma_dz(z_),pk_interp.sigma_d(),rtol=1e-4)
        assert np.allclose(xi_interp.sigma8_z(z_),pk_interp.sigma8(),rtol=1e-4)
        assert np.allclose(xi_interp.sigma8_z(z_),xi_interp.to_1d(z_).sigma8(),rtol=1e-4)


def test_extrap_1d(plot=True):
    if plot:
        from matplotlib import pyplot as plt
    cosmo = Cosmology()

    fo = Fourier(cosmo, engine='eisenstein_hu')
    k = np.logspace(-4,2,1000)
    k_extrap = np.logspace(-6,3,1000)
    pk_interp_callable = fo.pk_interpolator(k=k_extrap).to_1d()
    pk_interp_tab = PowerSpectrumInterpolator1D(k, pk_interp_callable(k), extrap_kmin=k_extrap[0], extrap_kmax=k_extrap[-1])
    k_eval = k_extrap[1:-1] # to avoid error with rounding
    assert np.allclose(pk_interp_tab(k_eval), pk_interp_callable(k_eval), atol=0, rtol=0.1)
    assert np.allclose(pk_interp_tab.extrap_kmin, pk_interp_callable.k[0])
    assert np.allclose(pk_interp_tab.extrap_kmax, pk_interp_callable.k[-1])
    pk_interp_callable(k_eval/2.)
    pk_interp_callable(k_eval*2.)
    pk_interp_tab(k_eval)
    assert np.allclose(pk_interp_tab(k_eval[0]/2., bounds_error=False), pk_interp_tab(k_extrap[0], bounds_error=False), atol=0.)
    assert np.allclose(pk_interp_tab(k_eval[-1]*2., bounds_error=False), pk_interp_tab(k_extrap[-1], bounds_error=False), atol=0.)
    with pytest.raises(ValueError):
        pk_interp_tab(k_eval/2.)
    with pytest.raises(ValueError):
        pk_interp_tab(k_eval*2.)

    if plot:
        plt.loglog(k_eval, pk_interp_callable(k_eval), label='callable')
        plt.loglog(k_eval, pk_interp_tab(k_eval), label='tab')
        plt.legend()
        plt.show()

    xi_interp_callable = pk_interp_callable.to_xi()
    xi_interp_tab = pk_interp_tab.to_xi()
    s_eval = xi_interp_tab.s
    xi_interp_tab(s_eval)
    assert np.allclose(xi_interp_tab(s_eval[0]/2., bounds_error=False), xi_interp_tab(s_eval[0], bounds_error=False), atol=0.)
    assert np.allclose(xi_interp_tab(s_eval[-1]*2., bounds_error=False), xi_interp_tab(s_eval[-1], bounds_error=False), atol=0.)
    with pytest.raises(ValueError):
        xi_interp_tab(s_eval/2.)
    with pytest.raises(ValueError):
        xi_interp_tab(s_eval*2.)
    assert np.allclose(xi_interp_tab(s_eval), xi_interp_callable(s_eval), rtol=0.1)
    if plot:
        plt.plot(s_eval, s_eval**2*xi_interp_callable(s_eval), label='callable')
        plt.plot(s_eval, s_eval**2*xi_interp_tab(s_eval), label='tab')
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
    k = np.logspace(-4,2,1000)
    z = np.linspace(0,4,10)
    k_extrap = np.logspace(-6,3,1000)
    pk_interp_callable = fo.pk_interpolator(k=k_extrap, z=z)
    pk_interp_tab = PowerSpectrumInterpolator2D(k, z, pk_interp_callable(k, z), extrap_kmin=k_extrap[0], extrap_kmax=k_extrap[-1])
    k_eval = k_extrap[1:-1] # to avoid error with rounding
    z_eval = z
    assert np.allclose(pk_interp_tab(k_eval, z_eval), pk_interp_callable(k_eval, z_eval), atol=0, rtol=0.1)
    assert np.allclose(pk_interp_tab.extrap_kmin, pk_interp_callable.k[0])
    assert np.allclose(pk_interp_tab.extrap_kmax, pk_interp_callable.k[-1])
    pk_interp_callable(k_eval/2., z_eval)
    pk_interp_callable(k_eval*2., z_eval)
    pk_interp_tab(k_eval, z_eval)
    assert np.allclose(pk_interp_tab(k_eval[0]/2., z_eval, bounds_error=False), pk_interp_tab(k_extrap[0], z_eval, bounds_error=False), atol=0.)
    assert np.allclose(pk_interp_tab(k_eval[-1]*2., z_eval, bounds_error=False), pk_interp_tab(k_extrap[-1], z_eval, bounds_error=False), atol=0.)
    with pytest.raises(ValueError):
        pk_interp_tab(k_eval/2., z_eval)
    with pytest.raises(ValueError):
        pk_interp_tab(k_eval*2., z_eval)
    pk_interp_tab(k_eval, z_eval*2.)
    pk_interp_noextrapz = PowerSpectrumInterpolator2D(k, z, pk_interp_callable(k, z), extrap_kmin=k_extrap[0], extrap_kmax=k_extrap[-1], extrap_z=False)
    assert np.allclose(pk_interp_noextrapz(k_eval, z_eval[-1]*2., bounds_error=False), pk_interp_noextrapz(k_eval, z_eval[-1], bounds_error=False), atol=0.)
    with pytest.raises(ValueError):
        pk_interp_noextrapz(k_eval, z_eval[-1]*2.)

    if plot:
        plt.loglog(k_eval, pk_interp_callable(k_eval, z_eval), linestyle='-')
        plt.loglog(k_eval, pk_interp_tab(k_eval, z_eval), linestyle='--')
        plt.show()

    xi_interp_callable = pk_interp_callable.to_xi()
    xi_interp_tab = pk_interp_tab.to_xi()
    s_eval = xi_interp_tab.s
    xi_interp_tab(s_eval, z_eval)
    assert np.allclose(xi_interp_tab(s_eval[0]/2., z_eval, bounds_error=False), xi_interp_tab(s_eval[0], z_eval, bounds_error=False), atol=0.)
    assert np.allclose(xi_interp_tab(s_eval[-1]*2., z_eval, bounds_error=False), xi_interp_tab(s_eval[-1], z_eval, bounds_error=False), atol=0.)
    with pytest.raises(ValueError):
        xi_interp_tab(s_eval/2., z_eval)
    with pytest.raises(ValueError):
        xi_interp_tab(s_eval*2., z_eval)
    xi_interp_tab(s_eval, z_eval*2.)
    xi_interp_noextrapz = pk_interp_noextrapz.to_xi()
    assert np.allclose(xi_interp_noextrapz(s_eval, z_eval[-1]*2., bounds_error=False), xi_interp_noextrapz(s_eval, z_eval[-1], bounds_error=False), atol=0.)
    with pytest.raises(ValueError):
        xi_interp_noextrapz(s_eval, z_eval[-1]*2.)

    assert np.allclose(xi_interp_tab(s_eval, z_eval), xi_interp_callable(s_eval, z_eval), rtol=0.1)
    if plot:
        plt.plot(s_eval, s_eval[:,None]**2*xi_interp_callable(s_eval, z_eval), linestyle='-')
        plt.plot(s_eval, s_eval[:,None]**2*xi_interp_tab(s_eval, z_eval), linestyle='--')
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
