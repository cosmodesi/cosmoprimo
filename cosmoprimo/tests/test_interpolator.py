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
    assert np.isscalar(interp(0.1))
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
    assert np.isscalar(interp(k[0],z=0))
    interp2 = interp.clone()
    assert np.all(interp2(k,z=[0]*2) == interp(k,z=[0]*2))

    z = np.random.uniform(0.,1.,10)
    interp = PowerSpectrumInterpolator2D(k,z=z,pk=np.array([pk]*len(z)).T)
    assert np.allclose(interp(k,z=np.random.uniform(0.,1.,10)),pk[:,None],atol=0,rtol=1e-5)

    cosmo = Cosmology()
    fo = Fourier(cosmo,engine='eisenstein_hu')
    pk_interp = fo.pk_interpolator()
    k = np.logspace(-4,2,100)
    z = np.linspace(0,4,10)
    pk = pk_interp(k,z)
    pk_interp2 = pk_interp.clone()
    assert np.allclose(pk_interp2(k,z),pk,rtol=1e-4)
    pk_interp2 = pk_interp.clone(pk=2*pk_interp.pk)
    assert np.allclose(pk_interp2(k,z),2*pk,rtol=1e-4)
    for iz,z_ in enumerate(z):
        assert np.allclose(pk_interp.to_1d(z=z_)(k),pk[:,iz])

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

    for z in [0.,0.1,1.,3.]:
        pk_interp = fo.pk_interpolator().to_1d(z=z)
        pk_interp2 = pk_interp.to_xi().to_pk()
        assert np.allclose(pk_interp(k),pk_interp2(k),rtol=1e-2)
        pk_interp2 = pk_interp.to_xi().clone().to_pk()
        assert np.allclose(pk_interp(k),pk_interp2(k),rtol=1e-2)


def plot_correlation_function():
    cosmo = Cosmology()
    fo = Fourier(cosmo,engine='eisenstein_hu')
    pk_interp = fo.pk_interpolator()
    xi_interp = pk_interp.to_xi()

    s = np.logspace(-2,4,1000)
    z = np.linspace(0,4,10)

    from matplotlib import pyplot as plt
    plt.plot(s,s[:,None]**2*xi_interp(s,z=z))
    plt.xscale('log')
    plt.legend()
    plt.show()

    pk_interp2 = xi_interp.to_pk()
    k = np.logspace(-4,2,100)
    plt.loglog(k,pk_interp(k,z=z))
    plt.loglog(k,pk_interp2(k,z=z),linestyle=':')
    plt.show()


if __name__ == '__main__':

    test_power_spectrum()
    test_correlation_function()
    #plot_correlation_function()
