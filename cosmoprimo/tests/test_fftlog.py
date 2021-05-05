import numpy as np
from scipy import integrate, interpolate

from cosmoprimo import Cosmology, Fourier
from cosmoprimo.fftlog import PowerToCorrelation, CorrelationToPower, TophatVariance, GaussianVariance, HankelTransform, pad


def wtophat_scalar(x):
    if x < 0.1:
        x2 = x**2
        return 1. + x2*(-1.0/10.0 + x2*(1.0/280.0 + x2*(-1.0/15120.0 + x2*(1.0/1330560.0 + x2*(-1.0/172972800.0)))))
    return 3.*(np.sin(x) - x*np.cos(x))/x**3


@np.vectorize
def _sigma_r(r, pk, kmin=1e-6, kmax=100, epsrel=1e-5):

    def integrand(logk):
        k = np.exp(logk)
        return pk(k)*(wtophat_scalar(r*k)*k)**2*k

    sigmasq = 1./2./np.pi**2*integrate.quad(integrand,np.log(kmin),np.log(kmax),epsrel=epsrel)[0]
    return np.sqrt(sigmasq)


def test_pad():

    a = b = np.ones((6, 6))
    padded_a = np.zeros((13, 6))
    padded_a[3:9, :] = 1
    padded_b = np.ones((6, 13))
    c = np.array([(i+1)*np.logspace(-3, 3, num=6, endpoint=False) for i in range(3)]).T
    padded_c = np.array([(i+1)*np.logspace(-12, 12, num=24, endpoint=False) for i in range(3)]).T

    assert np.allclose(pad(a, (3,4), extrap=0, axis=0), padded_a)
    assert np.allclose(pad(b, (4,3), extrap='edge', axis=1), padded_b)
    assert np.allclose(pad(c, (9,9), extrap='log', axis=0), padded_c)

    x = np.logspace(-3, 3, num=7, endpoint=True)
    padded_x = np.logspace(-15, 16, num=32, endpoint=True)
    y = np.logspace(-4, 2, num=7, endpoint=True)
    padded_y = np.logspace(-17, 14, num=32, endpoint=True)

    fftlog = HankelTransform(x, minfolds=3, xy=1, lowring=False)

    assert np.allclose(fftlog.padded_x, padded_x)
    assert np.allclose(fftlog.padded_y, padded_y)

    assert np.allclose(pad(x, (fftlog.padded_size_in_left,fftlog.padded_size_in_right), extrap='log'), padded_x)
    assert np.allclose(pad(y, (fftlog.padded_size_out_left,fftlog.padded_size_out_right), extrap='log'), padded_y)

    assert np.allclose(fftlog.padded_x[0,fftlog.padded_size_in_left:fftlog.padded_size_in_left+fftlog.size], x)
    assert np.allclose(fftlog.padded_y[0,fftlog.padded_size_out_left:fftlog.padded_size_out_left+fftlog.size], y)


def test_fftlog():

    def ffun(x): return 1 / (1 + x**2)**1.5
    def gfun(y): return np.exp(-y)

    for engine in ['numpy','fftw']:
        x = np.logspace(-3, 3, num=60, endpoint=False)
        f = ffun(x)
        hf = HankelTransform(x, nu=0, q=1, lowring=True, engine=engine)
        y, g = hf(f, extrap='log')
        assert np.allclose(g, gfun(y), rtol=1e-8, atol=1e-8)
        hf.inv()
        x2, f2 = hf(g, extrap='log')
        assert np.allclose(f2, f, rtol=1e-7, atol=1e-7)

        y = np.logspace(-4, 2, num=60, endpoint=False)
        g = gfun(y)
        hg = HankelTransform(y, nu=0, q=1, lowring=True, engine=engine)
        x, f = hg(g, extrap='log')
        assert np.allclose(f, ffun(x), rtol=1e-10, atol=1e-10)

        y = np.array([np.logspace(-4, 2, num=60, endpoint=False)]*3)
        scales = np.linspace(1.,3.,3)
        g = gfun(y)
        x, f = hg(g*scales[:,None], extrap='log')
        assert x.shape == (60,)
        assert f.shape == (3,60)
        assert np.allclose(f/scales[:,None], ffun(x), rtol=1e-10, atol=1e-10)


def test_power_to_correlation():
    cosmo = Cosmology()
    fo = Fourier(cosmo,engine='eisenstein_hu')
    pk_interp = fo.pk_interpolator().to_1d(z=0)
    k = np.logspace(-5,2,1000)
    pk = pk_interp(k)
    multipoles = []
    ells = [0,2,4]
    for ell in ells:
        s,xi = PowerToCorrelation(k,ell=ell,lowring=True)(pk)
        assert xi.shape == (1000,)
        k2,pk2 = CorrelationToPower(s,ell=ell,lowring=True)(xi)
        idx = (1e-2 < k2) & (k2 < 10.)
        assert np.allclose(pk2[idx],pk_interp(k2[idx]),rtol=1e-2)
        multipoles.append(xi)
    assert np.allclose(PowerToCorrelation(k,ell=ells,lowring=True,q=1.5)(pk)[-1],multipoles)


def test_sigmar():
    cosmo = Cosmology()
    fo = Fourier(cosmo,engine='eisenstein_hu')
    pk_interp = fo.pk_interpolator().to_1d(z=0)
    r = np.linspace(1.,20.,10)
    sigmar_ref = _sigma_r(r,pk_interp)
    k = np.logspace(-5,2,1000)
    pk = pk_interp(k)
    r2,sigmar2 = TophatVariance(k,lowring=True)(pk)
    sigmar = np.sqrt(interpolate.CubicSpline(r2,sigmar2)(r))
    assert np.allclose(sigmar,sigmar_ref,rtol=1e-5)
    sigmar = pk_interp.sigma_r(r)
    assert np.allclose(sigmar,sigmar_ref,rtol=1e-5)


def external_test_mcfit():
    cosmo = Cosmology()
    fo = Fourier(cosmo,engine='eisenstein_hu')
    k = np.logspace(-5,2,1000)
    pk = fo.pk_interpolator()(k,z=0)
    s,xi = PowerToCorrelation(k,ell=0,lowring=True,q=1.5)(pk)
    import mcfit
    s_ref,xi_ref = mcfit.P2xi(k,l=0,lowring=True,q=1.5)(pk)
    assert np.allclose(s,s_ref)
    assert np.allclose(xi,xi_ref)
    k,pk = CorrelationToPower(s,ell=0,lowring=True,q=1.5)(xi)
    k_ref,pk_ref = mcfit.xi2P(s,l=0,lowring=True,q=1.5)(xi)
    assert np.allclose(k,k_ref)
    assert np.allclose(pk,pk_ref)
    var = TophatVariance(k,lowring=True,q=1.5)(pk)
    var_ref = mcfit.TophatVar(k,lowring=True,q=1.5)(pk)
    assert np.allclose(var,var_ref)
    var = GaussianVariance(k,lowring=True,q=1.5)(pk)
    var_ref = mcfit.GaussVar(k,lowring=True,q=1.5)(pk)
    assert np.allclose(var,var_ref)


def benchmark():
    cosmo = Cosmology()
    fo = Fourier(cosmo,engine='eisenstein_hu')
    k = np.logspace(-5,2,1000)
    pk = fo.pk_interpolator()(k,z=0)
    pk = np.array([pk]*2)
    import timeit
    import mcfit
    d = {}
    d['full fftlog w/ numpy'] = {'stmt':'PowerToCorrelation(k,ell=0,lowring=True,q=1.5)(pk)','number':1000}
    d['full mcfit w/ numpy'] = {'stmt':'mcfit.P2xi(k,l=0,lowring=True,q=1.5)(pk)','number':1000}
    d['call fftlog w/ numpy'] = {'stmt':'fftlog(pk)',
                                'setup':'fftlog = PowerToCorrelation(k,ell=0,lowring=True,q=1.5)','number':10000}
    d['call fftlog w/ fftw'] = {'stmt':'fftlog(pk)',
                                'setup':'fftlog = PowerToCorrelation(k,ell=0,lowring=True,q=1.5,engine="fftw")','number':10000}
    d['call mcfit w/ numpy'] = {'stmt':'fftlog(pk)',
                                'setup':'fftlog = mcfit.P2xi(k,l=0,lowring=True,q=1.5)','number':10000}
    for key,value in d.items():
        dt = timeit.timeit(**value,globals={**globals(),**locals()})/value['number']*1e3
        print('{} takes {:.3f} milliseconds'.format(key,dt))


def plot():
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    fo = Fourier(cosmo,engine='eisenstein_hu')
    k = np.logspace(-5,2,1000)
    pk = fo.pk_interpolator()(k,z=0)
    q = 1.5
    s,xi = PowerToCorrelation(k,ell=0,lowring=True,q=q)(pk)
    plt.plot(s,s**2*xi)
    plt.xscale('log')
    plt.show()
    k2,pk2 = CorrelationToPower(s,ell=0,lowring=True,q=q)(xi)
    plt.plot(k2,pk2/pk)
    plt.xscale('log')
    plt.plot()
    plt.show()


if __name__ == '__main__':

    test_pad()
    test_fftlog()
    test_power_to_correlation()
    test_sigmar()
    #external_test_mcfit()
    #benchmark()
    #plot()
