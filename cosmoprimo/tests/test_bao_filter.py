import numpy as np
from scipy import optimize

from cosmoprimo import Cosmology, Transfer, Fourier, PowerSpectrumBAOFilter, CorrelationFunctionBAOFilter
from cosmoprimo.utils import SolveLeastSquares


def test_utils():

    x = np.linspace(1,100,10)
    gradient = np.array([1./x,np.ones_like(x),x,x**2,x**3])

    covs = [np.diag(x),np.diag(x)+0.1]
    rng = np.random.RandomState(seed=42)
    y = rng.uniform(0.,1.,x.size)

    for cov in covs:

        precision = np.linalg.inv(cov)

        def chi2(pars):
            delta = y - pars.dot(gradient)
            return np.sum(delta.dot(precision).dot(delta.T))

        x0 = np.zeros(len(gradient))
        result_ref = optimize.minimize(chi2,x0=x0,args=(),method='Nelder-Mead',tol=1e-6,options={'maxiter':1000000}).x

        sls = SolveLeastSquares(gradient,precision)
        result = sls(y)
        assert np.allclose(result,result_ref,rtol=1e-2,atol=1e-2)

    result_ref = SolveLeastSquares(gradient,precision=np.eye(x.size))(y)
    for precision in [1.,np.ones_like(x)]:
        result = SolveLeastSquares(gradient,precision=precision)(y)
        assert np.allclose(result,result_ref)

    sls = SolveLeastSquares(gradient,precision=np.eye(x.size))
    result_ref = sls(y)
    ys = np.array([y]*12)
    result = sls(ys)
    assert result.shape == (len(ys),len(gradient))
    assert np.allclose(result,result_ref)
    assert sls.model().shape == ys.shape
    assert sls.chi2().shape == (len(ys),)

    gradient = np.ones_like(x)
    sls = SolveLeastSquares(gradient,precision=np.eye(x.size))
    assert sls(y).ndim == 0
    assert sls(ys).shape == (len(ys),)


def plot_transfer():
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    tr = Transfer(cosmo,engine='eisenstein_hu')
    k = np.logspace(-5,1,1000)
    plt.plot(k,tr.transfer_k(k),label='w/ baryons')
    plt.plot(k,tr.transfer_k(k,frac_baryon=0.),label='w/o baryons')
    plt.xscale('log')
    plt.legend()
    plt.show()


def plot_wiggles():
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    #pk_interpolator = Fourier(cosmo,engine='eisenstein_hu').pk_interpolator().to_1d()
    #smooth_pk_interpolator = Fourier(cosmo,engine='eisenstein_hu_nowiggle').pk_interpolator().to_1d()

    fo = Fourier(cosmo,engine='eisenstein_hu')
    pk_interpolator = fo.pk_interpolator().to_1d()
    xi_interpolator = pk_interpolator.to_xi()
    #wiggles = pk_interpolator(k)/smooth_pk_interpolator(k)

    tr = Transfer(cosmo,engine='eisenstein_hu')
    k = np.logspace(-4,2,1000)
    wiggles = (tr.transfer_k(k)/tr.transfer_k(k,frac_baryon=0.))**2
    wiggles /= wiggles[-1]

    plt.plot(k,wiggles,label='truth')
    for engine in ['hinton2017','savgol','ehpoly','wallish2018']:
        flt = PowerSpectrumBAOFilter(pk_interpolator,engine=engine)
        plt.plot(k,pk_interpolator(k)/flt.smooth_pk_interpolator()(k),label=engine)
    for engine in ['kirkby2013']:
        flt = CorrelationFunctionBAOFilter(xi_interpolator,engine=engine)
        plt.plot(k,pk_interpolator(k)/flt.smooth_pk_interpolator(extrap_pk='lin')(k),label=engine)
    plt.xscale('log')
    plt.ylim(0.92,1.08)
    plt.legend()
    plt.show()


def plot_xi(engine='class'):
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    fo = Fourier(cosmo,engine=engine)
    pk_interpolator = fo.pk_interpolator().to_1d()
    xi_interpolator = pk_interpolator.to_xi()
    s = np.linspace(1e-2,300,1000)
    plt.plot(s,s**2*xi_interpolator(s),label='input')
    for engine in ['kirkby2013']:
        flt = CorrelationFunctionBAOFilter(xi_interpolator,engine=engine)
        plt.plot(s,s**2*flt.smooth_xi_interpolator()(s),label=engine)
    for engine in ['hinton2017','savgol','ehpoly','wallish2018']:
        flt = PowerSpectrumBAOFilter(pk_interpolator,engine=engine)
        plt.plot(s,s**2*flt.smooth_xi_interpolator()(s),label=engine)
    #plt.xscale('log')
    plt.legend()
    plt.show()


def plot_pk(engine='class'):
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    fo = Fourier(cosmo,engine=engine)
    pk_interpolator = fo.pk_interpolator().to_1d()
    xi_interpolator = pk_interpolator.to_xi()
    k = np.logspace(-5,2,1000)
    plt.loglog(k,pk_interpolator(k),label='input')
    for engine in ['hinton2017','savgol','ehpoly','wallish2018']:
        flt = PowerSpectrumBAOFilter(pk_interpolator,engine=engine)
        #plt.loglog(k,flt.pk_interpolator(k),label=engine)
        plt.loglog(k,flt.smooth_pk_interpolator()(k),label=engine)
    for engine in ['kirkby2013']:
        flt = CorrelationFunctionBAOFilter(xi_interpolator,engine=engine)
        plt.loglog(k,flt.smooth_pk_interpolator(extrap_pk='lin')(k),label=engine)
    plt.legend()
    plt.show()


def test_2d_pk():
    cosmo = Cosmology()
    #fo = Fourier(cosmo,engine='eisenstein_hu')
    fo = Fourier(cosmo,engine='class')
    pk_interpolator = fo.pk_interpolator()
    k = np.logspace(-3,2,1000)
    for engine in ['hinton2017','savgol','ehpoly','wallish2018']:
        flt = PowerSpectrumBAOFilter(pk_interpolator,engine=engine)
        z = pk_interpolator.z
        smooth_pk = flt.smooth_pk_interpolator()(k,z=z)
        for iz,z_ in enumerate(z):
            pk_interpolator_1d = pk_interpolator.to_1d(z=z_)
            flt_1d = PowerSpectrumBAOFilter(pk_interpolator_1d,engine=engine)
            assert np.allclose(smooth_pk[:,iz],flt_1d.smooth_pk_interpolator()(k),atol=1e-4,rtol=1e-3)


def test_2d_xi():
    cosmo = Cosmology()
    #fo = Fourier(cosmo,engine='eisenstein_hu')
    fo = Fourier(cosmo,engine='class')
    pk_interpolator = fo.pk_interpolator()
    xi_interpolator = pk_interpolator.to_xi()
    s = np.linspace(1e-2,300,1000)
    for engine in ['kirkby2013']:
        flt = CorrelationFunctionBAOFilter(xi_interpolator,engine=engine)
        z = xi_interpolator.z
        smooth_xi = flt.smooth_xi_interpolator()(s,z=z)
        for iz,z_ in enumerate(z):
            xi_interpolator_1d = pk_interpolator.to_1d(z=z_).to_xi()
            flt_1d = CorrelationFunctionBAOFilter(xi_interpolator_1d,engine=engine)
            assert np.allclose(smooth_xi[:,iz],flt_1d.smooth_xi_interpolator()(s),atol=1e-4,rtol=1e-3)


def plot_2d_xi():
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    #fo = Fourier(cosmo,engine='eisenstein_hu')
    fo = Fourier(cosmo,engine='class')
    pk_interpolator = fo.pk_interpolator()
    xi_interpolator = pk_interpolator.to_xi()
    s = np.linspace(1e-2,300,1000)
    z = 0.
    plt.plot(s,s**2*xi_interpolator(s,z=z),label='input')
    for engine in ['kirkby2013']:
        flt = CorrelationFunctionBAOFilter(xi_interpolator,engine=engine)
        plt.plot(s,s**2*flt.smooth_xi_interpolator()(s,z=z),label=engine)
    plt.legend()
    plt.show()


def plot_wallish2018():
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    fo = Fourier(cosmo,engine='class')
    pk_interpolator = fo.pk_interpolator().to_1d()
    flt = PowerSpectrumBAOFilter(pk_interpolator,engine='wallish2018')

    ind = np.arange(flt._even.size)
    mask = ind < 100
    plt.plot(ind[mask],ind[mask]*flt._dd_even[mask],color='C0',linestyle='-',label='dd even')
    plt.plot(ind[mask],ind[mask]*flt._dd_odd[mask],color='C1',linestyle='-',label='dd odd')
    plt.legend()
    plt.show()

    plt.plot(ind[mask],ind[mask]*flt._even[mask],color='C0',linestyle=':',label='even')
    plt.plot(ind[mask],ind[mask]*flt._even_now[mask],color='C0',linestyle='-',label='even no wiggle')
    plt.plot(ind[mask],ind[mask]*flt._odd[mask],color='C1',linestyle=':',label='odd')
    plt.plot(ind[mask],ind[mask]*flt._odd_now[mask],color='C1',linestyle='-',label='odd no wiggle')
    plt.legend()
    plt.show()

    k = np.logspace(-4,2,1000)
    plt.plot(k,pk_interpolator(k)/flt.smooth_pk_interpolator()(k),label='input')
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':

    test_utils()
    test_2d_pk()
    test_2d_xi()
    #plot_transfer()
    #plot_wiggles()
    #plot_pk()
    #plot_xi()
    #plot_wallish2018()
