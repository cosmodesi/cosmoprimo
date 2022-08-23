import numpy as np

from cosmoprimo import Cosmology, Transfer, Fourier, PowerSpectrumBAOFilter, CorrelationFunctionBAOFilter


def plot_wiggles():
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    pk_interpolator = Fourier(cosmo, engine='eisenstein_hu').pk_interpolator().to_1d()
    xi_interpolator = pk_interpolator.to_xi()
    smooth_pk_interpolator = Fourier(cosmo, engine='eisenstein_hu_nowiggle').pk_interpolator().to_1d()

    k = np.geomspace(1e-4, 10., 1000)
    wiggles = pk_interpolator(k) / smooth_pk_interpolator(k)
    plt.plot(k, wiggles, label='truth')
    cosmo.set_engine('eisenstein_hu')

    for engine in ['hinton2017', 'savgol', 'ehpoly', 'wallish2018', 'brieden2022', 'peakaverage']:
        flt = PowerSpectrumBAOFilter(pk_interpolator, engine=engine, cosmo=cosmo, cosmo_fid=cosmo)
        plt.plot(k, pk_interpolator(k) / flt.smooth_pk_interpolator()(k), label=engine)
    #for engine in ['kirkby2013']:
    #    flt = CorrelationFunctionBAOFilter(xi_interpolator, engine=engine)
    #    plt.plot(k, pk_interpolator(k) / flt.smooth_pk_interpolator(extrap_pk='lin')(k), label=engine)
    plt.xscale('log')
    plt.ylim(0.92, 1.08)
    plt.legend()
    plt.show()


def plot_numerical_stability(engine='class'):
    from matplotlib import pyplot as plt
    from cosmoprimo.fiducial import DESI
    #cosmo_fid = DESI(engine=engine)
    cosmo_fid = Cosmology(engine=engine)
    #param, values = 'h', np.linspace(0.6726, 0.6746, 10)
    #param, values = 'h', np.linspace(0.67, 0.68, 10)
    #param, values = 'Omega_m', np.linspace(0.3, 0.4, 10)
    param, values = 'Omega_m', np.linspace(0.3, 0.302, 10)
    colors = plt.cm.jet(np.linspace(0, 1, len(values)))
    k = np.geomspace(1e-4, 10., 1000)
    #Omega_m = np.linspace(2.09e-9, 2.1e-9, 10)
    z = 0.
    kp = 0.03
    Ap = {}
    for engine in ['wallish2018', 'brieden2022', 'peakaverage']:
        flt = PowerSpectrumBAOFilter(Fourier(cosmo_fid).pk_interpolator().to_1d(z=z), cosmo=cosmo_fid, cosmo_fid=cosmo_fid, engine=engine)
        Ap[engine] = []
        for value, color in zip(values, colors):
            cosmo = cosmo_fid.clone(**{param: value})
            flt(Fourier(cosmo).pk_interpolator().to_1d(z=z), cosmo=cosmo)
            Ap[engine].append(flt.smooth_pk_interpolator()(kp))
            plt.plot(k, flt.pk_interpolator(k) / flt.smooth_pk_interpolator()(k))
        plt.xscale('log')
        plt.show()
    for engine in Ap:
        plt.plot(values, Ap[engine], label=engine)
    plt.show()


def plot_xi(engine='class'):
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    fo = Fourier(cosmo, engine=engine)
    pk_interpolator = fo.pk_interpolator().to_1d()
    xi_interpolator = pk_interpolator.to_xi()
    s = np.linspace(1e-2, 300, 1000)
    plt.plot(s, s ** 2 * xi_interpolator(s), label='input')
    for engine in ['kirkby2013']:
        flt = CorrelationFunctionBAOFilter(xi_interpolator, engine=engine)
        plt.plot(s, s ** 2 * flt.smooth_xi_interpolator()(s), label=engine)
    for engine in ['hinton2017', 'savgol', 'ehpoly', 'wallish2018', 'brieden2022', 'peakaverage']:
        flt = PowerSpectrumBAOFilter(pk_interpolator, engine=engine, cosmo=cosmo, cosmo_fid=cosmo)
        plt.plot(s, s ** 2 * flt.smooth_xi_interpolator()(s), label=engine)
    # plt.xscale('log')
    plt.legend()
    plt.show()


def plot_pk(engine='class'):
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    fo = Fourier(cosmo, engine=engine)
    pk_interpolator = fo.pk_interpolator().to_1d()
    xi_interpolator = pk_interpolator.to_xi()
    k = np.logspace(-5, 2, 1000)
    plt.loglog(k, pk_interpolator(k), label='input')
    for engine in ['hinton2017', 'savgol', 'ehpoly', 'wallish2018', 'brieden2022', 'peakaverage']:
        flt = PowerSpectrumBAOFilter(pk_interpolator, engine=engine, cosmo=cosmo, cosmo_fid=cosmo)
        # plt.loglog(k, flt.pk_interpolator(k), label=engine)
        plt.loglog(k, flt.smooth_pk_interpolator()(k), label=engine)
    for engine in ['kirkby2013']:
        flt = CorrelationFunctionBAOFilter(xi_interpolator, engine=engine)
        plt.loglog(k, flt.smooth_pk_interpolator(extrap_pk='lin')(k), label=engine)
    plt.legend()
    plt.show()


def test_2d_pk(engine='class'):
    cosmo = Cosmology()
    # fo = Fourier(cosmo, engine='eisenstein_hu')
    fo = Fourier(cosmo, engine=engine)
    pk_interpolator = fo.pk_interpolator()
    k = np.logspace(-3, 2, 1000)
    for engine in ['hinton2017', 'savgol', 'ehpoly', 'wallish2018', 'brieden2022', 'peakaverage']:
        flt = PowerSpectrumBAOFilter(pk_interpolator, engine=engine, cosmo=cosmo, cosmo_fid=cosmo)
        z = pk_interpolator.z
        smooth_pk = flt.smooth_pk_interpolator()(k, z=z)
        flt_1d = PowerSpectrumBAOFilter(pk_interpolator.to_1d(z=0), engine=engine, cosmo=cosmo, cosmo_fid=cosmo)
        for iz, zz in enumerate(z):
            pk_interpolator_1d = pk_interpolator.to_1d(z=zz)
            flt_1d = flt_1d(pk_interpolator_1d)
            #flt_1d = PowerSpectrumBAOFilter(pk_interpolator_1d, engine=engine, cosmo=cosmo, cosmo_fid=cosmo)
            #print(np.abs(smooth_pk[:, iz] / flt_1d.smooth_pk_interpolator()(k) - 1).max())
            assert np.allclose(smooth_pk[:, iz], flt_1d.smooth_pk_interpolator()(k), atol=1e-6, rtol=1e-6)


def plot_2d_pk(engine='class'):
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    fo = Fourier(cosmo, engine=engine)
    pk_interpolator = fo.pk_interpolator()
    #pk_interpolator = pk_interpolator.clone(pk=pk_interpolator.pk[:, [0]] / np.linspace(1., 5., pk_interpolator.pk.shape[1]))
    k = np.logspace(-5, 2, 1000)
    z = np.linspace(0., 4., 5)
    colors = plt.cm.jet(np.linspace(0, 1, len(z)))
    for zz, color in zip(z, colors):
        plt.loglog(k, pk_interpolator(k, z=zz), color=color)
    engines = ['brieden2022', 'peakaverage'][1:]
    for engine in engines:
        flt = PowerSpectrumBAOFilter(pk_interpolator, engine=engine, cosmo=cosmo, cosmo_fid=cosmo)
        for zz, color in zip(z, colors):
            plt.loglog(k, flt.smooth_pk_interpolator()(k, z=zz), color=color)
    plt.show()
    for engine in engines:
        flt = PowerSpectrumBAOFilter(pk_interpolator, engine=engine, cosmo=cosmo, cosmo_fid=cosmo)
        for zz, color in zip(z, colors):
            plt.plot(k, flt.pk_interpolator(k, z=zz) / flt.smooth_pk_interpolator()(k, z=zz), color=color)
        plt.xscale('log')
        plt.show()


def test_2d_xi(engine='class'):
    cosmo = Cosmology()
    # fo = Fourier(cosmo, engine='eisenstein_hu')
    fo = Fourier(cosmo, engine=engine)
    pk_interpolator = fo.pk_interpolator()
    xi_interpolator = pk_interpolator.to_xi()
    s = np.linspace(1e-2, 300, 1000)
    for engine in ['kirkby2013']:
        flt = CorrelationFunctionBAOFilter(xi_interpolator, engine=engine)
        z = xi_interpolator.z
        smooth_xi = flt.smooth_xi_interpolator()(s, z=z)
        flt_1d = CorrelationFunctionBAOFilter(xi_interpolator.to_1d(z=0), engine=engine)
        for iz, zz in enumerate(z):
            xi_interpolator_1d = pk_interpolator.to_1d(z=zz).to_xi()
            #flt_1d = CorrelationFunctionBAOFilter(xi_interpolator_1d, engine=engine)
            flt_1d = flt_1d(xi_interpolator_1d)
            assert np.allclose(smooth_xi[:, iz], flt_1d.smooth_xi_interpolator()(s), atol=1e-4, rtol=1e-3)


def plot_2d_xi(engine='class'):
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    fo = Fourier(cosmo, engine=engine)
    pk_interpolator = fo.pk_interpolator()
    xi_interpolator = pk_interpolator.to_xi()
    s = np.linspace(1e-2, 300, 1000)
    z = np.linspace(0., 4., 5)
    colors = plt.cm.jet(np.linspace(0, 1, len(z)))
    for zz, color in zip(z, colors):
        plt.plot(s, s ** 2 * xi_interpolator(s, z=zz), color=color)
    for engine in ['kirkby2013']:
        flt = CorrelationFunctionBAOFilter(xi_interpolator, engine=engine)
        for zz, color in zip(z, colors):
            plt.plot(s, s ** 2 * flt.smooth_xi_interpolator()(s, z=zz), color=color)
    plt.show()


def plot_wallish2018(engine='class'):
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    fo = Fourier(cosmo, engine=engine)
    pk_interpolator = fo.pk_interpolator().to_1d()
    flt = PowerSpectrumBAOFilter(pk_interpolator, engine='wallish2018')
    ind = np.arange(flt._even.size)
    mask = ind < 100
    plt.plot(ind[mask], ind[mask] * flt._dd_even.flat[mask], color='C0', linestyle='-', label='dd even')
    plt.plot(ind[mask], ind[mask] * flt._dd_odd.flat[mask], color='C1', linestyle='-', label='dd odd')
    plt.legend()
    plt.show()

    plt.plot(ind[mask], ind[mask] * flt._even.flat[mask], color='C0', linestyle=':', label='even')
    plt.plot(ind[mask], ind[mask] * flt._even_now.flat[mask], color='C0', linestyle='-', label='even no wiggle')
    plt.plot(ind[mask], ind[mask] * flt._odd.flat[mask], color='C1', linestyle=':', label='odd')
    plt.plot(ind[mask], ind[mask] * flt._odd_now.flat[mask], color='C1', linestyle='-', label='odd no wiggle')
    plt.legend()
    plt.show()
    k = np.logspace(-4, 2, 1000)
    plt.plot(k, pk_interpolator(k) / flt.smooth_pk_interpolator()(k), label='input')
    plt.xscale('log')
    plt.show()


def plot_brieden2022(engine='class'):
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    pk_interpolator = Fourier(cosmo, engine='eisenstein_hu').pk_interpolator(ignore_norm=True).to_1d()
    xi_interpolator = pk_interpolator.to_xi()
    smooth_pk_interpolator = Fourier(cosmo, engine='eisenstein_hu_nowiggle').pk_interpolator(ignore_norm=True).to_1d()

    k = np.geomspace(1e-4, 10., 1000)
    wiggles = pk_interpolator(k) / smooth_pk_interpolator(k)
    plt.plot(k, wiggles, label='truth')
    cosmo.set_engine(engine)

    flt = PowerSpectrumBAOFilter(pk_interpolator, engine='brieden2022', cosmo=cosmo, cosmo_fid=cosmo)
    plt.plot(flt.k_fid, flt.ratio_fid)
    plt.plot(flt.k_fid, flt.ratio_now_fid)
    plt.plot(k, pk_interpolator(k) / flt.smooth_pk_interpolator()(k), label=engine)
    plt.xscale('log')
    plt.ylim(0.92, 1.08)
    plt.legend()
    plt.show()


def plot_peakaverage(engine='class'):
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    pk_interpolator = Fourier(cosmo, engine='eisenstein_hu').pk_interpolator(ignore_norm=True).to_1d()
    xi_interpolator = pk_interpolator.to_xi()
    smooth_pk_interpolator = Fourier(cosmo, engine='eisenstein_hu_nowiggle').pk_interpolator(ignore_norm=True).to_1d()

    k = np.geomspace(1e-4, 10., 1000)
    wiggles = pk_interpolator(k) / smooth_pk_interpolator(k)
    plt.plot(k, wiggles, label='truth')
    cosmo.set_engine(engine)

    flt = PowerSpectrumBAOFilter(pk_interpolator, engine='peakaverage', cosmo=cosmo, cosmo_fid=cosmo)
    plt.plot(k, pk_interpolator(k) / flt.smooth_pk_interpolator()(k), label=engine)
    plt.xscale('log')
    plt.ylim(0.92, 1.08)
    plt.legend()
    plt.show()


if __name__ == '__main__':

    plot_2d_pk()
    exit()
    plot_pk()
    test_2d_pk()
    plot_2d_pk()
    plot_xi()
    test_2d_xi()
    plot_2d_xi()
    plot_wiggles()
    plot_numerical_stability()
    # plot_wallish2018()
    # plot_brieden2022()
    # plot_peakaverage()
