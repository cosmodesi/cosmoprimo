import os
import tempfile

import pytest
import numpy as np

from cosmoprimo import (Cosmology, Background, Thermodynamics, Primordial,
                        Harmonic, Fourier, CosmologyError, constants)


def test_params():
    cosmo = Cosmology()
    with pytest.raises(CosmologyError):
        cosmo = Cosmology(sigma8=1., A_s=1e-9)
    params = {'Omega_cdm': 0.3, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96}
    cosmo = Cosmology(**params)
    assert cosmo['omega_cdm'] == 0.3 * 0.8**2
    assert len(cosmo['z_pk']) == 60
    assert cosmo['sigma8'] == 0.8
    for neutrino_hierarchy in ['normal', 'inverted', 'degenerate']:
        cosmo = Cosmology(m_ncdm=0.1, neutrino_hierarchy=neutrino_hierarchy)
    m_ncdm = [0.01, 0.02, 0.05]
    cosmo = Cosmology(m_ncdm=m_ncdm)
    Background(cosmo, engine='class')
    Fourier(cosmo)

    with tempfile.TemporaryDirectory() as tmp_dir:
        fn = os.path.join(tmp_dir, 'cosmo.npy')
        cosmo.save(fn)
        cosmo = Cosmology.load(fn)

    assert np.allclose(cosmo['m_ncdm'], m_ncdm)
    assert cosmo.engine.__class__.__name__ == 'ClassEngine'
    Fourier(cosmo)


def test_engine():
    cosmo = Cosmology(engine='class')
    cosmo.set_engine(engine='camb')
    cosmo.set_engine(engine=cosmo.engine)
    ba = cosmo.get_background()
    ba = Background(cosmo)
    assert ba._engine is cosmo.engine
    ba = cosmo.get_background(engine='camb', set_engine=False)
    ba = Background(cosmo, engine='camb', set_engine=False)
    assert cosmo.engine is not ba._engine
    assert type(ba) is not type(Background(cosmo, engine='class'))
    assert type(cosmo.get_background()) is not type(cosmo.get_background(engine='camb'))
    assert type(cosmo.get_background()) is type(cosmo.get_background(engine='camb'))


list_params = [{}, {'sigma8': 1.}, {'A_s': 2e-9, 'alpha_s': 0.3}, {'lensing': True},
               {'m_ncdm': 0.1, 'neutrino_hierarchy': 'normal'}, {'Omega_k': 0.1},
               {'w0_fld': -0.9, 'wa_fld': 0.1, 'cs2_fld': 0.9}, {'w0_fld': -1.1, 'wa_fld': 0.2}]


@pytest.mark.parametrize('params', list_params)
def test_background(params, seed=42):

    rng = np.random.RandomState(seed=seed)
    cosmo = Cosmology(**params)
    if 'A_s' in params:
        assert cosmo['A_s'] == params['A_s']
        for name in ['ln10^{10}A_s', 'ln10^10A_s']:
            assert cosmo[name] == np.log(10**10 * cosmo['A_s'])
        with pytest.raises(CosmologyError):
            cosmo['sigma8']
    else:
        assert cosmo['sigma8'] == params.get('sigma8', 0.8)  # sigma8 is set as default
        with pytest.raises(CosmologyError):
            cosmo['A_s']

    for engine in ['class', 'camb', 'astropy', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants', 'bbks']:
        ba = cosmo.get_background(engine=engine)
        for name in ['T0_cmb', 'T0_ncdm', 'Omega0_cdm', 'Omega0_b', 'Omega0_k', 'Omega0_g', 'Omega0_ur', 'Omega0_r',
                     'Omega0_pncdm', 'Omega0_pncdm_tot', 'Omega0_ncdm', 'Omega0_ncdm_tot',
                     'Omega0_m', 'Omega0_Lambda', 'Omega0_fld', 'Omega0_de']:
            assert np.allclose(getattr(ba, name), cosmo[name.replace('0', '')], atol=0, rtol=1e-3)
            assert np.allclose(getattr(ba, name), getattr(ba, name.replace('0', ''))(0.), atol=0, rtol=1e-3)

        for name in ['H0', 'h', 'N_ur', 'N_ncdm', 'm_ncdm', 'm_ncdm_tot', 'N_eff', 'w0_fld', 'wa_fld', 'cs2_fld']:
            assert np.allclose(getattr(ba, name), cosmo[name], atol=1e-9, rtol=1e-8 if name not in ['N_eff'] else 1e-4)

    ba_class = Background(cosmo, engine='class')

    def assert_allclose(ba, name, atol=0, rtol=1e-4):
        test, ref = getattr(ba, name), getattr(ba_class, name)
        shape = (cosmo['N_ncdm'], ) if name.endswith('ncdm') else ()
        z = rng.uniform(0., 1., 10)
        assert np.allclose(test(z), ref(z), atol=atol, rtol=rtol)
        assert test(0.).shape == shape
        assert test([]).shape == shape + (0, )
        z = np.array(0.)
        assert test(z).dtype.itemsize == z.dtype.itemsize
        z = np.array([0., 1.])
        assert test(z).shape == shape + z.shape
        z = np.array([[0., 1.]] * 4, dtype='f4')
        assert test(z).shape == shape + z.shape
        assert test(z).dtype.itemsize == z.dtype.itemsize

    for engine in ['class', 'camb', 'astropy', 'eisenstein_hu']:
        ba = cosmo.get_background(engine=engine)
        for name in ['T_cmb', 'T_ncdm']:
            assert_allclose(ba, name, atol=0, rtol=1e-4)
        for name in ['rho_crit', 'p_ncdm', 'p_ncdm_tot', 'Omega_pncdm', 'Omega_pncdm_tot']:
            assert_allclose(ba, name, atol=0, rtol=1e-4)
        for density in ['rho', 'Omega']:
            for species in ['cdm', 'b', 'k', 'g', 'ur', 'r', 'ncdm', 'ncdm_tot', 'm', 'Lambda', 'fld', 'de']:
                name = '{}_{}'.format(density, species)
                assert_allclose(ba, name, atol=0, rtol=1e-4)

        names = ['efunc', 'hubble_function']
        for name in names:
            assert_allclose(ba, name, atol=0, rtol=2e-4)
        names = []
        rtol = 2e-4
        if engine in ['class', 'camb', 'astropy']:
            names += ['time', 'comoving_radial_distance', 'luminosity_distance', 'angular_diameter_distance', 'comoving_angular_distance']
        if engine in ['class']:
            names += ['growth_factor', 'growth_rate']
        if engine in ['eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants', 'bbks'] and not cosmo['N_ncdm'] and not cosmo._has_fld:
            rtol = 2e-2
            names += ['growth_factor', 'growth_rate']
        for name in names:
            assert_allclose(ba, name, atol=0, rtol=rtol)


@pytest.mark.parametrize('params', list_params)
def test_thermodynamics(params):
    cosmo = Cosmology(**params)
    th_class = Thermodynamics(cosmo, engine='class')

    for engine in ['camb']:
        th = Thermodynamics(cosmo, engine=engine)
        for name in ['z_drag', 'rs_drag', 'z_star', 'rs_star']:  # weirdly enough, class's z_rec seems to match camb's z_star much better
            assert np.allclose(getattr(th, name), getattr(th_class, name), atol=0, rtol=5e-3 if 'star' in name else 1e-4)
        for name in ['theta_star', 'theta_cosmomc']:
            assert np.allclose(getattr(th, name), getattr(th_class, name), atol=0, rtol=5e-3 if 'star' in name else 5e-5)
        assert np.allclose(th_class.theta_cosmomc, cosmo['theta_cosmomc'], atol=0., rtol=3e-6)
        assert np.allclose(th.theta_cosmomc, cosmo['theta_cosmomc'], atol=0., rtol=3e-6)
    for engine in ['eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants']:
        for name in ['z_drag', 'rs_drag']:
            assert np.allclose(getattr(th, name), getattr(th_class, name), atol=0, rtol=1e-2)


@pytest.mark.parametrize('params', list_params)
def test_primordial(params, seed=42):
    rng = np.random.RandomState(seed=seed)
    cosmo = Cosmology(**params)
    pm_class = Primordial(cosmo, engine='class')

    for engine in ['camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants', 'bbks']:
        pm = Primordial(cosmo, engine=engine)
        for name in ['n_s', 'alpha_s', 'k_pivot']:
            assert np.allclose(getattr(pm_class, name), cosmo['k_pivot'] / cosmo['h'] if name == 'k_pivot' else cosmo[name])
            assert np.allclose(getattr(pm, name), getattr(pm_class, name), atol=0, rtol=1e-5)

    for engine in ['camb']:
        pm = Primordial(cosmo, engine=engine)
        for name in ['A_s', 'ln_1e10_A_s']:
            assert np.allclose(getattr(pm, name), getattr(pm_class, name), atol=0, rtol=2e-3)
        k = rng.uniform(1e-3, 10., 100)
        for mode in ['scalar', 'tensor']:
            assert np.allclose(pm.pk_k(k, mode=mode), pm_class.pk_k(k, mode=mode), atol=0, rtol=2e-3)
            assert np.allclose(pm.pk_interpolator(mode=mode)(k), pm_class.pk_interpolator(mode=mode)(k), atol=0, rtol=2e-3)

    for engine in ['eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants', 'bbks']:
        pm = Primordial(cosmo, engine=engine)
        for name in ['A_s', 'ln_1e10_A_s']:
            assert np.allclose(getattr(pm, name), getattr(pm_class, name), atol=0, rtol=1e-1)

    k = np.logspace(-3, 1, 100)
    for engine in ['camb', 'class', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants', 'bbks']:
        pm = Primordial(cosmo, engine=engine)
        assert np.allclose(pm.pk_interpolator(mode='scalar')(k), (cosmo.h**3 * pm.A_s * (k / pm.k_pivot) ** (pm.n_s - 1. + 1. / 2. * pm.alpha_s * np.log(k / pm.k_pivot))))


@pytest.mark.parametrize('params', list_params)
def test_harmonic(params):
    cosmo = Cosmology(**params)
    hr_class = Harmonic(cosmo, engine='class')
    test = hr_class.unlensed_cl()
    ref = hr_class.unlensed_table(of=['tt', 'ee', 'bb', 'te'])
    assert all(np.allclose(test[key], ref[key]) for key in ref.dtype.names)

    for engine in ['camb']:
        hr = Harmonic(cosmo, engine=engine)
        for name in ['lensed_cl', 'lens_potential_cl']:
            for ellmax in [100, -1]:
                if not cosmo['lensing']:
                    if engine == 'class':
                        from pyclass import ClassBadValueError
                        with pytest.raises(ClassBadValueError):
                            getattr(hr, name)(ellmax=ellmax)
                    if engine == 'camb':
                        from camb import CAMBError
                        with pytest.raises(CAMBError):
                            getattr(hr, name)(ellmax=ellmax)
                else:
                    tmp_class = getattr(hr_class, name)(ellmax=ellmax)
                    tmp = getattr(hr, name)(ellmax=ellmax)
                    assert tmp_class.dtype == tmp.dtype
                    for field in tmp_class.dtype.names[1:]:
                        if name == 'lensed_cl':
                            atol = tmp_class[field].std() * 1e-2  # to deal with 0 oscillating behavior
                            rtol = 1e-2
                            # print(name, field, tmp_class[field], tmp_camb[field])
                            # print(hr_class.unlensed_cl(ellmax=ellmax)[field], hr_camb.unlensed_cl(ellmax=ellmax)[field])
                        else:
                            atol = tmp_class[field].std() * 1e-1  # to deal with 0 oscillating behavior
                            rtol = 1e-1
                        assert np.allclose(tmp[field], tmp_class[field], atol=atol, rtol=rtol)
        for name in ['unlensed_cl']:
            for ellmax in [100, -1]:
                tmp_class = getattr(hr_class, name)(ellmax=ellmax)
                tmp = getattr(hr, name)(ellmax=ellmax)
                assert tmp_class.dtype == tmp.dtype
                for field in tmp_class.dtype.names[1:]:
                    atol = tmp_class[field].std() * 1e-2
                    assert np.allclose(tmp[field], tmp_class[field], atol=atol, rtol=2e-2)


@pytest.mark.parametrize('params', list_params)
def test_fourier(params, seed=42):
    rng = np.random.RandomState(seed=seed)
    cosmo = Cosmology(**params)
    fo_class = Fourier(cosmo, engine='class', gauge='newtonian')

    for engine in ['class', 'camb']:
        fo = Fourier(cosmo, engine=engine)
        z = rng.uniform(0., 10., 20)
        r = rng.uniform(1., 10., 10)
        if 'sigma8' in cosmo.params:
            assert np.allclose(fo.sigma8_z(0, of='delta_m'), cosmo['sigma8'], rtol=1e-3)
        for of in ['delta_m', 'delta_cb', ('delta_cb', 'theta_cb'), 'theta_cb']:
            assert np.allclose(fo.sigma_rz(r, z, of=of), fo_class.sigma_rz(r, z, of=of), rtol=1e-3)
            assert np.allclose(fo.sigma8_z(z, of=of), fo_class.sigma8_z(z, of=of), rtol=1e-3)

        z = rng.uniform(0., 3., 1)
        k = rng.uniform(1e-3, 1., 20)

        for of in ['delta_m', 'delta_cb']:
            assert np.allclose(fo.pk_interpolator(non_linear=False, of=of)(k, z=z), fo_class.pk_interpolator(non_linear=False, of=of)(k, z=z), rtol=2.5e-3)
            assert np.allclose(fo.pk_interpolator(non_linear=False, of=of).sigma8_z(z=z), fo.sigma8_z(z, of=of), rtol=1e-4)

        z = np.linspace(0., 4., 5)
        for of in ['theta_cb', ('delta_cb', 'theta_cb')]:
            assert np.allclose(fo.pk_interpolator(non_linear=False, of=of)(k, z=z), fo_class.pk_interpolator(non_linear=False, of=of)(k, z=z), rtol=2.5e-3)

        # if not cosmo['N_ncdm']:
        z = rng.uniform(0., 10., 20)
        r = rng.uniform(1., 10., 10)
        pk = fo.pk_interpolator(of='delta_cb')

        for z in np.linspace(0.2, 4., 5):
            for r in np.linspace(2., 20., 5):
                for dz in [1e-3, 1e-2]:
                    rtol = 1e-3
                    # assert np.allclose(ba_class.growth_rate(z), pk_class.growth_rate_rz(r=r, z=z, dz=dz), atol=0, rtol=rtol)
                    f = fo.sigma_rz(r, z, of='theta_cb') / fo.sigma_rz(r, z, of='delta_cb')
                    assert np.allclose(f, pk.growth_rate_rz(r=r, z=z, dz=dz), atol=0, rtol=rtol)

    for engine in ['eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants', 'bbks']:
        fo = Fourier(cosmo, engine=engine)
        pk_class = fo_class.pk_interpolator(non_linear=False, of='delta_m')
        pk = fo.pk_interpolator()
        rtol = 0.3 if engine == 'bbks' else 0.15
        assert np.allclose(pk(k, z=z), pk_class(k, z=z), atol=0., rtol=rtol)
        r = rng.uniform(1., 10., 10)
        assert np.allclose(pk.growth_rate_rz(r=r, z=z), pk_class.growth_rate_rz(r=r, z=z), atol=0., rtol=0.15)


def test_pk_norm():
    cosmo = Cosmology(engine='class')
    power_prim = cosmo.get_primordial().pk_interpolator()
    z = 1.
    k = np.logspace(-3., 1., 1000)
    assert np.allclose(cosmo.sigma8_z(0, of='delta_m'), cosmo['sigma8'], rtol=1e-3)
    power = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
    pk = power(k)
    pk_prim = power_prim(k)
    k0 = power.k[0]
    tk = (pk / power_prim(k) / k / (power(k0) / power_prim(k0) / k0))**0.5

    potential_to_density = (3. * cosmo.Omega0_m * 100**2 / (2. * (constants.c / 1e3)**2 * k**2)) ** (-2)
    curvature_to_potential = 9. / 25. * 2. * np.pi**2 / k**3 / cosmo.h**3
    znorm = 10.
    normalized_growth_factor = cosmo.growth_factor(z) / cosmo.growth_factor(znorm) / (1 + znorm)
    pk_test = normalized_growth_factor**2 * tk**2 * potential_to_density * curvature_to_potential * pk_prim
    assert np.allclose(pk_test, pk, atol=0., rtol=1e-3)


def plot_primordial_power_spectrum():
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    pm_class = Primordial(cosmo, engine='class')
    pr_camb = Primordial(cosmo, engine='camb')
    pk = pm_class.pk_interpolator()
    k = np.logspace(-6, 2, 500)
    plt.loglog(k, pk(k), label='class')
    pk = pr_camb.pk_interpolator()
    k = np.logspace(-6, 2, 500)
    plt.loglog(k, pk(k), label='camb')
    plt.legend()
    plt.show()


def plot_harmonic():
    from matplotlib import pyplot as plt
    cosmo = Cosmology(lensing=True)
    hr_class = Harmonic(cosmo, engine='class')
    cls = hr_class.lensed_cl()
    ells_factor = (cls['ell'] + 1) * cls['ell'] / (2 * np.pi)
    plt.plot(cls['ell'], ells_factor * cls['tt'], label='class')
    hr_camb = Harmonic(cosmo, engine='camb')
    cls = hr_camb.lensed_cl()
    plt.plot(cls['ell'], ells_factor * cls['tt'], label='camb')
    cosmo = Cosmology(lensing=True, m_ncdm=0.1)
    hr_class = Harmonic(cosmo, engine='class')
    cls = hr_class.lensed_cl()
    plt.plot(cls['ell'], ells_factor * cls['tt'], label='class + neutrinos')
    hr_camb = Harmonic(cosmo, engine='camb')
    cls = hr_camb.lensed_cl()
    plt.plot(cls['ell'], ells_factor * cls['tt'], label='camb + neutrinos')
    plt.legend()
    plt.show()


def plot_non_linear():
    from matplotlib import pyplot as plt
    cosmo = Cosmology(non_linear='mead')
    k = np.logspace(-3, 1, 1000)
    z = 1.
    for of in ['delta_m', 'delta_cb']:
        for engine, color in zip(['class', 'camb'], ['C0', 'C1']):
            for non_linear, linestyle in zip([False, True], ['-', '--']):
                pk = cosmo.get_fourier(engine=engine).pk_interpolator(non_linear=non_linear, of=of)(k, z=z)
                plt.loglog(k, pk, color=color, linestyle=linestyle, label=engine + (' non-linear' if non_linear else ''))
        plt.legend()
        plt.show()

    for engine, color in zip(['class', 'camb'], ['C0', 'C1']):
        for non_linear, linestyle in zip([False, True], ['-', '--']):
            cosmo = Cosmology(lensing=True, non_linear='mead' if non_linear else '')
            cls = cosmo.get_harmonic(engine=engine).lens_potential_cl()
            ells_factor = (cls['ell'] + 1)**2 * cls['ell']**2 / (2 * np.pi)
            plt.plot(cls['ell'], ells_factor * cls['pp'], color=color, linestyle=linestyle, label=engine + (' non-linear' if non_linear else ''))
            plt.xscale('log')
    plt.legend()
    plt.show()


def plot_matter_power_spectrum():
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    fo_class = Fourier(cosmo, engine='class')
    # k, z, pk = fo_class.table()
    # plt.loglog(k, pk)
    z = 1.
    k = np.logspace(-6, 2, 500)
    pk = fo_class.pk_interpolator(non_linear=False, of='delta_m', extrap_kmin=1e-7)
    # pk = fo_class.pk_kz
    plt.loglog(k, pk(k, z=z), label='class')
    pk = Fourier(cosmo, engine='camb').pk_interpolator(non_linear=False, of='delta_m', extrap_kmin=1e-7)
    plt.loglog(k, pk(k, z=z), label='camb')
    pk = Fourier(cosmo, engine='eisenstein_hu').pk_interpolator()
    plt.loglog(k, pk(k, z=z), label='eisenstein_hu')
    pk = Fourier(cosmo, engine='eisenstein_hu_nowiggle').pk_interpolator()
    plt.loglog(k, pk(k, z=z), label='eisenstein_hu_nowiggle')
    pk = Fourier(cosmo, engine='eisenstein_hu_nowiggle_variants').pk_interpolator()
    plt.loglog(k, pk(k, z=z), label='eisenstein_hu_nowiggle_variants')
    plt.legend()
    plt.show()


def plot_eisenstein_hu_nowiggle_variants():
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    z = 1.
    k = np.logspace(-6, 2, 500)
    for m_ncdm in [0., 1.1]:
        cosmo = cosmo.clone(m_ncdm=m_ncdm, T_ncdm_over_cmb=None)
        pk = Fourier(cosmo, engine='eisenstein_hu_nowiggle_variants').pk_interpolator()
        plt.loglog(k, pk(k, z=z), label=r'$m_{{ncdm}} = {:.2f} \mathrm{{eV}}$'.format(m_ncdm))
    plt.legend()
    plt.show()


def test_external_camb():
    import camb
    from camb import CAMBdata

    params = camb.CAMBparams(H0=70, omch2=0.15, ombh2=0.02)
    params.DoLensing = True
    params.Want_CMB_lensing = True
    tr = CAMBdata()
    tr.calc_power_spectra(params)
    print(tr.get_lens_potential_cls(lmax=100, CMB_unit=None, raw_cl=True))

    params = camb.CAMBparams(H0=70, omch2=0.15, ombh2=0.02)
    As = params.InitPower.As
    ns = params.InitPower.ns
    params.DoLensing = False
    # params.Want_CMB_lensing = True
    # params.Want_CMB_lensing = True
    tr = camb.get_transfer_functions(params)
    tr.Params.InitPower.set_params(As=As, ns=ns)
    tr.calc_power_spectra()
    tr.Params.DoLensing = True
    tr.Params.Want_CMB_lensing = True
    print(tr.get_lens_potential_cls(lmax=100, CMB_unit=None, raw_cl=True))

    params = camb.CAMBparams(H0=70, omch2=0.15, ombh2=0.02)
    #params.WantCls = False
    params.Want_CMB = False
    # params.WantTransfer = True
    tr = camb.get_transfer_functions(params)
    params.Want_CMB = True
    tr.calc_power_spectra(params)
    print(tr.get_unlensed_scalar_cls(lmax=100, CMB_unit=None, raw_cl=True))
    # print(tr.get_total_cls(lmax=100, CMB_unit=None, raw_cl=True))


def test_external_pyccl():
    try: import pyccl
    except ImportError: return
    print('With pyccl')
    params = {'sigma8': 0.8, 'Omega_c': 0.28, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96, 'm_nu': 0.1, 'm_nu_type': 'normal'}
    cosmo = pyccl.Cosmology(**params)
    print(pyccl.background.growth_rate(cosmo, 1))
    params = {'sigma8': 0.8, 'Omega_cdm': 0.28, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96, 'm_ncdm': 0.1, 'neutrino_hierarchy': 'normal'}
    cosmo = Cosmology(**params)
    print(Background(cosmo, engine='class').growth_rate(0))

    params = {'sigma8': 0.8, 'Omega_c': 0.28, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96}
    cosmo = pyccl.Cosmology(**params)
    print(pyccl.background.growth_rate(cosmo, 1))
    params = {'sigma8': 0.8, 'Omega_cdm': 0.28, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96}
    cosmo = Cosmology(**params)
    print(Background(cosmo, engine='class').growth_rate(0))


def benchmark():
    import timeit
    import pyccl
    params = {'sigma8': 0.8, 'Omega_cdm': 0.28, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96, 'm_ncdm': 0.1, 'neutrino_hierarchy': 'normal'}
    pyccl_params = {'sigma8': 0.8, 'Omega_c': 0.28, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96, 'm_nu': 0.1, 'm_nu_type': 'normal', 'transfer_function': 'boltzmann_class'}
    z = np.linspace(0., 10., 10000)
    z_pk = 1.  # ccl does not support vectorization over scale factor
    k = np.logspace(-4, 2, 500)
    a = 1. / (1 + z)
    a_pk = 1. / (1 + z_pk)
    d = {}
    d['cosmoprimo initialisation'] = {'stmt': "Cosmology(**params)", 'number': 1000}
    d['pyccl initialisation'] = {'stmt': "pyccl.Cosmology(**pyccl_params)", 'number': 1000}

    d['cosmoprimo initialisation + background'] = {'stmt': "cosmo = Cosmology(**params); cosmo.get_background('class').comoving_radial_distance(z)",
                                                   'number': 10}
    d['pyccl initialisation + background'] = {'stmt': "cosmo = pyccl.Cosmology(**pyccl_params); pyccl.background.comoving_radial_distance(cosmo, a)",
                                              'number': 10}

    d['cosmoprimo initialisation + background single z'] = {'stmt': "cosmo = Cosmology(**params); cosmo.get_background('class').comoving_radial_distance(z_pk)",
                                                            'number': 10}
    d['pyccl initialisation + background single z'] = {'stmt': "cosmo = pyccl.Cosmology(**pyccl_params); pyccl.background.comoving_radial_distance(cosmo, a_pk)",
                                                       'number': 10}

    d['cosmoprimo initialisation + pk'] = {'stmt': "cosmo = Cosmology(**params); cosmo.get_fourier('class').pk_interpolator()(k, z_pk)",
                                           'number': 2}
    d['pyccl initialisation + pk'] = {'stmt': "cosmo = pyccl.Cosmology(**pyccl_params); pyccl.linear_matter_power(cosmo, k*cosmo['h'], a_pk)",
                                      'number': 2}

    cosmo = Cosmology(**params)
    pyccl_cosmo = pyccl.Cosmology(**pyccl_params)
    ba_class = cosmo.get_background('class')
    fo_class = cosmo.get_fourier('class')
    d['cosmoprimo background'] = {'stmt': "ba_class.comoving_radial_distance(z)", 'number': 100}
    d['pyccl background'] = {'stmt': "pyccl.background.comoving_radial_distance(pyccl_cosmo, a)", 'number': 100}
    d['cosmoprimo pk'] = {'stmt': "fo_class.pk_interpolator()(k, z_pk)", 'number': 2}
    d['pyccl pk'] = {'stmt': "pyccl.linear_matter_power(pyccl_cosmo, k*pyccl_cosmo['h'], a_pk)", 'number': 2}

    for key, value in d.items():
        dt = timeit.timeit(**value, globals={**globals(), **locals()}) / value['number'] * 1e3
        print('{} takes {:.3f} milliseconds'.format(key, dt))


def test_repeats():
    import timeit
    params = {'sigma8': 0.8, 'Omega_cdm': 0.28, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96, 'm_ncdm': 0.1, 'neutrino_hierarchy': 'normal'}
    cosmo = Cosmology(**params)
    fo_class = cosmo.get_fourier('class')
    d = {}
    for section in ['background', 'fourier']:
        d['init {}'.format(section)] = {'stmt': "c = Cosmology(**params); c.get_{}('class')".format(section), 'number': 2}
        d['get {}'.format(section)] = {'stmt': "cosmo.get_{}()".format(section), 'number': 100}

    for key, value in d.items():
        dt = timeit.timeit(**value, globals={**globals(), **locals()}) / value['number'] * 1e3
        print('{} takes {: .3f} milliseconds'.format(key, dt))


def test_neutrinos():
    from cosmoprimo import constants
    from cosmoprimo.cosmology import _compute_ncdm_momenta

    T_eff = constants.TCMB * constants.TNCDM_OVER_CMB
    pncdm = _compute_ncdm_momenta(T_eff, 1e-14, z=0, epsrel=1e-7, out='p')
    rhoncdm = _compute_ncdm_momenta(T_eff, 1e-14, z=0, epsrel=1e-7, out='rho')
    assert np.allclose(3. * pncdm, rhoncdm, rtol=1e-6)

    for m_ncdm in [0.06, 0.1, 0.2, 0.4]:
        # print(_compute_ncdm_momenta(T_eff, m_ncdm, z=0, out='rho'), _compute_ncdm_momenta(T_eff, m_ncdm, z=0, out='p'))
        omega_ncdm = _compute_ncdm_momenta(T_eff, m_ncdm, z=0, out='rho') / constants.rho_crit_Msunph_per_Mpcph3
        assert np.allclose(omega_ncdm, m_ncdm / 93.14, rtol=1e-3)
        domega_over_dm = _compute_ncdm_momenta(T_eff, m_ncdm, out='drhodm', z=0) / constants.rho_crit_Msunph_per_Mpcph3
        assert np.allclose(domega_over_dm, 1. / 93.14, rtol=1e-3)

    for m_ncdm in [0.06, 0.1, 0.2, 0.4]:
        cosmo = Cosmology(m_ncdm=m_ncdm)
        # print(m_ncdm, cosmo['Omega_ncdm'], sum(cosmo['m_ncdm'])/(93.14*cosmo['h']**2))
        assert np.allclose(cosmo['Omega_ncdm'], sum(cosmo['m_ncdm']) / (93.14 * cosmo['h']**2), rtol=1e-3)
        cosmo = Cosmology(Omega_ncdm=cosmo['Omega_ncdm'])
        assert np.allclose(cosmo['m_ncdm'], m_ncdm)


def test_clone():

    cosmo = Cosmology(omega_cdm=0.2, engine='class')
    engine = cosmo.engine

    for factor in [1., 1.1]:
        cosmo_clone = cosmo.clone(omega_cdm=cosmo['omega_cdm'] * factor)
        assert type(cosmo_clone.engine) == type(engine)
        assert cosmo_clone.engine is not engine
        z = np.linspace(0.5, 2., 100)
        test = np.allclose(cosmo_clone.get_background().comoving_radial_distance(z), cosmo.get_background().comoving_radial_distance(z))
        if factor == 1:
            assert test
        else:
            assert not test
        cosmo_clone = cosmo.clone(sigma8=cosmo.sigma8_m * factor)
        assert np.allclose(cosmo_clone.get_fourier().sigma_rz(8, 0, of='delta_m'), cosmo.sigma8_m * factor, rtol=1e-4)  # interpolation error
        cosmo_clone = cosmo.clone(h=cosmo.h * factor)
        assert np.allclose(cosmo_clone.Omega0_m, cosmo.Omega0_m)
        cosmo_clone = cosmo.clone(base='input', h=cosmo.h * factor)
        assert np.allclose(cosmo_clone.Omega0_cdm, cosmo.Omega0_cdm / factor**2)


def test_shortcut():
    cosmo = Cosmology()
    z = [0.1, 0.3]
    with pytest.raises(AttributeError):
        d = cosmo.comoving_radial_distance(z)
    assert 'tau_reio' not in dir(cosmo)
    cosmo.set_engine('class')
    assert 'tau_reio' in dir(cosmo)
    assert 'table' not in dir(cosmo)
    assert 'table' in dir(Fourier(cosmo))
    d = cosmo.comoving_radial_distance(z)
    assert np.all(d == cosmo.get_background().comoving_radial_distance(z))
    assert cosmo.gauge == 'synchronous'  # default
    cosmo.set_engine('class', gauge='newtonian')
    assert cosmo.gauge == 'newtonian'


def test_theta_cosmomc():

    cosmo = Cosmology(engine='camb')
    from cosmoprimo.cosmology import _compute_rs_cosmomc

    rs, zstar = _compute_rs_cosmomc(cosmo.Omega0_b * cosmo.h**2, cosmo.Omega0_m * cosmo.h**2, cosmo.hubble_function)
    theta_cosmomc = rs * cosmo.h / cosmo.comoving_angular_distance(zstar)
    assert np.allclose(theta_cosmomc, cosmo.theta_cosmomc, atol=0., rtol=2e-6)


def test_isitgr():

    cosmo_camb = Cosmology(engine='camb')
    try:
        cosmo = Cosmology(engine='isitgr')
    except ImportError:
        return

    k = np.linspace(0.01, 1., 200)
    z = np.linspace(0., 2., 10)
    assert np.allclose(cosmo_camb.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=1e-4)

    cosmo = Cosmology(engine='isitgr', parameterization='mueta', E11=-0.5, E22=-0.5)
    assert not np.allclose(cosmo_camb.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=1e-4)
    cosmo.comoving_radial_distance(z)



if __name__ == '__main__':

    test_params()
    test_engine()
    for params in list_params:
        test_background(params)
        test_thermodynamics(params)
        test_primordial(params)
        test_harmonic(params)
        test_fourier(params)

    test_repeats()
    test_neutrinos()
    test_clone()
    test_shortcut()
    test_pk_norm()
    # plot_non_linear()
    # plot_primordial_power_spectrum()
    # plot_harmonic()
    # plot_matter_power_spectrum()
    # plot_eisenstein_hu_nowiggle_variants()
    # test_external_camb()
    test_external_pyccl()
    test_isitgr()
