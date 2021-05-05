import pytest
import numpy as np

from cosmoprimo import Cosmology, Background, Thermodynamics, Primordial, Transfer, Harmonic, Fourier, CosmologyError


def test_params():
    cosmo = Cosmology()
    with pytest.raises(CosmologyError):
        cosmo = Cosmology(sigma8=1.,A_s=1e-9)
    params = {'Omega_cdm':0.3,'Omega_b':0.02,'h':0.8,'n_s':0.96}
    cosmo = Cosmology(**params)
    assert cosmo['omega_cdm'] == 0.3*0.8**2
    assert len(cosmo['z_pk']) == 60
    assert cosmo['sigma8'] == 0.8
    for neutrino_hierarchy in ['normal','inverted','degenerate']:
        cosmo = Cosmology(m_ncdm=0.1,neutrino_hierarchy=neutrino_hierarchy)
    cosmo = Cosmology(m_ncdm=[0.01,0.02,0.05])
    ba_class = Background(cosmo,engine='class')
    fo_class = Fourier(cosmo)


list_params = [{},{'sigma8':1.},{'A_s':2e-9},{'lensing':True},{'m_ncdm':0.1,'neutrino_hierarchy':'normal'},{'Omega_k':0.1}]


@pytest.mark.parametrize('params',list_params)
def test_background(params):
    cosmo = Cosmology(**params)
    ba_camb = Background(cosmo,engine='camb')
    #cosmo2 = Cosmology(engine='class')
    #Transfer(cosmo2)
    ba_class = Background(cosmo,engine='class')
    z = np.sort(np.random.uniform(0.,1.,10))
    for name in ['h','H0']:
        assert np.allclose(getattr(ba_class,name),getattr(ba_camb,name),atol=0,rtol=1e-6)
    for density in ['Omega','rho']:
        for name in ['k','cdm','b','g','ur','ncdm','de']:
            func = '{}_{}'.format(density,name)
            assert np.allclose(getattr(ba_class,func)(z),getattr(ba_camb,func)(z),atol=0,rtol=1e-4)
    for name in ['k','cdm','b','g','ur']:
        density = 'Omega0_{}'.format(name)
        assert np.allclose(getattr(ba_class,density),cosmo['Omega_{}'.format(name)],atol=0,rtol=1e-4)
        assert np.allclose(getattr(ba_class,density),getattr(ba_camb,density),atol=0,rtol=1e-4)
    for name in ['time','efunc','hubble_function']:
        assert np.allclose(getattr(ba_class,name)(z),getattr(ba_camb,name)(z),atol=0,rtol=1e-4)
    for distance in ['comoving_radial','angular_diameter','comoving_angular','luminosity']:
        func = '{}_distance'.format(distance)
        assert np.allclose(getattr(ba_class,func)(z),getattr(ba_camb,func)(z),atol=0,rtol=2e-4)

    if not cosmo['N_ncdm']:
        for engine in ['eisenstein_hu','eisenstein_hu_nowiggle']:
            ba_eh = Background(cosmo,engine=engine)
            for density in ['Omega']:
                for name in ['m','Lambda']:
                    func = '{}_{}'.format(density,name)
                    assert np.allclose(getattr(ba_class,func)(z),getattr(ba_eh,func)(z),atol=0,rtol=1e-3)
            for name in ['efunc','hubble_function','growth_factor','growth_rate']:
                assert np.allclose(getattr(ba_class,name)(z),getattr(ba_eh,name)(z),atol=0,rtol=2e-2)


@pytest.mark.parametrize('params',list_params)
def test_thermodynamics(params):
    cosmo = Cosmology(**params)
    th_class = Thermodynamics(cosmo,engine='class')
    th_camb = Thermodynamics(cosmo,engine='camb')
    # class and camb z_star do not match...
    for name in ['z_drag','rs_drag','z_star','rs_star'][:2]:
        assert np.allclose(getattr(th_class,name),getattr(th_camb,name),atol=0,rtol=1e-4)

    for engine in ['eisenstein_hu','eisenstein_hu_nowiggle']:
        th_eh = Thermodynamics(cosmo,engine=engine)
        for name in ['z_drag','rs_drag']:
            #print(name,getattr(th_class,name),getattr(th_eh,name))
            assert np.allclose(getattr(th_class,name),getattr(th_eh,name),atol=0,rtol=1e-2)


@pytest.mark.parametrize('params',list_params)
def test_primordial(params):
    cosmo = Cosmology(**params)
    pr_class = Primordial(cosmo,engine='class')
    pr_camb = Primordial(cosmo,engine='camb')
    for name in ['A_s','ln_1e10_A_s','n_s','k_pivot']:
        assert np.allclose(getattr(pr_class,name),getattr(pr_camb,name),atol=0,rtol=2e-3)
    k = np.random.uniform(1e-3,10.,100)
    for mode in ['scalar','tensor']:
        assert np.allclose(pr_class.pk_k(k,mode=mode),pr_camb.pk_k(k,mode=mode),atol=0,rtol=2e-3)
        assert np.allclose(pr_class.pk_interpolator(mode=mode)(k),pr_camb.pk_interpolator(mode=mode)(k),atol=0,rtol=2e-3)

    for engine in ['eisenstein_hu','eisenstein_hu_nowiggle']:
        pr_eh = Primordial(cosmo,engine=engine)
        for name in ['n_s']:
            assert np.allclose(getattr(pr_class,name),getattr(pr_eh,name),atol=0,rtol=1e-4)


@pytest.mark.parametrize('params',list_params)
def test_harmonic(params):
    cosmo = Cosmology(**params)
    hr_class = Harmonic(cosmo,engine='class')
    hr_camb = Harmonic(cosmo,engine='camb')

    for name in ['lensed_cl','lens_potential_cl']:
        for ellmax in [100,-1]:
            if not cosmo['lensing']:
                from pyclass import ClassBadValueError
                from cosmolib.camb import CAMBError
                with pytest.raises(ClassBadValueError):
                    tmp_class = getattr(hr_class,name)(ellmax=ellmax)
                with pytest.raises(CAMBError):
                    tmp_camb = getattr(hr_camb,name)(ellmax=ellmax)
            else:
                tmp_class = getattr(hr_class,name)(ellmax=ellmax)
                tmp_camb = getattr(hr_camb,name)(ellmax=ellmax)
                assert tmp_class.dtype == tmp_camb.dtype
                for field in tmp_class.dtype.names[1:]:
                    if name == 'lensed_cl':
                        atol = tmp_class[field].std()*1e-2 # to deal with 0 oscillating behavior
                        rtol = 1e-2
                        #print(name,field,tmp_class[field],tmp_camb[field])
                        #print(hr_class.unlensed_cl(ellmax=ellmax)[field],hr_camb.unlensed_cl(ellmax=ellmax)[field])
                    else:
                        atol = tmp_class[field].std()*1e-1 # to deal with 0 oscillating behavior
                        rtol = 1e-1
                    assert np.allclose(tmp_class[field],tmp_camb[field],atol=atol,rtol=rtol)

    for name in ['unlensed_cl']:
        for ellmax in [100,-1]:
            tmp_class = getattr(hr_class,name)(ellmax=ellmax)
            tmp_camb = getattr(hr_camb,name)(ellmax=ellmax)
            assert tmp_class.dtype == tmp_camb.dtype
            for field in tmp_class.dtype.names[1:]:
                atol = tmp_class[field].std()*1e-2
                assert np.allclose(tmp_class[field],tmp_camb[field],atol=atol,rtol=2e-2)


@pytest.mark.parametrize('params',list_params)
def test_fourier(params):
    cosmo = Cosmology(**params)
    fo_class = Fourier(cosmo,engine='class',gauge='newtonian')
    fo_camb = Fourier(cosmo,engine='camb')

    z = np.random.uniform(0.,10.,20)
    r = np.random.uniform(1.,10.,10)
    if 'sigma8' in cosmo.params:
        assert np.allclose(fo_camb.sigma8_z(0,of='delta_m'),cosmo['sigma8'],rtol=1e-3)
    for of in ['delta_m','delta_cb',('delta_cb','theta_cb'),'theta_cb']:
        assert np.allclose(fo_class.sigma_rz(r,z,of=of),fo_camb.sigma_rz(r,z,of=of),rtol=1e-3)
        assert np.allclose(fo_class.sigma8_z(z,of=of),fo_camb.sigma8_z(z,of=of),rtol=1e-3)

    z = np.random.uniform(0.,3.,1)
    k = np.random.uniform(1e-3,1.,20)

    for of in ['delta_m','delta_cb']:
        assert np.allclose(fo_class.pk_interpolator(nonlinear=False,of=of)(k,z=z),fo_camb.pk_interpolator(nonlinear=False,of=of)(k,z=z),rtol=2e-3)
        assert np.allclose(fo_class.sigma8_z(z,of=of),fo_class.pk_interpolator(nonlinear=False,of=of).sigma8_z(z=z),rtol=1e-4)
        assert np.allclose(fo_camb.sigma8_z(z,of=of),fo_camb.pk_interpolator(nonlinear=False,of=of).sigma8_z(z=z),rtol=1e-4)

    z = np.array([0.,1.,2.,3.,4.])
    for of in ['theta_cb',('delta_cb','theta_cb')]:
        assert np.allclose(fo_class.pk_interpolator(nonlinear=False,of=of)(k,z=z),fo_camb.pk_interpolator(nonlinear=False,of=of)(k,z=z),rtol=2e-3)

    ba_class = Background(cosmo,engine='class')
    # if not cosmo['N_ncdm']:
    z = np.random.uniform(0.,10.,20)
    r = np.random.uniform(1.,10.,10)
    pk_class = fo_class.pk_interpolator(of='delta_cb')
    pk_camb = fo_camb.pk_interpolator(of='delta_cb')

    for z in np.linspace(0.2,4.,5):
        for r in np.linspace(2.,20.,5):
            for dz in [1e-3,1e-2]:
                rtol = 1e-3
                #assert np.allclose(ba_class.growth_rate(z),pk_class.growth_rate_rz(r=r,z=z,dz=dz),atol=0,rtol=rtol)
                f = fo_class.sigma_rz(r,z,of='theta_cb')/fo_class.sigma_rz(r,z,of='delta_cb')
                assert np.allclose(f,pk_class.growth_rate_rz(r=r,z=z,dz=dz),atol=0,rtol=rtol)
                f = fo_camb.sigma_rz(r,z,of='theta_cb')/fo_camb.sigma_rz(r,z,of='delta_cb')
                assert np.allclose(f,pk_camb.growth_rate_rz(r=r,z=z,dz=dz),atol=0,rtol=rtol)

    for engine in ['eisenstein_hu','eisenstein_hu_nowiggle']:
        fo_eh = Fourier(cosmo,engine=engine)
        if 'sigma8' not in cosmo.params:
            with pytest.raises(CosmologyError):
                fo_eh.pk_interpolator()
        else:
            rtol = {'eisenstein_hu':6e-2,'eisenstein_hu_nowiggle':8e-2}[engine]
            pk_class = fo_class.pk_interpolator(nonlinear=False,of='delta_m')
            pk_eh = fo_eh.pk_interpolator()
            assert np.allclose(pk_class(k,z=z),pk_eh(k,z=z),rtol=rtol)
            r = np.random.uniform(1.,10.,10)
            assert np.allclose(pk_class.growth_rate_rz(r=r,z=z),pk_eh.growth_rate_rz(r=r,z=z),rtol=rtol)


def test_registered():
    from cosmoprimo import Planck2018FullFlatLCDM
    cosmo = Planck2018FullFlatLCDM()
    assert cosmo['h'] == 0.6766


def plot_primordial_power_spectrum():
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    pr_class = Primordial(cosmo,engine='class')
    pr_camb = Primordial(cosmo,engine='camb')
    pk = pr_class.pk_interpolator()
    k = np.logspace(-6,2,500)
    plt.loglog(k,pk(k),label='class')
    pk = pr_camb.pk_interpolator()
    k = np.logspace(-6,2,500)
    plt.loglog(k,pk(k),label='camb')
    plt.legend()
    plt.show()


def plot_harmonic():
    from matplotlib import pyplot as plt
    cosmo = Cosmology(lensing=True)
    hr_class = Harmonic(cosmo,engine='class')
    cls = hr_class.lensed_cl()
    ells_factor = (cls['ell'] + 1) * cls['ell'] / (2 * np.pi)
    plt.plot(cls['ell'],ells_factor*cls['tt'],label='class')
    hr_camb = Harmonic(cosmo,engine='camb')
    cls = hr_camb.lensed_cl()
    plt.plot(cls['ell'],ells_factor*cls['tt'],label='camb')
    cosmo = Cosmology(lensing=True,m_ncdm=0.1)
    hr_class = Harmonic(cosmo,engine='class')
    cls = hr_class.lensed_cl()
    plt.plot(cls['ell'],ells_factor*cls['tt'],label='class + neutrinos')
    hr_camb = Harmonic(cosmo,engine='camb')
    cls = hr_camb.lensed_cl()
    plt.plot(cls['ell'],ells_factor*cls['tt'],label='camb + neutrinos')
    plt.legend()
    plt.show()


def plot_matter_power_spectrum():
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    fo_class = Fourier(cosmo,engine='class')
    #k,z,pk = fo_class.table()
    #plt.loglog(k,pk)
    z = 1.
    k = np.logspace(-6,2,500)
    pk = fo_class.pk_interpolator(nonlinear=False,of='m',extrap_kmin=1e-7)
    #pk = fo_class.pk_kz
    plt.loglog(k,pk(k,z=1),label='class')
    fo_camb = Fourier(cosmo,engine='camb')
    pk = fo_camb.pk_interpolator(nonlinear=False,of='m',extrap_kmin=1e-7)
    plt.loglog(k,pk(k,z=1),label='camb')
    fo_eh = Fourier(cosmo,engine='eisenstein_hu')
    pk = fo_eh.pk_interpolator()
    plt.loglog(k,pk(k,z=1),label='eisenstein_hu')
    fo_eh = Fourier(cosmo,engine='eisenstein_hu_nowiggle')
    pk = fo_eh.pk_interpolator()
    plt.loglog(k,pk(k,z=1),label='eisenstein_hu_nowiggle')
    plt.legend()
    plt.show()


def external_test_camb():
    import camb
    from camb import CAMBdata

    As = params.InitPower.As
    ns = params.InitPower.ns
    print(params.InitPower.As)
    params.DoLensing = False
    params.Want_CMB_lensing = False
    tr = camb.get_transfer_functions(params)
    tr.calc_power_spectra()
    print(tr.get_lens_potential_cls(lmax=100,CMB_unit=None,raw_cl=True))

    params = camb.CAMBparams(H0=70,omch2=0.15,ombh2=0.02)
    params.DoLensing = True
    params.Want_CMB_lensing = True
    tr = CAMBdata()
    tr.calc_power_spectra(params)
    print(tr.get_lens_potential_cls(lmax=100,CMB_unit=None,raw_cl=True))

    params = camb.CAMBparams(H0=70,omch2=0.15,ombh2=0.02)
    As = params.InitPower.As
    ns = params.InitPower.ns
    params.DoLensing = False
    #params.Want_CMB_lensing = True
    #params.Want_CMB_lensing = True
    tr = camb.get_transfer_functions(params)
    tr.Params.InitPower.set_params(As=As,ns=ns)
    tr.calc_power_spectra()
    tr.Params.DoLensing = True
    tr.Params.Want_CMB_lensing = True
    print(tr.get_lens_potential_cls(lmax=100,CMB_unit=None,raw_cl=True))

    params = camb.CAMBparams(H0=70,omch2=0.15,ombh2=0.02)
    #params.WantCls = False
    params.Want_CMB = False
    #params.WantTransfer = True
    tr = camb.get_transfer_functions(params)
    params.Want_CMB = True
    tr.calc_power_spectra(params)
    print(tr.get_unlensed_scalar_cls(lmax=100,CMB_unit=None,raw_cl=True))
    #print(tr.get_total_cls(lmax=100,CMB_unit=None,raw_cl=True))


def external_test_pyccl():
    import pyccl
    params = {'sigma8':0.8,'Omega_c':0.28,'Omega_b':0.02,'h':0.8,'n_s':0.96,'m_nu':0.1,'m_nu_type':'normal'}
    cosmo = pyccl.Cosmology(**params)
    print(pyccl.background.growth_rate(cosmo,1))
    params = {'sigma8':0.8,'Omega_cdm':0.28,'Omega_b':0.02,'h':0.8,'n_s':0.96,'m_ncdm':0.1,'neutrino_hierarchy':'normal'}
    cosmo = Cosmology(**params)
    print(Background(cosmo,engine='class').growth_rate(0))

    params = {'sigma8':0.8,'Omega_c':0.28,'Omega_b':0.02,'h':0.8,'n_s':0.96}
    cosmo = pyccl.Cosmology(**params)
    print(pyccl.background.growth_rate(cosmo,1))
    params = {'sigma8':0.8,'Omega_cdm':0.28,'Omega_b':0.02,'h':0.8,'n_s':0.96}
    cosmo = Cosmology(**params)
    print(Background(cosmo,engine='class').growth_rate(0))


def benchmark():
    import timeit
    import pyccl
    params = {'sigma8':0.8,'Omega_cdm':0.28,'Omega_b':0.02,'h':0.8,'n_s':0.96,'m_ncdm':0.1,'neutrino_hierarchy':'normal'}
    pyccl_params = {'sigma8':0.8,'Omega_c':0.28,'Omega_b':0.02,'h':0.8,'n_s':0.96,'m_nu':0.1,'m_nu_type':'normal','transfer_function':'boltzmann_class'}
    z = np.linspace(0.,10.,10000)
    z_pk = 1. # ccl does not support vectorization over scale factor
    k = np.logspace(-4,2,500)
    a = 1./(1+z)
    a_pk = 1./(1+z_pk)
    d = {}
    d['cosmoprimo initialisation'] = {'stmt':"Cosmology(**params)",'number':1000}
    d['pyccl initialisation'] = {'stmt':"pyccl.Cosmology(**pyccl_params)",'number':1000}

    d['cosmoprimo initialisation + background'] = {'stmt':"cosmo = Cosmology(**params); cosmo.get_background('class').comoving_radial_distance(z)",
                                                'number':10}
    d['pyccl initialisation + background'] = {'stmt':"cosmo = pyccl.Cosmology(**pyccl_params); pyccl.background.comoving_radial_distance(cosmo,a)",
                                                'number':10}

    d['cosmoprimo initialisation + background single z'] = {'stmt':"cosmo = Cosmology(**params); cosmo.get_background('class').comoving_radial_distance(z_pk)",
                                                'number':10}
    d['pyccl initialisation + background single z'] = {'stmt':"cosmo = pyccl.Cosmology(**pyccl_params); pyccl.background.comoving_radial_distance(cosmo,a_pk)",
                                                'number':10}

    d['cosmoprimo initialisation + pk'] = {'stmt':"cosmo = Cosmology(**params); cosmo.get_fourier('class').pk_interpolator()(k,z_pk)",
                                                'number':2}
    d['pyccl initialisation + pk'] = {'stmt':"cosmo = pyccl.Cosmology(**pyccl_params); pyccl.linear_matter_power(cosmo,k*cosmo['h'],a_pk)",
                                                'number':2}

    cosmo = Cosmology(**params)
    pyccl_cosmo = pyccl.Cosmology(**pyccl_params)
    ba_class = cosmo.get_background('class')
    fo_class = cosmo.get_fourier('class')
    d['cosmoprimo background'] = {'stmt':"ba_class.comoving_radial_distance(z)",'number':100}
    d['pyccl background'] = {'stmt':"pyccl.background.comoving_radial_distance(pyccl_cosmo,a)",'number':100}
    d['cosmoprimo pk'] = {'stmt':"fo_class.pk_interpolator()(k,z_pk)",'number':2}
    d['pyccl pk'] = {'stmt':"pyccl.linear_matter_power(pyccl_cosmo,k*pyccl_cosmo['h'],a_pk)",'number':2}

    for key,value in d.items():
        dt = timeit.timeit(**value,globals={**globals(),**locals()})/value['number']*1e3
        print('{} takes {:.3f} milliseconds'.format(key,dt))


if __name__ == '__main__':

    benchmark()
    exit()

    test_params()
    test_registered()
    for params in list_params:
        test_background(params)
        test_thermodynamics(params)
        test_primordial(params)
        test_harmonic(params)
        test_fourier(params)
    #plot_primordial_power_spectrum()
    #plot_harmonic()
    #plot_matter_power_spectrum()
    #external_test_camb()
    #external_test_pyccl()
