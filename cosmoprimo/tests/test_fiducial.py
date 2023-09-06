import os
import tempfile

import pytest
from matplotlib import pyplot as plt
import numpy as np

from cosmoprimo import fiducial, Cosmology


def test_planck():
    cosmo = fiducial.Planck2018FullFlatLCDM()
    assert cosmo['h'] == 0.6766


def test_boss():
    cosmo = fiducial.BOSS()
    assert cosmo['h'] == 0.676


def test_abacus():
    from cosmoprimo.fiducial import AbacusSummit_params, AbacusSummit
    dcosmos = AbacusSummit_params(params=['root', 'omega_b', 'omega_cdm', 'h', 'A_s', 'n_s', 'alpha_s', 'N_ur', 'omega_ncdm', 'w0_fld', 'wa_fld'])
    ncosmos = len(dcosmos)
    assert AbacusSummit_params(19)['omega_ncdm'] == (0.0006442, 0.0006442)
    assert list(AbacusSummit_params(19, params=['h']).keys()) == ['h']
    assert list(AbacusSummit_params(19, params=['omega_k', 'h']).keys()) == ['omega_k', 'h']
    for dcosmo in dcosmos:
        cosmo = AbacusSummit(dcosmo['root'])
        dcosmo.pop('root')
        assert cosmo == AbacusSummit().clone(T_ncdm_over_cmb=None, **dcosmo)
    with pytest.raises(ValueError):
        cosmo = AbacusSummit('0')

    try: from abacusnbody import metadata
    except ImportError: metadata = None

    if metadata is not None:
        print('With abacusnbody')
        plot = False
        for root in AbacusSummit_params(params=['root']):
            root = root['root'][-3:]
            #if root not in ['000', '009']: continue
            simname = 'AbacusSummit_base_c{}_ph000'.format(root)
            try:
                meta = metadata.get_meta(simname)
            except ValueError:
                continue
            cosmo = AbacusSummit(root)
            z = np.array(list(meta['GrowthTable'].keys()))
            dz_ref = np.array(list(meta['GrowthTable'].values()))
            if plot and root == '000':
                from matplotlib import pyplot as plt
                plt.plot(z, dz_ref * (1 + z), label='abacusnbody')
                dz_test = cosmo.growth_factor(z)
                dz_test *= dz_ref[0] / dz_test[0]
                plt.plot(z, dz_test * (1 + z), label='class')
                plt.legend()
                plt.show()
            mask = z <= 5.
            pivot = np.flatnonzero(z == 1.)
            assert pivot.size
            z, dz_ref = z[mask], dz_ref[mask]
            dz_test = cosmo.growth_factor(z)
            dz_test *= dz_ref[pivot] / dz_test[pivot]
            #print(dz_test / dz_ref - 1., cosmo.Omega0_r, cosmo.Omega0_ncdm, meta['Omega_Smooth'])
            sig_test = cosmo.get_fourier().sigma8_z(z)
            sig_test *= dz_ref[pivot] / sig_test[pivot]
            pk_test = cosmo.get_fourier().pk_interpolator()(0.2, z)**0.5
            pk_test *= dz_ref[pivot] / pk_test[pivot]
            assert np.allclose(dz_test, dz_ref, rtol=1e-3)
            assert np.allclose(sig_test, dz_ref, rtol=1e-3)

            try:
                z = np.sort(np.atleast_1d(meta['TimeSliceRedshifts']))
            except KeyError:
                continue
            vel_ref = np.array([metadata.get_meta(simname, redshift=zz)['VelZSpace_to_kms'] for zz in z]) / meta['BoxSize']
            vel_test =  1. / (1 + z) * 100 * cosmo.efunc(z)  #(cosmo.efunc(z)**2 - cosmo.Omega0_r * (1 + z)**4)**(0.5)
            #print(vel_test / vel_ref - 1.)
            assert np.allclose(vel_test, vel_ref, rtol=1e-3)


def test_desi(plot=False):

    cosmo = fiducial.DESI(precision='base', z_pk=np.linspace(0., 2., 11))
    assert cosmo.engine._extra_params

    with tempfile.TemporaryDirectory() as tmp_dir:
        fn = os.path.join(tmp_dir, 'cosmo.npy')
        cosmo.save(fn)
        cosmo = Cosmology.load(fn)

    assert np.allclose(cosmo['omega_ncdm'], 0.0006442)
    assert cosmo['N_ncdm'] == 1
    assert np.allclose(cosmo.get_primordial().A_s, 2.0830e-9, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_primordial().n_s, 0.9649, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_background().N_ur, 2.0328, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_background().N_ncdm, 1, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_background().Omega0_ncdm * cosmo.h**2, 0.0006442, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_background().w0_fld, -1, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_background().wa_fld, 0, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_fourier().sigma8_m, 0.807952, rtol=1e-4, atol=1e-9)
    assert np.allclose(cosmo.get_fourier().sigma8_cb, 0.811355, rtol=1e-4, atol=1e-9)
    assert np.allclose(cosmo.get_background().h, 0.6736, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_background().Omega0_cdm * cosmo.h**2, 0.12, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_background().Omega0_b * cosmo.h**2, 0.02237, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo['k_pivot'], 0.05, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_primordial().k_pivot * cosmo.h, 0.05, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_thermodynamics().tau_reio, 0.0544, rtol=1e-9, atol=1e-9)

    fo = cosmo.get_fourier()
    for of, fn in zip(['cb', 'cb', 'm'], ['AbacusSummitBase_CLASS_pk_cb.txt', 'abacus_cosm000_CLASSv3.1.1.00_z2_pk_cb.dat', 'abacus_cosm000_CLASSv3.1.1.00_z2_pk.dat']):
        pk = fo.pk_interpolator(of='delta_{}'.format(of)).to_1d(z=1.)
        # pk(np.logspace(-5.99, 1.99, 1000))
        fn = os.path.join('fiducial', fn)
        kref, pkref = np.loadtxt(fn, unpack=True)
        mask = (kref >= pk.k[0]) & (kref <= pk.k[-1])
        kref, pkref = kref[mask], pkref[mask]
        if plot:
            plt.plot(kref, pk(kref) / pkref)
            plt.xscale('log')
            plt.show()
        assert np.allclose(pk(kref), pkref, rtol=5e-4)

    cosmo = fiducial.DESI(sigma8=1.)
    assert np.allclose(cosmo.get_fourier().sigma8_m, 1., rtol=1e-4, atol=1e-9)


if __name__ == '__main__':

    test_planck()
    test_boss()
    test_abacus()
    test_desi(plot=False)
