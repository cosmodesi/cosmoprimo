import os
import tempfile

import pytest
from matplotlib import pyplot as plt
import numpy as np

from cosmoprimo import fiducial, Cosmology


def test_planck():

    cosmo = fiducial.Planck2018FullFlatLCDM()
    assert cosmo['h'] == 0.6766


def test_desi(plot=False):

    cosmo = fiducial.DESI(precision='base', z_pk=np.linspace(0., 2., 11))

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
    assert np.allclose(cosmo.get_background().Omega0_ncdm*cosmo.h**2, 0.0006442, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_background().w0_fld, -1, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_background().wa_fld, 0, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_fourier().sigma8_m, 0.807952, rtol=1e-4, atol=1e-9)
    assert np.allclose(cosmo.get_fourier().sigma8_cb, 0.811355, rtol=1e-4, atol=1e-9)
    assert np.allclose(cosmo.get_background().h, 0.6736, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_background().Omega0_cdm*cosmo.h**2, 0.12, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_background().Omega0_b*cosmo.h**2, 0.02237, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_primordial().k_pivot*cosmo.h, 0.05, rtol=1e-9, atol=1e-9)
    assert np.allclose(cosmo.get_thermodynamics().tau_reio, 0.0544, rtol=1e-9, atol=1e-9)

    fo = cosmo.get_fourier()
    for of, fn in zip(['cb', 'cb', 'm'], ['AbacusSummitBase_CLASS_pk_cb.txt', 'abacus_cosm000_CLASSv3.1.1.00_z2_pk_cb.dat', 'abacus_cosm000_CLASSv3.1.1.00_z2_pk.dat']):
        pk = fo.pk_interpolator(of='delta_{}'.format(of)).to_1d(z=1.)
        #pk(np.logspace(-5.99, 1.99, 1000))
        fn = os.path.join('fiducial', fn)
        kref, pkref = np.loadtxt(fn, unpack=True)
        mask = (kref >= pk.k[0]) & (kref <= pk.k[-1])
        kref, pkref = kref[mask], pkref[mask]
        if plot:
            plt.plot(kref, pk(kref)/pkref)
            plt.xscale('log')
            plt.show()
        assert np.allclose(pk(kref), pkref, rtol=5e-4)


if __name__ == '__main__':

    test_planck()
    test_desi(plot=False)
