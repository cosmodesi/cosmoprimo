import os
import tempfile

import pytest
import numpy as np

from cosmoprimo import fiducial, constants, Cosmology, CosmologyError


def test_desi():
    cosmo = fiducial.DESI()

    with tempfile.TemporaryDirectory() as tmp_dir:
        fn = os.path.join(tmp_dir, 'cosmo.npy')
        cosmo.save(fn)
        cosmo = Cosmology.load(fn)

    assert np.allclose(cosmo['omega_ncdm'], 0.0006442)
    assert cosmo['N_ncdm'] == 1

    cosmo_tabulated = fiducial.TabulatedDESI()

    with pytest.raises(CosmologyError):
        cosmo_tabulated.comoving_radial_distance(-1)

    z = np.linspace(0, 10, 100)
    assert np.allclose(cosmo_tabulated.comoving_radial_distance(z), cosmo.comoving_radial_distance(z), rtol=1e-7, atol=1e-10)
    # plt.plot(z, cosmo_tabulated.comoving_radial_distance(z) / cosmo.comoving_radial_distance(z))
    # plt.show()

    z0 = np.linspace(0, 9.9, 100)
    dz = 1e-6
    dcdz = (cosmo_tabulated.comoving_radial_distance(z0 + dz) - cosmo_tabulated.comoving_radial_distance(z0)) / dz
    dcdz_efunc = constants.c / (100. * 1e3 * cosmo_tabulated.efunc(z0))
    assert np.allclose(dcdz, dcdz_efunc, rtol=1e-2, atol=1e-10)

    cosmo_camb = fiducial.DESI(engine='camb')
    assert np.allclose(cosmo_camb.comoving_radial_distance(z), cosmo.comoving_radial_distance(z), rtol=2e-7, atol=1e-10)
    # plt.plot(z, cosmo_camb.comoving_radial_distance(z) / cosmo.comoving_radial_distance(z))
    # plt.show()

    cosmo_astropy = fiducial.DESI(engine='astropy')
    assert np.allclose(cosmo_astropy.comoving_radial_distance(z), cosmo.comoving_radial_distance(z), rtol=2e-5, atol=1e-10)
    # plt.plot(z, cosmo_astropy.comoving_radial_distance(z) / cosmo.comoving_radial_distance(z))
    # plt.show()

    try: import pyccl
    except ImportError: pyccl = None

    z = np.linspace(0, 10, 100)
    if pyccl is not None:
        print('With pyccl')
        params = {'sigma8': cosmo.sigma8_m, 'Omega_c': cosmo.Omega0_cdm, 'Omega_b': cosmo.Omega0_b, 'h': cosmo.h, 'n_s': cosmo.n_s, 'm_nu': cosmo['m_ncdm'][0], 'm_nu_type': 'single'}
        cosmo_pyccl = pyccl.Cosmology(**params)
        distance_pyccl = cosmo_pyccl.comoving_radial_distance(1. / (1. + z)) * cosmo.h
        assert np.allclose(distance_pyccl, cosmo.comoving_radial_distance(z), rtol=2e-6, atol=1e-10)
        # plt.plot(z, distance_pyccl / cosmo.comoving_radial_distance(z) - 1, label='pyccl')
        # plt.plot(z, cosmo_astropy.comoving_radial_distance(z) / cosmo.comoving_radial_distance(z) - 1, label='astropy')
        # plt.legend()
        # plt.show()


if __name__ == '__main__':

    # fiducial.save_TabulatedDESI()
    test_desi()
