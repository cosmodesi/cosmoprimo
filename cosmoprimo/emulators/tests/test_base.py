import os
import numpy as np

from cosmoprimo.fiducial import DESI
from cosmoprimo.emulators import Emulator, EmulatedEngine, setup_logging


def test_base():
    cosmo = DESI()
    emulator_dir = '_tests'
    fn = os.path.join(emulator_dir, 'emu.npy')
    params = {'Omega_cdm': (0.25, 0.26), 'h': (0.6, 0.8)}
    emulator = Emulator(cosmo, params, engine='point')
    emulator.set_samples()
    emulator.fit()
    emulator.save(fn)

    cosmo = DESI(engine=EmulatedEngine.load(fn))
    z = np.linspace(0., 3., 100)
    d1 = cosmo.comoving_radial_distance(z)
    d2 = cosmo.clone(Omega_m=0.3).comoving_radial_distance(z)
    assert np.allclose(d2, d1)

    cosmo.rs_drag
    cosmo.get_harmonic().unlensed_cl()
    cosmo.get_fourier().pk_interpolator(of='delta_cb')


if __name__ == '__main__':

    setup_logging()
    test_base()