import os

import numpy as np
import jax
from jax import numpy as jnp

from cosmoprimo.fiducial import DESI
from cosmoprimo.emulators import Emulator, EmulatedEngine, setup_logging


emulator_dir = '_tests'
emulator_fn = os.path.join(emulator_dir, 'emu.npy')


def test_base():
    cosmo = DESI()
    params = {'Omega_cdm': (0.25, 0.26), 'h': (0.6, 0.8)}
    emulator = Emulator(cosmo, params, engine='point')
    emulator.set_samples()
    emulator.fit()
    emulator.save(emulator_fn)

    cosmo = DESI(engine=EmulatedEngine.load(emulator_fn))
    z = np.linspace(0., 3., 100)
    d1 = cosmo.comoving_radial_distance(z)
    d2 = cosmo.clone(Omega_m=0.3).comoving_radial_distance(z)
    assert np.allclose(d2, d1)

    cosmo.rs_drag
    cosmo.get_harmonic().unlensed_cl()
    cosmo.get_fourier().pk_interpolator(of='delta_cb')


def test_jax():

    engine = EmulatedEngine.load(emulator_fn)
    cosmo = DESI(Omega_m=jnp.array(0.2), engine=engine)

    def test(Omega_m=0.3):
        cosmo = DESI(Omega_m=Omega_m, engine=engine)
        return cosmo.Omega0_cdm

    test = jax.jit(test)
    print(test(0.2))


if __name__ == '__main__':

    setup_logging()

    #test_base()
    test_jax()