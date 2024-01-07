import os

import numpy as np

from cosmoprimo.fiducial import DESI
from cosmoprimo.emulators import Emulator, EmulatedEngine, TaylorEmulatorEngine, DiffSampler, Samples, plot_residual_background, plot_residual_thermodynamics, plot_residual_harmonic, plot_residual_fourier, setup_logging


samples_fn = '_tests/diff_samples.npy'


def test_samples():
    cosmo = DESI()
    params = {'Omega_cdm': (0.25, 0.26), 'h': (0.6, 0.8)}
    sampler = DiffSampler(cosmo, params, order={'*': 1}, save_fn=samples_fn)
    sampler.run()


def test_taylor():
    cosmo = DESI()
    emulator_dir = '_tests'
    fn = os.path.join(emulator_dir, 'emu.npy')
    emulator = Emulator(samples=samples_fn, engine=TaylorEmulatorEngine())
    #ref_samples = emulator.samples['fourier.pk.delta_cb.delta_cb']
    emulator.fit()
    emulator.save(fn)

    ref_samples = Samples.load(samples_fn)

    cosmo = DESI(engine=EmulatedEngine.load(fn))
    z = np.linspace(0., 3., 100)
    d1 = cosmo.comoving_radial_distance(z)
    d2 = cosmo.clone(Omega_cdm=0.25).comoving_radial_distance(z)
    #assert not np.allclose(d2, d1)

    plot_residual_background(ref_samples, emulated_samples=cosmo, fn='_tests/background.png')
    plot_residual_thermodynamics(ref_samples, emulated_samples=cosmo, fn='_tests/thermodynamics.png')
    plot_residual_harmonic(ref_samples, emulated_samples=cosmo, fn='_tests/harmonic.png')
    plot_residual_fourier(ref_samples, emulated_samples=cosmo, fn='_tests/fourier.png')


if __name__ == '__main__':

    setup_logging()
    test_samples()
    test_taylor()