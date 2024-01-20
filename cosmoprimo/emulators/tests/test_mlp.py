import os

import numpy as np

from cosmoprimo.fiducial import DESI
from cosmoprimo.emulators import Emulator, EmulatedEngine, MLPEmulatorEngine, QMCSampler, Samples, FourierNormOperation, Log10Operation, PCAOperation, ChebyshevOperation, plot_residual_background, plot_residual_thermodynamics, plot_residual_harmonic, plot_residual_fourier, setup_logging


samples_fn = '_tests/samples.npz'


def test_samples():
    cosmo = DESI(lensing=True, non_linear='halofit')
    params = {'Omega_cdm': (0.25, 0.26), 'h': (0.6, 0.8), 'logA': (2.5, 3.5)}

    sampler = QMCSampler(cosmo, params, save_fn=samples_fn)
    sampler.run(save_every=5, niterations=100)


def test_mlp():
    fn = '_tests/emu.npy'

    ref_samples = Samples.load(samples_fn)
    k1d, z = ref_samples['Y.fourier.k'][0], ref_samples['Y.fourier.z'][0]
    k2d = np.meshgrid(k1d, z, indexing='ij')[0].ravel()
    k2d_non_linear = np.meshgrid(k1d, ref_samples['Y.fourier.z_non_linear'][0], indexing='ij')[0].ravel()

    def loss_pk_1d(y_true, y_pred):
        import tensorflow as tf
        return tf.reduce_sum(tf.multiply(tf.square(y_true - y_pred), tf.convert_to_tensor(
k1d**2, dtype=y_true.dtype)))

    def loss_pk_2d(y_true, y_pred):
        import tensorflow as tf
        return tf.reduce_sum(tf.multiply(tf.square(y_true - y_pred), tf.convert_to_tensor(
k2d**2, dtype=y_true.dtype)))

    def loss_pk_2d_non_linear(y_true, y_pred):
        import tensorflow as tf
        return tf.reduce_sum(tf.multiply(tf.square(y_true - y_pred), tf.convert_to_tensor(
k2d_non_linear**2, dtype=y_true.dtype)))

    #tofit = ['fourier']
    tofit = ['background']
    operations = []
    if 'fourier' in tofit:
        operations.append(FourierNormOperation(ref_pk_name='fourier.pk.delta_cb.delta_cb'))
    engine = {}
    #engine['background.*'] = MLPEmulatorEngine(nhidden=(10,) * 2, yoperation=PCAOperation(npcs=4))
    engine['background.*'] = MLPEmulatorEngine(nhidden=(10,) * 2, yoperation=ChebyshevOperation(order=50))
    engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(10,) * 2)
    engine['primordial.*'] = MLPEmulatorEngine(nhidden=(5,) * 2)
    #engine['fourier.*'] = MLPEmulatorEngine(nhidden=(10,) * 3, yoperation=[ChebyshevOperation(axis=0, order=100), ChebyshevOperation(axis=1, order=10)])
    #engine['fourier.pk.delta_cb.delta_cb'] = MLPEmulatorEngine(nhidden=(10,) * 3, yoperation=[ChebyshevOperation(axis=0, order=100)])
    #engine['fourier.pk.*'] = MLPEmulatorEngine(nhidden=(10,) * 3) #, ChebyshevOperation(axis=0, order=100), ChebyshevOperation(axis=1, order=10)])
    engine['fourier.pk*'] = MLPEmulatorEngine(nhidden=(10,) * 3, loss=loss_pk_2d)
    engine['fourier.pk_non_linear*'] = MLPEmulatorEngine(nhidden=(10,) * 3, loss=loss_pk_2d_non_linear)
    engine['fourier.pk.delta_cb.delta_cb'] = MLPEmulatorEngine(nhidden=(10,) * 3, loss=loss_pk_1d)
    engine['harmonic.*'] = MLPEmulatorEngine(nhidden=(10,) * 3, yoperation=PCAOperation(npcs=10))

    emulator = Emulator(engine=engine, yoperation=operations)
    if 'background' in tofit:
        emulator.set_samples(samples=ref_samples.select(['X.*', 'Y.background.*'], exclude='X.logA'))
        emulator.fit(name='background.*', batch_frac=(0.2, 1.), learning_rate=(1e-2, 1e-3), epochs=40, verbose=0)
        emulator.save(fn)
    if 'fourier' in tofit:
        emulator.set_samples(samples=ref_samples.select(['X.*', 'Y.fourier.*']))
        emulator.fit(name='fourier.pk*', batch_frac=(0.2, 1.), learning_rate=(1e-2, 1e-4), epochs=1000, verbose=1)
        emulator.save(fn)

    emulator = Emulator.load(fn)

    cosmo = DESI(engine=EmulatedEngine.load(fn))
    z = np.linspace(0., 3., 100)

    if 'background' in tofit:
        plot_residual_background(ref_samples, emulated_samples=cosmo, subsample=10, fn='_tests/background.png')
    if 'fourier' in tofit:
        cosmo = cosmo.clone(sigma8=0.8)
        print(cosmo.get_fourier().sigma8_m, cosmo.engine._rsigma8)
        plot_residual_fourier(ref_samples, emulated_samples=cosmo, subsample=2, fn='_tests/fourier.png')

    #plot_residual_thermodynamics(ref_samples, emulated_samples=cosmo, fn='_tests/thermodynamics.png')
    #plot_residual_harmonic(ref_samples, emulated_samples=cosmo, fn='_tests/harmonic.png')


def test_fourier_z():

    fn = '_tests/emu_z.npy'

    tofit = False
    ref_samples = Samples.load(samples_fn)

    emulator = Emulator()

    if tofit:
        # Adding z as a parameter
        samples_0 = ref_samples.select(['X.*', 'Y.fourier.k'])
        z = ref_samples['Y.fourier.z'][0]
        list_samples = []
        for iz, zz in enumerate(z[1:]):
            samples_z = samples_0.deepcopy()
            samples_z['X.z'] = np.full(samples_0.shape, zz)
            for name in ref_samples.columns('Y.fourier.pk.*'):
                samples_z[name] = ref_samples[name][..., iz]
            list_samples.append(samples_z)
        samples = samples_0.concatenate(list_samples)
        emulator.set_samples(samples=samples, engine=MLPEmulatorEngine(nhidden=(10,) * 3))
        emulator.fit(name='fourier.pk.*', batch_frac=(0.2, 1.), learning_rate=(1e-2, 1e-4), epochs=1000, verbose=1)
        emulator.save(fn)

    emulator = Emulator.load(fn)

    cosmo = DESI(engine=EmulatedEngine.load(fn))
    z = np.linspace(0., 3., 100)

    cosmo = cosmo.clone(sigma8=0.8)
    plot_residual_fourier(ref_samples, emulated_samples=cosmo, subsample=2, fn='_tests/fourier.png')


if __name__ == '__main__':

    setup_logging()
    #test_samples()
    test_mlp()
    #test_fourier_z()