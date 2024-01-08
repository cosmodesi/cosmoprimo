from pathlib import Path

import numpy as np

from cosmoprimo.fiducial import DESI
from cosmoprimo.emulators import Emulator, EmulatedEngine, MLPEmulatorEngine, QMCSampler, Samples, FourierNormOperation, PCAOperation, ChebyshevOperation, plot_residual_background, plot_residual_thermodynamics, plot_residual_primordial, plot_residual_fourier, plot_residual_harmonic, setup_logging


this_dir = Path(__file__).parent
train_dir = this_dir / '_train'
samples_fn = train_dir / 'classy/samples'
emulator_fn = this_dir / 'classy/emulator.npy'


def sample():
    cosmo = DESI(lensing=True, non_linear='mead', engine='class')
    params = {'logA': (2.5, 3.5), 'n_s': (0.88, 1.06), 'h': (0.4, 1.), 'omega_b': (0.019, 0.026), 'omega_cdm': (0.08, 0.2), 'm_ncdm': (0., 0.8),
              'Omega_k': (-0.3, 0.3), 'w0_fld': (-1.5, -0.5), 'wa_fld': (-0.7, 0.7), 'tau_reio': (0.02, 0.12)}
    sampler = QMCSampler(cosmo, params, save_fn=samples_fn)
    sampler.run(save_every=2, niterations=100)


def fit(tofit=('background', 'thermodynamics', 'primordial', 'fourier', 'harmonic')):
    samples = Samples.load(samples_fn)
    operations = []
    operations.append(FourierNormOperation(ref_pk_name='fourier.pk.delta_cb.delta_cb'))
    engine = {}
    engine['background.*'] = MLPEmulatorEngine(nhidden=(100,) * 2, yoperation=ChebyshevOperation(axis=0, order=100))
    engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(20,) * 2)
    engine['primordial.*'] = MLPEmulatorEngine(nhidden=(20,) * 2)
    engine['fourier.*'] = MLPEmulatorEngine(nhidden=(512,) * 3, yoperation=[ChebyshevOperation(axis=0, order=100), ChebyshevOperation(axis=1, order=10)])
    engine['fourier.pk.delta_cb.delta_cb'] = MLPEmulatorEngine(nhidden=(512,) * 3, yoperation=[ChebyshevOperation(axis=0, order=100)])
    engine['harmonic.*'] = MLPEmulatorEngine(nhidden=(512,) * 3, yoperation=[ChebyshevOperation(axis=0, order=100)])

    if emulator_fn.exists():
        emulator = Emulator.load(emulator_fn)
    else:
        emulator = Emulator()
    emulator.set_engine(engine)
    emulator.yoperations = operations
    if 'background' in tofit:
        emulator.set_samples(samples=samples.select(['X.*', 'Y.background.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio']))
        emulator.fit(name='background.*', batch_frac=(0.2, 0.4, 1.), learning_rate=(1e-2, 1e-4, 1e-6), epochs=1000)
        emulator.save(emulator_fn)
    if 'thermodynamics' in tofit:
        emulator.set_samples(samples=samples.select(['X.*', 'Y.thermodynamics.*'], exclude=['X.logA', 'X.n_s']))
        emulator.fit(name='thermodynamics.*', batch_frac=(0.2, 0.4, 1.), learning_rate=(1e-2, 1e-4, 1e-6), epochs=1000)
        emulator.save(emulator_fn)
    if 'primordial' in tofit:
        emulator.set_samples(samples=samples.select(['X.logA', 'X.n_s', 'Y.primordial.*']))
        emulator.fit(name='primordial.*', batch_frac=(0.2, 0.4, 1.), learning_rate=(1e-2, 1e-4, 1e-6), epochs=1000)
        emulator.save(emulator_fn)
    if 'fourier' in tofit:
        emulator.set_samples(samples=samples.select(['X.*', 'Y.fourier.*'], exclude=['X.tau_reio']))
        emulator.fit(name='fourier.*', batch_frac=(0.2, 0.4, 1.), learning_rate=(1e-2, 1e-4, 1e-6), epochs=1000)
        emulator.save(emulator_fn)
    if 'harmonic' in tofit:
        emulator.set_samples(samples=samples.select(['X.*', 'Y.harmonic.*']))
        emulator.fit(name='fourier.*', batch_frac=(0.2, 0.4, 1.), learning_rate=(1e-2, 1e-4, 1e-6), epochs=1000)
        emulator.save(emulator_fn)


def plot():
    samples = Samples.load(samples_fn)
    cosmo = DESI(engine=EmulatedEngine.load(emulator_fn))
    plot_residual_background(samples, emulated_samples=cosmo, fn=train_dir / 'background.png')
    plot_residual_thermodynamics(samples, emulated_samples=cosmo, fn=train_dir / 'thermodynamics.png')
    plot_residual_primordial(samples, emulated_samples=cosmo, fn=train_dir / 'primordial.png')
    plot_residual_fourier(samples, emulated_samples=cosmo, fn=train_dir / 'fourier.png')
    plot_residual_harmonic(samples, emulated_samples=cosmo, fn=train_dir / 'harmonic.png')


if __name__ == '__main__':

    """Uncomment to run."""

    setup_logging()
    sample()
    #fit()
    #plot()