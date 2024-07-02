import os
from pathlib import Path

this_dir = Path(__file__).parent
train_dir = Path(os.getenv('SCRATCH')) / 'emulators/train/classy/'
samples_fn = {'mpk': train_dir / 'samples_mpk', 'cmb': train_dir / 'samples_cmb'}
emulator_dir = this_dir / 'classy'
emulator_fn = emulator_dir / 'emulator.npy'


def sample(samples_fn, observable='mpk', start=0, stop=100000):
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.emulators import QMCSampler, get_calculator, setup_logging

    setup_logging()
    # What makes classy run so slow: harmonic, lensing, Omega_k far from 1.
    if 'mpk' in observable:
        cosmo = DESI(non_linear='mead', engine='class')
        params = {'logA': (2.5, 3.5), 'n_s': (0.88, 1.06), 'h': (0.4, 1.), 'omega_b': (0.019, 0.026), 'omega_cdm': (0.08, 0.2), 'm_ncdm': (0., 0.8),
                  'Omega_k': (-0.3, 0.3), 'w0_fld': (-1.5, -0.5), 'wa_fld': (-0.7, 0.7)}
        calculator = get_calculator(cosmo, section=['background', 'thermodynamics', 'primordial', 'fourier'])
        sampler = QMCSampler(calculator, params, engine='rqrs', seed=0.5, save_fn='{}_{:d}_{:d}'.format(samples_fn, start, stop))
        sampler.run(save_every=10, niterations=stop - start, nstart=start)

    if 'cmb' in observable:
        cosmo = DESI(lensing=True, engine='class')
        params = {'logA': (2.5, 3.5), 'n_s': (0.88, 1.06), 'h': (0.4, 1.), 'omega_b': (0.019, 0.026), 'omega_cdm': (0.08, 0.2), 'm_ncdm': (0., 0.8),
                  'Omega_k': (-0.3, 0.3), 'w0_fld': (-1.5, -0.5), 'wa_fld': (-0.7, 0.7), 'tau_reio': (0.02, 0.12)}
        cosmo = cosmo.clone(extra_params={'number_count_contributions': []})
        #cosmo = cosmo.clone(extra_params={'output': ['tCl', 'pCl', 'lCl']})
        calculator = get_calculator(cosmo, section=['background', 'thermodynamics', 'primordial', 'harmonic'])
        sampler = QMCSampler(calculator, params=params, engine='rqrs', seed=0.5, save_fn='{}_{:d}_{:d}'.format(samples_fn, start, stop))
        sampler.run(save_every=2, niterations=stop - start, nstart=start, timeout=1)


def fit(samples_fn, tofit=('background', 'thermodynamics', 'primordial', 'fourier', 'harmonic')):
    import glob
    import logging
    import numpy as np
    #import tensorflow as tf
    #tf.keras.backend.set_floatx('float64')

    from cosmoprimo.emulators import Emulator, EmulatedEngine, MLPEmulatorEngine, Samples, FourierNormOperation, Log10Operation, ArcsinhOperation, PCAOperation, ChebyshevOperation, setup_logging

    logger = logging.getLogger('Fit')
    setup_logging()

    def load_samples(**kwargs):
        samples = Samples.concatenate([Samples.load(fn, **kwargs) for fn in glob.glob(str(samples_fn) + '*')])
        mask = samples.isfinite()
        logger.info('Removing {:d} / {:d} NaN samples.'.format((~mask).sum(), mask.size))
        return samples[mask]

    operations = []
    operations.append(FourierNormOperation(ref_pk_name='fourier.pk.delta_cb.delta_cb'))
    engine = {}
    #engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(20,) * 2)
    engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(20,) * 3)
    engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(10,) * 5)
    engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(30,) * 5)
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
        samples = load_samples(include=['X.*', 'Y.background.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
        #z = np.linspace(0., 10., 1000)
        for name in samples.columns('Y.*'):
            if samples[name].ndim > 1:
                ee = MLPEmulatorEngine(nhidden=(512,) * 4, yoperation=ArcsinhOperation())  #, yoperation=Log10Operation()) #, cChebyshevOperation(axis=-1, order=100))
                #samples[name] = samples[name][...,1000:]
            else:
                ee = MLPEmulatorEngine(nhidden=(20,) * 2)
            engine[name[2:]] = ee
        emulator.set_engine(engine)
        emulator.set_samples(samples=samples)
        #emulator.fit(name='background.*', batch_frac=[0.02, 0.05, 0.1, 0.2, 0.4, 0.5], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], epochs=1000, verbose=True)
        emulator.fit(name='background.*', batch_frac=[0.02, 0.05, 0.3], learning_rate=[1e-2, 1e-3, 1e-4], epochs=1000, verbose=False)
        emulator.save(emulator_fn)
    if 'thermodynamics' in tofit:
        samples = load_samples(include=['X.*', 'Y.thermodynamics.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
        emulator.set_samples(samples=samples)
        #emulator.fit(name='thermodynamics.rs_drag', batch_frac=[0.02, 0.05, 0.1, 0.2, 0.4, 0.5], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], epochs=1000, verbose=True)
        emulator.fit(name='thermodynamics.rs_drag', batch_frac=[0.02, 0.05, 0.1, 0.2, 0.4, 0.5], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], epochs=1000, verbose=True)
        #emulator.fit(name='thermodynamics.rs_drag', batch_frac=[0.05, 0.1, 0.2, 0.4, 0.5], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-7], epochs=1000, verbose=True)
        #emulator.fit(name='thermodynamics.rs_drag', batch_frac=[0.05, 0.1, 0.2, 0.4], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5], epochs=1000, verbose=True)
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
        emulator.fit(name='harmonic.*', batch_frac=(0.2, 0.4, 1.), learning_rate=(1e-2, 1e-4, 1e-6), epochs=1000)
        emulator.save(emulator_fn)


def plot(samples_fn, toplot=('background', 'thermodynamics', 'primordial', 'fourier', 'harmonic')):
    import glob
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.emulators import Emulator, EmulatedEngine, Samples, plot_residual_background, plot_residual_thermodynamics, plot_residual_primordial, plot_residual_fourier, plot_residual_harmonic, setup_logging

    setup_logging()
    
    def load_samples(**kwargs):
        samples = Samples.concatenate([Samples.load(fn, **kwargs) for fn in glob.glob(str(samples_fn) + '*')[:1]])
        return samples
        #mask = samples.isfinite()
        #return samples[mask]

    emulator = Emulator.load(emulator_fn)
    #engine = emulator.engines['thermodynamics.rs_drag']
    #print(engine.params)
    #print([(operation._direct, operation._locals) for operation in engine.xoperations])
    #print([(operation._direct, operation._locals) for operation in engine.yoperations])
    #print([(operation._direct, operation._locals) for operation in engine.model_operations])
    #exit()
    cosmo = DESI(engine=EmulatedEngine.load(emulator_fn))

    samples = load_samples(include=['X.*', 'Y.thermodynamics.*'])
    for i in range(10):
        params = {name[2:]: samples[name][i] for name in samples.columns('X.*')}
        c = cosmo.clone(**params)
        ref = DESI(**params)
        print(c.rs_drag, ref.rs_drag, samples['Y.thermodynamics.rs_drag'][i])
    
    if 'background' in toplot:
        samples = load_samples(include=['X.*', 'Y.background.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
        plot_residual_background(samples, emulated_samples=cosmo, subsample=0.1, fn=emulator_dir / 'background.png')
    if 'thermodynamics' in toplot:
        samples = load_samples(include=['X.*', 'Y.thermodynamics.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
        plot_residual_thermodynamics(samples, emulated_samples=cosmo, subsample=0.1, fn=emulator_dir / 'thermodynamics.png')
    if 'primordial' in toplot:
        plot_residual_primordial(samples, emulated_samples=cosmo, fn=emulator_dir / 'primordial.png')
    if 'fourier' in toplot:
        plot_residual_fourier(samples, emulated_samples=cosmo, fn=emulator_dir / 'fourier.png')
    if 'harmonic' in toplot:
        plot_residual_harmonic(samples, emulated_samples=cosmo, fn=emulator_dir / 'harmonic.png')

        
def test():

    def load_samples(**kwargs):
        samples = Samples.concatenate([Samples.load(fn, **kwargs) for fn in glob.glob(str(samples_fn) + '*')])
        mask = samples.isfinite()
        logger.info('Removing {:d} / {:d} NaN samples.'.format((~mask).sum(), mask.size))
        return samples[mask]

    samples = load_samples(include=['X.*', 'Y.thermodynamics.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
    params = [name[2:] for name in samples.columns('X.*')]
    X = np.column_stack([samples['X.' + param] for param in samples])
    Y = samples['Y.thermodynamics.rs_drag']
    from cosmoprimo.emulators import MLPEmulatorEngine
    
    fn = 'tmp.npy'
    if True:
        engine = MLPEmulatorEngine(nhidden=(10,) * 3)
        engine.initialize(params=params)
        engine.fit(X, Y, batch_frac=[0.02, 0.05, 0.1, 0.2, 0.4, 0.5], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], epochs=1000, verbose=True)
        engine.save(fn)
    if True:
        engine = MLPEmulatorEngine.load(fn)
        from jax import vmap
        Y_pred = vmap(engine.predict)(X)
        diff = np.max(Y / Y_pred - 1.)
        print(diff.mean(), diff.max())


if __name__ == '__main__':

    """Uncomment to run."""

    #todo = ['sample']
    todo = ['fit', 'plot']

    if todo:
        from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

        setup_logging()

        queue = Queue('classy_emulator')
        queue.clear(kill=False)

        environ = Environment('nersc-cosmodesi', command='module unload cosmoprimo')
        output, error = '_sbatch/slurm-%j.out', '_sbatch/slurm-%j.err'
        tm = TaskManager(queue=queue, environ=environ)

        if 'sample' in todo:

            for observable in ['mpk', 'cmb'][1:]:

                if observable == 'mpk':
                    nsamples = 100000
                    nworkers = 10
                    tm_sample = tm.clone(scheduler=dict(max_workers=nworkers), provider=dict(provider='nersc', time='01:30:00', mpiprocs_per_worker=64, nodes_per_worker=1, output=output, error=error))
                if observable == 'cmb':
                    nsamples = 80000
                    nworkers = 80
                    tm_sample = tm.clone(scheduler=dict(max_workers=nworkers), provider=dict(provider='nersc', time='01:30:00', mpiprocs_per_worker=1, nodes_per_worker=1. / 16, output=output, error=error), environ=environ.clone(command='module unload cosmoprimo; export OMP_NUM_THREADS=16'))

                s = tm_sample.python_app(sample)
                steps = list(range(0, nsamples + 1, nsamples // nworkers))
                for start, stop in zip(steps[:-1], steps[1:]):
                    s(samples_fn[observable], observable=observable, start=start, stop=stop)
                    break
        
        if 'fit' in todo:
            for observable in ['mpk', 'cmb'][:1]:
                tofit = ['background', 'thermodynamics', 'primordial', 'fourier'][1:2]
                if observable == 'cmb': tofit = ['harmonic']
                for tofit in tofit:
                    fit(samples_fn[observable], tofit=tofit)
        
        if 'plot' in todo:
            for observable in ['mpk', 'cmb'][:1]:
                toplot = ['background', 'thermodynamics', 'primordial', 'fourier'][1:2]
                if observable == 'cmb': toplot = ['harmonic']
                for toplot in toplot:
                    plot(samples_fn[observable], toplot=toplot)
