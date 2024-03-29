import os
from pathlib import Path

this_dir = Path(__file__).parent
train_dir = Path(os.getenv('SCRATCH')) / 'emulators/classy/_train'
samples_fn = {'mpk': train_dir / 'classy/samples_mpk', 'cmb': train_dir / 'classy/samples_cmb'}
emulator_fn = this_dir / 'classy/emulator.npy'


def sample(samples_fn, observable='mpk', start=0, stop=100000):
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.emulators import QMCSampler, get_calculator, setup_logging

    setup_logging()
    # What makes classy run so slow: harmonic, lensing, Omega_k far from 1.
    if 'mpk' in observable:
        cosmo = DESI(lensing=True, non_linear='mead', engine='class')
        params = {'logA': (2.5, 3.5), 'n_s': (0.88, 1.06), 'h': (0.4, 1.), 'omega_b': (0.019, 0.026), 'omega_cdm': (0.08, 0.2), 'm_ncdm': (0., 0.8),
                  'Omega_k': (-0.3, 0.3), 'w0_fld': (-1.5, -0.5), 'wa_fld': (-0.7, 0.7)}
        calculator = get_calculator(cosmo, section=['background', 'thermodynamics', 'primordial', 'fourier'])
        sampler = QMCSampler(calculator, params, engine='rqrs', seed=0.5, save_fn='{}_{:d}_{:d}'.format(samples_fn, start, stop))
        sampler.run(save_every=10, niterations=stop - start, nstart=start)

    if 'cmb' in observable:
        params = {'logA': (2.5, 3.5), 'n_s': (0.88, 1.06), 'h': (0.4, 1.), 'omega_b': (0.019, 0.026), 'omega_cdm': (0.08, 0.2), 'm_ncdm': (0., 0.8),
                  'Omega_k': (-0.3, 0.3), 'w0_fld': (-1.5, -0.5), 'wa_fld': (-0.7, 0.7), 'tau_reio': (0.02, 0.12)}
        cosmo = cosmo.clone(extra_params={'number_count_contributions': []})
        #cosmo = cosmo.clone(extra_params={'output': ['tCl', 'pCl', 'lCl']})
        calculator = get_calculator(cosmo, section=['harmonic'])
        sampler = QMCSampler(calculator, params=params, engine='rqrs', seed=0.5, save_fn=samples_fn2)
        sampler.run(save_every=10, niterations=stop - start, nstart=start, save_fn='{}_{:d}_{:d}'.format(samples_fn, start, stop))


def fit(tofit=('background', 'thermodynamics', 'primordial', 'fourier', 'harmonic')):
    from cosmoprimo.emulators import Emulator, EmulatedEngine, MLPEmulatorEngine, Samples, FourierNormOperation, PCAOperation, ChebyshevOperation, setup_logging

    setup_logging()

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
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.emulators import EmulatedEngine, Samples, plot_residual_background, plot_residual_thermodynamics, plot_residual_primordial, plot_residual_fourier, plot_residual_harmonic

    samples = Samples.load(samples_fn)
    cosmo = DESI(engine=EmulatedEngine.load(emulator_fn))
    plot_residual_background(samples, emulated_samples=cosmo, fn=train_dir / 'background.png')
    plot_residual_thermodynamics(samples, emulated_samples=cosmo, fn=train_dir / 'thermodynamics.png')
    plot_residual_primordial(samples, emulated_samples=cosmo, fn=train_dir / 'primordial.png')
    plot_residual_fourier(samples, emulated_samples=cosmo, fn=train_dir / 'fourier.png')
    plot_residual_harmonic(samples, emulated_samples=cosmo, fn=train_dir / 'harmonic.png')


if __name__ == '__main__':

    setup_logging()
    """Uncomment to run."""

    #sample()
    #fit()
    #plot()

    todo = ['sample']

    if todo:
        from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

        queue = Queue('classy_emulator')
        queue.clear(kill=False)

        environ = Environment('nersc-cosmodesi', command='module unload cosmoprimo')
        output, error = '_sbatch/slurm-%j.out', '_sbatch/slurm-%j.err'
        tm = TaskManager(queue=queue, environ=environ)
        tm_sample = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='01:30:00', mpiprocs_per_worker=64, nodes_per_worker=1, output=output, error=error))

        if 'sample' in todo:
            sample = tm_sample.python_app(sample)

            for observable in ['mpk', 'cmb'][:1]:
                steps = list(range(0, 100000, 10000))
                for start, stop in zip(steps[:-1], steps[1:]):
                    sample(samples_fn[observable], observable=observable, start=start, stop=stop)
