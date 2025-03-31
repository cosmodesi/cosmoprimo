"""From: https://github.com/cosmodesi/cosmoprimo/blob/main/cosmoprimo/emulators/train/classy.py"""

import os
import glob
import logging
import argparse

from matplotlib import pyplot as plt
import numpy as np
from cosmoprimo.emulators import Emulator, EmulatedEngine, MLPEmulatorEngine, Samples, FourierNormOperation, Log10Operation, ArcsinhOperation, PCAOperation, ChebyshevOperation, Operation, setup_logging
from cosmoprimo.emulators import plot_residual_background, plot_residual_thermodynamics, plot_residual_fourier, plot_residual_harmonic
from cosmoprimo.fiducial import DESI


logger = logging.getLogger('camb emulator')


def sample(samples_fn, start=0, stop=100000, config='base_w_wa'):
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.emulators import QMCSampler, get_calculator, setup_logging

    setup_logging()
    extra_params = {'kmax': 10, 'k_per_logint': 130, 'lens_potential_accuracy': 8, 'lens_margin': 2050, 'lAccuracyBoost': 1.2,
                    'min_l_logl_sampling': 6000, 'DoLateRadTruncation': False, 'recombination_model': 'CosmoRec', 'halofit_version': 'mead2020'}
    cosmo = DESI(engine='camb', lensing=True, non_linear='hmcode', kmax_pk=10., ellmax_cl=9500, YHe='BBN', extra_params=extra_params)
    reparam = None

    if config == 'base_w_wa':
        cosmo = cosmo.clone(extra_params=extra_params)
        params = {'logA': (2.9, 3.2), 'n_s': (0.9, 1.04), 'h': (0.57, 0.80), 'omega_b': (0.019, 0.025), 'omega_cdm': (0.09, 0.16), 'tau_reio': (0.02, 0.13), 'w0_fld': (-2., 0.), 'wa_fld': (-3., 2.)}
        if True:
            import time
            params.pop('h')
            params['theta_MC_100'] = (1.02, 1.06)
    
            def reparam(X):
                from cosmoprimo.cosmology import CosmologyError
                from cosmoprimo.emulators.tools import CalculatorComputationError
                toret = dict(X)
                theta = toret.pop('theta_MC_100')
                try:
                    clone = cosmo.clone(**toret).solve('h', 'theta_MC_100', theta)
                    toret['h'] = clone.H0 / 100.
                except CosmologyError as exc:
                    raise CalculatorComputationError from exc
                return toret

    if config == 'base_mnu_w_wa':
        cosmo = cosmo.clone(neutrino_hierarchy='degenerate', extra_params=extra_params)
        params = {'logA': (2.9, 3.2), 'n_s': (0.9, 1.04), 'h': (0.57, 0.80), 'omega_b': (0.019, 0.025), 'omega_cdm': (0.09, 0.16), 'tau_reio': (0.02, 0.13), 'm_ncdm': (0., 1.), 'w0_fld': (-2., 0.), 'wa_fld': (-3., 2.)}
        if True:
            params.pop('h')
            params['theta_MC_100'] = (1.02, 1.06)
    
            def reparam(X):
                from cosmoprimo.cosmology import CosmologyError
                from cosmoprimo.emulators.tools import CalculatorComputationError
                toret = dict(X)
                theta = toret.pop('theta_MC_100')
                try:
                    clone = cosmo.clone(**toret).solve('h', 'theta_MC_100', theta)
                    toret['h'] = clone.H0 / 100.
                except CosmologyError as exc:
                    raise CalculatorComputationError from exc
                return toret

    calculator = get_calculator(cosmo, section=['background', 'thermodynamics', 'primordial', 'harmonic', 'fourier'])
    sampler = QMCSampler(calculator, params=params, engine='lhs', seed=5, reparam=reparam, save_fn='{}_{:d}_{:d}.npz'.format(samples_fn, start, stop))
    sampler.run(save_every=10, niterations=stop - start, nstart=start)


def load_samples(samples_fn, sl=slice(None), **kwargs):
    import glob
    from cosmoprimo.emulators import Samples
    list_samples = []
    ngood, ntotal = 0, 0
    for fn in sorted(glob.glob(str(samples_fn) + '*'))[sl]:
        samples = Samples.load(fn, **kwargs)
        if 'X.w0_fld' in samples.columns('X.*'):
            if 'X.wa_fld' in samples.columns('X.*'):
                samples = samples[samples['X.w0_fld'] + samples['X.wa_fld'] < 0.]
            else:
                samples = samples[samples['X.w0_fld'] < 0.]
        samples.pop('X.theta_MC_100', None)
        mask = samples.isfinite()
        ngood += mask.sum()
        ntotal += mask.size
        list_samples.append(samples[mask])
    print('Keeping {:d} / {:d} not NaN samples.'.format(ngood, ntotal))
    return Samples.concatenate(list_samples)
        

def fit(samples_fn, emulator_fn, section=None, name=None, load_samples=load_samples):
    import os
    import glob
    import logging
    import argparse
    import numpy as np
    from cosmoprimo.emulators import Emulator, EmulatedEngine, MLPEmulatorEngine, Samples, FourierNormOperation, Log10Operation, ArcsinhOperation, PCAOperation, ChebyshevOperation, Operation, setup_logging
    from cosmoprimo.fiducial import DESI

    import jax
    print(jax.default_backend(), jax.devices())

    operations = []
    operations.append(FourierNormOperation(ref_pk_name='fourier.pk.delta_cb.delta_cb'))
    engine = {}
    engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(10,) * 5, activation='tanh')
    engine['primordial.*'] = MLPEmulatorEngine(nhidden=(20,) * 2)
    ellmax = 9500
    ellnorm = np.maximum(np.arange(ellmax + 1), 1) / 500
    engine['harmonic.*'] = MLPEmulatorEngine(nhidden=(128,) * 3, yoperation=[Operation("v / jnp.exp(X['logA'] - 3.) / jnp.exp(-2 * X['tau_reio']) / ellnorm**(X['n_s'] - 0.96)", inverse="v * jnp.exp(X['logA'] - 3.) * jnp.exp(-2 * X['tau_reio']) * ellnorm**(X['n_s'] - 0.96)", locals={'ellnorm': ellnorm})], activation='tanh')
    #engine['harmonic.*'] = MLPEmulatorEngine(nhidden=(64,) * 6, yoperation=[Operation("v / jnp.exp(X['logA'] - 3.) / jnp.exp(-2 * X['tau_reio'])", inverse="v * jnp.exp(X['logA'] - 3.) * jnp.exp(-2 * X['tau_reio'])", locals={'ellnorm': ellnorm})], activation='tanh')
    # engine['fourier.*'] = MLPEmulatorEngine(nhidden=(64,) * 5, yoperation=[] if name == 'delta_cb.delta_cb' else [Log10Operation()], model_yoperation=[], activation='silu')
    engine['fourier.*'] = MLPEmulatorEngine(nhidden=(64,) * 5, yoperation=[Log10Operation()], model_yoperation=[], activation='silu')

    if emulator_fn.exists():
        emulator = Emulator.load(emulator_fn)
    else:
        emulator = Emulator()
    emulator.set_engine(engine)
    emulator.yoperations = operations

    if 'background' in section:
        samples = load_samples(samples_fn, include=['X.*', 'Y.background.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
        #z = np.linspace(0., 10., 1000)
        samples['X.Omega_m'] = (samples['X.omega_cdm'] + samples['X.omega_b']) / samples['X.h']**2
        for name in samples.columns('Y.*'):
            if samples[name].ndim > 1:
                ee = MLPEmulatorEngine(nhidden=(64,) * 4, activation='tanh')
            else:
                ee = MLPEmulatorEngine(nhidden=(20,))
            engine[name[2:]] = ee
        emulator.set_engine(engine)
        emulator.set_samples(samples=samples)
        emulator.fit(name='background.*', batch_frac=[0.5, 0.8, 0.8], learning_rate=[1e-2, 1e-3, 1e-4], patience=1000, epochs=50000)
        emulator.save(emulator_fn)

    if 'thermodynamics' in section:
        samples = load_samples(samples_fn, include=['X.*', 'Y.thermodynamics.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
        emulator.set_samples(samples=samples)
        emulator.fit(name='thermodynamics.*', batch_frac=[0.5, 0.8, 0.8, 1.], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5], patience=1000, epochs=50000)
        emulator.save(emulator_fn)
        
    if 'primordial' in section:
        samples = load_samples(samples_fn, include=['X.*', 'Y.primordial.*'])
        emulator.set_samples(samples=samples.select(['X.logA', 'X.n_s', 'Y.primordial.*']))
        emulator.fit(name='primordial.*', batch_frac=(0.2, 0.4, 1.), learning_rate=(1e-2, 1e-4, 1e-6), epochs=1000)
        emulator.save(emulator_fn)

    if 'fourier' in section:
        samples = load_samples(samples_fn, include=['X.*', 'Y.fourier.k', 'Y.fourier.z', 'Y.fourier.z_non_linear', 'Y.fourier.pk.delta_m.delta_m'], exclude=['X.tau_reio'])

        for kk, _ in samples.items():
            logger.info(kk)

        engine[f'{section}.{name}'] = MLPEmulatorEngine(nhidden=(64,) * 5, yoperation=[] if name == 'delta_cb.delta_cb' else [Log10Operation()], model_yoperation=[], activation='silu')
        emulator.set_samples(samples=samples)
        emulator.fit(name=f'{section}.{name}', batch_frac=[0.2, 0.3, 0.3, 0.4, 0.5, 1.], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], batch_norm=False, learning_rate_scheduling=False, epochs=10000, patience=1000)
        
        if name == 'delta_cb.delta_cb':
            yoperation = emulator.engines['fourier.pk.delta_cb.delta_cb'].yoperations[-1]
            emulator.engines['fourier.pk.delta_cb.delta_cb'].yoperations = emulator.engines['fourier.pk.delta_cb.delta_cb'].yoperations[:-1]

        emulator.save(emulator_fn)

    if 'harmonic' in section:
        samples = load_samples(samples_fn, include=['X.*', f'Y.harmonic.{name}'])
        print(samples.columns('X.*'))
        emulator.set_samples(samples=samples.select(['X.*', f'Y.harmonic.{name}']))
        #emulator.fit(name=f'{section}.{which_cl}', batch_frac=[0.2, 0.5, 0.5, 0.5, 0.5, 1.], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], patience=1000, epochs=50000)
        print(section, name)
        emulator.fit(name=f'{section}.{name}', batch_frac=[0.8, 0.8, 1.], learning_rate=[1e-2, 1e-3, 1e-3], patience=1000, epochs=5000)
        #emulator.fit(name=f'{section}.{name}', batch_frac=[0.8, 0.8, 1.], learning_rate=[1e-2, 1e-3, 1e-3], patience=2, epochs=5)
        emulator.save(emulator_fn)


def plot(samples_fn, emulator_fn, emulator_dir, section=('background', 'thermodynamics', 'harmonic')):
    #cosmo = DESI(engine=EmulatedEngine.load(emulator_fn), ellmax_cl=9500)
    cosmo = DESI(engine='emu_camb_mnu_w_wa_cmb', ellmax_cl=9500)
    if 'background' in section:
        samples = load_samples(samples_fn, include=['X.*', 'Y.background.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
        plot_residual_background(samples, emulated_samples=cosmo, subsample=0.01, fn=emulator_dir / 'background.png')
    if 'thermodynamics' in section:
        samples = load_samples(samples_fn, include=['X.*', 'Y.thermodynamics.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
        plot_residual_thermodynamics(samples, emulated_samples=cosmo, subsample=0.01, fn=emulator_dir / 'thermodynamics.png')
    if 'harmonic' in section:
        samples = load_samples(samples_fn, include=['X.*', f'Y.harmonic.lensed_cl.*', f'Y.harmonic.lens_potential_cl.*'], sl=slice(10))
        plot_residual_harmonic(samples, emulated_samples=cosmo, subsample=0.01, quantities=['lensed_cl.tt', 'lensed_cl.te', 'lensed_cl.ee', 'lensed_cl.bb', 'lens_potential_cl.pp'], fn=emulator_dir / 'harmonic.png')
    if 'fourier' in section:
        samples = load_samples(samples_fn, include=['X.*', 'Y.fourier.k', 'Y.fourier.z', 'Y.fourier.z_non_linear', 'Y.fourier.pk.delta_m.delta_m'], exclude=['X.tau_reio']) #'Y.fourier.pk.delta_cb.delta_cb', 
        plot_residual_fourier(samples, emulated_samples=cosmo, subsample=0.01, fn=emulator_dir / 'fourier.png')


def collect_argparser():
    parser = argparse.ArgumentParser(description="Let's build an emulator for camb. We need first build a sample with --todo sample. Then, train an emulator --todo fit for each --section that we want.")

    parser.add_argument("--todo", type=str, required=True, choices=['sample', 'fit', 'combine', 'plot'])
    parser.add_argument("--config", type=str, required=True, choices=['base_w_wa', 'base_mnu_w_wa'])
    parser.add_argument("--section", type=str, required=False, default='background', choices=['background', 'thermodynamics', 'fourier', 'harmonic'])
    return parser.parse_args()


if __name__ == '__main__':
    from pathlib import Path
    from cosmoprimo.emulators import setup_logging

    setup_logging()

    args = collect_argparser()
    logger.info(args)

    samples_fn = Path(os.getenv('SCRATCH', '')) / f'emulators/train/camb/{args.config}' / 'samples'
    emulator_dir = Path(__file__).parent / f'camb_{args.config}' 

    if 'sample' in args.todo:
        from desipipe import Queue, Environment, TaskManager, spawn, setup_logging
        queue = Queue(f'camb_{args.config}')
        queue.clear(kill=False)

        environ = Environment('nersc-cosmodesi')
        output, error = '_sbatch/slurm-%j.out', '_sbatch/slurm-%j.err'
        tm = TaskManager(queue=queue, environ=environ)

        nsamples = 80000
        # certainly to much worker and time, but once in queue it works ;)
        nworkers = 64 * 5
        #nsamples = nworkers * 2
        # module unload cosmoprimo -> to be sure that I'm running with my cosmoprimo version.
        # set up to 6 if need.
        tm_sample = tm.clone(scheduler=dict(max_workers=nworkers), provider=dict(provider='nersc', time='03:00:00', mpiprocs_per_worker=1, nodes_per_worker=1. / 64, output=output, error=error), environ=environ.clone(command='module unload cosmoprimo; export OMP_NUM_THREADS=1'))

        #compute = sample  # For test you don't want to use desipipe...
        compute = tm_sample.python_app(sample)
        steps = list(range(0, nsamples + 1, nsamples // nworkers))
        for start, stop in zip(steps[:-1], steps[1:]):
            compute(samples_fn, start=start, stop=stop, config=args.config)

    section_names = {'harmonic': ['lensed_cl.tt', 'lensed_cl.te', 'lensed_cl.ee', 'lensed_cl.bb', 'lens_potential_cl.pp'],
                     'background': None,
                     'thermodynamics': None,
                     'fourier': ['pk_delta_cb.delta_cb', 'pk_delta_m.delta_m', 'fourier.pk_pkz']}

    def get_emulator_fn(section, name):
        if name is not None:
            return emulator_dir / f'emulator_{section}_{name}.npy'
        return emulator_dir / f'emulator_{section}.npy'

    if 'fit' in args.todo:
        from desipipe import Queue, Environment, TaskManager, spawn, setup_logging
        queue = Queue(f'camb-{args.config}_{args.section}')
        queue.clear(kill=False)

        environ = Environment('nersc-cosmodesi')
        output, error = '_sbatch/slurm-%j.out', '_sbatch/slurm-%j.err'
        tm = TaskManager(queue=queue, environ=environ)
        #tm_fit = tm.clone(scheduler=dict(max_workers=6), provider=dict(provider='nersc', time='00:03:00', mpiprocs_per_worker=1, nodes_per_worker=0.25, output=output, error=error, constraint='gpu'), environ=environ.clone(command='module unload cosmoprimo; export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID'))
        tm_fit = tm.clone(scheduler=dict(max_workers=6), provider=dict(provider='nersc', time='00:15:00', mpiprocs_per_worker=1, nodes_per_worker=0.25, output=output, error=error, constraint='gpu'), environ=environ.clone(command='module unload cosmoprimo; export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID'))

        compute = fit  # For test you don't want to use desipipe...
        #compute = tm_fit.python_app(fit)

        names = section_names[args.section]
        if names is None:
            compute(samples_fn, get_emulator_fn(args.section, None), section=args.section, name=None)
        else:
            for name in names:
                compute(samples_fn, get_emulator_fn(args.section, name), section=args.section, name=name)

    if 'combine' in args.todo:
        emulator = Emulator()
        for section, names in section_names.items():
            emulator_section = Emulator()
            if names is None:
                paths = [get_emulator_fn(section, None)]
            else:
                paths = [get_emulator_fn(section, name) for name in names]
            for path in paths:
                if os.path.exists(path):
                    emulator_section.update(Emulator.load(path))
                    emulator.update(Emulator.load(path))
            emulator_section.save(get_emulator_fn(section, None))
        emulator.save(emulator_dir / 'emulator.npy')

    if 'plot' in args.todo:
        plot(samples_fn, emulator_dir / 'emulator.npy', emulator_dir, section=(args.section,))