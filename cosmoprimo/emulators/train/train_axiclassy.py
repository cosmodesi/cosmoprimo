import os
import glob
import logging
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from cosmoprimo.emulators import Emulator, EmulatedEngine, MLPEmulatorEngine, Samples, FourierNormOperation, Log10Operation, ArcsinhOperation, PCAOperation, ChebyshevOperation, Operation, setup_logging
from cosmoprimo.emulators import plot_residual_background, plot_residual_thermodynamics, plot_residual_fourier, plot_residual_harmonic
from cosmoprimo.fiducial import DESI

logger = logging.getLogger('axiclassy')

this_dir = Path(__file__).parent
train_dir = Path(os.getenv('SCRATCH', '')) / 'emulators/train/axiclassy/base/'
samples_fn = {name: train_dir / 'samples' for name in ['background', 'thermodynamics', 'fourier', 'harmonic']}
emulator_dir = this_dir / 'axiclassy_base'
emulator_fn = emulator_dir / 'emulator.npy'


def sample(samples_fn, section='background', start=0, stop=100000):
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.emulators import QMCSampler, get_calculator, setup_logging

    setup_logging()
    cosmo = DESI(engine='axiclass')
    #cosmo = DESI(engine='camb', neutrino_hierarchy='degenerate')

    if section == 'background':
        params = {'h': (0.2, 1.), 'omega_cdm': (0.01, 0.90), 'omega_b': (0.005, 0.05), 'm_ncdm': (0., 5.), 'w0_fld': (-3., 1.), 'wa_fld': (-3., 2.)}
        calculator = get_calculator(cosmo, section=[section])
        sampler = QMCSampler(calculator, params, engine='lhs', seed=42, save_fn='{}_{:d}_{:d}.npz'.format(samples_fn, start, stop))
        sampler.run(save_every=100, niterations=stop - start, nstart=start)

    if section == 'thermodynamics':
        #cosmo = DESI(engine='camb', neutrino_hierarchy='degenerate')
        params = {'h': (0.2, 1.), 'omega_cdm': (0.01, 0.90), 'omega_b': (0.005, 0.05), 'm_ncdm': (0., 5.), 'w0_fld': (-2., 0.), 'wa_fld': (-3., 2.)}
        #cosmo = DESI(engine='camb')
        #params = {'h': (0.4, 1.0), 'omega_cdm': (0.08, 0.20), 'omega_b': (0.01933, 0.02533)}
        #params = {'h': (0.4, 1.0), 'omega_cdm': (0.09, 0.15), 'omega_b': (0.015, 0.030), 'w0_fld': (-1.5, 0.), 'wa_fld': (-2., 1.5)}
        calculator = get_calculator(cosmo, section=[section])
        sampler = QMCSampler(calculator, params, engine='lhs', seed=42, save_fn='{}_{:d}_{:d}.npz'.format(samples_fn, start, stop))
        sampler.run(save_every=100, niterations=stop - start, nstart=start)

    if section == 'harmonic':
        #extra_params = {'nonlinear_min_k_max': 20, 'accurate_lensing': 1, 'delta_l_max': 800}
        extra_params = {'non_linear': 'hmcode', 'hmcode_version': '2020', 'recombination': 'HyRec', 'l_max_scalars': 9500, 'delta_l_max': 1800, 'P_k_max_h/Mpc': 100.0, 'l_logstep': 1.025, 'l_linstep': 20, 'perturbations_sampling_stepsize': 0.05, 'l_switch_limber': 30.0, 'hyper_sampling_flat': 32.0, 'l_max_g': 40, 'l_max_ur': 35, 'l_max_pol_g': 60, 'ur_fluid_approximation': 2, 'ur_fluid_trigger_tau_over_tau_k': 130.0, 'radiation_streaming_approximation': 2, 'radiation_streaming_trigger_tau_over_tau_k': 240.0, 'hyper_flat_approximation_nu': 7000.0, 'transfer_neglect_delta_k_S_t0': 0.17, 'transfer_neglect_delta_k_S_t1': 0.05, 'transfer_neglect_delta_k_S_t2': 0.17, 'transfer_neglect_delta_k_S_e': 0.17, 'accurate_lensing': True, 'start_small_k_at_tau_c_over_tau_h': 0.0004, 'start_large_k_at_tau_h_over_tau_k': 0.05, 'tight_coupling_trigger_tau_c_over_tau_h': 0.005, 'tight_coupling_trigger_tau_c_over_tau_k': 0.008, 'start_sources_at_tau_c_over_tau_h': 0.006, 'l_max_ncdm': 30, 'tol_ncdm_synchronous': 1e-06}
        cosmo = cosmo.clone(lensing=True, non_linear='hmcode', YHe='BBN', extra_params=extra_params, scf_potential='axion', n_axion=3.0, log10_axion_ac=-3.562, fraction_axion_ac=0.122, scf_parameters__1=2.83, scf_parameters__2=0.0, scf_evolve_as_fluid=False, scf_evolve_like_axionCAMB=False, scf_has_perturbations=True, attractor_ic_scf=False, compute_phase_shift=False, include_scf_in_delta_m=True, include_scf_in_delta_cb=True}
        params = {'logA': (2.5, 3.5), 'n_s': (0.88, 1.06), 'h': (0.4, 1.), 'omega_b': (0.019, 0.025), 'omega_cdm': (0.08, 0.2), 'tau_reio': (0.02, 0.12), 'log10_axion_ac': (-3.9, -3.2), 'fraction_axion_ac': (0.0, 0.3), 'scf_parameters__1': (0, 3.2)}
        calculator = get_calculator(cosmo, section=['background', 'thermodynamics', 'primordial', 'harmonic'])
        sampler = QMCSampler(calculator, params=params, engine='lhs', seed=42, save_fn='{}_{:d}_{:d}.npz'.format(samples_fn, start, stop))
        sampler.run(save_every=2, niterations=stop - start, nstart=start)


def load_samples(samples_fn, **kwargs):
    from cosmoprimo.emulators import Samples
    list_samples = []
    ngood, ntotal = 0, 0
    for fn in glob.glob(str(samples_fn) + '*'):
        samples = Samples.load(fn, **kwargs)
        if 'X.w0_fld' in samples.columns('X.*'):
            if 'X.wa_fld' in samples.columns('X.*'):
                samples = samples[samples['X.w0_fld'] + samples['X.wa_fld'] < 0.]
            else:
                samples = samples[samples['X.w0_fld'] < 0.]
        mask = samples.isfinite()
        ngood += mask.sum()
        ntotal += mask.size
        list_samples.append(samples[mask])
    logger.info('Keeping {:d} / {:d} not NaN samples.'.format(ngood, ntotal))
    return Samples.concatenate(list_samples)
        

def fit(samples_fn, section=('background', 'thermodynamics', 'primordial', 'fourier', 'harmonic')):

    operations = []
    operations.append(FourierNormOperation(ref_pk_name='fourier.pk.delta_cb.delta_cb'))
    engine = {}
    #engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(20,) * 2)
    #engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(20,) * 3)
    engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(10,) * 5, activation='tanh')
    #for name in ['rs_drag', 'rs_star']:
    #    engine['thermodynamics.{}'.format(name)] = MLPEmulatorEngine(nhidden=(10,) * 5, yoperation=[Operation("v / X['h']", inverse="v * X['h']"), Operation("v * (1. + 404. * jnp.exp(20.56 * (X['w0_fld'] - 1. / 3.)))", "v / (1. + 404. * jnp.exp(20.56 * (X['w0_fld'] - 1. / 3.)))")], activation='tanh')
        #engine['thermodynamics.{}'.format(name)] = MLPEmulatorEngine(nhidden=(64,) * 5, yoperation=[Operation("v / X['h']", inverse="v * X['h']")], activation='tanh')
    #engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(20,) * 5)
    #engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(20,) * 6)
    #engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(20,) * 7)
    #engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(20,) * 7, yoperation=Log10Operation())
    engine['primordial.*'] = MLPEmulatorEngine(nhidden=(20,) * 2)
    #engine['fourier.*'] = MLPEmulatorEngine(nhidden=(512,) * 3, yoperation=[ChebyshevOperation(axis=0, order=100), ChebyshevOperation(axis=1, order=10)])
    #engine['fourier.*'] = MLPEmulatorEngine(nhidden=(512,) * 3)
    #engine['fourier.pk.delta_cb.delta_cb'] = MLPEmulatorEngine(nhidden=(512,) * 3, yoperation=[ChebyshevOperation(axis=0, order=100)])
    #engine['fourier.*'] = MLPEmulatorEngine(nhidden=(64,) * 3, yoperation=PCAOperation(npcs=30), activation='silu')
    engine['harmonic.*'] = MLPEmulatorEngine(nhidden=(64,) * 4, yoperation=[Operation("v / jnp.exp(X['logA'] - 3.) / jnp.exp(-2 * X['tau_reio'])", inverse="v * jnp.exp(X['logA'] - 3.) * jnp.exp(-2 * X['tau_reio'])")]) #, yoperation=[ChebyshevOperation(axis=0, order=50)])

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
        del samples['X.omega_b']
        del samples['X.omega_cdm']
        for name in samples.columns('Y.*'):
            if samples[name].ndim > 1:
                #ee = MLPEmulatorEngine(nhidden=(64,) * 5, activation='tanh') #, yoperation=Log10Operation())
                #ee = MLPEmulatorEngine(nhidden=(64,) * 4, model_yoperation=PCAOperation(npcs=30), activation='silu')
                #ee = MLPEmulatorEngine(nhidden=(64,) * 6, model_yoperation=PCAOperation(npcs=30), activation='silu')
                #ee = MLPEmulatorEngine(nhidden=(128,) * 6, activation='silu')
                ee = MLPEmulatorEngine(nhidden=(64,) * 12, activation='silu')
                #ee = MLPEmulatorEngine(nhidden=(32,) * 6, model_yoperation=PCAOperation(npcs=30), activation='relu')
                #ee = MLPEmulatorEngine(nhidden=(64,) * 4, activation='silu', yoperation=Log10Operation())
                #ee = MLPEmulatorEngine(nhidden=(64,) * 2, activation='tanh')
            else:
                #ee = MLPEmulatorEngine(nhidden=(20,) * 7)
                ee = MLPEmulatorEngine(nhidden=(20,))
            engine[name[2:]] = ee
        emulator.set_engine(engine)
        emulator.set_samples(samples=samples)
        emulator.fit(name='background.comoving_radial_distance', batch_frac=[1.] * 6, learning_rate=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7], batch_norm=True, learning_rate_scheduling=False, epochs=50000, patience=10000)
        emulator.save(emulator_fn)
    if 'thermodynamics' in section:
        samples = load_samples(samples_fn, include=['X.*', 'Y.thermodynamics.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
        emulator.set_samples(samples=samples)
        #emulator.fit(name='thermodynamics.rs_drag', batch_frac=[0.02, 0.05, 0.1, 0.2, 0.4, 0.5], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], epochs=1000, verbose=True)
        emulator.fit(name='thermodynamics.*', batch_frac=[0.02, 0.05, 0.1, 0.2, 0.4, 0.5], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], patience=5000, epochs=50000)
        emulator.save(emulator_fn)
    if 'primordial' in section:
        emulator.set_samples(samples=samples.select(['X.logA', 'X.n_s', 'Y.primordial.*']))
        emulator.fit(name='primordial.*', batch_frac=(0.2, 0.4, 1.), learning_rate=(1e-2, 1e-4, 1e-6), epochs=1000)
        emulator.save(emulator_fn)
    if 'fourier' in section:
        names = ['fourier.pk.delta_cb.delta_cb', 'fourier.pk.delta_m.delta_m', 'fourier.pkz']
        samples = load_samples(samples_fn, include=['X.*', 'Y.fourier.k', 'Y.fourier.z', 'Y.fourier.z_non_linear'] + ['Y.' + name for name in names], exclude=['X.tau_reio'])
        for name in names:
            yoperation = []
            if name != 'fourier.pk.delta_cb.delta_cb':
                yoperation.append(Log10Operation())
            model_yoperation = []
            #model_yoperation = [PCAOperation(npcs=30)]
            #engine[name] = MLPEmulatorEngine(nhidden=(128,) * 5, model_yoperation=model_yoperation, activation='silu')
            engine[name] = MLPEmulatorEngine(nhidden=(64,) * 5, yoperation=yoperation, model_yoperation=model_yoperation, activation='silu')
        emulator.set_engine(engine)
        emulator.set_samples(samples=samples)
        for name in names[:1]:
            #emulator.fit(name=name, batch_frac=[0.2, 0.3, 0.4, 0.5, 1.], learning_rate=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7], batch_norm=False, learning_rate_scheduling=False, epochs=50000, patience=10000)
            emulator.fit(name=name, batch_frac=[0.2, 0.3, 0.3, 0.4, 0.5, 1.], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], batch_norm=False, learning_rate_scheduling=False, epochs=10000, patience=1000)
            #emulator.fit(name=name, batch_frac=[1.], learning_rate=[1e-2], batch_norm=False, learning_rate_scheduling=False, epochs=100, patience=100)
        pkname = 'fourier.pk.delta_cb.delta_cb'
        yoperation = emulator.engines[pkname].yoperations[-1]
        emulator.engines[pkname].yoperations = emulator.engines[pkname].yoperations[:-1]
        X = {name: emulator._samples_operations[pkname]['X.' + name] for name in emulator.engines[pkname].params}
        emulator.save(emulator_fn)
    if 'harmonic' in section:
        samples = load_samples(samples_fn, include=['X.*', 'Y.harmonic.*'])
        for name in samples.columns('X.*'):
            print(name, samples[name].min(), samples[name].max())
        emulator.set_samples(samples=samples.select(['X.*', 'Y.harmonic.*']))
        #emulator.fit(name='harmonic.lensed_cl.tt', batch_frac=[0.2, 0.3, 0.3, 0.4, 0.5, 1.], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], patience=1000, epochs=5000)
        emulator.fit(name='harmonic.lensed_cl.tt', batch_frac=[0.5, 0.7, 1., 1.], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5], validation_frac=0.2, patience=1000, epochs=5000)
        emulator.save(emulator_fn)


def plot(samples_fn, section=('background', 'thermodynamics', 'primordial', 'fourier', 'harmonic')):

    cosmo = DESI(engine=EmulatedEngine.load(emulator_fn), ellmax_cl=3500)
    #cosmo = DESI(engine=EmulatedEngine.load({Path(__file__).parent / 'cosmopower_jense2024_base_w_wa/emulator_{}.npy'.format(section): 'https://github.com/adematti/cosmoprimo-emulators/raw/refs/heads/main/cosmopower_jense2024_base_w_wa/emulator_{}.npy'.format(section) for section in ['thermodynamics']}))

    if 'background' in section:
        samples = load_samples(samples_fn, include=['X.*', 'Y.background.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
        plot_residual_background(samples, emulated_samples=cosmo, subsample=0.01, fn=emulator_dir / 'background.png')
    if 'thermodynamics' in section:
        samples = load_samples(samples_fn, include=['X.*', 'Y.thermodynamics.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
        plot_residual_thermodynamics(samples, emulated_samples=cosmo, subsample=0.01, fn=emulator_dir / 'thermodynamics.png')
    if 'primordial' in section:
        plot_residual_primordial(samples, emulated_samples=cosmo, fn=emulator_dir / 'primordial.png')
    if 'fourier' in section:
        samples = load_samples(samples_fn, include=['X.*', 'Y.fourier.k', 'Y.fourier.z', 'Y.fourier.z_non_linear', 'Y.fourier.pk.delta_cb.delta_cb', 'Y.fourier.pk.delta_m.delta_m'], exclude=['X.tau_reio'])
        plot_residual_fourier(samples, emulated_samples=cosmo, subsample=0.01, volume=None, fn=emulator_dir / 'fourier.png')
    if 'harmonic' in section:
        samples = load_samples(samples_fn, include=['X.*', 'Y.harmonic.lensed_cl.tt'])
        plot_residual_harmonic(samples, emulated_samples=cosmo, subsample=0.01, fsky=None, fn=emulator_dir / 'harmonic.png')

def plot_cl():
    from cosmoprimo.emulators import mask_subsample, InputSampler, get_calculator

    cosmo = DESI(engine=EmulatedEngine.load(emulator_fn))
    #cosmo = DESI(engine=EmulatedEngine.load('classy_base_mnu_w_wa/emulator.npy'))
    #cosmo = DESI(engine='capse')
    samples = load_samples(samples_fn['harmonic'], include=['X.*', 'Y.harmonic.lensed_cl.tt'])
    if True:
        values = samples['Y.harmonic.lensed_cl.tt']
        ells = np.arange(values.shape[-1])
        values /= np.median(values, axis=0)
        ax = plt.gca()
        qs = [0., 0.001, 0.01, 0.1, 0.9, 0.99, 0.999, 1.]
        for q in qs:
            ax.plot(ells, np.quantile(values, q=q, axis=0), label='quantile {:.1f}%'.format(100 * q))
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$C_\ell$ / median($C_\ell$)')
        ax.legend(frameon=False, ncols=2, loc=2)
        ax.set_yscale('log')
        plt.savefig('cl.png')
        
    else:
        samples = samples[mask_subsample(samples.size, factor=0.01, seed=42)][:2]
        print(samples.size)
        sampler = InputSampler(get_calculator(cosmo, section='harmonic'), samples=samples)
        emulated_samples = sampler.run()
        values = emulated_samples['Y.harmonic.lensed_cl.tt']
        ells = np.arange(values.shape[-1])
        ax = plt.gca()
        for value in values:
            mask = ells > 1
            ax.plot(ells[mask], (ells * (1 + ells) * value)[mask], color='k')
        plt.savefig('cl.png')   


if __name__ == '__main__':

    """Uncomment to run."""

    todo = ['sample']
    #todo = ['fit']
    #todo = ['plot']
    #todo = ['plot_compression']
    #todo = ['test']
    setup_logging()

    if todo:
        from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

        setup_logging()

        queue = Queue('classy_emulator')
        queue.clear(kill=False)

        environ = Environment('nersc-cosmodesi', command='module unload cosmoprimo')
        output, error = '_sbatch/slurm-%j.out', '_sbatch/slurm-%j.err'
        tm = TaskManager(queue=queue, environ=environ)

        if 'sample' in todo:

            for section in ['thermodynamics', 'fourier', 'harmonic'][2:]:
                if section == 'thermodynamics':
                    nsamples = 100000
                    nworkers = 10
                    tm_sample = tm.clone(scheduler=dict(max_workers=nworkers), provider=dict(provider='nersc', time='00:30:00', mpiprocs_per_worker=6, nodes_per_worker=0.1, output=output, error=error))
                if section == 'fourier':
                    nsamples = 100000
                    nworkers = 5
                    tm_sample = tm.clone(scheduler=dict(max_workers=nworkers), provider=dict(provider='nersc', time='01:30:00', mpiprocs_per_worker=64, nodes_per_worker=1, output=output, error=error))
                if section == 'harmonic':
                    nsamples = 80000
                    nworkers = 80
                    tm_sample = tm.clone(scheduler=dict(max_workers=nworkers), provider=dict(provider='nersc', time='02:00:00', mpiprocs_per_worker=1, nodes_per_worker=1. / 16, output=output, error=error), environ=environ.clone(command='export OMP_NUM_THREADS=8'))

                #compute = tm_sample.python_app(sample)
                compute = sample
                steps = list(range(0, nsamples + 1, nsamples // nworkers))
                for start, stop in zip(steps[:-1], steps[1:]):
                    compute(samples_fn[section], section=section, start=start, stop=stop)
                    #break

        if 'fit' in todo:
            for section in ['thermodynamics', 'fourier', 'harmonic'][2:]:
                fit(samples_fn[section], section=section)

        if 'plot' in todo:
            for section in ['thermodynamics', 'fourier', 'harmonic'][2:]:
                plot(samples_fn[section], section=section)

        if 'plot_compression' in todo:
            for section in ['fourier', 'harmonic'][:1]:
                plot_compression(samples_fn[section], section=section)

        if 'test' in todo:
            test()
