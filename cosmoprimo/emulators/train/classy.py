import os
from pathlib import Path

this_dir = Path(__file__).parent
train_dir = Path(os.getenv('SCRATCH', '')) / 'emulators/train/classy/new/'
samples_fn = {'bg': train_dir / 'samples_bg', 'mpk': train_dir / 'samples_mpk', 'cmb': train_dir / 'samples_cmb'}
emulator_dir = this_dir / 'classy'
emulator_fn = emulator_dir / 'emulator.npy'


def sample(samples_fn, observable='mpk', start=0, stop=100000):
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.emulators import QMCSampler, get_calculator, setup_logging

    setup_logging()
    if 'bg' in observable:
        cosmo = DESI(engine='class')
        params = {'h': (0.2, 1.), 'omega_cdm': (0.01, 0.90), 'omega_b': (0.005, 0.05), 'm_ncdm': (0., 5.), 'w0_fld': (-3., 1.), 'wa_fld': (-3., 2.)}
        calculator = get_calculator(cosmo, section=['background', 'thermodynamics'])
        sampler = QMCSampler(calculator, params, engine='lhs', seed=42, save_fn='{}_{:d}_{:d}.npz'.format(samples_fn, start, stop))
        sampler.run(save_every=10, niterations=stop - start, nstart=start)

    # What makes classy run so slow: harmonic, lensing, Omega_k far from 1.
    if 'mpk' in observable:
        cosmo = DESI(non_linear='mead', engine='class', neutrino_hierarchy='degenerate')
        #params = {'logA': (2.5, 3.5), 'n_s': (0.88, 1.06), 'h': (0.4, 1.), 'omega_b': (0.019, 0.026), 'omega_cdm': (0.08, 0.2), 'm_ncdm': (0., 0.8), 'Omega_k': (-0.3, 0.3), 'w0_fld': (-1.5, -0.5), 'wa_fld': (-0.7, 0.7)}
        params = {'h': (0.5, 0.9), 'omega_cdm': (0.03, 0.3), 'logA': (1.5, 4.0), 'n_s': (0.8, 1.2), 'omega_b': (0.005, 0.04), 'm_ncdm': (0., 3.), 'w0_fld': (-2., 1.), 'wa_fld': (-3., 2.)}
        calculator = get_calculator(cosmo, section=['background', 'thermodynamics', 'primordial', 'fourier'])
        #sampler = QMCSampler(calculator, params, engine='rqrs', seed=0.5, save_fn='{}_{:d}_{:d}.npz'.format(samples_fn, start, stop))
        sampler = QMCSampler(calculator, params, engine='lhs', seed=42, save_fn='{}_{:d}_{:d}.npz'.format(samples_fn, start, stop))
        sampler.run(save_every=10, niterations=stop - start, nstart=start)

    if 'cmb' in observable:
        cosmo = DESI(lensing=True, engine='class')
        params = {'logA': (2.5, 3.5), 'n_s': (0.88, 1.06), 'h': (0.4, 1.), 'omega_b': (0.019, 0.026), 'omega_cdm': (0.08, 0.2), 'm_ncdm': (0., 0.8),
                  'Omega_k': (-0.3, 0.3), 'w0_fld': (-1.5, -0.5), 'wa_fld': (-0.7, 0.7), 'tau_reio': (0.02, 0.12)}
        cosmo = cosmo.clone(extra_params={'number_count_contributions': []})
        #cosmo = cosmo.clone(extra_params={'output': ['tCl', 'pCl', 'lCl']})
        calculator = get_calculator(cosmo, section=['background', 'thermodynamics', 'primordial', 'harmonic'])
        sampler = QMCSampler(calculator, params=params, engine='rqrs', seed=0.5, save_fn='{}_{:d}_{:d}.npz'.format(samples_fn, start, stop))
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
        samples = Samples.concatenate([Samples.load(fn, **kwargs) for fn in glob.glob(str(samples_fn) + '*')[:1]])
        mask = samples.isfinite()
        logger.info('Removing {:d} / {:d} NaN samples.'.format((~mask).sum(), mask.size))
        return samples[mask]

    operations = []
    operations.append(FourierNormOperation(ref_pk_name='fourier.pk.delta_cb.delta_cb'))
    engine = {}
    #engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(20,) * 2)
    engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(20,) * 3)
    engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(10,) * 5)
    #engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(20,) * 5)
    #engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(20,) * 6)
    #engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(20,) * 7)
    engine['thermodynamics.*'] = MLPEmulatorEngine(nhidden=(20,) * 7, yoperation=Log10Operation())
    engine['primordial.*'] = MLPEmulatorEngine(nhidden=(20,) * 2)
    #engine['fourier.*'] = MLPEmulatorEngine(nhidden=(512,) * 3, yoperation=[ChebyshevOperation(axis=0, order=100), ChebyshevOperation(axis=1, order=10)])
    #engine['fourier.*'] = MLPEmulatorEngine(nhidden=(512,) * 3)
    #engine['fourier.pk.delta_cb.delta_cb'] = MLPEmulatorEngine(nhidden=(512,) * 3, yoperation=[ChebyshevOperation(axis=0, order=100)])
    #engine['fourier.*'] = MLPEmulatorEngine(nhidden=(64,) * 3, yoperation=PCAOperation(npcs=30), activation='silu')
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
        samples['X.Omega_m'] = (samples['X.omega_cdm'] + samples['X.omega_b']) / samples['X.h']**2
        del samples['X.omega_b']
        del samples['X.omega_cdm']
        for name in samples.columns('Y.*'):
            if samples[name].ndim > 1:
                #ee = MLPEmulatorEngine(nhidden=(64,) * 5, activation='tanh') #, yoperation=Log10Operation())
                #ee = MLPEmulatorEngine(nhidden=(256,) * 3, activation='tanh')
                #ee = MLPEmulatorEngine(nhidden=(128,) * 4, activation='tanh')
                #ee = MLPEmulatorEngine(nhidden=(16, 32, 64, 128), activation='identity-silu')
                #ee = MLPEmulatorEngine(nhidden=(64,) * 4, activation='silu')
                #ee = MLPEmulatorEngine(nhidden=(64,) * 4, model_yoperation=PCAOperation(npcs=30), activation='silu')
                #ee = MLPEmulatorEngine(nhidden=(64,) * 6, model_yoperation=PCAOperation(npcs=30), activation='silu')
                #ee = MLPEmulatorEngine(nhidden=(128,) * 6, activation='silu')
                ee = MLPEmulatorEngine(nhidden=(64,) * 12, activation='silu')
                #ee = MLPEmulatorEngine(nhidden=(32,) * 6, model_yoperation=PCAOperation(npcs=30), activation='relu')
                #ee = MLPEmulatorEngine(nhidden=(64,) * 4, activation='silu', yoperation=Log10Operation())
                #ee = MLPEmulatorEngine(nhidden=(64,) * 2, activation='tanh')
                #ee = MLPEmulatorEngine(nhidden=(64,), activation='tanh')
                #ee = MLPEmulatorEngine(nhidden=(20, 40, 80, 160, 320, 640), yoperation=ArcsinhOperation())  #, yoperation=Log10Operation()) #, cChebyshevOperation(axis=-1, order=100))
                #samples[name] = samples[name][...,1000:]
            else:
                #ee = MLPEmulatorEngine(nhidden=(20,) * 7)
                ee = MLPEmulatorEngine(nhidden=(20,))
            engine[name[2:]] = ee
        emulator.set_engine(engine)
        emulator.set_samples(samples=samples)
        #emulator.fit(name='background.*', batch_frac=[0.02, 0.05, 0.1, 0.2, 0.4, 0.5], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], epochs=1000, verbose=True)
        #emulator.fit(name='background.*', batch_frac=[0.02, 0.05, 0.3], learning_rate=[1e-2, 1e-3, 1e-4], epochs=1000)
        #emulator.fit(name='background.comoving_radial_distance', batch_frac=[0.02, 0.1, 0.2, 0.3, 0.4, 0.5], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], epochs=10000, verbose=2)
        # [0.2, 0.2, 0.3, 0.4, 0.4]
        #emulator.fit(name='background.comoving_radial_distance', batch_frac=[0.2, 0.3, 0.4, 0.5, 1.], learning_rate=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7], batch_norm=True, learning_rate_scheduling=False, epochs=50000, patience=10000)
        emulator.fit(name='background.comoving_radial_distance', batch_frac=[1.] * 6, learning_rate=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7], batch_norm=True, learning_rate_scheduling=False, epochs=50000, patience=10000)
        #emulator.fit(name='background.rho_ncdm', batch_frac=[0.2, 0.3, 0.4, 0.5, 1.], learning_rate=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7], batch_norm=False, learning_rate_scheduling=False, epochs=50000, patience=10000)
        #emulator.fit(name='background.comoving_radial_distance', batch_frac=[0.2], learning_rate=[1e-3], epochs=10, patience=10000)
        #emulator.fit(name='background.comoving_radial_distance', batch_frac=[0.2, 0.2], learning_rate=[1e-3, 1e-4], epochs=50000)
        #emulator.fit(name='background.comoving_radial_distance', batch_frac=[0.2, 0.3, 0.3, 0.4, 0.4], learning_rate=[3e-3, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], epochs=50000)
        emulator.save(emulator_fn)
    if 'thermodynamics' in tofit:
        samples = load_samples(include=['X.*', 'Y.thermodynamics.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
        emulator.set_samples(samples=samples)
        #emulator.fit(name='thermodynamics.rs_drag', batch_frac=[0.02, 0.05, 0.1, 0.2, 0.4, 0.5], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], epochs=1000, verbose=True)
        #emulator.fit(name='thermodynamics.rs_drag', batch_frac=[0.02, 0.05, 0.1, 0.2, 0.4, 0.5], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], epochs=3000)
        #emulator.fit(name='thermodynamics.*', batch_frac=[0.02, 0.1, 0.2, 0.3, 0.4, 0.5], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], epochs=4000)
        emulator.fit(name='thermodynamics.rs_drag', batch_frac=[0.02, 0.1, 0.2, 0.3, 0.4, 0.5], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], epochs=4000)
        #emulator.fit(name='thermodynamics.rs_drag', batch_frac=[0.05, 0.1, 0.2, 0.4, 0.5], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5, 1e-7], epochs=1000, verbose=True)
        #emulator.fit(name='thermodynamics.rs_drag', batch_frac=[0.05, 0.1, 0.2, 0.4], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5], epochs=1000, verbose=True)
        emulator.save(emulator_fn)
    if 'primordial' in tofit:
        emulator.set_samples(samples=samples.select(['X.logA', 'X.n_s', 'Y.primordial.*']))
        emulator.fit(name='primordial.*', batch_frac=(0.2, 0.4, 1.), learning_rate=(1e-2, 1e-4, 1e-6), epochs=1000)
        emulator.save(emulator_fn)
    if 'fourier' in tofit:
        names = ['fourier.pk.delta_cb.delta_cb', 'fourier.pk.delta_m.delta_m', 'fourier.pkz']
        samples = load_samples(include=['X.*', 'Y.fourier.k', 'Y.fourier.z', 'Y.fourier.z_non_linear'] + ['Y.' + name for name in names], exclude=['X.tau_reio'])
        for name in names:
            yoperation = []
            if name == '!fourier.pk.delta_cb.delta_cb':
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
        """
        from cosmoprimo.jax import vmap
        scaled = vmap(yoperation)(emulator._samples_operations[pkname]['Y.' + pkname])
        emulated_scaled = vmap(emulator.engines[pkname].predict)(X)
        #sl = slice(None)
        print('LOOOL', scaled[:3], emulated_scaled[:3])
        mse = np.mean((scaled - emulated_scaled)**2)
        maxi = np.abs(scaled - emulated_scaled).max()
        print('mse', mse, mse**0.5, maxi)
        #exit()
        """    
        emulator.save(emulator_fn)
    if 'harmonic' in tofit:
        emulator.set_samples(samples=samples.select(['X.*', 'Y.harmonic.*']))
        emulator.fit(name='harmonic.*', batch_frac=(0.2, 0.4, 1.), learning_rate=(1e-2, 1e-4, 1e-6), epochs=1000)
        emulator.save(emulator_fn)


def plot(samples_fn, toplot=('background', 'thermodynamics', 'primordial', 'fourier', 'harmonic')):
    import glob
    import logging
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.emulators import Emulator, EmulatedEngine, Samples, plot_residual_background, plot_residual_thermodynamics, plot_residual_primordial, plot_residual_fourier, plot_residual_harmonic, setup_logging

    logger = logging.getLogger('Plot')
    setup_logging()

    def load_samples(**kwargs):
        samples = Samples.concatenate([Samples.load(fn, **kwargs) for fn in glob.glob(str(samples_fn) + '*')[:1]])
        mask = samples.isfinite()
        logger.info('Removing {:d} / {:d} NaN samples.'.format((~mask).sum(), mask.size))
        return samples[mask]

    #emulator = Emulator.load(emulator_fn)
    #del emulator.engines['background.z']
    #emulator.save(emulator_fn)
    #engine = emulator.engines['thermodynamics.rs_drag']
    #print(engine.params)
    #print([(operation._direct, operation._locals) for operation in engine.xoperations])
    #print([(operation._direct, operation._locals) for operation in engine.yoperations])
    #print([(operation._direct, operation._locals) for operation in engine.model_operations])
    #exit()
    cosmo = DESI(engine=EmulatedEngine.load(emulator_fn))
    """
    samples = load_samples(include=['X.*', 'Y.thermodynamics.*'])
    import numpy as np
    #print(np.nanmin(samples['Y.thermodynamics.YHe']), np.nanmax(samples['Y.thermodynamics.YHe']))
    for i in range(10):
        params = {name[2:]: samples[name][i] for name in samples.columns('X.*')}
        c = cosmo.clone(**params)
        ref = DESI(**params)
        print(c.rs_drag, ref.rs_drag, samples['Y.thermodynamics.rs_drag'][i])
        #print(c.YHe, ref.YHe, samples['Y.thermodynamics.YHe'][i])
    """
    if 'background' in toplot:
        samples = load_samples(include=['X.*', 'Y.background.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
        plot_residual_background(samples, emulated_samples=cosmo, subsample=0.01, fn=emulator_dir / 'background.png')
    if 'thermodynamics' in toplot:
        samples = load_samples(include=['X.*', 'Y.thermodynamics.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
        plot_residual_thermodynamics(samples, emulated_samples=cosmo, subsample=0.1, fn=emulator_dir / 'thermodynamics.png')
    if 'primordial' in toplot:
        plot_residual_primordial(samples, emulated_samples=cosmo, fn=emulator_dir / 'primordial.png')
    if 'fourier' in toplot:
        samples = load_samples(include=['X.*', 'Y.fourier.k', 'Y.fourier.z', 'Y.fourier.z_non_linear', 'Y.fourier.pk.delta_cb.delta_cb', 'Y.fourier.pk.delta_m.delta_m'], exclude=['X.tau_reio'])
        plot_residual_fourier(samples, emulated_samples=cosmo, subsample=0.01, volume=None, fn=emulator_dir / 'fourier.png')
    if 'harmonic' in toplot:
        plot_residual_harmonic(samples, emulated_samples=cosmo, fn=emulator_dir / 'harmonic.png')


def plot_compression(samples_fn, toplot=('background', 'thermodynamics', 'primordial', 'fourier', 'harmonic')):
    import glob
    import logging
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.emulators import Emulator, EmulatedEngine, Samples, plot_residual_background, plot_residual_thermodynamics, plot_residual_primordial, plot_residual_fourier, plot_residual_harmonic, PCAOperation, setup_logging

    logger = logging.getLogger('Plot')
    setup_logging()

    def load_samples(**kwargs):
        samples = Samples.concatenate([Samples.load(fn, **kwargs) for fn in glob.glob(str(samples_fn) + '*')])
        mask = samples.isfinite()
        logger.info('Removing {:d} / {:d} NaN samples.'.format((~mask).sum(), mask.size))
        return samples[mask]

    if 'background' in toplot:
        quantities = ['rho_ncdm', 'p_ncdm', 'time', 'comoving_radial_distance']
        samples = load_samples(include=['Y.background.{}'.format(name) for name in quantities])
        samples_compression = samples.deepcopy()
        operation = PCAOperation(npcs=30)
        import jax
        for quantity in quantities:
            quantity = 'Y.background.' + quantity
            operation.initialize(samples[quantity])
            samples_compression[quantity] = jax.vmap(operation.inverse)(jax.vmap(operation)(samples[quantity]))
        plot_residual_background(samples, emulated_samples=samples_compression, quantities=quantities, subsample=0.01, fn=emulator_dir / 'compression_background.png')

    if 'fourier' in toplot:
        quantities = ['pk.delta_cb.delta_cb']
        samples = load_samples(include=['Y.fourier.{}'.format(name) for name in quantities])
        samples_compression = samples.deepcopy()
        operation = PCAOperation(npcs=30)
        import jax
        for quantity in quantities:
            quantity = 'Y.fourier.' + quantity
            operation.initialize(samples[quantity])
            samples_compression[quantity] = jax.vmap(operation.inverse)(jax.vmap(operation)(samples[quantity]))
        plot_residual_background(samples, emulated_samples=samples_compression, quantities=quantities, subsample=0.01, fn=emulator_dir / 'compression_background.png')


def test():
    import glob
    import numpy as np
    from matplotlib import pyplot as plt
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.cosmology import Cosmology, DefaultBackground
    from cosmoprimo.emulators import Emulator, EmulatedEngine, Samples, QMCSampler, InputSampler, get_calculator, HarmonicNormOperation, FourierNormOperation, plot_residual_fourier
    from cosmoprimo.jax import vmap
    from cosmoprimo.emulators.tools.base import batch_vmap

    if 0:
        cosmo = DESI(engine='class', m_ncdm=0., lensing=True)
        params = {'h': (0.7, 0.7001), 'omega_cdm': (0.02, 0.02001), 'omega_b': (0.005, 0.005001), 'm_ncdm': (0.06, 0.06001), 'logA': (3., 3.0001), 'w0_fld': (-3., 1.), 'wa_fld': (-3., 2.)}
        calculator = get_calculator(cosmo, section=['harmonic'])
        sampler = QMCSampler(calculator, params, engine='lhs', seed=42)
        samples = sampler.run(niterations=4)
        samples = samples[samples.isfinite()]
        X = {name[2:]: samples[name] for name in samples.columns('X.*')}
        Y = {name[2:]: samples[name] for name in samples.columns('Y.*')}
        Yref = dict(Y)
        operation = HarmonicNormOperation()
        operation.initialize(Y)
        ax = plt.gca()
        ax.plot(operation.wells['unlensed_cl'], operation.windows['unlensed_cl'])
        ax.set_xscale('log')
        plt.savefig('tmp.png')
        plt.close(plt.gcf())
        Y = batch_vmap(operation, (0, 0))(Y, X)
        ax = plt.gca()
        for i, cl in enumerate(Y['harmonic.unlensed_cl.tt']):
            ell = np.arange(cl.shape[-1])
            factor = ell * (ell + 1)
            print(i, cl)
            ax.plot(ell, factor * cl)
        ax.set_xscale('log')
        plt.savefig('tmp1.png')
        plt.close(plt.gcf())
        Y = batch_vmap(operation.inverse, (0, 0))(Y, X)
        ax = plt.gca()
        name = 'harmonic.unlensed_cl.tt'
        for i, cl in enumerate(Y[name]):
            ax.plot(ell, cl / Yref[name][i])
        ax.set_xscale('log')
        plt.savefig('tmp2.png')
        plt.close(plt.gcf())
        return

    if 0:
        cosmo = DESI(engine='class', m_ncdm=0.)
        #params = {'h': (0.7, 0.9), 'logA': (3., 4.)} #, 'w0_fld': (-3., 1.), 'wa_fld': (-3., 2.)}
        params = {'h': (0.7, 0.7001), 'omega_cdm': (0.02, 0.02001), 'omega_b': (0.005, 0.005001), 'm_ncdm': (0.06, 0.06001), 'logA': (3., 3.0001), 'w0_fld': (-3., 1.), 'wa_fld': (-3., 2.)}
        calculator = get_calculator(cosmo, section=['fourier'])
        sampler = QMCSampler(calculator, params, engine='lhs', seed=42)
        samples = sampler.run(niterations=4)
        samples = samples[samples.isfinite()]
        X = {name[2:]: samples[name] for name in samples.columns('X.*')}
        Y = {name[2:]: samples[name] for name in samples.columns('Y.*')}
        Yref = dict(Y)
        operation = FourierNormOperation()
        operation.initialize(Y)
        Y = batch_vmap(operation, (0, 0))(Y, X)
        k = Y['fourier.k'][0]
        ax = plt.gca()
        for i, pk in enumerate(Y['fourier.pk.delta_cb.delta_cb']):
            ax.loglog(k, pk)
        plt.savefig('tmp.png')
        plt.close(plt.gcf())
        Y = batch_vmap(operation.inverse, (0, 0))(Y, X)
        ax = plt.gca()
        name = 'fourier.pk.delta_cb.delta_cb'
        for i, pk in enumerate(Y[name]):
            k = Y['fourier.k'][i]
            ax.plot(k, pk[..., 0] / Yref[name][i][..., 0])
        ax.set_xscale('log')
        plt.savefig('tmp2.png')
        plt.close(plt.gcf())
        return

    if 1:
        engine = Emulator.load(emulator_fn)
        engine.yoperations = engine.yoperations[:-1]
        pkname ='Y.fourier.pk.delta_cb.delta_cb'
        engine.engines = {name: engine for name, engine in engine.engines.items() if name in [pkname[2:]]}
        predict = batch_vmap(engine.predict)

        def load_samples(**kwargs):
            samples = Samples.concatenate([Samples.load(fn, **kwargs) for fn in glob.glob(str(samples_fn['mpk']) + '*')[:1]])
            mask = samples.isfinite()
            return samples[mask]

        samples = load_samples(include=['X.*', 'Y.fourier.k', 'Y.fourier.z', 'Y.fourier.z_non_linear', pkname], exclude=['X.tau_reio'])[:10]
        X = {name[2:]: samples[name] for name in samples.columns('X.*')}
        Y = {name[2:]: samples[name] for name in samples.columns('Y.*')}
        operation = FourierNormOperation()
        operation.initialize(Y)
        Y = batch_vmap(operation, (0, 0))(Y, X)
        emulated_Y = predict(X)
        samples = Samples({**{'X.' + name: value for name, value in X.items()}, **{'Y.' + name: value for name, value in Y.items()}})
        emulated_samples = Samples({**{'X.' + name: value for name, value in X.items()}, **{'Y.' + name: value for name, value in emulated_Y.items()}})
        '''
        operation = engine.engines[pkname[2:]].yoperations[-1]
        scaled = vmap(operation)(samples[pkname])
        emulated_scaled = vmap(operation)(emulated_samples[pkname])
        diff = scaled - emulated_scaled
        mse = np.mean(diff**2)
        print('mse', mse, mse**0.5)
        print(diff[..., :5], scaled[..., :5], diff[..., :5] / scaled[..., :5])
        '''
        samples[pkname] = samples[pkname][..., None]
        emulated_samples[pkname] = emulated_samples[pkname][..., None]
        plot_residual_fourier(samples, emulated_samples=emulated_samples, quantities=[pkname[len('Y.fourier.'):]], volume=None, fn='tmp3.png')

    def load_samples(**kwargs):
        samples = Samples.concatenate([Samples.load(fn, **kwargs) for fn in glob.glob(str(samples_fn['mpk']) + '*')[:1]])
        mask = samples.isfinite()
        return samples[mask]

    #samples = load_samples(include=['X.*', 'Y.thermodynamics.*'], exclude=['X.logA', 'X.n_s', 'X.tau_reio'])
    samples = load_samples(include=['X.*', 'Y.thermodynamics.*'])
    #params = [name[2:] for name in samples.columns('X.*')]

    def get_default_k_callable():
        k = np.concatenate([np.array([1e-6]),
                            np.logspace(-5, -4, num=20, endpoint=False),
                            np.logspace(-4, -3, num=40, endpoint=False),
                            np.logspace(-3, -2, num=60, endpoint=False),
                            np.logspace(-2, -1, num=80, endpoint=False),
                            np.logspace(-1, 0, num=100, endpoint=False),
                            np.logspace(0, 1, num=120, endpoint=True),
                            np.array([1e2])])
        return k

    k = get_default_k_callable()

    def test(X):
        cosmo = Cosmology(**X)
        #return cosmo.get_background(engine='bbks').comoving_radial_distance(1.)
        h = cosmo['h']
        return cosmo.get_primordial(engine='eisenstein_hu').pk_k(k=k * h) / h**3

    """
    def test(params):
        #cosmo = Cosmology(**params, engine=None)
        cosmo = DESI(**params, engine=None)
        background = DefaultBackground(cosmo)
        #background = BaseBackground(cosmo)
        return background.comoving_radial_distance(1.)
    """
    import time, jax
    test = jax.jit(test)
    size = 100
    X = {name[2:]: samples[name][:size] for name in samples.columns('X.*')}
    t0 = time.time()
    for i in range(size): test({name: X[name][i] for name in X})
    print(time.time() - t0)
    test = jax.jit(jax.vmap(test))
    X = {name[2:]: samples[name][:size] for name in samples.columns('X.*')}
    t0 = time.time()
    test(X)
    print(time.time() - t0)
    t0 = time.time()
    test(X)
    print(time.time() - t0)
    return
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


def test_pk():
    import numpy as np
    from cosmoprimo.fiducial import DESI
    from matplotlib import pyplot as plt
    cosmo = DESI()

    k = np.logspace(-2., 1., 1000)
    ax = plt.gca()

    list_params = [{'w0_fld': -1., 'wa_fld': 0., 'h': 0.6, 'Omega_k': -0.3}, {'w0_fld': -2., 'wa_fld': -1., 'h': 0.8, 'Omega_k': 0.3}]

    for ip, params in enumerate(list_params):
        cosmo = DESI(**params)
        pk = cosmo.get_fourier().pk_interpolator(of='delta_cb').to_1d(z=1.)
        s = cosmo['h']
        tmp = pk(k / s) / pk(0.1 / s)
        if ip == 0:
            ref = tmp
        else:
            ax.plot(k, tmp / ref)
            ax.set_xscale('log')
            #ax.set_yscale('log')

    plt.show()


def test_cl():
    import numpy as np
    from cosmoprimo.fiducial import DESI
    from matplotlib import pyplot as plt
    cosmo = DESI()

    k = np.logspace(-2., 1., 1000)
    ax = plt.gca()

    #list_params = [{'w0_fld': -1., 'wa_fld': 0., 'h': 0.6, 'Omega_k': -0.3}, {'w0_fld': -2., 'wa_fld': -1., 'h': 0.8, 'Omega_k': 0.3}]
    #list_params = [{'w0_fld': -3., 'wa_fld': -3.}, {'w0_fld': 1., 'wa_fld': 2.}]
    #list_params = [{'w0_fld': -0.78606431, 'wa_fld': 1.83985796}]
    list_params = [{'w0_fld': -1.37079802, 'wa_fld': -0.40845624}]
    ref = DESI()

    if 0:
        for ip, params in enumerate(list_params):
            cosmo = DESI(**params)
            cl = cosmo.get_harmonic().unlensed_cl()
            ell = cl['ell']
            factor = (ell + 1) * ell
            s = (ref.comoving_angular_distance(cosmo.z_star) / ref.rs_star) / (cosmo.comoving_angular_distance(cosmo.z_star) / cosmo.rs_star)
            ax.plot(ell * s, factor * cl['tt'])
            ax.set_xscale('log')
        plt.savefig('tmp3.png')
        plt.close(plt.gcf())
    if 1:
        from cosmoprimo.jax import Interpolator1D
        for ip, params in enumerate(list_params):
            cosmo = DESI(**params)
            cl = cosmo.get_harmonic().unlensed_cl()
            ell = cl['ell']
            factor = (ell + 1) * ell
            s = (ref.comoving_angular_distance(cosmo.z_star) / ref.rs_star) / (cosmo.comoving_angular_distance(cosmo.z_star) / cosmo.rs_star)
            ax.plot(ell, Interpolator1D(ell, Interpolator1D(ell, cl['tt'])(ell / s))(ell * s))
            ax.set_xscale('log')
        plt.savefig('tmp4.png')
        plt.close(plt.gcf())
    plt.show()


if __name__ == '__main__':

    """Uncomment to run."""

    todo = ['sample']
    #todo = ['fit', 'plot'][:1]
    #todo = ['plot_compression']
    #todo = ['fit', 'test'][1:]

    if todo:
        from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

        setup_logging()

        queue = Queue('classy_emulator')
        queue.clear(kill=False)

        environ = Environment('nersc-cosmodesi', command='module unload cosmoprimo')
        output, error = '_sbatch/slurm-%j.out', '_sbatch/slurm-%j.err'
        tm = TaskManager(queue=queue, environ=environ)

        if 'sample' in todo:

            for observable in ['bg', 'mpk', 'cmb'][1:2]:
                if observable == 'bg':
                    nsamples = 100000
                    nworkers = 10
                    tm_sample = tm.clone(scheduler=dict(max_workers=nworkers), provider=dict(provider='nersc', time='00:30:00', mpiprocs_per_worker=6, nodes_per_worker=0.1, output=output, error=error))
                if observable == 'mpk':
                    nsamples = 100000
                    nworkers = 10
                    tm_sample = tm.clone(scheduler=dict(max_workers=nworkers), provider=dict(provider='nersc', time='01:30:00', mpiprocs_per_worker=64, nodes_per_worker=1, output=output, error=error))
                if observable == 'cmb':
                    nsamples = 80000
                    nworkers = 80
                    tm_sample = tm.clone(scheduler=dict(max_workers=nworkers), provider=dict(provider='nersc', time='01:30:00', mpiprocs_per_worker=1, nodes_per_worker=1. / 16, output=output, error=error), environ=environ.clone(command='module unload cosmoprimo; export OMP_NUM_THREADS=16'))

                compute = tm_sample.python_app(sample)
                #compute = sample
                steps = list(range(0, nsamples + 1, nsamples // nworkers))
                for start, stop in zip(steps[:-1], steps[1:]):
                    compute(samples_fn[observable], observable=observable, start=start, stop=stop)
                    #break

        if 'fit' in todo:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
            for observable in ['bg', 'mpk', 'cmb'][1:2]:
                tofit = ['background', 'thermodynamics'][:1]
                if observable == 'mpk': tofit = ['fourier']
                if observable == 'cmb': tofit = ['harmonic']
                for tofit in tofit:
                    fit(samples_fn[observable], tofit=tofit)

        if 'plot' in todo:
            for observable in ['bg', 'mpk', 'cmb'][1:2]:
                toplot = ['background', 'thermodynamics'][:1]
                if observable == 'mpk': toplot = ['fourier']
                if observable == 'cmb': toplot = ['harmonic']
                for toplot in toplot:
                    plot(samples_fn[observable], toplot=toplot)

        if 'plot_compression' in todo:
            for observable in ['bg', 'mpk', 'cmb'][:1]:
                toplot = ['background', 'thermodynamics'][:1]
                if observable == 'mpk': toplot = ['fourier']
                if observable == 'cmb': toplot = ['harmonic']
                for toplot in toplot:
                    plot_compression(samples_fn[observable], toplot=toplot)

        if 'test' in todo:
            #test_pk()
            #test_cl()
            test()
