from pathlib import Path

import numpy as np
from cosmoprimo.emulators import Operation, Emulator


def convert_jaxcapse_to_cosmoprimo(fn, quantities=None, params=None):
    # For Cl
    import jaxcapse

    if quantities is None:
        quantities = ['TT', 'TE', 'EE', 'PP']

    if params is None:
        params = ['logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'tau_reio']

    def operations(params, activations, nlayers):
        operations = []
        for ilayer in range(nlayers):
            # linear network operation
            player = params['Dense_{:d}'.format(ilayer)]
            operations.append(Operation('v @ kernel + bias', locals={name: np.asarray(player[name]) for name in ['kernel', 'bias']}))
            # non-linear activation function
            if ilayer < nlayers - 1:
                activation = activations[ilayer]
                if activation == 'silu':
                    operations.append(Operation('v / (1 + jnp.exp(-v))', locals={}))
                elif activation == 'relu':
                    operations.append(Operation('jnp.maximum(v, 0.)', locals={}))
                elif activation == 'tanh':
                    operations.append(Operation('jnp.tanh(v)', locals={}))
        return operations

    state = {'engines': {}, 'xoperations': [], 'yoperations': [], 'defaults': {}, 'fixed': {}}
    for quantity in quantities:
        emu = jaxcapse.load_emulator(str(Path(fn) / quantity) + '/')
        model_operations = operations(emu.NN_params['params'], emu.activations, len(emu.features))
        xoperations = [Operation('(v - limits[0]) / (limits[1] - limits[0])', locals={'limits': np.asarray(emu.in_MinMax.T)})]
        limits = np.asarray(emu.out_MinMax.T)
        ells = np.arange(emu.out_MinMax.shape[0] + 2)
        # Conversion muK -> 1, ell * (ell + 1) / (2 pi) -> raw
        TCMB = 2.7255
        CMB_unit = TCMB * 1e6
        ells2 = (ells * (ells + 1))[2:]
        lens_potential = quantity.lower() == 'pp'

        if lens_potential:
            limits = limits / (ells2**2 / (2. * np.pi))
        else:
            limits = limits / (CMB_unit**2 * (ells2 / (2. * np.pi)))

        yoperations = []
        if 'm_ncdm_tot' in params:
            yoperations.append(Operation("v / jnp.exp(X['logA'] - 3.) / jnp.exp(-2 * X['tau_reio'])", inverse="v * jnp.exp(X['logA'] - 3.) * jnp.exp(-2 * X['tau_reio'])"))
        else:
            yoperations.append(Operation("v / jnp.exp(X['logA'] - 3.)", inverse="v * jnp.exp(X['logA'] - 3.)"))
        yoperations.append(Operation('((v - limits[0]) / (limits[1] - limits[0]))[:2]', inverse='jnp.insert(v * (limits[1] - limits[0]) + limits[0], 0, jnp.zeros(2))', locals={'limits': limits}))
        name = quantity.lower()
        basename = 'lens_potential_cl' if lens_potential else 'lensed_cl'
        state['engines']['harmonic.{}.{}'.format(basename, name)] = {'name': 'mlp', 'params': params, 'xshape': (len(params),), 'yshape': (emu.out_MinMax.shape[0],), 'xoperations': [operation.__getstate__() for operation in xoperations], 'yoperations': [operation.__getstate__() for operation in yoperations], 'model_operations': [operation.__getstate__() for operation in model_operations], 'model_yoperations': []}
        state['fixed']['harmonic.{}.ell'.format(basename)] = ells
    emulator = Emulator.from_state(state)
    return emulator


if __name__ == '__main__':

    train_dir = Path(__file__).parent / 'train'

    convert, test = [], []
    convert = ['jaxcapse_mnu_w0wa']
    test = ['jaxcapse_mnu_w0wa']
    #convert = ['jaxcapse']
    #test = ['jaxcapse']

    def get_source(name, return_params=False):
        if 'jaxcapse' in name:
            base_dir = Path(jaxcapse.__file__).parent.parent
            if 'mnu_w0wa' in name:
                toret = base_dir / 'new_batch_trained_desi', ['logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'tau_reio', 'm_ncdm_tot', 'w0_fld', 'wa_fld']
            else:
                toret = base_dir / 'trained_emu', ['logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'tau_reio']
        if return_params:
            return toret
        return toret[0]

    for name in convert:
        if 'jaxcapse' in name:
            import jaxcapse
            source_fn, params = get_source(name, return_params=True)
            emulator = convert_jaxcapse_to_cosmoprimo(source_fn, params=params)
            emulator_fn = train_dir / name / 'emulator.npy'
            emulator.save(emulator_fn)

    if test:
        from matplotlib import pyplot as plt
        import jax
        from cosmoprimo import constants
        from cosmoprimo.fiducial import DESI
        from cosmoprimo.emulators import EmulatedEngine

        for name in test:
            emulator_fn = train_dir / name / 'emulator.npy'
            """
            cosmo = DESI(engine=EmulatedEngine.load(emulator_fn))
            cosmo.clone(h=0.6).get_harmonic().lensed_cl().shape

            def cl(omega_cdm):
                return cosmo.clone(omega_cdm=omega_cdm).get_harmonic().lensed_cl()['tt']

            #cl = jax.jit(jax.jacfwd(cl))
            #cl(0.1)
            """
            import jaxcapse
            source_fn, params = get_source(name, return_params=True)

            def to_dict(array):
                return {name: array[name] for name in array.dtype.names}

            cosmo = DESI(m_ncdm=0.06, kmax_pk=10., engine=EmulatedEngine.load(emulator_fn))
            test = to_dict(cosmo.get_harmonic().lensed_cl())
            test.update(to_dict(cosmo.get_harmonic().lens_potential_cl()))

            cosmo = DESI(lensing=True, m_ncdm=0.06, kmax_pk=10., engine='camb',
                         extra_params=dict(AccuracyBoost=2, lSampleBoost=2, lAccuracyBoost=2, DoLateRadTruncation=False), non_linear='hmcode')
            ref = to_dict(cosmo.get_harmonic().lensed_cl())
            ref.update(to_dict(cosmo.get_harmonic().lens_potential_cl()))

            for name in ['TT', 'TE', 'EE', 'PP']:
                name = name.lower()
                tt = jaxcapse.load_emulator(str(source_fn / name.upper()) + '/')
                test2 = np.insert(tt.get_Cl(np.array([cosmo[name] for name in params])), 0, [0.] * 2)

                ell = ref['ell']
                if name == 'pp':
                    factor = (ell * (ell + 1))**2 / (2. * np.pi)
                else:
                    factor = ell * (ell + 1) / (2. * np.pi) * (1e6 * constants.TCMB)**2
                ax = plt.gca()
                ax.set_title(name)
                ax.plot(ell, factor * test[name], label='cosmoprimo - jaxcapse')
                ax.plot(ell, test2[:ell[-1] + 1], label='jaxcapse')
                ax.plot(ell, factor * ref[name], label='camb')
                if name == 'pp':
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                ax.legend()
                plt.show()