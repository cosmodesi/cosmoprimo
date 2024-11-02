from pathlib import Path

import numpy as np
from cosmoprimo.emulators import Operation, Emulator


def convert_jaxcapse_to_cosmoprimo(fn, params=None, quantities=None):
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
    #convert = ['jaxcapse_mnu_w0wa']
    test = ['jaxcapse_mnu_w0wa']
    #convert = ['jaxcapse']
    #test = ['jaxcapse']

    def get_source(name, return_params=False, return_quantities=False):
        if 'jaxcapse' in name:
            base_dir = Path(jaxcapse.__file__).parent.parent
            if 'mnu_w0wa' in name:
                #toret = base_dir / 'new_batch_trained_desi', ['logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'tau_reio', 'm_ncdm_tot', 'w0_fld', 'wa_fld'], ['TT', 'TE', 'EE', 'PP']
                toret = base_dir / 'batch_trained_desi', ['logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'tau_reio', 'm_ncdm_tot', 'w0_fld', 'wa_fld'], ['TT', 'TE', 'EE', 'PP', 'BB']
            else:
                toret = base_dir / 'trained_emu', ['logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'tau_reio'], ['TT', 'TE', 'EE', 'PP']
        toret = list(toret)
        if not return_quantities:
            del toret[-1]
        if not return_params:
            del toret[1]
        if len(toret) == 1:
            return toret[0]
        return tuple(toret)

    for name in convert:
        if 'jaxcapse' in name:
            import jaxcapse
            source_fn, params, quantities = get_source(name, return_params=True, return_quantities=True)
            emulator = convert_jaxcapse_to_cosmoprimo(source_fn, params=params, quantities=quantities)
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
            source_fn, params, quantities = get_source(name, return_params=True, return_quantities=True)

            def to_dict(array):
                return {name: array[name] for name in array.dtype.names}

            cosmo = DESI(m_ncdm=0.06, kmax_pk=10., engine=EmulatedEngine.load(emulator_fn))
            test = to_dict(cosmo.get_harmonic().lensed_cl())
            test.update(to_dict(cosmo.get_harmonic().lens_potential_cl()))

            cosmo = DESI(lensing=True, m_ncdm=0.06, kmax_pk=10., engine='camb', ellmax_cl=4000, non_linear='mead',
                         #extra_params=dict(AccuracyBoost=2, lSampleBoost=2, lAccuracyBoost=2, DoLateRadTruncation=False), non_linear='mead2016')
                         extra_params=dict(lens_margin=1250, lens_potential_accuracy=4, AccuracyBoost=1, lSampleBoost=1, lAccuracyBoost=1, DoLateRadTruncation=False))
            ref_camb = to_dict(cosmo.get_harmonic().lensed_cl())
            ref_camb.update(to_dict(cosmo.get_harmonic().lens_potential_cl()))

            cosmo = DESI(lensing=True, m_ncdm=0.06, kmax_pk=10., engine='class', ellmax_cl=4000, non_linear='hmcode',
                         extra_params=dict(halofit_k_per_decade=3000., l_switch_limber=40., accurate_lensing=1, num_mu_minus_lmax=1000., delta_l_max=1000.))
            ref_class = to_dict(cosmo.get_harmonic().lensed_cl())
            ref_class.update(to_dict(cosmo.get_harmonic().lens_potential_cl()))

            for name in quantities[-1:]:
                name = name.lower()
                tt = jaxcapse.load_emulator(str(source_fn / name.upper()) + '/')
                test2 = np.insert(tt.get_Cl(np.array([cosmo[name] for name in params])), 0, [0.] * 2)

                ellmax = min(len(test[name]), len(ref_camb[name]))
                ell = np.arange(ellmax)
                if name == 'pp':
                    factor = (ell * (ell + 1))**2 / (2. * np.pi)
                else:
                    factor = ell * (ell + 1) / (2. * np.pi) * (1e6 * constants.TCMB)**2
                ax = plt.gca()
                ax.set_title(name)
                ax.plot(ell[:ellmax], factor[:ellmax] * test[name][:ellmax], label='cosmoprimo - jaxcapse')
                ax.plot(ell[:ellmax], test2[:ellmax], label='jaxcapse')
                ax.plot(ell[:ellmax], factor[:ellmax] * ref_camb[name][:ellmax], label='camb')
                ax.plot(ell[:ellmax], factor[:ellmax] * ref_class[name][:ellmax], label='class')
                if name == 'pp':
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                ax.legend()
                plt.show()