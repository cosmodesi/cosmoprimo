import glob
from pathlib import Path

import numpy as np
from cosmoprimo.emulators import Operation, Emulator
from cosmoprimo.emulators.tools import utils
from cosmoprimo.cosmology import Cosmology


def convert_jaxcapse_to_cosmoprimo(fn, params=None, include_quantities=None):
    # For Cl
    import jaxcapse
    fn = Path(fn)

    def get_conversion():
        conversion = {}
        for name in ['tt', 'te', 'ee', 'ee', 'bb']:
            conversion['harmonic.lensed_cl.{}'.format(name)] = name.upper()
        for name in ['pp']:
            conversion['harmonic.lens_potential_cl.{}'.format(name)] = name.upper()
        return conversion

    def get_fn(fn, quantity):
        conversion = get_conversion()
        fn = fn / conversion.get(quantity, quantity)
        return str(fn)

    quantities = list(get_conversion())
    quantities = [quantity for quantity in quantities if glob.glob(get_fn(fn, quantity))]
    if include_quantities is not None:
        quantities = utils.find_names(quantities, include_quantities)

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
        emu = jaxcapse.load_emulator(get_fn(fn, quantity) + '/')
        model_operations = operations(emu.NN_params['params'], emu.activations, len(emu.features))
        xoperations = [Operation('(v - limits[0]) / (limits[1] - limits[0])', locals={'limits': np.asarray(emu.in_MinMax.T)})]
        limits = np.asarray(emu.out_MinMax.T)
        ells = np.arange(emu.out_MinMax.shape[0] + 2)
        # Conversion muK -> 1, ell * (ell + 1) / (2 pi) -> raw
        TCMB = 2.7255
        CMB_unit = TCMB * 1e6
        ells2 = (ells * (ells + 1))[2:]

        if 'lens_potential' in quantity:
            limits = limits / (ells2**2 / (2. * np.pi))
        else:
            limits = limits / (CMB_unit**2 * (ells2 / (2. * np.pi)))

        yoperations = []
        if 'm_ncdm_tot' in params:
            yoperations.append(Operation("v / jnp.exp(X['logA'] - 3.) / jnp.exp(-2 * X['tau_reio'])", inverse="v * jnp.exp(X['logA'] - 3.) * jnp.exp(-2 * X['tau_reio'])"))
        else:
            yoperations.append(Operation("v / jnp.exp(X['logA'] - 3.)", inverse="v * jnp.exp(X['logA'] - 3.)"))
        yoperations.append(Operation('((v - limits[0]) / (limits[1] - limits[0]))[:2]', inverse='jnp.insert(v * (limits[1] - limits[0]) + limits[0], 0, jnp.zeros(2))', locals={'limits': limits}))
        state['engines'][quantity] = {'name': 'mlp', 'params': params, 'xshape': (len(params),), 'yshape': (emu.out_MinMax.shape[0],), 'xoperations': [operation.__getstate__() for operation in xoperations], 'yoperations': [operation.__getstate__() for operation in yoperations], 'model_operations': [operation.__getstate__() for operation in model_operations], 'model_yoperations': []}
        if 'harmonic' in quantity:
            state['fixed']['.'.join(quantity.split('.')[:2]) + '.ell'] = ells
    for name in ['xoperations', 'yoperations']: state[name] = [operation.__getstate__() for operation in state[name]]
    emulator = Emulator.from_state(state)
    return emulator


def convert_cosmopower_to_cosmoprimo(fn, include_quantities=None):
    # https://colab.research.google.com/drive/1YB9rUzUSKx6LeugtDU0eWRlA0yLxpM1-?usp=sharing

    fn = Path(fn)
    version = '2' if 'jense' in str(fn) else '1'

    def operations(fpz):
        operations = []
        nlayers = fpz['n_layers']

        if "weights_" in fpz:
            # Assign the list of weight arrays from 'weights_' directly
            kernels = fpz["weights_"]
        else:
            # Use individual weight arrays if available
            kernels = [fpz[f"W_{i}"] for i in range(nlayers)]

        # Fallback to 'biases_' if individual 'b_i' are not found
        if "biases_" in fpz:
            biases = fpz["biases_"]
        else:
            biases = [fpz[f"b_{i}"] for i in range(nlayers)]

        alphas = fpz.get("alphas_", [fpz.get(f"alphas_{i}") for i in range(nlayers - 1)])
        betas = fpz.get("betas_", [fpz.get(f"betas_{i}") for i in range(nlayers - 1)])

        for ilayer in range(nlayers):
            # linear network operation
            operations.append(Operation('v @ kernel + bias', locals={'kernel': kernels[ilayer], 'bias': biases[ilayer]}))
            # non-linear activation function
            if ilayer < nlayers - 1:
                operations.append(Operation('(beta + (1 - beta) / (1 + jnp.exp(-alpha * v))) * v', locals={'alpha': alphas[ilayer], 'beta': betas[ilayer]}))
        return operations

    def get_conversion():
        if version == '2':
            conversion = {}
            for name in ['tt', 'te', 'ee', 'ee', 'bb']:
                conversion['harmonic.lensed_cl.{}'.format(name)] = 'Cl_{}'.format(name)
            for name in ['pp']:
                conversion['harmonic.lens_potential_cl.{}'.format(name)] = 'Cl_{}'.format(name)
            conversion['fourier.pk.delta_m.delta_m'] = 'Pk_lin'
            # ['thetastar', 'sigma8', 'YHe', 'zrei', 'taurend', 'zstar', 'rstar', 'zdrag', 'rdrag', 'N_eff']
            conversion['thermodynamics.all'] = 'derived'
        else:
            conversion = {}
            for name in ['tt', 'te', 'ee', 'ee', 'bb']:
                conversion['harmonic.lensed_cl.{}'.format(name)] = '{}_'.format(name.upper())
            for name in ['pp']:
                conversion['harmonic.lens_potential_cl.{}'.format(name)] = '{}_'.format(name.upper())
            #conversion['background.comoving_transverse_distance'] = 'DAZ'
            #conversion['background.efunc'] = 'HZ'
            conversion['fourier.pk.delta_m.delta_m'] = 'PKL_'
            conversion['thermodynamics.all'] = 'DER_'
        return conversion

    def get_fn(fn, quantity):
        conversion = get_conversion()
        if version == '2':
            fn = fn / 'networks' / '*{}*.npz'.format(conversion.get(quantity, quantity))
        else:
            if 'harmonic' in quantity:
                folder = 'TTTEEE'
                if 'pp' in quantity:
                    folder = 'PP'
            elif 'fourier' in quantity:
                folder = 'PK'
            elif 'background' in quantity:
                folder = 'growth-and-distances'
            else:
                folder = 'derived-parameters'
            fn = fn / folder / '*{}*.npz'.format(conversion.get(quantity, quantity))
        return str(fn)

    quantities = list(get_conversion())
    quantities = [quantity for quantity in quantities if glob.glob(get_fn(fn, quantity))]
    if include_quantities is not None:
        quantities = utils.find_names(quantities, include_quantities)

    state = {'engines': {}, 'xoperations': [], 'yoperations': [], 'defaults': {}, 'fixed': {}}

    if any('thermodynamics' in quantity for quantity in quantities):
        if version == '2':
            # thetastar, sigma8, YHe, zrei, taurend, zstar, rstar, zdrag, N_eff
            conversion = {'thermodynamics.{}'.format(name): value for name, value in zip(['z_star', 'rs_star', 'z_drag', 'rs_drag'], [5, 6, 7, 8])}
        else:
            # theta_s_100, sigma8, Y_p, z_reio, Neff, taurec, z_rec, rs_rec, ra_rec, tau_star, z_star, rs_star, ra_star, r_drag
            conversion = {'thermodynamics.{}'.format(name): value for name, value in zip(['z_star', 'rs_star', 'z_drag', 'rs_drag'], [10, 11, 12, 13])}
        state['yoperations'].append(Operation('', 'derived = v.pop("thermodynamics.all")\nfor name, index in conversion.items(): v[name] = derived[index]\nfor name in ["rs_drag", "rs_star"]: v["thermodynamics.{}".format(name)] *= X["h"]\nv', locals={'conversion': conversion}))
        #state['yoperations'].append(Operation('', 'derived = v.pop("thermodynamics.all")\nfor name, index in conversion.items(): v[name] = derived[index]\nv', locals={'conversion': conversion}))

    if any('fourier' in quantity for quantity in quantities):
        if version == '1':
            state['yoperations'].append(Operation('', inverse='v["fourier.k"] = v["fourier.k"] / X["h"]\n{name: value * X["h"]**3 if name.startswith("fourier.pk") else value for name, value in v.items()}'))
        else:
            state['yoperations'].append(Operation('', inverse='v["fourier.k"] = v["fourier.k"] / X["h"]\n{name: value if name.startswith("fourier.pk") else value for name, value in v.items()}'))
        #state['xoperations'].append(Operation('v["A_b"], v["eta_b"], v["logT_AGN"] = [3., 0.75, 7.8]', ''))
        state['defaults'] = {'A_b': 3., 'eta_b': 0.75, 'logT_AGN': 7.8}

    if version == '2':
        z_background = np.linspace(0., 20., 5000)
        k_fourier = np.geomspace(5.e-5, 50., 1000)
    else:
        z_background = np.linspace(0., 20., 5000)
        k_fourier = np.geomspace(1e-4, 50., 5000)[::10]

    def rename_param(param):
        conversion = {'m_ncdm': 'm_ncdm_tot', 'z_pk_save_nonclass': 'z'}
        toret = param
        for rename, aliases in Cosmology._alias_parameters.items():
            if param in (rename,) + aliases:
                toret = rename
                break
        return conversion.get(toret, toret)

    for quantity in quantities:
        ff = glob.glob(get_fn(fn, quantity))
        if len(ff) != 1:
            raise ValueError('could not find paths for quantity {}: {}'.format(quantity, ff))
        fpz = np.load(ff[0], allow_pickle=True)
        if version == '1': fpz = fpz["arr_0"].flatten()[0]
        # Parameters (X)
        params = [rename_param(param) for param in fpz["parameters"]]
        mean = fpz.get("parameters_mean", fpz.get("param_train_mean"))
        std = fpz.get("parameters_std", fpz.get("param_train_std"))
        limits = np.array([mean, mean + std])

        if 'H0' in params:
            idx = params.index('H0')
            params[idx] = 'h'
            limits[:, idx] /= 100.
        xoperations = [Operation('(v - limits[0]) / (limits[1] - limits[0])', locals={'limits': limits})]

        # Features (Y)
        mean = fpz.get("features_mean", fpz.get("feature_train_mean"))
        std = fpz.get("features_std", fpz.get("feature_train_std"))
        limits = np.array([mean, mean + std])

        yoperations, model_yoperations = [], []
        model_operations = operations(fpz)
        if 'pca_mean' in fpz:
            model_yoperations.append(Operation('(v @ matrix.T - mean) / std', inverse='(v * std + mean) @ matrix', locals={'mean': fpz["pca_mean"], 'std': fpz["pca_std"], 'matrix': fpz["pca_transform_matrix"]}))

        yoperations.insert(0, Operation('(v - limits[0]) / (limits[1] - limits[0])', inverse='v * (limits[1] - limits[0]) + limits[0]', locals={'limits': limits}))

        if 'harmonic' in quantity:
            if any(name in quantity for name in ['tt', 'ee', 'pp']):
                yoperations.insert(0, Operation('jnp.log10(v)', inverse='10**v'))
            ells = np.arange(limits[0].size + 2)
            ells2 = (ells * (ells + 1))[2:]
            if 'lens_potential' in quantity:
                factor = ells2**2 / (2. * np.pi)
            else:
                factor = ells2 / (2. * np.pi)
            yoperations.insert(0, Operation('v[:2] * factor', inverse='jnp.insert(v / factor, 0, jnp.zeros(2))', locals={'factor': factor}))
            state['fixed']['.'.join(quantity.split('.')[:2]) + '.ell'] = ells

        if 'thermodynamics' in quantity and version == '1':
            yoperations.insert(0, Operation('jnp.log10(v)', inverse='10**v'))
        if quantity == 'background.comoving_transverse_distance':
            state['fixed']['background.z'] = z_background
            yoperations.insert(0, Operation('v * (1. + z)', 'v / (1. + z)', locals={'z': state['fixed']['background.z']}))
        if quantity == 'background.efunc':
            state['fixed']['background.z'] = z_background
            yoperations.insert(0, Operation('v * (1e4 * X["h"]) / c', 'v * c / (1e4 * X["h"])', locals={'c': constants.c}))
        if 'fourier.pk' in quantity:
            yoperations.insert(0, Operation('jnp.log10(v)', inverse='10**v'))
            # Conversion Mpc => Mpc/h
            if version == '1':
                ells = np.arange(2, 5000 + 2)[::10]
                factor = (ells * (ells + 1)) / (2. * np.pi)
                yoperations.insert(0, Operation('v * factor', inverse='v / factor', locals={'factor': factor}))
            state['fixed']['fourier.k'] = k_fourier

        state['engines'][quantity] = {'name': 'mlp', 'params': params, 'xshape': (len(params),), 'yshape': (limits[0].size,),
                                'xoperations': [operation.__getstate__() for operation in xoperations], 'yoperations': [operation.__getstate__() for operation in yoperations],
                                'model_operations': [operation.__getstate__() for operation in model_operations], 'model_yoperations': [operation.__getstate__() for operation in model_yoperations]}
    for name in ['xoperations', 'yoperations']: state[name] = [operation.__getstate__() for operation in state[name]]
    emulator = Emulator.from_state(state)
    return emulator


if __name__ == '__main__':

    package_dir = Path(__file__).parent.parent.parent.parent
    train_dir = package_dir / 'cosmoprimo-emulators'
    #train_dir = Path(__file__).parent / 'train'

    convert, test = [], []
    convert = ['jaxcapse_base_mnu_w_wa']
    #convert = ['jaxcapse']
    #convert = ['cosmopower_bolliet2023_base', 'cosmopower_bolliet2023_base_mnu', 'cosmopower_bolliet2023_base_w',
    #           'cosmopower_jense2024_base', 'cosmopower_jense2024_base_mnu', 'cosmopower_jense2024_base_w_wa']
    test = convert

    def get_source_jaxcapse(name, return_params=False):
        if 'jaxcapse' in name:
            base_dir = Path(jaxcapse.__file__).parent.parent
            if 'mnu_w_wa' in name:
                toret = base_dir / 'batch_trained_desi', ['logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'tau_reio', 'm_ncdm_tot', 'w0_fld', 'wa_fld']
            else:
                toret = base_dir / 'trained_emu', ['logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'tau_reio']
        if not return_params:
            del toret[1]
        if len(toret) == 1:
            return toret[0]
        return tuple(toret)

    def get_source_cosmopower(name):
        conversion = {'cosmopower_bolliet2023_base': 'lcdm',
                      'cosmopower_bolliet2023_base_mnu': 'mnu',
                      'cosmopower_bolliet2023_base_w': 'wcdm',
                      'cosmopower_jense2024_base': 'jense_2024_emulators/jense_2023_camb_lcdm',
                      'cosmopower_jense2024_base_mnu': 'jense_2024_emulators/jense_2023_camb_mnu',
                      'cosmopower_jense2024_base_w_wa': 'jense_2024_emulators/jense_2023_camb_wcdm'}
        return package_dir / 'cosmopower-organization' / conversion.get(name, name)

    for name in convert:
        if 'capse' in name:
            import jaxcapse
            source_fn, params = get_source_jaxcapse(name, return_params=True)
            emulator = convert_jaxcapse_to_cosmoprimo(source_fn, params=params)
            emulator_fn = train_dir / name / 'emulator.npy'
            emulator.save(emulator_fn)
        if 'cosmopower' in name:
            #from cosmopower import YAMLParser
            #parser = YAMLParser('../../../cosmopower-organization/jense_2024_emulators/jense_2023_cmb_camb_mnu.yaml')
            #print(parser.computed_parameters)
            source_fn = get_source_cosmopower(name)
            for section in ['thermodynamics', 'harmonic', 'fourier']:
                emulator = convert_cosmopower_to_cosmoprimo(source_fn, include_quantities=[section + '.*'])
                emulator_fn = train_dir / name / 'emulator_{}.npy'.format(section)
                emulator.save(emulator_fn)

    if test:
        from matplotlib import pyplot as plt
        import jax
        from cosmoprimo import constants
        from cosmoprimo.fiducial import DESI
        from cosmoprimo.emulators import EmulatedEngine

        for name in test:

            if 'capse' in name:
                emulator_fn = train_dir / name / 'emulator.npy'

                import jaxcapse
                source_fn, params = get_source_jaxcapse(name, return_params=True)

                def to_dict(array):
                    return {name: array[name] for name in array.dtype.names}

                kw = dict(logA=3., tau_reio=0.06, m_ncdm=0.2)
                cosmo = DESI(**kw, kmax_pk=10., engine=EmulatedEngine.load(emulator_fn))
                test = to_dict(cosmo.get_harmonic().lensed_cl())
                test.update(to_dict(cosmo.get_harmonic().lens_potential_cl()))

                cosmo = DESI(**kw, lensing=True, kmax_pk=10., engine='camb', ellmax_cl=4000, non_linear='mead',
                            #extra_params=dict(AccuracyBoost=2, lSampleBoost=2, lAccuracyBoost=2, DoLateRadTruncation=False), non_linear='mead2016')
                            extra_params=dict(lens_margin=1250, lens_potential_accuracy=4, AccuracyBoost=1, lSampleBoost=1, lAccuracyBoost=1, DoLateRadTruncation=False))
                ref_camb = to_dict(cosmo.get_harmonic().lensed_cl())
                ref_camb.update(to_dict(cosmo.get_harmonic().lens_potential_cl()))

                cosmo = DESI(**kw, lensing=True, kmax_pk=10., engine='class', ellmax_cl=4000, non_linear='hmcode',
                            extra_params=dict(halofit_k_per_decade=3000., l_switch_limber=40., accurate_lensing=1, num_mu_minus_lmax=1000., delta_l_max=1000.))
                ref_class = to_dict(cosmo.get_harmonic().lensed_cl())
                ref_class.update(to_dict(cosmo.get_harmonic().lens_potential_cl()))

                for name in ['TT', 'TE', 'EE', 'PP', 'BB']:
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

            if 'cosmopower' in name:
                ellmax = 2000
                kw = dict()
                cosmo_emu = DESI(**kw, kmax_pk=10., ellmax_cl=ellmax,
                                 engine=EmulatedEngine.load({train_dir / name / 'emulator_{}.npy'.format(section): None for section in ['thermodynamics', 'harmonic', 'fourier']}))
                cosmo_camb = DESI(**kw, lensing=True, kmax_pk=10., engine='camb', ellmax_cl=ellmax, non_linear='mead',
                            #extra_params=dict(AccuracyBoost=2, lSampleBoost=2, lAccuracyBoost=2, DoLateRadTruncation=False), non_linear='mead2016')
                            extra_params=dict(lens_margin=1250, lens_potential_accuracy=4, AccuracyBoost=1, lSampleBoost=1, lAccuracyBoost=1, DoLateRadTruncation=False))
                cosmo_class = DESI(**kw, lensing=True, kmax_pk=10., engine='class', ellmax_cl=ellmax, non_linear='hmcode',
                            extra_params=dict(halofit_k_per_decade=3000., l_switch_limber=40., accurate_lensing=1, num_mu_minus_lmax=1000., delta_l_max=1000.))
                #print(cosmo.rs_star, cosmo.rs_drag)
                emu = cosmo_emu.get_fourier().pk_interpolator()
                camb = cosmo_camb.get_fourier().pk_interpolator()
                classy = cosmo_class.get_fourier().pk_interpolator()

                z = 1.
                k = np.geomspace(1e-3, 1, 1000)
                ax = plt.gca()
                #ax.plot(k, emu(k, z) / camb(k, z), label='cosmoprimo - cosmopower')
                ax.plot(k, emu(k, z), label='cosmoprimo - cosmopower')
                ax.plot(k, camb(k, z), label='camb')
                ax.plot(k, classy(k, z), label='class')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.legend()
                plt.show()

                def to_dict(array):
                    return {name: array[name] for name in array.dtype.names}

                emu = to_dict(cosmo_emu.get_harmonic().lensed_cl())
                emu.update(to_dict(cosmo_emu.get_harmonic().lens_potential_cl()))

                camb = to_dict(cosmo_camb.get_harmonic().lensed_cl())
                camb.update(to_dict(cosmo_camb.get_harmonic().lens_potential_cl()))

                classy = to_dict(cosmo_class.get_harmonic().lensed_cl())
                classy.update(to_dict(cosmo_class.get_harmonic().lens_potential_cl()))

                for name in ['TT', 'TE', 'EE']:
                    name = name.lower()
                    ell = np.arange(ellmax)
                    if name == 'pp':
                        factor = (ell * (ell + 1))**2 / (2. * np.pi)
                    else:
                        factor = ell * (ell + 1) / (2. * np.pi) * (1e6 * constants.TCMB)**2
                    ax = plt.gca()
                    ax.set_title(name)
                    #ax.plot(ell[:ellmax], test[name][:ellmax] / ref_camb[name][:ellmax], label='cosmoprimo - cosmopower')
                    ax.plot(ell[:ellmax], factor[:ellmax] * emu[name][:ellmax], label='cosmoprimo - cosmopower')
                    ax.plot(ell[:ellmax], factor[:ellmax] * camb[name][:ellmax], label='camb')
                    ax.plot(ell[:ellmax], factor[:ellmax] * classy[name][:ellmax], label='class')
                    if name == 'pp':
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                    ax.legend()
                    plt.show()