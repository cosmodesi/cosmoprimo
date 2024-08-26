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
        yoperations = [Operation('(v - limits[0]) / (limits[1] - limits[0])', inverse='v * (limits[1] - limits[0]) + limits[0]', locals={'limits': np.asarray(emu.out_MinMax.T)})]
        state['engines']['harmonic.lensed_cl.{}'.format(quantity.lower())] = {'name': 'mlp', 'params': params, 'xshape': (len(params),), 'yshape': (emu.out_MinMax.shape[0],), 'xoperations': [operation.__getstate__() for operation in xoperations], 'yoperations': [operation.__getstate__() for operation in yoperations], 'model_operations': [operation.__getstate__() for operation in model_operations], 'model_yoperations': []}
        state['fixed']['harmonic.lensed_cl.ell'] = np.arange(emu.out_MinMax.shape[0])
    emulator = Emulator.from_state(state)
    return emulator


if __name__ == '__main__':

    train_dir = Path(__file__).parent / 'train'

    convert = ['jaxcapse']
    test = True

    for name in convert:
        if name == 'jaxcapse':
            import jaxcapse
            source_fn = Path(jaxcapse.__file__).parent.parent / 'trained_emu'
            emulator = convert_jaxcapse_to_cosmoprimo(source_fn)
            emulator_fn = train_dir / name / 'emulator.npy'
            emulator.save(emulator_fn)
            if test:
                import jax
                from cosmoprimo.fiducial import DESI
                from cosmoprimo.emulators import EmulatedEngine

                cosmo = DESI(engine=EmulatedEngine.load(emulator_fn))
                cosmo.clone(h=0.6).get_harmonic().lensed_cl().shape
                
                def cl(omega_cdm):
                    return cosmo.clone(omega_cdm=omega_cdm).get_harmonic().lensed_cl()['tt']

                cl = jax.jit(jax.jacfwd(cl))
                cl(0.1)