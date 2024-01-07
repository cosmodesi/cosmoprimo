import os

import numpy as np
from matplotlib import pyplot as plt

from cosmoprimo.emulators.tools import Emulator, EmulatedCalculator, TaylorEmulatorEngine, DiffSampler, setup_logging


def calculator(a=0, b=0):
    x = np.linspace(0., 1., 10)
    return {'x': x, 'y': a * x + b}


def plot(calculator, emulator, params):
    ax = plt.gca()
    values = np.array(np.meshgrid(*[np.linspace(*limits, 3) for limits in params.values()], indexing='ij')).T.reshape(-1, len(params))
    cmap = plt.get_cmap('jet', len(values))

    for ivalue, value in enumerate(values):
        value = dict(zip(params, value))
        ref = calculator(**value)
        emulated = emulator(**value)
        color = cmap(ivalue / len(values))
        ax.plot(ref['x'], ref['y'], linestyle='--', color=color)
        ax.plot(emulated['x'], emulated['y'], linestyle='-', color=color)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')


def test_taylor(show=True):
    emulator_dir = '_tests'
    fn = os.path.join(emulator_dir, 'emu.npy')
    params = {'a': (0., 1.), 'b': (0., 1.)}
    emulator = Emulator(calculator, params, engine=TaylorEmulatorEngine(order={'*': 4}))
    emulator.set_samples()
    emulator.fit()
    emulator.save(fn)
    emulator = emulator.to_calculator()
    emulator = EmulatedCalculator.load(fn)
    state = emulator(a=1)
    print(state)
    sampler = DiffSampler(calculator, params, order={'*': 4})
    samples = sampler.run()
    emulator = Emulator(samples=samples, engine=TaylorEmulatorEngine())
    emulator.fit()
    emulator.save(fn)
    emulator = emulator.to_calculator()

    if show:
        plot(calculator, emulator, params)
        plt.show()


if __name__ == '__main__':

    setup_logging()
    test_taylor()