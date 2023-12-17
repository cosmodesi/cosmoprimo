import os
import numpy as np

from cosmoprimo.emulators.tools import Emulator, EmulatedCalculator, setup_logging


def calculator(a=0, b=0):
    x = np.linspace(0., 1., 10)
    return {'x': x, 'y': a * x + b}


def test_base():
    emulator_dir = '_tests'
    fn = os.path.join(emulator_dir, 'emu.npy')
    params = {'a': (0., 1.), 'b': (0., 1.)}
    emulator = Emulator(calculator, params, engine='point')
    emulator.set_samples()
    emulator.fit()
    emulator.save(fn)
    emulator = emulator.to_calculator()
    emulator = EmulatedCalculator.load(fn)
    state = emulator(a=1)
    print(state)


if __name__ == '__main__':

    setup_logging()
    test_base()