import os
import numpy as np

from cosmoprimo.emulators.tools import Emulator, EmulatedCalculator, Operation, ScaleOperation, NormOperation, Log10Operation, ArcsinhOperation, PCAOperation, ChebyshevOperation, setup_logging


def calculator(a=0, b=0):
    x = np.linspace(0., 1., 10)
    return {'x': x, 'y': a * x + b}


def test_base():
    emulator_dir = '_tests'
    fn = os.path.join(emulator_dir, 'emu.npy')
    params = {'a': (0., 1.), 'b': (0., 1.)}
    emulator = Emulator(calculator=calculator, params=params, engine='point')
    emulator.set_samples()
    emulator.fit()
    emulator.save(fn)
    emulator.mpicomm.Barrier()
    emulator = emulator.to_calculator()
    emulator = EmulatedCalculator.load(fn)
    state = emulator(a=1)
    print(state)


def test_operation():
    fn = '_tests/operation.npy'
    operation = Operation('v + 2', inverse='v - 2')
    assert operation.inverse(operation(42.)) == 42.
    rng = np.random.RandomState(seed=42)
    shape = (5, 7, 9)
    x = rng.uniform(0., 1., size=(10,) + shape)

    operation = Operation("v['a'] += 2; v")
    assert operation({'a': 2}) == {'a': 4}
    try:
        operation({'b': 2})
    except KeyError:
        pass

    operation = ScaleOperation()
    operation.initialize(x)
    operation.save(fn)
    operation = Operation.load(fn)
    operation(np.ones((2,) + shape))

    operation = NormOperation()
    operation.initialize(x)
    operation.save(fn)
    operation = Operation.load(fn)
    operation(np.ones((2,) + shape))

    operation = Log10Operation()
    operation.save(fn)
    operation = Operation.load(fn)
    assert np.allclose(operation.inverse(operation(np.ones((2,) + shape))), 1.)

    operation = ArcsinhOperation()
    operation.save(fn)
    operation = Operation.load(fn)
    assert np.allclose(operation.inverse(operation(np.ones((2,) + shape))), 1.)

    operation = PCAOperation(npcs=1)
    operation.initialize(x)
    operation.save(fn)
    operation = Operation.load(fn)
    operation.inverse(operation(np.ones(shape)))

    operation = ChebyshevOperation(order=3)
    operation.initialize(x)
    operation.save(fn)
    operation = Operation.load(fn)
    print(operation.inverse(operation(np.ones(shape))))

    operation = ChebyshevOperation(order=3, axis=2)
    operation.initialize(x)
    operation.save(fn)
    operation = Operation.load(fn)
    operation(np.ones(shape))
    print(operation.inverse(operation(np.ones(shape))))


def test_chebyshev():

    x = np.pi + np.linspace(0., 5 * np.pi, 1000)
    a, b = 1., 1.
    y = a * x * np.sin(b * x)
    from matplotlib import pyplot as plt
    ax = plt.gca()
    ax.plot(x, y, label='original')
    for order in [10, 20, 50, 60, 100]:
        operation = ChebyshevOperation(order=order)
        operation.initialize(y[None, :])
        yc = operation.inverse(operation(y))
        ax.plot(x, yc, label='chebyshev order = {:d}'.format(order))
    ax.legend()
    plt.show()



if __name__ == '__main__':

    setup_logging()
    test_base()
    test_operation()
    test_chebyshev()