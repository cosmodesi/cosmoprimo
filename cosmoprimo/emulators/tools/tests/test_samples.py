import os
import numpy as np

from cosmoprimo.emulators.tools import Samples, InputSampler, GridSampler, DiffSampler, QMCSampler, setup_logging


def test_samples():
    size = 10
    samples = Samples({'a': np.linspace(0., 1., size), 'b': np.linspace(0., 1., size)})
    print(list(samples))
    assert samples.size == size
    assert samples[:3].size == 3
    for fn in ['_tests/samples.npy', '_tests/samples.npz']:
        samples.save(fn)
        samples2 = Samples.load(fn)
        assert samples2 == samples


def test_samplers():

    def calculator(a=0, b=0):
        x = np.linspace(0., 1., 10)
        return {'x': x, 'y': a * x + b}

    sampler = QMCSampler(calculator, params={'a': (0.8, 1.2), 'b': (0.8, 1.2)})
    sampler.run(niterations=10)
    sampler.samples

    def calculator(a=0, b=0):
        x = np.linspace(0., 1., 10)
        return {'x': x, 'y': a * x + b, 'z': a * x**2 + b}

    sampler = InputSampler(calculator, samples=sampler.samples)
    sampler.run()
    columns = sampler.mpicomm.bcast(sampler.samples.columns() if sampler.mpicomm.rank == 0 else None, root=0)
    assert set(columns) == set(['X.a', 'X.b', 'Y.x', 'Y.y', 'Y.z'])

    sampler = GridSampler(calculator, params={'a': (0.8, 1.2), 'b': (0.8, 1.2)})
    sampler.run()
    sampler.samples

    sampler = DiffSampler(calculator, params={'a': (0.8, 1.2), 'b': (0.8, 1.2)})
    sampler.run()
    sampler.samples


if __name__ == '__main__':

    setup_logging()
    test_samples()
    test_samplers()