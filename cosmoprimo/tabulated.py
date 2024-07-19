import numpy as np

from .cosmology import BaseEngine, BaseSection, CosmologyError


class TabulatedEngine(BaseEngine):

    """Engine using tabulated values from an ASCII file."""
    name = 'tabulated'

    def __init__(self, *args, **kwargs):
        super(TabulatedEngine, self).__init__(*args, **kwargs)
        self._names = self._extra_params.get('names', ['efunc', 'comoving_radial_distance'])
        arrays = np.loadtxt(self._extra_params['filename'], comments='#', usecols=range(len(self._names) + 1), unpack=True)
        self.z = arrays[0]
        for name, array in zip(self._names, arrays[1:]):
            setattr(self, name, array)


class Background(BaseSection):

    """Tabulated background quantities."""

    def __init__(self, engine):
        super().__init__(engine)
        self.ba = self._engine


def make_func(name):

    def func(self, z):
        z = self._np.asarray(z)
        mask = (z < self.ba.z[0]) | (z > self.ba.z[-1])
        if mask.any(): raise CosmologyError('Input z outside of tabulated range.')
        return self._np.interp(z, self.ba.z, getattr(self.ba, name), left=None, right=None)

    return func


for name in ['efunc', 'comoving_radial_distance']:
    setattr(Background, name, make_func(name))
