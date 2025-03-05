import os

from pathlib import Path

from cosmoprimo.emulators import emulated
from cosmoprimo.emulators.emulated import EmulatedEngine, Thermodynamics, Harmonic


def get_train_dir():
    dirname = os.getenv('COSMOPRIMO_EMULATOR_DIR', '')
    if not dirname:
        dirname = Path(__file__).parent
    return dirname / 'train'


from cosmoprimo.cosmology import DefaultBackground


class Background(DefaultBackground):

    """Background quantities."""


class Fourier(emulated.Fourier):

    def __getstate__(self):
        """Return this class' state dictionary."""
        state = emulated.Fourier.__getstate__(self)
        state['sigma8_m'] = self.sigma8_m
        return state

    def __setstate__(self, state):
        """Set this class' state dictionary."""
        self.sigma8_m_custom = self._sigma8_m = state.pop('sigma8_m')
        super().__setstate__(state)


class CustomEngine(EmulatedEngine):

    name = 'custom'
