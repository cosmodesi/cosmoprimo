from pathlib import Path

from .emulated import EmulatedEngine, Thermodynamics, Harmonic

train_dir = Path(__file__).parent / 'train'


from cosmoprimo.cosmology import DefaultBackground

    
class Background(DefaultBackground):
    
    """Background quantities."""


class CAPSEEngine(EmulatedEngine):

    name = 'capse'
    path = (train_dir / 'classy/emulator.npy', train_dir / 'jaxcapse_mnu_w0wa/emulator.npy')
