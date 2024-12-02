import os

from pathlib import Path

from .emulated import EmulatedEngine, Thermodynamics, Harmonic


train_dir = Path(__file__).parent / 'train'


from cosmoprimo.cosmology import DefaultBackground


class Background(DefaultBackground):

    """Background quantities."""


class CAPSEEngine(EmulatedEngine):

    name = 'capse'
    path = {train_dir / 'cosmopower_bolliet2023_base_mnu/emulator_thermodynamics.npy': 'https://github.com/adematti/cosmoprimo-emulators/raw/refs/heads/main/cosmopower_bolliet2023_base_mnu/emulator_thermodynamics.npy',
            train_dir / 'jaxcapse_base_mnu_w_wa/emulator.npy': 'https://github.com/adematti/cosmoprimo-emulators/raw/refs/heads/main/jaxcapse_base_mnu_w_wa/emulator.npy'}


class CosmopowerBolliet2023Engine(EmulatedEngine):

    name = 'cosmopower_bolliet2023'
    path = {train_dir / 'cosmopower_bolliet2023_base_mnu/emulator_{}.npy'.format(section): 'https://github.com/adematti/cosmoprimo-emulators/raw/refs/heads/main/cosmopower_bolliet2023_base_mnu/emulator_{}.npy'.format(section) for section in ['thermodynamics', 'harmonic', 'fourier']}