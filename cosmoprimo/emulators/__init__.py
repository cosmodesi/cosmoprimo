"""Emulators for cosmoprimo.

    from cosmoprimo import Cosmology
    from cosmoprimo.emulators import emulate, Space

    cosmo = Cosmology(engine='camb', lensing=True, ellmax_cl=3000)
    emu = Emulator(cosmo, Space(samples=chain), section='harmonic')
    emu.train(budget=4, checkpoint='cl.npz', chunk='30min')      # or emulate(...) for both

    emu.predict(h=0.68, omega_cdm=0.12)                        # {'lensed_cl.tt': array, ...}
    fast = emu.to_cosmology()                                 # a Cosmology
    fast.clone(h=0.68).get_harmonic().lensed_cl()

The emulation machinery itself knows nothing about cosmology and lives in
:mod:`cosmoprimo.emulators.tools`; what a cosmology knows about itself -- that
:math:`C_\\ell \\propto A_s`, which spectra the optical depth screens -- is written as code in
:mod:`cosmoprimo.emulators.cosmology`.
"""

from .tools import Space, TrainingSet, Validation, validate, CoverageError, NotTrained
# NOTE `Emulator` here is the cosmology entry point, which dispatches on `section`. The template
# class to subclass is `cosmoprimo.emulators.tools.Emulator`.
from .cosmology import (Emulator, emulate, read, SectionEmulator, CosmologyEmulator,
                        HarmonicEmulator, BackgroundEmulator, FourierEmulator,
                        ThermodynamicsEmulator, emulated_engine, read_engine)

# `Cosmology(engine='ace')`: the packaged jaxace / jaxmapse / jaxcapse networks. Imported
# lazily by `get_engine`, so the heavy jax packages are not pulled in by `import cosmoprimo`.

# Physicality gates for the modified-gravity engines: which (alpha_B, alpha_M, ...) a Boltzmann
# code would accept, decided from the parameters alone, so a training sample or a prediction can
# be screened without running it. numpy-only, hence imported here rather than lazily.
from . import heftcamb, mochiclass
