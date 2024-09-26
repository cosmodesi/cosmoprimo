"""Cosmological calculation with the Boltzmann code MochiCLASS."""

from pyclass import mochiclass

from .cosmology import BaseEngine, CosmologyInputError, CosmologyComputationError
from . import classy


class MochiClassEngine(classy.ClassEngine):

    """Engine for the Boltzmann code mochiclass."""

    name = 'mochiclass'

    _default_cosmological_parameters = dict()

    def _set_classy(self, params):

        class _ClassEngine(mochiclass.ClassEngine):

            def compute(self, tasks):
                try:
                    return super(_ClassEngine, self).compute(tasks)
                except mochiclass.ClassInputError as exc:
                    raise CosmologyInputError from exc
                except mochiclass.ClassComputationError as exc:
                    raise CosmologyComputationError from exc

        self.classy = _ClassEngine(params=params)


class Background(classy.BaseClassBackground, mochiclass.Background):

    """Your modifications, if any."""


class Primordial(classy.BaseClassPrimordial, mochiclass.Primordial):

     """Your modifications, if any."""


class Perturbations(classy.BaseClassPerturbations, mochiclass.Perturbations):

     """Your modifications, if any."""


class Transfer(classy.BaseClassTransfer, mochiclass.Transfer):

     """Your modifications, if any."""


class Harmonic(classy.BaseClassHarmonic, mochiclass.Harmonic):
     """Your modifications, if any."""


class Fourier(classy.BaseClassFourier, mochiclass.Fourier):
     """Your modifications, if any."""
