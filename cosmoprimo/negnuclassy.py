"""Cosmological calculation with the Boltzmann code NegnuCLASS."""

from pyclass import negnuclass

from .cosmology import BaseEngine, CosmologyInputError, CosmologyComputationError
from . import classy


class NegnuClassEngine(classy.ClassEngine):

    """Engine for the Boltzmann code negnuclass."""

    name = 'negnuclass'

    _default_cosmological_parameters = dict()
    _check_ignore = ['m_ncdm']

    def _set_classy(self, params):

        class _ClassEngine(negnuclass.ClassEngine):

            def compute(self, tasks):
                try:
                    return super(_ClassEngine, self).compute(tasks)
                except negnuclass.ClassInputError as exc:
                    raise CosmologyInputError from exc
                except negnuclass.ClassComputationError as exc:
                    raise CosmologyComputationError from exc

        self.classy = _ClassEngine(params=params)


class Background(classy.BaseClassBackground, negnuclass.Background):

    """Your modifications, if any."""


class Thermodynamics(classy.BaseClassThermodynamics, negnuclass.Thermodynamics):

    """Your modifications, if any."""


class Primordial(classy.BaseClassPrimordial, negnuclass.Primordial):

     """Your modifications, if any."""


class Perturbations(classy.BaseClassPerturbations, negnuclass.Perturbations):

     """Your modifications, if any."""


class Transfer(classy.BaseClassTransfer, negnuclass.Transfer):

     """Your modifications, if any."""


class Harmonic(classy.BaseClassHarmonic, negnuclass.Harmonic):
     """Your modifications, if any."""


class Fourier(classy.BaseClassFourier, negnuclass.Fourier):
     """Your modifications, if any."""
