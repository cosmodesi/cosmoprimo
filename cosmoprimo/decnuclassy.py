"""Cosmological calculation with the Boltzmann code DecnuCLASS."""

from pyclass import decnuclass

from .cosmology import BaseEngine, CosmologyInputError, CosmologyComputationError
from . import classy


class DecnuClassEngine(classy.ClassEngine):

    """Engine for the Boltzmann code decnuclass."""

    name = 'decnuclass'

    _default_cosmological_parameters = dict()
    _check_ignore = ['m_ncdm']

    def _set_classy(self, params):

        class _ClassEngine(decnuclass.ClassEngine):

            def compute(self, tasks):
                try:
                    return super(_ClassEngine, self).compute(tasks)
                except decnuclass.ClassInputError as exc:
                    raise CosmologyInputError from exc
                except decnuclass.ClassComputationError as exc:
                    raise CosmologyComputationError from exc

        self.classy = _ClassEngine(params=params)


class Background(classy.BaseClassBackground, decnuclass.Background):

    """Your modifications, if any."""


class Thermodynamics(classy.BaseClassThermodynamics, decnuclass.Thermodynamics):

    """Your modifications, if any."""


class Primordial(classy.BaseClassPrimordial, decnuclass.Primordial):

     """Your modifications, if any."""


class Perturbations(classy.BaseClassPerturbations, decnuclass.Perturbations):

     """Your modifications, if any."""


class Transfer(classy.BaseClassTransfer, decnuclass.Transfer):

     """Your modifications, if any."""


class Harmonic(classy.BaseClassHarmonic, decnuclass.Harmonic):
     """Your modifications, if any."""


class Fourier(classy.BaseClassFourier, decnuclass.Fourier):
     """Your modifications, if any."""
