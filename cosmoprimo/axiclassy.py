"""Cosmological calculation with the Boltzmann code AxiCLASS, wrapping by Rafaela Gsponer."""

from pyclass import axiclass

from .cosmology import BaseEngine, CosmologyInputError, CosmologyComputationError
from . import classy


class AxiClassEngine(classy.ClassEngine):

    """Engine for the Boltzmann code AxiCLASS."""

    name = 'axiclass'

    _default_cosmological_parameters = dict()

    def _set_classy(self, params):
        # Enable passing scf_parameters individually, works only for scf_parameters = [theta_i, tehta_dot_i]
        if 'scf_parameters__1' in params:
            if 'scf_parameters__2' not in params:
                raise CosmologyInputError('scf_parameters__2 not found in params')
            params['scf_parameters'] = [params['scf_parameters__1'], params['scf_parameters__2']]
            # _ClassEngine do not expect scf_parameters__X to be in params
            del params['scf_parameters__1']
            del params['scf_parameters__2']

        class _ClassEngine(axiclass.ClassEngine):

            def compute(self, tasks):
                try:
                    return super(_ClassEngine, self).compute(tasks)
                except axiclass.ClassInputError as exc:
                    raise CosmologyInputError from exc
                except axiclass.ClassComputationError as exc:
                    raise CosmologyComputationError from exc

        self.classy = _ClassEngine(params=params)


class Background(classy.BaseClassBackground, axiclass.Background):

    """Your modifications, if any."""


class Thermodynamics(classy.BaseClassThermodynamics, axiclass.Thermodynamics):

    """Your modifications, if any."""


class Primordial(classy.BaseClassPrimordial, axiclass.Primordial):

     """Your modifications, if any."""


class Perturbations(classy.BaseClassPerturbations, axiclass.Perturbations):

     """Your modifications, if any."""


class Transfer(classy.BaseClassTransfer, axiclass.Transfer):

     """Your modifications, if any."""


class Harmonic(classy.BaseClassHarmonic, axiclass.Harmonic):
     """Your modifications, if any."""


class Fourier(classy.BaseClassFourier, axiclass.Fourier):
     """Your modifications, if any."""
