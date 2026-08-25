"""Cosmological calculation with the Boltzmann code CAMB adapted for isitIDE."""


import warnings

import numpy as np

from .camb import CambEngine, Background as CambBackground, Thermodynamics, Primordial, Transfer, Harmonic, Fourier
from . import utils, constants


np.int = int

class Background(CambBackground):

    """Implementing functions for IDE growth rates"""

    def _pk_results(self):
        """Return the CAMB results object with matter power spectra computed.

        growth_rate/growth_factor are now derived from P(k, z) rather than the background-only
        growth ODE, so they need the 'fourier' task (which also pulls in 'transfer'), not just
        'background'. Requesting it here (only when these two methods are actually called).
        """
        return self._engine.get_fourier().fo

    @utils.flatarray(dtype=np.float64)
    def growth_rate(self, z):
        r"""Growth rate :math:`f(z) = d\ln D / d\ln a`, where :math:`D` is the growth factor
        and is rescaled with the fQ=f+g term for IDE models."""
        return self._pk_results().get_fQ_growth_rate_from_pk(z=z)

    @utils.flatarray(dtype=np.float64)
    def growth_factor(self, z):
        r"""Growth factor :math:`D(z)` normalized to D(0)=1"""
        return self._pk_results().get_growth_factor_from_pk(z=z)

class isitideEngine(CambEngine):

    """Engine for the isitide version of the Boltzmann code CAMB."""
    name = 'isitide'
    _default_cosmological_parameters = dict(w=-1.0, wa=0.0)
    _default_calculation_parameters = dict(dark_energy_model='IDEModel1')  

    def _set_camb(self):
        import isitide
        self.camb = isitide

