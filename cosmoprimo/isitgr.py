"""Cosmological calculation with the Boltzmann code CAMB."""


import warnings

import numpy as np

from .camb import CambEngine, Background, Thermodynamics, Primordial, Transfer, Harmonic, Fourier, CosmologyError, enum
from . import utils, constants


class IsitgrEngine(CambEngine):

    """Engine for the isitgr version of the Boltzmann code CAMB."""
    name = 'isitgr'
    specific_params = dict(Q0=0, Qinf=0, D0=0, R0=0, Dinf=0, Rinf=0, s=0, k_c=0,
                           E11=0, E22=0, mu0=0, Sigma0=0, c1=1, c2=1, Lambda=0,
                           z_div=0, z_TGR=0, z_tw=0, k_tw=0, Q1=0, Q2=0, Q3=0, Q4=0, D1=0,
                           D2=0, D3=0, D4=0, mu1=0, mu2=0, mu3=0, mu4=0, eta1=0, eta2=0,
					       eta3=0, eta4=0, Sigma1=0, Sigma2=0, Sigma3=0, Sigma4=0,
                           parameterization=None, binning=None)

    def _set_camb(self):
        import isitgr
        self.camb = isitgr