"""Cosmological calculation with the Boltzmann code CAMB."""

import warnings

import numpy as np

from .camb import CambEngine, Background, Thermodynamics, Primordial, Transfer, Harmonic, Fourier
from . import utils, constants

np.int = int  # Otherwise, AttributeError in isitgr in recent numpy


class IsitgrEngine(CambEngine):

    """Engine for the isitgr version of the Boltzmann code CAMB."""
    name = 'isitgr'

    _default_cosmological_parameters = dict(
        E11=0.0,
        E22=0.0,
        c1=1.0,
        c2=1.0,
        lambda_k=0.0,
        mu0=0.0,
        Sigma0=0.0,
        mu1=1.0,
        mu2=1.0,
        mu3=1.0,
        mu4=1.0,
        eta1=1.0,
        eta2=1.0,
        eta3=1.0,
        eta4=1.0,
        Sigma1=1.0,
        Sigma2=1.0,
        Sigma3=1.0,
        Sigma4=1.0,
        z_div=1.0,
        z_TGR=2.0,
        z_tw=0.05,
        k_c=0.01,
        k_tw=0.001,
        k_TGR=0.001,
        k_S=0.5,
        beta_1=1.0,
        lambda_1=0.0,
        exp_s=1.0,
        beta_2=1.0,
        lambda_2=0.0,
        gamma_0=0.54545,
        gamma_a=0.0,
        t_k=10.0,
        d_s=2.0,
        r_c=0.0,
        fR0_HS=0.0,
        n_HS=1.0,
    )

    _default_calculation_parameters = dict(
        MG_parameterization="muSigma",
        use_growth_index=None,
        damping_yukawa=False,
        use_BZ_form=False,
        use_HS_form=False,
        redshift_bins=None,
        scale_bins=None,
        use_nDGP=False,
    )

    def _set_camb(self):
        import isitgr
        self.camb = isitgr