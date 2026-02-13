"""Cosmological calculation with the Boltzmann code CAMB."""


import warnings

import numpy as np

from .camb import CambEngine, Background, Thermodynamics, Primordial, Transfer, Harmonic, Fourier
from . import utils, constants


np.int = int  # Otherwise, AttributeError in isitgr in recent numpy, see https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations


class IsitgrEngine(CambEngine):

    """Engine for the isitgr version of the Boltzmann code CAMB."""
    name = 'isitgr'
    _default_cosmological_parameters = dict(
        # existing ones...
        Q0=0, Qinf=0, D0=0, R0=0, Dinf=0, Rinf=0, s=0, k_c=0.1,
        E11=0, E22=0, mu0=0, Sigma0=0, c1=1, c2=1, Lambda=0,
        z_div=1.0, z_TGR=3.0, z_tw=0.05, k_tw=0.001,
        Q1=0, Q2=0, Q3=0, Q4=0, D1=0, D2=0, D3=0, D4=0,
        mu1=1.0, mu2=1.0, mu3=1.0, mu4=1.0,
        eta1=1.0, eta2=1.0, eta3=1.0, eta4=1.0,
        Sigma1=1.0, Sigma2=1.0, Sigma3=1.0, Sigma4=1.0,
        beta_1=1.0, beta_2=1.0, lambda_1=1.0, lambda_2=1.0, exp_s=1.0,

        # ---- For growth index ----
        gamma_0=0.54545,
        gamma_a=0.0,
        GI_tk=10.0,
        GI_ds=2.0,
        k_TGR=0.01,

        # ---- For binning ----
        redshift_bins=True,
        scale_bins=False,
        scale_bins_method="traditional",  # "traditional" or "hybrid"
        k_S=0.2, # k value above which we return to GR
    )

    _default_calculation_parameters = dict(
        MG_parameterization='muSigma',
        use_BZ_form=False,
        use_growth_index=None,
        damping_yukawa=False,
    )

    def _set_camb(self):
        import isitgr
        self.camb = isitgr

