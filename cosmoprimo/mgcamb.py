"""Cosmological calculation with the Boltzmann code CAMB."""


import warnings

import numpy as np

from .camb import CambEngine, Background, Thermodynamics, Primordial, Transfer, Harmonic, Fourier
from . import utils, constants


np.int = int  # Otherwise, AttributeError in mgcamb in recent numpy, see https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations


class MGCambEngine(CambEngine):

    """Engine for the mgcamb version of the Boltzmann code CAMB."""
    name = 'mgcamb'
    _default_cosmological_parameters = dict(GRtrans=0.001, B1=1.333, lambda1_2=1000., B2=0.5, lambda2_2=1000., ss=4.0, E11=1.0, E22=1.0,
                                            ga=0.5, nn=2.0, mu0=0.0, sigma0=0.0, MGQfix=1.0, MGRfix=1.0, Qnot=1.0,
                                            Rnot=1.0, sss=0.0, Linder_gamma=0.545, B0=0.001, beta_star=1.0, a_star=0.5, xi_star=0.001,
                                            beta0=0.0, xi0=0.0001, DilS=0.24, DilR=1.0, F_R0=0.0001, FRn=1.0,
                                            w0DE=-1.0, waDE=0.0, MGCAMB_Mu_idx_1=1.0, MGCAMB_Mu_idx_2=1.0,
                                            MGCAMB_Mu_idx_3=1.0, MGCAMB_Mu_idx_4=1.0, MGCAMB_Mu_idx_5=1.0, MGCAMB_Mu_idx_6=1.0, MGCAMB_Mu_idx_7=1.0,
                                            MGCAMB_Mu_idx_8=1.0, MGCAMB_Mu_idx_9=1.0, MGCAMB_Mu_idx_10=1.0, MGCAMB_Mu_idx_11=1.0, MGCAMB_Sigma_idx_1=1.0,
                                            MGCAMB_Sigma_idx_2=1.0, MGCAMB_Sigma_idx_3=1.0, MGCAMB_Sigma_idx_4=1.0, MGCAMB_Sigma_idx_5=1.0,
                                            MGCAMB_Sigma_idx_6=1.0, MGCAMB_Sigma_idx_7=1.0, MGCAMB_Sigma_idx_8=1.0, MGCAMB_Sigma_idx_9=1.0,
                                            MGCAMB_Sigma_idx_10=1.0, MGCAMB_Sigma_idx_11=1.0, Funcofw_1=0.7, Funcofw_2=0.7, Funcofw_3=0.7, Funcofw_4=0.7,
                                            Funcofw_5=0.7, Funcofw_6=0.7, Funcofw_7=0.7, Funcofw_8=0.7, Funcofw_9=0.7, Funcofw_10=0.7, Funcofw_11=0.7)

    _default_calculation_parameters = dict(MG_wrapped=True, MG_flag=0, pure_MG_flag=1, alt_MG_flag=1, QSA_flag=1, CDM_flag=1, muSigma_flag=1,
                                           DE_model=0, MGDE_pert=False, mugamma_par=1, musigma_par=1, QR_par=1)  #dict(parameterization=None, binning=None)

    def _set_camb(self):
        import mgcamb
        self.camb = mgcamb