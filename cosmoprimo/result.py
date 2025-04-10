import os
import numpy as np

from .cosmology import Cosmology, get_engine
from . import constants

from .fiducial import AbacusSummit


def DESIDR2Flatw0waCDM(engine='class', precision=None, extra_params=None, **params):
    """
    Initialize :class:`Cosmology` with the best fit values for a flat, w0wa, CDM cosmology from CMB + DESI BAO DR2 + DESY5 (https://arxiv.org/pdf/2503.14738).

    The best fit values are from: '/global/cfs/cdirs/desicollab/science/cpe/y3_bao_cosmo/bao_v1p2/bao/iminuit/camb/run1/base_w_wa/desi-bao-all_desy5sn_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck-NPIPE-highl-CamSpec-TTTEEE_planck-act-dr6-lensing'

    Parameters
    ----------
    engine : string, default='class'
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`Cosmology.engine`.

    precision : string, default=None
        Precision for computation; pass 'base' to get same precision as AbacusSummitBase (few minutes).

    extra_params : dict, default=None
        Extra engine parameters, typically precision parameters.

    params : dict
        Cosmological and calculation parameters which take priority over the default ones.

    Returns
    -------
    cosmology : Cosmology
    """

    # Read from bestfit.minimum:
    bestfit_params = {'Omega_m':0.3191980194, 'omega_b':0.02221485621, 'H0':66.73428704, 
                      'logA':3.038847745, 'n_s':0.9644215278, 'tau_reio':0.05271118001, 
                      'w0_fld':-0.7536302620, 'wa_fld':-0.8574714585}

    cosmo = AbacusSummit(engine=engine, precision=precision, extra_params=extra_params, **bestfit_params)

    return cosmo.clone(**params)

