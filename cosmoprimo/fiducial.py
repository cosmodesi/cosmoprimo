from .cosmology import Cosmology, merge_params
from . import constants


def Planck2018FullFlatLCDM(engine=None, extra_params=None, **params):
    """
    Initialize :class:`Cosmology` based on Planck2018 TT, TE, EE, lowE, lensing and BAO data.

    Parameters
    ----------
    engine : string, default=None
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`Cosmology.engine`.

    extra_params : dict, default=None
        Extra engine parameters, typically precision parameters.

    params : dict
        Cosmological and calculation parameters which take priority over the default ones.

    Returns
    -------
    cosmology : Cosmology
    """
    _default_cosmological_parameters = dict(h=0.6766, omega_cdm=0.11933, omega_b=0.02242, Omega_k=0., sigma8=0.8102, k_pivot=0.05, n_s=0.9665,
    m_ncdm=[0.06], neutrino_hierarchy=None, T_ncdm=constants.TNCDM, N_eff=constants.NEFF, tau_reio=0.0561, A_L=1.0, w0_fld=-1., wa_fld=0., cs2_fld=1.)
    params = merge_params(_default_cosmological_parameters,params)
    return Cosmology(engine=engine,extra_params=extra_params,**params)


def AbacusSummitBase(engine='class', extra_params=None, **params):
    """
    Initialize :class:`Cosmology` with base AbacusSummit cosmological parameters (Planck2018, base_plikHM_TTTEEE_lowl_lowE_lensing mean).

    Note
    ----
    Original AbacusSummit initial power spectrum was computed with CLASS, with:
    https://github.com/abacusorg/AbacusSummit/blob/master/Cosmologies/abacus_cosm000/CLASS.ini

    Parameters
    ----------
    engine : string, default='class'
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`Cosmology.engine`.

    extra_params : dict, default=None
        Extra engine parameters, typically precision parameters.

    params : dict
        Cosmological and calculation parameters which take priority over the default ones.

    Returns
    -------
    cosmology : Cosmology
    """
    _default_cosmological_parameters = dict(h=0.6736, omega_cdm=0.12, omega_b=0.02237, Omega_k=0., A_s=2.083e-09, k_pivot=0.05, n_s=0.9649,
    omega_ncdm=[0.0006442], neutrino_hierarchy=None, T_ncdm=constants.TNCDM, N_ur=2.0328, tau_reio=0.0561, A_L=1.0, w0_fld=-1., wa_fld=0., cs2_fld=1.)
    params = merge_params(_default_cosmological_parameters,params)
    return Cosmology(engine=engine,extra_params=extra_params,**params)


DESI = AbacusSummitBase

"""Tabulated cosmologies."""

import os
import numpy as np

_dir_tabulated = os.path.join(os.path.dirname(__file__), 'data')


_DESI_filename = os.path.join(_dir_tabulated, 'desi.dat')


def TabulatedDESI():
    """
    Tabulated DESI cosmology.

    Note
    ----
    Redshift interpolation range is [0, 10]; returned values outside this range are constant (no error is raised).
    Relative interpolation precision is 1e-7; relative difference with camb prediction is 1e-7, with astropy 1e-5 and pyccl 1e-6
    (see tests/test_tabulated.py).
    """
    return DESI(engine='tabulated', extra_params={'filename':_DESI_filename, 'names':['efunc','comoving_radial_distance']})


def save_TabulatedDESI():
    cosmo = DESI()
    bins_log = 'np.logspace(-8, 1, 40001)'
    z = np.concatenate([[0], eval(bins_log,{'np':np})],axis=0)
    array = np.array([z, cosmo.efunc(z), cosmo.comoving_radial_distance(z)]).T
    header = 'z = [0] + {}\nz efunc(z) comoving_radial_distance(z) [Mpc/h]'.format(bins_log)
    np.savetxt(_DESI_filename, array, fmt='%.18e', header=header, comments='# ')
