from .cosmology import Cosmology, merge_params
from . import constants


def Planck2018FullFlatLCDM(engine=None, extra_params=None, **params):
    """
    Initialize :class:`Cosmology` based on Planck2018 TT, TE, EE, lowE, lensing and BAO data.

    Parameters
    ----------
    engine : string
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`Cosmology.engine`.

    extra_params : dict
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


def AbacusBaseline(engine=None, extra_params=None, **params):
    """
    Initialize :class:`Cosmology` with baseline Abacus cosmological parameters (Planck2018, base_plikHM_TTTEEE_lowl_lowE_lensing mean).

    Note
    ----
    Original Abacus initial power spectrum was computed with CLASS.

    Parameters
    ----------
    engine : string
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`Cosmology.engine`.

    extra_params : dict
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
