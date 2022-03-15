from .cosmology import Cosmology, merge_params, get_engine
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
    m_ncdm=[0.06], neutrino_hierarchy=None, T_ncdm_over_cmb=constants.TNCDM_OVER_CMB, N_eff=constants.NEFF, tau_reio=0.0561, A_L=1.0, w0_fld=-1., wa_fld=0.)
    params = merge_params(_default_cosmological_parameters,params)
    return Cosmology(engine=engine,extra_params=extra_params,**params)


def AbacusSummitBase(engine='class', precision=None, extra_params=None, **params):
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
    _default_cosmological_parameters = dict(h=0.6736, omega_cdm=0.12, omega_b=0.02237, Omega_k=0., A_s=2.083e-09, k_pivot=0.05, n_s=0.9649,
    omega_ncdm=[0.0006442], neutrino_hierarchy=None, T_ncdm_over_cmb=constants.TNCDM_OVER_CMB, N_ur=2.0328, tau_reio=0.0544, A_L=1.0, w0_fld=-1., wa_fld=0.)
    params = merge_params(_default_cosmological_parameters, params)
    if extra_params is None: extra_params = {}
    engine = get_engine(engine)
    from .classy import ClassEngine
    if engine is ClassEngine:
        extra_params.setdefault('recombination', 'HyRec')
        if precision == 'base':
            prec = dict(tol_ncdm_bg=1.e-10, recfast_Nz0=100000, tol_thermo_integration=1.e-5, recfast_x_He0_trigger_delta=0.01, recfast_x_H0_trigger_delta=0.01, evolver=0, k_min_tau0=0.002, k_max_tau0_over_l_max=3.,
                        k_step_sub=0.015, k_step_super=0.0001, k_step_super_reduction=0.1, start_small_k_at_tau_c_over_tau_h=0.0004, start_large_k_at_tau_h_over_tau_k=0.05, tight_coupling_trigger_tau_c_over_tau_h=0.005,
                        tight_coupling_trigger_tau_c_over_tau_k=0.008, start_sources_at_tau_c_over_tau_h=0.006, l_max_g=50, l_max_pol_g=25, l_max_ur=150, l_max_ncdm=50, tol_perturb_integration=1.e-6, perturb_sampling_stepsize=0.01,
                        radiation_streaming_approximation=2, radiation_streaming_trigger_tau_over_tau_k=240., radiation_streaming_trigger_tau_c_over_tau=100., ur_fluid_approximation=2, ur_fluid_trigger_tau_over_tau_k=50.,
                        ncdm_fluid_approximation=3, ncdm_fluid_trigger_tau_over_tau_k=51., tol_ncdm_synchronous=1.e-10, tol_ncdm_newtonian=1.e-10, l_logstep=1.026, l_linstep=25, hyper_sampling_flat=12., hyper_sampling_curved_low_nu=10.,
                        hyper_sampling_curved_high_nu=10., hyper_nu_sampling_step=10., hyper_phi_min_abs=1.e-10, hyper_x_tol=1.e-4, hyper_flat_approximation_nu=1.e6, q_linstep=0.20, q_logstep_spline=20., q_logstep_trapzd=0.5,
                        q_numstep_transition=250, transfer_neglect_delta_k_S_t0=100., transfer_neglect_delta_k_S_t1=100., transfer_neglect_delta_k_S_t2=100., transfer_neglect_delta_k_S_e=100., transfer_neglect_delta_k_V_t1=100.,
                        transfer_neglect_delta_k_V_t2=100., transfer_neglect_delta_k_V_e=100., transfer_neglect_delta_k_V_b=100., transfer_neglect_delta_k_T_t2=100., transfer_neglect_delta_k_T_e=100., transfer_neglect_delta_k_T_b=100.,
                        neglect_CMB_sources_below_visibility=1.e-30, transfer_neglect_late_source=3000., halofit_k_per_decade=3000., l_switch_limber=40., accurate_lensing=1, num_mu_minus_lmax=1000., delta_l_max=1000.)
            for name in ['recfast_Nz0', 'tol_perturb_integration', 'perturb_sampling_stepsize']: prec.pop(name) # these do not exist anymore
            prec.update(extra_params)
            extra_params = prec
    return Cosmology(engine=engine, extra_params=extra_params, **params)


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
    bins_log = 'np.logspace(-8, 2, 40001)'
    z = np.concatenate([[0], eval(bins_log,{'np':np})],axis=0)
    array = np.array([z, cosmo.efunc(z), cosmo.comoving_radial_distance(z)]).T
    header = 'z = [0] + {}\nz efunc(z) comoving_radial_distance(z) [Mpc/h]'.format(bins_log)
    np.savetxt(_DESI_filename, array, fmt='%.18e', header=header, comments='# ')
