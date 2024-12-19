import os
import numpy as np

from .cosmology import Cosmology, get_engine
from . import constants


_dir_data = os.path.join(os.path.dirname(__file__), 'data')


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
    default_params = dict(h=0.6766, omega_cdm=0.11933, omega_b=0.02242, Omega_k=0., sigma8=0.8102, k_pivot=0.05, n_s=0.9665,
                          m_ncdm=[0.06], neutrino_hierarchy=None, T_ncdm_over_cmb=constants.TNCDM_OVER_CMB, N_eff=constants.NEFF,
                          tau_reio=0.0561, A_L=1.0, w0_fld=-1., wa_fld=0.)
    return Cosmology(engine=engine, extra_params=extra_params, **default_params).clone(**params)


def BOSS(engine=None, extra_params=None, **params):
    """
    Initialize :class:`Cosmology` based on BOSS fiducial cosmology.

    Note
    ----
    Can be found in https://arxiv.org/abs/1607.03155

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
    default_params = dict(h=0.676, Omega_m=0.31, omega_b=0.022, Omega_k=0., sigma8=0.8, k_pivot=0.05, n_s=0.97,
                          m_ncdm=[0.06], neutrino_hierarchy=None, T_ncdm_over_cmb=constants.TNCDM_OVER_CMB, N_eff=constants.NEFF,
                          A_L=1.0, w0_fld=-1., wa_fld=0.)
    return Cosmology(engine=engine, extra_params=extra_params, **default_params).clone(**params)


_AbacusSummit_params_filename = os.path.join(_dir_data, 'abacus_cosmologies.csv')


def AbacusSummit_params(name=None, filename=_AbacusSummit_params_filename, params=None):
    """
    Return AbacusSummit cosmological parameters.

    Note
    ----
    Described in:
    https://github.com/abacusorg/AbacusSummit/tree/master/Cosmologies

    Parameters
    ----------
    name : string, int, default=0
        AbacusSummit cosmology number, e.g. ``0`` or ``'000'``.

    filename : string, default='data/abacus_cosmologies.csv'
        Where Abacus cosmology parameters are saved as a 'csv' file.

    params : list, default=['omega_b', 'omega_cdm', 'h', 'A_s', 'n_s', 'alpha_s', 'N_ur', 'omega_ncdm', 'omega_k', 'tau_reio', 'w0_fld', 'wa_fld']
        Optionally, list of parameters to read from.
    """
    if name is not None:
        if not isinstance(name, str):
            name = '{:03d}'.format(name)

    if params is None:
        params = ['omega_b', 'omega_cdm', 'h', 'A_s', 'n_s', 'alpha_s', 'N_ur', 'omega_ncdm', 'omega_k', 'tau_reio', 'w0_fld', 'wa_fld']
    decode = {'root': str, 'notes': str, 'N_ncdm': int}
    default = {'tau_reio': 0.0544, 'omega_k': 0.}
    params = list(params)
    for param in list(default.keys()):
        if param in params:
            del params[params.index(param)]  # renove from list of params to read from csv
        else:
            default.pop(param)  # remove from returned params

    import re
    import csv
    toret = []
    with open(filename) as file:
        for iline, line in enumerate(csv.reader(file, delimiter=',')):
            line = [el.strip() for el in line]
            if iline == 0:  # header
                iparams = [line.index(param) for param in params]
                iroot = line.index('root')
                incdm = line.index('N_ncdm')
            else:
                tmp = default.copy()
                ncdm = int(line[incdm])
                for ii, param in zip(iparams, params):
                    value = line[ii]
                    value = decode.get(param, eval)(value)
                    if param == 'omega_ncdm' and not ncdm:  # no ncdm
                        value = tuple()
                    tmp[param] = value
                if name is not None:
                    if re.match('[^0-9]*{}$'.format(name), line[iroot]):
                        return tmp
                else:
                    toret.append(tmp)
    if name is not None:
        raise ValueError('AbacusSummit cosmology {} not found'.format(name))
    return toret


def AbacusSummit(name=0, engine='class', precision=None, extra_params=None, **params):
    """
    Initialize :class:`Cosmology` with AbacusSummit cosmological parameters.

    Note
    ----
    Original AbacusSummit initial power spectrum was computed with CLASS, with:
    https://github.com/abacusorg/AbacusSummit/blob/master/Cosmologies/abacus_cosm000/CLASS.ini

    Warning
    -------
    Be careful of the parameterization when calling :meth:`Cosmology.clone`:
    for AbacusSummit input parameters are ['omega_b', 'omega_cdm', 'h', 'A_s', 'n_s', 'alpha_s', 'N_ur', 'omega_ncdm', 'omega_k', 'tau_reio', 'w0_fld', 'wa_fld'].
    We recast the ``N_ur`` specification into ``N_eff`` with ``clone(N_eff=cosmo['N_eff'])`` such that changes with 'm_ncdm' are continuous.

    Parameters
    ----------
    name : string, int, default=0
        AbacusSummit cosmology number, e.g. ``0`` or ``'000'``.

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
    default_params = dict(k_pivot=0.05, neutrino_hierarchy=None, T_ncdm_over_cmb=constants.TNCDM_OVER_CMB, A_L=1.0)
    default_params.update(AbacusSummit_params(name=name))
    engine = get_engine(engine)
    default_extra_params = {}
    if engine is not None and engine.name == 'class':
        default_extra_params = {'recombination': 'HyRec'}  # {'recombination': 'HyRec', 'sBBN file': 'bbn/sBBN.dat'}  # TODO: change this for Y3? ~5e-5 change in rs_drag
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
            for name in ['recfast_Nz0', 'tol_perturb_integration', 'perturb_sampling_stepsize']: prec.pop(name)  # these do not exist anymore
            default_extra_params.update(prec)
    extra_params = {**default_extra_params, **(extra_params or {})}
    cosmo = Cosmology(engine=engine, extra_params=extra_params, **default_params)
    cosmo = cosmo.clone(base='input', N_eff=cosmo['N_eff'])
    return cosmo.clone(**params)


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
    return AbacusSummit(name='000', engine=engine, precision=precision, extra_params=extra_params, **params)


DESI = AbacusSummitBase

"""Tabulated cosmologies."""


_DESI_filename = os.path.join(_dir_data, 'desi.dat')


def TabulatedDESI():
    """
    Tabulated DESI cosmology.

    Note
    ----
    Redshift interpolation range is [0, 10]; returned values outside this range are constant (no error is raised).
    Relative interpolation precision is 1e-7; relative difference with camb prediction is 1e-7, with astropy 1e-5 and pyccl 1e-6
    (see tests/test_tabulated.py).
    """
    return DESI(engine='tabulated', extra_params={'filename': _DESI_filename, 'names': ['efunc', 'comoving_radial_distance']})


def save_TabulatedDESI():
    cosmo = DESI()
    bins_log = 'np.logspace(-8, 2, 40001)'
    z = np.concatenate([[0], eval(bins_log, {'np': np})], axis=0)
    array = np.array([z, cosmo.efunc(z), cosmo.comoving_radial_distance(z)]).T
    header = 'z = [0] + {}\nz efunc(z) comoving_radial_distance(z) [Mpc/h]'.format(bins_log)
    np.savetxt(_DESI_filename, array, fmt='%.18e', header=header, comments='# ')
