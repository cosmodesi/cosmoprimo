"""Cosmology class"""

import logging
import functools
import sys

import numpy as np

from .utils import BaseClass
from . import constants


class CosmologyError(Exception):

    """Exception raise by :class:`Codsmology`."""


class BaseEngine(BaseClass):

    """Base engine for cosmological calculation."""

    def __init__(self, extra_params=None, **params):
        """
        Initialise engine.

        Parameters
        ----------
        extra_params : dict
            Extra engine parameters, typically precision parameters.

        params : dict
            Engine parameters.
        """
        self.params = params
        self.extra_params = extra_params or {}

    def __getitem__(self, name):
        """Return an input (or easily derived) parameter."""
        return self.get(name)

    def get(self, *args, **kwargs):
        """Return an input (or easily derived) parameter."""
        if len(args) == 1:
            name = args[0]
            has_default = 'default' in kwargs
            default = kwargs.get('default',None)
        else:
            name,default = args
            has_default = True
        if name in self.params:
            return self.params[name]
        if name.startswith('omega'):
            return self['O'+name[1:]]*self.params['h']**2
        if name == 'H0':
            return self.params['h']*100
        if name == 'ln10^{10}A_s':
            return np.log(10**10*self.params['As'])
        #if name == 'rho_crit':
        #    return constants.rho_crit_Msunph_per_Mpcph3
        if name == 'Omega_g':
            rho = 4./constants.c**3 * constants.Stefan_Boltzmann * self.params['T_cmb']**4 # density, kg/m^3
            return rho/(self['h']**2*constants.rho_crit_kgph_per_mph3)
        if name == 'T_ur':
            return self.params['T_cmb'] * (4./11.)**(1./3.)
        if name == 'Omega_ur':
            rho = self['N_ur'] * 7./8. * 4./constants.c**3 * constants.Stefan_Boltzmann * self['T_ur']**4 # density, kg/m^3
            return rho/(self['h']**2*constants.rho_crit_kgph_per_mph3)
        if name == 'Omega_ncdm':
            self.params['Omega_ncdm'] = self._get_Omega_ncdm(z=0)
            return self.params['Omega_ncdm']
        if name == 'Omega_m':
            return  self['Omega_b'] + self['Omega_cdm'] + self['Omega_ncdm']
        if has_default:
            return default
        raise CosmologyError('Parameter {} not found.'.format(name))

    def _get_A_s_fid(self):
        """First guess for power spectrum amplitude :math:`A_{s}` (given input :math:`sigma_{8}`)."""
        # https://github.com/lesgourg/class_public/blob/4724295b527448b00faa28bce973e306e0e82ef5/source/input.c#L1161
        if 'A_s' in self.params:
            return self.params['A_s']
        return 2.43e-9*(self['sigma8']/0.87659)**2

    def _get_Omega_ncdm(self, z=0, epsrel=1e-7):
        """
        Returne nergy density of non-CDM components (massive neutrinos) by integrating over the phase-space distribution (frozen since CMB).

        Parameters
        ----------
        z : float
            Redshift.

        epsrel : float
            Relative precision (for :meth:`scipy.integrate.quad` integration).

        Returns
        -------
        Omega_ncdm : Density parameter of massive neutrinos.
        """
        T_eff = self['T_cmb'] * self['T_ncdm']
        a = 1./(1. + z)
        toret = 0.
        from scipy import integrate
        for m in self['m_ncdm']:
            # arXiv 1812.05995 eq. 6
            m_over_T = m*constants.electronvolt/(constants.Boltzmann*(T_eff/a))

            def phasespace_integrand(q):
                return q**2*np.sqrt(q**2+m_over_T**2)/(1.+np.exp(q))

            # upper bound of 100 enough (10^⁻16 error)
            toret += integrate.quad(phasespace_integrand,0,100,epsrel=epsrel)[0]/(7.*np.pi**4/120.)

        rho_crit = constants.rho_crit_kgph_per_mph3 * self['h']**2
        toret *= 7./8. * 4/constants.c**3 * constants.Stefan_Boltzmann/rho_crit * (T_eff/a)**4
        return toret

    def get_background(self):
        """Return :class:`Background` calculations."""
        return sys.modules[self.__class__.__module__].Background(self)

    def get_thermodynamics(self):
        """Return :class:`Thermodynamics` calculations."""
        return sys.modules[self.__class__.__module__].Thermodynamics(self)

    def get_primordial(self):
        """Return :class:`Primordial` calculations."""
        return sys.modules[self.__class__.__module__].Primordial(self)

    def get_transfer(self):
        """Return :class:`Transfer` calculations."""
        return sys.modules[self.__class__.__module__].Transfer(self)

    def get_harmonic(self):
        """Return :class:`Transfer` calculations."""
        return sys.modules[self.__class__.__module__].Harmonic(self)

    def get_fourier(self):
        """Return :class:`Fourier` calculations."""
        return sys.modules[self.__class__.__module__].Fourier(self)


def get_engine(cosmology, engine=None, set_engine=True, **extra_params):
    """
    Return engine for cosmological calculation.

    Parameters
    ----------
    cosmology : Cosmology
        Current cosmology.

    engine : BaseEngine, string
        Engine or one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`cosmology.engine`.

    set_engine : bool
        Whether to attach returned engine to ``cosmology``.
        (Set ``False`` if e.g. you want to use this engine for a single calculation).

    extra_params : dict
        Extra engine parameters, typically precision parameters.

    Returns
    -------
    engine : BaseEngine
    """
    if engine is None:
        if cosmology.engine is None:
            raise CosmologyError('Please provide an engine')
        engine = cosmology.engine
    elif engine == 'class':
        from .classy import ClassEngine
        engine = ClassEngine(**cosmology.params,extra_params=extra_params)
    elif engine == 'camb':
        from .camb import CambEngine
        engine = CambEngine(**cosmology.params,extra_params=extra_params)
    elif engine == 'eisenstein_hu':
        from .eisenstein_hu import EisensteinHuEngine
        engine = EisensteinHuEngine(**cosmology.params,extra_params=extra_params)
    elif engine == 'eisenstein_hu_nowiggle':
        from .eisenstein_hu_nowiggle import EisensteinHuNoWiggleEngine
        engine = EisensteinHuNoWiggleEngine(**cosmology.params,extra_params=extra_params)
    elif engine == 'bbks':
        from .bbks import BBKSEngine
        engine = BBKSEngine(**cosmology.params,extra_params=extra_params)
    elif isinstance(engine,str):
        raise CosmologyError('Unknown engine {}'.format(engine))
    if set_engine:
        cosmology.engine = engine
    return engine


def Background(cosmology, engine=None, **extra_params):
    """
    Return :class:`Background` calculations.

    Parameters
    ----------
    cosmology : Cosmology
        Current cosmology.

    engine : string
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`cosmology.engine`.

    set_engine : bool
        Whether to attach returned engine to ``cosmology``
        (Set ``False`` if e.g. you want to use this engine for a single calculation).

    extra_params : dict
        Extra engine parameters, typically precision parameters.

    Returns
    -------
    engine : BaseEngine
    """
    engine = get_engine(cosmology,engine=engine,**extra_params)
    return engine.get_background()


def Thermodynamics(cosmology, engine=None, **extra_params):
    """
    Return :class:`Thermodynamics` calculations.

    Parameters
    ----------
    cosmology : Cosmology
        Current cosmology.

    engine : string
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`cosmology.engine`.

    set_engine : bool
        Whether to attach returned engine to ``cosmology``.
        (Set ``False`` if e.g. you want to use this engine for a single calculation).

    extra_params : dict
        Extra engine parameters, typically precision parameters.

    Returns
    -------
    engine : BaseEngine
    """
    engine = get_engine(cosmology,engine=engine,**extra_params)
    return engine.get_thermodynamics()


def Primordial(cosmology, engine=None, **extra_params):
    """
    Return :class:`Primordial` calculations.

    Parameters
    ----------
    cosmology : Cosmology
        Current cosmology.

    engine : string
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`cosmology.engine`.

    set_engine : bool
        Whether to attach returned engine to ``cosmology``.
        (Set ``False`` if e.g. you want to use this engine for a single calculation).

    extra_params : dict
        Extra engine parameters, typically precision parameters.

    Returns
    -------
    engine : BaseEngine
    """
    engine = get_engine(cosmology,engine=engine,**extra_params)
    return engine.get_primordial()


def Transfer(cosmology, engine=None, **extra_params):
    """
    Return :class:`Transfer` calculations.

    Parameters
    ----------
    cosmology : Cosmology
        Current cosmology.

    engine : string
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`cosmology.engine`.

    set_engine : bool
        Whether to attach returned engine to ``cosmology``.
        (Set ``False`` if e.g. you want to use this engine for a single calculation).

    extra_params : dict
        Extra engine parameters, typically precision parameters.

    Returns
    -------
    engine : BaseEngine
    """
    engine = get_engine(cosmology,engine=engine,**extra_params)
    return engine.get_transfer()


def Harmonic(cosmology, engine=None, **extra_params):
    """
    Return :class:`Harmonic` calculations.

    Parameters
    ----------
    cosmology : Cosmology
        Current cosmology.

    engine : string
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`cosmology.engine`.

    set_engine : bool
        Whether to attach returned engine to ``cosmology``.
        (Set ``False`` if e.g. you want to use this engine for a single calculation).

    extra_params : dict
        Extra engine parameters, typically precision parameters.

    Returns
    -------
    engine : BaseEngine
    """
    engine = get_engine(cosmology,engine=engine,**extra_params)
    return engine.get_harmonic()


def Fourier(cosmology, engine=None, **extra_params):
    """
    Return :class:`Fourier` calculations.

    Parameters
    ----------
    cosmology : Cosmology
        Current cosmology.

    engine : string
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`cosmology.engine`.

    set_engine : bool
        Whether to attach returned engine to ``cosmology``.
        (Set ``False`` if e.g. you want to use this engine for a single calculation).

    extra_params : dict
        Extra engine parameters, typically precision parameters.

    Returns
    -------
    engine : BaseEngine
    """
    engine = get_engine(cosmology,engine=engine,**extra_params)
    return engine.get_fourier()


def _include_conflicts(params):
    """Add in conflicting parameters to input ``params`` dictionay (in-place operation)."""
    for name in list(params.keys()):
        for conf in find_conflicts(name):
            params[conf] = params[name]


class Cosmology(BaseEngine):

    """Cosmology, defined as a set of parameters (and possibly a current engine attached to it)."""

    _default_cosmological_parameters = dict(h=0.7, Omega_cdm=0.25, Omega_b=0.05, Omega_k=0., sigma8=0.8, k_pivot=0.05, n_s=0.96, alpha_s=0., r=0., T_cmb=2.7255,
    N_ur=None, m_ncdm=None, neutrino_hierarchy=None, T_ncdm=constants.TNCDM, N_eff=constants.NEFF, tau_reio=0.06, reionization_width=0.5, A_L=1.0,
    w0_fld=-1., wa_fld=0., cs2_fld=1.)
    _default_calculaton_parameters = dict(non_linear='', modes='s', lensing=False, z_pk=None, kmax_pk=10., ellmax_cl=2500)

    def __init__(self, engine=None, extra_params=None, **params):
        """
        Initialise :class:`Cosmology`.

        Parameters
        ----------
        engine : string
            Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
            If ``None``, returns current :attr:`cosmology.engine`.

        extra_params : dict
            Extra engine parameters, typically precision parameters.

        params : dict
            Cosmological and calculation parameters which take priority over the default ones.
        """
        check_params(params)
        self.params = compile_params(merge_params(self.__class__.get_default_parameters(include_conflicts=False),params))
        self.engine = engine
        if self.engine is not None:
            self.set_engine(self.engine, **(extra_params or {}))

    def set_engine(self, engine, **extra_params):
        """
        Set engine for cosmological calculation.

        Parameters
        ----------
        engine : string
            Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
            If ``None``, returns current :attr:`cosmology.engine`.

        set_engine : bool
            Whether to attach returned engine to ``cosmology``.
            (Set ``False`` if e.g. you want to use this engine for a single calculation).

        extra_params : dict
            Extra engine parameters, typically precision parameters.
        """
        self.engine = get_engine(self, engine, **extra_params)

    @classmethod
    def get_default_parameters(cls, of=None, include_conflicts=True):
        """
        Return default input parameters.

        Parameters
        ----------
        of : string
            One of ['cosmology','calculation'].
            If ``None``, returns all parameters.

        include_conflicts : bool
            Whether to include conflicting parameters (then all accepted parameters).

        Returns
        -------
        params : dict
            Dictionary of default parameters.
        """
        if of == 'cosmology':
            toret = cls._default_cosmological_parameters.copy()
            if include_conflicts: _include_conflicts(toret)
            return toret
        if of == 'calculation':
            toret = cls._default_calculaton_parameters.copy()
            if include_conflicts: _include_conflicts(toret)
            return toret
        if of is None:
            toret = cls.get_default_parameters(of='cosmology',include_conflicts=include_conflicts)
            toret.update(cls.get_default_parameters(of='calculation',include_conflicts=include_conflicts))
            return toret
        raise CosmologyError('No default parameters for {}'.format(of))


class BaseSection(object):

    """Base section."""

    def __init__(self, engine):
        self.engine = engine


def _make_section_getter(section):

    def getter(self,  engine=None, set_engine=True, **extra_params):
        """
        Get {}.

        Parameters
        ----------
        engine : string
            Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
            If ``None``, returns current :attr:`cosmology.engine`.

        set_engine : bool
            Whether to attach returned engine to ``cosmology``.
            (Set ``False`` if e.g. you want to use this engine for a single calculation).

        extra_params : dict
            Extra engine parameters, typically precision parameters.
        """.format(section)
        engine = get_engine(self,engine=engine,set_engine=set_engine,**extra_params)
        toret = getattr(engine,'get_{}'.format(section),None)
        if toret is None:
            raise CosmologyError('Engine {} does not provide {}'.format(engine.__class__.__name__,section))
        return toret()

    return getter


for section in ['background','thermodynamics','primordial','perturbations','harmonic','fourier']:
    setattr(Cosmology,'get_{}'.format(section),_make_section_getter(section))


def compile_params(args):
    """
    Compile parameters ``args``:
    - normalise parameter names
    - perform immediate parameter derivations (e.g. omega => Omega)
    - setup neutrino masses if relevant

    Parameters
    ----------
    args : dict
        Input parameter dictionary, without parameter conflicts.

    Returns
    -------
    params : dict
        Normalised parameter dictionary.

    References
    ----------
    https://github.com/bccp/nbodykit/blob/master/nbodykit/cosmology/cosmology.py
    """
    params = {}
    params.update(args)

    if 'H0' in params:
        params['h'] = params.pop('H0')/100.

    for name,value in args.items():
        if name.startswith('omega'):
            params[name.replace('omega','Omega')] = params.pop(name)/params['h']**2

    def set_alias(params_name, args_name):
        if args_name not in args: return
        # pop because we copied everything
        params[params_name] = params.pop(args_name)

    set_alias('T_cmb', 'T0_cmb')
    set_alias('Omega_cdm', 'Omega0_cdm')
    set_alias('Omega_b', 'Omega0_b')
    set_alias('Omega_k', 'Omega0_k')
    set_alias('Omega_ur', 'Omega0_ur')
    set_alias('Omega_Lambda', 'Omega_lambda')
    set_alias('Omega_Lambda', 'Omega0_lambda')
    set_alias('Omega_Lambda', 'Omega0_Lambda')
    set_alias('Omega_fld', 'Omega0_fld')
    set_alias('Omega_ncdm', 'Omega0_ncdm')
    set_alias('Omega_g', 'Omega0_g')

    if 'ln10^{10}A_s' in params:
        params['A_s'] = np.exp(params.pop('ln10^{10}A_s'))*10**(-10)

    if 'Omega_g' in params:
        params['T_cmb'] = (params.pop('Omega_g')*params['h']**2*constants.rho_crit_kgph_per_mph3/(4./constants.c**3 * constants.Stefan_Boltzmann))**(0.25)

    # no massive neutrinos
    if 'm_ncdm' in params:
        m_ncdm = params.pop('m_ncdm')
        if m_ncdm is None:
            m_ncdm = []

        m_single = np.ndim(m_ncdm) == 0
        if m_single:
            # a single massive neutrino
            m_ncdm = [m_ncdm]

        if isinstance(m_ncdm,(list,np.ndarray)):
            m_ncdm = list(m_ncdm)
        else:
            raise TypeError('m_ncdm should be a list of mass values in eV')

        if 'neutrino_hierarchy' in args:
            neutrino_hierarchy = params.pop('neutrino_hierarchy')
            # Taken from https://github.com/LSSTDESC/CCL/blob/66397c7b53e785ae6ee38a688a741bb88d50706b/pyccl/core.py#L461
            # Sum changes in the lower bounds...
            if neutrino_hierarchy is not None:
                if not m_single:
                    raise CosmologyError('neutrino_hierarchy {} cannot be passed with a list '
                                        'for m_ncdm, only with a sum.'.format(neutrino_hierarchy))
                sum_ncdm = m_ncdm[0]
                if sum_ncdm < 0:
                    raise CosmologyError('Sum of neutrino masses must be positive.')
                # Lesgourges & Pastor 2012, arXiv:1212.6154
                deltam21sq = 7.62e-5

                def solve_newton(m_ncdm, deltam21sq, deltam31sq):
                    # m_ncdm is a starting guess
                    sum_check = sum(m_ncdm)
                    # This is the Newton's method, solving s = m1 + m2 + m3,
                    # with dm2/dm1 = dsqrt(deltam21^2 + m1^2) / dm1, same for m3
                    while (np.abs(sum_ncdm - sum_check) > 1e-15):
                        dsdm1 = 1. + m_ncdm[0] / m_ncdm[1] + m_ncdm[0] / m_ncdm[2]
                        m_ncdm[0] = m_ncdm[0] - (sum_check - sum_ncdm) / dsdm1
                        m_ncdm[1] = np.sqrt(m_ncdm[0]**2 + deltam21sq)
                        m_ncdm[2] = np.sqrt(m_ncdm[0]**2 + deltam31sq)
                        sum_check = sum(m_ncdm)
                    return m_ncdm

                if (neutrino_hierarchy == 'normal'):
                    deltam31sq = 2.55e-3
                    if sum_ncdm**2 < deltam21sq + deltam31sq:
                        raise ValueError('If neutrino_hierarchy is normal, we are using the normal hierarchy and so m_nu must be greater than (~)0.0592')
                    # Split the sum into 3 masses under normal hierarchy, m3 > m2 > m1
                    m_ncdm = [0.,deltam21sq,deltam31sq]
                    solve_newton(m_ncdm,deltam21sq,deltam31sq)

                elif (neutrino_hierarchy == 'inverted'):
                    deltam31sq = -2.43e-3
                    if sum_ncdm**2 < -deltam31sq + deltam21sq - deltam31sq:
                        raise ValueError('If neutrino_hierarchy is inverted, we are using the inverted hierarchy and so m_nu must be greater than (~)0.0978')
                    # Split the sum into 3 masses under inverted hierarchy, m2 > m1 > m3, here ordered as m1, m2, m3
                    m_ncdm = [np.sqrt(-deltam31sq),np.sqrt(-deltam31sq + deltam21sq),1e-5]
                    solve_newton(m_ncdm,deltam21sq,deltam31sq)

                elif (neutrino_hierarchy == 'degenerate'):
                    m_ncdm = [sum_ncdm/3.]*3

                else:
                    raise CosmologyError('Unkown neutrino mass type {}'.format(neutrino_hierarchy))

        if args.get('N_ur',None) is None:
            # Check which of the neutrino species are non-relativistic today
            m_massive = 0.00017 # Lesgourges et al. 2012
            m_ncdm = np.array(m_ncdm)
            N_m_ncdm = np.sum(m_ncdm > m_massive)
            # arxiv: 1812.05995 eq. 84
            N_ur = params.get('N_eff',constants.NEFF) - (N_m_ncdm * params.get('T_ncdm',constants.TNCDM)**4 * (4./11.)**(-4./3.))
            if N_ur < 0.:
                raise ValueError('N_ur and m_ncdm must result in a number of relativistic neutrino species greater than or equal to zero.')
            # Fill an array with the non-relativistic neutrino masses
            m_ncdm = m_ncdm[m_ncdm > m_massive].tolist()
            params['N_ur'] = N_ur

        # number of massive neutrino species
        params['N_ncdm'] = len(m_ncdm)
        params['m_ncdm'] = m_ncdm

    if params.get('z_pk',None) is None:
        # same as pyccl, https://github.com/LSSTDESC/CCL/blob/d2a5630a229378f64468d050de948b91f4480d41/src/ccl_core.c
        from . import interpolator
        params['z_pk'] = interpolator.get_default_z_callable()
    if params.get('modes',None) is None:
        params['modes'] = ['s']
    for name in ['modes','z_pk']:
        if np.ndim(params[name]) == 0:
            params[name] = [params[name]]
    if 0 not in params['z_pk']:
        params['z_pk'].append(0) # in order to normalise CAMB power spectrum

    return params


def merge_params(args, moreargs):
    """
    Merge ``moreargs`` parameters into ``args``.
    ``moreargs`` parameters take priority over those defined in ``args``.

    Parameters
    ----------
    args : dict
        Base parameter dictionary.

    moreargs : dict
        Parameter dictionary to be merged into ``args``.

    Returns
    -------
    args : dict
        Merged parameter dictionary.
    """
    for name in moreargs.keys():
        # pop those conflicting with me from the old pars
        for eq in find_conflicts(name):
            if eq in args: args.pop(eq)

    args.update(moreargs)
    return args


def check_params(args):
    """Check for conflicting parameters in ``args`` parameter dictionary."""
    conf = {}
    for name in args:
        conf[name] = []
        for eq in find_conflicts(name):
            if eq == name: continue
            if eq in args: conf[name].append(eq)

    for name in conf:
        if conf[name]:
            raise CosmologyError('Conflicted parameters are given: {}'.format(conf))


def find_conflicts(name):
    """
    Return conflicts corresponding to input parameter name.

    Parameters
    ---------
    name : string
        Parameter name.

    Returns
    -------
    conflicts : tuple
        Conflicting parameter names.
    """
    # dict that defines input parameters that conflict with each other
    conflicts = [('h', 'H0'),
                 ('T_cmb', 'Omega_g', 'omega_g', 'Omega0_g'),
                 ('Omega_b', 'omega_b', 'Omega0_b'),
                 #('Omega_fld', 'Omega0_fld'),
                 #('Omega_Lambda', 'Omega0_Lambda'),
                 ('N_ur', 'Omega_ur', 'omega_ur', 'Omega0_ur'),
                 ('Omega_cdm', 'omega_cdm', 'Omega0_cdm'),
                 #('m_ncdm', 'Omega_ncdm', 'omega_ncdm', 'Omega0_ncdm'),
                 ('A_s', 'ln10^{10}A_s', 'sigma8'),
                 ('tau_reio','z_reio')
                ]

    for conf in conflicts:
        if name in conf:
            return conf
    return ()


def Planck2018FullFlatLCDM(engine=None, extra_params=None, **params):
    """
    Initialise :class:`Cosmology` based on Planck2018 TT, TE, EE, lowE, lensing and BAO data.

    Parameters
    ----------
    engine : string
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`cosmology.engine`.

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
