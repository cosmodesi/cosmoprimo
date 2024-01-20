"""Cosmology class"""

import os
import sys

import numpy as np
from scipy import integrate

from .utils import BaseClass
from . import utils, constants


_Sections = ['Background', 'Thermodynamics', 'Primordial', 'Perturbations', 'Transfer', 'Harmonic', 'Fourier']


class class_or_instancemethod(classmethod):
    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)


def _deepeq(obj1, obj2):
    # Deep equality test
    if type(obj2) is type(obj1):
        if isinstance(obj1, dict):
            if obj2.keys() == obj1.keys():
                return all(_deepeq(obj1[name], obj2[name]) for name in obj1)
        elif isinstance(obj1, (tuple, list)):
            if len(obj2) == len(obj1):
                return all(_deepeq(o1, o2) for o1, o2 in zip(obj1, obj2))
        elif isinstance(obj1, np.ndarray):
            return np.all(obj2 == obj1)
        else:
            return obj2 == obj1
    return False


class CosmologyError(Exception):

    """Exception raised by :class:`Cosmology`."""


class CosmologyInputError(CosmologyError):

    """Exception raised when error in input parameters."""


class CosmologyComputationError(CosmologyError):

    """Exception raised when error in cosmology computation."""


def is_sequence(item):
    return isinstance(item, (tuple, list))


def _compute_ncdm_momenta(T_eff, m, z=0, epsrel=1e-7, out='rho'):
    r"""
    Return momenta of non-CDM components (massive neutrinos)
    by integrating over the phase-space distribution (frozen since CMB).

    Parameters
    ----------
    T_eff : float
        Effective temperature; typically T_cmb * T_ncdm_over_cmb.

    m : float
        Mass in :math:`\mathrm{eV}`.

    z : float, default=0
        Redshift.

    epsrel : float, default=1e-7
        Relative precision (for :meth:`scipy.integrate.quad` integration).

    out : string, default='rho'
        If 'rho', return energy density.
        If 'drhodm', return derivative of energy density w.r.t. to mass ``m``.
        If 'p', return pressure.

    Returns
    -------
    out : float
        Required momentum, in units of :math:`10^{10} M_{\odot} / \mathrm{Mpc}^{3}` (/ :math:`\mathrm{eV}` if ``out`` is 'drhodm')
    """
    a = 1. / (1. + z)
    over_T = constants.electronvolt / (constants.Boltzmann * (T_eff / a))
    m2_over_T2 = (m * over_T) ** 2
    m_over_T2 = m * over_T ** 2

    if out == 'rho':
        def phase_space_integrand(q):
            return q**2 * np.sqrt(q**2 + m2_over_T2) / (1. + np.exp(q))
    elif out == 'drhodm':
        def phase_space_integrand(q):
            return m_over_T2 * q**2 / np.sqrt(q**2 + m2_over_T2) / (1. + np.exp(q))
    elif out == 'p':
        def phase_space_integrand(q):
            return 1. / 3. * q**4 / np.sqrt(q**2 + m2_over_T2) / (1. + np.exp(q))
    else:
        raise ValueError('Cannot compute ncdm momenta {}; choices are ["rho", "drhodm", "p"]', out)
    # upper bound of 100 enough (10^⁻16 error)
    toret = integrate.quad(phase_space_integrand, 0, 100, epsrel=epsrel)[0] / (7. * np.pi**4 / 120.)
    return 7. / 8. * 4 / constants.c**3 * constants.Stefan_Boltzmann * (T_eff / a)**4 * toret / (1e10 * constants.msun) * constants.megaparsec**3


def _compute_rs_cosmomc(omega_b, omega_m, hubble_function):

    """Return sound horizon in proper Mpc, and redshift of the last scattering surface in the CosmoMC approximation."""

    zstar = 1048 * (1 + 0.00124 * omega_b**(-0.738))\
            * (1 + (0.0783 * omega_b**(-0.238) / (1 + 39.5 * omega_b**0.763))\
            * omega_m**(0.560 / (1 + 21.1 * omega_b**1.81)))

    astart = 1e-8
    astar = 1. / (1 + zstar)
    atol = 1e-6

    def dtauda(a):
        return 1. / (a**2 * hubble_function(1 / a - 1.) / (constants.c / 1e3))

    def dsoundda_approx(a):
        # https://github.com/cmbant/CAMB/blob/758c6c2359764297e332ee2108df599506a754c3/fortran/results.f90#L1138
        R = 3e4 * a * omega_b
        cs = (3 * (1 + R))**(-0.5)
        return dtauda(a) * cs

    return integrate.romberg(dsoundda_approx, astart, astar, tol=atol, rtol=atol, vec_func=True), zstar


class BaseCosmology(BaseClass):

    _default_cosmological_parameters = dict()
    _default_calculation_parameters = dict()
    _conflict_parameters = []

    def __init__(self, extra_params=None, **params):
        """
        Initialize engine.

        Parameters
        ----------
        extra_params : dict, default=None
            Extra engine parameters, typically precision parameters.

        params : dict
            Engine parameters.
        """
        check_params(params, conflicts=self.__class__._conflict_parameters)
        self._derived = {}
        self._input_params = merge_params(self.get_default_params(include_conflicts=False), params, conflicts=self.__class__._conflict_parameters)
        self._params = self._compile_params(self._input_params)
        self._extra_params = dict(extra_params or {})

    @classmethod
    def get_default_params(cls, of=None, include_conflicts=True):
        """
        Return default input parameters.

        Parameters
        ----------
        of : string, default=None
            One of ['cosmology', 'calculation'].
            If ``None``, returns all parameters.

        include_conflicts : bool, default=True
            Whether to include conflicting parameters (then all accepted parameters).

        Returns
        -------
        params : dict
            Dictionary of default parameters.
        """
        if of is None:
            toret = cls.get_default_params(of='cosmology', include_conflicts=include_conflicts)
            toret.update(cls.get_default_params(of='calculation', include_conflicts=include_conflicts))
            return toret

        def _include_conflicts(params):
            """Add in conflicting parameters to input ``params`` dictionay (in-place operation)."""
            for name in list(params.keys()):
                for conf in find_conflicts(name, conflicts=cls._conflict_parameters):
                    params[conf] = params[name]

        if of == 'cosmology':
            toret = cls._default_cosmological_parameters.copy()
            if include_conflicts: _include_conflicts(toret)
            return toret
        if of == 'calculation':
            toret = cls._default_calculation_parameters.copy()
            if include_conflicts: _include_conflicts(toret)
            return toret
        raise CosmologyInputError('No default parameters for {}'.format(of))

    def get_params(self, of='base'):
        """
        Return parameters.

        Parameters
        ----------
        of : string, default='base'
            One of ['cosmology', 'calculation', 'base', 'derived', 'extra', 'all'].
            If ``all``, returns base, derived and extra parameters.

        Returns
        -------
        params : dict
            Dictionary of parameters.
        """
        if of == 'derived':
            return dict(self._derived)
        if of == 'extra':
            return dict(self._extra_params)
        toret = dict(self._params)
        if of == 'base':
            return toret
        if of in ['cosmology', 'calculation']:
            params = self.get_default_params(of=of)
            toret = {name: toret.get(name, value) for name, value in params.items()}
            return toret
        if of == 'all':
            toret.update(self.get_params(of='derived'))
            toret.update(self.get_params(of='extra'))
            return toret
        raise CosmologyInputError('No parameters for {}'.format(of))

    @classmethod
    def _compile_params(cls, params):
        """Return input parameters in a standard basis."""
        return dict(params)

    def __getitem__(self, name):
        """Return an input (or easily derived) parameter."""
        return self.get(name)

    def get(self, *args, **kwargs):
        """Return an input (or easily derived) parameter."""
        if len(args) == 1:
            name = args[0]
            has_default = 'default' in kwargs
            default = kwargs.get('default', None)
        else:
            name, default = args
            has_default = True
        params = self.get_params(of='base')
        derived = self.get_params(of='derived')
        try:
            if name in params:
                return params[name]
            if name in derived:
                return derived[name]
            if name.startswith('omega'):
                return self.get('O' + name[1:]) * params['h']**2
            if name == 'H0':
                return params['h'] * 100
            if name in ['logA', 'ln10^{10}A_s', 'ln10^10A_s', 'ln_A_s_1e10']:
                return np.log(1e10 * params['A_s'])
            # if name == 'rho_crit':
            #     return constants.rho_crit_Msunph_per_Mpcph3
            if name == 'Omega_g':
                rho = params['T_cmb']**4 * 4. / constants.c**3 * constants.Stefan_Boltzmann  # density, kg/m^3
                return rho / (self.get('h')**2 * constants.rho_crit_kgph_per_mph3)
            if name == 'T_ur':
                return params['T_cmb'] * (4. / 11.)**(1. / 3.)
            if name == 'T_ncdm':
                return np.array(params['T_ncdm_over_cmb']) * params['T_cmb']
            if name == 'Omega_ur':
                rho = params['N_ur'] * 7. / 8. * self.get('T_ur')**4 * 4. / constants.c**3 * constants.Stefan_Boltzmann  # density, kg/m^3
                return rho / (self.get('h')**2 * constants.rho_crit_kgph_per_mph3)
            if name == 'Omega_r':
                rho = (params['T_cmb']**4 + params['N_ur'] * 7. / 8. * self.get('T_ur')**4) * 4. / constants.c**3 * constants.Stefan_Boltzmann
                return rho / (self.get('h')**2 * constants.rho_crit_kgph_per_mph3) + self.get('Omega_pncdm_tot')
            if name == 'm_ncdm_tot':
                return sum(params['m_ncdm'])
            if name == 'Omega_ncdm':
                derived['Omega_ncdm'] = self._get_rho_ncdm(z=0) / constants.rho_crit_Msunph_per_Mpcph3
                return derived['Omega_ncdm']
            if name == 'Omega_ncdm_tot':
                return np.sum(self.get('Omega_ncdm'))
            if name == 'Omega_pncdm':
                derived['Omega_pncdm'] = 3. * self._get_p_ncdm(z=0) / constants.rho_crit_Msunph_per_Mpcph3
                return derived['Omega_pncdm']
            if name == 'Omega_pncdm_tot':
                return np.sum(self.get('Omega_pncdm'))
            if name == 'Omega_m':
                return self.get('Omega_b') + self.get('Omega_cdm') + self.get('Omega_ncdm_tot') - self.get('Omega_pncdm_tot')
            if name == 'Omega_de':
                return 1. - sum(self.get(name) for name in ['Omega_cdm', 'Omega_b', 'Omega_g', 'Omega_ur', 'Omega_ncdm_tot', 'Omega_k'])
            if name == 'Omega_Lambda':
                if self._has_fld: return 0.
                return self.get('Omega_de')
            if name == 'Omega_fld':
                if self._has_fld: return self.get('Omega_de')
                return 0.
            if name == 'K':
                return - 100.**2 / (constants.c / 1e3)**2 * params['Omega_k']  # in (h / Mpc)^2
            if name == 'N_ncdm':
                return len(params['m_ncdm'])
            #if name == 'N_ur':
            #    return params['N_eff'] - sum(T_ncdm_over_cmb**4 * (4. / 11.)**(-4. / 3.) for T_ncdm_over_cmb in params['T_ncdm_over_cmb'])
            if name == 'N_eff':
                return sum(T_ncdm_over_cmb**4 * (4. / 11.)**(-4. / 3.) for T_ncdm_over_cmb in params['T_ncdm_over_cmb']) + params['N_ur']
            if name == 'theta_cosmomc':
                ba = self.get_background()
                rs, zstar = _compute_rs_cosmomc(self['omega_b'], self['omega_m'], ba.hubble_function)
                derived['theta_cosmomc'] = rs * ba.h / ba.comoving_angular_distance(zstar)
                return derived['theta_cosmomc']
            if name == 'theta_MC_100':
                return self.get('theta_cosmomc') * 100.
        except KeyError:
            pass
        if has_default:
            return default
        raise CosmologyError('Parameter {} not found.'.format(name))

    @property
    def _has_fld(self):
        return (self._params['w0_fld'], self._params['wa_fld'], self._params['cs2_fld']) != (-1, 0., 1.)

    def _get_rho_ncdm(self, z=0, species=None, epsrel=1e-7):
        r"""
        Return energy density of non-CDM components (massive neutrinos) for each species by integrating over the phase-space distribution (frozen since CMB),
        including non-relativistic (contributing to :math:`\Omega_{m}`) and relativistic (contributing to :math:`\Omega_{r}`) components.
        Usually close to :math:`\sum m/(93.14 h^{2})` by definition of T_ncdm_over_cmb.

        Parameters
        ----------
        z : float, default=0
            Redshift.

        epsrel : float, default=1e-7
            Relative precision (for :meth:`scipy.integrate.quad` integration).

        Returns
        -------
        rho_ncdm : array
            Energy density, in units of :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`.
        """
        h2 = self['h']**2
        T_cmb, T_ncdm_over_cmb, m_ncdm = self['T_cmb'], self['T_ncdm_over_cmb'], self['m_ncdm']

        def compute(T_ncdm_over_cmb, m_ncdm):
            return _compute_ncdm_momenta(T_cmb * T_ncdm_over_cmb, m_ncdm, z=z, epsrel=epsrel, out='rho') / (1 + z)**3 / h2

        if species is None:
            species = list(range(len(m_ncdm)))

        if is_sequence(species):
            return np.asarray([compute(T_ncdm_over_cmb[s], m_ncdm[s]) for s in species])

        return compute(self['T_ncdm_over_cmb'][species], self['m_ncdm'][species])

    def _get_p_ncdm(self, z=0, species=None, epsrel=1e-7):
        r"""
        Return pressure of non-CDM components (massive neutrinos) for each species by integrating over the phase-space distribution (frozen since CMB).

        Parameters
        ----------
        z : float, default=0
            Redshift.

        epsrel : float, default=1e-7
            Relative precision (for :meth:`scipy.integrate.quad` integration).

        Returns
        -------
        p_ncdm : array
            Pressure, in units of :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`.
        """
        h2 = self['h']**2
        T_cmb, T_ncdm_over_cmb, m_ncdm = self['T_cmb'], self['T_ncdm_over_cmb'], self['m_ncdm']

        def compute(T_ncdm_over_cmb, m_ncdm):
            return _compute_ncdm_momenta(T_cmb * T_ncdm_over_cmb, m_ncdm, z=z, epsrel=epsrel, out='p') / (1 + z)**3 / h2

        if species is None:
            species = list(range(len(m_ncdm)))

        if is_sequence(species):
            return np.asarray([compute(T_ncdm_over_cmb[s], m_ncdm[s]) for s in species])

        return compute(self['T_ncdm_over_cmb'][species], self['m_ncdm'][species])

    def __eq__(self, other):
        r"""Is ``other`` same as ``self``?"""
        return type(other) == type(self) and _deepeq(other._params, self._params) and _deepeq(other._extra_params, self._extra_params)


class RegisteredEngine(type(BaseCosmology)):

    """Metaclass registering :class:`BaseEngine`-derived classes."""

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls
        return cls


class BaseEngine(BaseCosmology, metaclass=RegisteredEngine):

    """Base engine for cosmological calculation."""
    name = 'base'

    def __init__(self, extra_params=None, **params):
        """
        Initialize engine.

        Parameters
        ----------
        extra_params : dict, default=None
            Extra engine parameters, typically precision parameters.

        params : dict
            Engine parameters.
        """
        super(BaseEngine, self).__init__(extra_params=extra_params, **params)
        self._Sections = {}
        module = sys.modules[self.__class__.__module__]
        for name in _Sections:
            section = getattr(module, name, None)
            if section is not None:
                self._Sections[name.lower()] = section
        self._sections = {}

    def _get_A_s_fid(self):
        r"""First guess for power spectrum amplitude :math:`A_{s}` (given input :math:`\sigma_{8}`)."""
        # https://github.com/lesgourg/class_public/blob/4724295b527448b00faa28bce973e306e0e82ef5/source/input.c#L1161
        if 'A_s' in self._params:
            return self._params['A_s']
        return 2.43e-9 * (self['sigma8'] / 0.87659)**2

    def _get_sigma8_fid(self):
        r"""First guess for power spectrum amplitude :math:`\sigma_{8}` (given input :math:`A_s`)."""
        # https://github.com/lesgourg/class_public/blob/4724295b527448b00faa28bce973e306e0e82ef5/source/input.c#L1161
        if 'sigma8' in self._params:
            return self._params['sigma8']
        return (self['A_s'] / 2.43e-9)**0.5 * 0.87659

    def _rescale_sigma8(self):
        """Rescale perturbative quantities to match input sigma8."""
        if hasattr(self, '_rsigma8'):
            return self._rsigma8
        self._rsigma8 = 1.
        if 'sigma8' in self._params:
            self._sections.clear()  # to remove fourier with potential _rsigma8 != 1
            #fo = self.get_fourier()
            self._rsigma8 = self._params['sigma8'] / self.get_fourier().sigma8_m
            self._sections.clear()  # to reinitialize fourier with correct _rsigma8
        return self._rsigma8


def _make_section_getter(section):

    def getter(self):
        name = section.lower()
        if name not in self._sections:
            self._sections[name] = self._Sections[name](self)
        return self._sections[name]

    getter.__doc__ = """Return :class:`{}` calculations.""".format(section)

    return getter


for section in _Sections:
    setattr(BaseEngine, 'get_{}'.format(section.lower()), _make_section_getter(section))


def get_engine(engine):
    """
    Return engine (class) for cosmological calculation.

    Parameters
    ----------
    engine : type, string
        Engine or one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants', 'bbks'].

    Returns
    -------
    engine : BaseEngine
    """
    if isinstance(engine, str):
        engine = engine.lower()
        if engine == 'class':
            from . import classy
        elif engine == 'camb':
            from . import camb
        elif engine == 'isitgr':
            from . import isitgr
        elif engine == 'eisenstein_hu':
            from . import eisenstein_hu
        elif engine == 'eisenstein_hu_nowiggle':
            from . import eisenstein_hu_nowiggle
        elif engine == 'eisenstein_hu_nowiggle_variants':
            from . import eisenstein_hu_nowiggle_variants
        elif engine == 'bbks':
            from . import bbks
        elif engine == 'astropy':
            from . import astropy
        elif engine == 'tabulated':
            from . import tabulated

        try:
            engine = BaseEngine._registry[engine]
        except KeyError:
            raise CosmologyInputError('Unknown engine {}.'.format(engine))

    if isinstance(engine, BaseEngine):
        engine = engine.__class__

    return engine


def _get_cosmology_engine(cosmology, engine=None, set_engine=True, **extra_params):
    """
    Return engine for cosmological calculation.

    Parameters
    ----------
    cosmology : Cosmology
        Current cosmology.

    engine : BaseEngine, string, default=None
        Engine or one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`Cosmology.engine`.

    set_engine : bool, default=True
        Whether to attach returned engine to ``cosmology``.
        (Set ``False`` if e.g. you want to use this engine for a single calculation).

    extra_params : dict
        Extra engine parameters, typically precision parameters.

    Returns
    -------
    engine : BaseEngine
    """
    if engine is None:
        if cosmology._engine is None:
            raise CosmologyInputError('Please provide an engine')
        engine = cosmology._engine
    elif not isinstance(engine, BaseEngine):
        engine = get_engine(engine)(**cosmology._params, extra_params=extra_params)
    if set_engine:
        cosmology._engine = engine
    return engine


def _make_section_getter(section):

    def getter(cosmology, engine=None, set_engine=True, **extra_params):
        engine = _get_cosmology_engine(cosmology, engine=engine, set_engine=set_engine, **extra_params)
        return getattr(engine, 'get_{}'.format(section.lower()))()

    getter.__doc__ = """
    Return :class:`{}` calculations.

    Parameters
    ----------
    cosmology : Cosmology
        Current cosmology.

    engine : string, default=None
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`Cosmology.engine`.

    set_engine : bool, default=True
        Whether to attach returned engine to ``cosmology``
        (Set ``False`` if e.g. you want to use this engine for a single calculation).

    extra_params : dict
        Extra engine parameters, typically precision parameters.

    Returns
    -------
    engine : BaseEngine
    """.format(section)

    return getter


for section in _Sections:
    globals()[section] = _make_section_getter(section)


@utils.addproperty('engine')
class Cosmology(BaseCosmology):

    """Cosmology, defined as a set of parameters (and possibly a current engine attached to it)."""

    _default_cosmological_parameters = dict(h=0.7, Omega_cdm=0.25, Omega_b=0.05, Omega_k=0., sigma8=0.8, k_pivot=0.05, n_s=0.96, alpha_s=0., beta_s=0.,
                                            r=0., n_t='scc', alpha_t='scc', T_cmb=constants.TCMB,
                                            m_ncdm=None, neutrino_hierarchy=None, T_ncdm_over_cmb=constants.TNCDM_OVER_CMB, N_eff=constants.NEFF,
                                            tau_reio=0.06, reionization_width=0.5, A_L=1.0, w0_fld=-1., wa_fld=0., cs2_fld=1.)
    _default_calculation_parameters = dict(non_linear='', modes='s', lensing=False, z_pk=None, kmax_pk=10., ellmax_cl=2500, YHe='BBN', use_ppf=True)
    _conflict_parameters = [('h', 'H0'),
                            ('T_cmb', 'Omega_g', 'omega_g', 'Omega0_g'),
                            ('Omega_b', 'omega_b', 'Omega0_b'),
                            # ('Omega_fld', 'Omega0_fld'),
                            # ('Omega_Lambda', 'Omega0_Lambda'),
                            ('Omega_k', 'omega_k', 'Omega0_k'),
                            ('N_ur', 'Omega_ur', 'omega_ur', 'Omega0_ur', 'N_eff'),
                            ('Omega_cdm', 'omega_cdm', 'Omega0_cdm', 'Omega_c', 'omega_c', 'Omega_m', 'omega_m', 'Omega0_m'),
                            ('m_ncdm', 'Omega_ncdm', 'omega_ncdm', 'Omega0_ncdm'),
                            ('A_s', 'logA', 'ln10^{10}A_s', 'ln10^10A_s', 'ln_A_s_1e10', 'sigma8'),
                            ('tau_reio', 'z_reio')]

    def __init__(self, engine=None, extra_params=None, **params):
        r"""
        Initialize :class:`Cosmology`.

        Note
        ----
        If ``Omega_m`` (or ``omega_m``) is provided, ``Omega_cdm`` is infered by subtracting ``Omega_b`` and the non-relativistic part of ``Omega_ncdm`` from ``Omega_m``.
        Massive neutrinos can be provided e.g. through ``m_ncdm`` or ``Omega_ncdm``/``omega_ncdm`` with their temperatures w.r.t. CMB ``T_ncdm_over_cmb``.
        In the case of ``Omega_ncdm``, the neutrino energy density (see :func:`_compute_ncdm_momenta`) will be inverted to recover ``m_ncdm``.
        If a single value for ``m_ncdm`` or ``Omega_ncdm`` is provided, ``neutrino_hierarchy`` can be set to ``None`` (default, single massive neutrino)
        or 'normal', 'inverted', 'degenerate' (all neutrinos with same mass), which will determine the masses of the 3 neutrinos.
        If the number of relativistic species ``N_ur`` is not provided (or ``None``), it will be determined
        from the desired effective number of neutrinos ``N_eff`` (typically kept at 3.044 for 3 neutrinos whatever ``m_ncdm`` or ``Omega_ncdm``/``omega_ncdm``)
        and the number of massless neutrinos (:math:`m \leq 0.00017 \; \mathrm{eV}`), which are then removed from the list ``m_ncdm``.
        Parameter ``Omega_ncdm``/``omega_ncdm`` (accessed as ``cosmo['Omega_ncdm']``/``cosmo['omega_ncdm']``)
        will always provide the total energy density of neutrinos (single value).
        The pivot scale ``k_pivot`` is in :math:`\mathrm{Mpc}^{-1}`.
        If 'non_linear' is required, we recommend "mead" for class and camb matching.

        Parameters
        ----------
        engine : string, default=None
            Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
            If ``None``, no engine is set.

        extra_params : dict, default=None
            Extra engine parameters, typically precision parameters.

        params : dict
            Cosmological and calculation parameters which take priority over the default ones.
        """
        check_params(params, conflicts=self._conflict_parameters)
        self._derived = {}
        self._engine = None
        self._input_params = merge_params(self.get_default_params(include_conflicts=False), params, conflicts=self._conflict_parameters)
        self._params = self._compile_params(self._input_params)
        self._extra_params = {}
        if engine is not None:
            self.set_engine(engine, **(extra_params or {}))

    @class_or_instancemethod
    def get_default_params(cls, of=None, include_conflicts=True):
        """
        Return default input parameters.

        Parameters
        ----------
        of : string, default=None
            One of ['cosmology', 'calculation'].
            If ``None``, returns all parameters.

        include_conflicts : bool, default=True
            Whether to include conflicting parameters (then all accepted parameters).

        Returns
        -------
        params : dict
            Dictionary of default parameters.
        """
        toret = super().get_default_params(of=of, include_conflicts=include_conflicts)
        engine = getattr(cls, '_engine', None)
        if engine is not None:  # cls is self
            toret.update(engine.get_default_params(of=of, include_conflicts=include_conflicts))
        return toret

    @class_or_instancemethod
    def get_default_parameters(cls):
        import warnings
        warnings.warn('get_default_parameters is deprecated, use get_default_params')
        return cls.get_default_params()

    def get_params(self, of='base'):
        """
        Return parameters.

        Parameters
        ----------
        of : string, default='input'
            One of ['cosmology', 'calculation', 'base', 'derived', 'extra', 'all'].
            If ``all``, returns base, derived and extra parameters.

        Returns
        -------
        params : dict
            Dictionary of parameters.
        """
        toret = super().get_params(of=of)
        if self._engine is not None:
            toret.update(self._engine.get_params(of=of))
        return toret

    @classmethod
    def _compile_params(cls, args):
        """
        Compile parameters ``args``:

        - normalise parameter names
        - perform immediate parameter derivations (e.g. omega => Omega)
        - set neutrino masses if relevant
        - if Omega_m provided, compute Omega_cdm from Omega_b and non-relativistic ncdm

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
            params['h'] = params.pop('H0') / 100.

        h = params['h']
        for name, value in args.items():
            if name.startswith('omega'):
                omega = params.pop(name)
                Omega = np.array(omega) / h**2  # array to cope with tuple, lists for e.g. omega_ncdm
                params_name = name.replace('omega', 'Omega')
                if params_name in params: raise CosmologyInputError('found both {} and {}, must be added to _conflict_parameters'.format(name, params_name))
                params[params_name] = Omega

        def set_alias(params_name, args_name):
            if args_name not in args: return
            # pop because we copied everything
            if params_name in params: raise CosmologyInputError('found both {} and {}, must be added to _conflict_parameters'.format(args_name, params_name))
            params[params_name] = params.pop(args_name)

        set_alias('Omega_m', 'Omega0_m')
        set_alias('Omega_cdm', 'Omega0_cdm')
        set_alias('Omega_cdm', 'Omega_c')
        set_alias('Omega_ncdm', 'Omega0_ncdm')
        set_alias('Omega_b', 'Omega0_b')
        set_alias('Omega_k', 'Omega0_k')
        set_alias('Omega_ur', 'Omega0_ur')
        # set_alias('Omega_Lambda', 'Omega_lambda')
        # set_alias('Omega_Lambda', 'Omega0_lambda')
        # set_alias('Omega_Lambda', 'Omega0_Lambda')
        set_alias('Omega_fld', 'Omega0_fld')
        set_alias('Omega_ncdm', 'Omega0_ncdm')
        set_alias('T_cmb', 'T0_cmb')
        set_alias('Omega_g', 'Omega0_g')
        set_alias('logA', 'ln10^10A_s')
        set_alias('logA', 'ln10^{10}A_s')
        set_alias('logA', 'ln_A_s_1e10')

        if 'logA' in params:
            params['A_s'] = np.exp(params.pop('logA')) * 10**(-10)

        if 'Omega_g' in params:
            params['T_cmb'] = (params.pop('Omega_g') * h**2 * constants.rho_crit_kgph_per_mph3 / (4. / constants.c**3 * constants.Stefan_Boltzmann))**(0.25)

        def _make_list(li, name):
            if isinstance(li, (tuple, list, np.ndarray)):
                return list(li)
            raise TypeError('{} must be a list'.format(name))

        T_ncdm_over_cmb = params.get('T_ncdm_over_cmb', None)
        if T_ncdm_over_cmb in (None, []):
            T_ncdm_over_cmb = constants.TNCDM_OVER_CMB

        if 'm_ncdm' in params:
            m_ncdm = params.pop('m_ncdm')
            Omega_ncdm = None
        else:
            if 'Omega_ncdm' in params:
                Omega_ncdm = params.pop('Omega_ncdm')
                single_ncdm = False
                if Omega_ncdm is None:
                    Omega_ncdm = []
                else:
                    single_ncdm = np.ndim(Omega_ncdm) == 0
                if single_ncdm:  # a single massive neutrino
                    Omega_ncdm = [Omega_ncdm]
                Omega_ncdm = _make_list(Omega_ncdm, 'Omega_ncdm')
                if np.ndim(T_ncdm_over_cmb) == 0:
                    T_ncdm_over_cmb = [T_ncdm_over_cmb] * len(Omega_ncdm)
                T_ncdm_over_cmb = _make_list(T_ncdm_over_cmb, 'T_ncdm_over_cmb')
                if len(T_ncdm_over_cmb) != len(Omega_ncdm):
                    raise TypeError('T_ncdm_over_cmb and Omega_ncdm must be of same length')
                m_ncdm = []
                h = params['h']

                def solve_newton(omega_ncdm, m, T_eff):
                    # m is a starting guess
                    omega_check = _compute_ncdm_momenta(T_eff, m, z=0, out='rho') / constants.rho_crit_Msunph_per_Mpcph3

                    while (np.abs(omega_ncdm - omega_check) > 1e-15):
                        domegadm = _compute_ncdm_momenta(T_eff, m, z=0, out='drhodm') / constants.rho_crit_Msunph_per_Mpcph3
                        # domegadm = 1./93.14 # this approximation works as well
                        m = m + (omega_ncdm - omega_check) / domegadm
                        omega_check = _compute_ncdm_momenta(T_eff, m, z=0, out='rho') / constants.rho_crit_Msunph_per_Mpcph3

                    return m

                for Omega, T in zip(Omega_ncdm, T_ncdm_over_cmb):
                    if Omega == 0.:
                        m_ncdm.append(0.)
                    else:
                        T_ncdm = params['T_cmb'] * T
                        m = solve_newton(Omega * h**2, Omega * h**2 * 93.14, T_ncdm)
                        # print(m, Omega * h**2 * 93.14)
                        m_ncdm.append(m)

                if single_ncdm: m_ncdm = m_ncdm[0]

            else:
                m_ncdm = []

        single_ncdm = False
        if m_ncdm is None:
            m_ncdm = []
        else:
            single_ncdm = np.ndim(m_ncdm) == 0
        if single_ncdm:  # a single massive neutrino
            m_ncdm = [m_ncdm]

        m_ncdm = _make_list(m_ncdm, 'm_ncdm')

        if np.ndim(T_ncdm_over_cmb) == 0:
            T_ncdm_over_cmb = [T_ncdm_over_cmb] * len(m_ncdm)
        T_ncdm_over_cmb = _make_list(T_ncdm_over_cmb, 'T_ncdm_over_cmb')
        if len(T_ncdm_over_cmb) != len(m_ncdm):
            raise TypeError('T_ncdm_over_cmb and m_ncdm must be of same length')

        if 'neutrino_hierarchy' in params:
            neutrino_hierarchy = params.pop('neutrino_hierarchy')
            # Taken from https://github.com/LSSTDESC/CCL/blob/66397c7b53e785ae6ee38a688a741bb88d50706b/pyccl/core.py#L461
            # Sum changes in the lower bounds...
            if neutrino_hierarchy is not None:
                if not single_ncdm:
                    raise CosmologyInputError('neutrino_hierarchy {} cannot be passed with a list '
                                            'for m_ncdm, only with a sum.'.format(neutrino_hierarchy))
                sum_ncdm = m_ncdm[0]
                if sum_ncdm < 0:
                    raise CosmologyInputError('Sum of neutrino masses must be positive.')
                # Lesgourges & Pastor 2012, arXiv:1212.6154
                #deltam21sq = 7.62e-5
                # https://arxiv.org/pdf/1907.12598.pdf
                deltam21sq = 7.39e-5

                def solve_newton(sum_ncdm, m_ncdm, deltam21sq, deltam31sq):
                    # m_ncdm is a starting guess
                    sum_check = sum(m_ncdm)
                    # This is the Newton's method, solving s = m1 + m2 + m3,
                    # with dm2/dm1 = dsqrt(deltam21^2 + m1^2) / dm1 = m1/m2, similarly for m3
                    while (np.abs(sum_ncdm - sum_check) > 1e-15):
                        dsdm1 = 1. + m_ncdm[0] / m_ncdm[1] + m_ncdm[0] / m_ncdm[2]
                        m_ncdm[0] = m_ncdm[0] + (sum_ncdm - sum_check) / dsdm1
                        m_ncdm[1] = np.sqrt(m_ncdm[0]**2 + deltam21sq)
                        m_ncdm[2] = np.sqrt(m_ncdm[0]**2 + deltam31sq)
                        sum_check = sum(m_ncdm)
                    return m_ncdm

                if (neutrino_hierarchy == 'normal'):
                    #deltam31sq = 2.55e-3
                    deltam31sq = 2.525e-3
                    if sum_ncdm**2 < deltam21sq + deltam31sq:
                        raise ValueError('If neutrino_hierarchy is normal, we are using the normal hierarchy and so m_nu must be greater than (~)0.0592')
                    # Split the sum into 3 masses under normal hierarchy, m3 > m2 > m1
                    m_ncdm = [0., deltam21sq, deltam31sq]
                    solve_newton(sum_ncdm, m_ncdm, deltam21sq, deltam31sq)

                elif (neutrino_hierarchy == 'inverted'):
                    #deltam31sq = -2.43e-3
                    deltam32sq = -2.512e-3
                    deltam31sq = deltam32sq + deltam21sq
                    if sum_ncdm**2 < -deltam31sq - deltam32sq:
                        raise ValueError('If neutrino_hierarchy is inverted, we are using the inverted hierarchy and so m_nu must be greater than (~)0.0978')
                    # Split the sum into 3 masses under inverted hierarchy, m2 > m1 > m3, here ordered as m1, m2, m3
                    m_ncdm = [np.sqrt(-deltam31sq), np.sqrt(-deltam32sq), 1e-5]
                    solve_newton(sum_ncdm, m_ncdm, deltam21sq, deltam31sq)

                elif (neutrino_hierarchy == 'degenerate'):
                    m_ncdm = [sum_ncdm / 3.] * 3

                else:
                    raise CosmologyInputError('Unkown neutrino mass type {}'.format(neutrino_hierarchy))

                T_ncdm_over_cmb = [T_ncdm_over_cmb[0]] * 3

        N_ur = params.pop('N_ur', None)

        if 'Omega_ur' in params:
            T_ur = params['T_cmb'] * (4. / 11.)**(1. / 3.)
            rho = 7. / 8. * 4. / constants.c**3 * constants.Stefan_Boltzmann * T_ur**4  # density, kg/m^3
            N_ur = params.pop('Omega_ur') / (rho / (h**2 * constants.rho_crit_kgph_per_mph3))

        # Check which of the neutrino species are non-relativistic today
        m_massive = 0.  #0.00017  # Lesgourges et al. 2012
        m_ncdm = np.array(m_ncdm)
        T_ncdm_over_cmb = np.array(T_ncdm_over_cmb)
        mask_m = m_ncdm >= m_massive
        # Fill an array with the non-relativistic neutrino masses
        m_ncdm = m_ncdm[mask_m].tolist()
        T_ncdm_over_cmb = T_ncdm_over_cmb[mask_m].tolist()
        # arxiv: 1812.05995 eq. 84
        N_eff = params.pop('N_eff', constants.NEFF)
        # We remove massive neutrinos
        if N_ur is None:
            N_ur = N_eff - sum(T_ncdm_over_cmb**4 * (4. / 11.)**(-4. / 3.) for T_ncdm_over_cmb in T_ncdm_over_cmb)
            # Which is just the high-redshift limit of what is below; leaving it there for clarity
            # N_eff = (rho_r / rho_g - 1) / (7. / 8. * (4. / 11.)**(4. / 3.))  # as defined in class_public https://github.com/lesgourg/class_public/blob/aa92943e4ab86b56970953589b4897adf2bd0f99/source/background.c#L2051
            # with rho_r = rho_g + rho_ur + 3. * pncdm and rho_ur = 7. / 8. * (4. / 11.)**(4. / 3.) * N_ur * rho_g
            # so N_ur = N_eff - 3 * pncdm / rho_g / (7. / 8. * (4. / 11.)**(4. / 3.))
            # z = 1e10
            # pncdm = sum(_compute_ncdm_momenta(params['T_cmb'] * T, m, z=z, out='p') for T, m in zip(T_ncdm_over_cmb, m_ncdm))
            # rho_g = params['T_cmb']**4 * (1. + z)**4 * 4. / constants.c**3 * constants.Stefan_Boltzmann * constants.megaparsec**3 / (1e10 * constants.msun)
            # N_ur = N_eff - 3. * pncdm / rho_g / (7. / 8. * (4. / 11.)**(4. / 3.))
        if N_ur < 0.:
            raise ValueError('N_ur and m_ncdm must result in a number of relativistic neutrino species greater than or equal to zero.')

        params['N_ur'] = float(N_ur)
        #params['N_eff'] = N_ur + sum(T_ncdm_over_cmb**4 * (4. / 11.)**(-4. / 3.) for T_ncdm_over_cmb in T_ncdm_over_cmb)
        # number of massive neutrino species
        params['m_ncdm'] = m_ncdm
        params['T_ncdm_over_cmb'] = T_ncdm_over_cmb

        if params.get('z_pk', None) is None:
            # same as pyccl, https://github.com/LSSTDESC/CCL/blob/d2a5630a229378f64468d050de948b91f4480d41/src/ccl_core.c
            from . import interpolator
            params['z_pk'] = interpolator.get_default_z_callable()
        if params.get('modes', None) is None:
            params['modes'] = ['s']
        for name in ['modes', 'z_pk']:
            if np.ndim(params[name]) == 0:
                params[name] = [params[name]]
        params['z_pk'] = np.sort(params['z_pk'])
        if 0. not in params['z_pk']:
            params['z_pk'] = np.insert(params['z_pk'], 0, 0.)  # in order to normalise CAMB power spectrum with sigma8

        if 'Omega_m' in params:
            nonrelativistic_ncdm = (BaseEngine._get_rho_ncdm(params, z=0).sum() - 3 * BaseEngine._get_p_ncdm(params, z=0).sum()) / constants.rho_crit_Msunph_per_Mpcph3
            params['Omega_cdm'] = params.pop('Omega_m') - params['Omega_b'] - nonrelativistic_ncdm

        defaults = {'w0_fld': (float, -1.), 'wa_fld': (float, 0.), 'cs2_fld': (float, 1.), 'use_ppf': (bool, True)}
        for name, (type, default) in defaults.items():
            params[name] = type(params.get(name, default))

        for basename in ['Omega_cdm', 'Omega_b', 'T_cmb', 'h', 'A_s', 'sigma8', 'm_ncdm', 'T_ncdm_over_cmb']:
            if basename in params:
                value = np.asarray(params[basename])
                if np.any(value < 0.):
                    raise CosmologyInputError('Parameter {} should be positive, found {}'.format(basename, value))

        def is_str(name, default_string, allowed_strings):
            value = params[name]
            if value is None:
                value = default_string
            if isinstance(value, str):
                value = value.upper()
                if value not in allowed_strings:
                    raise CosmologyInputError('Parameter {} should be either a float or one of {}'.format(name, allowed_strings))
                params[name] = value
                return True
            params[name] = float(value)
            return False

        is_str('YHe', 'BBN', allowed_strings=('BBN',))
        is_str('n_t', 'SCC', allowed_strings=('SCC',))
        is_str('alpha_t', 'SCC', allowed_strings=('SCC',))
        r, n_s = params['r'], params['n_s']
        # e.g. https://github.com/cmbant/CAMB/blob/master/camb/initialpower.py
        if params['n_t'] == 'SCC':
            params['n_t'] = - r / 8.0 * (2.0 - n_s - r / 8.0)
        if params['alpha_t'] == 'SCC':
            params['alpha_t'] = r / 8.0 * (r / 8.0 + n_s - 1)

        return params

    def set_engine(self, engine, set_engine=True, **extra_params):
        """
        Set engine for cosmological calculation.

        Parameters
        ----------
        engine : string
            Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].

        set_engine : bool, default=True
            Whether to attach returned engine to ``cosmology``.
            (Set ``False`` if e.g. you want to use this engine for a single calculation).

        extra_params : dict
            Extra engine parameters, typically precision parameters.
        """
        self._engine = _get_cosmology_engine(self, engine, set_engine=set_engine, **extra_params)

    def clone(self, base='input', engine=None, extra_params=None, **params):
        r"""
        Clone current cosmology instance, optionally updating engine and parameters.

        Parameters
        ----------
        base : string, default='input'
            If 'internal' or ``None``, update parameters in the internal :math:`h, \Omega, m_{cdm}` basis.
            If, e.g. input parameters are :math:`h, \omega_{b}, \omega_{cdm}`, ``clone(base='internal', h=0.7)``
            returns the same cosmology, but with :math:`h = 0.7`; since :math:`\Omega_{b}, \Omega_{cdm}` are kept fixed,
            :math:`\omega_{b}, \omega_{cdm}` are modified.
            If 'input', update input parameters.
            With ``clone(base='input', h=0.7)`` :math:`\omega_{b}, \omega_{cdm}` are left unchanged,
            but :math:`\Omega_{b}, \Omega_{cdm}` are modified.

        engine : string, default=None
            Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
            If ``None``, use same engine (class) as current instance.

        extra_params : dict, default=None
            Extra engine parameters, typically precision parameters.

        params : dict
            Cosmological and calculation parameters which take priority over the current ones.

        Returns
        -------
        new : Cosmology
            Copy of current instance, with updated engine and parameters.
        """
        new = self.copy()
        check_params(params, conflicts=new.__class__._conflict_parameters)
        new._derived = {}
        if base == 'input':
            base_params = self._input_params.copy()
        elif base in ['internal', None]:
            base_params = self._params.copy()
        else:
            raise CosmologyInputError('Unknown parameter base {}'.format(base))
        new._input_params = merge_params(base_params, params, conflicts=new.__class__._conflict_parameters)
        new._params = new._compile_params(new._input_params)
        if engine is None and self._engine is not None:
            engine = self._engine.__class__
        if engine is not None:
            if extra_params is None:
                extra_params = getattr(self._engine, '_extra_params', {})
            new.set_engine(engine, **extra_params)
        return new

    def solve(self, param, func, target=0., limits=None, xtol=1e-6, rtol=1e-6, maxiter=100):
        """
        Return cosmology ``cosmo`` that verifies ``func(cosmo) == target``, by varying parameter ``param``.

        Parameters
        ----------
        param : string
            Input parameter name, e.g. 'h'.

        func : callable, string
            Function that takes a :class:`Cosmology` instance (clone of ``self``) and returns a value,
            e.g. ``lambda cosmo: cosmo.get_thermodynamics().theta_star``.
            If 'theta_MC_100', match ``100 * cosmo['theta_cosmomc']`` to ``target`` (and engine should be defined).

        target : float, default=0.
            Target value.

        xtol : float, default=1e-6
            Absolute tolerance on the value of ``param``. See :func:`scipy.optimize.bisect`.

        rtol : float, default=1e-6
            Relative tolerance on the value of ``param``. See :func:`scipy.optimize.bisect`.

        maxiter : int, default=100
            If convergence is not achieved in ``maxiter`` iterations, an error is raised. Must be >= 0.

        Returns
        -------
        new : Cosmology
        """
        default_limits = {'h': [0.1, 2.], 'H0': [10., 200.]}
        default_tol = {'h': (1e-6, 1e-6), 'H0': (1e-4, 1e-6)}

        if func == 'theta_MC_100':
            func = lambda cosmo: 100. * cosmo['theta_cosmomc']
        if func is None:
            raise CosmologyInputError('Provide func')
        if limits is None:
            limits = default_limits.get(param, None)
            if limits is None:
                raise CosmologyInputError('Provide limits')
        if xtol is None:
            xtol = default_tol.get(param, [1e-6] * 2)[0]
        if rtol is None:
            rtol = default_tol.get(param, [1e-6] * 2)[1]

        value = self[param]

        def f(value):
            new = self.clone(base='input', **{param: value})
            return func(new) - target

        from scipy import optimize
        try:
            value = optimize.bisect(f, *limits, xtol=xtol, rtol=rtol, maxiter=maxiter, disp=True)
        except ValueError as exc:
            raise CosmologyInputError('Could not find proper {} value in the interval that matches target = {:.4f} with [f({:.3f}), f({:.3f})] = [{:.4f}, {:.4f}]'.format(param, target, *limits, *list(map(f, limits)))) from exc

        return self.clone(base='input', **{param: value})

    def __setstate__(self, state):
        """Set the class state dictionary."""
        for name in ['params', 'input_params', 'derived']:
            setattr(self, '_{}'.format(name), state.get(name, {}))
        # Backward-compatibility
        #if 'N_eff' not in self._params:
        #    self._params['N_eff'] = self._params['N_ur'] + sum(T_ncdm_over_cmb**4 * (4. / 11.)**(-4. / 3.) for T_ncdm_over_cmb in self._params['T_ncdm_over_cmb'])
        #    del self._params['N_ur']
        if state.get('engine', None) is not None:
            self.set_engine(state['engine']['name'], **state['engine']['extra_params'])

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {'engine': None}
        for name in ['params', 'input_params', 'derived']:
            state[name] = getattr(self, '_{}'.format(name))
        if getattr(self, '_engine', None) is not None:
            state['engine'] = {'name': self._engine.name, 'extra_params': self._engine._extra_params}
        return state

    @classmethod
    def from_state(cls, state):
        """Instantiate and initalise class with state dictionary."""
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    @classmethod
    def load(cls, filename):
        """Load class from disk."""
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.from_state(state)
        return new

    def save(self, filename):
        """Save class to disk."""
        dirname = os.path.dirname(filename)
        utils.mkdir(dirname)
        np.save(filename, self.__getstate__())

    def __dir__(self):
        """
        List of non-duplicate members from all sections.
        Adapted from https://github.com/bccp/nbodykit/blob/master/nbodykit/cosmology/cosmology.py.
        """
        toret = super(Cosmology, self).__dir__()
        if self._engine is None:
            return toret
        for Section in self._engine._Sections.values():
            section_dir = dir(Section)
            for item in section_dir:
                if item in toret:
                    toret.remove(item)
                else:
                    toret.append(item)
        return toret

    def __getattr__(self, name):
        """
        Find the proper section, initialize it, and return its attribute.
        For example, calling ``cosmo.comoving_radial_distance`` will actually return ``cosmo.get_background().comoving_radial_distance``.
        Adapted from https://github.com/bccp/nbodykit/blob/master/nbodykit/cosmology/cosmology.py.
        """
        if self._engine is None:
            raise AttributeError('Attribute {} not found; try setting an engine ("set_engine")?'.format(name))
        # Resolving a name from the sections : cosmo.Omega0_m => cosmo.get_background().Omega0_m
        Sections = self._engine._Sections
        for section_name, Section in Sections.items():
            if hasattr(Section, name) and not any(hasattr(OtherSection, name) for OtherSection in Sections.values() if OtherSection is not Section):  # keep only single elements
                section = getattr(self._engine, 'get_{}'.format(section_name))()
                return getattr(section, name)
        raise AttributeError("Attribute {} not found in any of {} engine's products (rejecting duplicates)".format(name, self.engine.__class__.__name__))

    def __eq__(self, other):
        r"""Is ``other`` same as ``self``?"""
        return type(other) == type(self) and _deepeq(other._params, self._params) and other._engine == self._engine


@utils.addproperty('engine')
class BaseSection(object):

    """Base section."""

    def __init__(self, engine):
        self._engine = engine


def _make_section_getter(section):

    def getter(self, engine=None, set_engine=True, **extra_params):
        engine = _get_cosmology_engine(self, engine=engine, set_engine=set_engine, **extra_params)
        toret = getattr(engine, 'get_{}'.format(section), None)
        if toret is None:
            raise CosmologyInputError('Engine {} does not provide {}'.format(engine.__class__.__name__, section))
        return toret()

    getter.__doc__ = """
    Get {}.

    Parameters
    ----------
    engine : string, default=None
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_variants', 'bbks'].
        If ``None``, returns current :attr:`Cosmology.engine`.

    set_engine : bool, default=True
        Whether to attach returned engine to ``cosmology``.
        (Set ``False`` if e.g. you want to use this engine for a single calculation).

    extra_params : dict
        Extra engine parameters, typically precision parameters.
    """.format(section)

    return getter


for section in _Sections:
    setattr(Cosmology, 'get_{}'.format(section.lower()), _make_section_getter(section.lower()))


def merge_params(args, moreargs, **kwargs):
    """
    Merge ``moreargs`` parameters into ``args``.
    ``moreargs`` parameters take priority over those defined in ``args``.

    Note
    ----
    ``args`` is modified in-place.

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
        for eq in find_conflicts(name, **kwargs):
            if eq in args: args.pop(eq)

    args.update(moreargs)
    return args


def check_params(args, **kwargs):
    """Check for conflicting parameters in ``args`` parameter dictionary."""
    conf = {}
    for name in args:
        conf[name] = []
        for eq in find_conflicts(name, **kwargs):
            if eq == name: continue
            if eq in args: conf[name].append(eq)

    for name in conf:
        if conf[name]:
            raise CosmologyInputError('Conflicting parameters are given: {}'.format([name] + conf[name]))


def find_conflicts(name, conflicts=tuple()):
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
    for conf in conflicts:
        if name in conf:
            return conf
    return ()


@utils.addproperty('H0', 'h', 'N_ur', 'N_ncdm', 'm_ncdm', 'm_ncdm_tot', 'N_eff', 'T0_cmb', 'T0_ncdm', 'w0_fld', 'wa_fld', 'cs2_fld',
                   'Omega0_cdm', 'Omega0_b', 'Omega0_k', 'K', 'Omega0_g', 'Omega0_ur', 'Omega0_r',
                   'Omega0_pncdm', 'Omega0_pncdm_tot', 'Omega0_ncdm', 'Omega0_ncdm_tot',
                   'Omega0_m', 'Omega0_Lambda', 'Omega0_fld', 'Omega0_de')
class BaseBackground(BaseSection):

    """Base background engine, including a few definitions."""

    def __init__(self, engine):
        self._engine = engine
        for name in ['H0', 'h', 'N_ur', 'N_ncdm', 'm_ncdm', 'm_ncdm_tot', 'N_eff', 'w0_fld', 'wa_fld', 'cs2_fld', 'K']:
            setattr(self, '_{}'.format(name), self._engine[name])
        self._T0_cmb = self._engine['T_cmb']
        self._T0_ncdm = np.array(self._engine['T_ncdm_over_cmb']) * self._T0_cmb
        for name in ['cdm', 'b', 'k', 'g', 'ur', 'r', 'ncdm', 'ncdm_tot', 'pncdm', 'pncdm_tot', 'm', 'Lambda', 'fld', 'de']:
            setattr(self, '_Omega0_{}'.format(name), self._engine['Omega_{}'.format(name)])
        for name in ['_m_ncdm', '_Omega0_pncdm', '_Omega0_ncdm']:
            setattr(self, name, np.array(getattr(self, name), dtype='f8'))

    @utils.flatarray()
    def rho_ncdm(self, z, species=None):
        r"""
        Comoving density of non-relativistic part of massive neutrinos for each species, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`.
        If ``species`` is ``None`` returned shape is (N_ncdm,) if ``z`` is a scalar, else (N_ncdm, len(z)).
        Else if ``species`` is between 0 and N_ncdm, return density for this species.
        """
        if species is None:
            species = list(range(self.N_ncdm))

        def compute(z, species):
            return np.array([self._engine._get_rho_ncdm(z=zz, species=species) for zz in z])

        if is_sequence(species):
            return np.array([compute(z, species=s) for s in species]).reshape(len(species), len(z))

        return compute(z, species)

    def rho_ncdm_tot(self, z):
        r"""Total comoving density of non-relativistic part of massive neutrinos, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return np.sum(self.rho_ncdm(z, species=None), axis=0)

    @utils.flatarray()
    def p_ncdm(self, z, species=None):
        r"""
        Pressure of non-relativistic part of massive neutrinos for each species, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`.
        If ``species`` is ``None`` returned shape is (N_ncdm,) if ``z`` is a scalar, else (N_ncdm, len(z)).
        Else if ``species`` is between 0 and N_ncdm, return pressure for this species.
        """
        if species is None:
            species = list(range(self.N_ncdm))

        def compute(z, species):
            return np.array([self._engine._get_p_ncdm(z=zz, species=species) for zz in z])

        if is_sequence(species):
            return np.array([compute(z, species=s) for s in species]).reshape(len(species), len(z))

        return compute(z, species)

    def p_ncdm_tot(self, z):
        r"""Total pressure of non-relativistic part of massive neutrinos, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return np.sum(self.p_ncdm(z, species=None), axis=0)

    @utils.flatarray()
    def rho_g(self, z):
        r"""Comoving density of photons :math:`\rho_{g}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.Omega0_g * (1 + z) * constants.rho_crit_Msunph_per_Mpcph3

    @utils.flatarray()
    def rho_b(self, z):
        r"""Comoving density of baryons :math:`\rho_{b}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.Omega0_b * np.ones_like(z) * constants.rho_crit_Msunph_per_Mpcph3

    @utils.flatarray()
    def rho_ur(self, z):
        r"""Comoving density of massless neutrinos :math:`\rho_{ur}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.Omega0_ur * (1 + z) * constants.rho_crit_Msunph_per_Mpcph3

    def rho_r(self, z):
        r"""Comoving density of radiation :math:`\rho_{r}`, including photons and relativistic part of massive and massless neutrinos, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.rho_g(z) + self.rho_ur(z) + 3. * self.p_ncdm_tot(z)

    @utils.flatarray()
    def rho_cdm(self, z):
        r"""Comoving density of cold dark matter :math:`\rho_{cdm}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.Omega0_cdm * np.ones_like(z) * constants.rho_crit_Msunph_per_Mpcph3

    @utils.flatarray()
    def rho_m(self, z):
        r"""Comoving density of matter :math:`\rho_{m}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.rho_cdm(z) + self.rho_b(z) + self.rho_ncdm_tot(z) - 3. * self.p_ncdm_tot(z)

    @utils.flatarray()
    def rho_k(self, z):
        r"""Comoving density of curvature :math:`\rho_{k}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.Omega0_k / (1 + z) * constants.rho_crit_Msunph_per_Mpcph3

    @utils.flatarray()
    def rho_Lambda(self, z):
        r"""Comoving density of cosmological constant :math:`\rho_{\Lambda}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.Omega0_Lambda / (1 + z)**3 * constants.rho_crit_Msunph_per_Mpcph3

    @utils.flatarray()
    def rho_fld(self, z):
        r"""Comoving density of dark energy fluid :math:`\rho_{\mathrm{fld}}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        if self.Omega0_fld == 0.:
            return np.zeros_like(z)
        # return self.Omega0_fld * (1 + z) ** (3. * (1 + self.w0_fld + self.wa_fld)) * np.exp(3. * self.wa_fld * (1. / (1 + z) - 1)) * constants.rho_crit_Msunph_per_Mpcph3 / (1 + z)**3
        return self.Omega0_fld * (1 + z) ** (3. * (self.w0_fld + self.wa_fld)) * np.exp(3. * self.wa_fld * (1. / (1 + z) - 1)) * constants.rho_crit_Msunph_per_Mpcph3

    @utils.flatarray()
    def rho_de(self, z):
        r"""Total comoving density of dark energy :math:`\rho_{\mathrm{de}}` (fluid + cosmological constant), in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.rho_fld(z) + self.rho_Lambda(z)

    @utils.flatarray()
    def rho_tot(self, z):
        r"""Comoving total density :math:`\rho_{\mathrm{tot}}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        m = self.rho_cdm(z) + self.rho_b(z) + self.rho_ncdm_tot(z)  # - 3 * self.p_ncdm_tot(z)
        r = self.rho_g(z) + self.rho_ur(z)  # + 3 * self.p_ncdm_tot(z)
        de = self.rho_fld(z) + self.rho_Lambda(z)
        return m + r + de

    @utils.flatarray()
    def rho_crit(self, z):
        r"""
        Comoving critical density excluding curvature :math:`\rho_{c}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`.

        This is defined as:

        .. math::

              \rho_{\mathrm{crit}}(z) = \frac{3 H(z)^{2}}{8 \pi G}.
        """
        return self.rho_tot(z) + self.rho_k(z)

    @utils.flatarray()
    def efunc(self, z):
        r"""Function giving :math:`E(z)`, where the Hubble parameter is defined as :math:`H(z) = H_{0} E(z)`, unitless."""
        return np.sqrt(self.rho_crit(z) * (1 + z)**3 / constants.rho_crit_Msunph_per_Mpcph3)

    @utils.flatarray()
    def hubble_function(self, z):
        r"""Hubble function ``ba.index_bg_H``, in :math:`\mathrm{km}/\mathrm{s}/\mathrm{Mpc}`."""
        return self.efunc(z) * self.H0

    @utils.flatarray()
    def T_cmb(self, z):
        r"""The CMB temperature, in :math:`K`."""
        return self.T0_cmb * (1 + z)

    @utils.flatarray()
    def T_ncdm(self, z, species=None):
        r"""
        Return the ncdm temperature (massive neutrinos), in :math:`K`.
        Returned shape is (N_ncdm,) if ``z`` is a scalar, else (N_ncdm, len(z)).
        """
        return self.T0_ncdm[species if species is not None else Ellipsis, None] * (1 + z)

    @utils.flatarray()
    def Omega_cdm(self, z):
        r"""Density parameter of cold dark matter, unitless."""
        return self.rho_cdm(z) / self.rho_crit(z)

    @utils.flatarray()
    def Omega_b(self, z):
        r"""Density parameter of baryons, unitless."""
        return self.rho_b(z) / self.rho_crit(z)

    @utils.flatarray()
    def Omega_k(self, z):
        r"""Density parameter of curvature, unitless."""
        return self.rho_k(z) / self.rho_crit(z)

    @utils.flatarray()
    def Omega_g(self, z):
        r"""Density parameter of photons, unitless."""
        return self.rho_g(z) / self.rho_crit(z)

    @utils.flatarray()
    def Omega_ur(self, z):
        r"""Density parameter of massless neutrinos, unitless."""
        return self.rho_ur(z) / self.rho_crit(z)

    @utils.flatarray()
    def Omega_r(self, z):
        r"""Density parameter of radiation, including photons and relativistic part of massive and massless neutrinos, unitless."""
        return self.rho_r(z) / self.rho_crit(z)

    @utils.flatarray()
    def Omega_m(self, z):
        r"""
        Density parameter of non-relativistic (matter-like) component, including
        non-relativistic part of massive neutrino, unitless.
        """
        return self.rho_m(z) / self.rho_crit(z)

    @utils.flatarray()
    def Omega_ncdm(self, z, species=None):
        r"""
        Density parameter of massive neutrinos, unitless.
        If ``species`` is ``None`` returned shape is (N_ncdm,) if ``z`` is a scalar, else (N_ncdm, len(z)).
        Else if ``species`` is between 0 and N_ncdm, return density for this species.
        """
        return self.rho_ncdm(z, species=species) / self.rho_crit(z)

    @utils.flatarray()
    def Omega_ncdm_tot(self, z):
        r"""Total density parameter of massive neutrinos, unitless."""
        return self.rho_ncdm_tot(z) / self.rho_crit(z)

    @utils.flatarray()
    def Omega_pncdm(self, z, species=None):
        r"""
        Density parameter of pressure of non-relativistic part of massive neutrinos, unitless.
        If ``species`` is ``None`` returned shape is (N_ncdm,) if ``z`` is a scalar, else (N_ncdm, len(z)).
        Else if ``species`` is between 0 and N_ncdm, return density for this species.
        """
        return 3 * self.p_ncdm(z, species=species) / self.rho_crit(z)

    @utils.flatarray()
    def Omega_pncdm_tot(self, z):
        r"""Total density parameter of pressure of non-relativistic part of massive neutrinos, unitless."""
        return 3 * self.p_ncdm_tot(z) / self.rho_crit(z)

    @utils.flatarray()
    def Omega_Lambda(self, z):
        r"""Density of cosmological constant, unitless."""
        return self.rho_Lambda(z) / self.rho_crit(z)

    @utils.flatarray()
    def Omega_fld(self, z):
        r"""Density of cosmological constant, unitless."""
        return self.rho_fld(z) / self.rho_crit(z)

    @utils.flatarray()
    def Omega_de(self, z):
        r"""Density of total dark energy (fluid + cosmological constant), unitless."""
        return self.rho_de(z) / self.rho_crit(z)