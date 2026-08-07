"""Cosmological calculation with the H-EFTCAMB version of CAMB."""

import warnings

import numpy as np

from .camb import (CambEngine, Background as CambBackground, Thermodynamics, Primordial,
                   Transfer, Harmonic, Fourier)
from .cosmology import CosmologyComputationError
from . import utils, constants


np.int = int


# ----------------------------------------------------------------------
# EFTCAMB parametrized_function_1D model flags -> parameter name suffixes.
#
# A parametrized 1D function called NAME reads its parameters from the
# (virtual) ini file under the names built by its `parameter_names`
# routine in the Fortran source.  The default naming of
# 04_abstract_parametrizations_1D.f90 is NAME0, NAME1, ...; some
# parametrizations override it.
#
# This mapping is used to strip the shape parameters that the currently
# selected model flag will NOT read.  This matters because HEFTCAMB's
# `camb.set_params` marks as "used" only the keys the Fortran side
# actually queried; anything left over raises CAMBUnknownArgumentError.
# ----------------------------------------------------------------------
_RPH_FLAG_PARAM_SUFFIXES = {
    0: (),                          # zero
    1: ('0',),                      # constant
    2: ('0',),                      # linear
    3: ('0', 'Exp'),                # power law
    4: ('0', 'Exp'),                # exponential
    8: ('_v1', '_v2', '_at', '_delta'),   # steplog
    11: ('0', '1'),                 # exponential 2
    12: ('_cM', '_at', '_tau'),     # hill/valley, Brando et al. JCAP 11 (2019) 018 eq. (3.2)
}

# w_DE parametrization flags of 007p2_RPH.f90 (these set explicit names).
_RPH_WDE_FLAG_PARAMS = {
    0: (),                                              # LCDM
    1: ('RPHw0',),                                      # constant w
    2: ('RPHw0', 'RPHwa'),                              # CPL
    3: ('RPHw0', 'RPHwa', 'RPHwn'),                     # JBP
    4: ('RPHw0', 'RPHwa', 'RPHwat'),                    # turning point
    5: ('RPHw0', 'RPHwa', 'RPHw2', 'RPHw3'),            # Taylor
}

# (model flag name, name of the parametrized function on the Fortran side)
# Branches that are always active.
_RPH_COMMON_BRANCHES = (
    ('RPHbraidingmodel', 'RPHbraiding'),
    ('RPHbraidingmodel_ODE', 'RPHbraiding_ODE'),
    ('RPHkineticitymodel', 'RPHkineticity'),
    ('RPHkineticitymodel_ODE', 'RPHkineticity_ODE'),
    ('RPHtensormodel', 'RPHtensor'),
    ('RPHtensormodel_ODE', 'RPHtensor_ODE'),
)

# Read only when RPHusealphaM = True (007p2_RPH.f90,
# EFTCAMBRPHInitModelParametersFromFile).
_RPH_ALPHAM_BRANCHES = (
    ('RPHalphaMmodel', 'RPHalphaM'),
    ('RPHalphaMmodel_ODE', 'RPHalphaM_ODE'),
)

# Read only when RPHusealphaM = False.
_RPH_MASSP_BRANCHES = (
    ('RPHmassPmodel', 'RPHmassP'),
    ('RPHmassPmodel_ODE', 'RPHmassP_ODE'),
)

_RPH_ALPHA_BRANCHES = _RPH_COMMON_BRANCHES + _RPH_ALPHAM_BRANCHES + _RPH_MASSP_BRANCHES

# Model flag of the hill/valley parametrization added to
# fortran/eftcamb/04f_parametrizations_1D/04p020_hillvalley_parametrizations_1D.f90
# and registered as case(12) in 04p1_parametrizations_1D_allocator.f90.
HILLVALLEY_FLAG = 12


class Background(CambBackground):
    """CAMB background section, with the GR-only growth solver disabled.

    Recent cosmoprimo gives ``camb.Background`` a ``growth_factor`` /
    ``growth_rate`` that integrates the *general relativistic* growth equation

        D'' + [1 - (addot/a)/H^2] D' = (3/2) Omega_m(z) D

    via ``DefaultBackground``, using cosmoprimo's analytic ``w0_fld`` /
    ``wa_fld`` background. Both ingredients are wrong for an EFTCAMB run: the
    source term carries no alpha_M / alpha_B modification of the Poisson
    equation, and the expansion history is cosmoprimo's, not the one EFTCAMB
    integrates from ``RPHwDE``. Since these methods are inherited silently,
    they are blocked here rather than returning plausible but GR numbers.

    Get the modified-gravity growth from the perturbations instead, e.g.

        fo = cosmo.get_fourier()
        f_sigma8 = fo.sigma8_z(z, of='theta_cb') / fo.sigma8_z(z, of='delta_cb')

    Pass ``heftcamb_gr_growth=True`` in ``extra_params`` to re-enable the GR
    computation (for instance to compare against a GR reference).
    """

    def _check_gr_growth(self, name):
        if getattr(self._engine, '_heftcamb_gr_growth', False):
            return
        raise CosmologyComputationError(
            "Background.{}() on the 'heftcamb' engine would solve the general relativistic "
            "growth ODE on cosmoprimo's (w0_fld, wa_fld) background, ignoring the EFTCAMB "
            "modification of gravity and of the expansion history; the result would not "
            "describe this model. Use the perturbations instead, e.g. "
            "fo = cosmo.get_fourier(); fo.sigma8_z(z, of='theta_cb') / fo.sigma8_z(z, of='delta_cb'), "
            "or pass heftcamb_gr_growth=True to force the GR calculation.".format(name)
        )

    def growth_factor(self, z, mass='m', znorm=None):
        self._check_gr_growth('growth_factor')
        return super().growth_factor(z, mass=mass, znorm=znorm)

    def growth_rate(self, z, mass='m'):
        self._check_gr_growth('growth_rate')
        return super().growth_rate(z, mass=mass)


class HEFTCAMBEngine(CambEngine):
    """Engine for the H-EFTCAMB version of CAMB.

    This follows the same cosmoprimo pattern as isitgr.py:

      - _default_cosmological_parameters contains non-standard MG/EFT
        parameters that can vary like cosmological parameters.

      - _default_calculation_parameters contains switches, model selectors,
        and numerical options.

    Do NOT put standard cosmoprimo parameters such as logA, omega_b,
    omega_cdm, h, n_s, etc. here. CambEngine already handles those and maps
    them to CAMB names. Putting them here can forward logA directly to
    camb.set_params(...), causing CAMBUnknownArgumentError.

    Two alpha-function shapes are wired up here:

      - the default one, alpha_X(a) = cX * Omega_DE(a), obtained with
        RPHXmodel = 0 and RPHXmodel_ODE = 2;

      - the "hill/valley" shape of Brando, Falciano, Linder & Velten,
        JCAP 11 (2019) 018, eq. (3.2),

            alpha(a) = c tanh[(tau/2) ln(a/a_t)] / cosh^2[(tau/2) ln(a/a_t)]

        obtained with RPHXmodel = 12 and the three parameters
        RPHX_cM, RPHX_at, RPHX_tau.  This requires the patched HEFTCAMB
        build that provides parametrization flag 12.

    Shape parameters that the selected model flag would not read are
    stripped before reaching camb.set_params(...); see _prune_rph_params.
    """

    name = "heftcamb"

    # ------------------------------------------------------------------
    # Non-standard EFT/MG parameters.
    #
    # This is analogous to IsitgrEngine putting mu0, Sigma0, etc. here.
    # These are accepted by HEFTCAMB's camb.set_params(...).
    # ------------------------------------------------------------------
    _default_cosmological_parameters = dict(
        # --- alpha_X(a) = RPHX_ODE0 * Omega_DE(a)  (RPHXmodel_ODE = 2) ---

        # alpha_K(a) = RPHkineticity_ODE0 * Omega_DE(a)
        RPHkineticity_ODE0=1.0,

        # alpha_B(a) = RPHbraiding_ODE0 * Omega_DE(a)
        RPHbraiding_ODE0=0.0,

        # alpha_M(a) = RPHalphaM_ODE0 * Omega_DE(a)
        RPHalphaM_ODE0=0.0,

        # alpha_T(a) = RPHtensor_ODE0 * Omega_DE(a)
        RPHtensor_ODE0=0.0,

        # --- hill/valley shape (RPHXmodel = 12), eq. (3.2) of
        # --- Brando et al. JCAP 11 (2019) 018.  Only forwarded to
        # --- HEFTCAMB when the corresponding model flag is 12.
        RPHalphaM_cM=0.0,
        RPHalphaM_at=0.5,
        RPHalphaM_tau=1.0,

        RPHbraiding_cM=0.0,
        RPHbraiding_at=0.5,
        RPHbraiding_tau=1.0,
    )

    # ------------------------------------------------------------------
    # CAMB / EFTCAMB calculation switches.
    # ------------------------------------------------------------------
    _default_calculation_parameters = dict(
        # CAMB / HEFTCAMB
        dark_energy_model="EFTCAMB",

        # EFTCAMB model selection
        EFTflag=2,
        AltParEFTmodel=1,

        # EFTCAMB turn-on / stability settings
        EFTCAMB_back_turn_on=1.0e-8,
        EFTCAMB_turn_on_time=1.0e-8,
        EFTCAMB_skip_stability=True,
        feedback_level=0,

        # Optional stability flags
        EFT_ghost_math_stability=False,
        EFT_mass_math_stability=False,
        EFT_ghost_stability=True,
        EFT_gradient_stability=True,
        EFT_mass_stability=False,
        EFT_additional_priors=False,

        # RPH alpha-basis setup
        RPHintegratefromtoday=False,
        RPHusealphaM=True,

        # Initial condition of the M^2 ODE:
        # 1 + RPH_M0 = M_eff^2 / m_0^2 at the starting time of the
        # integration (early times when RPHintegratefromtoday = False).
        RPH_M0=0.0,

        # alpha_K branch
        RPHkineticitymodel=0,
        RPHkineticitymodel_ODE=2,

        # alpha_B branch
        RPHbraidingmodel=0,
        RPHbraidingmodel_ODE=2,

        # alpha_M branch
        RPHalphaMmodel=0,
        RPHalphaMmodel_ODE=2,

        # alpha_T branch
        RPHtensormodel=0,
        RPHtensormodel_ODE=2,
    )

    # Wrapper-only options. These must not reach camb.set_params(...).
    _wrapper_private_keys = [
        "eftcamb_params",
        "eftcamb_print_header",
        "heftcamb_debug",
        "heftcamb_map_w0wa",
        "heftcamb_gr_growth",
        "RPH_massP0",
        "RPH_braiding0",
        "RPH_kinetic0",
    ]

    def __init__(self, *args, **kwargs):
        # ------------------------------------------------------------
        # Wrapper-only options
        # ------------------------------------------------------------
        eftcamb_params = kwargs.pop("eftcamb_params", None)
        eftcamb_print_header = kwargs.pop("eftcamb_print_header", False)
        heftcamb_debug = kwargs.pop("heftcamb_debug", eftcamb_print_header)

        # Whether to translate cosmoprimo's (w0_fld, wa_fld) into the
        # EFTCAMB background parametrization (RPHwDE / RPHw0 / RPHwa).
        # Without this, w0_fld/wa_fld are silently ignored by EFTCAMB,
        # which drives its own background from RPHwDE.
        heftcamb_map_w0wa = kwargs.pop("heftcamb_map_w0wa", True)

        # Allow the GR growth ODE of Background.growth_factor / growth_rate,
        # which does not describe the modified gravity model (see Background).
        heftcamb_gr_growth = kwargs.pop("heftcamb_gr_growth", False)

        # Stash on self: _set_camb() runs inside super().__init__().
        self._heftcamb_debug = bool(heftcamb_debug)
        self._heftcamb_map_w0wa = bool(heftcamb_map_w0wa)
        self._heftcamb_gr_growth = bool(heftcamb_gr_growth)
        self._pruned_rph_params = {}

        # Convenience aliases.
        # Use None defaults so these aliases do not accidentally overwrite
        # a full eftcamb_params dictionary.
        RPH_massP0 = kwargs.pop("RPH_massP0", None)
        RPH_braiding0 = kwargs.pop("RPH_braiding0", None)
        RPH_kinetic0 = kwargs.pop("RPH_kinetic0", None)

        # ------------------------------------------------------------
        # Build parameter dictionary to push through CambEngine.
        # ------------------------------------------------------------
        params = {}
        params.update(self._default_calculation_parameters)
        params.update(self._default_cosmological_parameters)

        if eftcamb_params is not None:
            params.update(dict(eftcamb_params))

        # Convenience aliases override defaults / eftcamb_params only
        # if explicitly supplied.
        if RPH_massP0 is not None:
            params["RPHalphaM_ODE0"] = float(RPH_massP0)

        if RPH_braiding0 is not None:
            params["RPHbraiding_ODE0"] = float(RPH_braiding0)

        if RPH_kinetic0 is not None:
            params["RPHkineticity_ODE0"] = float(RPH_kinetic0)

        # Push EFTCAMB/RPH params into kwargs before CambEngine is built.
        #
        # Use setdefault so explicit top-level kwargs like
        # RPHalphaM_ODE0=... still win.
        for key, value in params.items():
            kwargs.setdefault(key, value)

        # Ensure wrapper-only keys do not leak.
        for key in self._wrapper_private_keys:
            kwargs.pop(key, None)

        # Remember whether flag 12 was asked for, so that a failure can be
        # attributed to a HEFTCAMB build without the hill/valley parametrization.
        self._hillvalley_in_kwargs = False
        for flag_name, _ in _RPH_ALPHA_BRANCHES:
            try:
                if int(kwargs.get(flag_name, 0)) == HILLVALLEY_FLAG:
                    self._hillvalley_in_kwargs = True
                    break
            except (TypeError, ValueError):
                continue

        if heftcamb_debug:
            self._debug_kwargs_before_super(kwargs)

        # Parent CambEngine now sees the complete HEFTCAMB parameter set.
        try:
            super().__init__(*args, **kwargs)
        except Exception as exc:
            raise self._annotate_init_error(exc) from exc

        # read_parameters() is cached Python-side; clear before debug.
        self._clear_eftcamb_read_cache()

        if heftcamb_debug:
            self._debug_eftcamb_parameters("after CambEngine")

    # ------------------------------------------------------------------
    # RPH parameter bookkeeping
    # ------------------------------------------------------------------
    def _rph_setting(self, name, default=0):
        """Value of an EFTCAMB switch as it will be seen by set_params.

        ``CambEngine`` builds ``all_params = self._extra_params | base_params``
        with ``base_params`` derived from ``self._params``, so ``self._params``
        has priority.
        """
        for container in (getattr(self, '_params', {}), getattr(self, '_extra_params', {})):
            if name in container:
                return container[name]
        return self._default_calculation_parameters.get(name, default)

    def _resolve_default_param_conflicts(self):
        """Let explicit ``extra_params`` win over untouched class defaults.

        ``BaseEngine.__init__`` merges *both* ``_default_cosmological_parameters``
        and ``_default_calculation_parameters`` into ``self._params``, while
        anything the user passes as ``extra_params`` lands in
        ``self._extra_params``. ``CambEngine`` then builds
        ``all_params = self._extra_params | base_params`` with ``base_params``
        derived from ``self._params``, so ``self._params`` wins.

        The consequence is that an engine switch such as ``RPHalphaMmodel``,
        which has a class default of 0, keeps that default even when the user
        writes ``extra_params={'RPHalphaMmodel': 12}``: the class default in
        ``self._params`` overrides the explicit request. When the value in
        ``self._params`` is still the untouched class default and
        ``extra_params`` carries something else, the latter is used.

        A value set directly on the ``Cosmology`` (so that ``self._params``
        differs from the class default) still has the last word.
        """
        defaults = dict(self._default_calculation_parameters)
        defaults.update(self._default_cosmological_parameters)
        for key, default in defaults.items():
            if key in self._params and key in self._extra_params:
                if self._params[key] == default and self._extra_params[key] != default:
                    self._params.pop(key)

    def _eftcamb_drives_background(self):
        """Whether EFTCAMB, rather than CAMB's DarkEnergy, sets the background."""
        try:
            return int(self._rph_setting('EFTflag', 0)) != 0
        except (TypeError, ValueError):
            return False

    @property
    def _has_fld(self):
        r"""Report no CAMB fluid dark energy while EFTCAMB drives the background.

        ``CambEngine.__init__`` forwards ``(w0_fld, wa_fld, cs2_fld)`` to
        ``cp.DarkEnergy.set_params`` only when ``_has_fld`` is true. With
        ``dark_energy_model = 'EFTCAMB'``, HEFTCAMB's ``set_classes`` allocates a
        *fluid* ``DarkEnergy`` object that EFTCAMB then never uses: whenever
        ``EFTFlag /= 0``, ``dtauda`` and the perturbation equations in
        equations.f90 set ``grhov_t = 0`` and take the dark energy density from
        the EFTCAMB cache instead, which the RPH model integrates from
        ``RPHwDE``. That unused fluid object is nevertheless validated, and
        ``DarkEnergyFluid.validate_params`` raises "fluid dark energy model does
        not support w crossing -1" for any phantom-crossing CPL -- including the
        mirage line w_a = -3.6 (1 + w_0) of Brando et al., for which
        1 + w_0 + w_a < 0.

        The equation of state is not lost: :meth:`_map_w0wa_to_rph` has already
        copied it into ``RPHwDE`` / ``RPHw0`` / ``RPHwa``, and ``self._params``
        keeps ``w0_fld`` / ``wa_fld`` untouched, so cosmoprimo's own bookkeeping
        and the CAMB-derived background quantities are unaffected.
        """
        if self._eftcamb_drives_background():
            return False
        return CambEngine._has_fld.fget(self)

    def _check_background_consistency(self):
        """Warn if a non-trivial w(a) would be dropped on both sides."""
        if not self._eftcamb_drives_background():
            return
        w0 = float(self._params.get('w0_fld', -1.))
        wa = float(self._params.get('wa_fld', 0.))
        if w0 == -1. and wa == 0.:
            return
        if int(self._rph_setting('RPHwDE', 0)) == 0:
            warnings.warn(
                "HEFTCAMBEngine: w0_fld={}, wa_fld={} but RPHwDE=0, so EFTCAMB will use a "
                "cosmological constant background. CAMB's own dark energy is bypassed when "
                "EFTflag != 0, so the equation of state would be silently ignored. Set "
                "RPHwDE / RPHw0 / RPHwa explicitly, or leave heftcamb_map_w0wa=True.".format(w0, wa)
            )

    def _map_w0wa_to_rph(self):
        """Translate (w0_fld, wa_fld) into the EFTCAMB background flags.

        Only applied when the dark energy equation of state is not the
        cosmological constant and when the user has not set RPHwDE
        explicitly. EFTCAMB drives its own background from RPHwDE, so
        without this the cosmoprimo w0/wa would have no effect on the
        EFTCAMB side.
        """
        if not getattr(self, '_heftcamb_map_w0wa', True):
            return
        if 'RPHwDE' in self._params or 'RPHwDE' in self._extra_params:
            return

        w0 = float(self._params.get('w0_fld', -1.))
        wa = float(self._params.get('wa_fld', 0.))
        if w0 == -1. and wa == 0.:
            return

        self._extra_params['RPHwDE'] = 2   # CPL
        self._extra_params['RPHw0'] = w0
        self._extra_params['RPHwa'] = wa

    def _prune_rph_params(self):
        """Drop RPH shape parameters that the active model flags will not read.

        HEFTCAMB's ``camb.set_params`` hands the whole parameter dictionary
        to EFTCAMB, then marks as used only the keys the Fortran side
        actually queried (``Ini%ReadValues``). Any key left over is fed to
        ``setattr(CAMBparams, ...)`` and raises ``CAMBUnknownArgumentError``.

        A parametrized function only reads the parameters of the model it was
        allocated with, so e.g. ``RPHalphaM_cM`` must not be forwarded unless
        ``RPHalphaMmodel == 12``.
        """
        known, keep = set(), set()

        use_alpham = bool(self._rph_setting('RPHusealphaM', True))
        active = _RPH_COMMON_BRANCHES + (_RPH_ALPHAM_BRANCHES if use_alpham else _RPH_MASSP_BRANCHES)

        # RPH_M0 is read only on the alpha_M branch.
        known.add('RPH_M0')
        if use_alpham:
            keep.add('RPH_M0')

        for flag_name, func_name in _RPH_ALPHA_BRANCHES:
            flag = int(self._rph_setting(flag_name, 0))
            is_active = (flag_name, func_name) in active
            for candidate_flag, suffixes in _RPH_FLAG_PARAM_SUFFIXES.items():
                names = {func_name + suffix for suffix in suffixes}
                known |= names
                if is_active and candidate_flag == flag:
                    keep |= names

        wde_flag = int(self._rph_setting('RPHwDE', 0))
        for candidate_flag, names in _RPH_WDE_FLAG_PARAMS.items():
            known |= set(names)
            if candidate_flag == wde_flag:
                keep |= set(names)

        drop = known - keep

        self._pruned_rph_params = {}
        for container in (self._params, self._extra_params):
            for key in list(container):
                if key in drop:
                    self._pruned_rph_params[key] = container.pop(key)

        if self._pruned_rph_params and getattr(self, '_heftcamb_debug', False):
            print(
                "HEFTCAMBEngine: not forwarding unused RPH shape parameters "
                + ", ".join(sorted(self._pruned_rph_params)),
                flush=True,
            )

    def _annotate_init_error(self, exc):
        """Add a build hint when initialisation fails with flag 12 requested.

        ``allocate_parametrized_1D_function`` only prints "No model corresponding
        to flag" when ``feedback_level > 0``; with the default of 0 an unknown
        model flag surfaces as a bare EFTCAMB initialisation failure. Rather than
        warning on every run that uses flag 12, the hint is attached here, where
        something has actually gone wrong.
        """
        if not self._hillvalley_requested():
            return exc
        message = (
            "EFTCAMB initialisation failed while RPH model flag {} (hill/valley alpha, "
            "Brando et al. JCAP 11 (2019) 018 eq. 3.2) was requested. If this HEFTCAMB build "
            "does not provide flag {}, add "
            "fortran/eftcamb/04f_parametrizations_1D/04p020_hillvalley_parametrizations_1D.f90 "
            "with its case({}) entry in 04p1_parametrizations_1D_allocator.f90 and rebuild "
            "(make eftcamb_dep && make clean && make camb). Setting feedback_level=1 makes "
            "EFTCAMB report an unknown model flag explicitly. Original error: {!r}".format(
                HILLVALLEY_FLAG, HILLVALLEY_FLAG, HILLVALLEY_FLAG, exc)
        )
        try:
            return type(exc)(message)
        except Exception:
            return RuntimeError(message)

    def _hillvalley_requested(self):
        """Whether any alpha branch asks for the hill/valley parametrization."""
        if getattr(self, '_hillvalley_in_kwargs', False):
            return True
        for flag_name, _ in _RPH_ALPHA_BRANCHES:
            try:
                if int(self._rph_setting(flag_name, 0)) == HILLVALLEY_FLAG:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _set_camb(self):
        import camb as heftcamb

        try:
            pars = heftcamb.CAMBparams()
            has_eftcamb = hasattr(pars, "EFTCAMB")
        except Exception:
            has_eftcamb = False

        if not has_eftcamb:
            raise ImportError(
                "Imported `camb`, but it does not look like HEFTCAMB: "
                "`CAMBparams()` has no `EFTCAMB` attribute. Make sure the "
                "HEFTCAMB_fullshape/camb build directory is first on PYTHONPATH."
            )

        self.camb = heftcamb

        # Clean only wrapper-private keys.
        # Do NOT remove real EFTCAMB/RPH parameters.
        if hasattr(self, "_extra_params"):
            for key in self._wrapper_private_keys:
                self._extra_params.pop(key, None)

        if hasattr(self, "_params"):
            for key in self._wrapper_private_keys:
                self._params.pop(key, None)

        # _set_camb() is called by CambEngine.__init__ after _params and
        # _extra_params are set and before base_params is assembled, so this
        # is the right place to finalise the EFTCAMB parameter set.
        self._resolve_default_param_conflicts()
        self._map_w0wa_to_rph()
        self._check_background_consistency()
        self._prune_rph_params()

        if getattr(self, '_heftcamb_debug', False):
            self._debug_effective_params()

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    @staticmethod
    def _build_rph_eftcamb_params(
        *,
        RPH_massP0=0.0,
        RPH_braiding0=0.0,
        RPH_kinetic0=1.0,
        feedback_level=0,
        EFTCAMB_back_turn_on=1.0e-8,
        EFTCAMB_turn_on_time=1.0e-8,
        EFTCAMB_skip_stability=True,
        EFT_ghost_math_stability=False,
        EFT_mass_math_stability=False,
        EFT_ghost_stability=True,
        EFT_gradient_stability=True,
        EFT_mass_stability=False,
        EFT_additional_priors=False,
    ):
        """Build RPH Horndeski alpha-basis EFTCAMB parameters.

        alpha_M(a) = RPH_massP0    * Omega_DE(a)
        alpha_B(a) = RPH_braiding0 * Omega_DE(a)
        alpha_K(a) = RPH_kinetic0  * Omega_DE(a)
        alpha_T(a) = 0
        """

        return {
            # Model selection
            "EFTflag": 2,
            "AltParEFTmodel": 1,

            # Runtime / stability
            "EFTCAMB_back_turn_on": float(EFTCAMB_back_turn_on),
            "EFTCAMB_turn_on_time": float(EFTCAMB_turn_on_time),
            "EFTCAMB_skip_stability": bool(EFTCAMB_skip_stability),
            "feedback_level": int(feedback_level),

            # RPH alpha-basis setup
            "RPHintegratefromtoday": False,
            "RPHusealphaM": True,

            # alpha_K(a) = cK * Omega_DE(a)
            "RPHkineticitymodel": 0,
            "RPHkineticitymodel_ODE": 2,
            "RPHkineticity_ODE0": float(RPH_kinetic0),

            # alpha_B(a) = cB * Omega_DE(a)
            "RPHbraidingmodel": 0,
            "RPHbraidingmodel_ODE": 2,
            "RPHbraiding_ODE0": float(RPH_braiding0),

            # alpha_M(a) = cM * Omega_DE(a)
            "RPHalphaMmodel": 0,
            "RPHalphaMmodel_ODE": 2,
            "RPHalphaM_ODE0": float(RPH_massP0),

            # alpha_T(a) = 0
            "RPHtensormodel": 0,
            "RPHtensormodel_ODE": 2,
            "RPHtensor_ODE0": 0.0,

            # Optional stability flags
            "EFT_ghost_math_stability": bool(EFT_ghost_math_stability),
            "EFT_mass_math_stability": bool(EFT_mass_math_stability),
            "EFT_ghost_stability": bool(EFT_ghost_stability),
            "EFT_gradient_stability": bool(EFT_gradient_stability),
            "EFT_mass_stability": bool(EFT_mass_stability),
            "EFT_additional_priors": bool(EFT_additional_priors),
        }

    @staticmethod
    def _build_hillvalley_eftcamb_params(
        *,
        cM=-0.05,
        at=0.5,
        tau=1.0,
        no_slip=True,
        RPH_kinetic0=1.0,
        w0=None,
        wa=None,
        feedback_level=0,
        EFTCAMB_back_turn_on=1.0e-8,
        EFTCAMB_turn_on_time=1.0e-8,
        EFTCAMB_skip_stability=True,
        EFT_ghost_math_stability=False,
        EFT_mass_math_stability=False,
        EFT_ghost_stability=True,
        EFT_gradient_stability=True,
        EFT_mass_stability=False,
        EFT_additional_priors=False,
    ):
        """Build the hill/valley alpha_M parameters of eq. (3.2) of
        Brando, Falciano, Linder & Velten, JCAP 11 (2019) 018.

            alpha_M(a) = cM tanh[(tau/2) ln(a/at)] / cosh^2[(tau/2) ln(a/at)]

        Stability requires cM < 0, and the extremal amplitude is
        0.385 |cM|, not |cM|.

        With ``no_slip=True`` the braiding is locked to the No Slip Gravity
        condition alpha_B = -2 alpha_M in the Bellini-Sawicki convention.
        EFTCAMB stores alpha_B^EFTCAMB = -alpha_B^BS / 2 (see the comment at
        008p0_Horndeski.f90:1783 and the assignment in 007p2_RPH.f90), so in
        EFTCAMB's own convention this is alpha_B = +alpha_M: the same
        functional form with the same three parameters.

        ``w0`` / ``wa`` optionally select a CPL background (RPHwDE = 2).
        """
        params = {
            # Model selection
            "EFTflag": 2,
            "AltParEFTmodel": 1,

            # Runtime / stability
            "EFTCAMB_back_turn_on": float(EFTCAMB_back_turn_on),
            "EFTCAMB_turn_on_time": float(EFTCAMB_turn_on_time),
            "EFTCAMB_skip_stability": bool(EFTCAMB_skip_stability),
            "feedback_level": int(feedback_level),

            # RPH alpha-basis setup.  alpha_M -> 0 in the early universe for
            # this shape, so integrate the M^2 ODE from the radiation era
            # starting at M^2 = m_0^2.
            "RPHintegratefromtoday": False,
            "RPHusealphaM": True,
            "RPH_M0": 0.0,

            # alpha_M: hill/valley
            "RPHalphaMmodel": HILLVALLEY_FLAG,
            "RPHalphaM_cM": float(cM),
            "RPHalphaM_at": float(at),
            "RPHalphaM_tau": float(tau),
            "RPHalphaMmodel_ODE": 0,

            # alpha_K(a) = cK * Omega_DE(a); irrelevant on sub-horizon scales
            # but must be non-zero to avoid a singular kinetic term.
            "RPHkineticitymodel": 0,
            "RPHkineticitymodel_ODE": 2,
            "RPHkineticity_ODE0": float(RPH_kinetic0),

            # alpha_T = 0 (GW speed = c)
            "RPHtensormodel": 0,
            "RPHtensormodel_ODE": 0,

            # Optional stability flags
            "EFT_ghost_math_stability": bool(EFT_ghost_math_stability),
            "EFT_mass_math_stability": bool(EFT_mass_math_stability),
            "EFT_ghost_stability": bool(EFT_ghost_stability),
            "EFT_gradient_stability": bool(EFT_gradient_stability),
            "EFT_mass_stability": bool(EFT_mass_stability),
            "EFT_additional_priors": bool(EFT_additional_priors),
        }

        if no_slip:
            params.update({
                "RPHbraidingmodel": HILLVALLEY_FLAG,
                "RPHbraiding_cM": float(cM),
                "RPHbraiding_at": float(at),
                "RPHbraiding_tau": float(tau),
                "RPHbraidingmodel_ODE": 0,
            })
        else:
            params.update({
                "RPHbraidingmodel": 0,
                "RPHbraidingmodel_ODE": 2,
                "RPHbraiding_ODE0": 0.0,
            })

        if w0 is not None or wa is not None:
            params.update({
                "RPHwDE": 2,
                "RPHw0": float(-1.0 if w0 is None else w0),
                "RPHwa": float(0.0 if wa is None else wa),
            })

        return params

    def _clear_eftcamb_read_cache(self):
        """Clear EFTCAMB Python-side read_parameters() cache."""
        try:
            self._camb_params.EFTCAMB._read_parameters = None
        except Exception:
            pass

    def _debug_effective_params(self):
        """Print the EFTCAMB parameters exactly as set_params will see them.

        Reproduces ``CambEngine``'s ``all_params = self._extra_params | base_params``
        priority. This is the dump to trust: the kwargs dump printed before
        ``super().__init__()`` shows only the engine defaults, not the values
        carried on the ``Cosmology`` itself.
        """
        print("\n" + "=" * 80, flush=True)
        print("DEBUG HEFTCAMBEngine effective parameters passed to set_params", flush=True)
        print("=" * 80, flush=True)
        merged = {**self._extra_params, **self._params}
        keys = [key for key in merged
                if key.startswith('RPH') or key in ('EFTflag', 'AltParEFTmodel', 'dark_energy_model')]
        for key in sorted(keys):
            print(f"  {key}: {merged[key]}", flush=True)
        if self._pruned_rph_params:
            print("  (pruned, not forwarded):", sorted(self._pruned_rph_params), flush=True)

    def _debug_kwargs_before_super(self, kwargs):
        print("\n" + "=" * 80, flush=True)
        print("DEBUG HEFTCAMBEngine kwargs before CambEngine", flush=True)
        print("(engine defaults + extra_params only; values set on the Cosmology "
              "itself appear later, in the effective-parameter dump)", flush=True)
        print("=" * 80, flush=True)

        keys = [
            "dark_energy_model",
            "EFTflag",
            "AltParEFTmodel",
            "RPHintegratefromtoday",
            "RPHusealphaM",
            "RPH_M0",
            "RPHwDE",
            "RPHw0",
            "RPHwa",
            "RPHalphaMmodel",
            "RPHalphaMmodel_ODE",
            "RPHalphaM_ODE0",
            "RPHalphaM_cM",
            "RPHalphaM_at",
            "RPHalphaM_tau",
            "RPHbraidingmodel",
            "RPHbraidingmodel_ODE",
            "RPHbraiding_ODE0",
            "RPHbraiding_cM",
            "RPHbraiding_at",
            "RPHbraiding_tau",
            "RPHkineticitymodel",
            "RPHkineticitymodel_ODE",
            "RPHkineticity_ODE0",
            "RPHtensormodel",
            "RPHtensormodel_ODE",
            "RPHtensor_ODE0",
            "EFTCAMB_back_turn_on",
            "EFTCAMB_turn_on_time",
            "EFTCAMB_skip_stability",
            "feedback_level",
        ]

        for key in keys:
            print(f"  {key}: {kwargs.get(key, '<MISSING>')}", flush=True)

    def _debug_eftcamb_parameters(self, label="EFTCAMB"):
        """Print what EFTCAMB actually read."""
        print("\n" + "=" * 80, flush=True)
        print(f"DEBUG {label}", flush=True)
        print("=" * 80, flush=True)

        if self._pruned_rph_params:
            print("  pruned (not forwarded):", sorted(self._pruned_rph_params), flush=True)

        if not hasattr(self, "_camb_params"):
            print("No self._camb_params available.", flush=True)
            return

        if not hasattr(self._camb_params, "EFTCAMB"):
            print("self._camb_params has no EFTCAMB.", flush=True)
            return

        self._clear_eftcamb_read_cache()

        try:
            read = self._camb_params.EFTCAMB.read_parameters()

            for key in [
                "EFTflag",
                "AltParEFTmodel",
                "RPHintegratefromtoday",
                "RPHusealphaM",
                "RPH_M0",
                "RPHwDE",
                "RPHw0",
                "RPHwa",
                "RPHalphaMmodel",
                "RPHalphaMmodel_ODE",
                "RPHalphaM_ODE0",
                "RPHalphaM_cM",
                "RPHalphaM_at",
                "RPHalphaM_tau",
                "RPHbraidingmodel",
                "RPHbraidingmodel_ODE",
                "RPHbraiding_ODE0",
                "RPHbraiding_cM",
                "RPHbraiding_at",
                "RPHbraiding_tau",
                "RPHkineticitymodel",
                "RPHkineticitymodel_ODE",
                "RPHkineticity_ODE0",
                "RPHtensormodel",
                "RPHtensormodel_ODE",
                "RPHtensor_ODE0",
                "EFTCAMB_back_turn_on",
                "EFTCAMB_turn_on_time",
                "EFTCAMB_skip_stability",
                "feedback_level",
            ]:
                print(f"  read {key}: {read.get(key, '<MISSING>')}", flush=True)

        except Exception as exc:
            print("Could not read EFTCAMB read_parameters:", repr(exc), flush=True)

        try:
            print("EFTCAMB model_name:", self._camb_params.EFTCAMB.model_name(), flush=True)
        except Exception as exc:
            print("Could not read EFTCAMB model_name:", repr(exc), flush=True)

        try:
            print("EFTCAMB param_names:", self._camb_params.EFTCAMB.param_names(), flush=True)
            print("EFTCAMB param_values:", self._camb_params.EFTCAMB.param_values(), flush=True)
        except Exception as exc:
            print("Could not read EFTCAMB param names/values:", repr(exc), flush=True)

    @staticmethod
    def _initialize_eftcamb(camb_params, eftcamb_params=None, print_header=False):
        """Legacy helper.

        Kept for compatibility. The engine itself should pass EFTCAMB
        parameters into CambEngine before camb.get_results(...).
        """
        if not eftcamb_params:
            return camb_params

        if not hasattr(camb_params, "EFTCAMB"):
            raise AttributeError(
                "CAMBparams has no EFTCAMB object. Did you import HEFTCAMB's camb?"
            )

        if not hasattr(camb_params.EFTCAMB, "initialize_parameters"):
            raise AttributeError(
                "CAMBparams.EFTCAMB has no initialize_parameters method. "
                "This does not look like the expected HEFTCAMB Python wrapper."
            )

        try:
            camb_params.EFTCAMB._read_parameters = None
        except Exception:
            pass

        camb_params.EFTCAMB.initialize_parameters(
            camb_params,
            dict(eftcamb_params),
            print_header=bool(print_header),
        )

        try:
            camb_params.EFTCAMB._read_parameters = None
        except Exception:
            pass

        return camb_params
