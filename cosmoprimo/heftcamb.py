"""Cosmological calculation with the H-EFTCAMB version of CAMB."""

import numpy as np

from .camb import CambEngine, Background, Thermodynamics, Primordial, Transfer, Harmonic, Fourier
from . import utils, constants


np.int = int


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
    """

    name = "heftcamb"

    # ------------------------------------------------------------------
    # Non-standard EFT/MG parameters.
    #
    # This is analogous to IsitgrEngine putting mu0, Sigma0, etc. here.
    # These are accepted by HEFTCAMB's camb.set_params(...).
    # ------------------------------------------------------------------
    _default_cosmological_parameters = dict(
        # alpha_K(a) = RPHkineticity_ODE0 * Omega_DE(a)
        RPHkineticity_ODE0=1.0,

        # alpha_B(a) = RPHbraiding_ODE0 * Omega_DE(a)
        RPHbraiding_ODE0=0.0,

        # alpha_M(a) = RPHalphaM_ODE0 * Omega_DE(a)
        RPHalphaM_ODE0=0.0,

        # alpha_T(a) = RPHtensor_ODE0 * Omega_DE(a)
        RPHtensor_ODE0=0.0,
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

        if heftcamb_debug:
            self._debug_kwargs_before_super(kwargs)

        # Parent CambEngine now sees the complete HEFTCAMB parameter set.
        super().__init__(*args, **kwargs)

        # read_parameters() is cached Python-side; clear before debug.
        self._clear_eftcamb_read_cache()

        if heftcamb_debug:
            self._debug_eftcamb_parameters("after CambEngine")

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

    def _clear_eftcamb_read_cache(self):
        """Clear EFTCAMB Python-side read_parameters() cache."""
        try:
            self._camb_params.EFTCAMB._read_parameters = None
        except Exception:
            pass

    def _debug_kwargs_before_super(self, kwargs):
        print("\n" + "=" * 80, flush=True)
        print("DEBUG HEFTCAMBEngine kwargs before CambEngine", flush=True)
        print("=" * 80, flush=True)

        keys = [
            "dark_energy_model",
            "EFTflag",
            "AltParEFTmodel",
            "RPHintegratefromtoday",
            "RPHusealphaM",
            "RPHalphaMmodel",
            "RPHalphaMmodel_ODE",
            "RPHalphaM_ODE0",
            "RPHbraidingmodel",
            "RPHbraidingmodel_ODE",
            "RPHbraiding_ODE0",
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
                "RPHalphaMmodel",
                "RPHalphaMmodel_ODE",
                "RPHalphaM_ODE0",
                "RPHbraidingmodel",
                "RPHbraidingmodel_ODE",
                "RPHbraiding_ODE0",
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