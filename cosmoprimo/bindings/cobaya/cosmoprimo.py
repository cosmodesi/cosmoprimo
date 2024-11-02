# Global
import sys
import os
import numpy as np
from copy import deepcopy
from typing import NamedTuple, Sequence, Union, Optional, Callable, Any

# Local
from cobaya.theories.cosmo import BoltzmannBase
from cobaya.log import LoggedError, get_logger
from cobaya.install import download_github_release, pip_install, check_gcc_version
from cobaya.component import ComponentNotInstalledError, load_external_module
from cobaya.tools import Pool1D, Pool2D, PoolND, combine_1d, get_compiled_import_path, \
    VersionCheckError

from cobaya.theories.cosmo import BoltzmannBase


# Result collector
class Collector(NamedTuple):
    section: str
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence, None] = None
    z_pool: Optional[PoolND] = None
    post: Optional[Callable] = None


def get_from_cosmo(cosmo, name):
    conversions = {'Omega_nu_massive': 'Omega_ncdm_tot', 'm_nu_massive': 'm_ncdm_tot'}
    name = conversions.get(name, name)
    if name.lower().startswith('omega_'):
        name = name[:5] + '0' + name[5:]
    if name.startswith('omega'):
        return get_from_cosmo(cosmo, 'O' + name[1:]) * cosmo.h ** 2
    scale = None
    if name == 'theta_MC_100':
        name = 'theta_cosmomc'
        scale = 100.
    if name == 'k_pivot':
        return cosmo.k_pivot * cosmo.h
    try:
        toret = getattr(cosmo, name)
    except AttributeError:
        toret = cosmo[name]
    if not toret:
        return 0.
    if scale is not None:
        return scale * toret
    return toret


class cosmoprimo(BoltzmannBase):

    # Name of the cosmoprimo repo/folder and version to download
    _cosmoprimo_repo_name = "cosmodesi/cosmoprimo"
    _min_cosmoprimo_version = "1.0.0"
    _cosmoprimo_repo_version = os.environ.get('COSMOPRIMO_REPO_VERSION', _min_cosmoprimo_version)

    cosmoprimo_module: Any
    ignore_obsolete: bool

    def initialize(self):
        try:
            install_path = (lambda p: self.get_path(p) if p else None)(self.packages_path)
            min_version = None if self.ignore_obsolete else self._cosmoprimo_repo_version
            self.cosmoprimo_module = load_external_module(
                "cosmoprimo", path=self.path, install_path=install_path,
                min_version=min_version, get_import_path=self.get_import_path,
                logger=self.log, not_installed_level="debug")
        except VersionCheckError as excpt:
            raise VersionCheckError(
                str(excpt) + " If you are using cosmoprimo unmodified, upgrade with"
                "`cobaya-install cosmoprimo --upgrade`. If you are using a modified cosmoprimo, "
                "set the option `ignore_obsolete: True` for cosmoprimo.")
        except ComponentNotInstalledError as excpt:
            raise ComponentNotInstalledError(
                self.log, (f"Could not find cosmoprimo: {excpt}. "
                           "To install it, run `cobaya-install cosmoprimo`"))
        super().initialize()
        # Derived parameters that may not have been requested, but will be necessary later
        self.derived_extra = []

    def set_cl_reqs(self, reqs):
        """Sets some common settings for both lensed and unlensed Cl's."""
        # For l_max_scalars, remember previous entries.
        self.extra_args["ellmax_cl"] = \
            max(self.extra_args.get("ellmax_cl", 0), max(reqs.values()))

    def must_provide(self, **requirements):
        # Computed quantities required by the likelihood
        super().must_provide(**requirements)
        conversions_of = {'delta_tot': 'delta_m', 'delta_nonu': 'delta_cb', 'v_newtonian_cdm': 'theta_cdm', 'v_newtonian_baryon': 'theta_b', 'Weyl': 'phi_plus_psi'}

        def get_of(pair):
            pair = list(conversions_of.get(of, of) for of in pair)
            if 'class' in self.engine:
                if 'theta_b' in pair or 'theta_cdm' in pair:
                    import warnings
                    warnings.warn('cosmoprimo - pyclass wrappings do not expose theta_b, theta_cdm individually; will return theta_cb. It is fine if you only need theta_cb (e.g. RSD analysis), but in other cases please post an issue on the cosmoprimo github.')
                pair = [{'theta_b': 'theta_cb', 'theta_cdm': 'theta_cb'}.get(of, of) for of in pair]
            return tuple(pair)

        for k, v in self._must_provide.items():
            # Products and other computations
            if k == "Cl":
                self.set_cl_reqs(v)
                # For modern experiments, always lensed Cl's!
                self.extra_args["lensing"] = True
                self.extra_args.setdefault('non_linear', 'hmcode')
                self.collectors[k] = Collector(section="harmonic", method="lensed_cl")
            elif k == "unlensed_Cl":
                self.set_cl_reqs(v)
                self.collectors[k] = Collector(section="harmonic", method="unlensed_cl")
            elif k == "Hubble":
                self.set_collector_with_z_pool(k, v["z"], section="background", method="hubble_function", args_names=["z"])
            elif k in ["Omega_b", "Omega_cdm", "Omega_nu_massive"]:
                func_name = {"Omega_nu_massive": "Omega_ncdm_tot"}.get(k, k)
                self.set_collector_with_z_pool(k, v["z"], section="background", method=func_name, args_names=["z"])
            elif k == "angular_diameter_distance":
                self.set_collector_with_z_pool(k, v["z"], section="background", method="angular_diameter_distance", args_names=["z"])
            elif k == "comoving_radial_distance":
                self.set_collector_with_z_pool(k, v["z"], section="background", method="comoving_radial_distance", args_names=["z"])
            elif k == "angular_diameter_distance_2":
                self.set_collector_with_z_pool(k, v["z_pairs"], section="background", method="angular_diameter_distance_2", args_names=["z1", "z2"], d=2)
            elif isinstance(k, tuple) and k[0] == "Pk_grid":
                v = deepcopy(v)
                kmax = v.pop("k_max")
                self.add_P_k_max(kmax, units="1/Mpc")
                self.add_z_for_matter_power(v.pop("z"))
                if v["nonlinear"]:
                    if "non_linear" not in self.extra_args:
                        # this is redundant with initialisation, but just in case
                        self.extra_args["non_linear"] = True
                    elif not self.extra_args["non_linear"]:
                        raise LoggedError(
                            self.log, (f"Non-linear Pk requested, but `non_linear: {self.extra_args['non_linear']}` imposed in `extra_args`"))
                pair = k[2:]
                v["of"] = get_of(pair)
                v['non_linear'] = v.pop('nonlinear')
                v['extrap_kmax'] = 10 * kmax
                self.collectors[k] = Collector(section="fourier", method="pk_interpolator", kwargs=v)
            elif k == "sigma8_z":
                self.add_z_for_matter_power(v["z"])
                self.set_collector_with_z_pool(k, v["z"], section="fourier", method="sigma8_z", args_names=["z"], kwargs={'of': 'delta_m'})
            elif k == "fsigma8":
                self.add_z_for_matter_power(v["z"])
                self.set_collector_with_z_pool(k, v["z"], section="fourier", method="sigma8_z", args_names=["z"], kwargs={'of': 'theta_cb'})
            elif isinstance(k, tuple) and k[0] == "sigma_R":
                self.add_P_k_max(v.pop("k_max"), units="1/Mpc")
                self.add_z_for_matter_power(v["z"])
                pair = k[1:]
                v["of"] = get_of(pair)
                self.collectors[k] = Collector(section="fourier", method="sigma_rz", args=[v["R"], v["z"]], args_names=["R", "z"])
            elif k in [f"get_{q}" for q in ["background", "thermodynamics", "primordial", "perturbations"]]:
                # Get direct cosmoprimo results
                self.collectors[k] = Collector(section=q)
            elif v is None:
                k_translated = self.translate_param(k)
                if k_translated not in self.derived_extra:
                    self.derived_extra += [k_translated]
            else:
                raise LoggedError(self.log, "Requested product not known: %r", {k: v})
        # Derived parameters (if some need some additional computations)
        if any(("sigma" in s) for s in set(self.output_params).union(requirements)):
            self.add_P_k_max(1, units="1/Mpc")
        # Adding tensor modes if requested
        if self.extra_args.get("r") or "r" in self.input_params:
            self.extra_args["modes"] = ["s", "t"]
        # If B spectrum with l>50, or lensing, recommend using a non-linear code
        cls = self._must_provide.get("Cl", {})
        has_BB_l_gt_50 = (any(("b" in cl.lower()) for cl in cls) and
                          max(cls[cl] for cl in cls if "b" in cl.lower()) > 50)
        has_lensing = any(("p" in cl.lower()) for cl in cls)
        if (has_BB_l_gt_50 or has_lensing) and not self.extra_args.get("non_linear"):
            self.log.warning("Requesting BB for ell>50 or lensing Cl's: "
                             "using a non-linear code is recommended (and you are not "
                             "using any). To activate it, set "
                             "'non_linear: halofit|hmcode|...' in cosmoprimo's 'extra_args'.")
        self.check_no_repeated_input_extra()

    def add_z_for_matter_power(self, z):
        if getattr(self, "z_for_matter_power", None) is None:
            self.z_for_matter_power = np.empty(0)
        self.z_for_matter_power = np.flip(combine_1d(z, self.z_for_matter_power))
        self.extra_args["z_pk"] = self.z_for_matter_power

    def set_collector_with_z_pool(self, k, zs, section=None, method=None, args=(), args_names=(),
                                  kwargs=None, arg_array=None, post=None, d=1):
        """
        Creates a collector for a z-dependent quantity, keeping track of the pool of z's.

        If ``z`` is an arg, i.e. it is in ``args_names``, then omit it in the ``args``,
        e.g. ``args_names=["a", "z", "b"]`` should be passed together with
        ``args=[a_value, b_value]``.
        """
        if k in self.collectors:
            z_pool = self.collectors[k].z_pool
            z_pool.update(zs)
        else:
            Pool = {1: Pool1D, 2: Pool2D}[d]
            z_pool = Pool(zs)
        # Insert z as arg or kwarg
        kwargs = kwargs or {}
        if d == 1 and "z" in kwargs:
            kwargs = deepcopy(kwargs)
            kwargs["z"] = z_pool.values
        elif d == 1 and "z" in args_names:
            args = deepcopy(args)
            i_z = args_names.index("z")
            args = list(args[:i_z]) + [z_pool.values] + list(args[i_z:])
        elif d == 2 and "z1" in args_names and "z2" in args_names:
            # z1 assumed appearing before z2!
            args = deepcopy(args)
            i_z1 = args_names.index("z1")
            i_z2 = args_names.index("z2")
            args = (list(args[:i_z1]) + [z_pool.values[:, 0]] + list(args[i_z1:i_z2]) + [z_pool.values[:, 1]] + list(args[i_z2:]))
        else:
            raise LoggedError(
                self.log,
                f"I do not know how to insert the redshift for collector method {method} "
                f"of requisite {k}")
        self.collectors[k] = Collector(
            section=section, method=method, z_pool=z_pool, args=args, args_names=args_names, kwargs=kwargs,
            arg_array=arg_array, post=post)

    def add_P_k_max(self, k_max, units):
        r"""
        Unifies treatment of :math:`k_\mathrm{max}` for matter power spectrum:
        ``P_k_max_[1|h]/Mpc``.

        Make ``units="1/Mpc"|"h/Mpc"``.
        """
        # Fiducial h conversion (high, though it may slow the computations)
        h_fid = 1
        if units == "h/Mpc":
            k_max *= h_fid
        # Take into account possible manual set of P_k_max_***h/Mpc*** through extra_args
        k_max_old = self.extra_args.pop(
            "kmax_pk", h_fid * self.extra_args.pop("kmax_pk", 0))
        self.extra_args["kmax_pk"] = max(k_max, k_max_old)

    def set(self, params_values_dict):
        # Prepare parameters to be passed: this-iteration + extra
        args = {self.translate_param(p): v for p, v in params_values_dict.items()}
        args.update(self.extra_args)
        # Generate and save
        self.log.debug("Setting parameters: %r", args)
        try:
            self.cosmo = self.cosmoprimo_module.Cosmology(**args, engine=self.engine)
        except self.cosmoprimo_module.CosmologyError as e:
            self.log.error("Serious error setting parameters. The parameters passed were %r. "
                           "To see the original cosmoprimo error traceback, make 'debug: True'.", args)
            raise  # No LoggedError, so that cosmoprimo traceback gets printed

    def calculate(self, state, want_derived=True, **params_values_dict):
        # Set parameters
        self.set(params_values_dict)
        self.cosmo.get_background()
        # Gather products
        for product, collector in self.collectors.items():
            try:
                section = getattr(self.cosmo, 'get_{}'.format(collector.section))()
                method = getattr(section, collector.method)
            # cosmoprimo not correctly initialized, or input parameters not correct
            except self.cosmoprimo_module.CosmologyError as e:
                self.log.error("Serious error setting parameters or computing results. "
                               "The parameters passed were %r and %r. To see the original "
                                "cosmoprimo error traceback, make 'debug: True'.",
                                state["params"], self.extra_args)
                raise  # No LoggedError, so that cosmoprimo traceback gets printed
            args = collector.args
            if product[0] == "sigma_R":
                args[0] = args[0] * self.cosmo.h
            result = method(*args, **collector.kwargs)
            if collector.post:
                result = collector.post(*result)
            if 'distance' in product:
                result /= self.cosmo.h
            if product == 'Hubble':
                result /= (self.cosmoprimo_module.constants.c / 1e3)
            if product[0] == "Pk_grid":
                h = self.cosmo.h
                pair = product[2:]
                nweyl = sum(of == 'Weyl' for of in pair)
                kmin, kmax = 1e-4, self.extra_args["kmax_pk"]
                nk = 125 * int(np.log10(kmax / kmin) + 0.5)
                k = np.geomspace(1e-4, self.extra_args["kmax_pk"], nk)
                z = np.copy(self.z_for_matter_power)
                pk = result(k / h, z, grid=True).T
                #k = result.k * h
                #z = result.z
                #pk = result.pk.T
                # We returned (phi + psi), but we want k^2 (phi + psi) / 2
                pk = pk / h**3 * k**(2 * nweyl) / 2**nweyl
                result = (k, z, pk)
            if product[0] == "sigma_R":
                result = (args[1], args[0], result.T)  # z, r, sigma
            if 'Cl' in product:
                result = {name: result[name] for name in result.dtype.names}
                if collector.method == 'lensed_cl':
                    tmp = self.cosmo.get_harmonic().lens_potential_cl()
                    result.update({name: tmp[name] for name in tmp.dtype.names})
            state[product] = result
        # Prepare derived parameters
        d, d_extra = self._get_derived_all(derived_requested=want_derived)
        if want_derived:
            state["derived"] = {p: d.get(p) for p in self.output_params}
            # Prepare necessary extra derived parameters
        state["derived_extra"] = deepcopy(d_extra)

    def _get_derived_all(self, derived_requested=True):
        """
        Returns a dictionary of derived parameters with their values,
        using the *current* state (i.e. it should only be called from
        the ``compute`` method).

        Parameter names are returned in cosmoprimo nomenclature.

        To get a parameter *from a likelihood* use `get_param` instead.
        """
        requested = [self.translate_param(p) for p in (
            self.output_params if derived_requested else [])]
        requested_and_extra = dict.fromkeys(set(requested).union(self.derived_extra))
        requested_and_extra.update({p: get_from_cosmo(self.cosmo, p) for p, v in requested_and_extra.items() if v is None})
        for name in ['rs_drag']:
            if name in requested_and_extra:
                requested_and_extra[name] /= self.cosmo.h
        # Separate the parameters before returning
        # Remember: self.output_params is in sampler nomenclature,
        # but self.derived_extra is in cosmoprimo
        derived = {p: requested_and_extra[self.translate_param(p)] for p in self.output_params}
        derived_extra = {p: requested_and_extra[p] for p in self.derived_extra}
        return derived, derived_extra

    def _get_Cl(self, ell_factor=False, units="FIRASmuK2", lensed=True):
        which_key = "Cl" if lensed else "unlensed_Cl"
        which_error = "lensed" if lensed else "unlensed"
        try:
            cls = deepcopy(self.current_state[which_key])
        except:
            raise LoggedError(self.log, "No %s Cl's were computed. Are you sure that you "
                                        "have requested them?", which_error)
        # unit conversion and ell_factor
        ells_factor = \
            ((cls["ell"] + 1) * cls["ell"] / (2 * np.pi))[2:] if ell_factor else 1
        units_factor = self._cmb_unit_factor(units, self.cosmo['T_cmb'])
        for cl in cls:
            if cl not in ['pp', 'ell']:
                cls[cl][2:] *= units_factor ** 2 * ells_factor
        if lensed and "pp" in cls and ell_factor:
            cls['pp'][2:] *= ells_factor ** 2 * (2 * np.pi)
        return cls

    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        return self._get_Cl(ell_factor=ell_factor, units=units, lensed=True)

    def get_unlensed_Cl(self, ell_factor=False, units="FIRASmuK2"):
        return self._get_Cl(ell_factor=ell_factor, units=units, lensed=False)

    def get_background(self):
        """Direct access to ``get_background`` from the cosmoprimo interface."""
        return self.cosmo.get_background()

    def get_thermodynamics(self):
        """Direct access to ``get_thermodynamics`` from the cosmoprimo interface."""
        return self.cosmo.get_thermodynamics()

    def get_primordial(self):
        """Direct access to ``get_primordial`` from the cosmoprimo interface."""
        return self.cosmo.get_primordial()

    def get_perturbations(self):
        """Direct access to ``get_perturbations`` from the cosmoprimo interface."""
        return self.cosmo.get_perturbations()

    def get_fourier(self):
        """Direct access to ``get_fourier`` from the cosmoprimo interface."""
        return self.cosmo.get_fourier()

    def get_harmonic(self):
        """Direct access to ``get_harmonic`` from the cosmoprimo interface."""
        return self.cosmo.get_harmonic()

    def close(self):
        del self.cosmo

    def get_can_provide_params(self):
        names = ['h', 'H0', 'Omega_Lambda', 'Omega_m', 'Omega_k',
                 'rs_drag', 'z_drag', 'tau_reio', 'z_reio', 'z_rec', 'tau_rec', 'm_ncdm_tot',
                 'N_eff', 'YHe', 'age', 'sigma8_m', 'sigma8_cb', 'theta_s_100']
        for name, mapped in self.renames.items():
            if mapped in names:
                names.append(name)
        return names

    def get_can_support_params(self):
        # non-exhaustive list of supported input parameters that will be assigned to
        # cosmoprimo if they are varied
        return ['H0']

    def get_version(self):
        return getattr(self.cosmoprimo_module, '__version__', None)

    # Installation routines

    @classmethod
    def get_path(cls, path):
        return os.path.realpath(os.path.join(path, "code", cls.__name__))

    @staticmethod
    def get_import_path(path):
        return get_compiled_import_path(os.path.join(path, "python"))

    @classmethod
    def is_compatible(cls):
        import platform
        if platform.system() == "Windows":
            return False
        return True

    @classmethod
    def is_installed(cls, reload=False, **kwargs):
        if not kwargs.get("code", True):
            return True
        try:
            return bool(load_external_module(
                "cosmoprimo", path=kwargs["path"], get_import_path=cls.get_import_path,
                min_version=cls._cosmoprimo_repo_version, reload=reload,
                logger=get_logger(cls.__name__), not_installed_level="debug"))
        except ComponentNotInstalledError:
            return False

    @classmethod
    def install(cls, path=None, code=True, no_progress_bars=False, **_kwargs):
        log = get_logger(cls.__name__)
        if not code:
            log.info("Code not requested. Nothing to do.")
            return True
        log.info("Downloading cosmoprimo...")
        success = download_github_release(
            os.path.join(path, "code"), cls._cosmoprimo_repo_name, cls._cosmoprimo_repo_version,
            directory=cls.__name__, no_progress_bars=no_progress_bars, logger=log)
        if not success:
            log.error("Could not download cosmoprimo.")
            return False
        cosmoprimo_path = cls.get_path(path)
        log.info("Compiling cosmoprimo...")
        from subprocess import Popen, PIPE
        process_make = Popen([sys.executable, "setup.py", "install"],
                             cwd=cosmoprimo_path, stdout=PIPE, stderr=PIPE)
        out, err = process_make.communicate()
        if process_make.returncode:
            log.info(out.decode())
            log.info(err.decode())
        return True
