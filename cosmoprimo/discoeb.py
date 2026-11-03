r"""Cosmological calculation with the differentiable Einstein-Boltzmann solver DISCO-EB.

DISCO-EB (`github.com/ohahn/DISCO-EB <https://github.com/ohahn/DISCO-EB>`_) is a linear
Einstein-Boltzmann solver written in JAX, a module of the DISCO-DJ framework.  It is
autodifferentiable, and solves the synchronous-gauge Boltzmann hierarchy for photons, massless
and (one degenerate family of) massive neutrinos, cold dark matter, baryons and a quintessence
dark-energy fluid, together with a RECFAST (or MB95) thermal history.

References
----------
Hahn et al. 2024, https://arxiv.org/abs/2311.03291
https://github.com/ohahn/DISCO-EB

Note
----
DISCO-EB requires ``jax``, ``diffrax``, ``equinox``, ``jaxtyping``, ``jax_cosmo`` and ``lineax``,
and 64-bit precision (``jax_enable_x64``, which :mod:`cosmoprimo.jax` already turns on).  It is
developed for, and by far fastest on, GPU platforms; on CPU the JIT compilation of the (stiff,
vmapped) perturbation solver is slow and memory-hungry.

Only the *linear* matter power spectrum is produced (DISCO-EB has no non-linear module and does
not compute CMB anisotropies), so the :class:`Harmonic` section is not available and
``non_linear`` must be turned off.

Differentiability
-----------------
DISCO-EB is autodifferentiable, and the engine keeps the whole ``parameters -> P(k)`` chain
traceable, so derivatives / Jacobians of the matter power spectrum w.r.t. the cosmological
parameters are obtained with cosmoprimo's standard jax pattern::

    import jax
    from cosmoprimo import Cosmology

    def pk(params):
        return Cosmology(**params, engine='discoeb').get_fourier().table(of='delta_m')[2]

    jax.jacfwd(pk)(dict(Omega_cdm=0.25, Omega_b=0.05, h=0.677, A_s=2.1e-9, n_s=0.968))

DISCO-EB's *internal* splines (its thermal history, neutrino background, ...) are simply traced
through as part of the differentiable solve; they are never exposed.  On the cosmoprimo side the
power spectrum is handed to cosmoprimo's own :class:`~cosmoprimo.PowerSpectrumInterpolator2D`,
which is itself jax-friendly (via ``interpax``), so the quantities built on it --
:meth:`Fourier.pk_interpolator`, :meth:`Fourier.sigma_rz`, ``sigma8`` -- are differentiable too,
exactly as for the other jax-friendly engines.  Only :class:`Thermodynamics` and :class:`Transfer`
post-process DISCO-EB's output with ``numpy.interp`` and are therefore evaluated concretely.
As for the values themselves, differentiating requires DISCO-EB's (heavy) perturbation solve to be
JIT-compiled.
"""

import warnings

import numpy as np

from .cosmology import (BaseEngine, BaseSection, DefaultBackground, CosmologyInputError,
                        CosmologyComputationError, _compute_rs_cosmomc)
from .interpolator import PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D
from . import utils
from .jax import exception


#: DISCO-EB output variables and their index in the state vector returned by
#: ``discoeb.perturbations.evolve_perturbations`` (see
#: ``discoeb.perturbations.convert_to_output_variables``).  ``theta_*`` variables are the
#: velocity divergences divided by the conformal Hubble rate :math:`\theta / (a H)`, which is the
#: normalisation :mod:`cosmoprimo` uses (so that :math:`P_{\theta\theta} = f^{2} P_{\delta\delta}`).
DISCOEB_VARIABLES = {
    'eta': 0, 'eta_prime': 1, 'h_prime': 2, 'alpha': 3,
    'delta_m': 4, 'theta_m': 5,
    'delta_cb': 6, 'theta_cb': 7,
    'delta_c': 8, 'theta_c': 9,
    'delta_b': 10, 'theta_b': 11,
    'delta_g': 12, 'theta_g': 13,
    'delta_ur': 14, 'theta_ur': 15,
    'delta_ncdm': 16, 'theta_ncdm': 17,
    'delta_fld': 18, 'theta_fld': 19,
}

#: Names of the perturbed quantities the :class:`Fourier` (and :class:`Transfer`) sections accept
#: as ``of``, mapped to the DISCO-EB state-vector index.
_OF_TO_INDEX = {name: index for name, index in DISCOEB_VARIABLES.items()
                if name.startswith(('delta_', 'theta_'))}

# Atomic masses (in atomic mass units), for the helium nucleon -> mass fraction conversion
# (same convention as CAMB's camb.bbn).
_M_H = 1.00782503207
_M_HE4 = 4.00260325413

# Standard (instantaneous-decoupling) neutrino-to-photon temperature ratio.
_TNU_OVER_TCMB_STANDARD = (4. / 11.)**(1. / 3.)

#: DISCO-EB's quintessence equations of motion (Ballesteros & Lesgourgues 2010, eqs. 3.5-3.6)
#: contain a ``cs2_DE / (1 + w_DE)`` term that is ``0 / 0`` for a pure cosmological constant
#: (``w0_fld = -1``, cosmoprimo's default): the density perturbation starts at zero (it is
#: proportional to ``1 + w_DE``) so the ratio is finite but the solver only sees ``inf * 0 = nan``
#: and then rejects every single ODE step.  Offsetting ``w_DE_0`` by this small amount removes the
#: singularity.  Because the offset is *constant* it shifts ``w_DE`` without changing
#: ``dw_DE0 / dw0_fld = 1``, so autodiff w.r.t. ``w0_fld`` is unaffected; the induced change in the
#: background is ``O(3 * eps * ln(1/a)) ~ 1e-5`` at ``a = 1e-3`` and entirely negligible for
#: ``w0_fld > -1``.
_ONE_PLUS_W_EPS = 1e-6


def _yhe_from_yp(Yp):
    r"""Convert the BBN helium *nucleon* fraction :math:`Y_{p}` to the helium *mass* fraction."""
    return -Yp * _M_HE4 / (-Yp * _M_HE4 + 4. * Yp * _M_H - 4. * _M_H)


def yhe_bbn(omega_b, N_eff=3.044, tau_neutron=880.3):
    r"""
    Primordial helium mass fraction from Big Bang Nucleosynthesis.

    Uses the PArthENoPE fitting formulae of the Planck 2015 papers (as implemented in
    ``camb.bbn.BBN_fitting_parthenope``), for a neutron lifetime of ``tau_neutron``.

    Parameters
    ----------
    omega_b : float
        Physical baryon density :math:`\Omega_{b} h^{2}`.

    N_eff : float, default=3.044
        Effective number of relativistic species; the fit is expressed in terms of
        :math:`\Delta N_\mathrm{eff}` relative to 3.046.

    tau_neutron : float, default=880.3
        Neutron lifetime, in seconds.

    Returns
    -------
    YHe : float
        Primordial helium mass fraction.
    """
    delta_neff = N_eff - 3.046
    Yp = (0.2311 + 0.9502 * omega_b - 11.27 * omega_b**2
          + delta_neff * (0.01356 + 0.008581 * omega_b - 0.1810 * omega_b**2)
          + delta_neff**2 * (-0.0009795 - 0.001370 * omega_b + 0.01746 * omega_b**2))
    Yp *= (tau_neutron / 880.3)**0.728
    return _yhe_from_yp(Yp)


def _import_discoeb():
    """Import DISCO-EB (and jax), raising a helpful error if it is not installed."""
    try:
        import jax
        import jax.numpy as jnp
        from discoeb.background import evolve_background
        from discoeb.perturbations import (evolve_perturbations, evolve_perturbations_batched,
                                           get_power)
    except ImportError as exc:
        raise CosmologyInputError(
            'Could not import DISCO-EB; please install it with '
            '"pip install cosmoprimo[discoeb]" (or, directly, '
            '"pip install git+https://github.com/dforero0896/DISCO-EB.git@master"). '
            'It requires jax, diffrax, equinox, jaxtyping, jax_cosmo and lineax. '
            'Original error: {}'.format(exc)) from exc
    # DISCO-EB requires double precision; cosmoprimo.jax already sets this, but be defensive.
    if not jax.config.jax_enable_x64:
        jax.config.update('jax_enable_x64', True)
    return jax, jnp, evolve_background, evolve_perturbations, evolve_perturbations_batched, get_power


#: Compiled-function cache.  ``jax.jit`` caches on the *jitted callable object*, so rebuilding the
#: wrapper for every engine -- which is what constructing many :class:`Cosmology` instances does,
#: e.g. a finite-difference or ``jax.jacfwd`` sweep -- would recompile every time (each DISCO-EB
#: compile is ~10-15 s on CPU).  Keying on the static settings plus the solver callables lets all
#: engines with the same settings share one compilation; the callables are in the key so that a
#: monkeypatched/stubbed DISCO-EB can never pick up a real compiled function.
_JIT_CACHE = {}


def _discoeb_background_jit(evolve_background, thermo_module, num_thermo, rtol, atol):
    """Return a jitted ``evolve_background`` for the given (static) thermal-history settings."""
    import jax

    key = ('background', evolve_background, thermo_module, int(num_thermo), float(rtol),
           float(atol))
    if key not in _JIT_CACHE:
        def background(param):
            return evolve_background(param=param, thermo_module=thermo_module,
                                     num_thermo=num_thermo, rtol=rtol, atol=atol)

        _JIT_CACHE[key] = jax.jit(background)
    return _JIT_CACHE[key]


def _discoeb_solve_jit(evolve_background, evolve_perturbations, evolve_perturbations_batched,
                       *, nk, Nmnu, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax,
                       rtol, atol, max_steps, pcoeff, icoeff, dcoeff, factormax, factormin,
                       thermo_module, num_thermo, bg_rtol, bg_atol, batched, batch_size):
    r"""
    Build a jitted function computing the DISCO-EB perturbations.

    The returned callable takes the (differentiable) dictionary of DISCO-EB cosmological
    parameters ``param``, the array of output scale factors ``aexp_out`` and the (differentiable)
    wavenumber bounds ``kmin``, ``kmax`` (in :math:`\mathrm{Mpc}^{-1}`), and returns
    ``(kmodes, y)``, where ``kmodes`` has shape ``(nk,)`` and ``y`` the perturbations, of shape
    ``(nk, len(aexp_out), 20)``.  All of ``param``, ``aexp_out``, ``kmin`` and ``kmax`` are traced,
    so the whole solve is autodifferentiable (DISCO-EB's headline feature); only the solver
    settings above are compile-time constants.

    The jitted callable is cached (see :data:`_JIT_CACHE`) so that engines sharing the same static
    settings reuse a single compilation.
    """
    import jax

    key = ('solve', evolve_background, evolve_perturbations, evolve_perturbations_batched,
           nk, Nmnu, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax, rtol, atol, max_steps, pcoeff, icoeff,
           dcoeff, factormax, factormin, thermo_module, num_thermo, bg_rtol, bg_atol, batched,
           batch_size)
    if key in _JIT_CACHE:
        return _JIT_CACHE[key]

    def solve(param, aexp_out, kmin, kmax):
        param = dict(param)
        # ``Nmnu`` is a count, hence a compile-time (static) constant of the traced function.
        param['Nmnu'] = Nmnu
        param = evolve_background(param=param, thermo_module=thermo_module, num_thermo=num_thermo,
                                  rtol=bg_rtol, atol=bg_atol)
        common = dict(kmin=kmin, kmax=kmax, num_k=nk, aexp_out=aexp_out,
                      lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax,
                      rtol=rtol, atol=atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff,
                      factormax=factormax, factormin=factormin, max_steps=max_steps)
        if batched:
            y, kmodes = evolve_perturbations_batched(param=param, batch_size=batch_size, **common)
        else:
            y, kmodes, _ = evolve_perturbations(param=param, **common)
        return kmodes, y

    _JIT_CACHE[key] = jax.jit(solve)
    return _JIT_CACHE[key]


class DISCOEBEngine(BaseEngine):
    r"""
    Engine for the differentiable Einstein-Boltzmann solver DISCO-EB.

    DISCO-EB solves for the linear matter (and photon, neutrino, dark-energy) perturbations,
    from which the matter power spectrum is built.  It supports the standard
    :math:`\Lambda\mathrm{CDM}` model extended with a :math:`(w_{0}, w_{a})` quintessence dark
    energy fluid, spatial curvature and one family of ``N_ncdm`` *degenerate* massive neutrinos.

    Extra parameters (precision / solver settings), passed as e.g.
    ``Cosmology(engine='discoeb', nk=512, rtol=1e-4)``:

    - ``nk`` (int, default=400): number of wavenumbers, log-spaced between ``kmin`` and ``kmax``.
    - ``kmin``, ``kmax`` (float, default=1e-4 and ``kmax_pk``): wavenumber range, in
      :math:`h/\mathrm{Mpc}` (internally converted to :math:`\mathrm{Mpc}^{-1}`).
    - ``z`` (array, default=``z_pk``): redshifts at which the power spectrum is computed.
    - ``thermo_module`` (str, default='RECFAST'): thermal history module, 'RECFAST', 'MB95'.
    - ``num_thermo`` (int, default=1024): number of thermal-history sampling points.
    - ``rtol``, ``atol`` (float, default=1e-3): perturbation ODE solver tolerances.
    - ``bg_rtol``, ``bg_atol`` (float, default=1e-5, 1e-7): background/thermal ODE tolerances.
    - ``lmaxg``, ``lmaxgp``, ``lmaxr`` (int, default=11): photon temperature / polarization and
      massless-neutrino Boltzmann hierarchy truncation.
    - ``lmaxnu`` (int, default=8), ``nqmax`` (int, default=3): massive-neutrino hierarchy
      truncation and number of momentum bins.
    - ``max_steps`` (int, default=2048): maximum number of ODE solver steps.
    - ``batched`` (bool, default=False): use the batched solver (recommended on GPU).
    - ``batch_size`` (int, default=16): number of modes per batch, if ``batched``.
    - ``YHe`` (float, default=None): primordial helium mass fraction; if ``None``, taken from the
      cosmology's ``YHe`` parameter (computed from BBN consistency if it is ``'BBN'``).

    References
    ----------
    Hahn et al. 2024, https://arxiv.org/abs/2311.03291
    https://github.com/ohahn/DISCO-EB
    """
    name = 'discoeb'

    #: Extra (precision) parameters and their default values.
    _default_extra_params = dict(
        nk=400, kmin=1e-4, kmax=None, z=None,
        thermo_module='RECFAST', num_thermo=1024,
        rtol=1e-3, atol=1e-3, bg_rtol=1e-5, bg_atol=1e-7,
        lmaxg=11, lmaxgp=11, lmaxr=11, lmaxnu=8, nqmax=3,
        max_steps=16384, pcoeff=0.25, icoeff=0.80, dcoeff=0.0,
        factormax=20.0, factormin=0.3,
        batched=False, batch_size=16, YHe=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- Extra (precision) parameters, with defaults ---
        extra = dict(self._default_extra_params)
        extra.update(self._extra_params)
        self._extra_params = extra
        for name in ['nk', 'num_thermo', 'lmaxg', 'lmaxgp', 'lmaxr', 'lmaxnu', 'nqmax',
                     'max_steps', 'batch_size']:
            extra[name] = int(extra[name])
        for name in ['rtol', 'atol', 'bg_rtol', 'bg_atol', 'pcoeff', 'icoeff', 'dcoeff',
                     'factormax', 'factormin', 'kmin']:
            extra[name] = float(extra[name])
        if extra['kmax'] is None:
            extra['kmax'] = float(self['kmax_pk'])
        else:
            extra['kmax'] = float(extra['kmax'])
        if extra['kmin'] >= extra['kmax']:
            raise CosmologyInputError(
                'discoeb: kmin ({}) must be smaller than kmax ({})'.format(
                    extra['kmin'], extra['kmax']))
        if extra['batched'] and extra['nk'] % extra['batch_size']:
            raise CosmologyInputError(
                'discoeb: nk ({}) must be a multiple of batch_size ({}) when batched'.format(
                    extra['nk'], extra['batch_size']))

        # --- Reject / warn about unsupported features ---
        # ``non_linear`` is a string parameter (always concrete): DISCO-EB is linear-only.
        if self['non_linear']:
            raise CosmologyInputError(
                '{} is a linear-only solver; it cannot compute the non-linear power spectrum '
                '(non_linear={}). Please set non_linear=False.'.format(
                    self.__class__.__name__, self['non_linear']))

        def warn(N_ncdm, m_ncdm, alpha_s, beta_s, T_ncdm_over_cmb):
            name = self.__class__.__name__
            if N_ncdm > 1:
                m_ncdm = np.asarray(m_ncdm, dtype='f8')
                if not np.allclose(m_ncdm, m_ncdm[0], rtol=1e-3):
                    warnings.warn(
                        '{} treats the {} massive neutrinos as a single degenerate family of '
                        'mean mass {:.4g} eV (sum {:.4g} eV); non-degenerate masses are not '
                        'supported.'.format(name, N_ncdm, m_ncdm.mean(), m_ncdm.sum()))
            if np.any(np.abs(np.asarray(T_ncdm_over_cmb, dtype='f8')
                            / _TNU_OVER_TCMB_STANDARD - 1.) > 0.02):
                warnings.warn(
                    '{} assumes the standard neutrino temperature T_ncdm = (4/11)^(1/3) T_cmb; '
                    'the requested T_ncdm_over_cmb={} departs from it.'.format(
                        name, T_ncdm_over_cmb))
            if alpha_s:
                warnings.warn('{} does not implement running of the spectral index; '
                              'alpha_s={} is ignored.'.format(name, alpha_s))
            if beta_s:
                warnings.warn('{} does not implement running of the running; '
                              'beta_s={} is ignored.'.format(name, beta_s))
        exception(warn, self['N_ncdm'], self['m_ncdm'] if self['N_ncdm'] else [0.],
                  self['alpha_s'], self['beta_s'],
                  self['T_ncdm_over_cmb'] if self['N_ncdm'] else [_TNU_OVER_TCMB_STANDARD])

        # --- First guess for A_s (exact if sigma8 was requested: DISCO-EB P(k) is linear in A_s) ---
        self._A_s = self._get_A_s_fid()

        # --- Compile the DISCO-EB cosmological parameters ---
        self._discoeb_params = self._get_discoeb_params()

        # --- Redshift grid ---
        # keep the extra parameters serialisable (state dict / write-to-disk): an array ``z``
        # provided by the user is stored as a plain list.
        if extra['z'] is not None:
            extra['z'] = [float(zz) for zz in np.asarray(extra['z'], dtype='f8').ravel()]
        z = extra['z']
        if z is None:
            z = np.asarray(self['z_pk'], dtype='f8')
        z = np.unique(np.asarray(z, dtype='f8'))
        if not np.any(z == 0.):
            z = np.unique(np.concatenate([[0.], z]))
        self._z_pk = z

        # --- Lazy caches for the (expensive) DISCO-EB computations ---
        self._discoeb_bg = None
        self._discoeb_pert = None

    def _get_discoeb_params(self):
        r"""
        Map the :class:`Cosmology` parameters to DISCO-EB's input parameter dictionary.

        Notes
        -----
        DISCO-EB's ``Omegam`` is the cold dark matter + baryon density (massive neutrinos are
        accounted for separately, through ``mnu`` and ``Nmnu``), ``Neff`` is the number of
        *massless* neutrino species (so that ``Neff + Nmnu = N_eff``), and ``H0`` is in
        :math:`\mathrm{km}/\mathrm{s}/\mathrm{Mpc}`.

        ``w_DE_0`` is offset by :data:`_ONE_PLUS_W_EPS` to stay away from the ``1 + w_DE = 0``
        singularity of DISCO-EB's quintessence equations (see the note there), which is what makes
        the default ``w0_fld = -1`` models solvable.
        """
        N_ncdm = int(self['N_ncdm'])
        m_ncdm_tot = self['m_ncdm_tot'] if N_ncdm else 0.
        # DISCO-EB models a single degenerate massive family of Nmnu neutrinos of mass mnu each.
        Nmnu = N_ncdm
        mnu = m_ncdm_tot / Nmnu if Nmnu else 0.
        # Number of massless species: DISCO-EB adds one relativistic d.o.f. per massive species.
        # NB: values are kept as-is (not cast to ``float``) so that they may be jax tracers, which
        # is what makes the engine differentiable w.r.t. the cosmological parameters.
        Neff = self['N_eff'] - Nmnu

        YHe = self._extra_params.get('YHe', None)
        if YHe is None:
            YHe = self['YHe']
            if isinstance(YHe, str):  # 'BBN'
                YHe = yhe_bbn(self['omega_b'], N_eff=self['N_eff'])

        return dict(
            Omegam=self['Omega_cdm'] + self['Omega_b'],
            Omegab=self['Omega_b'],
            w_DE_0=self['w0_fld'] + _ONE_PLUS_W_EPS,
            w_DE_a=self['wa_fld'],
            cs2_DE=self['cs2_fld'],
            Omegak=self['Omega_k'],
            A_s=self._A_s,
            n_s=self['n_s'],
            H0=self['H0'],
            Tcmb=self['T_cmb'],
            YHe=YHe,
            Neff=Neff,
            Nmnu=Nmnu,
            mnu=mnu,
            k_p=self['k_pivot'])

    def _get_kgrid(self):
        r"""Return the ``(kmin, kmax)`` wavenumber bounds in :math:`\mathrm{Mpc}^{-1}`."""
        h = self['h']  # may be a jax tracer; kept as-is for differentiability
        # cosmoprimo k is in h/Mpc; DISCO-EB k is in 1/Mpc:  k[1/Mpc] = k[h/Mpc] * h
        return self._extra_params['kmin'] * h, self._extra_params['kmax'] * h

    def get_discoeb_background(self):
        r"""
        Run (and cache) DISCO-EB's background and thermal-history evolution.

        Returns
        -------
        bg : dict
            Dictionary of numpy arrays / floats holding the DISCO-EB thermal history
            (``aexp``, ``tau``, ``xe``, ``opac``, ``Tm``) and the background densities
            (``grhom``, ``grhog``, ``Omegab``) needed by :class:`Thermodynamics`.
        """
        if self._discoeb_bg is None:
            jax, jnp, evolve_background, _, _, _ = _import_discoeb()
            extra = self._extra_params
            jit_background = _discoeb_background_jit(
                evolve_background, extra['thermo_module'], extra['num_thermo'],
                extra['bg_rtol'], extra['bg_atol'])
            param = {name: jnp.asarray(value, dtype='f8')
                     for name, value in self._discoeb_params.items()}
            param['Nmnu'] = int(param['Nmnu'])
            out = jit_background(param)
            aexp = np.asarray(out['aexp'], dtype='f8')
            tau = np.asarray(out['tau'], dtype='f8')
            # RECFAST stores the baryon temperature ``Tm`` directly; the MB95 module does not, but
            # both branches store the composed spline ``a * T_b``, from which ``Tm`` is recovered.
            if 'Tm' in out:
                Tm = np.asarray(out['Tm'], dtype='f8')
            else:
                Tm = np.asarray(out['tempba_of_tau_spline'].evaluate(tau), dtype='f8') / aexp
            # keep only plain numpy/float values, so the engine stays a clean pytree
            self._discoeb_bg = {
                'aexp': aexp,
                'tau': tau,
                'xe': np.asarray(out['xe'], dtype='f8'),
                'opac': np.asarray(out['opac'], dtype='f8'),
                'Tm': Tm,
                'grhom': float(out['grhom']),
                'grhog': float(out['grhog']),
                'Omegab': float(out['Omegab'])}
        return self._discoeb_bg

    def get_discoeb_perturbations(self):
        r"""
        Run (and cache) DISCO-EB's background and perturbation evolution.

        Returns
        -------
        kmodes : array
            Wavenumbers, in :math:`\mathrm{Mpc}^{-1}`, shape ``(nk,)``.

        y : array
            Perturbations, of shape ``(nk, nz, 20)``; see :data:`DISCOEB_VARIABLES` for the
            meaning of the last axis.

        z : array
            Redshifts corresponding to the second axis of ``y``, ascending.
        """
        if self._discoeb_pert is None:
            jax, jnp, evolve_background, evolve_perturbations, evolve_perturbations_batched, _ = \
                _import_discoeb()
            extra = self._extra_params
            kmin, kmax = self._get_kgrid()
            # ``self._z_pk`` is ascending, so ``aexp_out`` is descending.  DISCO-EB's perturbation
            # solver requires its output scale factors -- hence its conformal-time ``saveat.ts`` --
            # to be *increasing*, so solve in ascending ``aexp`` (descending z) and reorder back.
            aexp_out = 1. / (1. + self._z_pk)
            order = np.argsort(aexp_out)
            inverse_order = np.argsort(order)
            solve = _discoeb_solve_jit(
                evolve_background, evolve_perturbations, evolve_perturbations_batched,
                nk=extra['nk'], Nmnu=int(self._discoeb_params['Nmnu']),
                lmaxg=extra['lmaxg'], lmaxgp=extra['lmaxgp'], lmaxr=extra['lmaxr'],
                lmaxnu=extra['lmaxnu'], nqmax=extra['nqmax'], rtol=extra['rtol'],
                atol=extra['atol'], max_steps=extra['max_steps'], pcoeff=extra['pcoeff'],
                icoeff=extra['icoeff'], dcoeff=extra['dcoeff'], factormax=extra['factormax'],
                factormin=extra['factormin'], thermo_module=extra['thermo_module'],
                num_thermo=extra['num_thermo'], bg_rtol=extra['bg_rtol'], bg_atol=extra['bg_atol'],
                batched=extra['batched'], batch_size=extra['batch_size'])
            param = {name: jnp.asarray(value, dtype='f8')
                     for name, value in self._discoeb_params.items()}
            kmodes, y = solve(param, jnp.asarray(aexp_out[order], dtype='f8'),
                              jnp.asarray(kmin, dtype='f8'), jnp.asarray(kmax, dtype='f8'))
            y = y[:, inverse_order, :]      # back to the engine's ascending-z ordering
            # in jax mode keep the tracers (so the result stays differentiable); else to numpy
            self._discoeb_pert = (self._np.asarray(kmodes), self._np.asarray(y), self._z_pk)
        return self._discoeb_pert


def _make_of_tuple(of):
    """Return ``of`` as a list of strings (a single string is a 1-element list)."""
    if isinstance(of, str):
        return [of]
    return list(of)


class Background(DefaultBackground):
    r"""
    Background quantities.

    DISCO-EB evolves the same Friedmann background as :class:`DefaultBackground`
    (:math:`\Lambda\mathrm{CDM}` + :math:`(w_{0}, w_{a})` dark energy + curvature + massive
    neutrinos), so the (well-tested, analytic) default implementation is used, which provides
    distances, the growth factor/rate and all the energy densities.
    """


@utils.addproperty('rs_drag', 'z_drag', 'z_star', 'rs_star', 'theta_star', 'YHe')
class Thermodynamics(BaseSection):
    r"""
    Thermodynamics, from DISCO-EB's thermal history (RECFAST by default).

    Provides the recombination redshift :attr:`z_star`, the baryon drag redshift :attr:`z_drag`
    and the comoving sound horizon at those epochs, computed by integrating DISCO-EB's baryon
    sound speed over its conformal-time grid.
    """

    def __init__(self, engine):
        super().__init__(engine)
        self._h = engine['h']
        bg = engine.get_discoeb_background()
        np_ = np

        aexp = np_.asarray(bg['aexp'], dtype='f8')
        tau = np_.asarray(bg['tau'], dtype='f8')
        xe = np_.asarray(bg['xe'], dtype='f8')
        opac = np_.asarray(bg['opac'], dtype='f8')
        grhom = float(bg['grhom'])
        grhog = float(bg['grhog'])
        Omegab = float(bg['Omegab'])

        # keep only the physical, increasing-in-a part of the grid
        order = np_.argsort(aexp)
        aexp, tau, xe, opac = (arr[order] for arr in (aexp, tau, xe, opac))
        mask = (aexp > 0.) & np_.isfinite(tau)
        aexp, tau, xe, opac = (arr[mask] for arr in (aexp, tau, xe, opac))

        z = 1. / aexp - 1.

        # baryon-photon acoustic sound speed: c_s^2 = 1 / (3 (1 + R)), R = 3 rho_b / (4 rho_gamma).
        # NB: DISCO-EB's ``cs2`` is the (tiny) *baryon* thermal sound speed used in the baryon Euler
        # equation; the acoustic sound horizon is set by the photon pressure, hence 1/sqrt(3(1+R)).
        R = 3. * grhom * Omegab * aexp / (4. * grhog)
        cs = 1. / np_.sqrt(3. * (1. + R))
        # comoving sound horizon r_s(tau) = int c_s dtau, in Mpc (tau in Mpc, c_s in units of c)
        rs_Mpc = _cumulative_trapezoid(cs, tau)
        # Thomson optical depth from tau to today: kappa(tau) = int_tau^{tau0} opac dtau
        kappa = _reverse_cumulative_trapezoid(opac, tau)

        # store (descending in z, i.e. ascending in a/tau) for interpolation
        self._z_grid = z
        self._tau_grid = tau
        self._rs_grid = rs_Mpc * self._h          # in Mpc/h
        self._kappa_grid = kappa
        self._xe_grid = xe
        self._cs_grid = cs

        # recombination: kappa = 1
        self._z_star = _interp_root(z, kappa, 1.)
        self._rs_star = float(np_.interp(-self._z_star, -z, self._rs_grid))
        ba = engine.get_background()
        self._theta_star = self._rs_star / ba.comoving_transverse_distance(self._z_star)

        # drag epoch: Eisenstein & Hu (1998) fitting formula (as in CLASS / the eisenstein_hu engine)
        omega_b = float(engine['omega_b'])
        omega_m = float(engine['omega_cdm'] + engine['omega_b'])
        b1 = 0.313 * omega_m**(-0.419) * (1. + 0.607 * omega_m**0.674)
        b2 = 0.238 * omega_m**0.223
        self._z_drag = float(1345. * omega_m**0.251 / (1. + 0.659 * omega_m**0.828)
                             * (1. + b1 * omega_b**b2))
        self._rs_drag = float(np_.interp(-self._z_drag, -z, self._rs_grid))

        self._YHe = float(engine._discoeb_params['YHe'])

        # CosmoMC-approximation sound-horizon angle (computed once; keeps the section jax-safe,
        # i.e. holding only scalars/arrays rather than a reference to the engine).
        rs_cosmomc, zstar_cosmomc = _compute_rs_cosmomc(ba.Omega0_b * ba.h**2, ba.Omega0_m * ba.h**2,
                                                        ba.hubble_function)
        self._theta_cosmomc = float(rs_cosmomc * ba.h / ba.comoving_transverse_distance(zstar_cosmomc))

        # baryon temperature grid (DISCO-EB stores Tm directly, in K)
        Tm = np_.asarray(bg['Tm'], dtype='f8')[order][mask]
        self._Tb_grid = Tm

    @property
    def theta_cosmomc(self):
        r"""CosmoMC-approximation sound horizon angle :math:`r_{s}/D_{M}(z_{*})`, unitless."""
        return self._theta_cosmomc

    @utils.flatarray()
    def x_e(self, z):
        r"""Free electron fraction :math:`x_{e}` at redshift ``z`` (from DISCO-EB's RECFAST)."""
        return np.interp(-np.asarray(z), -self._z_grid, self._xe_grid, left=None, right=None)

    @utils.flatarray()
    def rs(self, z):
        r"""Comoving sound horizon at redshift ``z``, in :math:`\mathrm{Mpc}/h`."""
        return np.interp(-np.asarray(z), -self._z_grid, self._rs_grid, left=None, right=None)

    @utils.flatarray()
    def kappa(self, z):
        r"""Thomson optical depth from ``z`` to today (unitless)."""
        return np.interp(-np.asarray(z), -self._z_grid, self._kappa_grid, left=None, right=None)

    @utils.flatarray()
    def T_b(self, z):
        r"""Baryon temperature at redshift ``z``, in K."""
        return np.interp(-np.asarray(z), -self._z_grid, self._Tb_grid, left=None, right=None)


def _cumulative_trapezoid(y, x):
    """Cumulative trapezoidal integral of ``y`` over ``x``, starting at 0 (same length as ``y``)."""
    dx = np.diff(x)
    segments = 0.5 * (y[1:] + y[:-1]) * dx
    return np.concatenate([[0.], np.cumsum(segments)])


def _reverse_cumulative_trapezoid(y, x):
    r"""Return :math:`\int_{x}^{x_\mathrm{max}} y\, dx'` on the grid (same length as ``y``)."""
    forward = _cumulative_trapezoid(y, x)
    return forward[-1] - forward


def _interp_root(x, fvalue, target):
    """Return ``x`` where the monotonic ``fvalue(x)`` crosses ``target`` (linear interpolation)."""
    fvalue = np.asarray(fvalue, dtype='f8')
    x = np.asarray(x, dtype='f8')
    g = fvalue - target
    # find a bracketing interval where the sign changes
    sign_change = np.where(np.sign(g[:-1]) * np.sign(g[1:]) <= 0)[0]
    if len(sign_change) == 0:
        raise CosmologyComputationError('Could not locate crossing of {} in thermodynamics.'.format(
            target))
    i = sign_change[0]
    x0, x1, g0, g1 = x[i], x[i + 1], g[i], g[i + 1]
    if g1 == g0:
        return float(x0)
    return float(x0 + (x1 - x0) * (-g0) / (g1 - g0))


@utils.addproperty('k_pivot', 'n_s', 'alpha_s', 'beta_s')
class Primordial(BaseSection):
    r"""Primordial power spectrum (power-law, as used by DISCO-EB)."""

    def __init__(self, engine):
        super().__init__(engine)
        self._h = engine['h']
        self._A_s = engine._A_s
        self._n_s = engine['n_s']
        # DISCO-EB does not implement running; expose zeros for interface completeness.
        self._alpha_s = 0.
        self._beta_s = 0.
        self._k_pivot = engine['k_pivot'] / self._h      # to h/Mpc
        self._rsigma8 = engine._rescale_sigma8()

    @property
    def A_s(self):
        r"""Scalar amplitude of the primordial power spectrum at :math:`k_\mathrm{pivot}`."""
        return self._A_s * self._rsigma8**2

    @property
    def ln_1e10_A_s(self):
        r""":math:`\ln(10^{10}A_s)`, unitless."""
        return self._np.log(1e10 * self.A_s)

    def pk_k(self, k, mode='scalar'):
        r"""
        The primordial spectrum of curvature perturbations at ``k``, in :math:`(\mathrm{Mpc}/h)^{3}`.

        Parameters
        ----------
        k : array_like
            Wavenumbers, in :math:`h/\mathrm{Mpc}`.

        mode : string, default='scalar'
            Only 'scalar' is implemented.

        Returns
        -------
        pk : array
            The primordial power spectrum.
        """
        if mode != 'scalar':
            raise CosmologyInputError('discoeb only implements the scalar primordial spectrum; '
                                      'got mode={}.'.format(mode))
        k = self._np.asarray(k)
        return self._h**3 * self.A_s * (k / self._k_pivot)**(self._n_s - 1.)

    def pk_interpolator(self, mode='scalar'):
        """Return :class:`PowerSpectrumInterpolator1D` for the primordial power spectrum."""
        return PowerSpectrumInterpolator1D.from_callable(
            pk_callable=lambda k: self.pk_k(k, mode=mode))


class Transfer(BaseSection):
    r"""
    Matter transfer function, from DISCO-EB's linear perturbations.

    On super-horizon scales the matter density contrast of adiabatic perturbations behaves as
    :math:`\delta_{m} \propto k^{2}`, so the transfer function -- normalised to unity on large
    scales, as for the other :mod:`cosmoprimo` engines -- is obtained as

    .. math::

        T(k) = \frac{\delta_{m}(k) / k^{2}}{\lim_{k \to 0} \delta_{m}(k) / k^{2}}

    This is independent of the growth normalisation (which cancels between numerator and
    denominator), so ``T(k)`` is the scale-dependent *shape* of the transfer function.
    """

    def __init__(self, engine):
        super().__init__(engine)
        self._h = engine['h']
        kmodes, y, self._z = engine.get_discoeb_perturbations()
        self._k_deb = np.asarray(kmodes, dtype='f8')                      # 1/Mpc, ascending
        self._delta_m = np.asarray(y[..., DISCOEB_VARIABLES['delta_m']], dtype='f8')  # (nk, nz)
        # super-horizon normalisation delta_m/k^2 at the smallest k, for each redshift
        self._norm = self._delta_m[0, :] / self._k_deb[0]**2              # (nz,)

    def transfer_k(self, k, z=0.):
        r"""
        Return the matter transfer function, normalised to unity on large scales.

        Parameters
        ----------
        k : array_like
            Wavenumbers, in :math:`h/\mathrm{Mpc}`.

        z : float, default=0
            Redshift at which to evaluate the transfer function (nearest computed redshift).

        Returns
        -------
        transfer : array
            Matter transfer function.
        """
        k = self._np.asarray(k, dtype='f8')
        iz = int(np.argmin(np.abs(self._z - z)))
        k_deb = k * self._h
        delta_m = np.interp(k_deb, self._k_deb, self._delta_m[:, iz])
        with np.errstate(divide='ignore', invalid='ignore'):
            ksq = np.where(k_deb > 0., k_deb**2, np.inf)
            transfer = (delta_m / ksq) / self._norm[iz]
        return np.where(k_deb > 0., transfer, 1.)


class Fourier(BaseSection):
    r"""
    Fourier-space quantities, from DISCO-EB's linear perturbations.

    Provides the linear matter power spectrum (and cross-spectra between the density and velocity
    divergence of the various species), from which :math:`\sigma_{8}`, :math:`\sigma(r, z)` and the
    growth rate are derived.
    """

    def __init__(self, engine):
        super().__init__(engine)
        self._h = engine['h']
        self._rsigma8 = engine._rescale_sigma8()
        # store the DISCO-EB outputs and primordial parameters, so the section holds only
        # arrays/scalars (no reference back to the engine).  Kept as ``self._np`` arrays so the
        # power spectrum remains differentiable in jax mode.
        kmodes, y, z = engine.get_discoeb_perturbations()
        self._k_deb = self._np.asarray(kmodes)             # 1/Mpc
        self._y = self._np.asarray(y)                      # (nk, nz, nvar)
        self._z = z                                        # ascending (concrete)
        dp = engine._discoeb_params
        self._A_s = dp['A_s']
        self._n_s = dp['n_s']
        self._k_p = dp['k_p']
        # primordial factor 2 pi^2 A_s (k/k_p)^(n_s-1) k^-3, in Mpc^3 (k in 1/Mpc)
        self._prim_deb = 2. * np.pi**2 * self._A_s * (self._k_deb / self._k_p)**(self._n_s - 1.)\
            * self._k_deb**(-3.)
        self._pk_interpolators = {}

    def _pk_deb(self, idx1, idx2):
        r"""Power spectrum of ``y[idx1] x y[idx2]`` on the (k, z) grid, in :math:`\mathrm{Mpc}^3`."""
        return self._prim_deb[:, None] * self._y[..., idx1] * self._y[..., idx2]

    def _index_of(self, of_):
        if of_ not in _OF_TO_INDEX:
            raise CosmologyInputError(
                'discoeb does not provide "{}"; available quantities are {}'.format(
                    of_, sorted(_OF_TO_INDEX)))
        return _OF_TO_INDEX[of_]

    def table(self, non_linear=False, of='delta_m'):
        r"""
        Return the linear power spectrum table, in :math:`(\mathrm{Mpc}/h)^{3}`.

        Parameters
        ----------
        non_linear : bool, default=False
            DISCO-EB is linear-only; ``True`` raises an error.

        of : string, tuple, default='delta_m'
            Perturbed quantity (or pair, for the cross-spectrum).

        Returns
        -------
        k : array
            Wavenumbers, in :math:`h/\mathrm{Mpc}`.

        z : array
            Redshifts.

        pk : array
            Power spectrum, of shape ``(len(k), len(z))``.
        """
        if non_linear:
            raise CosmologyInputError(
                'discoeb is a linear-only solver; the non-linear power spectrum is not available.')
        of = _make_of_tuple(of)
        indices = [self._index_of(of_) for of_ in of]
        indices = (indices + [indices[0]])[:2]
        pk_deb = self._pk_deb(*indices)
        # to cosmoprimo units: k[h/Mpc] = k[1/Mpc] / h ; P[(Mpc/h)^3] = P[Mpc^3] * h^3
        k = self._k_deb / self._h
        pk = pk_deb * self._h**3 * self._rsigma8**2
        return k, self._z, pk

    def pk_interpolator(self, non_linear=False, of='delta_m', **kwargs):
        r"""
        Return a :class:`PowerSpectrumInterpolator2D` instance.

        Parameters
        ----------
        non_linear : bool, default=False
            DISCO-EB is linear-only; ``True`` raises an error.

        of : string, tuple, default='delta_m'
            Perturbed quantity (or pair, for the cross-spectrum), e.g. ``delta_m``, ``delta_cb``,
            ``('delta_cb', 'theta_cb')``.

        kwargs : dict
            Arguments for :class:`PowerSpectrumInterpolator2D`.

        Returns
        -------
        PowerSpectrumInterpolator2D
        """
        key = (tuple(_make_of_tuple(of)), tuple(sorted((k, str(v)) for k, v in kwargs.items())))
        if key not in self._pk_interpolators:
            ka, za, pka = self.table(non_linear=non_linear, of=of)
            # self._np (not np) so the table stays a jax array in jax mode: cosmoprimo's
            # PowerSpectrumInterpolator2D is itself jax-friendly (via interpax), so sigma_rz,
            # sigma8, ... remain differentiable w.r.t. the cosmological parameters.
            #
            # With a single output redshift the 2D interpolator has nothing to interpolate along
            # ``z`` and requires a ``growth_factor_sq``; there is no growth information in a
            # one-point grid, so pass a constant unit growth.  Evaluation at the single computed
            # redshift (e.g. ``sigma8_m`` for z=[0.]) is then exact.
            if self._np.asarray(za).size == 1 and 'growth_factor_sq' not in kwargs:
                from .jax import Partial
                kwargs['growth_factor_sq'] = Partial(self._np.ones_like)
            self._pk_interpolators[key] = PowerSpectrumInterpolator2D(
                ka, za, self._np.abs(pka), **kwargs)
        return self._pk_interpolators[key]

    def pk_kz(self, k, z, non_linear=False, of='delta_m'):
        r"""
        Return the linear power spectrum at ``k`` (in :math:`h/\mathrm{Mpc}`) and ``z``,
        in :math:`(\mathrm{Mpc}/h)^{3}`.
        """
        return self.pk_interpolator(non_linear=non_linear, of=of)(k, z=z)

    def sigma_rz(self, r, z, of='delta_m', **kwargs):
        r"""Return the r.m.s. of ``of`` perturbations in spheres of ``r`` (in :math:`\mathrm{Mpc}/h`)."""
        return self.pk_interpolator(non_linear=False, of=of, **kwargs).sigma_rz(r, z)

    def sigma8_z(self, z, of='delta_m'):
        r"""Return the r.m.s. of ``of`` perturbations in spheres of :math:`8\,\mathrm{Mpc}/h`."""
        return self.sigma_rz(8., z, of=of)

    @property
    def sigma8_m(self):
        r"""Current r.m.s. of matter perturbations in spheres of :math:`8\,\mathrm{Mpc}/h`."""
        return self.sigma8_z(0., of='delta_m')

    @property
    def sigma8_cb(self):
        r"""Current r.m.s. of cold dark matter + baryon perturbations in spheres of
        :math:`8\,\mathrm{Mpc}/h`."""
        return self.sigma8_z(0., of='delta_cb')
