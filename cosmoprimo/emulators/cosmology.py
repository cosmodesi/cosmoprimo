r"""Cosmologies, as :class:`~cosmoprimo.emulators.tools.Emulator` subclasses.

This is everything :mod:`cosmoprimo.emulators.tools` deliberately does not know. It is CMB and
large-scale-structure physics:

- :math:`C_\ell \propto A_s`, :math:`P(k) \propto A_s` -- exact for the primary anisotropies and
  for the linear power spectrum, so the amplitude can leave the interpolation grid entirely and
  cost no nodes at all. not exact once lensing or halofit is applied: both are non-linear in the
  amplitude, so there dividing by :math:`A_s` only flattens the dependence and the parameter
  stays on the grid.
- :math:`e^{-\tau}` per screened leg -- ``tt`` and ``ee`` carry :math:`e^{-2\tau}`, ``tp`` and
  ``ep`` one factor, ``pp`` none. Also only a flattening: below :math:`\ell \sim 30` reionization
  puts power back, which no prefactor describes, so :math:`\tau` stays on the grid too.

There is deliberately no :math:`\theta_\ast` rescaling of the :math:`\ell` axis, though it works
(applied to :math:`D_\ell`, not :math:`C_\ell` -- 76x on the residual otherwise). Whitening the
space onto the posterior's principal axes, which :mod:`~cosmoprimo.emulators.tools` does by itself,
beat that hand-built coordinate by 86x in the median and 300x in the 90th percentile.

What the analytic calculations buy
----------------------------------
``DefaultBackground`` solves the background ODEs straight from the parameters, with no Boltzmann
call. Measured against CAMB: ``efunc``, ``growth_factor`` and ``growth_rate`` agree to 6e-13 and
``comoving_radial_distance`` to 2.3e-4. So the background and fourier sections divide by it and
fit the ratio (``analytic=True``, the default), which for growth is 1 to machine precision. The
thermodynamics section does the same with the Eisenstein & Hu fitting formulae. A preconditioner
does not have to be correct physics -- what the formula gets wrong stays on the grid.

The harmonic section has no such divisor and does not pretend to: the natural candidate is an
acoustic-scale rescaling of the :math:`\ell` axis, and that was measured to be beaten 86x in the
median by simply whitening the space (see below).

Adding a section
----------------
Subclass :class:`SectionEmulator` and write three things: :meth:`~SectionEmulator.extract` (what
comes out of a computed cosmology), :meth:`~SectionEmulator.scaling` (what to divide out, if
anything), and :meth:`~SectionEmulator.section_class` (how to serve it back as a cosmoprimo
section). Then register it in ``_SECTIONS``. Nothing else in the package needs to change.
"""

import numpy as np

from cosmoprimo.cosmology import Cosmology
from cosmoprimo.jax import numpy as jnp, numpy_jax

# aliased: in this module `Emulator` is the user-facing entry point below,
# which dispatches on `section`; this is the template it all derives from
from .tools import Emulator as _Emulator, NotTrained


# the formulae divided out live in `analytic`, shared with desilike's cosmology emulators
from .analytic import (AMPLITUDES as _AMPLITUDES, nonzero as _nonzero,
                       eisenstein_hu_scales as _eisenstein_hu_scales, harmonic_scaling,
                       theta_analytic, solve_theta_analytic, dilate)

# the columns each cosmoprimo harmonic getter returns, so the scaling can be built without a run
_SPECTRA = {'lensed_cl': ('tt', 'ee', 'bb', 'te'),
            'unlensed_cl': ('tt', 'ee', 'bb', 'te'),
            'lens_potential_cl': ('pp', 'tp', 'ep')}


class _Table(dict):
    r"""A structured-array lookalike, holding whatever array type it was given.

    cosmoprimo's sections return numpy structured arrays, and a structured dtype has no tracer
    equivalent -- so an emulated section that built one could not be used inside ``jit``, which is
    most of the point of having an emulator. This supports what callers actually do with the
    table (``table['tt']``, ``table['ell']``, ``table[mask]``, ``len``, ``.dtype.names``) while
    holding jax arrays.
    """
    @property
    def dtype(self):
        """A real structured dtype, built from the columns.

        Read off each array rather than assumed, so a tracer's dtype is reported truthfully; and a
        genuine ``np.dtype`` rather than a stand-in, so ``.names``, ``.fields`` and comparisons
        against the native sections' dtype all behave.
        """
        return np.dtype([(name, getattr(value, 'dtype', np.float64))
                         for name, value in self.items()])

    @property
    def size(self):
        return len(self)

    def __len__(self):
        for value in self.values():
            return len(value)
        return 0

    def __getitem__(self, name):
        if isinstance(name, str):
            return super().__getitem__(name)
        # a slice or a boolean mask applies to every column, as it would to a structured array
        return type(self)({key: value[name] for key, value in self.items()})


#: A physical-density basis: what the CMB and the matter power spectrum actually respond to.
#: Everything not named here is passed through unchanged, so it doubles as the list of names that
#: get replaced.
PHYSICAL = ('omega_cdm', 'omega_b', 'h')

#: What ``basis='theta'`` is shorthand for: the physical densities with the acoustic scale in
#: place of :math:`h`. See :meth:`SectionEmulator.to_training`.
THETA = ('omega_cdm', 'omega_b', 'theta_MC_100')


def _basis(basis):
    """``basis`` as a list of names: a shorthand resolved, anything else taken as given."""
    if basis is None:
        return None
    return list({'physical': PHYSICAL, 'theta': THETA}.get(basis, basis)) \
        if isinstance(basis, str) else list(basis)


class SectionEmulator(_Emulator):
    r"""One section of a cosmology. Clone the fiducial, compute, extract.

    The target is :meth:`compute` -- a bound method, nothing more. The split between
    :meth:`compute` and :meth:`extract` is what lets several sections share one Boltzmann call
    when they are emulated together: :class:`CosmologyEmulator` clones once and calls each
    section's :meth:`extract` on the same cosmology.

    Parameters
    ----------
    basis : list, str, default=None
        Train in these parameters, whatever the space was written in. ``None`` trains in the
        space's own; ``'physical'`` is shorthand for :data:`PHYSICAL`, ``'theta'`` for
        :data:`THETA`.

        A chain may be run in :math:`\Omega_m`, but the spectra respond simply to the physical
        density :math:`\omega_{cdm} = \Omega_{cdm} h^2`, and that map mixes in :math:`h`: at fixed
        :math:`\Omega_m = 0.31`, :math:`\omega_{cdm}` runs from 0.107 to 0.135 over
        :math:`h \in [0.64, 0.72]`. Being non-linear, it is not something whitening can absorb --
        whitening is a rotation and a rescaling, and this is neither.

        Measured on lensed TT, 25 nodes either way, over a Planck-like posterior in
        ``(Omega_m, Omega_b, h)`` given as samples: 1.5x better in the median and 3.2x at the 90th
        percentile. Worth having, but an order of magnitude less than whitening was, which is why
        it is not the default.

        And it is the default for nobody, because it can also lose.

        ``'theta'`` is the physical basis with :math:`100\,\theta_\mathrm{MC}` in place of
        :math:`h`. The argument for it is that ``h``, ``w0`` and ``wa`` act on the spectra
        almost entirely through the acoustic scale, so a box in ``h`` holds cosmologies whose
        peaks are translated along :math:`\ell`, which is not something a low-order interpolant
        follows.
    """
    section = None

    def __init__(self, cosmo, space, basis=None, **options):
        self.cosmo = cosmo
        self.basis = _basis(basis)
        super().__init__(self.compute, space, **options)

    # ── the basis ─────────────────────────────────────────────────────────────
    def to_training(self, params):
        r"""The user's parameters, read back in :attr:`basis` -- by the cosmology itself.

        :meth:`~cosmoprimo.cosmology.Cosmology._get_params` does the work, because the conversion
        is a cosmology's business: it needs the parameter compilation, and the fiducial supplies
        everything the user did not vary.

        ``theta_MC_100`` is the exception, and is computed by :func:`~.analytic.theta_analytic`
        instead. Two reasons, and the second is the one that matters: deriving it the ordinary way
        runs a background per point, and it is not an input :meth:`~cosmoprimo.cosmology.Cosmology.clone`
        accepts, so the basis needs a genuine inverse (:meth:`from_training`) rather than another
        call to the same converter. Using the same closed form in both directions makes the round
        trip exact, which is what lets the formula's own ~1.6 sigma offset from the engine's
        ``theta_MC_100`` cancel: the box is built by mapping points through this, the nodes are
        evaluated by inverting it, and predictions enter through it again. It would only matter if
        a ``theta`` from somewhere else -- a published box, a chain -- were fed in.
        """
        if self.basis is None:
            return params
        from cosmoprimo import Cosmology

        names = self._basis_names()
        theta = 'theta_MC_100' in names
        names = ['h' if name == 'theta_MC_100' else name for name in names]
        if set(names) <= set(params):
            # Already in the basis, so the converter is a pass-through -- measured exactly zero
            # difference -- and skipping it is not an optimisation of the arithmetic but of the
            # cost of getting there: `_get_params` normalises a whole parameter compilation,
            # neutrino-mass solve included, at 12.6 ms a point. That is paid once per point of
            # the space when `training_space` maps it, so on the default 100000 draws the
            # difference is 21 minutes of construction against none.
            params = {name: params[name] for name in names}
        else:
            params = Cosmology._get_params(dict(params), names, base=self.cosmo._input_params)
        if theta:
            params['theta_MC_100'] = 100. * theta_analytic(
                params.pop('h'), params['omega_b'], params['omega_cdm'],
                **self._theta_kwargs(params))
        return params

    def from_training(self, params):
        """The inverse: the calculator is cloned with these, and ``theta_MC_100`` is not
        something :meth:`~cosmoprimo.cosmology.Cosmology.clone` accepts.

        ``h`` comes back through :func:`~.analytic.solve_theta_analytic`, a bisection on the same
        closed form -- not through :meth:`~cosmoprimo.cosmology.Cosmology.solve`, which runs a
        background per iteration and would break the exact round trip above. A bisection whose
        bracket misses the root returns nan rather than an endpoint, so a node that cannot be
        placed is reported instead of being fitted at the wrong theta.

        Every other basis needs no inverse: its names are all ``clone`` inputs.
        """
        if self.basis is None or 'theta_MC_100' not in params:
            return params
        params = dict(params)
        params['h'] = solve_theta_analytic(params.pop('theta_MC_100'), params['omega_b'],
                                           params['omega_cdm'], **self._theta_kwargs(params))
        return params

    def _theta_kwargs(self, params):
        """What :func:`~.analytic.theta_analytic` needs besides the densities: the dark energy
        and the radiation content, varied when the space varies them and the fiducial's
        otherwise."""
        # Each read the same way: the sampled value where the space varies it, the cosmology's
        # otherwise. None may be captured as a constant -- a captured one evaluates the emulator's
        # basis at the fiducial while the calculator uses the sampled value, and the two bases then
        # disagree point by point. That is what put an earlier box 5.3 sigma off its posterior.
        kwargs = {'w0': params.get('w0_fld', self.cosmo['w0_fld']),
                  'wa': params.get('wa_fld', self.cosmo['wa_fld']),
                  'm_ncdm': params.get('m_ncdm', self.cosmo['m_ncdm']),
                  'N_ur': params.get('N_ur', self.cosmo['N_ur']),
                  'T_cmb': params.get('T_cmb', self.cosmo['T_cmb'])}
        # as an array, so a compiled `theta_analytic` sees one argument structure rather than
        # retracing between a tuple of masses and an array of them
        kwargs['m_ncdm'] = np.atleast_1d(kwargs['m_ncdm'])
        return kwargs

    #: The names a density basis stands in for. Anything else the space varies is passed
    #: through untouched.
    _REPLACED = ('Omega_m', 'Omega_cdm', 'Omega_b', 'H0', 'h', 'omega_m', 'omega_cdm', 'omega_b')

    def _basis_names(self):
        """The training names: the requested basis, plus everything it does not replace.

        A basis change is a reparametrisation, so it must not change the dimension.
        """
        replaced = [name for name in self.space.params if name in self._REPLACED]
        if len(self.basis) != len(replaced):
            raise ValueError(
                f'basis {self.basis} has {len(self.basis)} parameters, but the space varies '
                f'{len(replaced)} of the ones it stands in for ({replaced}). A basis change is a '
                f'reparametrisation and cannot add a direction: either vary the missing '
                f'parameter in the Space, or give a basis of {len(replaced)} names.')
        passthrough = [name for name in self.space.params if name not in self._REPLACED]
        return list(self.basis) + [name for name in passthrough if name not in self.basis]

    def training_space(self):
        if self.basis is None:
            return self.space
        return self.space.map(self.to_training)

    def clone(self, params):
        try:
            return self.cosmo.clone(**params)
        except Exception as exc:
            raise ValueError(f'{type(self).__name__} could not clone the fiducial cosmology with '
                             f'{sorted(params)}: {exc}') from exc

    # ── what a subclass writes ────────────────────────────────────────────────
    def extract(self, cosmo):
        """Named arrays out of an already computed cosmology. No scaling here."""
        raise NotImplementedError

    def amplitude(self, params):
        """``A_s``, in whatever spelling the space uses -- ``logA``, ``ln10^10A_s`` -- or None if
        the space varies no amplitude at all.

        Derived by the cosmology rather than by hand: ``A_s = 1e-10 exp(logA)`` is a convention
        that :meth:`~cosmoprimo.cosmology.Cosmology._compile_params` already implements, along
        with every alias, and a second copy here would be one more thing to keep in step.
        """
        if not any(name in params for name in _AMPLITUDES):
            return None
        from cosmoprimo import Cosmology

        return Cosmology._get_params(dict(params), ['A_s'],
                                     base=self.cosmo._input_params)['A_s']

    def scaling(self, params):
        """{output name: factor} divided out at training, multiplied back at prediction.

        Empty by default -- a section that knows nothing exact about itself divides out nothing.
        """
        return {}

    def dilation(self, params):
        r"""``{output name: (k grid, scale)}``: outputs read back in a reference frame before the
        fit, at :math:`k s`, and returned to the point's own frame at prediction.

        Empty by default. It is the second thing a section can know about itself, and it is not a
        factor: :math:`h` moves a spectrum along its k axis, and no prefactor describes that. See
        :meth:`FourierEmulator.dilation`.
        """
        return {}

    def section_class(self, source, prefix=''):
        """A :class:`~cosmoprimo.cosmology.BaseSection` serving the predictions back.

        ``source(engine)`` returns the predicted dict for that engine's cosmology; ``prefix`` is
        what the composite prepended to the output names.
        """
        raise NotImplementedError

    # ── the analytic divisor ──────────────────────────────────────────────────
    def analytic_background(self, params):
        """:class:`~cosmoprimo.cosmology.DefaultBackground` for these parameters, without running
        the Boltzmann code -- it solves the same ODEs directly from the parameters.

        Measured against CAMB over the default grid: ``efunc``, ``growth_factor`` and
        ``growth_rate`` agree to 6e-13, ``comoving_radial_distance`` to 2.3e-4. So dividing by it
        leaves a ratio that is 1 to machine precision for the first three -- there is essentially
        nothing left for an interpolant to do.

        It costs about 2.5 ms per call (0.8 ms of which is building the engine), paid once per
        node at training and once per prediction. That is the trade: nodes for milliseconds.
        """
        from cosmoprimo.cosmology import BaseEngine, DefaultBackground

        key = tuple(sorted((name, float(value)) for name, value in params.items()))
        cached = getattr(self, '_analytic_cache', None)
        if cached is None or cached[0] != key:
            # `transform` and `inverse_transform` are called with the same params back to back
            self._analytic_cache = (key, DefaultBackground(BaseEngine(self.clone(dict(params)))))
        return self._analytic_cache[1]

    # ── the Emulator hooks ────────────────────────────────────────────────────
    def compute(self, params):
        return self.extract(self.clone(dict(params)))

    def transform(self, values, params):
        factors, dilations = self.scaling(params), self.dilation(params)
        out = {}
        for name, value in values.items():
            # the factor first, at the point's own k grid, and the dilation second: the tilt is
            # k-dependent, so the order is part of the definition and `inverse_transform` undoes
            # it in reverse. Dividing the tilt here, where `k h` is the live one, is what leaves
            # the dilated spectrum's primordial factor h-free
            if name in factors:
                value = value / factors[name]
            if name in dilations:
                k, scale = dilations[name]
                value = dilate(k, value, 1. / scale, axis=0) / scale**3
            out[name] = value
        return out

    def inverse_transform(self, values, params):
        factors, dilations = self.scaling(params), self.dilation(params)
        out = {}
        for name, value in values.items():
            if name in dilations:
                k, scale = dilations[name]
                value = scale**3 * dilate(k, value, scale, axis=0)
            if name in factors:
                value = value * factors[name]
            out[name] = value
        return out

    @property
    def sections(self):
        """``{name: emulator}`` -- itself, so a single section and a composite look the same."""
        return {self.section: self}

    def to_cosmology(self):
        """A :class:`~cosmoprimo.cosmology.Cosmology` whose section is predicted."""
        if not self.trained:
            raise NotTrained('call train() first')
        return self.cosmo.clone(engine=emulated_engine(self))

    # ── state ─────────────────────────────────────────────────────────────────
    def section_options(self):
        """The keyword arguments needed to rebuild this section. Saved and replayed.

        not called ``options``: :class:`~cosmoprimo.emulators.tools.Emulator` already keeps the
        engine options under that name, and a method would be shadowed by the attribute.
        """
        return {'basis': self.basis}

    def __getstate__(self):
        state = super().__getstate__()
        state['cosmo'] = _cosmology_state(self.cosmo)
        state['section_options'] = self.section_options()
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.cosmo = _cosmology_from_state(state['cosmo'])
        for name, value in state['section_options'].items():
            setattr(self, name, value)
        self.target = self.compute


def _cosmology_state(cosmo):
    """Input parameters and engine, rather than the cosmology's own ``__getstate__``.

    Deliberate: the fiducial is fully described by what was asked for, and rebuilding it from
    that is robust to the engine's internals changing shape between versions.
    """
    engine = getattr(cosmo, '_engine', None)
    return {'input_params': dict(cosmo._input_params),
            'engine': getattr(engine, 'name', None),
            'extra_params': dict(getattr(engine, '_extra_params', {}) or {})}


def _cosmology_from_state(state):
    from cosmoprimo import Cosmology

    cosmo = Cosmology(**state['input_params'])
    if state['engine'] is not None:
        cosmo.set_engine(state['engine'], **state['extra_params'])
    return cosmo


# ── harmonic ──────────────────────────────────────────────────────────────────

class HarmonicEmulator(SectionEmulator):
    r"""CMB :math:`C_\ell`, with the amplitude and the optical depth divided out.

    Parameters
    ----------
    cosmo : Cosmology
        The fiducial; ``lensing=True`` is required for lensed spectra.
    space : Space
        Where accuracy is required.
    of : tuple, str, default=('lensed_cl',)
        Which of ``'lensed_cl'``, ``'unlensed_cl'``, ``'lens_potential_cl'`` to emulate. Outputs
        are named ``'<of>.<spectrum>'``, e.g. ``'lensed_cl.tt'``.
    ellmax : int, default=None
        Truncate at this multipole; the fiducial's own ``ellmax_cl`` by default.
    """
    section = 'harmonic'

    def __init__(self, cosmo, space, of=('lensed_cl',), ellmax=None, **options):
        self.of = (of,) if isinstance(of, str) else tuple(of)
        unknown = [name for name in self.of if name not in _SPECTRA]
        if unknown:
            raise ValueError(f'unknown harmonic spectra {unknown}; available {sorted(_SPECTRA)}')
        self.ellmax = ellmax
        self.ell = None                 # captured at the first evaluation, with the arrays
        super().__init__(cosmo, space, **options)

    @property
    def lensed(self):
        """Lensing makes the amplitude channel approximate, so it decides whether A_s leaves the
        grid or is merely flattened on it."""
        return any(name != 'unlensed_cl' for name in self.of)

    def extract(self, cosmo):
        harmonic = cosmo.get_harmonic()
        values, ell = {}, None
        for name in self.of:
            table = getattr(harmonic, name)(ellmax=self.ellmax if self.ellmax is not None else -1)
            ell = np.asarray(table['ell'], dtype='i8')
            for spectrum in table.dtype.names:
                if spectrum != 'ell':
                    values[f'{name}.{spectrum}'] = np.asarray(table[spectrum], dtype='f8')
        if self.ell is None:
            self.ell = ell
        elif len(ell) != len(self.ell):
            raise ValueError(f'the engine returned {len(ell)} multipoles, {len(self.ell)} before; '
                             f'the l range must be the same at every node')
        return values

    def select_params(self, names):
        if self.lensed:
            return list(names)
        return [name for name in names if name not in _AMPLITUDES]

    def scaling(self, params):
        r"""Amplitude, and one :math:`e^{-\tau}` per screened leg.

        Keyed by output name because the optical depth screens a different number of legs in each
        spectrum -- getting that per-leg count wrong is a silent factor of :math:`e^{\tau}`.
        """
        return harmonic_scaling([f'{name}.{spectrum}' for name in self.of for spectrum in _SPECTRA[name]],
                                self.amplitude(params), params.get('tau_reio', None))

    def section_options(self):
        return {**super().section_options(), 'of': self.of, 'ellmax': self.ellmax,
                'ell': self.ell}

    def to_cosmology(self):
        if self.ell is None:
            raise RuntimeError('never evaluated, so the l range is unknown; train first')
        return super().to_cosmology()

    def section_class(self, source, prefix=''):
        from cosmoprimo.cosmology import BaseSection

        emulator = self

        class Harmonic(BaseSection):

            def __init__(self, engine):
                super().__init__(engine)
                self._engine = engine
                self._cl = source(engine)
                self.ell = emulator.ell
                self.ellmax_cl = int(self.ell[-1])

            def _table(self, of, ellmax):
                if of not in emulator.of:
                    raise ValueError(f'{of} was not emulated; this one has {list(emulator.of)}')
                if ellmax is None or ellmax < 0:
                    ellmax = self.ellmax_cl + 1 + (ellmax if ellmax is not None else -1)
                if ellmax > self.ellmax_cl:
                    raise ValueError(f'emulated up to l = {self.ellmax_cl}, asked for {ellmax}')
                names = [name for name in _SPECTRA[of] if f'{prefix}{of}.{name}' in self._cl]
                # a lookalike rather than a structured array, so the whole route -- clone, get
                # the section, read a spectrum -- stays inside a jax trace
                return _Table({'ell': np.asarray(self.ell[:ellmax + 1]),
                               **{name: self._cl[f'{prefix}{of}.{name}'][:ellmax + 1]
                                  for name in names}})

            def lensed_cl(self, ellmax=-1):
                r"""Emulated lensed :math:`C_\ell`, unitless."""
                return self._table('lensed_cl', ellmax)

            def unlensed_cl(self, ellmax=-1):
                r"""Emulated unlensed :math:`C_\ell`, unitless."""
                return self._table('unlensed_cl', ellmax)

            def lens_potential_cl(self, ellmax=-1):
                r"""Emulated lensing-potential :math:`C_\ell`, unitless."""
                return self._table('lens_potential_cl', ellmax)

        return Harmonic


# ── background ────────────────────────────────────────────────────────────────

# every one of these is a smooth function of z, so a modest grid plus a spline is enough; they are
# emulated as arrays over that grid rather than one emulator per redshift
_BACKGROUND = ('efunc', 'comoving_radial_distance', 'angular_diameter_distance',
               'luminosity_distance', 'growth_factor', 'growth_rate')


class BackgroundEmulator(SectionEmulator):
    """Distances and growth over a redshift grid.

    Parameters
    ----------
    z : array, default=None
        The grid. Log-spaced in ``1 / (1 + z)`` out to z = 1000 by default, which resolves both
        the low-z distances and the recombination-era ones on the same axis.
    of : tuple, default=None
        Which quantities; all of :data:`_BACKGROUND` by default.
    analytic : bool, default=True
        Fit the ratio to :meth:`~SectionEmulator.analytic_background` rather than the quantity.

        On by default because the ratio is 1 to 6e-13 for ``efunc``, ``growth_factor`` and
        ``growth_rate``, and to 2.3e-4 for the distances -- the analytic background solves the
        same ODEs, so the Boltzmann code adds almost nothing here.
        If the background is all you want, in (open) w0wamnuCDM cosmology, do not emulate it at all, just use
        ``DefaultBackground``. This section earns its place when it rides along with a harmonic
        or fourier one, sharing their Boltzmann call for free.
    """
    section = 'background'

    def __init__(self, cosmo, space, z=None, of=None, analytic=True, **options):
        self.analytic = bool(analytic)
        self.z = np.asarray(z, dtype='f8') if z is not None \
            else 1. / np.logspace(-3., 0., 256)[::-1] - 1.
        self.of = tuple(of) if of is not None else _BACKGROUND
        unknown = [name for name in self.of if name not in _BACKGROUND]
        if unknown:
            raise ValueError(f'unknown background quantities {unknown}; '
                             f'available {list(_BACKGROUND)}')
        super().__init__(cosmo, space, **options)

    def extract(self, cosmo):
        background = cosmo.get_background()
        return {name: np.asarray(getattr(background, name)(self.z), dtype='f8')
                for name in self.of}

    def scaling(self, params):
        if not self.analytic:
            return {}
        background = self.analytic_background(params)
        return {name: _nonzero(np.asarray(getattr(background, name)(self.z), dtype='f8'))
                for name in self.of}

    def section_options(self):
        return {**super().section_options(), 'z': self.z, 'of': self.of,
                'analytic': self.analytic}

    def section_class(self, source, prefix=''):
        from cosmoprimo.cosmology import BaseSection
        from cosmoprimo.jax import Interpolator1D

        emulator = self

        class Background(BaseSection):

            def __init__(self, engine):
                super().__init__(engine)
                self._engine = engine
                values = source(engine)
                self._interp = {name: Interpolator1D(emulator.z, values[f'{prefix}{name}'])
                                for name in emulator.of}

        def _make(name):

            def getter(self, z):
                return self._interp[name](np.asarray(z))

            getter.__name__ = name
            getter.__doc__ = f'Emulated :meth:`{name}`, interpolated over the training grid.'
            return getter

        for name in emulator.of:
            setattr(Background, name, _make(name))
        return Background


# ── fourier ───────────────────────────────────────────────────────────────────

class FourierEmulator(SectionEmulator):
    r"""The matter power spectrum on a :math:`(k, z)` grid.

    Parameters
    ----------
    k : array, default=None
        Wavenumbers, :math:`h/\mathrm{Mpc}`. Log-spaced over 1e-4 to 10 by default.
    z : array, default=None
        Redshifts. ``linspace(0, sqrt(10))**2`` by default, which is dense where growth moves.
    of : tuple, default=('delta_m',)
        Which transfer combinations.
    non_linear : bool, default=False
        Emulate the non-linear spectrum. It is not linear in the amplitude, so the amplitude
        then stays on the grid instead of leaving it.
    analytic : bool, default=True
        Divide out the analytic growth :math:`D(z)^2` as well as the amplitude, so the
        interpolant sees a single k-shape instead of one per redshift.

        On by default: the analytic ``growth_factor`` matches CAMB to 6e-13, so this removes
        essentially all of the z dependence for a linear spectrum at the cost of one ODE solve
        (about 0.8 ms) per prediction. It is only a flattening for ``non_linear``, where the
        growth of the halofit correction is not the linear one.
    dilate : bool, default=False
        Read the spectra back in a reference frame at :math:`k s`, :math:`s = h /
        h_\mathrm{fid}`, so that :math:`h` is carried by the dilation rather than by the
        interpolant, and leaves the grid where the space is written in physical densities.

        Off by default, because whether it wins depends on how wide the box in :math:`h` is and
        how dense ``k`` is, and the defaults here are neither. It replaces an interpolation error
        by a resampling one: the dilation reads the spectrum off its own k grid at shifted
        wavenumbers, and that cubic resampling costs :math:`\mathrm{d}\ln k^4` -- 1.7e-3 on the
        default 200-point grid, and about 1e-4 at 480 points over 2.7 decades. Measured over
        ``h`` in [0.62, 0.72] at budget 2, worst ``|emulated / exact - 1|``: 1.7e-3 on 5 nodes
        with it, 1.5e-4 on 13 without. It is the wide-box option -- the interpolation error it
        removes grows with the box while the resampling error does not, and on a box three times
        wider the same switch was worth 19x in desilike's FOLPS emulator -- so turn it on with a
        dense ``k`` and a wide ``h``, and leave it off otherwise.
    tilt : bool, default=True
        Divide out :math:`(k h / k_\mathrm{pivot})^{n_s - n_s^\mathrm{fid}}`, so that
        :math:`n_s` leaves the grid entirely.

        Exact, not a flattening: the tilt enters a linear spectrum through the primordial one
        alone, and the transfer function knows nothing of it. So this is a whole dimension off
        the interpolation grid rather than a smaller thing to interpolate -- the same trade the
        amplitude already gets, and for the same reason. Off for ``non_linear``, where halofit
        mixes scales and the factorisation fails.
    """
    section = 'fourier'
    #: 2: the analytic growth divisor changed convention (see :meth:`scaling`), so a file
    #: written before it would predict confidently and wrongly rather than fail.
    version = 2

    def __init__(self, cosmo, space, k=None, z=None, of=('delta_m',), non_linear=False,
                 analytic=True, tilt=True, dilate=False, **options):
        self.analytic = bool(analytic)
        self.tilt = bool(tilt) and not non_linear
        self.dilate = bool(dilate) and not non_linear
        self.k = np.asarray(k, dtype='f8') if k is not None else np.logspace(-4., 1., 200)
        self.z = np.asarray(z, dtype='f8') if z is not None else np.linspace(0., 10.**0.5, 30)**2
        self.of = (of,) if isinstance(of, str) else tuple(of)
        self.non_linear = bool(non_linear)
        super().__init__(cosmo, space, **options)

    def extract(self, cosmo):
        fourier = cosmo.get_fourier()
        values = {}
        for name in self.of:
            # passed only when asked for: `non_linear=False` is every engine's default, and the
            # ones that cannot do halofit at all (eisenstein_hu) reject the keyword outright
            # rather than ignore it, so naming it made them unemulatable
            interpolator = fourier.pk_interpolator(of=name,
                                                   **({'non_linear': True} if self.non_linear else {}))
            values[f'pk.{name}'] = np.asarray(interpolator(self.k, self.z), dtype='f8')
        return values

    def select_params(self, names):
        # P(k) is exactly linear in A_s and an exact power law in n_s -- but halofit is neither,
        # so they only leave the grid for the linear spectrum
        if self.non_linear:
            return list(names)
        exact = _AMPLITUDES + (('n_s',) if self.tilt else ())
        # `h` only where the rest of the space is h-free: the dilation holds the physical
        # densities fixed, and a space written in `Omega_m` does not -- taking `h` off the grid
        # there would interpolate in `Omega_m` at an implied `omega_cdm` that moves with the `h`
        # the dilation is meanwhile handling
        if self.dilate and not any(name in names for name in ('Omega_m', 'Omega_cdm', 'Omega_b',
                                                              'omega_m')):
            exact = exact + ('h', 'H0')
        return [name for name in names if name not in exact]

    def scaling(self, params):
        amplitude = self.amplitude(params)
        factor = 1. if amplitude is None else amplitude
        if self.tilt:
            # `k` is in h/Mpc and `k_pivot` in 1/Mpc, so the tilt is a power of `k h`: measured,
            # cosmoprimo's primordial spectrum is h^3 A_s (k h / k_pivot)^(n_s - 1) read on a
            # grid in h/Mpc. Anchored at the fiducial's n_s so the factor is 1 there.
            #
            # `h` is read back through `from_training` because a basis may have replaced it --
            # in the theta basis it is not among the training parameters at all.
            user = self.from_training(dict(params))
            n_s, h = (user.get(name, self.cosmo[name]) for name in ('n_s', 'h'))
            tilt = (self.k * h / self.cosmo['k_pivot']) ** (n_s - self.cosmo['n_s'])
            # pk arrays are (k, z); the tilt varies along the first axis
            factor = factor * np.asarray(tilt, dtype='f8')[:, None]
        if self.analytic:
            # znorm=0, the convention the engines' own spectra carry (D ~ a in matter
            # domination), not the default normalisation to 1 at z = 0. The default divides out
            # only the relative growth: being 1 at z = 0 for every cosmology by construction, it
            # leaves the absolute normalisation, and that is a real cosmology dependence -- it
            # moves with h at fixed physical densities, since Omega_m = omega_m / h^2 does, and
            # desilike measured the same leftover at 12% across a (w0, wa) box against ~1e-3
            # once the absolute growth is taken out.
            #
            # With h on the grid the interpolant absorbs it either way (measured: 1.65e-3
            # against 1.61e-3), which is why this went unnoticed. It is fatal once `dilate`
            # takes h off the grid: 6.5% over h in [0.63, 0.71], against 2.2e-3 with znorm=0.
            growth = np.asarray(self.analytic_background(params).growth_factor(self.z, znorm=0.),
                                dtype='f8')
            # ... and the growth along the last
            factor = factor * _nonzero(growth**2)[None, :]
        elif amplitude is None and not self.tilt:
            return {}
        return {f'pk.{name}': factor for name in self.of}

    def dilation(self, params):
        r"""``{output: (k, s)}`` with :math:`s = h / h_\mathrm{fid}`.

        At fixed physical densities the transfer function in :math:`\mathrm{Mpc}^{-1}` does not
        move with :math:`h`, so a spectrum in :math:`(\mathrm{Mpc}/h)^3` on a grid in
        :math:`h/\mathrm{Mpc}` is :math:`P_h(k) = s^3 P_\mathrm{fid}(k s)` -- exactly, but for
        the late-time growth, which :attr:`analytic` divides out separately. What is left for the
        interpolant is a reference-frame spectrum plus a smooth residual, instead of the BAO
        wiggles sliding through the k grid, which no low-order polynomial follows.
        """
        if not self.dilate:
            return {}
        user = self.from_training(dict(params))
        scale = user.get('h', self.cosmo['h']) / self.cosmo['h']
        return {f'pk.{name}': (self.k, scale) for name in self.of}

    def section_options(self):
        return {**super().section_options(), 'k': self.k, 'z': self.z, 'of': self.of,
                'non_linear': self.non_linear, 'analytic': self.analytic, 'tilt': self.tilt,
                'dilate': self.dilate}

    def section_class(self, source, prefix=''):
        from cosmoprimo.cosmology import BaseSection
        from cosmoprimo.interpolator import PowerSpectrumInterpolator2D

        emulator = self

        class Fourier(BaseSection):

            def __init__(self, engine):
                super().__init__(engine)
                self._engine = engine
                self._pk = source(engine)

            def pk_interpolator(self, of='delta_m', non_linear=False, **kwargs):
                r"""Emulated :math:`P(k, z)`, as the usual 2D interpolator."""
                if isinstance(of, (list, tuple)):
                    of = of[0] if len(set(of)) == 1 else '_'.join(of)
                if of not in emulator.of:
                    raise ValueError(f'{of!r} was not emulated; this one has {list(emulator.of)}')
                if bool(non_linear) != emulator.non_linear:
                    raise ValueError(f'emulated with non_linear={emulator.non_linear}, '
                                     f'asked for {bool(non_linear)}')
                return PowerSpectrumInterpolator2D(emulator.k, emulator.z,
                                                   self._pk[f'{prefix}pk.{of}'], **kwargs)

            def pk_kz(self, k, z, of='delta_m', **kwargs):
                return self.pk_interpolator(of=of, **kwargs)(k, z)

            def sigma_rz(self, r, z, of='delta_m', **kwargs):
                return self.pk_interpolator(of=of, **kwargs).sigma_rz(r, z)

            def sigma8_z(self, z, of='delta_m'):
                return self.sigma_rz(8., z, of=of)

            @property
            def sigma8_m(self):
                return self.sigma8_z(0., of='delta_m')

        return Fourier


# ── thermodynamics ────────────────────────────────────────────────────────────

_THERMODYNAMICS = ('rs_drag', 'z_drag', 'rs_star', 'z_star', 'theta_star', 'theta_cosmomc')


class ThermodynamicsEmulator(SectionEmulator):
    """Recombination and drag-epoch scalars.

    Parameters
    ----------
    of : tuple, default=None
        Which quantities; all of :data:`_THERMODYNAMICS` by default.
    analytic : bool, default=True
        Divide the sound horizon and drag redshift by their Eisenstein & Hu fitting formulae, and
        the angular scales by the analytic ``rs_drag / comoving_radial_distance(z_drag)``.

        These are scalars, so the saving is not in array size -- it is that a ratio to a formula
        carrying the right parameter scalings is far flatter across the space than the quantity
        itself, and a flatter function needs fewer nodes for the same error. What the formula
        gets wrong (massive neutrinos, curvature, dark energy: exactly the cases its own engine
        refuses) simply stays on the grid.
    """
    section = 'thermodynamics'

    def __init__(self, cosmo, space, of=None, analytic=True, **options):
        self.analytic = bool(analytic)
        self.of = tuple(of) if of is not None else _THERMODYNAMICS
        unknown = [name for name in self.of if name not in _THERMODYNAMICS]
        if unknown:
            raise ValueError(f'unknown thermodynamics quantities {unknown}; '
                             f'available {list(_THERMODYNAMICS)}')
        super().__init__(cosmo, space, **options)

    def extract(self, cosmo):
        thermodynamics = cosmo.get_thermodynamics()
        return {name: np.asarray(getattr(thermodynamics, name), dtype='f8')
                for name in self.of}

    def scaling(self, params):
        if not self.analytic:
            return {}
        cosmo = self.clone(dict(params))
        scales = _eisenstein_hu_scales(cosmo)
        # an angle is a sound horizon over a distance to the same epoch; the analytic background
        # supplies the distance, so the whole ratio is available without a Boltzmann call
        distance = float(self.analytic_background(params).comoving_radial_distance(
            scales['z_drag']))
        angle = scales['rs_drag'] / distance if distance else 1.
        formula = {'rs_drag': scales['rs_drag'], 'z_drag': scales['z_drag'],
                   'rs_star': scales['rs_drag'], 'z_star': scales['z_drag'],
                   'theta_star': angle, 'theta_cosmomc': angle}
        return {name: _nonzero(formula[name]) for name in self.of if name in formula}

    def section_options(self):
        return {**super().section_options(), 'of': self.of, 'analytic': self.analytic}

    def section_class(self, source, prefix=''):
        from cosmoprimo.cosmology import BaseSection

        emulator = self

        class Thermodynamics(BaseSection):

            def __init__(self, engine):
                super().__init__(engine)
                self._engine = engine
                self._values = source(engine)

        def _make(name):

            def getter(self):
                return float(self._values[f'{prefix}{name}'])

            getter.__doc__ = f'Emulated :attr:`{name}`.'
            return property(getter)

        for name in emulator.of:
            setattr(Thermodynamics, name, _make(name))
        return Thermodynamics
_SECTIONS = {'harmonic': HarmonicEmulator, 'background': BackgroundEmulator,
             'fourier': FourierEmulator, 'thermodynamics': ThermodynamicsEmulator}


# ── several sections at once ──────────────────────────────────────────────────

class CosmologyEmulator(_Emulator):
    """Several sections, sharing one Boltzmann call per node.

    That sharing is the whole point. Training harmonic and fourier as two separate emulators
    runs the Boltzmann code twice per node for the same cosmology, and the Boltzmann call is the
    entire cost -- everything else is a spline fit. So the composite clones once and hands the
    same computed cosmology to every section's ``extract``.

    Outputs are prefixed by section (``'harmonic.lensed_cl.tt'``), and each section's own
    ``scaling`` is applied to its own outputs. A parameter leaves the grid only if every section
    handles it exactly -- one section that needs it expanded settles it for all of them, since
    they share the node set.
    """
    section = None
    #: 2: a composite may hold a fourier section, whose divisor changed convention.
    version = 2

    def __init__(self, cosmo, space, sections, basis=None, **options):
        self.cosmo = cosmo
        self.basis = _basis(basis)
        # the sections share the node set, so they share the basis too
        self.sections = {name: _SECTIONS[name](cosmo, space, basis=basis, **dict(kwargs))
                         for name, kwargs in sections.items()}
        super().__init__(self.compute, space, **options)

    to_training = SectionEmulator.to_training
    from_training = SectionEmulator.from_training
    _theta_kwargs = SectionEmulator._theta_kwargs
    _basis_names = SectionEmulator._basis_names
    training_space = SectionEmulator.training_space

    def compute(self, params):
        cosmo = self.cosmo.clone(**dict(params))         # one Boltzmann call for every section
        values = {}
        for name, section in self.sections.items():
            values.update({f'{name}.{key}': value
                           for key, value in section.extract(cosmo).items()})
        return values

    def select_params(self, names):
        keep = set()
        for section in self.sections.values():
            keep |= set(section.select_params(names))
        return [name for name in names if name in keep]

    def _factors(self, params):
        factors = {}
        for name, section in self.sections.items():
            factors.update({f'{name}.{key}': value
                            for key, value in section.scaling(params).items()})
        return factors

    def transform(self, values, params):
        factors = self._factors(params)
        return {name: value / factors[name] if name in factors else value
                for name, value in values.items()}

    def inverse_transform(self, values, params):
        factors = self._factors(params)
        return {name: value * factors[name] if name in factors else value
                for name, value in values.items()}

    def to_cosmology(self):
        if not self.trained:
            raise NotTrained('call train() first')
        return self.cosmo.clone(engine=emulated_engine(self))

    # ── state ─────────────────────────────────────────────────────────────────
    def __getstate__(self):
        state = super().__getstate__()
        state['cosmo'] = _cosmology_state(self.cosmo)
        state['sections'] = {name: section.section_options()
                             for name, section in self.sections.items()}
        state['basis'] = self.basis
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.cosmo = _cosmology_from_state(state['cosmo'])
        self.basis = state['basis']
        self.sections = {}
        for name, options in state['sections'].items():
            section = _SECTIONS[name].__new__(_SECTIONS[name])
            section.cosmo, section.space = self.cosmo, self.space
            for key, value in options.items():
                setattr(section, key, value)
            self.sections[name] = section
        self.target = self.compute


# ── putting it back where the original was ────────────────────────────────────

def emulated_engine(emulator):
    """An engine class serving ``emulator``'s predictions, usable anywhere an engine name is.

        cosmo = Cosmology(..., engine=emulated_engine(emu))
        cosmo.get_harmonic().lensed_cl()

    A class, not an instance, because that is what cosmoprimo's engine plumbing takes: ``clone``
    and ``set_engine`` instantiate it themselves against the cosmology being asked about.
    The emulator is then queried with that cosmology's parameters.
    """
    from cosmoprimo.cosmology import BaseEngine

    composite = emulator.section is None
    sections = emulator.sections

    def source(engine):
        """The prediction for this engine's cosmology, computed once and shared by the sections."""
        if getattr(engine, '_predicted', None) is None:
            # read the user's parameters off this cosmology -- `predict` converts them to the
            # training basis itself, and raises outside the trained box
            engine._predicted = emulator.predict(
                **{name: engine[name] for name in emulator.space.params})
        return engine._predicted

    built = {name: section.section_class(source, prefix=f'{name}.' if composite else '')
             for name, section in sections.items()}

    class EmulatedEngine(BaseEngine):
        """Engine backed by a trained emulator."""
        name = 'emulated'

        def __init__(self, cosmo, **extra_params):
            super().__init__(cosmo, **extra_params)
            self._predicted = None
            self._Sections = dict(built)

    return EmulatedEngine


def read(path):
    """Read a trained emulator back, of whatever kind wrote it.

    The counterpart to :meth:`~cosmoprimo.emulators.tools.Emulator.write`, exposed here so that
    loading never requires importing the template class, which in this namespace would collide
    with :func:`Emulator`, the cosmology entry point. That collision is why the template is
    imported as ``_Emulator`` in this module.
    """
    return _Emulator.read(path)


def read_engine(path):
    """The engine behind ``Cosmology(engine='my_emulator.npy')``."""
    return emulated_engine(read(path))


# ── the user-facing entry point ───────────────────────────────────────────────

#: Keywords :func:`emulate` forwards to :meth:`~cosmoprimo.emulators.tools.Emulator.train`
#: rather than to the emulator it builds. ``train`` passes anything else it is given on to the
#: engine, so a name may safely appear in both places.
_TRAIN_OPTIONS = ('budget', 'checkpoint', 'chunk', 'batch_size', 'mpicomm')


def Emulator(cosmo, space, section='harmonic', **options):
    """Build an emulator of one or more sections of a cosmology. To be trained.

        cosmo = Cosmology(engine='camb', lensing=True, ellmax_cl=3000)
        emu = Emulator(cosmo, Space(samples=chain), section='harmonic')
        emu.nodes(budget=4)                          # size the run before paying for it
        emu.train(budget=4, checkpoint='cl.npz', chunk='30min')

        emu.predict(h=0.68, omega_cdm=0.12)          # {'lensed_cl.tt': array, ...}
        emu.write('cl.h5')                            # and later, Cosmology(engine='cl.h5')
        fast = emu.to_cosmology()                   # a Cosmology, engine and all
        fast.clone(h=0.68, omega_cdm=0.12).get_harmonic().lensed_cl()

    Training is the expensive part (minutes/hours of Boltzmann calls), so it is deliberately a separate
    step here, giving you the chance to size it first. :func:`emulate` does both in one call
    when you already know what you want.

    Several sections at once share one Boltzmann call per node, which is the entire cost -- so ask
    for them together rather than training two emulators over the same grid::

        emulate(cosmo, space, section=['harmonic', 'background'])
        emulate(cosmo, space, section={'harmonic': dict(of=('lensed_cl', 'lens_potential_cl')),
                                       'fourier': dict(z=np.linspace(0., 3., 20))})

    Parameters
    ----------
    cosmo : Cosmology
        The fiducial: its engine and precision settings are what every training node is computed
        with, so set ``lensing``, ``ellmax_cl`` and any precision parameters on it first.
    space : Space
        Where accuracy is required. ``Space(samples=chain)`` is worth orders of magnitude more
        than plain ranges -- whitening onto the posterior's axes beat a box 350x at equal cost.
    section : str, list, dict, default='harmonic'
        A name gives that section's emulator, with its outputs named plainly
        (``'lensed_cl.tt'``). A list or dict gives a :class:`CosmologyEmulator` over all of them,
        outputs prefixed by section (``'harmonic.lensed_cl.tt'``); a dict also carries each
        section's own options.

    Other keyword arguments go to the emulator: a single section's own (``of``, ``ellmax``,
    ``k``, ``z``, ``non_linear``) plus ``engine``, ``coverage``, ``budget``, ``levels``.

    Returns
    -------
    emulator : SectionEmulator, CosmologyEmulator
        UNtrained. Call :meth:`~cosmoprimo.emulators.tools.Emulator.train`.

    Notes
    -----
    A function with a class's name, deliberately: it dispatches on ``section``, so what comes back
    is a :class:`HarmonicEmulator` or a :class:`CosmologyEmulator` rather than one fixed type.
    """
    from cosmoprimo import Cosmology

    if not isinstance(cosmo, Cosmology):
        raise TypeError(f'cannot emulate {type(cosmo).__name__}: give a Cosmology. For a plain '
                        f'callable, subclass cosmoprimo.emulators.tools.Emulator.')
    if isinstance(section, str):
        names = {section: {}}
    elif isinstance(section, dict):
        names = {name: dict(kwargs) for name, kwargs in section.items()}
    else:
        names = {name: {} for name in section}
    unknown = [name for name in names if name not in _SECTIONS]
    if unknown:
        raise ValueError(f'no emulator for section(s) {unknown}; available: {sorted(_SECTIONS)}')
    if not names:
        raise ValueError('no section given')
    if len(names) == 1 and isinstance(section, str):
        return _SECTIONS[section](cosmo, space, **options)
    return CosmologyEmulator(cosmo, space, names, **options)


def emulate(cosmo, space, section='harmonic', **options):
    """Build and train, in one call.

        emu = emulate(cosmo, Space(samples=chain), budget=2)
        emu.write('cl.h5')
        fast = emu.to_cosmology()

    The same as :func:`Emulator` followed by
    :meth:`~cosmoprimo.emulators.tools.Emulator.train`, for when the run is small enough that you
    do not need to size it first. For anything expensive prefer the two steps, and pass
    ``checkpoint`` and ``chunk``: a kill then costs one node rather than the training.

    Keyword arguments are routed by name -- :data:`_TRAIN_OPTIONS` to ``train``, the rest to
    :func:`Emulator`.

    Returns
    -------
    emulator : SectionEmulator, CosmologyEmulator
        trained. Call :meth:`~cosmoprimo.emulators.tools.Emulator.to_cosmology` for a cosmology,
        or :meth:`~cosmoprimo.emulators.tools.Emulator.write` to keep it.
    """
    training = {name: options.pop(name) for name in _TRAIN_OPTIONS if name in options}
    if 'engine' in options:                 # `engine` selects the interpolant in both places
        training['engine'] = options['engine']
    return Emulator(cosmo, space, section=section, **options).train(**training)
