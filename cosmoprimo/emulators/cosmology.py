r"""Cosmologies, as :class:`~cosmoprimo.emulators.tools.Emulator` subclasses.

This is everything :mod:`cosmoprimo.emulators.tools` deliberately does not know. It is CMB and
large-scale-structure physics:

- :math:`C_\ell \propto A_s`, :math:`P(k) \propto A_s` -- exact for the primary anisotropies and
  for the linear power spectrum, so the amplitude can leave the interpolation grid entirely and
  cost no nodes at all. not exact once lensing or halofit is applied: both are non-linear in the
  amplitude, so there dividing by :math:`A_s` only flattens the dependence and the parameter
  stays on the grid.

  :math:`\sigma_8` is the same statement in another spelling -- at fixed shape :math:`P \propto
  A_s \propto \sigma_8^2` -- so a space may be written in either, and a trained emulator serves a
  cosmology written in the other (:func:`_rsigma8`, and cosmoprimo's own rescaling convention).
- :math:`P(k, z) = D(z)^2 P(k)`, with one :math:`f(z)` per velocity leg, so a spectrum's z
  dependence goes to the interpolator as a callable rather than into the fitted array: what is
  left to spline in z is the scale-dependent residual, and a 8-node grid then does what 20 nodes
  could not.
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

from cosmoprimo.cosmology import Cosmology, BaseEngine, DefaultBackground
from cosmoprimo.jax import numpy as jnp, numpy_jax

# aliased: in this module `Emulator` is the user-facing entry point below,
# which dispatches on `section`; this is the template it all derives from
from .tools import Emulator as _Emulator, CoverageError, NotTrained


# the formulae divided out live in `analytic`, shared with desilike's cosmology emulators
from .analytic import (AMPLITUDES as _AMPLITUDES,
                       eisenstein_hu_scales as _eisenstein_hu_scales, harmonic_scaling,
                       theta_analytic, solve_theta_analytic, dilate)

# the columns each cosmoprimo harmonic getter returns, so the scaling can be built without a run
_SPECTRA = {'lensed_cl': ('tt', 'ee', 'bb', 'te'),
            'unlensed_cl': ('tt', 'ee', 'bb', 'te'),
            'lens_potential_cl': ('pp', 'tp', 'ep')}

#: sigma8 in every spelling, alongside :data:`~.analytic.AMPLITUDES`. Sampling it is sampling the
#: amplitude: at fixed shape :math:`P \propto A_s \propto \sigma_8^2` exactly, so whichever of the
#: two the space is written in, it is the same single number that scales out.
_SIGMA8 = ('sigma8',) + tuple(Cosmology._alias_parameters.get('sigma8', ()))


def _guard(values):
    r"""*values* where they can be divided by, and 1 where they cannot.

    A divisor has two ways of not being one, and both are silent: an exact zero (a distance at
    :math:`z = 0`) and a nan. The nan is not hypothetical --
    :meth:`~cosmoprimo.cosmology.DefaultBackground.growth_factor` integrates on
    :math:`\eta = \ln a \in [-6, 0]`, so it is nan above :math:`z \simeq 402`, which the default
    background grid (out to :math:`z = 1000`) reaches: 13.3% of its nodes came back nan, and
    dividing by them puts nan into the training values, where it poisons every coefficient of the
    fit rather than failing.

    Leaving a 1 there means those entries are fitted unconditioned, which is the honest fallback:
    the conditioning is a preconditioner, and a node it cannot precondition is still a node.

    The dispatching numpy, because the deployed sections apply this inside a jax trace.
    """
    np_ = numpy_jax(values)
    values = np_.asarray(values)
    return np_.where(np_.isfinite(values) & (values != 0.), values, 1.)


def _growth_factor_sq(background, of):
    r"""The growth a spectrum between these legs carries, as a callable of z: one :math:`D(z)` per
    leg, times :math:`f(z)` for each leg that is a velocity divergence.

    :math:`P_{\delta\delta} \propto D^2`, :math:`P_{\delta\theta} \propto f D^2` and
    :math:`P_{\theta\theta} \propto f^2 D^2` -- so the divisor is read off the spectrum's own legs
    rather than assumed to be :math:`D^2`. Getting that wrong is not a small error: at the
    redshifts a DESI analysis uses, :math:`f` runs from 0.5 to 0.9, and dividing a
    :math:`\theta\theta` spectrum by :math:`D^2` leaves the whole of :math:`f^2` for the
    interpolant, which is exactly the z dependence the divisor exists to remove.

    ``znorm=0``, the convention the engines' own spectra carry (:math:`D \sim a` in matter
    domination), not the default normalisation to 1 at :math:`z = 0`: being 1 at :math:`z = 0` for
    every cosmology by construction, the default leaves the absolute normalisation in, and that is
    a real cosmology dependence.

    A callable rather than an array, because that is what
    :class:`~cosmoprimo.interpolator.PowerSpectrumInterpolator2D` takes to apply the growth
    itself (see :meth:`FourierEmulator.section_class`); the training divisor is the same callable
    evaluated on the training grid, which is what keeps the two ends in step.
    """
    nrate = sum(str(leg).startswith('theta') for leg in _of_legs(of))

    def growth_factor_sq(z):
        # `jnp`, not `np`: at prediction this runs inside the trace (`inverse_transform` applies
        # the divisor), where the parameters are tracers and the growth is one too
        factor = jnp.asarray(background.growth_factor(z, znorm=0.)) ** 2
        if nrate:
            factor = factor * jnp.asarray(background.growth_rate(z)) ** nrate
        return _guard(factor)

    return growth_factor_sq


#: The dark-energy parameters, in every spelling a training basis may give them (see
#: :meth:`SectionEmulator.to_training`).
DARK_ENERGY = ('w0_fld', 'wa_fld', 'w', 'wa', 'w0pwa')


def _rsigma8(engine):
    r"""The amplitude ratio a cosmology given :math:`\sigma_8` needs, or 1.

    cosmoprimo's convention for a :math:`\sigma_8`-parameterised cosmology is that the engine
    runs at a first-guess amplitude and its perturbative outputs are then rescaled so that
    :math:`\sigma_8` comes out exactly right (:meth:`~cosmoprimo.cosmology.BaseEngine._rescale_sigma8`).
    That measurement needs a fourier section to read :math:`\sigma_8` back off, so an emulator
    without one -- a harmonic-only emulator, say -- cannot make it and does not have to: a space
    written in :math:`\sigma_8` already had :math:`\sigma_8^2` divided out as the amplitude
    (:meth:`SectionEmulator.amplitude`), so its predictions are at the asked-for amplitude
    already, and the ratio it would compute is 1 by construction.

    Exact either way, because the quantities it scales are exactly linear in the amplitude.
    """
    if 'sigma8' not in engine._params or 'fourier' not in engine._Sections:
        return 1.
    return engine._rescale_sigma8()


def _has_tensors(cosmo):
    """Whether this cosmology carries tensor modes, which is what makes a lensed ``bb`` a sum.

    Without them ``bb`` is generated entirely by the lensing and so is quadratic in the scalar
    amplitude; with them it also holds a tensor spectrum, linear in its own.
    """
    modes = cosmo['modes']
    modes = [modes] if isinstance(modes, str) else list(modes)
    return any('t' in str(mode) for mode in modes)


def _of_legs(of):
    """The two perturbed quantities a spectrum is between, as a 2-tuple.

    ``'delta_cb'`` is the auto spectrum ``('delta_cb', 'delta_cb')``; a pair is taken as given.
    """
    if isinstance(of, str):
        return (of, of)
    of = tuple(of)
    if len(of) == 1:
        return of * 2
    if len(of) != 2:
        raise ValueError(f'a spectrum is between two quantities; got {of}')
    return of


def _of_name(of):
    """The name a spectrum is stored under: ``'delta_cb'`` for an auto spectrum,
    ``'delta_cb_theta_cb'`` for a cross one.

    One name for one spectrum, whichever way the caller spelled it -- ``'delta_cb'``,
    ``('delta_cb',)`` and ``('delta_cb', 'delta_cb')`` are the same request, and an emulator that
    stored them under different keys would emulate the same thing twice and serve neither
    reliably.
    """
    legs = _of_legs(of)
    return legs[0] if legs[0] == legs[1] else '_'.join(legs)


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


def _get_basis(basis):
    """``basis`` as a list of names: a shorthand resolved, anything else taken as given.

    ``'physical'`` is the density basis the CMB and the matter power spectrum actually respond
    to; ``'theta'`` is that with the acoustic scale in place of :math:`h` (see
    :meth:`SectionEmulator.to_training`). Anything not named in a basis is passed through
    unchanged, so a shorthand doubles as the list of names it replaces.
    """
    if basis is None:
        return None
    shorthands = {'physical': ('omega_cdm', 'omega_b', 'h'),
                  'theta': ('omega_cdm', 'omega_b', 'theta_MC_100')}
    return list(shorthands.get(basis, basis)) if isinstance(basis, str) else list(basis)


def _with_total_m_ncdm(params):
    """``m_ncdm`` as the emulator holds it: the total.

    A cosmology carries one mass per massive species; a ``neutrino_hierarchy`` is the statement
    about how a total splits into them. So the total is the one number that describes the set, it
    is the number `clone` takes back, and it is what the emulator uses throughout -- in the space,
    in the node table, and in what `predict` stacks, where a (3,) among scalars is what refused
    with "All input arrays must have the same shape".
    """
    if 'm_ncdm' not in params:
        return params
    return {**params, 'm_ncdm': jnp.sum(jnp.asarray(params['m_ncdm']))}


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
        space's own; ``'physical'`` and ``'theta'`` are the shorthands :func:`_get_basis`
        resolves.

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

        ``'w0pwa'`` is not a density basis and composes with any of them -- a shorthand cannot
        be spelled alongside it, so give the names in full: ``basis=['omega_cdm', 'omega_b',
        'theta_MC_100', 'w0pwa']``. It re-expresses :math:`(w_0, w_a)` by :math:`(w_0, w_0 +
        w_a)` so that :meth:`transforms` can put a logit on the sum, which is the quantity a
        w0waCDM analysis actually bounds: CAMB's PPF refuses :math:`w_0 + w_a > 0` ("giving
        w > 0 at high redshift"), no per-axis limit expresses that, and a Smolyak grid is
        unisolvent, so one node past the bound is not a smaller problem but a singular one.
    """
    section = None

    def __init__(self, cosmo, space, basis=None, **options):
        self.cosmo = cosmo
        self.basis = _get_basis(basis)
        # analytic backgrounds already built, see `analytic_background`. A dict rather than one
        # entry, because the sections of a composite share it and each may ask at its own
        # parameters, and shared rather than per section because they all clone the same fiducial.
        self._analytic_cache = {}
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
        params = _with_total_m_ncdm(params)
        if self.basis is None:
            return params
        from cosmoprimo import Cosmology

        names = self._basis_names()
        theta = 'theta_MC_100' in names
        w0pwa = 'w0pwa' in names
        # `_get_params` is asked for the names it knows; the two derived ones are put back below
        names = ['h' if name == 'theta_MC_100' else
                 'wa_fld' if name == 'w0pwa' else name for name in names]
        if set(names) <= set(params):
            # Already in the basis, so the converter is a pass-through -- measured exactly zero
            # difference -- and skipping it is not an optimisation of the arithmetic but of the
            # cost of getting there: `_get_params` normalises a whole parameter compilation,
            # neutrino-mass solve included, at 12.6 ms a point. That is paid once per point of
            # the space when `training_space` maps it, so on the default 100000 draws the
            # difference is 21 minutes of construction against none.
            params = {name: params[name] for name in names}
        else:
            # `_get_params` hands `m_ncdm` back as the masses, being a cosmology's converter
            params = _with_total_m_ncdm(
                Cosmology._get_params(dict(params), names, base=self.cosmo._input_params))
        if theta:
            params['theta_MC_100'] = 100. * theta_analytic(
                params.pop('h'), params['omega_b'], params['omega_cdm'],
                **self._theta_kwargs(params))
        if w0pwa:
            # after theta, which reads the pair through `_theta_kwargs` and wants it in the
            # cosmology's own names
            params['w0pwa'] = params.pop('wa_fld') + params['w0_fld']
        return params

    def from_training(self, params):
        """The inverse: the calculator is cloned with these, and ``theta_MC_100`` is not
        something :meth:`~cosmoprimo.cosmology.Cosmology.clone` accepts.

        ``h`` comes back through :func:`~.analytic.solve_theta_analytic`, a bisection on the same
        closed form -- not through :meth:`~cosmoprimo.cosmology.Cosmology.solve`, which runs a
        background per iteration and would break the exact round trip above. A bisection whose
        bracket misses the root returns nan rather than an endpoint, so a node that cannot be
        placed is reported instead of being fitted at the wrong theta.

        ``w0pwa`` is undone first and by subtraction, because ``theta`` reads the dark-energy pair
        (:meth:`_theta_kwargs`) and must see it in the cosmology's own names -- the pair it is
        given has to be the one :meth:`to_training` used, or the round trip is not the identity.

        Every other basis needs no inverse: its names are all ``clone`` inputs.
        """
        if self.basis is None or not any(name in params for name in ('theta_MC_100', 'w0pwa')):
            return params
        params = dict(params)
        if 'w0pwa' in params:
            params['wa_fld'] = params.pop('w0pwa') - params['w0_fld']
        if 'theta_MC_100' in params:
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
                  'N_ur': params.get('N_ur', self.cosmo['N_ur']),
                  'T_cmb': params.get('T_cmb', self.cosmo['T_cmb'])}
        # `m_ncdm` is a total here (:func:`_with_total_m_ncdm`) and theta wants the species, so
        # scale the fiducial's own masses to it: exact under the degenerate hierarchy base_mnu
        # runs, the identity with a single massive species. Handing theta the total as a single mass instead
        # is 7.7% off at every total from 0.06 to 0.45, because `N_ur` still counts three massive
        # species (0.0064 for the DESI fiducial) and two of them then contribute nothing -- the
        # scaled split sits 0.13% from cosmoprimo's own `theta_MC_100`, the near-constant offset
        # the round trip cancels, where the single-mass offset drifts. An array either way, so a
        # compiled `theta_analytic` sees one argument structure rather than retracing between a
        # tuple of masses and an array of them.
        m_ncdm = np.atleast_1d(self.cosmo['m_ncdm'])
        if 'm_ncdm' in params:
            m_ncdm = m_ncdm * (params['m_ncdm'] / np.sum(m_ncdm))
        kwargs['m_ncdm'] = m_ncdm
        return kwargs

    #: The names a density basis stands in for. Anything else the space varies is passed
    #: through untouched.
    _REPLACED = ('Omega_m', 'Omega_cdm', 'Omega_b', 'H0', 'h', 'omega_m', 'omega_cdm', 'omega_b')

    @property
    def w0pwa(self):
        """Whether the basis re-expresses the dark-energy pair by its sum, and can."""
        return (self.basis is not None and 'w0pwa' in self.basis
                and {'w0_fld', 'wa_fld'} <= set(self.space.params))

    def _basis_names(self):
        """The training names: the requested basis, plus everything it does not replace.

        A basis change is a reparametrisation, so it must not change the dimension. The
        density part and the dark-energy part are counted separately, because they replace
        different things and either may be used without the other.
        """
        density = [name for name in self.basis if name != 'w0pwa']
        replaced = [name for name in self.space.params if name in self._REPLACED]
        # An empty density part asks for no density reparametrisation at all -- `basis=['w0pwa']`
        # is a legitimate request, and the densities are then pass-through like everything else.
        if density and len(density) != len(replaced):
            raise ValueError(
                f'basis {density} has {len(density)} parameters, but the space varies '
                f'{len(replaced)} of the ones it stands in for ({replaced}). A basis change is a '
                f'reparametrisation and cannot add a direction: either vary the missing '
                f'parameter in the Space, or give a basis of {len(replaced)} names.')
        if 'w0pwa' in self.basis and not self.w0pwa:
            raise ValueError(
                f"basis asks for 'w0pwa', which stands in for 'wa_fld', but the space varies "
                f"{sorted(set(self.space.params) & {'w0_fld', 'wa_fld'})} -- the sum is only a "
                f"reparametrisation when both are varied.")
        # what this basis actually stands in for, which is not the same as what another basis could:
        # with no density part, `h` and the densities are pass-through like anything else. Getting
        # this wrong drops them from the training parameters altogether, and an emulator trained
        # with the densities frozen answers confidently and cannot respond to them at all
        # (measured: |dchi2| against exact CLASS 377 where the same box through another backend
        # gave 32).
        replaced = (list(self._REPLACED) if density else []) + (['wa_fld'] if self.w0pwa else [])
        passthrough = [name for name in self.space.params if name not in replaced]
        return list(self.basis) + [name for name in passthrough if name not in self.basis]

    def transforms(self):
        """The expansion variables the training box needs, as ``{name: transform}``.

        Only for a name the basis introduces: :meth:`~.tools.space.Space.map` refuses one for a
        pass-through, whose limits are already in its own expansion variable, and applying a
        transform twice is how a logit becomes a nan.

        Two of them, both making a bound unreachable rather than an edge to trim back from:
        the logit on ``'w0pwa'``, and a log on each density fraction. Such a fraction is strictly
        positive and its mapped box is a plain rectangle, so a wide one crosses zero -- measured
        at 3.75 sigma on a DESI+CMB box, ``Omega_cdm`` reached -0.037 and ``Omega_b`` -0.0065,
        and a negative density is a non-finite background rather than a slightly wrong one.
        """
        introduced = [name for name in self._basis_names() if name not in self.space.params]
        transforms = {}
        for name in introduced:
            if name == 'w0pwa':
                transforms[name] = 'logit_w0pwa'
            elif name.startswith('Omega_'):
                transforms[name] = 'log'
        return transforms

    def training_space(self, uncorrelated=True):
        """The space to lay the nodes over: the user's, re-expressed in :attr:`basis`.

        Uncorrelated by default, which is a statement about the node cloud rather than about the
        region. Whitened, the nodes fill a band across the box and its off-diagonal corners hold
        none, so :meth:`~.tools.engines.BaseEngine.outside` refuses points that are inside every
        one of their own limits -- and refusing them is right, since the fit has nothing there.
        Measured on a DESI full-shape box, 20 draws at a quarter of it: the whitened emulator
        answered 14, the uncorrelated one all 20. desilike's own cosmology emulators found the
        same thing at production scale (66.68% of a posterior answered against 99.87%), where the
        truncation reached the chain as a hard wall while Gelman-Rubin read 1.00.

        The whitening is not what buys the accuracy: at equal budget both arms scored the same
        |dchi2| there, and with a regression engine drawing its candidates from the space's own
        samples the nodes are posterior-shaped either way. It is worth reconsidering for a
        collocation grid, where the rotation is what orients the grid onto the posterior.
        """
        space = self.space if self.basis is None else \
            self.space.map(self.to_training, transforms=self.transforms())
        return space.uncorrelated() if uncorrelated else space

    def clone(self, params, engine=None):
        """The fiducial with *params* applied, in the cosmology's own names.

        A name the cosmology does not know is refused rather than passed on: `clone` ignores such a
        name silently, so a training parameter that reached it -- ``theta_MC_100`` or ``w0pwa``,
        which only a basis speaks -- left that quantity at the fiducial's value and said nothing.
        That is how the analytic divisor came to be built at ``wa_fld = 0`` for nodes with
        ``wa_fld = -1.65``; see :meth:`analytic_background`. Call :meth:`from_training` first.
        """
        known = set(self.cosmo.get_default_params(include_conflicts=True))
        unknown = [name for name in params if name not in known]
        if unknown:
            raise ValueError(f'{type(self).__name__} cannot clone the fiducial cosmology with '
                             f'{sorted(unknown)}: not cosmology parameters. A training basis has '
                             f'to be undone with `from_training` before cloning -- `clone` would '
                             f'ignore these and hold what they stand for at the fiducial.')
        try:
            return self.cosmo.clone(engine=engine, **params)
        except Exception as exc:
            raise ValueError(f'{type(self).__name__} could not clone the fiducial cosmology with '
                             f'{sorted(params)}: {exc}') from exc

    # ── what a subclass writes ────────────────────────────────────────────────
    def extract(self, cosmo):
        """Named arrays out of an already computed cosmology. No scaling here."""
        raise NotImplementedError

    def amplitude(self, params):
        r"""The amplitude to divide out, in whatever spelling the space uses -- ``logA``,
        ``ln10^10A_s``, or ``sigma8`` -- or None if the space varies no amplitude at all.

        ``A_s`` is derived by the cosmology rather than by hand: ``A_s = 1e-10 exp(logA)`` is a
        convention that :meth:`~cosmoprimo.cosmology.Cosmology._compile_params` already
        implements, along with every alias, and a second copy here would be one more thing to keep
        in step.

        A space written in :math:`\sigma_8` gets :math:`\sigma_8^2`, and that is not an
        approximation of the ``A_s`` route but the same statement: at fixed shape :math:`P \propto
        A_s \propto \sigma_8^2` exactly, so either divides the amplitude out completely and lets
        the parameter leave the grid. It is deliberately left unconverted: turning it into
        ``A_s`` takes a Boltzmann call (:meth:`~cosmoprimo.cosmology.BaseEngine._rescale_sigma8`
        runs the code once and reads :math:`\sigma_8` back off it), so at prediction time there
        is nothing to convert it with, and the fitting formula that stands in for it
        (:meth:`~cosmoprimo.cosmology.BaseEngine._get_A_s_fid`) is good to a few per cent, which
        is not a divisor. Using :math:`\sigma_8^2` needs no conversion in either direction.
        The measured invariance of :math:`\sigma_8^2 / A_s` under the amplitude is 1e-15.
        """
        from cosmoprimo import Cosmology

        if any(name in params for name in _AMPLITUDES):
            return Cosmology._get_params(dict(params), ['A_s'],
                                         base=self.cosmo._input_params)['A_s']
        for name in _SIGMA8:
            if name in params:
                return params[name] ** 2
        return None

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

        The clone asks for a plain :class:`~cosmoprimo.cosmology.BaseEngine` rather than
        inheriting the fiducial's. Inheriting it re-runs the Boltzmann engine's constructor, which
        is both wasted work -- nothing of it is used, the ODEs are solved from the parameters --
        and not traceable: `classy` tests `_has_fld`, a comparison on ``w0_fld`` and ``wa_fld``,
        as a python bool, so a jitted prediction over a dark energy space raised before it ever
        reached the divisor.

        *params* may be in the training basis, and that is converted here rather than at each call
        site. It has to be: ``clone`` takes the cosmology's own names and quietly ignores anything
        else, so a basis name reached it as a no-op and the divisor was built at the fiducial value
        of whatever the basis had replaced. Measured on a w0waCDM box in the ``theta_MC_100`` +
        ``w0pwa`` basis, at a node with ``wa_fld = -1.65`` and ``h = 0.64``: the divisor came out at
        ``wa_fld = 0`` and ``h = 0.6736``, the fiducial's, which is 15% in ``efunc`` over z < 3.
        The section then fits that residual instead of dividing it out -- invisible while every
        output was expanded in the basis names, since the polynomial absorbed it, and 1e-2 errors
        in the background as soon as a section was fitted in coordinates of its own.
        """
        try:
            key = tuple(sorted((name, float(value)) for name, value in params.items()))
        except (TypeError, ValueError):
            # a traced parameter has no float(), so key on the identity of the values instead: a
            # tracer is only ever equal to itself, and within one trace the same tracer reaching
            # this twice is the same point. The parameters are kept alive alongside the result,
            # which is what makes the identities safe -- an id is only unique while its object is.
            key = ('traced',) + tuple(sorted((name, id(value)) for name, value in params.items()))
        if key not in self._analytic_cache:
            if len(self._analytic_cache) >= 8:
                # the sections of a composite hold a few keys at once (they select different
                # parameters); more than that is a new point, and the old ones are dead weight
                self._analytic_cache.clear()
            # `transform` and `inverse_transform` are called with the same params back to back,
            # and every section of a composite asks for the same background again. Building it
            # each time is what a GPU feels: the background carries cubic splines, and a spline
            # build is a tridiagonal solve, which is a kernel launch that dwarfs the arithmetic.
            self._analytic_cache[key] = (dict(params), DefaultBackground(
                BaseEngine(self.clone(self.from_training(dict(params)), engine=BaseEngine))))
        return self._analytic_cache[key][1]

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
        # not saved, since it holds computed backgrounds rather than anything describing the
        # emulator, but a reloaded section predicts and therefore needs one
        self._analytic_cache = {}
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

# What a Boltzmann code needs to be asked for before its Cl are worth emulating. These are
# accuracy settings, so they belong to the cosmology the nodes are computed with rather than to
# the emulator: the nodes are the truth the fit is only as good as, and an under-resolved node is
# an error no budget removes. `non_linear` and `ellmax_cl` are cosmoprimo calculation parameters (set like
# any other); the rest are raw engine precision knobs, forwarded through `extra_params`.
_LENSING_PARAMS = {'camb': dict(non_linear='mead2016'), 'class': dict(non_linear='hmcode')}
_LENS_POTENTIAL_EXTRA_PARAMS = {
    'camb': dict(lens_margin=1250, lens_potential_accuracy=4,
                 AccuracyBoost=1, lSampleBoost=1, lAccuracyBoost=1),
    'class': dict(nonlinear_min_k_max=20, accurate_lensing=1, delta_l_max=800)}
# CAMB needs ell reach beyond the requested ellmax for `lens_margin` to have room to work with;
# CLASS's `delta_l_max` already provides that margin relative to whatever ellmax_cl is.
_LENS_POTENTIAL_MIN_ELLMAX = {'camb': 4000}


def with_harmonic_precision(cosmo, of=('lensed_cl',), ellmax=None):
    r"""*cosmo*, cloned with what the requested spectra need computed and how well.

        cosmo = with_harmonic_precision(Cosmology(engine='camb'), of=('lensed_cl',), ellmax=3000)
        emu = Emulator(cosmo, space, section='harmonic', of=('lensed_cl',), ellmax=3000)

    This is where a fixed :math:`\ell_\mathrm{max}` is fixed. An emulator's grids are settled
    when it is built, not by whoever reads it later, and the two halves of that -- what the
    training cosmology is asked to compute, and what the emulator stores -- have to agree or the
    nodes are truncated where the fit is not. Pass the same ``ellmax`` to both.

    What it sets, and why each is not optional:

    * ``lensing``: without it, ``lensed_cl`` and ``lens_potential_cl`` are not computed at all and
      the getters raise.
    * a non-linear matter power (``mead2016`` / ``hmcode``): the default settings under-resolve
      the deflection power that lensing is applied with.
    * for the lensing potential, the reconstruction accuracy boost, and CAMB's internal
      :math:`\ell` reach raised to at least 4000.

    Left alone otherwise: an ``ellmax_cl`` already on *cosmo* is only ever raised, never lowered,
    since a larger one is an accuracy choice that a smaller request must not undercut.
    """
    of = (of,) if isinstance(of, str) else tuple(of)
    unknown = [name for name in of if name not in _SPECTRA]
    if unknown:
        raise ValueError(f'unknown harmonic spectra {unknown}; available {sorted(_SPECTRA)}')
    engine = getattr(getattr(cosmo, '_engine', None), 'name', None)
    lens_potential = 'lens_potential_cl' in of
    lensing = lens_potential or 'lensed_cl' in of
    params, extra_params = {}, {}
    if lensing:
        params['lensing'] = True
        params.update(_LENSING_PARAMS.get(engine, {}))
    if lens_potential:
        extra_params.update(_LENS_POTENTIAL_EXTRA_PARAMS.get(engine, {}))
        ellmax = max(ellmax or 0, _LENS_POTENTIAL_MIN_ELLMAX.get(engine, 0)) or None
    if ellmax is not None:
        params['ellmax_cl'] = max(int(ellmax), int(cosmo['ellmax_cl']))
    if extra_params:
        merged = dict(getattr(getattr(cosmo, '_engine', None), '_extra_params', None) or {})
        merged.update(extra_params)
        params['extra_params'] = merged
    return cosmo.clone(**params)


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
        # sigma8 is the amplitude under another name (see `SectionEmulator.amplitude`), so it
        # leaves the grid on the same terms: exactly for the primary anisotropies, not once
        # lensing has been applied
        return [name for name in names if name not in _AMPLITUDES + _SIGMA8]

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

    def section_class(self, source, prefix=''):
        from cosmoprimo.cosmology import BaseSection

        emulator = self

        class Harmonic(BaseSection):

            def __init__(self, engine):
                super().__init__(engine)
                self._engine = engine
                self._cl = source(engine)
                # a cosmology given sigma8 where the emulator was trained in A_s: the same ratio
                # the fourier section is rescaled by applies here. 1 whenever sigma8 is not an
                # input, so this costs nothing then -- and a harmonic-only emulator, which has no
                # fourier section to measure the ratio with, only ever sees that case.
                self._rsigma8 = _rsigma8(engine)
                # lensing B modes are generated by the lensing itself, so they carry the amplitude
                # twice: `_amplitude_power` is the exponent each spectrum takes. Measured against
                # CAMB over 2 <= l <= 2500 by computing at A_s and at r^2 A_s for
                # r^2 = 0.94 and 1.06 and fitting C(r^2 A_s) = r^(2p) C(A_s): p = 1.000 for every
                # unlensed spectrum, for lensed tt and te and for the lensing potential, 0.996 for
                # lensed ee -- and 2.06 for lensed bb. Scaling bb linearly leaves 6.7e-2 of its own
                # peak; scaling it quadratically leaves 7.7e-4. The same with the non-linear matter
                # power on, which changes none of these by more than 30%: what the linear rescaling
                # misses here is the lensing, not halofit.
                self._amplitude_power = {}
                if not _has_tensors(emulator.cosmo):
                    # with tensor modes the lensed bb is a sum of a tensor spectrum (linear in its
                    # own amplitude) and the lensing one, and no single power describes it
                    self._amplitude_power['lensed_cl.bb'] = 2.
                # the l grid is captured from the engine at the first evaluation -- but a
                # training that resumes from a complete checkpoint evaluates no node, and neither
                # does a rank that was given none, so the object that wrote the file may never
                # have seen a spectrum. The stored arrays say the same thing: a spectrum starts at
                # l = 0 and is contiguous, so their length is the grid.
                held = next(len(value) for name, value in self._cl.items()
                            if name.startswith(prefix))
                self.ell = np.arange(held) if emulator.ell is None else np.asarray(emulator.ell)
                if len(self.ell) != held:
                    raise ValueError(f'the emulator holds {held} multipoles and its l grid has '
                                     f'{len(self.ell)}: the file disagrees with itself')
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
                                  * self._rsigma8**(2 * self._amplitude_power.get(f'{of}.{name}', 1.))
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
_BACKGROUND = ('efunc', 'comoving_radial_distance', 'comoving_transverse_distance',
               'angular_diameter_distance', 'luminosity_distance', 'growth_factor',
               'growth_rate', 'time',
               'Omega_b', 'Omega_cdm', 'Omega_m', 'Omega_ncdm_tot', 'Omega_de', 'Omega_k',
               'Omega_g', 'Omega_ur', 'Omega_r')

#: What a background emulator serves when it is not told: distances and growth.
#:
#: The density fractions are available but not on by default, because for the standard expansion
#: they are a function of the input parameters and :math:`E(z)` and so are already served exactly
#: by :class:`~cosmoprimo.cosmology.DefaultBackground` -- there is nothing for a node to add.
#: They earn their place the moment the expansion is not the standard one: a dark-energy or
#: modified-gravity engine solves a different :math:`H(z)`, every :math:`\Omega_i(z) = \rho_i /
#: \rho_\mathrm{crit}(z)` moves with it, and the analytic core then has the wrong denominator.
#: Ask for them there -- ``of=_BACKGROUND`` -- and the analytic version is used as a divisor
#: rather than as an answer, which is what conditioning is for: what the formula gets right costs
#: no accuracy, and what it gets wrong stays on the grid.
_BACKGROUND_DEFAULT = ('efunc', 'comoving_radial_distance', 'angular_diameter_distance',
                       'luminosity_distance', 'growth_factor', 'growth_rate')

#: Background quantities that are one number rather than a function of z.
_BACKGROUND_SCALAR = ('age',)


class BackgroundEmulator(SectionEmulator):
    """Distances and growth over a redshift grid.

    Parameters
    ----------
    z : array, default=None
        The grid. Log-spaced in ``1 / (1 + z)`` out to
        :data:`~cosmoprimo.cosmology._GROWTH_ZMAX` by default, which resolves the low-z distances
        and the early ones on the same axis. That is where it stops because it is where the growth
        stops: the same grid has to carry both, and a distance the growth cannot accompany is a
        node the analytic divisor cannot condition.
    of : tuple, default=None
        Which quantities; :data:`_BACKGROUND_DEFAULT` by default, and anything in
        :data:`_BACKGROUND` (the density fractions included) or :data:`_BACKGROUND_SCALAR`
        (``age``) on request.
    analytic : bool, default=True
        Fit the ratio to :meth:`~SectionEmulator.analytic_background` rather than the quantity.

        On by default because the ratio is 1 to 6e-13 for ``efunc``, ``growth_factor`` and
        ``growth_rate``, and to 2.3e-4 for the distances -- the analytic background solves the
        same ODEs, so the Boltzmann code adds almost nothing here.
        If the background is all you want, in (open) w0wamnuCDM cosmology, do not emulate it at all, just use
        ``DefaultBackground``. This section earns its place when it rides along with a harmonic
        or fourier one, sharing their Boltzmann call for free -- or when the engine's expansion
        is not the one the analytic core solves, which is exactly when the ratio stops being 1
        and the nodes start earning their keep.

        Where the analytic background cannot be a divisor the quantity is fitted unconditioned
        rather than divided by a nan. That used to be 13% of the default grid: the growth was
        integrated only to :math:`z \\simeq 402` while the grid ran to :math:`z = 1000`, so every
        node above it divided by a nan, silently, and nan in the training values poisons every
        coefficient of the fit. The two now share :data:`~cosmoprimo.cosmology._GROWTH_ZMAX`, and
        the guard is what remains for a quantity that is genuinely undefined somewhere.
    """
    section = 'background'
    #: 2: nan/zero divisors are guarded rather than propagated, so a file written before it holds
    #: coefficients fitted against nan.
    #: 3: the divisor is built at the node's own cosmology, not at the fiducial's `h` and `wa_fld`
    #: -- see :class:`CosmologyEmulator`.
    version = 3

    def __init__(self, cosmo, space, z=None, of=None, analytic=True, **options):
        self.analytic = bool(analytic)
        from cosmoprimo.cosmology import _GROWTH_ZMAX

        self.z = np.asarray(z, dtype='f8') if z is not None \
            else 1. / np.logspace(-np.log10(1. + _GROWTH_ZMAX), 0., 256)[::-1] - 1.
        self.of = tuple(of) if of is not None else _BACKGROUND_DEFAULT
        known = _BACKGROUND + _BACKGROUND_SCALAR
        unknown = [name for name in self.of if name not in known]
        if unknown:
            raise ValueError(f'unknown background quantities {unknown}; '
                             f'available {list(known)}')
        super().__init__(cosmo, space, **options)

    @property
    def scalars(self):
        """Those of :attr:`of` that are one number rather than a function of z."""
        return tuple(name for name in self.of if name in _BACKGROUND_SCALAR)

    def _values(self, background):
        """``{name: value}`` off any background object -- the engine's, or the analytic one.

        The dispatching ``asarray``, because the analytic background is read at prediction too,
        inside the trace, where every one of these is a tracer.
        """
        values = {}
        for name in self.of:
            value = getattr(background, name)
            if name not in _BACKGROUND_SCALAR:
                value = value(self.z)
            values[name] = numpy_jax(value).asarray(value)
        return values

    def extract(self, cosmo):
        return self._values(cosmo.get_background())

    def select_params(self, names):
        r"""Everything but the dark energy, once the analytic background is divided out.

        :class:`~cosmoprimo.cosmology.DefaultBackground` solves the same w0waCDM expansion the
        engine does -- measured against CAMB, ``efunc``, ``growth_factor`` and ``growth_rate``
        agree to 6e-13 and the distances to 2.3e-4 -- so what :math:`w_0` and :math:`w_a` do to a
        background is what the divisor already does, and a node spent on them buys nothing.

        It also keeps the grid off `w0 + wa > 1/3`, which CLASS refuses outright.
        """
        if not self.analytic:
            return list(names)
        return [name for name in names if name not in DARK_ENERGY]

    def scaling(self, params):
        if not self.analytic:
            return {}
        analytic = self._values(self.analytic_background(params))
        return {name: _guard(value) for name, value in analytic.items()}

    def section_options(self):
        return {**super().section_options(), 'z': self.z, 'of': self.of,
                'analytic': self.analytic}

    def section_class(self, source, prefix=''):
        from cosmoprimo.cosmology import BaseSection
        from cosmoprimo.jax import Interpolator1D

        emulator = self

        scalars = emulator.scalars

        class Background(BaseSection):

            def __init__(self, engine):
                super().__init__(engine)
                self._engine = engine
                values = source(engine)
                self._values = {name: values[f'{prefix}{name}'] for name in scalars}
                # a nan prediction (a point outside the trained region) passes straight through:
                # `Interpolator1D` sanitises what its solve sees and puts the nan back itself
                self._interp = {name: Interpolator1D(emulator.z, values[f'{prefix}{name}'])
                                for name in emulator.of if name not in scalars}

        def _make(name):

            def getter(self, z):
                # `jnp` rather than `np`: a redshift may be a tracer -- a supernova likelihood
                # that fits its own redshift errors differentiates through this -- and
                # `np.asarray` on a tracer is a TracerArrayConversionError
                return self._interp[name](jnp.asarray(z))

            getter.__name__ = name
            getter.__doc__ = f'Emulated :meth:`{name}`, interpolated over the training grid.'
            return getter

        def _make_scalar(name):

            def getter(self):
                return self._values[name]

            getter.__doc__ = f'Emulated :attr:`{name}`.'
            return property(getter)

        for name in emulator.of:
            setattr(Background, name, _make_scalar(name) if name in scalars else _make(name))
        return Background


# ── fourier ───────────────────────────────────────────────────────────────────

class FourierEmulator(SectionEmulator):
    r"""The matter power spectrum on a :math:`(k, z)` grid.

    Parameters
    ----------
    k : array, default=None
        Wavenumbers, :math:`h/\mathrm{Mpc}`. 512 points log-spaced over 1e-4 to 10 by default.

        This grid is settled at training time, so every consumer reads the spectrum back through
        :class:`~cosmoprimo.interpolator.PowerSpectrumInterpolator2D` rather than on its own
        wavenumbers, and the resampling that costs is the price of not having to know who asks.
        Measured against CLASS's own dense evaluation of the same cosmology, worst of five
        cosmologies across a DESI-like w0waCDM box, max :math:`|P_\mathrm{interp}/P - 1|` over
        :math:`[10^{-4}, 10]`: 6.6e-4 at 200 points, 9.9e-5 at 300, 3.7e-5 at 400, 1.2e-5 at 600.
        The error is a cubic spline's :math:`\mathrm{d}\ln k^4` and it sits on the BAO wiggles, so
        it is spread over exactly the scales an analysis fits. 512 buys ~2e-5 for an array that is
        still small; drop to 300 only if the outputs are the bottleneck.
    z : array, default=None
        Redshifts. 12 points, spaced as :math:`\sqrt{z}` out to 4, by default.

        Far fewer than the raw z dependence would need, because ``analytic`` hands the growth to
        the interpolator separately (see :meth:`section_class`) and what is splined in z is the
        ratio, which is flat. Same measurement as above, max over k and z, worst cosmology and
        worst ``of``: splining :math:`P` itself needs 20 nodes to reach 1.6e-5 and is at 4.8e-2
        with 5; splining :math:`P / D^2` reaches 3.7e-5 at 8 nodes and 1.7e-3 at 5. One node is
        not enough (2.8e-2): what is left in z after the growth is divided out is the
        scale-dependent part -- massive neutrinos, and the w0waCDM growth the analytic core only
        approximates -- and that is a real z dependence, not a resampling artefact.
    of : tuple, default=('delta_m',)
        Which spectra. A name is an auto spectrum (``'delta_cb'``); a pair is a cross one
        (``('delta_cb', 'theta_cb')``), stored as ``'pk.delta_cb_theta_cb'``.
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
    #: 3: the growth divisor counts the velocity legs (see :func:`_growth_factor_sq`), and the
    #: served spectrum hands the growth to the interpolator rather than baking it into the array,
    #: so a file written before it would predict confidently and wrongly rather than fail.
    #: 4: the divisor is built at the node's own cosmology -- see :class:`CosmologyEmulator`.
    version = 4

    def __init__(self, cosmo, space, k=None, z=None, of=('delta_m',), non_linear=False,
                 analytic=True, tilt=True, dilate=False, **options):
        self.analytic = bool(analytic)
        self.tilt = bool(tilt) and not non_linear
        self.dilate = bool(dilate) and not non_linear
        self.k = np.asarray(k, dtype='f8') if k is not None else np.logspace(-4., 1., 512)
        self.z = np.asarray(z, dtype='f8') if z is not None else np.linspace(0., 4.**0.5, 12)**2
        # a bare name is one spectrum, not a sequence of one-letter ones
        self.of = (of,) if isinstance(of, str) else tuple(of)
        self.non_linear = bool(non_linear)
        super().__init__(cosmo, space, **options)

    @property
    def names(self):
        """``{output name: of}`` -- the canonical name each requested spectrum is stored under."""
        return {f'pk.{_of_name(of)}': of for of in self.of}

    def extract(self, cosmo):
        fourier = cosmo.get_fourier()
        values = {}
        for name, of in self.names.items():
            # passed only when asked for: `non_linear=False` is every engine's default, and the
            # ones that cannot do halofit at all (eisenstein_hu) reject the keyword outright
            # rather than ignore it, so naming it made them unemulatable
            interpolator = fourier.pk_interpolator(of=of,
                                                   **({'non_linear': True} if self.non_linear else {}))
            values[name] = np.asarray(interpolator(self.k, self.z), dtype='f8')
        return values

    def select_params(self, names):
        # P(k) is exactly linear in A_s and an exact power law in n_s -- but halofit is neither,
        # so they only leave the grid for the linear spectrum
        if self.non_linear:
            return list(names)
        # sigma8 alongside A_s: the divisor is sigma8^2 there (see `amplitude`), exact for the
        # same reason, so the parameter leaves the grid for the same reason
        exact = _AMPLITUDES + _SIGMA8 + (('n_s',) if self.tilt else ())
        # `h` only where the rest of the space is h-free: the dilation holds the physical
        # densities fixed, and a space written in `Omega_m` does not -- taking `h` off the grid
        # there would interpolate in `Omega_m` at an implied `omega_cdm` that moves with the `h`
        # the dilation is meanwhile handling
        if self.dilate and not any(name in names for name in ('Omega_m', 'Omega_cdm', 'Omega_b',
                                                              'omega_m')):
            exact = exact + ('h', 'H0')
        # The dark energy, when the analytic growth is divided out: at fixed physical densities
        # `w0` and `wa` reach a linear spectrum only through the late-time growth -- the transfer
        # function is set long before dark energy matters -- so the divisor removes them rather
        # than flattening them, and they cost no node. desilike's cosmology emulators take the
        # same view and measured the residual below 1.4e-4 from k = 3e-3 up for `wa` moved by
        # 0.5, reaching 7e-3 only at k = 1e-3.
        #
        # It also keeps the node set off a corner no Boltzmann code will evaluate: a grid that
        # varies the pair reaches `w0 + wa > 1/3`, where CLASS refuses outright.
        if self.analytic:
            exact = exact + DARK_ENERGY
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
            # pk arrays are (k, z); the tilt varies along the first axis. `jnp`, since at
            # prediction `h` and `n_s` are tracers and so is the tilt.
            factor = factor * jnp.asarray(tilt)[:, None]
        if not self.analytic:
            if amplitude is None and not self.tilt:
                return {}
            return {name: factor for name in self.names}
        # The growth is the one part of the scaling that is not common to every output -- it
        # counts each spectrum's own legs (`_growth_factor_sq`) -- which is why the dict is built
        # per name rather than shared. Its absolute normalisation, which that function keeps, is
        # what `dilate` depends on: with h on the grid the interpolant absorbs a relative one
        # either way (measured: 1.65e-3 against 1.61e-3), and once the dilation takes h off it
        # that leftover is 6.5% against 2.2e-3.
        background = self.analytic_background(params)
        return {name: factor * _growth_factor_sq(background, of)(self.z)[None, :]
                for name, of in self.names.items()}

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
        return {name: (self.k, scale) for name in self.names}

    def section_options(self):
        return {**super().section_options(), 'k': self.k, 'z': self.z, 'of': self.of,
                'non_linear': self.non_linear, 'analytic': self.analytic, 'tilt': self.tilt,
                'dilate': self.dilate}

    def section_class(self, source, prefix=''):
        from cosmoprimo.cosmology import BaseSection, DefaultBackground

        emulator = self
        names = {_of_name(of): of for of in self.of}

        class Fourier(BaseSection):

            def __init__(self, engine):
                super().__init__(engine)
                self._engine = engine
                self._pk = source(engine)
                # sigma8 given as an input parameter, on an emulator trained in A_s (or the other way
                # round): cosmoprimo's own generic mechanism, which runs this section once with
                # _rsigma8 = 1, reads sigma8_m back off it and returns the ratio. Exact, because
                # the spectrum is exactly linear in the amplitude -- so a cosmology may be
                # written in whichever of the two the caller has, whatever the emulator was
                # trained in.
                self._rsigma8 = _rsigma8(engine)
                self._background, self._growth = None, {}
                # interpolators already built for this cosmology, see `pk_interpolator`
                self._interpolator = {}

            def pk_interpolator(self, of='delta_m', non_linear=False, **kwargs):
                r"""Emulated :math:`P(k, z)`, as the usual 2D interpolator.

                With ``analytic``, the array it is built on holds :math:`P / D^2` and the growth
                goes along as a callable for it to apply itself. That is what lets the fixed z
                grid be short: what the spline sees in z is only the scale-dependent residual
                (massive neutrinos, and what the analytic w0waCDM core does not capture), while
                the growth is put back at the z actually asked for. Measured, worst of five
                cosmologies: 3.7e-5 on 8 nodes this way against 4.8e-2 on 5 and 1.6e-5 on 20
                without.
                """
                from cosmoprimo.interpolator import PowerSpectrumInterpolator2D

                # One per spectrum and interpolation setting, held for the life of the engine --
                # which is the life of one cosmology, so it cannot go stale. A caller asks for the
                # same spectrum several times without meaning to (`pk_now_interpolator`,
                # `sigma8_z` and `sigma_rz` all go through here), and each rebuild is a cubic
                # spline over the whole k grid: on a GPU that is one tridiagonal solve per column,
                # which is what the profile of a full-shape likelihood was made of.
                key = (_of_name(of), bool(non_linear)) + tuple(sorted(kwargs.items()))
                if key in self._interpolator:
                    return self._interpolator[key]
                # the C1 cubic, not the C2 one order 3 means: on this grid -- dense in k, short
                # in z -- it is both cheaper (0.28 ms against 10.5 on a GPU, where the C2 solve is
                # a kernel launch per system) and more accurate against CLASS (7.7e-4 against
                # 1.4e-3). See `_interpax_method`.
                kwargs.setdefault('interp_order_k', 'cubic')
                kwargs.setdefault('interp_order_z', 'cubic')

                name = _of_name(of)
                if name not in names:
                    raise ValueError(f'{of!r} was not emulated; this one has '
                                     f'{sorted(names)}')
                if bool(non_linear) != emulator.non_linear:
                    raise ValueError(f'emulated with non_linear={emulator.non_linear}, '
                                     f'asked for {bool(non_linear)}')
                # A point outside the trained region is predicted as nan, deliberately: it maps to
                # -inf in a posterior, so a sampler rejects it instead of being handed an
                # extrapolation. Nothing is done about that here -- `Interpolator2D` keeps the nan
                # out of its solve and puts it back on the way out, which is where that belongs.
                pk = self._pk[f'{prefix}pk.{name}'] * self._rsigma8**2
                if not emulator.analytic:
                    self._interpolator[key] = PowerSpectrumInterpolator2D(
                        emulator.k, emulator.z, pk, **kwargs)
                    return self._interpolator[key]
                if name not in self._growth:
                    if self._background is None:
                        # off the engine itself rather than off a clone: this is the cosmology
                        # being served, and the same function the training divisor came from
                        self._background = DefaultBackground(self._engine)
                    self._growth[name] = _growth_factor_sq(self._background, names[name])
                growth_factor_sq = self._growth[name]
                # divided out here and handed back as the callable: the same array on the same
                # nodes, so the spectrum is returned exactly at a training redshift and
                # interpolated only in what is left between them
                self._interpolator[key] = PowerSpectrumInterpolator2D(
                    emulator.k, emulator.z, pk / growth_factor_sq(emulator.z)[None, :],
                    growth_factor_sq=growth_factor_sq, **kwargs)
                return self._interpolator[key]

            def pk_now_interpolator(self, of='delta_m', engine='peakaverage', cosmo='auto',
                                    cosmo_fid='auto', **kwargs):
                r"""The emulated spectrum with its BAO wiggles filtered out.

                Filtered rather than emulated: the no-wiggle spectrum is a functional of the
                spectrum this section already predicts, so running the filter on the prediction
                costs no node and cannot disagree with it, which a separately emulated
                :math:`P_\mathrm{nw}` is free to do. The filter object is built once and re-called
                (``filter(pk_interpolator)``), which is the form the peak-finding filters are
                jax-differentiable in: built fresh per call, ``peakaverage`` and ``hinton2017``
                locate their peak maxima with ``scipy.signal.find_peaks``, which a trace cannot
                follow.

                ``cosmo`` defaults to the cosmology this section belongs to, so the filter's
                :math:`r_\mathrm{drag}` rescaling is the emulated one -- which needs a
                thermodynamics section on the same emulator, and says so if there is none.
                Pass ``cosmo=None`` explicitly to switch that rescaling off.
                """
                from cosmoprimo import PowerSpectrumBAOFilter

                interpolator = self.pk_interpolator(of=of, **kwargs)
                if isinstance(cosmo, str) and cosmo == 'auto':
                    cosmo = getattr(self._engine, '_cosmology', None)
                if isinstance(cosmo_fid, str) and cosmo_fid == 'auto':
                    cosmo_fid = emulator.cosmo
                key = (_of_name(of), str(engine))
                cached = getattr(self, '_bao_filter', None)
                if cached is None:
                    cached = self._bao_filter = {}
                if key not in cached:
                    cached[key] = PowerSpectrumBAOFilter(interpolator, engine=engine,
                                                         cosmo=cosmo, cosmo_fid=cosmo_fid)
                else:
                    cached[key](interpolator, cosmo=cosmo)
                return cached[key].smooth_pk_interpolator()

            def pk_kz(self, k, z, of='delta_m', **kwargs):
                return self.pk_interpolator(of=of, **kwargs)(k, z)

            def sigma_rz(self, r, z, of='delta_m', **kwargs):
                return self.pk_interpolator(of=of, **kwargs).sigma_rz(r, z)

            def sigma8_z(self, z, of='delta_m'):
                r"""Emulated :math:`\sigma_8(z)`.

                Off the emulated spectrum rather than emulated on its own: the top-hat integral
                over the fixed k grid reproduces the engine's own :math:`\sigma_8` to 1e-7
                (measured over the same five cosmologies), so a leaf of its own would buy nothing
                and would be free to disagree with the spectrum it is supposed to summarise.
                """
                return self.sigma_rz(8., z, of=of)

            @property
            def sigma8_m(self):
                return self.sigma8_z(0., of='delta_m' if 'delta_m' in names else
                                     next(iter(names)))

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
    #: 2: the Eisenstein-Hu scales are computed at the node's own cosmology. Before, a training
    #: basis reached `clone`, which ignored it, so `h` was the fiducial's -- and the sound horizon
    #: it divides out carries one factor of `h`.
    version = 2

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

    def select_params(self, names):
        """Everything but the dark energy: these scalars are set before recombination, and what
        happens to the expansion long afterwards does not reach them.

        The angular scales are the exception in principle -- an angle divides by a distance,
        which the dark energy does move -- but ``analytic`` divides by the analytic background's
        own distance to the same epoch, so that part is removed rather than interpolated.
        """
        if not self.analytic:
            return list(names)
        return [name for name in names if name not in DARK_ENERGY]

    def scaling(self, params):
        if not self.analytic:
            return {}
        # a plain engine, for the reason `analytic_background` gives: the formulae below read
        # parameters only, and building the fiducial's own engine again would both waste a
        # Boltzmann setup and break the trace. `from_training` for the reason it gives too: the
        # sound horizon is multiplied by `h`, and in the theta basis `h` is not among the training
        # parameters -- `clone` would silently use the fiducial's.
        cosmo = self.clone(self.from_training(dict(params)), engine=BaseEngine)
        scales = _eisenstein_hu_scales(cosmo)
        # an angle is a sound horizon over a distance to the same epoch; the analytic background
        # supplies the distance, so the whole ratio is available without a Boltzmann call
        # `_guard` rather than a `float()` and a truth test: at prediction the parameters are
        # tracers, and both are things a tracer cannot do
        distance = _guard(self.analytic_background(params).comoving_radial_distance(
            scales['z_drag']))
        angle = scales['rs_drag'] / distance
        formula = {'rs_drag': scales['rs_drag'], 'z_drag': scales['z_drag'],
                   'rs_star': scales['rs_drag'], 'z_star': scales['z_drag'],
                   'theta_star': angle, 'theta_cosmomc': angle}
        return {name: _guard(formula[name])
                for name in self.of if name in formula}

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
                # never float(): these are read inside a jit -- a BAO likelihood divides its
                # distances by `rs_drag`, a CMB one reads `theta_cosmomc` -- and float() on a
                # tracer is a ConcretizationTypeError. A 0-d array behaves like a scalar
                # everywhere a float would, eagerly included.
                return self._values[f'{prefix}{name}']

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
    #: 3: the sections are handed the cosmology's own names, so the analytic divisor is built at
    #: the node rather than at the fiducial value of whatever a basis had replaced. A file written
    #: before that holds coefficients fitted against the wrong divisor, and this code multiplies
    #: the right one back: the two do not compose.
    version = 3

    def __init__(self, cosmo, space, sections, basis=None, **options):
        self.cosmo = cosmo
        # `basis` may be one basis for everything, or a dict keyed by section name for a section
        # that wants its own. The entry under `None` is where the nodes are laid out -- one node
        # set is shared whatever the sections are fitted in -- and defaults to the space's own
        # parameters, which is what a caller who names only some sections gets.
        given = isinstance(basis, dict)
        per_section = dict(basis) if given else {}
        unknown = [name for name in per_section if name is not None and name not in sections]
        if unknown:
            raise ValueError(f'basis given for {unknown}, which are not sections of this '
                             f'emulator ({sorted(sections)})')
        # `given`, not `if per_section`: a dict carrying only the `None` key is empty once the node
        # basis is taken out of it, and falling back then would hand each section the dict itself
        self.basis = _get_basis(per_section.pop(None, None) if given else basis)
        self.sections = {name: _SECTIONS[name](
                             cosmo, space, basis=per_section.get(name, None) if given else basis,
                             **dict(kwargs))
                         for name, kwargs in sections.items()}
        # and they share the analytic divisor: they clone the same fiducial with the same
        # parameters, so the background one of them solves is the background all of them want
        self._analytic_cache = {}
        for section in self.sections.values():
            section._analytic_cache = self._analytic_cache
        super().__init__(self.compute, space, **options)

    to_training = SectionEmulator.to_training
    from_training = SectionEmulator.from_training
    _theta_kwargs = SectionEmulator._theta_kwargs
    _basis_names = SectionEmulator._basis_names
    _REPLACED = SectionEmulator._REPLACED
    w0pwa = SectionEmulator.w0pwa
    transforms = SectionEmulator.transforms
    training_space = SectionEmulator.training_space

    def compute(self, params):
        cosmo = self.cosmo.clone(**dict(params))         # one Boltzmann call for every section
        values = {}
        for name, section in self.sections.items():
            values.update({f'{name}.{key}': value
                           for key, value in section.extract(cosmo).items()})
        return values

    def output_coordinates(self, name):
        r"""The coordinates the section behind output *name* is fitted in.

        ``None`` whenever that section's basis is the composite's, which is the ordinary case and
        costs nothing. When it differs, the section supplies its own space, its own parameters and
        its own map, and the tools layer fits that output there -- over the same nodes, since one
        Boltzmann call per node is what makes the extra sections cheap.

        What this buys is what desilike gets from one emulator per sector: :math:`C_\ell` are
        nearly stationary in :math:`\theta_\mathrm{MC}` and a translation along :math:`\ell` in
        :math:`h`, while a power spectrum wants :math:`h` itself -- so a cosmology serving both
        need not put either on the other's grid.
        """
        section = self.sections.get(name.split('.', 1)[0], None)
        if section is None or section.basis == self.basis:
            return None
        space = section.training_space()

        def to_training(params):
            # from the composite's coordinates, which is what the tools layer hands over, back to
            # the cosmology's own and into the section's
            return section.to_training(self.from_training(dict(params)))

        return space, section.select_params(list(space.params)), to_training

    def select_params(self, names):
        keep = set()
        for section in self.sections.values():
            keep |= set(section.select_params(names))
        return [name for name in names if name in keep]

    def _prefixed(self, params, what):
        """Each section's ``scaling`` or ``dilation``, under the names the composite stores.

        The basis is undone once, here: a section's own :meth:`from_training` knows only its own
        basis, and with a basis per section that is often none at all, so a section handed
        ``theta_MC_100`` or ``w0pwa`` would have cloned the fiducial with names it cannot read --
        which `clone` used to ignore, leaving `h` and `wa_fld` at the fiducial and the divisor
        wrong by 15% in ``efunc``. Sections speak the cosmology's own names.
        """
        params = self.from_training(dict(params))
        out = {}
        for name, section in self.sections.items():
            out.update({f'{name}.{key}': value
                        for key, value in getattr(section, what)(params).items()})
        return out

    def scaling(self, params):
        return self._prefixed(params, 'scaling')

    def dilation(self, params):
        """The sections' dilations, which a composite has to apply as a single section does.

        Not applying them was silent in the worst way: a `FourierEmulator` asked for
        ``dilate=True`` on its own reads its spectra back in the reference frame, and the same
        section inside a `CosmologyEmulator` did not -- the option was set, stored and reported,
        and did nothing. What that costs is the whole point of the dilation: at fixed physical
        densities `h` slides the acoustic peaks through the k grid, and measured on a full-shape
        box the residual then alternated in sign with the BAO period out to +-6.5% across the
        fitted range.
        """
        return self._prefixed(params, 'dilation')

    # the same pair a single section applies, which is what makes a composite of one section
    # and that section agree
    transform = SectionEmulator.transform
    inverse_transform = SectionEmulator.inverse_transform

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
        self._analytic_cache = {}
        for name, options in state['sections'].items():
            section = _SECTIONS[name].__new__(_SECTIONS[name])
            section.cosmo, section.space = self.cosmo, self.space
            # the same dict for every section, as in `__init__`
            section._analytic_cache = self._analytic_cache
            for key, value in options.items():
                setattr(section, key, value)
            self.sections[name] = section
        self.target = self.compute


# ── putting it back where the original was ────────────────────────────────────

def _check_fiducial(emulator, engine):
    """Refuse a cosmology that differs from the training fiducial in a parameter the emulator
    does not vary.

    Such a value is not interpolated, not extrapolated and not reported: it is ignored, and the
    prediction is the fiducial's. Measured while wiring this into desilike, an emulator trained
    from ``Cosmology()`` and asked for a DESI fiducial was 3% off in every spectrum, silently.

    Only the cosmological inputs are compared, not the calculation settings -- a caller
    legitimately raises ``ellmax_cl`` or turns ``lensing`` on, and none of that reaches a
    prediction. A tracer, or anything that is not a number, is skipped.
    """
    fiducial = emulator.cosmo._input_params
    cosmological = set(Cosmology.get_default_params(of='cosmology'))
    varied = set(emulator.space.params)
    # the amplitude is a conflict group: `sigma8` and `logA` are the same parameter differently
    # spelled, and `source` already handles a cosmology that names the other one
    if varied & set(_AMPLITUDES + _SIGMA8):
        varied |= set(_AMPLITUDES + _SIGMA8)
    differ = {}
    for name, value in engine._cosmology._input_params.items():
        if name in varied or name not in fiducial or name not in cosmological:
            continue
        try:
            given, trained = np.asarray(value, dtype='f8'), np.asarray(fiducial[name], dtype='f8')
        except (TypeError, ValueError):
            continue    # not a number: a hierarchy name, a species list, an engine setting
        if given.shape == trained.shape and not np.allclose(given, trained, rtol=1e-9, atol=0.,
                                                            equal_nan=True):
            differ[name] = (given.tolist(), trained.tolist())
    if differ:
        detail = ', '.join(f'{name}={given} against {trained}' for name, (given, trained) in differ.items())
        raise CoverageError(
            f'this cosmology differs from the one the emulator was trained on in {sorted(differ)}, '
            f'which it does not vary: {detail}. Those values would be ignored and the prediction '
            f'made at the training fiducial instead. Retrain from this cosmology, add the '
            f'parameters to the Space, or set them back to the fiducial values.')


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
        r"""The prediction for this engine's cosmology, computed once and shared by the sections.

        The parameters are read off the cosmology -- `predict` converts them to the training
        basis itself, and raises outside the trained box. The amplitude is the one that may not
        be there to read: it is a conflict group, so a cosmology written in :math:`\sigma_8` has
        no ``A_s`` and vice versa, and an emulator trained in either should still serve the
        other. What stands in is the engine's own first guess
        (:meth:`~cosmoprimo.cosmology.BaseEngine._get_A_s_fid`, good to a few per cent), and it
        does not have to be better than that: :func:`_rsigma8` rescales the prediction afterwards
        by the exact ratio, and the quantities it rescales are exactly linear in the amplitude.
        The same two-step cosmoprimo already uses to run CLASS or CAMB at a given
        :math:`\sigma_8`.
        """
        if getattr(engine, '_predicted', None) is None:
            _check_fiducial(emulator, engine)
            params = {}
            for name in emulator.space.params:
                try:
                    params[name] = engine[name]
                except Exception:
                    if name in _AMPLITUDES:
                        params[name] = Cosmology._get_params({'A_s': engine._get_A_s_fid()},
                                                             [name])[name]
                    elif name in _SIGMA8:
                        params[name] = engine._get_sigma8_fid()
                    else:
                        raise
            engine._predicted = emulator.predict(**params)
        return engine._predicted

    built = {name: section.section_class(source, prefix=f'{name}.' if composite else '')
             for name, section in sections.items()}

    class EmulatedEngine(BaseEngine):
        """Engine backed by a trained emulator."""
        name = 'emulated'
        # Declared jax-backed rather than inferred from the parameters, which is what
        # `BaseEngine._set_jax` does otherwise. Its predictions come out of jax engines whatever
        # the parameters look like, and the two disagree in a case that is not exotic: inside
        # someone else's jit, an operation on constant inputs is still staged out, so a
        # prediction made at concrete parameters is a tracer while the parameters are not. The
        # sections then mix the two -- a numpy-backed analytic background handed a traced
        # redshift by a jax-backed interpolator -- and numpy raises on the conversion.
        _use_jax = True
        _np = jnp

        def __init__(self, cosmo, **extra_params):
            super().__init__(cosmo, **extra_params)
            self._predicted = None
            # the cosmology this engine was attached to, which `BaseEngine` reads and discards.
            # Kept because a section may need to hand a cosmology to something that takes one --
            # the BAO filter's rs_drag rescaling, which should be this cosmology's own
            self._cosmology = cosmo
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
