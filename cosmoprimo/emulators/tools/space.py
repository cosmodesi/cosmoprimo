"""Where an emulator must be accurate -- not what it is trained on.

The distinction matters because the single largest lever on emulator accuracy is the region the
interpolant has to cover, and a product of per-parameter ranges is a poor description of a
posterior with degeneracies. Measured on an 8-parameter CMB emulator at fixed node count, with a
posterior whose largest off-diagonal correlation is 0.896:

    coordinates                     median dchi2   90th percentile
    product of marginal ranges         0.0352          0.728
    hand-built theta* coordinate       0.0086          0.0618
    principal axes of the covariance   0.0001          0.0002

A rectangle around a thin ellipsoid is mostly empty space that the chain never visits, and the
interpolant spends its resolution there. So prefer, in order::

    Space(samples=chain)                                   # mean, covariance and true support
    Space(mean=best_fit, covariance=fisher, params=names)  # a Fisher matrix
    Space(bounds={'omega_cdm': (0.10, 0.14)})              # plain ranges, the weakest form

They combine: ``Space(samples=chain, bounds={'tau_reio': (0.01, 0.1)})`` keeps the chain's
correlations and hard-bounds one parameter.

Three words for ranges, and they are not interchangeable. :attr:`Space.bounds` is what the
analysis refuses to leave and only ever tightens; a *measured reach* only ever widens;
:attr:`Space.limits` is the box the two produce. Only a bound is a constraint an engine may shrink
a box to fit inside -- see :meth:`~.engines.BaseEngine._shrink_to_limits` for what happens when a
measured range is mistaken for a declared one. The constructor takes only ``bounds``; the other
two directions are the verbs :meth:`Space.bound` and :meth:`Space.widen`, which name which is
which at the call site instead of leaving it to argument order.

Two vocabularies, deliberately kept apart:

* ``forward`` / ``inverse`` change **coordinates**, on parameters -- :meth:`Space.forward`,
  :meth:`Space.inverse`, the ``(forward, inverse)`` pairs in :data:`~.utils.TRANSFORMS`, and
  :meth:`~.engines.BaseEngine._transform_pair`. A ``transform`` spec names such a pair.
* ``transform`` / ``inverse_transform`` precondition **values**, on emulator outputs --
  :meth:`~.emulate.Emulator.transform`. Different objects, different direction, and applying one
  where the other belongs produces numbers rather than errors.
"""

import numpy as np

from .utils import TRANSFORMS


def _forward(spec):
    """The forward callable of a transform spec, or ``None``.

    A key into :data:`~.utils.TRANSFORMS` (``'sqrt'``), or a ``(forward, inverse)`` pair of
    callables, which is what a parameterised transform needs -- a logit carries its interval, and
    a name-keyed registry cannot express that.
    """
    if spec is None:
        return None
    return TRANSFORMS[spec][0] if isinstance(spec, str) else spec[0]


def _ranges(given, transforms=None):
    """``{name: (low, high)}`` as floats, each mapped through its transform if one is declared."""
    ranges = {}
    for name, value in (given or {}).items():
        forward = _forward((transforms or {}).get(name))
        value = [float(bound) for bound in value]
        ranges[name] = tuple(sorted(float(forward(bound)) for bound in value)) if forward is not None \
            else tuple(value)
    return ranges


class Space(object):
    """The region an emulator must be accurate over, in the user's own parameters.

    Every input is an explicit keyword -- parameter names are not keywords, since a parameter
    called ``mean`` or ``samples`` would collide with the API.

    Parameters
    ----------
    samples : mapping, default=None
        {parameter: chain values}. Any mapping; an ``X.`` prefix on the keys is stripped, so a
        chain from a sampler works unchanged. The most informative form -- mean, covariance and
        the true support, so coverage can be checked against the points a chain actually visits
        rather than an assumed ``nsigma``.
    weights : array, default=None
        Multiplicities for ``samples``, if it is a weighted chain. Taken from the chain itself
        when not given -- ``samples.weights`` (getdist) or ``samples.weight`` (desilike) -- since
        dropping them silently describes a different distribution, one that over-samples the
        tails. Everything measured from the samples uses them: :attr:`mean`, :attr:`covariance`,
        and :meth:`coverage`.
    mean, covariance : array, default=None
        Joint description; ``params`` then names them, in order.
    bounds : dict, default=None
        {parameter: (low, high)}, a hard bound -- what the calculator or the analysis refuses.
        Alone, the weakest form. With a covariance it only ever tightens the parameters it names,
        leaving the correlations intact, and it is the only thing an engine narrows its box to fit
        inside. Use :meth:`widen` for a range that merely describes where the region reaches:
        shrinking a box to fit inside one of those enforces nothing and costs every axis.
    nsigma : float, default=3.
        Half-width of the box, in sigma, wherever it comes from a covariance.
    levels : dict, default=None
        {parameter: level}, per-axis resolution. A different knob from the training budget: the
        level sets one axis's own error (raising one from 2 to 3 cut that axis 276x for 4 extra
        nodes), the budget buys only interaction terms.
    transforms : dict, default=None
        {parameter: transform}, e.g. ``'sqrt'`` for a neutrino mass. A key into
        :data:`~.utils.TRANSFORMS`, or a ``(forward, inverse)`` pair of callables where the map
        carries parameters of its own that a name cannot express -- a logit and its interval.

        Naming one makes it the *expansion variable* for that parameter, which changes what this
        whole object is measured in, not just where the nodes fall: see the Notes.

    Attributes
    ----------
    params : list
    limits : dict
        The box: ``mean +- nsigma sigma``, cut by ``bounds``. Adjusted afterwards by
        :meth:`bound` and :meth:`widen`, each of which returns a new Space.
    bounds : dict
        The hard bounds it was given, as given -- the subset of :attr:`limits` an engine may
        narrow a box to fit inside.
    samples : array
    weights : array
        ``None`` where the samples are unweighted, which every consumer reads as weight one.
    levels : dict
    transforms : dict
    mean : array

    Notes
    -----
    Every quantity here -- :attr:`limits`, :attr:`bounds`, :attr:`mean`, :attr:`covariance` and
    :attr:`samples` -- is in the expansion variable, and so are the points :meth:`draw` returns
    and the ones :meth:`contains` expects. Only :attr:`params` keeps the user's own names. Start
    from :meth:`forward` when you hold a value in the user's parameter and want to compare it with
    any of them; a comparison made in the wrong variable is silent, and wrong in both directions.

    It has to work this way round. Nodes are placed on the principal axes of :attr:`covariance`,
    so that covariance must describe the transformed variable -- and it can only be measured
    there, from transformed samples. Propagating a physical covariance through the map instead
    would be a linear approximation to a map chosen for being nonlinear: measured on
    ``sqrt(m_ncdm)`` over 0.02-0.40, a Jacobian puts the mean 13.1% of a sigma off and sigma
    itself 8.4% off, which tilts the very axes the whitening exists to find. This is also why
    ``transforms`` cannot simply move to the engine, which sees a covariance and never the samples
    behind it.

    Mapping happens once, on the way in, and nowhere else: ``__init__`` is the only method that
    maps a range from the user's parameter into the expansion variable. Every method that adjusts
    a region -- :meth:`marginal`, :meth:`bound`, :meth:`widen`, :meth:`uncorrelated`, :meth:`map`
    -- goes through :meth:`clone`, which copies this one's state
    rather than rebuilding through ``__init__``, so there is no path on which a transform is
    applied to an already transformed range.

    What this class does not decide is how the region is tiled. Whether a hard bound is honoured
    by narrowing the box (``shrink_to_limits``) or by holding that axis out of the whitening
    rotation (``unrotated``) is the engine's choice, and both live on
    :class:`~.engines.BaseEngine`. :attr:`nsigma` here is descriptive: it says how far out the
    region reaches, and is never reduced to make a box fit.
    """
    def __init__(self, samples=None, mean=None, covariance=None, bounds=None, params=None,
                 nsigma=3., levels=None, transforms=None, weights=None):
        self.nsigma = float(nsigma)
        transforms = dict(transforms or {})
        # `bounds` arrives in the user's own parameter, and everything this object holds is in the
        # expansion variable, so it is mapped here, before it meets mean +- nsigma sigma. This is
        # the only place a range is ever transformed, which is what lets every method below take
        # and return expansion-variable values without asking which system it is in.
        bounds = _ranges(bounds, transforms)

        # What is known about the region, from whichever description was given: `params`, `mean`,
        # the covariance, and the samples with their weights. The box itself is assembled from
        # these and the declared ranges, further down.
        self._covariance, self.samples, self.weights, self.mean = None, None, None, None
        if samples is not None:
            names = list(samples.columns('X.*')) if hasattr(samples, 'columns') else list(samples)
            self.params = [name[2:] if name.startswith('X.') else name for name in names]
            self.samples = np.column_stack([np.asarray(samples[name]) for name in names])
            # A chain's multiplicities are part of the posterior, not metadata: ignoring them
            # describes a different distribution, one that over-samples the tails -- measured on
            # a cobaya CMB chain, the mean of `h` moved 0.065 sigma and every sigma came out
            # wide. So they are picked up from the chain itself when the caller does not pass
            # them (getdist spells the attribute `weights`, desilike `weight`), because the
            # failure is silent in both directions and the object always knows.
            if weights is None:
                for attr in ('weights', 'weight'):
                    if getattr(samples, attr, None) is not None:
                        weights = getattr(samples, attr)
                        break
            if weights is not None:
                weights = np.ravel(np.asarray(weights, dtype='f8'))
                if len(weights) != len(self.samples):
                    raise ValueError(f'{len(weights)} weights for {len(self.samples)} samples')
                if np.any(weights < 0.) or not weights.sum() > 0.:
                    raise ValueError('weights must be non-negative and not all zero')
            self.weights = weights
            # A declared transform makes that parameter the expansion variable: the engine
            # composes transform-then-whiten, so mean and covariance must describe the
            # transformed samples, not the raw ones. Doing it here means a caller never
            # has to know -- passing raw samples and a transform is enough, and the two
            # cannot drift apart. (Getting this wrong is silent: the box is then in one
            # variable and the nodes in another.)
            for name, spec in transforms.items():
                if name in self.params:
                    index = self.params.index(name)
                    self.samples[:, index] = _forward(spec)(self.samples[:, index])
            self.mean = np.average(self.samples, axis=0, weights=self.weights)
            self._covariance = np.cov(self.samples, rowvar=False, aweights=self.weights)
        elif covariance is not None:
            if params is None:
                raise ValueError('`params` is required with `covariance` (it names and orders them)')
            self.params = list(params)
            self.mean = np.asarray(mean, dtype='f8')
            self._covariance = np.atleast_2d(np.asarray(covariance, dtype='f8'))
            if self._covariance.shape != (len(self.params),) * 2:
                raise ValueError(f'covariance is {self._covariance.shape}, expected '
                                 f'{(len(self.params),) * 2} for {len(self.params)} parameters')
        elif bounds:
            self.params = list(bounds)
        else:
            raise ValueError('provide samples=, (mean=, covariance=, params=), or bounds=')

        for what, given in (('bounds', bounds),
                            ('levels', levels or {}), ('transforms', transforms)):
            unknown = [name for name in given if name not in self.params]
            if unknown:
                raise ValueError(f'{what} names unknown parameters {unknown}; '
                                 f'space has {self.params}')

        # The box: mean +- nsigma sigma, then cut by `bounds`. A bound has to cut -- w0 + wa < 0,
        # a positive mass -- and one merely looser than the box does not widen it; use
        # :meth:`widen` for that, which is the verb for "the region was measured to reach here".
        limits = {}
        if self._covariance is not None:
            sigma = np.sqrt(np.diag(self._covariance))
            limits = {name: (self.mean[index] - self.nsigma * sigma[index],
                             self.mean[index] + self.nsigma * sigma[index])
                      for index, name in enumerate(self.params)}
        for name, (low, high) in bounds.items():
            current = limits.get(name)
            limits[name] = (low, high) if current is None else \
                (max(current[0], low), min(current[1], high))
        for name, (low, high) in limits.items():
            if not high > low:
                raise ValueError(f'empty range for {name!r}: [{low}, {high}]')
        self.bounds, self.limits = bounds, limits

        if self.mean is None:  # no covariance to centre on: the box speaks for itself
            self.mean = np.array([sum(self.limits[name]) / 2. for name in self.params])
        self.levels = {name: int((levels or {}).get(name, 2)) for name in self.params}
        self.transforms = {name: transforms.get(name) for name in self.params}

    # ── description ────────────────────────────────────────────────────────────
    @property
    def covariance(self):
        """Joint covariance; diagonal from the limits when none was given."""
        if self._covariance is not None:
            return self._covariance
        return np.diag([((high - low) / 2. / self.nsigma)**2
                        for low, high in (self.limits[name] for name in self.params)])

    @property
    def correlation(self):
        sigma = np.sqrt(np.diag(self.covariance))
        return self.covariance / np.outer(sigma, sigma)

    @property
    def center(self):
        return {name: sum(self.limits[name]) / 2. for name in self.params}

    def is_correlated(self, threshold=0.1):
        """Whether whitening can buy anything: a diagonal covariance whitens to a pure rescaling,
        which changes nothing and only obscures the parameter names."""
        off = np.abs(self.correlation - np.eye(len(self.params)))
        return bool(off.max() > threshold)

    def geometry(self):
        """The box, as the keyword arguments an engine's constructor takes.

        Here rather than in :meth:`~.emulate.Emulator._engine` so that what the geometry *is*
        stays with the class that owns it, and an engine keeps taking plain arrays: an engine is
        serialised and rebuilt in a fresh process, and what it persists is the factorised
        whitening -- mean, rotation, scale -- not the covariance, nor the chain in :attr:`samples`
        that a Space may carry. Handing it a Space would only move the unpacking into
        ``__init__``, and split it from ``__getstate__``.

        The whitening keys are added only when :meth:`is_correlated`, so the nodes go on the
        posterior's principal axes instead of a rectangle around them: measured 350x in the median
        at equal node count, the largest single lever. It stays internal -- the engine's parameter
        names remain physical.

        Describes the region and nothing more. :attr:`nsigma` goes out as given, and it is the
        engine that decides how to fit a box inside :attr:`bounds` -- see
        :meth:`~.engines.BaseEngine._shrink_to_limits`.
        """
        geometry = dict(params=list(self.params), limits=dict(self.limits),
                        bounds=dict(self.bounds),
                        levels=dict(self.levels), transform=dict(self.transforms))
        if self.is_correlated():
            geometry.update(mean=self.mean, covariance=self.covariance, nsigma=self.nsigma)
        return geometry

    def marginal(self, names):
        """The space restricted to ``names``, marginalising over the rest.

        Used when a target handles some parameters exactly: what remains must be described by the
        marginal covariance -- the sub-block -- not the conditional one (a Schur complement),
        which describes the region at fixed values of the removed parameters and would shrink the
        box wrongly.

        A restriction, so every field is sliced out of this object rather than re-derived from a
        description of it. Rebuilding through ``__init__`` would hand it ranges that are already
        in the expansion variable and have them mapped a second time -- ``sqrt`` twice, ``logit``
        twice -- and the sub-block's diagonal is the parent's anyway, so there is nothing a
        rebuild would compute differently.
        """
        names = list(names)
        unknown = [name for name in names if name not in self.params]
        if unknown:
            raise ValueError(f'unknown parameters {unknown}; space has {self.params}')
        index = [self.params.index(name) for name in names]
        state = self.__getstate__()
        state.update(
            params=names,
            limits={name: self.limits[name] for name in names},
            bounds={name: self.bounds[name] for name in names if name in self.bounds},
            levels={name: self.levels[name] for name in names},
            transforms={name: self.transforms[name] for name in names},
            mean=self.mean[index],
            covariance=None if self._covariance is None else self._covariance[np.ix_(index, index)],
            samples=None if self.samples is None else self.samples[:, index])
        return Space.from_state(state)

    def clone(self, **fields):
        """A copy of this space with *fields* replaced, built from state rather than rebuilt.

        Every method that adjusts a region goes through here. Rebuilding through ``__init__``
        would hand it ranges that are already in the expansion variable and map them a second
        time -- ``sqrt`` twice, ``logit`` of a logit is nan -- which is the invariant the whole
        class rests on: ``__init__`` is the only place a range crosses between the user's
        parameter and the expansion variable.
        """
        state = self.__getstate__()
        state.update(fields)
        return Space.from_state(state)

    def bound(self, **ranges):
        """The same region, cut to *ranges*: a hard constraint the target must respect.

        Intersects the box and records the range in :attr:`bounds`, which is what the engine
        shrinks its grid against. The verb for ``w0 + wa < 0`` or a positive mass. Ranges are in
        the expansion variable; a transform declared for that parameter has already been applied
        to everything this object holds.
        """
        unknown = [name for name in ranges if name not in self.params]
        if unknown:
            raise ValueError(f'unknown parameters {unknown}; space has {self.params}')
        limits, bounds = dict(self.limits), dict(self.bounds)
        for name, value in ranges.items():
            low, high = sorted(float(bound) for bound in value)
            bounds[name] = (low, high)
            current = limits[name]
            limits[name] = (max(current[0], low), min(current[1], high))
            if not limits[name][1] > limits[name][0]:
                raise ValueError(f'empty range for {name!r} after bounding: {limits[name]}')
        return self.clone(limits=limits, bounds=bounds)

    def widen(self, **ranges):
        """The same region, widened to reach *ranges*: a measurement, not a constraint.

        Unions the box and leaves :attr:`bounds` alone -- the opposite of :meth:`bound`, and the
        distinction matters because the engine shrinks against bounds and not against a measured
        reach. Ranges are in the expansion variable.
        """
        unknown = [name for name in ranges if name not in self.params]
        if unknown:
            raise ValueError(f'unknown parameters {unknown}; space has {self.params}')
        limits = dict(self.limits)
        for name, value in ranges.items():
            low, high = sorted(float(bound) for bound in value)
            current = limits[name]
            limits[name] = (min(current[0], low), max(current[1], high))
        return self.clone(limits=limits)

    def uncorrelated(self):
        """The same box, with the covariance dropped and the samples kept as a candidate pool.

        :meth:`is_correlated` is then false, :meth:`geometry` emits no whitening keys, and the
        engine lays a plain axis-aligned grid -- while ``measure='samples'`` still has the points
        it draws candidates from.

        Wanted when the correlation is an artefact of a basis change rather than the posterior:
        ``Omega_b`` and ``Omega_cdm`` are both ``omega / h^2``, so their image is a narrow band
        (correlation 0.85) that says nothing about where the target is. A whitened grid follows
        that band and then refuses a point moving ``omega_cdm`` alone at fixed ``h`` as off the
        node cloud -- measured, ``omega_cdm = 0.1511`` inside a box reaching 0.1599, which cut the
        tail off a chain.
        """
        return self.clone(covariance=None)

    def map(self, mapping, size=100000, seed=42, levels=None, transforms=None):
        r"""The same region, expressed in other parameters.

        Used when the parameters a chain was run in are not the ones an emulator should expand.
        The mapping is done by transforming points, not by propagating a Jacobian: the change of
        variables that matters here -- :math:`\Omega_m \rightarrow \omega_{cdm}` -- mixes in
        :math:`h` and is not linear, so a Jacobian would be an approximation where this is exact.

        A space built from samples maps its actual chain points, which is why samples are worth
        keeping: the image of a posterior is described by the image of its draws, however curved
        the map.

        The box is set per name category rather than re-measured as ``mean +- nsigma`` of the
        image: a pass-through parameter keeps the source's, an introduced one takes the image's
        own bounding box over the points this space accepts. See the comment on that block below
        for what each rule is worth in practice. What the caller does need is that a point
        inside the source box lands inside the target box, or a perfectly valid prediction fails
        coverage, and that does not come for free: the image of an ellipsoid under a non-linear
        map is not an ellipsoid, and measured on a Planck-like posterior in
        ``(Omega_m, Omega_b, h)`` a draw well inside 3 sigma landed outside the mapped 3-sigma box
        in ``omega_cdm``. The bounding box is a superset of the curved image and so over-covers a
        little; the whitening still uses the image's own covariance, so the grid sits on its
        principal axes rather than in the corners.

        Parameters
        ----------
        mapping : callable
            ``mapping(params) -> params`` in the new names. May mix parameters and need not be
            invertible; a per-axis invertible change belongs in `transforms`, which the engine
            inverts per node.
        size : int, default=100000
            How many points to map, when the space has no samples of its own.
        """
        transforms, levels = dict(transforms or {}), dict(levels or {})
        drawn = ([dict(zip(self.params, row)) for row in self.samples]
                 if self.samples is not None else self.draw(size=size, seed=seed))
        weights = self.weights if self.samples is not None else None
        keep = [index for index, point in enumerate(drawn) if self.contains(point)]
        if not keep:
            raise ValueError('no point of this space is inside its own limits; nothing to map')
        points = [drawn[index] for index in keep]
        # the same rows, so a weighted chain stays the posterior it was through the map
        weights = None if weights is None else weights[keep]
        rows = [mapping(point) for point in points]
        names = list(rows[0])

        unknown = [name for name in list(transforms) + list(levels) if name not in names]
        if unknown:
            raise ValueError(f'{unknown} are not among the mapped parameters {names}')
        # A transform may only be declared for a parameter the mapping INTRODUCES. A pass-through
        # keeps the source's limits, which are already in the source's expansion variable;
        # re-declaring its transform would need them re-expressed, and applying one twice is the
        # "logit of a logit is nan" failure. Both real callers only ever transform names their
        # mapping introduces.
        redeclared = [name for name in transforms if name in self.params]
        if redeclared:
            raise ValueError(f'{redeclared} are pass-through parameters; a transform may only be '
                             f'declared for one the mapping introduces, or the source limits '
                             f'would have to be re-expressed')

        samples = {name: np.array([float(row[name]) for row in rows]) for name in names}
        # A declared transform asserts a domain, and a point outside it is not in the region at
        # all -- `logit_w0pwa` is undefined at w0 + wa >= 0, which a uniform draw from a rectangle
        # in (w0, wa) reaches. Dropping those is the same contract as `contains` above; keeping
        # them puts nan in the limits, which is how a single map that both renames and derives
        # used to fail.
        finite = np.ones(len(rows), dtype='?')
        for name, spec in transforms.items():
            forward = _forward(spec)
            with np.errstate(all='ignore'):
                finite &= np.isfinite(np.asarray(forward(samples[name]), dtype='f8'))
        if not finite.any():
            raise ValueError(f'every mapped point is outside the domain of {list(transforms)}')
        if not finite.all():
            samples = {name: values[finite] for name, values in samples.items()}
            weights = None if weights is None else weights[finite]

        # Only the NEW transforms are declared to the constructor, which applies them to the
        # samples. A pass-through's values arrived from `mapping` already in the source's
        # expansion variable -- `drawn` is `self.samples`, which is stored transformed -- so
        # declaring its transform here would apply it a second time, and the logit of a logit is
        # nan. Its declaration is recorded afterwards instead, where nothing re-applies it.
        inherited = {name: self.transforms[name] for name in names
                     if name in self.params and self.transforms.get(name) is not None}
        inherited_levels = {name: self.levels[name] for name in names if name in self.params}
        mapped = Space(samples=samples, weights=weights, nsigma=self.nsigma,
                       levels={**inherited_levels, **levels}, transforms=transforms)
        if inherited:
            mapped = mapped.clone(transforms={**mapped.transforms, **inherited})

        # The box, per name category. Neither category is `mean +- nsigma sigma` of the image.
        #
        # A PASS-THROUGH parameter keeps the source's limits: the image has nothing to measure
        # that the source did not already know, and re-measuring inflates a bounds-defined box by
        # 3/sqrt(12) ~ 1.7x on axes the mapping never touched -- measured, a `wa_fld` box of +-0.9
        # came back +-1.56 and training died on a node at w0 + wa = 0.56, non-finite in CLASS.
        #
        # An INTRODUCED one takes the image's own bounding box, which is what the region actually
        # is and contains every point the source box maps to. `mean +- nsigma sigma` of a skewed
        # image reaches outside it: `Omega_cdm = omega_cdm / h^2` over the ACE domain runs 0.10 to
        # 0.64, and three sigma about its mean reaches negative density -- a node at
        # `Omega_cdm = -0.053`, where CLASS returns non-finite and one node poisons every
        # coefficient.
        #
        # `bounds` as well as `limits` for an introduced name: when the space is correlated the
        # engine whitens and lays its grid on a rotated ellipsoid whose axis-aligned hull reaches
        # outside the per-axis limits, and `_shrink_to_limits` cuts against `bounds`. With
        # `limits` alone the box was unchanged.
        limits, bounds = {}, {}
        for name in mapped.params:
            if name in self.params:
                limits[name] = self.limits[name]
                if name in self.bounds:
                    bounds[name] = self.bounds[name]
            else:
                index = mapped.params.index(name)
                column = mapped.samples[:, index]   # already the expansion variable
                limits[name] = bounds[name] = (float(column.min()), float(column.max()))
        return mapped.clone(limits=limits, bounds=bounds)

    # ── use ────────────────────────────────────────────────────────────────────
    def draw(self, size=1, seed=42):
        """Draw from the space: the joint Gaussian when a covariance is known, else uniform.

        Uniform draws in a high-dimensional box sit overwhelmingly near its boundary, so they
        measure corners a chain never visits; prefer a covariance whenever there is one.
        """
        rng = np.random.default_rng(seed)
        if self._covariance is not None:
            values = rng.multivariate_normal(self.mean, self._covariance, size=size)
        else:
            low = np.array([self.limits[name][0] for name in self.params])
            high = np.array([self.limits[name][1] for name in self.params])
            values = rng.uniform(low, high, size=(size, len(self.params)))
        return [dict(zip(self.params, row)) for row in values]

    def forward(self, point):
        """A point in the user's own parameters, mapped into the expansion variable.

        Everything this class holds -- :attr:`limits`, :attr:`mean`, :attr:`covariance`,
        :attr:`samples` -- is in the expansion variable, because that is what a declared transform
        makes the interpolant work in, and mean and covariance have to describe the same variable
        the nodes are placed in. So a value arriving in the user's parameter has to be mapped
        before it can be compared with any of them.

        This is the whole boundary between the two coordinate systems, and it is worth calling
        rather than open-coding: a comparison made in the wrong one is silent and wrong in both
        directions -- it rejects points well inside a declared bound and accepts points outside it.
        Dispatches through the value, so it survives a jax trace.
        """
        mapped = {}
        for name, value in point.items():
            forward = _forward(self.transforms.get(name))
            mapped[name] = forward(value) if forward is not None else value
        return mapped

    def inverse(self, point):
        """A point in the expansion variable, mapped back to the user's own parameters.

        The other half of :meth:`forward`. Everything this class holds is in the expansion
        variable, so reporting a limit in the parameter the user actually named -- an error
        message, a prior published from a box -- has to come back through here. Callers used to
        reach into ``TRANSFORMS[spec][1]`` by hand for want of it.
        """
        mapped = {}
        for name, value in point.items():
            spec = self.transforms.get(name)
            inverse = None if spec is None else (TRANSFORMS[spec] if isinstance(spec, str) else spec)[1]
            mapped[name] = inverse(value) if inverse is not None else value
        return mapped

    def contains(self, point):
        """Is this point inside the box? Coverage is a contract: a point outside must be an
        error, never a silent clip.

        *point* is in the expansion variable, as :attr:`limits` and :attr:`samples` are; start from
        :meth:`forward` when you have the user's own parameters.
        """
        return all(self.limits[name][0] <= point[name] <= self.limits[name][1]
                   for name in self.params)

    # ── state ──────────────────────────────────────────────────────────────────
    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def __getstate__(self):
        return {'params': list(self.params), 'limits': dict(self.limits),
                'bounds': dict(self.bounds),
                'levels': dict(self.levels), 'transforms': dict(self.transforms),
                'nsigma': self.nsigma, 'mean': self.mean,
                'covariance': self._covariance, 'samples': self.samples,
                'weights': self.weights}

    def __setstate__(self, state):
        self.params = list(state['params'])
        self.limits = {name: tuple(value) for name, value in state['limits'].items()}
        # `.get`, for a state written before hard bounds were told apart from derived ones: every
        # limit was then treated as a bound, which is what reading them all back reproduces.
        self.bounds = {name: tuple(value)
                       for name, value in state.get('bounds', state['limits']).items()}
        self.levels, self.transforms = dict(state['levels']), dict(state['transforms'])
        self.nsigma, self.mean = float(state['nsigma']), state['mean']
        self._covariance, self.samples = state['covariance'], state['samples']
        # `.get`, for a state written before weights were carried: those samples were treated as
        # unweighted, which is what reading back `None` reproduces.
        self.weights = state.get('weights', None)

    def __repr__(self):
        kind = 'samples' if self.samples is not None else (
            'covariance' if self._covariance is not None else 'limits')
        return f'Space({len(self.params)} params from {kind}, correlated={self.is_correlated()})'
