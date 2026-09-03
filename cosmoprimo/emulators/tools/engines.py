"""Interpolation engines: fit a set of node evaluations, predict anywhere in the box.

The base geometry here -- the box, the transforms, the whitening -- is shared by every engine,
including the ones in :mod:`mlp`, :mod:`polynomial` and :mod:`taylor`. What each engine adds is
the node set it wants and how it turns those values into a prediction. Of those,
:class:`ChebyshevEngine` and :class:`~.polynomial.PolynomialEngine` also share their *output*
form -- ``coefficients @ phi(x)`` over a tensor-product basis -- which is what
:class:`LinearBasisEngine` holds.

The Chebyshev engine is a sparse-grid (Smolyak) interpolant on Chebyshev-Lobatto nodes: exact on
polynomials, near-minimax over the box, and one coefficient contraction to evaluate. Two
properties of the node set matter in practice:

- the levels are nested, so raising ``budget`` later reuses every evaluation already made;
- the per-axis ``level`` and the total ``budget`` are different knobs. The level sets one axis's
  own error -- raising one axis from 2 to 3 cut its error 276x for 4 extra nodes -- while the
  budget buys only interaction terms, and left every per-axis number unchanged.

Whitening is internal. Given the posterior's mean and covariance the grid is laid out on its
principal axes rather than in a rectangle around them (measured 350x in the median at equal node
count, the largest single lever). The engine's parameter names stay physical: whitened
coordinates are not cosmology parameters, and exposing them breaks anything downstream that looks
parameters up by name.

Fitting that box inside a hard bound is decided here rather than by the space it came from: a
:class:`~.space.Space` says where the emulator must be accurate, and how to tile that region is
the engine's business. Two mutually exclusive treatments, ``shrink_to_limits`` and ``unrotated``,
and choosing between them needs to know how the nodes are laid out -- which only the engine does.

:class:`~.taylor.TaylorEngine` does not interpolate, it expands about the
centre, and its knobs are ``order`` and ``accuracy`` rather than ``levels`` and ``budget``.
:class:`~.polynomial.PolynomialEngine` does not interpolate either: it declares a small basis and
fits it by least squares to scattered points, which costs exactness at the nodes and buys a node
set that need not be complete -- reach for it when part of the box is a region the calculator
refuses.
"""

import itertools
import logging

import numpy as np

from cosmoprimo.jax import numpy as jnp, numpy_jax

from .utils import (chebyshev_lobatto_nodes, chebyshev_vandermonde_inverse,
                    nested_level_nodes, smolyak_combination, tensor_basis, TRANSFORMS)


ENGINES = {}


def engine_from_state(state):
    """Rebuild whichever engine wrote this state.

    The registry only holds the engines whose module has been imported, and reading a saved
    emulator in a fresh process imports none of them, so the ones that live elsewhere are pulled
    in here rather than being reported as unknown.
    """
    from . import mlp, polynomial, taylor    # noqa: F401 -- registers the engines living elsewhere

    name = state.get('name', 'chebyshev')
    if name not in ENGINES:
        raise ValueError(f'unknown engine {name!r} in the saved state; have {sorted(ENGINES)}')
    return ENGINES[name].from_state(state)


class BaseEngine(object):
    """The geometry every engine shares: the box, the transforms, and the whitening.

    Parameters
    ----------
    params : list
        Parameter names, ordering the columns of the node array.
    limits : dict
        {name: (low, high)}, in the expansion variable -- the same coordinate as ``mean`` and
        ``covariance``, not the user's parameter, wherever ``transform`` declares the two differ.
        :meth:`~.space.Space.geometry` supplies all three together and already in that variable.
    transform : dict, default=None
        {name: transform name}, e.g. ``'sqrt'``. Says which parameters the expansion variable
        differs for, so :meth:`nodes` can hand the calculator its own parameter back.
    mean, covariance : array, default=None
        When given, the nodes are laid out on the covariance's principal axes.
    nsigma : float, default=3.
        Half-width of the whitened box.
    unrotated : list, default=None
        Parameters held out of that rotation, so each keeps its own physical axis. Two
        consequences. Its ``limits`` entry then places nodes -- the only case where it does, since
        a rotated axis spans ``+- nsigma`` and reads no limit at all -- so a hard bound on it
        becomes a face of the box rather than a half-space cutting obliquely across the
        parallelepiped, and a one-sided or asymmetric bound (``w0 + wa < 0``) is expressible where
        ``mean +- nsigma sigma`` cannot express it. And the parameters left behind rotate onto the
        principal axes of their own sub-block, so naming one here changes the geometry whether or
        not it is bounded: the box becomes a product region, strictly larger than the ellipsoid --
        no loss of coverage, only of efficiency, as some nodes go to corners the posterior does
        not occupy.

        Policy rather than geometry, which is why it is a flag and not inferred from ``limits``:
        ``shrink_to_limits`` also respects a bound, and the covariance cannot say which trade is
        wanted. Measured on a CMB-only w0waCDM box with ``w0 + wa < 0``, shrinking pulls nsigma
        3.75 -> 2.771 in all eight directions, where naming that one parameter here leaves the
        other seven at 3.75.
    bounds : dict, default=None
        The subset of ``limits`` that are hard bounds -- what the calculator or the analysis
        refuses, not where a region happens to reach. Only these narrow the box or cap an
        ``unrotated`` axis; :meth:`_shrink_to_limits` is where the distinction is spelt out, and
        where the cost of losing it is measured. ``None`` reads every limit as a bound, which is
        what an engine built by hand, or a state written before the two were told apart, means.
        :meth:`~.space.Space.geometry` supplies it.
    shrink_to_limits : bool, default=True
        Narrow the box until it lies inside ``bounds``, for those parameters not in ``unrotated``.
        The other treatment of a hard bound, and the two are alternatives -- see
        :meth:`_shrink_to_limits`.
    """
    name = None

    #: Whether a planned node is allowed to have no value. Never, for an interpolant: its
    #: coefficients come from inverting a Vandermonde over the full grid, so a hole is not a
    #: smaller problem, it is an unsolvable one -- and a NaN mixes into every coefficient. A
    #: regression engine (see :class:`~.mlp.MLPEngine`) sets this False: dropping a sample from a
    #: least-squares fit costs that sample and nothing else, which is what lets it cover a box
    #: whose corners the calculator cannot evaluate.
    requires_all_nodes = True

    #: Whether this engine can use the samples the Space was measured from, over and above the
    #: mean and covariance :meth:`~.space.Space.geometry` hands every engine. Only a scattered
    #: node set can: a grid's nodes are determined by the box. :meth:`~.emulate.Emulator._engine`
    #: reads this rather than passing a chain to engines that would ignore it.
    wants_samples = False

    #: Margin, in units of that axis' own sigma, kept between an unrotated axis and a hard
    #: bound. Chebyshev node sets include their endpoints, so a bound used as-is places a node
    #: on the bound itself -- and a bound like ``w0 + wa < 0`` is strict.
    bound_margin = 1e-6

    def __init__(self, params, limits, transform=None, mean=None, covariance=None, nsigma=3.,
                 unrotated=None, shrink_to_limits=True, bounds=None, **unused):
        self.params = list(params)
        self.limits = {name: tuple(float(value) for value in limits[name]) for name in self.params}
        # `None` means "every limit is a bound": that is what a hand-built engine says, and what a
        # saved state written before the distinction existed meant.
        bounds = self.limits if bounds is None else bounds
        self.bounds = {name: tuple(float(value) for value in bounds[name])
                       for name in self.params if name in bounds}
        self.nsigma = float(nsigma)
        self.transforms = {name: (transform or {}).get(name) for name in self.params}
        self.mean = None if mean is None else np.asarray(mean, dtype='f8')
        self.unrotated = [name for name in self.params if name in set(unrotated or ())]
        self._rotation = self._scale = None
        if covariance is not None:
            covariance = np.atleast_2d(np.asarray(covariance, dtype='f8'))
            # A parameter named in `unrotated` is held out of the rotation: its row and column
            # are dropped and it keeps its own physical axis, so a hard bound on it is a face of
            # the box rather than a half-space cutting obliquely across it. That is the
            # difference between capping one axis and shrinking every one: with `w0 + wa < 0` on a
            # CMB-only w0waCDM box, `shrink_to_limits` pulls nsigma 3.75 -> 2.771 in every one of
            # eight directions (measured sigma 0.21 -> 0.50), where capping the one axis leaves the
            # other seven at 3.75.
            #
            # What is given up is that parameter's correlation with the rest, so the box becomes a
            # product region -- strictly larger than the ellipsoid, hence no loss of coverage,
            # only of efficiency: some nodes go to corners the posterior does not occupy.
            held = {self.params.index(name) for name in self.unrotated}
            rotated = [index for index in range(len(self.params)) if index not in held]
            rotation = np.eye(len(self.params))
            scale = np.sqrt(np.diag(covariance))
            if rotated:
                block = covariance[np.ix_(rotated, rotated)]
                eigenvalues, eigenvectors = np.linalg.eigh(block)
                if np.any(eigenvalues <= 0.):
                    raise ValueError('covariance is not positive definite; cannot whiten')
                rotation[np.ix_(rotated, rotated)] = eigenvectors
                scale[rotated] = np.sqrt(eigenvalues)
            if np.any(scale <= 0.):
                raise ValueError('covariance is not positive definite; cannot whiten')
            self._rotation, self._scale = rotation, scale
            condition = float(self._scale.max() / self._scale.min())
            if condition > 1e8:
                # a direction the space cannot actually move in: whitening it divides by nothing,
                # and the round trip through the whitened coordinates stops being exact. Left
                # alone this surfaces much later as an unfindable node.
                raise ValueError(
                    f'the covariance is effectively singular (condition number {condition:.1e}): '
                    f'some combination of {list(self.params)} is a deterministic function of the '
                    f'others, so there is no volume to interpolate over. Drop the redundant '
                    f'parameter, or give a Space that varies it independently.')
            if shrink_to_limits:
                self._shrink_to_limits(covariance)

    def _shrink_to_limits(self, covariance):
        """Reduce :attr:`nsigma` until the whitened box lies inside :attr:`limits`.

        The box is what the nodes fill, and it exceeds the marginal limits at its corners by
        design: a limit encoding a hard bound would otherwise be recorded and then ignored, since
        node placement reads nsigma and not limits. Measured, 113 of 817 nodes past ``w0 + wa = 0``
        with the bound set.

        The box reaches ``mean_i +- nsigma sum_j |A_ij|`` with ``A = rotation @ diag(scale)``, so
        the largest admissible nsigma is a ratio rather than a search. Conservative twice over: it
        bounds the full tensor box, of which a Smolyak grid is a subset, and it shrinks every axis
        by the same factor.

        Two things it must not do. It must not bind on anything but a hard bound, which is why it
        reads :attr:`bounds` and not :attr:`limits`: every parameter carries a limit, most of them
        derived -- ``mean +- nsigma sigma``, or the bounding box
        :meth:`~.space.Space.map` measures on the image of a chain -- and since ``reach`` exceeds
        ``sigma`` whenever the axes are correlated, taking those at face value shrinks a box that
        was never constrained. A geometric test alone does not catch it: a derived limit falling a
        little short of ``mean +- nsigma sigma``, which is what a finite sample does, is *tighter*
        and so passes. Measured on the CMB w0waCDM box, that read the chain's rail at ``w0 = -2``
        -- 1.57 sigma from the mean -- as a bound on the training region and pulled nsigma
        3.75 -> 1.27 in all eight directions, leaving the emulator able to answer on 15.8% of the
        chain it was built from. The geometric test is kept on top, for a bound the caller
        declared that is looser than the marginal extent and so constrains nothing.

        And it must skip :attr:`unrotated` axes, whose bound is already a face of the box (see
        :meth:`_domain`). Shrinking for those is the whole cost the flag exists to avoid: nsigma
        3.75 -> 2.771 in every direction, to enforce something one axis was enforcing alone.
        """
        sigma = np.sqrt(np.diag(covariance))
        reach = (np.abs(self._rotation) * self._scale).sum(axis=1)
        factor = 1.
        for index, name in enumerate(self.params):
            if name in self.unrotated or reach[index] <= 0. or name not in self.bounds:
                continue
            low, high = self.bounds[name]
            for room in (self.mean[index] - low, high - self.mean[index]):
                # `tightened`: a limit sitting at the marginal extent is the default one, and the
                # tolerance is there because it arrives via a float round trip, not exact.
                tightened = np.isfinite(room) and room < self.nsigma * sigma[index] * (1. - 1e-9)
                if tightened:
                    factor = min(factor, float(room) / (self.nsigma * reach[index]))
        self.nsigma = self.nsigma * max(factor, 0.)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name is not None:
            ENGINES[cls.name] = cls

    # ── geometry ──────────────────────────────────────────────────────────────
    @property
    def whitened(self):
        return self._rotation is not None

    def condition_number(self):
        return float(self._scale.max() / self._scale.min()) if self.whitened else 1.

    def whiten(self, values):
        return (self._rotation.T @ (np.asarray(values, dtype='f8') - self.mean)) / self._scale

    def unwhiten(self, values):
        return self.mean + self._rotation @ (np.asarray(values, dtype='f8') * self._scale)

    def _domain(self, name):
        """The interval this axis's nodes fill, in the coordinates the interpolant works in.

        Every geometric input -- :attr:`limits` as much as :attr:`mean` and the covariance -- is in
        the expansion variable, so nothing is transformed here. A :class:`~.space.Space` maps the
        limits it was given at construction, alongside the samples it measures mean and covariance
        from, which is what keeps the box and the nodes in one variable rather than two.
        """
        if self.whitened:
            if name in self.unrotated:
                # held out of the rotation, so this axis is the parameter itself: take the
                # stricter of its limits and the usual +- nsigma sigma, asymmetrically if that is
                # what the limits say. A one-sided bound then costs width on that side only.
                index = self.params.index(name)
                # `bounds`, for the reason `_shrink_to_limits` reads it: a derived extent is not
                # something to cut this axis at.
                low, high = self.bounds.get(name, (-np.inf, np.inf))
                centre, scale = float(self.mean[index]), float(self._scale[index])
                lower, upper = -self.nsigma, self.nsigma
                if np.isfinite(low) and (low - centre) / scale > lower:
                    lower = (low - centre) / scale + self.bound_margin
                if np.isfinite(high) and (high - centre) / scale < upper:
                    upper = (high - centre) / scale - self.bound_margin
                if not upper > lower:
                    raise ValueError(
                        f'the limits on {name} leave no room inside the box: '
                        f'[{low}, {high}] against a centre {centre} and sigma {scale}')
                return float(lower), float(upper)
            return -self.nsigma, self.nsigma
        return self.limits[name]

    def _transform_pair(self, name):
        """``(forward, inverse)`` for *name*, or None. Accepts a registry key (``'sqrt'``) or a
        pair of callables, which is what a parameterised transform needs -- a logit carries its
        interval, and a name-keyed registry cannot."""
        spec = self.transforms[name]
        if spec is None:
            return None
        return TRANSFORMS[spec] if isinstance(spec, str) else tuple(spec)

    def _internal(self, values):
        """Physical parameters -> the coordinates the interpolant works in.

        Transform first and only then whiten: the two compose rather than exclude each other.
        Whitening rotates onto the principal axes of the covariance, which must therefore be that
        of the transformed variables -- so a Space that declares transforms must supply mean and
        covariance measured in them. The two together are what a hard bound wants: the transform
        makes it unreachable, the rotation keeps the box on the posterior's axes.
        """
        values = np.asarray(values, dtype='f8')
        pairs = [self._transform_pair(name) for name in self.params]
        if any(pair is not None for pair in pairs):
            values = np.array([float(pair[0](value)) if pair else float(value)
                               for value, pair in zip(values, pairs)])
        return self.whiten(values) if self.whitened else values

    def _physical(self, values):
        """Internal coordinates -> physical parameters: the inverse of :meth:`_internal`."""
        values = self.unwhiten(values) if self.whitened else np.asarray(values, dtype='f8')
        pairs = [self._transform_pair(name) for name in self.params]
        if any(pair is not None for pair in pairs):
            values = np.array([float(pair[1](value)) if pair else float(value)
                               for value, pair in zip(values, pairs)])
        return values

    def _traced(self, values):
        """Physical parameters -> internal coordinates, safe inside a jax trace.

        The eager :meth:`_internal` casts to float, which a tracer refuses; this is the same map
        written with the dispatching numpy so ``predict`` works under ``jit``. Both must agree,
        or the fit and the evaluation are in different coordinates.
        """
        xnp = numpy_jax(values)
        values = xnp.asarray(values)
        pairs = [self._transform_pair(name) for name in self.params]
        if any(pair is not None for pair in pairs):
            values = xnp.stack([pair[0](values[index]) if pair else values[index]
                                for index, pair in enumerate(pairs)])
        if self.whitened:
            return (xnp.asarray(self._rotation).T @ (values - xnp.asarray(self.mean))) \
                / xnp.asarray(self._scale)
        return values

    #: Fraction of an axis' own width tolerated on each side of the fitted domain, so a point
    #: sitting exactly on a node -- Chebyshev-Lobatto sets include their endpoints -- is not
    #: rejected by the round trip through the whitening.
    domain_atol = 1e-9

    def outside(self, values):
        """Boolean mask: is this point outside the region the fit is defined over?

        A different question from :attr:`limits`, which is what makes it worth asking. ``limits``
        is one low/high pair per parameter -- a rectangle with its sides along the axes -- and it
        is all :meth:`Emulator._check` can test. But when the nodes are whitened they do not fill
        that rectangle: they fill a band lying along the directions the parameters vary together,
        and the rectangle's off-diagonal corners hold no node at all.
        Measured on an (h, omega_cdm) pair at correlation -0.95, 70.6% of the rectangle falls off
        the band, its worst corner at 6.1 times the band's half-width, with each parameter well
        inside its own range. A polynomial interpolant asked there does not degrade gracefully;
        it answers confidently from coefficients nothing constrained.

        Here rather than on :class:`~.space.Space` or :class:`~.emulate.Emulator` because this
        class already owns every piece the answer needs -- the transforms, the whitening rotation
        and :meth:`_domain`, including the asymmetric domain an unrotated axis gets from a
        one-sided bound. A Space stores a covariance but never builds the rotation, so asking it
        would mean reimplementing :meth:`_traced`, and a second implementation of the map that
        placed the nodes is exactly the thing that drifts.

        Mechanism, not policy: this says where the fit is defined, and nothing here refuses. It is
        :meth:`Emulator.outside` that enforces it, under the user's ``coverage`` switch, so an
        engine used directly stays a plain interpolant. A subclass whose fit is defined somewhere
        other than the per-axis domain -- a regression that may legitimately be trusted a little
        beyond it -- overrides this.

        Elementwise, so a batched/vmapped call marks only the offending members, and traceable, so
        the answer survives a ``jit``.

        Parameters
        ----------
        values : array
            Physical parameters, stacked in :attr:`params` order -- as :meth:`predict` takes them.

        Returns
        -------
        mask : array, bool
        """
        internal = self._traced(values)
        mask = None
        for index, name in enumerate(self.params):
            low, high = self._domain(name)
            margin = self.domain_atol * (high - low)
            this = (internal[index] < low - margin) | (internal[index] > high + margin)
            mask = this if mask is None else (mask | this)
        return mask

    def _geometry_state(self):
        return {'name': self.name, 'params': list(self.params), 'limits': dict(self.limits),
                'bounds': dict(self.bounds),
                'transforms': dict(self.transforms), 'nsigma': self.nsigma,
                'unrotated': list(self.unrotated),
                'mean': self.mean, 'rotation': self._rotation, 'scale': self._scale}

    def _set_geometry(self, state):
        self.params = list(state['params'])
        self.limits = {name: tuple(value) for name, value in state['limits'].items()}
        # `.get`: a state written before hard bounds were told apart from derived ones read every
        # limit as a bound, and `nsigma` in that same state was already shrunk accordingly.
        self.bounds = {name: tuple(value)
                       for name, value in state.get('bounds', state['limits']).items()}
        self.transforms, self.nsigma = dict(state['transforms']), float(state['nsigma'])
        # Needed by :meth:`_domain`, which is how :meth:`outside` gets its box back when the
        # engine did not save one (only Chebyshev stores `domains`). Left off, the attribute is
        # missing on a reloaded engine and `_domain` raises AttributeError -- which `outside`
        # catches, so the coverage check turns itself off silently rather than failing.
        # `.get`, because a state written before this was saved has none.
        self.unrotated = list(state.get('unrotated', ()))
        self.mean, self._rotation, self._scale = state['mean'], state['rotation'], state['scale']


class LinearBasisEngine(BaseEngine):
    """An engine whose prediction is ``coefficients @ phi(x)`` over a fixed polynomial basis.

    Two engines are of this form and only differ in how they get the coefficients:
    :class:`ChebyshevEngine` inverts a Vandermonde over a collocation grid,
    :class:`~.polynomial.PolynomialEngine` solves a least-squares problem over scattered points.
    Everything downstream of the coefficients is therefore shared, and shared rather than copied
    because :meth:`predict` and a fit's design matrix must evaluate the *same* basis: two
    implementations of it drift, and the symptom -- a fit that reproduces its own nodes and
    nothing else -- points nowhere near the cause.

    Subclasses set :attr:`powers` ``(nterms, nparams)``, :attr:`coefficients`
    ``(nterms, noutputs)`` and :attr:`domains` ``(nparams, 2)`` when they fit.
    """
    #: Which orthogonal family, a key of :data:`~.utils.BASES`. A class attribute where the
    #: family is fixed, an instance one where the engine lets it be chosen.
    basis = 'chebyshev'

    def predict(self, values):
        """``values``: physical parameters, in :attr:`params` order."""
        if self.coefficients is None:
            raise ValueError('not fitted')
        basis = tensor_basis(self._traced(values), self.powers, self.domains, basis=self.basis)
        return jnp.tensordot(self.coefficients, basis, axes=(0, 0))

    def contract(self, matrix):
        """Left-multiply the output by a fixed ``matrix``, exactly, by contracting coefficients.

        The prediction is linear in its coefficients, so for any fixed ``M``

            ``M @ predict(x) == (engine with coefficients M @ C).predict(x)``

        identically. Applying ``M`` here rather than downstream therefore costs nothing at
        evaluation time and, when ``M`` reduces the output size, shrinks the emulator with it.
        """
        if self.coefficients is None:
            raise ValueError('not fitted')
        matrix = np.asarray(matrix, dtype='f8')
        if matrix.shape[1] != self.coefficients.shape[1]:
            raise ValueError(f'matrix is {matrix.shape}, cannot act on an output of '
                             f'{self.coefficients.shape[1]}')
        self.coefficients = self.coefficients @ matrix.T
        return self


class ChebyshevEngine(LinearBasisEngine):
    """Sparse-grid Chebyshev interpolation.

    Parameters
    ----------
    levels : dict, default=None
        {name: level}; level ``l`` gives ``2^l + 1`` nested nodes (degree ``2^l``).
    budget : int, default=None
        Smolyak total-level budget; ``None`` gives the full tensor grid.

    See :class:`BaseEngine` for the geometry arguments.
    """
    name = 'chebyshev'
    logger = logging.getLogger('ChebyshevEngine')

    def __init__(self, params, limits, levels=None, budget=None, **kwargs):
        super().__init__(params, limits, **kwargs)
        self.levels = {name: int((levels or {}).get(name, 2)) for name in self.params}
        self.budget = budget
        self.powers = self.coefficients = self.domains = None
        self._node_map, self._n_missing = {}, 0

    def _grids(self, budget=-1):
        """(levels, weight, per-axis node arrays) for each sub-grid of the combination.

        *budget* defaults to this engine's own; a least-squares fit asks for a smaller one, to
        get its index set without the node set changing under it.
        """
        budget = self.budget if budget == -1 else budget
        domains = {name: self._domain(name) for name in self.params}
        if budget is None:
            return [(None, 1, [chebyshev_lobatto_nodes(self.levels[name] + 1, limits=domains[name])
                               for name in self.params])], domains
        combination = smolyak_combination([self.levels[name] for name in self.params],
                                          int(budget))
        return [(levels, weight, [nested_level_nodes(level, limits=domains[name])
                                  for name, level in zip(self.params, levels)])
                for levels, weight in combination.items()], domains

    def nodes(self):
        """The node set, in physical parameters: an ``(nnodes, nparams)`` array."""
        grids, _ = self._grids()
        seen, rows = set(), []
        for _, _, axes in grids:
            for point in itertools.product(*axes):
                key = tuple(round(float(value), 12) for value in point)
                if key in seen:
                    continue
                seen.add(key)
                rows.append([float(value) for value in point])
        rows = np.array(rows)
        # `_physical`, not `unwhiten` and not the inverse transform on its own: the two compose
        # (transform, then whiten), so both have to be undone, and only `_physical` knows about a
        # transform given as a pair of callables rather than a registry name -- which is what a
        # parameterised one has to be, since a logit carries its interval. Undoing the whitening
        # alone hands the calculator the expansion variable instead of the parameter: measured,
        # 226 of 817 nodes then landed past a bound the transform exists to make unreachable.
        physical = np.array([self._physical(row) for row in rows])
        if self.whitened:
            # Remember the whitened coordinates these very nodes came from. `fit` needs the
            # grid key for each evaluated node, and recovering it by whitening the physical value
            # back is not exact: measured on a condition number of 8354, 2 of 817 nodes came back
            # 1.0e-12 away, which a 12-decimal round turns into 'missing evaluation at node ...'
            # after the whole training has been paid for. The generator knows the exact key, so
            # it keeps it.
            self._node_map = {self._round(row): self._round(whitened)
                              for row, whitened in zip(physical, rows)}
        return physical

    @staticmethod
    def _round(row):
        return tuple(round(float(value), 12) for value in np.asarray(row))

    def _node_key(self, row):
        """The grid key for one evaluated node: looked up, not recomputed.

        Falls back to whitening the value when the node is not one this engine generated (a
        caller fitting its own points), which is the old behaviour and exact when unwhitened.
        """
        if self.whitened:
            if not self._node_map:
                self.nodes()          # deterministic and free: geometry only, no evaluations
            found = self._node_map.get(self._round(row))
            if found is not None:
                return found
        return self._round(self._internal(row))

    # ── fit / predict ─────────────────────────────────────────────────────────
    def _fit_lstsq(self, inputs, outputs, grids, domains, rcond=None):
        """Coefficients from a least-squares solve, for a node set with holes in it.

        Interpolation cannot survive a missing node -- the square system loses a row and there
        is no freedom anywhere to absorb it. A least-squares fit over the same basis can, as
        long as enough nodes remain to determine the coefficients. The cost is that the result
        no longer passes exactly through the nodes; the gain is that a box whose corners the
        calculator cannot evaluate becomes usable at all.
        """
        # The sparse index set, which needs no node value at all -- the same combination the
        # interpolation accumulates over, collected as indices alone, so the design matrix can be
        # built before anything has been solved for.
        indices = set()
        for _, _, axes in grids:
            indices.update(itertools.product(*[range(len(axis)) for axis in axes]))
        powers = np.array(sorted(indices), dtype='i4')

        # Phi[i, k] = prod_d T_{powers[k, d]}(x_i, d): the interpolant is linear in its
        # coefficients, which is the whole reason a least-squares fit is available at all.
        internal = np.array([self._node_key(row) for row in np.asarray(inputs, dtype='f8')])
        box = np.array([domains[name] for name in self.params])
        design = np.asarray(tensor_basis(internal.T, powers, box)).T   # (nnodes, nterms)

        nnodes, nterms = design.shape
        coefficients, _, rank, singular = np.linalg.lstsq(design, outputs, rcond=rcond)
        condition = float(singular.max() / singular.min()) if singular.min() > 0 else np.inf
        # A Smolyak grid is unisolvent -- as many nodes as terms -- so losing even one node
        # leaves the system rank-deficient, and a minimum-norm solve would then invent the
        # unconstrained directions rather than measure them. The way to absorb a hole is to fit a
        # smaller basis to the same nodes, which is what `basis_budget` buys: the redundancy comes
        # from giving up polynomial degree, not from wishing the missing node away.
        deficiency = nterms - int(rank)
        if deficiency:
            raise ValueError(
                f'the usable nodes span only {rank} of {nterms} basis directions '
                f'({deficiency / nterms:.1%} unconstrained). Lower `basis_budget` so the basis '
                f'is smaller than the node set, or recover the missing nodes; fitting this many '
                f'free directions would invent structure rather than measure it.')
        self.logger.info(f'least-squares fit on {nnodes}/{nnodes + self._n_missing} nodes, '
                         f'{nterms} terms, condition number {condition:.1f}')
        self.powers = powers
        self.coefficients = coefficients
        self.domains = np.array([domains[name] for name in self.params])
        return self

    def fit(self, inputs, outputs, method='auto', rcond=None, basis_budget=None):
        """``inputs``: (nnodes, nparams), physical. ``outputs``: (nnodes, noutputs).

        ``method='auto'`` interpolates when every planned node is present -- exact, and the
        cheap tensor algebra -- and falls back to a least-squares fit when some are missing.
        ``'interpolate'`` and ``'lstsq'`` force one or the other.
        """
        grids, domains = self._grids()
        inputs = np.asarray(inputs, dtype='f8')
        outputs = np.asarray(outputs, dtype='f8')
        table = {self._node_key(row): out for row, out in zip(inputs, outputs)}
        if method not in ('auto', 'interpolate', 'lstsq'):
            raise ValueError(f"method must be 'auto', 'interpolate' or 'lstsq'; got {method!r}")
        if method != 'interpolate':
            planned = set()
            for _, _, axes in grids:
                for index in itertools.product(*[range(len(axis)) for axis in axes]):
                    planned.add(self._round([axis[position]
                                             for axis, position in zip(axes, index)]))
            self._n_missing = len(planned - set(table))
            if method == 'lstsq' or self._n_missing:
                # A Smolyak grid carries exactly as many nodes as basis terms, so a fit over its
                # own basis is square and one hole already makes it underdetermined. A smaller
                # basis over the same nodes is what creates the room to absorb one.
                basis = grids if basis_budget is None else self._grids(int(basis_budget))[0]
                return self._fit_lstsq(inputs, outputs, basis, domains, rcond=rcond)
        sparse = {}
        for _, weight, axes in grids:
            shape = tuple(len(axis) for axis in axes)
            values = np.empty(shape + (outputs.shape[1],))
            for index in itertools.product(*[range(size) for size in shape]):
                key = self._round([axis[position] for axis, position in zip(axes, index)])
                if key not in table:
                    raise ValueError(f'missing evaluation at node {key}; the node set must be the '
                                     f'one this engine produced')
                values[index] = table[key]
            for axis, name in enumerate(self.params):
                inverse = chebyshev_vandermonde_inverse(axes[axis], limits=domains[name])
                values = np.moveaxis(np.tensordot(inverse, values, axes=([1], [axis])), 0, axis)
            for index in itertools.product(*[range(size) for size in shape]):
                sparse[index] = sparse.get(index, 0.) + weight * values[index]
        powers = sorted(sparse)
        self.powers = np.array(powers, dtype='i4')
        self.coefficients = np.array([sparse[index] for index in powers])
        self.domains = np.array([domains[name] for name in self.params])
        return self

    # ── state ──────────────────────────────────────────────────────────────────
    def __getstate__(self):
        state = self._geometry_state()
        state.update({'levels': dict(self.levels), 'budget': self.budget,
                      'powers': self.powers, 'coefficients': self.coefficients,
                      'domains': self.domains})
        return state

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new._set_geometry(state)
        new.levels, new.budget = dict(state['levels']), state['budget']
        new.powers, new.coefficients = state['powers'], state['coefficients']
        new.domains = state['domains']    # set by fit(), and predict() reads it
        new._node_map, new._n_missing = {}, 0
        return new
