"""Interpolation engines: fit a set of node evaluations, predict anywhere in the box.

The base geometry here -- the box, the transforms, the whitening -- is shared by every engine,
including the ones in :mod:`mlp` and :mod:`taylor`. What each engine adds is the node set it
wants and how it turns those values into a prediction.

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

:class:`~.taylor.TaylorEngine` does not interpolate, it expands about the
centre, and its knobs are ``order`` and ``accuracy`` rather than ``levels`` and ``budget``.
"""

import itertools

import numpy as np

from cosmoprimo.jax import numpy as jnp, numpy_jax

from .utils import (chebyshev_values, chebyshev_lobatto_nodes, chebyshev_vandermonde_inverse,
                    nested_level_nodes, smolyak_combination, TRANSFORMS)


ENGINES = {}


def engine_from_state(state):
    """Rebuild whichever engine wrote this state.

    The registry only holds the engines whose module has been imported, and reading a saved
    emulator in a fresh process imports none of them, so the ones that live elsewhere are pulled
    in here rather than being reported as unknown.
    """
    from . import mlp, taylor    # noqa: F401 -- registers MLPEngine and TaylorEngine

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
        {name: (low, high)} in physical units.
    transform : dict, default=None
        {name: transform name}, e.g. ``'sqrt'``.
    mean, covariance : array, default=None
        When given, the nodes are laid out on the covariance's principal axes.
    nsigma : float, default=3.
        Half-width of the whitened box.
    """
    name = None

    def __init__(self, params, limits, transform=None, mean=None, covariance=None, nsigma=3.,
                 **unused):
        self.params = list(params)
        self.limits = {name: tuple(float(value) for value in limits[name]) for name in self.params}
        self.nsigma = float(nsigma)
        self.transforms = {name: (transform or {}).get(name) for name in self.params}
        self.mean = None if mean is None else np.asarray(mean, dtype='f8')
        self._rotation = self._scale = None
        if covariance is not None:
            eigenvalues, self._rotation = np.linalg.eigh(np.atleast_2d(np.asarray(covariance, dtype='f8')))
            if np.any(eigenvalues <= 0.):
                raise ValueError('covariance is not positive definite; cannot whiten')
            self._scale = np.sqrt(eigenvalues)
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
        if self.whitened:
            return -self.nsigma, self.nsigma
        low, high = self.limits[name]
        forward = TRANSFORMS[self.transforms[name]][0] if self.transforms[name] else None
        return (float(forward(low)), float(forward(high))) if forward else (low, high)

    def _internal(self, values):
        """Physical parameters -> the coordinates the interpolant works in."""
        if self.whitened:
            return self.whiten(values)
        return np.array([float(TRANSFORMS[self.transforms[name]][0](value))
                         if self.transforms[name] else float(value)
                         for value, name in zip(np.asarray(values), self.params)])

    def _physical(self, values):
        """Internal coordinates -> physical parameters: the inverse of :meth:`_internal`."""
        if self.whitened:
            return self.unwhiten(values)
        return np.array([float(TRANSFORMS[self.transforms[name]][1](value))
                         if self.transforms[name] else float(value)
                         for value, name in zip(np.asarray(values), self.params)])

    def _traced(self, values):
        """Physical parameters -> internal coordinates, safe inside a jax trace.

        The eager :meth:`_internal` casts to float, which a tracer refuses; this is the same map
        written with the dispatching numpy so ``predict`` works under ``jit``. Both must agree,
        or the fit and the evaluation are in different coordinates.
        """
        xnp = numpy_jax(values)
        values = xnp.asarray(values)
        if self.whitened:
            return (xnp.asarray(self._rotation).T @ (values - xnp.asarray(self.mean))) \
                / xnp.asarray(self._scale)
        return xnp.stack([TRANSFORMS[self.transforms[name]][0](values[index])
                          if self.transforms[name] else values[index]
                          for index, name in enumerate(self.params)])

    def _geometry_state(self):
        return {'name': self.name, 'params': list(self.params), 'limits': dict(self.limits),
                'transforms': dict(self.transforms), 'nsigma': self.nsigma,
                'mean': self.mean, 'rotation': self._rotation, 'scale': self._scale}

    def _set_geometry(self, state):
        self.params = list(state['params'])
        self.limits = {name: tuple(value) for name, value in state['limits'].items()}
        self.transforms, self.nsigma = dict(state['transforms']), float(state['nsigma'])
        self.mean, self._rotation, self._scale = state['mean'], state['rotation'], state['scale']


class ChebyshevEngine(BaseEngine):
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

    def __init__(self, params, limits, levels=None, budget=None, **kwargs):
        super().__init__(params, limits, **kwargs)
        self.levels = {name: int((levels or {}).get(name, 2)) for name in self.params}
        self.budget = budget
        self.powers = self.coefficients = None

    def _grids(self):
        """(levels, weight, per-axis node arrays) for each sub-grid of the combination."""
        domains = {name: self._domain(name) for name in self.params}
        if self.budget is None:
            return [(None, 1, [chebyshev_lobatto_nodes(self.levels[name] + 1, limits=domains[name])
                               for name in self.params])], domains
        combination = smolyak_combination([self.levels[name] for name in self.params],
                                          int(self.budget))
        return [(levels, weight, [nested_level_nodes(level, limits=domains[name])
                                  for name, level in zip(self.params, levels)])
                for levels, weight in combination.items()], domains

    def nodes(self):
        """The node set, in physical parameters: an ``(n_nodes, n_params)`` array."""
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
        if self.whitened:
            return np.array([self.unwhiten(row) for row in rows])
        return np.array([[float(TRANSFORMS[self.transforms[name]][1](value))
                          if self.transforms[name] else value
                          for value, name in zip(row, self.params)] for row in rows])

    # ── fit / predict ─────────────────────────────────────────────────────────
    def fit(self, inputs, outputs):
        """``inputs``: (n_nodes, n_params), physical. ``outputs``: (n_nodes, n_outputs)."""
        grids, domains = self._grids()
        inputs = np.asarray(inputs, dtype='f8')
        outputs = np.asarray(outputs, dtype='f8')
        table = {tuple(round(float(value), 12) for value in self._internal(row)): out
                 for row, out in zip(inputs, outputs)}
        sparse = {}
        for _, weight, axes in grids:
            shape = tuple(len(axis) for axis in axes)
            values = np.empty(shape + (outputs.shape[1],))
            for index in itertools.product(*[range(size) for size in shape]):
                key = tuple(round(float(axes[axis][position]), 12)
                            for axis, position in enumerate(index))
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

    def predict(self, values):
        """``values``: physical parameters, in :attr:`params` order."""
        if self.coefficients is None:
            raise ValueError('not fitted')
        xnp = numpy_jax(values)
        values = self._traced(values)
        factors = []
        for index in range(len(self.params)):
            low, high = self.domains[index]
            scaled = (2. * values[index] - low - high) / (high - low)
            cheb = chebyshev_values(scaled, int(self.powers[:, index].max()))
            factors.append(cheb[self.powers[:, index]])
        return jnp.tensordot(self.coefficients, jnp.prod(jnp.stack(factors), axis=0), axes=(0, 0))

    def contract(self, matrix):
        """Left-multiply the output by a fixed ``matrix``, exactly, by contracting coefficients.

        The interpolant is linear in its coefficients, so for any fixed ``M``

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

    # ── state ──────────────────────────────────────────────────────────────────
    def __getstate__(self):
        state = self._geometry_state()
        state.update({'levels': dict(self.levels), 'budget': self.budget,
                      'powers': self.powers, 'coefficients': self.coefficients,
                      'domains': getattr(self, 'domains', None)})
        return state

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new._set_geometry(state)
        new.levels, new.budget = dict(state['levels']), state['budget']
        new.powers, new.coefficients = state['powers'], state['coefficients']
        new.domains = state['domains']    # set by fit(), and predict() reads it
        return new
