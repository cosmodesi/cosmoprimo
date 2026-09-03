"""A Taylor expansion engine: derivatives at one point, from a finite-difference stencil.

Not the sparse grid in :mod:`engines`, though the two used to share a name. Chebyshev
*interpolates* -- nodes spread over the whole box, error near-minimax across it. This is
*local*: derivatives at one centre, exact there and degrading away from it, accuracy bought by
raising the order. Measured on a two-parameter spectrum at 25 nodes, max relative error over 400
draws of the box, median / 90th: chebyshev 4.8e-3 / 2.1e-2, taylor 7.8e-3 / 1.3e-1. The tail is
the story -- the expansion's corners are where it has no claim -- but inside the middle 60% of
that box it is the more accurate of the two. So: reach for it when the derivatives are
themselves the output (a Fisher forecast, a response coefficient), when the region is genuinely
small, or to reproduce an analysis built on one; otherwise take the grid.

Two knobs, and they are not the grid's::

    emu.train(engine='taylor', order=3, accuracy=2)

``order`` is the highest total degree kept -- the truncation, and the only thing setting the
truncation error. ``accuracy`` is how well each derivative is *estimated*, and is a floor: a
stencil is widened past it when a narrower one could not return the derivative it claims (see
:func:`axis_stencils`). Both take a dict with an optional ``'*'`` default. The widest
stencil spans the box, as in the previous implementation, so the step is not a separate knob;
the box, transforms and whitening come from :class:`BaseEngine`.
"""
import itertools
import math

import numpy as np

from cosmoprimo.jax import numpy_jax

from .engines import BaseEngine
from .utils import fd_stencil, expand_dict


def axis_stencils(order, accuracy, exact_through, half_width):
    """Every finite-difference stencil one axis needs, and the step they share.

    Returns ``(step, [(offsets, weights)])``, indexed by derivative order from 0 to ``order``.
    The weights already carry the ``1 / step**k``, so a product of them across axes is the mixed
    partial itself, and only the points that actually carry weight appear.

    Two demands set a stencil's half-width, and the wider wins. The first is the requested
    ``accuracy``: ``(k + accuracy - 1) // 2`` points either side, error of order ``h**accuracy``.
    The second is exactness, and it is the one that is easy to miss: a ``2n+1``-point stencil
    returns the true derivative only for polynomials of degree ``<= 2n``, so a centred first
    difference is *not* the first derivative of a cubic -- it carries a term ``h**2`` times the
    third derivative. Sized on ``accuracy`` alone, an order-3 expansion would get its linear
    coefficient wrong by an amount the cubic term put there, and would fail to reproduce even
    the polynomial it is long enough to hold. So every stencil is widened until it is exact
    through ``exact_through``, the highest degree the expansion keeps along this axis. That
    costs no nodes: the widest stencil is the highest-order one either way, and the widened
    low-order stencils reuse points the node set already contains.

    ``half_width`` is half the box along this axis, and fixes the step: the widest stencil
    reaches the edge exactly, which is why the step is not a knob of its own.
    """
    unit = (np.array([0]), np.array([1.]))
    if not order:
        return 1., [unit]

    def nside(k):
        return max((k + accuracy - 1) // 2, -(-exact_through // 2))

    step = half_width / nside(order)
    stencils = [unit]
    for k in range(1, order + 1):
        # `fd_stencil` sizes itself as `(k + accuracy - 1) // 2`; invert that for the width above
        offsets, coefficients = fd_stencil(k, 2 * nside(k) - k + 1)
        stencils.append((offsets, coefficients / step ** k))
    return step, stencils


class TaylorEngine(BaseEngine):
    """Multivariate Taylor expansion about the centre of the box.

    Parameters
    ----------
    order : int, dict, default=3
        Highest derivative order per parameter; a dict may carry a ``'*'`` default. The expansion
        keeps every mixed term whose total degree is within :attr:`budget` and whose degree in
        each parameter is within that parameter's ``order``.
    accuracy : int, dict, default=2
        Finite-difference accuracy per parameter -- a positive even integer, the order of the
        error of the derivative *estimate*, and a floor: a stencil is widened past it when
        exactness demands it. Not the truncation order: see the module docstring.
    budget : int, default=None
        Cap on the total degree of the mixed terms, ``max(order)`` by default (the full Taylor
        polynomial of that degree). Lowering it drops cross terms -- and the nodes that only
        those terms needed -- while leaving every pure-derivative term alone.
    levels : dict, default=None
        Accepted and ignored: the sparse grid's knob, passed through by the emulator so that
        swapping engines needs no other change.

    Unlike the grid's, these nodes are **not nested**: the widest stencil spans the box, so
    raising ``order`` past the next even step rescales the spacing and the previous evaluations
    are no longer on it. Pick the order before paying for the training, rather than expecting to
    top it up. (Lowering ``budget`` is safe -- it only drops terms and the nodes only they
    needed.)

    See :class:`BaseEngine` for the geometry arguments.
    """
    name = 'taylor'

    def __init__(self, params, limits, order=3, accuracy=2, budget=None, levels=None, **kwargs):
        super().__init__(params, limits, **kwargs)
        self.order = {name: int(value) for name, value
                      in expand_dict(order, self.params, 'order').items()}
        self.accuracy = {name: int(value) for name, value
                         in expand_dict(accuracy, self.params, 'accuracy').items()}
        for name in self.params:
            if self.order[name] < 0:
                raise ValueError(f'order is {self.order[name]} < 0 for {name!r}')
            if not self.order[name]:
                continue
            if self.accuracy[name] <= 0 or self.accuracy[name] % 2:
                raise ValueError(f'accuracy is {self.accuracy[name]} for {name!r}, and must be '
                                 f'a positive even integer')
        if not max(self.order.values()):
            raise ValueError(f'every parameter is at order 0, so the expansion is a constant; '
                             f'give order >= 1 for at least one of {list(self.params)}')
        self.budget = None if budget is None else int(budget)
        self._setup()
        self.derivatives = None

    def _setup(self):
        """Everything the box, ``order``, ``accuracy`` and ``budget`` fix between them.

        Rebuilt on load rather than saved, so a restored engine and a fresh one cannot end up
        disagreeing about the terms they keep or the stencil they were fitted on.
        """
        highest = max(self.order.values())
        #: Highest total degree the expansion actually keeps -- neither ``order`` nor ``budget``
        #: on its own, since either can be the binding one.
        self.cap = highest if self.budget is None else min(self.budget, highest)
        #: The kept terms: degree ``p_i <= order_i`` per axis, total degree within :attr:`cap`.
        self.powers = np.array(
            [power for power in
             itertools.product(*[range(self.order[name] + 1) for name in self.params])
             if sum(power) <= self.cap], dtype='i4')
        center, steps, self.stencils = [], [], []
        for name in self.params:
            low, high = self._domain(name)
            center.append(0.5 * (low + high))
            step, stencils = axis_stencils(self.order[name], self.accuracy[name],
                                           min(self.order[name], self.cap), 0.5 * (high - low))
            steps.append(step)
            self.stencils.append(stencils)
        self.center, self.steps = np.array(center), np.array(steps)

    # ── nodes ─────────────────────────────────────────────────────────────────
    def nodes(self):
        """The node set, in physical parameters: an ``(nnodes, nparams)`` array.

        The union over kept terms of the points each one's own stencil needs, so a term dropped
        by ``budget`` takes with it any node no other term asked for.
        """
        rows, seen = [], set()
        for power in self.powers:
            axes = [self.center[index] + self.steps[index] * self.stencils[index][order][0]
                    for index, order in enumerate(power)]
            for point in itertools.product(*axes):
                key = tuple(round(float(value), 12) for value in point)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(self._physical(np.array(point)))
        return np.array(rows)

    # ── fit / predict ─────────────────────────────────────────────────────────
    def fit(self, inputs, outputs):
        """``inputs``: (nnodes, nparams), physical. ``outputs``: (nnodes, noutputs).

        Each mixed partial is the tensor product of one-dimensional stencils -- the difference
        operators along different axes commute, so an order-``p`` mixed derivative is a weighted
        sum over the product of their points, with no recursion needed. The Taylor coefficient is
        that derivative over ``prod(p_i!)``.
        """
        inputs = np.asarray(inputs, dtype='f8')
        outputs = np.asarray(outputs, dtype='f8')
        if len(inputs) != len(outputs):
            raise ValueError(f'{len(inputs)} inputs against {len(outputs)} outputs')
        table = {tuple(round(float(value), 12) for value in self._internal(row)): output
                 for row, output in zip(inputs, outputs)}

        derivatives = []
        for power in self.powers:
            stencils = [self.stencils[index][order] for index, order in enumerate(power)]
            derivative = 0.
            for point in itertools.product(*[zip(offsets, weights)
                                             for offsets, weights in stencils]):
                key = tuple(round(float(self.center[index] + self.steps[index] * offset), 12)
                            for index, (offset, _) in enumerate(point))
                if key not in table:
                    raise ValueError(f'missing evaluation at node {key}; the node set must be the '
                                     f'one this engine produced')
                weight = float(np.prod([weight for _, weight in point]))
                derivative = derivative + weight * table[key]
            derivatives.append(derivative / np.prod([math.factorial(order) for order in power]))
        self.derivatives = np.array(derivatives)
        return self

    def predict(self, values):
        """``values``: physical parameters, in :attr:`params` order."""
        if self.derivatives is None:
            raise ValueError('not fitted')
        xnp = numpy_jax(values)
        values = self._traced(values)
        powers = xnp.asarray(self.powers)
        diffs = values - xnp.asarray(self.center)
        # `where` rather than a bare power: an axis at the centre gives 0**0, whose derivative is
        # a NaN that would propagate through a jitted likelihood's gradient
        terms = xnp.prod(xnp.where(powers > 0, diffs ** powers, 1.), axis=-1)
        return xnp.tensordot(xnp.asarray(self.derivatives), terms, axes=(0, 0))

    def contract(self, matrix):
        """Left-multiply the output by a fixed ``matrix``, exactly, by contracting derivatives.

        The expansion is linear in its derivatives, so ``M @ predict(x)`` is the expansion whose
        derivatives are ``M @ D`` -- exactly, and at no evaluation-time cost. See
        :meth:`ChebyshevEngine.contract`.
        """
        if self.derivatives is None:
            raise ValueError('not fitted')
        matrix = np.asarray(matrix, dtype='f8')
        if matrix.shape[1] != self.derivatives.shape[1]:
            raise ValueError(f'matrix is {matrix.shape}, cannot act on an output of '
                             f'{self.derivatives.shape[1]}')
        self.derivatives = self.derivatives @ matrix.T
        return self

    # ── state ──────────────────────────────────────────────────────────────────
    def __getstate__(self):
        state = self._geometry_state()
        state.update({'order': dict(self.order), 'accuracy': dict(self.accuracy),
                      'budget': self.budget, 'derivatives': self.derivatives})
        return state

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new._set_geometry(state)
        new.order = {name: int(value) for name, value in state['order'].items()}
        new.accuracy = {name: int(value) for name, value in state['accuracy'].items()}
        new.budget = state['budget']
        new._setup()
        new.derivatives = state['derivatives']
        return new
