import itertools

import numpy as np
from .jax import jit
from .jax import numpy as jnp
from .base import BaseEmulatorEngine
from . import mpi


def deriv_ncoeffs(order, acc=2):
    """Return number of coefficients given input derivative order and accuracy."""
    return 2 * ((order + 1) // 2) - 1 + acc


def coefficients(order, acc, coords, idx):
    """
    Calculate the finite difference coefficients for given derivative order and accuracy order.
    Assume that the underlying grid is non-uniform.

    Adapted from https://github.com/maroba/findiff/blob/master/findiff/coefs.py

    Parameters
    ----------
    order : int
        The derivative order (positive integer).

    acc : int
        The accuracy order (even positive integer).

    coords : np.ndarray
        The coordinates of the axis for the partial derivative.

    idx : int
        Index of the grid position where to calculate the coefficients.

    Returns
    -------
    coeffs, offsets
    """
    import math

    if acc % 2 or acc <= 0:
        raise ValueError('Accuracy order acc must be positive EVEN integer')

    if order < 0:
        raise ValueError('Derive degree must be positive integer')

    order, acc = int(order), int(acc)

    ncoeffs = deriv_ncoeffs(order, acc=acc)
    nside = ncoeffs // 2
    ncoeffs += (order % 2 == 0)

    def _build_rhs(offsets, order):
        """The right hand side of the equation system matrix"""
        b = [0 for _ in offsets]
        b[order] = math.factorial(order)
        return np.array(b, dtype='float')

    def _build_matrix_non_uniform(p, q, coords, k):
        """Constructs the equation matrix for the finite difference coefficients of non-uniform grids at location k"""
        A = [[1] * (p + q + 1)]
        for i in range(1, p + q + 1):
            line = [(coords[k + j] - coords[k])**i for j in range(-p, q + 1)]
            A.append(line)
        return np.array(A, dtype='float')

    if idx < nside:
        matrix = _build_matrix_non_uniform(0, ncoeffs - 1, coords, idx)

        offsets = list(range(ncoeffs))
        rhs = _build_rhs(offsets, order)

        return np.linalg.solve(matrix, rhs), np.array(offsets)

    if idx >= len(coords) - nside:
        matrix = _build_matrix_non_uniform(ncoeffs - 1, 0, coords, idx)

        offsets = list(range(-ncoeffs + 1, 1))
        rhs = _build_rhs(offsets, order)

        return np.linalg.solve(matrix, rhs), np.array(offsets)

    matrix = _build_matrix_non_uniform(nside, nside, coords, idx)

    offsets = list(range(-nside, nside + 1))
    rhs = _build_rhs(offsets, order)

    return np.linalg.solve(matrix, rhs), np.array([p for p in range(-nside, nside + 1)])


def deriv_nd(X, Y, orders, center=None, atol=0.):
    """
    Compute n-dimensional derivative.

    Parameters
    ----------
    X : array
        Array of shape (nsamples, ndim), with ndim the number of variables.

    Y : array
        Array of shape (nsamples, ysize), with ysize the size of the vector to derive.

    orders : list
        List of tuples (derivation axis between 0 and ndim - 1, derivative order, derivative accuracy).

    center : array, default=None
        The center around which to take derivatives, of size ndim.
        If ``None``, defaults to the median of input ``X``.

    atol : list, float
        Absolute tolerance to find the center.

    Returns
    -------
    deriv : array
        Derivative of Y, of size ysize.
    """
    uorders = []
    for axis, order, acc in orders:
        if not order: continue
        uorders.append((axis, order, acc))
    orders = uorders
    if center is None:
        center = [np.median(np.unique(xx)) for xx in X.T]
    if np.ndim(atol) == 0:
        atol = [atol] * X.shape[1]
    atol = list(atol)
    if not len(orders):
        toret = Y[np.all([np.isclose(xx, cc, rtol=0., atol=at) for xx, cc, at in zip(X.T, center, atol)], axis=0)]
        if not toret.size:
            raise ValueError('Global center point not found')
        return toret[0]
    axis, order, acc = orders[-1]
    ncoeffs = deriv_ncoeffs(order, acc=acc)
    coord = np.unique(X[..., axis])
    if coord.size < ncoeffs:
        raise ValueError('Grid is not large enough ({:d} < {:d}) to estimate {:d}-th order derivative'.format(coord.size, ncoeffs, order))
    cidx = np.flatnonzero(np.isclose(coord, center[axis], rtol=0., atol=atol[axis]))
    if not cidx.size:
        raise ValueError('Global center point not found')
    cidx = cidx[0]
    toret = 0.
    for coeff, offset in zip(*coefficients(order, acc, coord, cidx)):
        mask = X[..., axis] == coord[cidx + offset]
        ncenter = center.copy()
        ncenter[axis] = coord[cidx + offset]
        # We could fill in atol[axis] = 0., but it should be useless?
        y = deriv_nd(X[mask], Y[mask], orders[:-1], center=ncenter, atol=atol)
        toret += y * coeff
    return toret


def deriv_grid(grids, current_order=0):
    """
    Return grid of points where to compute function to estimate its derivatives.

    Parameters
    ----------
    grids : list
        List of tuples (1D grid coordinates, array of (minimum) derivative orders corresponding to 1D grid, derivative accuracy).

    Returns
    -------
    grid : list
        List of coordinates.
    """
    grid, orders, maxorder = grids[-1]
    toret = []
    for order in np.unique(orders)[::-1]:
        if order == 0 or order + current_order <= maxorder:
            mask = orders == order
            if len(grids) > 1:
                mgrid = deriv_grid(grids[:-1], current_order=order + current_order)
            else:
                mgrid = [[]]
            toret += [mg + [gg] for mg in mgrid for gg in grid[mask]]
    return toret


class TaylorEmulatorEngine(BaseEmulatorEngine):
    """
    Taylor expansion emulator engine, based on Stephen Chen and Mark Maus' velocileptors' Taylor expansion:
    https://github.com/cosmodesi/desi-y1-kp45/tree/main/ShapeFit_Velocileptors
    """
    name = 'taylor'

    def __init__(self, order=3, accuracy=2):
        self.sampler_options = dict(order=order, accuracy=accuracy)

    def get_default_samples(self, calculator, **kwargs):
        """
        Returns samples with derivatives.

        Parameters
        ----------
        order : int, dict, default=3
            A dictionary mapping parameter name (including wildcard) to maximum derivative order.
            If a single value is provided, applies to all varied parameters.

        accuracy : int, dict, default=2
            A dictionary mapping parameter name (including wildcard) to derivative accuracy (number of points used to estimate it).
            If a single value is provided, applies to all varied parameters.
            Not used if ``method = 'auto'``  for this parameter.

        delta_scale : float, default=1.
            Parameter grid ranges for the estimation of finite derivatives are inferred from parameters' :attr:`Parameter.delta`.
            These values are then scaled by ``delta_scale`` (< 1. means smaller ranges).
        """
        from .samples import DiffSampler
        options = {**self.sampler_options, **kwargs}
        sampler = DiffSampler(calculator, self.params, **options, mpicomm=self.mpicomm)
        sampler.run()
        return sampler.samples

    def fit(self, X, Y, attrs):
        if self.mpicomm.rank == 0:
            cidx = attrs['cidx']
            saccuracy = [attrs['accuracy'][param] for param in self.params]
            sorder = [attrs['order'][param] for param in self.params]
            ndim = X.shape[1]
            self.center = X[cidx]
            self.derivatives, self.powers = [], []
            self.powers = [(0,) * ndim]
            self.derivatives = [Y[cidx]]
            prefactor = 1
            for order in range(1, max(sorder + [0]) + 1):
                prefactor /= order
                for indices in itertools.product(range(ndim), repeat=order):
                    power = tuple(np.bincount(indices, minlength=ndim).astype('i4'))
                    if sum(power) > min(order for o, order in zip(power, sorder) if o):
                        continue
                    value = prefactor * deriv_nd(X, Y, [(iparam, order, accuracy) for iparam, (order, accuracy) in enumerate(zip(power, saccuracy)) if order > 0],
                                                 center=self.center, atol=0.)
                    if power in self.powers:
                        self.derivatives[self.powers.index(power)] += value
                    else:
                        self.derivatives.append(value)
                        self.powers.append(power)
            self.derivatives, self.powers = np.array(self.derivatives), np.array(self.powers)
        self.derivatives = mpi.bcast(self.derivatives if self.mpicomm.rank == 0 else None, mpicomm=self.mpicomm, mpiroot=0)
        self.powers = self.mpicomm.bcast(self.powers if self.mpicomm.rank == 0 else None, root=0)
        self.center = self.mpicomm.bcast(self.center if self.mpicomm.rank == 0 else None, root=0)

    @jit(static_argnums=[0])
    def predict(self, X):
        diffs = jnp.array(X - self.center)
        #diffs = jnp.where(self.powers > 0, diffs, 0.)  # a trick to avoid NaNs in the derivation
        #powers = jnp.prod(jnp.power(diffs, self.powers), axis=-1)
        powers = jnp.prod(jnp.where(self.powers > 0, diffs ** self.powers, 1.), axis=-1)
        return jnp.tensordot(self.derivatives, powers, axes=(0, 0))

    def __getstate__(self):
        state = {}
        for name in ['center', 'derivatives', 'powers']:
            state[name] = getattr(self, name)
        return state