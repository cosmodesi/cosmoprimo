"""Utilities for **cosmoprimo**."""

import os
import functools
import inspect
import logging

import numpy as np
from scipy import interpolate


def mkdir(dirname):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname)  # MPI...
    except OSError:
        return


class BaseClass(object):
    """
    Base class to be used throughout the **cosmoprimo** package.
    Implements a :meth:`copy` method.
    """
    def __copy__(self):
        """Return shallow copy of ``self``."""
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def copy(self):
        return self.__copy__()


def addproperty(*attrs):

    """Add properties ``attrs`` to class ``cls`` (assumed to be internally stored as '_attr')."""

    def decorator(cls):

        def _make_property(name):

            @property
            def func(self):
                return getattr(self, '_{}'.format(name))

            return func

        for attr in attrs:
            setattr(cls, attr, _make_property(attr))

        return cls

    return decorator


def _bcast_dtype(*args):
    r"""If input arrays are all float32, return float32; else float64."""
    toret = np.result_type(*(getattr(arg, 'dtype', None) for arg in args))
    if not np.issubdtype(toret, np.floating):
        toret = np.float64
    return toret


def flatarray(iargs=[0], dtype=np.float64):
    """Decorator that flattens input array(s) and reshapes the output in the same form."""
    def make_wrapper(func):
        sig = inspect.signature(func)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ba = sig.bind_partial(*args, **kwargs)
            ba.apply_defaults()
            self, args = ba.args[0], list(ba.args[1:])
            toret_dtype = _bcast_dtype(*[args[iarg] for iarg in iargs])
            input_dtype = dtype
            if input_dtype is None:
                input_dtype = toret_dtype
            shape = None
            for iarg in iargs:
                array = np.asarray(args[iarg], dtype=input_dtype)
                if shape is not None:
                    if array.shape != shape:
                        raise ValueError('input arrays must have same shape, found {}, {}'.format(shape, array.shape))
                else:
                    shape = array.shape
                args[iarg] = array.ravel()

            toret = func(self, *args, **ba.kwargs)

            def reshape(toret):
                toret = np.asarray(toret, dtype=toret_dtype)
                toret.shape = toret.shape[:-1] + shape
                return toret

            if isinstance(toret, dict):
                for key, value in toret.items():
                    toret[key] = reshape(value)
            else:
                toret = reshape(toret)

            return toret

        return wrapper

    return make_wrapper


class LeastSquareSolver(BaseClass):
    r"""
    Class that solves the least square problem, i.e. solves :math:`d\chi^{2}/d\mathbf{p} = 0` for :math:`\mathbf{p}`, with:

    .. math::

        \chi^{2} = \left(\mathbf{\delta} - \mathbf{p} \cdot \mathbf{grad}\right)^{T} \mathbf{F} \left(\mathbf{\delta} - \mathbf{p} \cdot \mathbf{grad}\right)

    >>> lss = LeastSquareSolver(np.ones(4))
    >>> lss(2 * np.ones(4))
    2.0
    >>> lss.model()
    array([2., 2., 2., 2.])
    >>> lss.chi2()
    0.0
    """
    def __init__(self, gradient, precision=1., constraint_gradient=None, compute_inverse=True):
        r"""
        Initialize :class:`SolveLeastSquares`.

        Parameters
        ----------
        gradient : array_like, 1D array, 2D array
            Gradient :math:`\mathbf{grad}`` of the model (assumed constant),
            such that the model is :math:`\mathbf{p} \cdot \mathbf{grad}`.
            If 1D, shape is the data vector size (one parameter model).
            If 2D, first dimension is the number of model parameters, second is data vector size.

        precision : scalar, 1D array, 2D array
            Precision matrix i.e. :math:`\mathbf{F}`.
            If scalar, equivalent to diagonal matrix with all elements set to ``precision``.
            If 1D, equivalent to diagonal matrix filled with ``precision``.
            If 2D, symmetric precision matrix.

        constraint_gradient : array_like, 2D array
            Gradient of constraints w.r.t. model parameters.
            First dimension is the number of model parameters, second is number of constraints.

        compute_inverse : bool, default=True
            Compute matrix inverse to get Fisher matrix; this is faster in case of many calls to the solver.
            Else, ``np.linalg.solve`` is used to solve the system at each call, which may be numerically more stable
            than computing the inverse.
        """
        # gradient shape = (nparams,ndata)
        self.gradient = np.atleast_1d(gradient)
        self.isscalar = self.gradient.ndim == 1
        if self.isscalar:
            self.gradient = self.gradient[None, :]
        elif self.gradient.ndim != 2:
            raise ValueError('gradient must be at most 2D')
        self.precision = np.asarray(precision)
        if self.precision.ndim == 1:
            hv = self.gradient * self.precision
        else:
            hv = self.gradient.dot(self.precision)
        invfisher = hv.dot(self.gradient.T)
        if constraint_gradient is None:
            self.nconstraints = 0
        else:
            constraint_gradient = np.atleast_2d(constraint_gradient)
            self.nconstraints = constraint_gradient.shape[-1]
            if constraint_gradient.ndim != 2 or constraint_gradient.shape[0] != self.gradient.shape[0]:
                raise ValueError('constraint_gradient must be 2D, of first dimension the number of model parameters (gradient first dimension)')
            dtype = constraint_gradient.dtype
            # Possible improvement: block-inverse
            invfisher = np.bmat([[invfisher, - constraint_gradient],
                                 [constraint_gradient.T, np.zeros((self.nconstraints,) * 2, dtype=dtype)]]).A
            hv = np.bmat([[hv, np.zeros(constraint_gradient.shape, dtype=dtype)],
                          [np.zeros((self.nconstraints, self.gradient.shape[-1]), dtype=dtype), np.eye(self.nconstraints, dtype=dtype)]]).A
        self.inverse_fisher = invfisher
        self.gradient_precision = hv

        if compute_inverse:
            fisher = np.linalg.inv(invfisher)

            # Check inversion
            tmp = fisher.dot(invfisher)
            ref = np.eye(tmp.shape[0], dtype=tmp.dtype)
            if not np.allclose(tmp, ref, rtol=1e-04, atol=1e-04):
                import warnings
                warnings.warn('Numerically inaccurate inverse matrix, max absolute diff {:.6f}.'.format(np.max(np.abs(tmp - ref))))

            self.projector = fisher.dot(hv).T

    def compute(self, delta, constraint=None):
        """Solve least square problem for ``delta`` given :attr:`gradient`, :attr:`precision`."""
        self.delta = delta = np.atleast_1d(delta)
        if constraint is not None:
            delta = np.concatenate([self.delta, np.atleast_1d(constraint)], axis=-1)
        if hasattr(self, 'projector'):
            params = delta.dot(self.projector)
        else:
            params = np.linalg.solve(self.inverse_fisher, self.gradient_precision.dot(delta.T)).T
        self.params = params[..., :self.gradient.shape[0]]

    def __call__(self, delta, constraint=None):
        r"""Main method to be called; return parameters :math:`\mathbf{p}` best fitting ``delta``."""
        self.compute(delta, constraint=constraint)
        if self.isscalar: return self.params[..., 0]
        return self.params

    def model(self):
        r"""Return model at :math:`\mathbf{p}`."""
        return self.params.dot(self.gradient)

    def chi2(self):
        r"""Return :math:`\chi^{2}` at :math:`\mathbf{p}`."""
        delta = self.delta - self.model()
        if self.precision.ndim == 1:
            return np.sum((delta * self.precision) * delta, axis=-1)
        return np.sum(delta.dot(self.precision) * delta, axis=-1)


class DistanceToRedshift(BaseClass):

    """Class that holds a conversion distance -> redshift."""

    def __init__(self, distance, zmax=100., nz=2048, interp_order=3):
        """
        Initialize :class:`DistanceToRedshift`.
        Creates an array of redshift -> distance in log(redshift) and instantiates
        a spline interpolator distance -> redshift.

        Parameters
        ----------
        distance : callable
            Callable that provides distance as a function of redshift (array).

        zmax : float, default=100.
            Maximum redshift for redshift <-> distance mapping.

        nz : int, default=2048
            Number of points for redshift <-> distance mapping.

        interp_order : int, default=3
            Interpolation order, e.g. 1 for linear interpolation, 3 for cubic splines.
        """
        self.zgrid = np.insert(np.logspace(-8, np.log10(zmax), nz), 0, 0.)
        self.rgrid = distance(self.zgrid)
        self.interp = interpolate.UnivariateSpline(self.rgrid, self.zgrid, k=interp_order, s=0, ext='raise')

    def __call__(self, distance):
        """Return (interpolated) redshift at distance ``distance`` (scalar or array)."""
        return self.interp(distance)


logger = logging.getLogger('Plotting')


def savefig(filename, fig=None, bbox_inches='tight', pad_inches=0.1, dpi=200, **kwargs):
    """
    Save figure to ``filename``.

    Warning
    -------
    Take care to close figure at the end, ``plt.close(fig)``.

    Parameters
    ----------
    filename : string
        Path where to save figure.

    fig : matplotlib.figure.Figure, default=None
        Figure to save. Defaults to current figure.

    kwargs : dict
        Optional arguments for :meth:`matplotlib.figure.Figure.savefig`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from matplotlib import pyplot as plt
    mkdir(os.path.dirname(filename))
    logger.info('Saving figure to {}.'.format(filename))
    if fig is None:
        fig = plt.gcf()
    fig.savefig(filename, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=dpi, **kwargs)
    return fig