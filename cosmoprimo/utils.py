"""Utilities for **cosmoprimo**."""

import os
import functools

import numpy as np
from scipy import interpolate


def mkdir(dirname):
    """Try to create ``dirnm`` and catch :class:`OSError`."""
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


def flatarray(dtype=None):
    """Decorator that flattens input array and reshapes the output in the same form."""
    def make_wrapper(func):

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            toret_dtype = getattr(args[0], 'dtype', np.float64)
            input_dtype = dtype
            if not np.issubdtype(toret_dtype, np.floating):
                toret_dtype = np.float64
            if input_dtype is None:
                input_dtype = toret_dtype
            array = np.asarray(args[0], dtype=input_dtype)
            shape = array.shape
            array.shape = (-1,)
            toret = func(self, array, *args[1:], **kwargs)
            array.shape = shape
            toret.shape = toret.shape[:-1] + shape
            return toret.astype(dtype=toret_dtype, copy=False)

        return wrapper

    return make_wrapper


class SolveLeastSquares(BaseClass):
    r"""
    Class that solves the least square problem, i.e. solves :math:`d\chi^{2}/d\mathbf{p}` for :math:`\mathbf{p}`, with:

    .. math::

        \chi^{2} = \left(\mathbf{\delta} - \mathbf{p} \cdot \mathbf{grad}\right)^{T} \mathbf{F} \left(\mathbf{\delta} - \mathbf{p} \cdot \mathbf{grad}\right)

    >>> sls = SolveLeastSquares(np.ones(4))
    >>> sls(2*np.ones(4))
    2.0
    >>> sls.model()
    array([2., 2., 2., 2.])
    >>> sls.chi2()
    0.0
    """
    def __init__(self, gradient, precision=1.):
        r"""
        Initialize :class:`SolveLeastSquares`.

        Parameters
        ----------
        gradient : array_like, 1D array, 2D array
            Gradient :math:`\mathbf{grad}`` of the model (assumed constant), such that the model is :math:`\mathbf{p} \cdot \mathbf{grad}`.
            If 1D, corresponds to the data vector size (one parameter model).
            If 2D, first dimension is the number of model parameters, second is data vector size.

        precision : scalar, 1D array, 2D array
            Precision matrix i.e. :math:`\mathbf{F}`.
            If scalar, equivalent to diagonal matrix with all elements set to ``precision``.
            If 1D, equivalent to diagonal matrix filled with ``precision``.
            If 2D, symmetric precision matrix.
        """
        # gradient shape = (nparams,ndata)
        self.gradient = np.asarray(gradient)
        self.isscalar = self.gradient.ndim == 1
        if self.isscalar: self.gradient = self.gradient[None, :]
        self.precision = np.asarray(precision)
        if self.precision.ndim == 1:
            hv = self.gradient * self.precision
        else:
            hv = self.gradient.dot(self.precision)
        self.projector = np.linalg.inv(hv.dot(self.gradient.T)).dot(hv).T

    def compute(self, delta):
        """Solve least square problem for ``delta`` given :attr:`gradient`, :attr:`precision`."""
        self.delta = delta
        self.params = self.delta.dot(self.projector)

    def __call__(self, delta):
        r"""Main method to be called; return parameters :math:`\mathbf{p}` best fitting ``delta``."""
        self.compute(delta)
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
        zgrid = np.logspace(-8, np.log10(zmax), nz)
        self.zgrid = np.concatenate([[0.], zgrid])
        self.rgrid = distance(self.zgrid)
        self.interp = interpolate.UnivariateSpline(self.rgrid, self.zgrid, k=interp_order, s=0, ext='raise')

    def __call__(self, distance):
        """Return (interpolated) redshift at distance ``distance`` (scalar or array)."""
        return self.interp(distance)
