"""Utilities for **cosmoprimo**."""

import os
import numpy as np


def mkdir(dirname):
    """Try to create ``dirnm`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname) # MPI...
    except OSError:
        return


class BaseClass(object):
    """
    BaseClass to be used throughout the **cosmoprimo** package.
    Implements a :meth:`copy` method.
    """
    def __copy__(self):
        """Return shallow copy of ``self``."""
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    copy = __copy__


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
        """
        Initialise :class:`SolveLeastSquares`.

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
        if self.isscalar: self.gradient = self.gradient[None,:]
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
        """Main method to be called; return parameters :math:`\mathbf{p}` best fitting ``delta``."""
        self.compute(delta)
        if self.isscalar: return self.params[...,0]
        return self.params

    def model(self):
        """Return model at :math:`\mathbf{p}`."""
        return self.params.dot(self.gradient)

    def chi2(self):
        """Return :math:`\chi^{2}` at :math:`\mathbf{p}`."""
        delta = self.delta - self.model()
        if self.precision.ndim == 1:
            return np.sum((delta * self.precision) * delta,axis=-1)
        return np.sum(delta.dot(self.precision) * delta,axis=-1)
