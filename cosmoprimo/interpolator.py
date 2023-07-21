"""
Utilities for power spectrum and correlation function interpolations.
Useful classes are :class:`PowerSpectrumInterpolator1D`, :class:`PowerSpectrumInterpolator2D`,
:class:`CorrelationFunctionInterpolator1D`, :class:`CorrelationFunctionInterpolator2D`.
"""

import inspect

import numpy as np
from scipy import interpolate, integrate

from .utils import BaseClass
from .fftlog import PowerToCorrelation, CorrelationToPower, TophatVariance
from . import constants


def get_default_k_callable():
    return np.logspace(-6., 2., 500)


def get_default_s_callable():
    return np.logspace(-6., 2., 500)


def get_default_z_callable():
    return np.linspace(0., 10., 60)


def _pad_log(k, pk, extrap_kmin=1e-6, extrap_kmax=1e2):
    """
    Pad ``pk`` and ``k`` in log10-log10-space between ``extrap_kmin`` and ``k[0]`` and ``k[-1]`` and ``extrap_kmax``.

    Parameters
    ----------
    k : array_like
        Wavenumbers.

    pk : array_like
        Power spectrum.

    extrap_kmin : float, default=1e-6
        Minimum wavenumber of extrapolation range.

    extrap_kmax : float, default=1e2
        Maximum wavenumber of extrapolation range.

    Returns
    -------
    logk : array
        log10 of wavenumbers.

    logpk : array
        log10 of power spectrum.
    """
    logk = np.log10(k)
    logpk = np.log10(pk)
    padlowk, padhighk = [], []
    padlowpk, padhighpk = None, None
    log_extrap_kmax = np.log10(extrap_kmax)
    log_extrap_kmin = np.log10(extrap_kmin)
    if log_extrap_kmax > logk[-1]:
        dlogpkdlogk = (logpk[-1] - logpk[-2]) / (logk[-1] - logk[-2])
        padhighk = [logk[-1] * 0.1 + log_extrap_kmax * 0.9, log_extrap_kmax]
        delta = [dlogpkdlogk * (padhighk[0] - logk[-1]), dlogpkdlogk * (padhighk[1] - logk[-1])]
        padhighpk = np.array([logpk[-1] + delta[0], logpk[-1] + delta[1]])
        # if log_extrap_kmax too close to logk
        if padhighk[1] <= padhighk[0] or padhighk[0] <= logk[-1]:
            logk = logk[:-1]
            logpk = logpk[:-1]
            padhighk = padhighk[1:]
            padhighpk = padhighpk[1:]
    if log_extrap_kmin < logk[0]:
        dlogpkdlogk = (logpk[1] - logpk[0]) / (logk[1] - logk[0])
        padlowk = [log_extrap_kmin, logk[0] * 0.1 + log_extrap_kmin * 0.9]
        delta = [dlogpkdlogk * (padlowk[0] - logk[0]), dlogpkdlogk * (padlowk[1] - logk[0])]
        padlowpk = np.array([logpk[0] + delta[0], logpk[0] + delta[1]])
        if padlowk[1] <= padlowk[0] or padlowk[1] >= logk[0]:
            logk = logk[1:]
            logpk = logpk[1:]
            padlowk = padlowk[:-1]
            padlowpk = padlowpk[:-1]
    logk = np.concatenate([padlowk, logk, padhighk], axis=0)
    s = [logpk]
    if padlowpk is not None: s = [padlowpk] + s
    if padhighpk is not None: s = s + [padhighpk]
    logpk = np.concatenate(s, axis=0)
    return logk, logpk


def _kernel_tophat_lowx(x2):
    r"""
    Maclaurin expansion of :math:`W(x) = 3 (\sin(x)-x\cos(x))/x^{3}` to :math:`\mathcal{O}(x^{10})`.
    Necessary numerically because at low x W(x) relies on the fine cancellation of two terms.

    Note
    ----
    Taken from https://github.com/LSSTDESC/CCL/blob/66397c7b53e785ae6ee38a688a741bb88d50706b/src/ccl_power.c
    """
    return 1. + x2 * (-1.0 / 10.0 + x2 * (1.0 / 280.0 + x2 * (-1.0 / 15120.0 + x2 * (1.0 / 1330560.0 + x2 * (-1.0 / 172972800.0)))))


def _kernel_tophat_highx(x):
    r"""Tophat function math:`W(x) = 3 (\sin(x)-x\cos(x))/x^{3}`."""
    return 3. * (np.sin(x) - x * np.cos(x)) / x**3


def kernel_tophat2(x):
    """Non-vectorized tophat function."""
    if x < 0.1: return _kernel_tophat_lowx(x**2)**2
    return _kernel_tophat_highx(x)**2


def integrate_sigma_d(pk, kmin=1e-6, kmax=1e2, epsrel=1e-5):
    r"""
    Return the r.m.s. of the displacement field, i.e.:

    .. math::

        \sigma_{d} = \sqrt{\frac{1}{6 \pi^{2}} \int dk P(k)}

    Parameters
    ----------
    pk : callable
        Power spectrum.

    kmin : float, default=1e-6
        Minimum wavenumber.

    kmax : float, default=1e2
        Maximum wavenumber.

    epsrel : float, default=1e-5
        Relative precision (for :meth:`scipy.integrate.quad` integration)

    Returns
    -------
    sigmad : float
        r.m.s. of the displacement field.

    """
    def integrand(logk):
        k = np.exp(logk)
        return k * pk(k)  # extra k factor because log integration

    sigma2 = 1. / 6. / np.pi**2 * integrate.quad(integrand, np.log(kmin), np.log(kmax), epsrel=epsrel)[0]
    return np.sqrt(sigma2)


@np.vectorize
def integrate_sigma_r2(r, pk, kmin=1e-6, kmax=1e2, epsrel=1e-5, kernel=kernel_tophat2):
    r"""
    Return the variance of perturbations smoothed by a kernel :math:`W` of radius :math:`r`, i.e.:

    .. math::

        \sigma_{r}^{2} = \frac{1}{2 \pi^{2}} \int dk k^{2} P(k) W^{2}(kr)

    Parameters
    ----------
    r : float
        Smoothing radius.

    pk : callable
        Power spectrum.

    kmin : float, default=1e-6
        Minimum wavenumber.

    kmax : float, default=1e2
        Maximum wavenumber.

    epsrel : float, default=1e-5
        Relative precision (for :meth:`scipy.integrate.quad` integration).

    kernel : callable, default=kernel_tophat2
        Kernel :math:`W^{2}`; defaults to (square of) top-hat kernel.

    Returns
    -------
    sigmar2 : float
        Variance of perturbations.
    """
    def integrand(logk):
        k = np.exp(logk)
        return pk(k) * kernel(k * r) * k**3  # extra k factor because log integration

    return 1. / 2. / np.pi**2 * integrate.quad(integrand, np.log(kmin), np.log(kmax), epsrel=epsrel)[0]


def _get_default_kwargs(func, start=0, remove=()):
    """
    Extract default parameters of ``func`` as a dictionary.

    Parameters
    ----------
    func : callable
        Function.

    start : int, default=0
        Ignore ``start`` first arguments.

    remove : tuple, default=()
        Remove these arguments.

    Returns
    -------
    default_params : dict
        Default ``func`` parameters.
    """
    parameters = inspect.signature(func).parameters
    default_params = {}
    for iname, (name, param) in enumerate(parameters.items()):
        if iname >= start:
            default_params[name] = param.default
    for rm in remove:
        default_params.pop(remove)
    return default_params


def _bcast_dtype(*args):
    """If input arrays are all float32, return float32; else float64."""
    toret = np.result_type(*(getattr(arg, 'dtype', None) for arg in args))
    if not np.issubdtype(toret, np.floating):
        toret = np.float64
    return toret


class GenericSpline(BaseClass):

    """Base class that handles 1D and 2D splines."""

    def __init__(self, x, y=0, fun=None, interp_x='log', extrap_fun='lin', extrap_xmin=None, extrap_xmax=None, interp_order_x=3, interp_order_y=None, extrap_y=False):
        """
        Initialize :class:`GenericSpline`.

        Parameters
        ----------
        x : array_like
            x-coordinates.

        y : array_like, float, default=0
            y-coordinates.

        fun : array_like, default=None
            Data to be interpolated.
            If ``y`` is scalar, should be 1D; else 2D, with shape ``(x.size, y.size)``.

        interp_x : string, default='log'
            If 'log', interpolation is performed in log-x coordinates.

        extrap_fun : string, default='lin'
            If 'log' (and ``interp_x`` is 'log'), ``fun`` is log-log extrpolated up to ``extrap_xmin``, ``extrap_xmax``.

        extrap_xmin : float, default=1e-6
            Minimum extrapolation range in ``x``.

        extrap_xmax : float, default=1e2
            Maximum extrapolation range in ``y``.

        interp_order_x : int, default=3
            Interpolation order, i.e. degree of smoothing spline along ``x``.

        interp_order_y : int, default=None
            Interpolation order, i.e. degree of smoothing spline along ``y``.
            If ``None``, the maximum order given ``y`` size (see :meth:`min_spline_order`) is considered.

        extrap_y : bool, default=False
            If ``True``, clip out-of-bounds ``y`` input coordinates in the 2D case.
        """
        #  Check order
        if np.ndim(y) == 0:
            fun = fun[:, None]
        x, y = (np.atleast_1d(x_) for x_ in [x, y])
        i_x = np.argsort(x)
        i_y = np.argsort(y)
        self.x, self.y, self.fun = x[i_x], y[i_y], fun[i_x, :][:, i_y]
        self.extrap_xmin, self.extrap_xmax = self.xmin, self.xmax
        self.interp_x = interp_x
        self.extrap_fun = extrap_fun
        self.interp_order_x = interp_order_x
        self.interp_order_y = interp_order_y
        self.extrap_y = extrap_y

        x = self.x
        if self.interp_x == 'log':
            x = np.log10(self.x)
        fun = self.fun

        if self.extrap_fun == 'log':
            if self.interp_x != 'log':
                raise ValueError('log-log extrapolation requires log-x interpolation')
            if extrap_xmin is None:
                extrap_xmin = min(1e-6, self.x[0])
            if extrap_xmax is None:
                extrap_xmax = max(1e2, self.x[-1])
            x, fun = _pad_log(self.x, self.fun, extrap_kmin=extrap_xmin, extrap_kmax=extrap_xmax)
            self.extrap_xmin = 10**x[0]
            self.extrap_xmax = 10**x[-1]

        if interp_order_y is None:
            self.interp_order_y = min(len(y) - 1, 3)
        if self.interp_order_y == 0:
            self.spline = interpolate.UnivariateSpline(x, fun, k=self.interp_order_x, s=0, ext='const')
        else:
            self.spline = interpolate.RectBivariateSpline(x, self.y, fun, kx=self.interp_order_x, ky=self.interp_order_y, s=0)

    @staticmethod
    def min_spline_order(x):
        """Return maximum spline order given ``x`` size."""
        return min(len(x) - 1, 3)

    @property
    def xmin(self):
        """Minimum (interpolated) ``x`` coordinate."""
        return self.x[0]

    @property
    def xmax(self):
        """Maximum (interpolated) ``x`` coordinate."""
        return self.x[-1]

    @property
    def ymin(self):
        """Minimum (interpolated) ``y`` coordinate."""
        return self.y[0]

    @property
    def ymax(self):
        """Maximum (interpolated) ``y`` coordinate."""
        return self.y[-1]

    def __call__(self, x, y=0, grid=True, islogx=False, bounds_error=True):
        """
        Evaluate spline (or its nu-th derivative) at positions x (and y in 2D case).

        Parameters
        ----------
        x : array_like
            1D array of points where to evaluate the spline.

        y : array_like, default=0
            1D array of points where to evaluate the spline (2D case).

        grid : bool, default=True
            Whether ``x``, ``y`` coordinates should be interpreted as a grid, in which case the output will be of shape ``x.shape + y.shape``.

        islogx : bool, default=False
            Whether input ``x`` is already in log10-space.

        bounds_error : bool, default=True
            If ``True``, raise a :class:`ValueError` for out-of-range values.
            Else, set out-of-range values to the boundary value.

        Returns
        -------
        toret : array
        """
        dtype = _bcast_dtype(x, y) if self.interp_order_y else _bcast_dtype(x)
        x, y = (np.asarray(xx, dtype=dtype) for xx in (x, y))
        if grid:
            toret_shape = x.shape + y.shape
        else:
            toret_shape = x.shape
        if bounds_error and (np.any(x < self.extrap_xmin) or np.any(x > self.extrap_xmax)):
            raise ValueError('Input x outside of extrapolation range (min: {} vs. {}; max: {} vs. {})'.format(x.min(), self.extrap_xmin, x.max(), self.extrap_xmax))
        if self.interp_x == 'log' and not islogx:
            x = np.log10(x)
        if self.interp_order_y == 0:
            toret = self.spline(x, ext='const')
            if grid and y.size:
                toret = np.repeat(toret[..., None], y.size, axis=-1)
        else:
            if self.interp_order_y != 0 and self.extrap_y:
                y = np.clip(y, self.ymin, self.ymax)
            elif bounds_error and (np.any(y < self.ymin) or np.any(y > self.ymax)):
                raise ValueError('Input y outside of interpolation range (min: {} vs. {}; max: {} vs. {})'.format(y.min(), self.ymin, y.max(), self.ymax))
            if grid:
                i_x = np.argsort(x.flat)
                i_y = np.argsort(y.flat)
                toret = self.spline(x.flat[i_x], y.flat[i_y], grid=grid)[np.ix_(np.argsort(i_x), np.argsort(i_y))]
            else:
                toret = self.spline(x, y, grid=False)

        toret.shape = toret_shape

        if self.extrap_fun == 'log':
            toret = 10**toret

        return toret.astype(dtype, copy=False)


class _BasePowerSpectrumInterpolator(BaseClass):

    """Base class for power spectrum interpolators."""

    def params(self):
        """Return interpolator parameter dictionary."""
        return {name: getattr(self, name) for name in self.default_params}

    def as_dict(self):
        """Return interpolator as a dictionary."""
        state = self.params()
        for name in ['k', 'pk']:
            state[name] = getattr(self, name)
        return state

    def clone(self, **kwargs):
        """
        Clone interpolator, i.e. return a deepcopy with (possibly) other attributes in ``kwargs``
        (:meth:`clone` witout arguments is the same as :meth:`deepcopy`).

        See :meth:`deepcopy` doc for warning about interpolators built from callables.
        """
        return self.__class__(**{**self.as_dict(), **kwargs})

    def deepcopy(self):
        """
        Deep copy interpolator.

        If interpolator ``interp1`` is built from callable, requires its evaluation at ``k``, such that e.g.:
        >>> interp2 = interp1.clone()
        will not be provide exactly the same interpolated values as ``interp1`` (due to interpolation errors).
        """
        return self.__class__(**self.as_dict())

    @property
    def kmin(self):
        """Minimum (interpolated) ``k`` value."""
        return self.k[0]

    @property
    def kmax(self):
        """Maximum (interpolated) ``k`` value."""
        return self.k[-1]


class PowerSpectrumInterpolator1D(_BasePowerSpectrumInterpolator):
    """
    1D power spectrum interpolator, broadly adapted from CAMB's P(k) interpolator by Antony Lewis
    in https://github.com/cmbant/CAMB/blob/master/camb/results.py, providing extra useful methods,
    such as :meth:`sigma_r` or :meth:`to_xi`.
    """

    def __init__(self, k, pk, interp_k='log', extrap_pk='log', extrap_kmin=None, extrap_kmax=None, interp_order_k=3):
        """
        Initialize :class:`PowerSpectrumInterpolator1D`.

        Parameters
        ----------
        k : array_like
            Wavenumbers.

        pk : array_like
            Power spectrum to be interpolated.

        interp_k : string, default='log'
            If 'log', interpolation is performed in log-k coordinates.

        extrap_pk : string, default='log'
            If 'log' (and ``interp_k`` is 'log'), ``fun`` is log-log extrapolated up to ``extrap_kmin``, ``extrap_kmax``.

        extrap_kmin : float, default=1e-6
            Minimum extrapolation range in ``k``.

        extrap_kmax : float, default=1e2
            Maximum extrapolation range in ``k``.

        interp_order_k : int, default=3
            Interpolation order, i.e. degree of smoothing spline along ``k``.
        """
        self._rsigma8sq = 1.
        self.spline = GenericSpline(k, fun=pk, interp_x=interp_k, extrap_fun=extrap_pk, extrap_xmin=extrap_kmin, extrap_xmax=extrap_kmax, interp_order_x=interp_order_k)
        self.k = self.spline.x
        self.extrap_kmin, self.extrap_kmax = self.spline.extrap_xmin, self.spline.extrap_xmax
        self.interp_k, self.extrap_pk = self.spline.interp_x, self.spline.extrap_fun
        self.interp_order_k = self.spline.interp_order_x
        self.is_from_callable = False

        def interp(k, islogk=False, **kwargs):
            k = np.asarray(k)
            return self.spline(k, islogx=islogk, **kwargs) * self._rsigma8sq

        self.interp = interp

    default_params = _get_default_kwargs(__init__, start=3)

    @property
    def pk(self):
        """Return power spectrum array (if interpolator built from callable, evaluate it), with normalisation."""
        if self.is_from_callable:
            return self(self.k)
        return self.spline.fun[:, 0] * self._rsigma8sq

    @classmethod
    def from_callable(cls, k=None, pk_callable=None):
        """
        Build :class:`PowerSpectrumInterpolator1D` from callable.

        Parameters
        ----------
        k : array_like, default=None
            Array of wavenumbers where the provided ``pk_callable`` can be trusted.
            It will be used if :attr:`pk` is requested.
            Must be strictly increasing.

        pk_callable : callable, default=None
            Power spectrum callable.

        Returns
        -------
        new : PowerSpectrumInterpolator1D
        """
        if k is None: k = get_default_k_callable()
        self = cls.__new__(cls)
        self.__dict__.update(self.default_params)
        self._rsigma8sq = 1.
        self.k = np.atleast_1d(k)
        self.extrap_kmin, self.extrap_kmax = self.kmin, self.kmax
        self.is_from_callable = True

        def interp(k, islogk=False, **kwargs):
            dtype = _bcast_dtype(k)
            k = np.asarray(k, dtype=dtype)
            if islogk: k = 10**k
            toret = pk_callable(k, **kwargs) * self._rsigma8sq
            return toret.astype(dtype=dtype, copy=False)

        self.interp = interp

        return self

    def __call__(self, k, islogk=False, **kwargs):
        """
        Evaluate power spectrum at wavenumbers ``k``.

        Parameters
        ----------
        k : array_like
            Wavenumbers where to evaluate the power spectrum.

        islogk : bool, default=False
            Whether input ``k`` is already in log10-space.
        """
        return self.interp(k, **kwargs)

    def sigma_d(self, nk=1024, epsrel=1e-5):
        r"""
        Return the r.m.s. of the displacement field, i.e.:

        .. math::

            \sigma_{d} = \sqrt{\frac{1}{6 \pi^{2}} \int dk P(k)}

        Parameters
        ----------
        nk : int, default=1024
            If not ``None``, performs trapezoidal integration with ``nk`` points between :attr:`extrap_kmin` and :attr:`extrap_kmax`.
            Else, uses `scipy.integrate.quad`.

        epsrel : float, default=1e-5
            Relative precision (for :meth:`scipy.integrate.quad` integration).

        Returns
        -------
        sigmad : array_like
        """
        if nk is None:
            return integrate_sigma_d(self, kmin=self.extrap_kmin, kmax=self.extrap_kmax, epsrel=epsrel)
        k = np.geomspace(self.extrap_kmin, self.extrap_kmax, nk)
        sigmasq = 1. / 6. / constants.pi**2 * integrate.trapz(self(k), x=k, axis=-1)
        return np.sqrt(sigmasq)

    def sigma_r(self, r, nk=1024, epsrel=1e-5):
        r"""
        Return the r.m.s. of perturbations in a sphere of :math:`r`, i.e.:

        .. math::

            \sigma_{r} = \sqrt{\frac{1}{2 \pi^{2}} \int dk k^{2} P(k) W^{2}(kr))

        Parameters
        ----------
        r : array_like
            Sphere radius.

        nk : int, default=1024
            If not ``None``, performs trapezoidal integration with ``nk`` points between :attr:`extrap_kmin` and :attr:`extrap_kmax`.
            Else, uses `scipy.integrate.quad`.

        epsrel : float, default=1e-5
            Relative precision (for :meth:`scipy.integrate.quad` integration).

        Returns
        -------
        sigmar : array_like
            Array of shape ``(r.size,)`` (null dimensions are squeezed).
        """
        if nk is None:
            return integrate_sigma_r2(r, self, kmin=self.extrap_kmin, kmax=self.extrap_kmax, epsrel=epsrel)**0.5
        k = np.geomspace(self.extrap_kmin, self.extrap_kmax, nk)
        s, var = TophatVariance(k)(self(k))
        return np.sqrt(GenericSpline(s, [0], var[:, None])(r))

    def sigma8(self, **kwargs):
        """Return the r.m.s. of perturbations in a sphere of 8."""
        return self.sigma_r(8., **kwargs)

    def rescale_sigma8(self, sigma8=1.):
        """Rescale power spectrum to the provided ``sigma8`` normalisation."""
        self._rsigma8sq = 1.  # reset rsigma8 for interpolation
        self._rsigma8sq = sigma8**2 / self.sigma8()**2

    def to_xi(self, nk=1024, fftlog_kwargs=None, **kwargs):
        """
        Transform power spectrum into correlation function using :class:`FFTlog`.

        nk : int, default=1024
            Number of wavenumbers used in FFTlog transform.

        fftlog_kwargs : dict, default=None
            Arguments for :class:`FFTlog`.

        kwargs : dict
            Arguments for the new :class:`CorrelationFunctionInterpolator1D` instance.

        Returns
        -------
        xi : CorrelationFunctionInterpolator1D
        """
        k = np.geomspace(self.extrap_kmin, self.extrap_kmax, nk)
        s, xi = PowerToCorrelation(k, complex=False, **(fftlog_kwargs or {}))(self(k))
        default_params = dict(interp_s='log', interp_order_s=self.interp_order_k)
        default_params.update(kwargs)
        return CorrelationFunctionInterpolator1D(s, xi=xi, **default_params)


class PowerSpectrumInterpolator2D(_BasePowerSpectrumInterpolator):
    """
    2D power spectrum interpolator, broadly adapted from CAMB's P(k) interpolator by Antony Lewis
    in https://github.com/cmbant/CAMB/blob/master/camb/results.py, providing extra useful methods,
    such as :meth:`sigma_rz` or :meth:`to_xi`.
    """

    def __init__(self, k, z=0, pk=None, interp_k='log', extrap_pk='log', extrap_kmin=None, extrap_kmax=None,
                 interp_order_k=3, interp_order_z=None, extrap_z=None, growth_factor_sq=None):
        r"""
        Initialize :class:`PowerSpectrumInterpolator2D`.

        ``growth_factor_sq`` is a callable that can be prodided to rescale the output of the base spline interpolation.
        Indeed, variations of :math:`z \rightarrow P(k,z)` are (mostly) :math:`k` scale independent, such that more accurate interpolation in ``z``
        can be achieved by providing the `z` variations separately in a well-sampled ``growth_factor_sq``.

        Parameters
        ----------
        k : array_like
            Wavenumbers.

        z : array_like, float, default=0
            Redshifts.

        pk : array_like
            Power spectrum to be interpolated.
            If ``z`` is scalar, should be 1D; else 2D, with shape ``(k.size, z.size)``.

        interp_k : string, default='log'
            If 'log', interpolation is performed in log-k coordinates.

        extrap_pk : string, default='log'
            If 'log' (and ``interp_k`` is 'log'), ``fun`` is log-log extrpolated up to ``extrap_kmin``, ``extrap_kmax``.

        extrap_kmin : float, default=1e-6
            Minimum extrapolation range in ``k``.

        extrap_kmax : float, default=1e2
            Maximum extrapolation range in ``k``.

        interp_order_k : int, default=3
            Interpolation order, i.e. degree of smoothing spline along ``k``.

        interp_order_z : int, default=None
            Interpolation order, i.e. degree of smoothing spline along ``z``.
            If ``None``, the maximum order given ``z`` size (see :meth:`GenericSpline.min_spline_order`) is considered.

        extrap_z : bool, default=None
            If ``True``, clip out-of-bounds ``z`` input coordinates.
            If ``None``, and ``growth_factor_sq`` is provided, defaults to ``True``
            (hence assuming ``growth_factor_sq`` will provide the extrapolation to out-of-bounds ``z`` input coordinates).

        growth_factor_sq : callable, default=None
            Function that takes ``z`` as argument and returns the growth factor squared at that redshift.
            This will rescale the output of the base spline interpolation.
            Therefore, make sure that provided ``pk`` does not contain the redundant ``z`` variations.
        """
        self._rsigma8sq = 1.
        self.growth_factor_sq = growth_factor_sq
        if extrap_z is None: extrap_z = self.growth_factor_sq is not None
        self.spline = GenericSpline(k, y=z, fun=pk, interp_x=interp_k, extrap_fun=extrap_pk, extrap_xmin=extrap_kmin, extrap_xmax=extrap_kmax,
                                    interp_order_x=interp_order_k, interp_order_y=interp_order_z, extrap_y=extrap_z)
        self.k, self.z = self.spline.x, self.spline.y
        self.extrap_kmin, self.extrap_kmax = self.spline.extrap_xmin, self.spline.extrap_xmax
        self.interp_k, self.extrap_pk, self.extrap_z = self.spline.interp_x, self.spline.extrap_fun, self.spline.extrap_y
        self.interp_order_k, self.interp_order_z = self.spline.interp_order_x, self.spline.interp_order_y
        self.is_from_callable = False

        def interp(k, z=0, islogk=False, ignore_growth=False, **kwargs):
            toret = self.spline(k, z, islogx=islogk, **kwargs)
            if self.growth_factor_sq is not None and not ignore_growth:
                toret = toret * self.growth_factor_sq(z)
            return toret * self._rsigma8sq

        self.interp = interp

    default_params = _get_default_kwargs(__init__, start=4)

    def as_dict(self):
        """Return interpolator as a dictionary."""
        state = super(PowerSpectrumInterpolator2D, self).as_dict()
        state['z'] = self.z
        return state

    @property
    def pk(self):
        """Return power spectrum array (if interpolator built from callable, evaluate it), without growth factor, but with normalisation."""
        if self.is_from_callable:
            kwargs = {'ignore_growth': True} if self.growth_factor_sq is not None else {}
            return self(self.k, self.z, **kwargs)
        return self.spline.fun * self._rsigma8sq

    @property
    def zmin(self):
        """Minimum (spline-interpolated) redshift."""
        return self.z[0]

    @property
    def zmax(self):
        """Maximum (spline-interpolated) redshift."""
        return self.z[-1]

    @classmethod
    def from_callable(cls, k=None, z=None, pk_callable=None, growth_factor_sq=None):
        """
        Build :class:`PowerSpectrumInterpolator2D` from callable.

        Parameters
        ----------
        k : array_like, default=None
            Array of wavenumbers where the provided ``pk_callable`` can be trusted.
            It will be used if :attr:`pk` is requested.
            Must be strictly increasing.

        z : array_like, default=None
            Array of redshifts where the provided ``pk_callable`` can be trusted.
            Same remark as for ``k``.

        pk_callable : callable, default=None
            Power spectrum callable.
            If ``growth_factor_sq`` is not provided, should take ``k``, ``z``, ``grid`` as arguments (see :meth:`__call__`)
            else, should take ``k`` as arguments.

        growth_factor_sq : callable, default=None
            Function that takes ``z`` as argument and returns the growth factor squared at that redshift.
            See remark above.

        Returns
        -------
        self : PowerSpectrumInterpolator2D
        """
        if k is None: k = get_default_k_callable()
        if z is None: z = get_default_z_callable()
        self = cls.__new__(cls)
        self.__dict__.update(self.default_params)
        self._rsigma8sq = 1.
        self.k, self.z = (np.atleast_1d(xx) for xx in (k, z))
        self.extrap_kmin, self.extrap_kmax = self.kmin, self.kmax
        self.interp_order_z = GenericSpline.min_spline_order(self.z)
        self.growth_factor_sq = growth_factor_sq
        self.is_from_callable = True

        if self.growth_factor_sq is not None:

            def interp(k, z=0, grid=True, islogk=False, ignore_growth=False):
                dtype = _bcast_dtype(k, z)
                k, z = (np.asarray(xx, dtype=dtype) for xx in (k, z))
                if islogk: k = 10**k
                toret = pk_callable(k) * self._rsigma8sq
                if grid:
                    toret_shape = k.shape + z.shape
                else:
                    toret_shape = k.shape
                if not ignore_growth:
                    growth = self.growth_factor_sq(z)
                    if grid:
                        toret = toret[..., None] * growth.ravel()
                    else:
                        toret = toret * growth
                elif z.size:
                    toret = np.repeat(toret[..., None], z.size, axis=-1)
                toret.shape = toret_shape
                return toret.astype(dtype=dtype, copy=False)

        else:

            def interp(k, z=0, grid=True, islogk=False):
                dtype = _bcast_dtype(k, z)
                k, z = (np.asarray(xx, dtype=dtype) for xx in (k, z))
                if islogk: k = 10**k
                toret = pk_callable(k, z=z, grid=grid) * self._rsigma8sq
                return toret.astype(dtype=dtype, copy=False)

        self.interp = interp
        return self

    def __call__(self, k, z=0, grid=True, islogk=False, **kwargs):
        """
        Evaluate power spectrum at wavenumbers ``k`` and redshifts ``z``.

        Parameters
        ----------
        k : array_like
            Wavenumbers where to evaluate the power spectrum.

        z : array_like, default=0
            Redshifts where to evaluate the power spectrum.

        grid : bool, default=True
            Whether ``k``, ``z`` coordinates should be interpreted as a grid, in which case the output will be of shape ``(k.size, z.size)``.

        islogk : bool, default=False
            Whether input ``k`` is already in log10-space.

        ignore_growth : bool, default=False
            Whether to ignore multiplication by growth function (if provided).
        """
        return self.interp(k, z=z, grid=grid, islogk=islogk, **kwargs)

    def sigma_dz(self, z=0, nk=1024, epsrel=1e-5):
        r"""
        Return the r.m.s. of the displacement field, i.e.:

        .. math::

            \sigma_{d}(z) = \sqrt{\frac{1}{6 \pi^{2}} \int dk P(k,z)}

        Parameters
        ----------
        z : array_like, default=0
            Redshifts.

        nk : int, default=1024
            If not ``None``, performs trapezoidal integration with ``nk`` points between :attr:`extrap_kmin` and :attr:`extrap_kmax`.
            Else, uses `scipy.integrate.quad`.

        epsrel : float, default=1e-5
            Relative precision (for :meth:`scipy.integrate.quad` integration).

        Returns
        -------
        sigmadz : array_like
        """
        dtype = _bcast_dtype(z)
        z = np.asarray(z, dtype=dtype)
        if nk is None:
            toret = np.array([self.to_1d(z=zz).sigma_d(epsrel=epsrel) for zz in z.flat])
            toret.shape = z.shape
            return toret.astype(dtype=dtype, copy=False)
        k = np.geomspace(self.extrap_kmin, self.extrap_kmax, nk)
        sigmasq = 1. / 6. / constants.pi**2 * integrate.trapz(self(k, z), x=k, axis=0)
        return np.sqrt(sigmasq).astype(dtype=dtype, copy=False)

    def sigma_rz(self, r, z=0, nk=1024, epsrel=1e-5):
        r"""
        Return the r.m.s. of perturbations in a sphere of :math:`r`, i.e.:

        .. math::

            \sigma_{r}(z) = \sqrt{\frac{1}{2 \pi^{2}} \int dk k^{2} P(k, z) W^{2}(kr)}

        Parameters
        ----------
        r : array_like
            Sphere radii.

        z : array_like, default=0
            Redshifts.

        nk : int, default=1024
            If not ``None``, performs trapezoidal integration with ``nk`` points between :attr:`extrap_kmin` and :attr:`extrap_kmax`.
            Else, uses `scipy.integrate.quad`.

        epsrel : float, default=1e-5
            Relative precision (for :meth:`scipy.integrate.quad` integration).

        Returns
        -------
        sigmarz : array_like
            Array of shape ``(r.size, z.size)`` (null dimensions are squeezed).
        """
        dtype = _bcast_dtype(z)
        z = np.asarray(z, dtype=dtype)
        if nk is None:
            toret = np.array([self.to_1d(z=zz).sigma_r(r, epsrel=epsrel) for zz in z.flat]).T
            toret.shape = z.shape
            return toret.astype(dtype=dtype, copy=False)
        k = np.geomspace(self.extrap_kmin, self.extrap_kmax, nk)
        s, var = TophatVariance(k)(self(k, z=self.z).T)
        return np.sqrt(GenericSpline(s, self.z, var.T)(r, z, grid=True)).astype(dtype=dtype, copy=False)

    def sigma8_z(self, z=0, **kwargs):
        """Return the r.m.s. of perturbations in a sphere of 8."""
        return self.sigma_rz(8., z=z, **kwargs)

    def rescale_sigma8(self, sigma8=1.):
        """Rescale power spectrum to the provided ``sigma8`` normalisation  at :math:`z = 0`."""
        self._rsigma8sq = 1.  # reset rsigma8 for interpolation
        self._rsigma8sq = sigma8**2 / self.sigma8_z(z=0)**2

    def growth_rate_rz(self, r, z=0, dz=1e-3, nk=1024, epsrel=1e-5):
        r"""
        Evaluate the growth rate at the log-derivative of perturbations in a sphere of :math:`r`, i.e.:

        .. math:

            f(r,z) = \frac{d ln \sigma_r(z)}{d ln a}

        With :math:`z = ln(a)`.

        Parameters
        ----------
        r : array_like
            Sphere radii.

        z : array_like, default=0
            Redshifts.

        dz : float, default=1e-3
            ``z`` interval used for finite differentiation.

        nk : int, default=1024
            If not ``None``, performs trapezoidal integration with ``nk`` points between :attr:`extrap_kmin` and :attr:`extrap_kmax`.
            Else, uses `scipy.integrate.quad`.

        epsrel : float, default=1e-5
            Relative precision (for :meth:`scipy.integrate.quad` integration).
        """
        if self.interp_order_z == 0 and self.growth_factor_sq is None:
            import warnings
            warnings.warn('No redshift evolution provided, growth rate is 0')
            return 0.
        hdz = dz / 2.

        dtype = _bcast_dtype(r, z)
        r, z = (np.asarray(xx, dtype=dtype) for xx in (r, z))
        toret_shape = r.shape + z.shape
        z.shape = -1

        def finite_difference(fun):
            mask = z < self.zmin + hdz
            toret = np.empty(r.shape + z.shape, dtype=dtype)
            # See eq. 6 of https://arxiv.org/abs/2102.05049, corrected
            toret[..., mask] = (-fun(z[mask] + dz) + 4 * fun(z[mask] + hdz) - 3 * fun(z[mask])) / dz
            toret[..., ~mask] = (fun(z[~mask] + hdz) - fun(z[~mask] - hdz)) / dz
            return toret

        dsigdz = finite_difference(lambda z: np.log(self.sigma_rz(r, z, nk=nk, epsrel=epsrel)))
        # a = 1/(1 + z) => da = -1/(1+z)^2 dz => dln(a) = -1/(1 + z) dz
        dsigdlna = -dsigdz * (1 + z)
        dsigdlna.shape = toret_shape
        return dsigdlna

    def to_1d(self, z=0, **kwargs):
        """
        Return :class:`PowerSpectrumInterpolator1D` instance corresponding to interpolator at ``z``.

        Parameters
        ----------
        z : float, default=0
            Redshift.

        kwargs : dict
            Arguments for the new :class:`PowerSpectrumInterpolator1D` instance.
            By default, the new instance inherits the same ``k`` interpolattion and extrapolation settings.

        Returns
        -------
        interp : PowerSpectrumInterpolator1D
        """
        if self.is_from_callable:
            return PowerSpectrumInterpolator1D.from_callable(self.k, pk_callable=lambda k: self.interp(k, z=z))
        default_params = dict(extrap_pk=self.extrap_pk, extrap_kmin=self.extrap_kmin, extrap_kmax=self.extrap_kmax, interp_order_k=self.interp_order_k)
        default_params.update(kwargs)
        return PowerSpectrumInterpolator1D(self.k, self(self.k, z=z), **default_params)

    def to_xi(self, nk=1024, fftlog_kwargs=None, **kwargs):
        """
        Transform power spectrum into correlation function using :class:`FFTlog`.

        nk : int, default=1024
            Number of wavenumbers used in FFTlog transform.

        fftlog_kwargs : dict, default=None
            Arguments for :class:`FFTlog`.

        kwargs : dict
            Arguments for the new :class:`CorrelationFunctionInterpolator2D` instance.

        Returns
        -------
        xi : CorrelationFunctionInterpolator2D
        """
        k = np.geomspace(self.extrap_kmin, self.extrap_kmax, nk)
        s, xi = PowerToCorrelation(k, complex=False, **(fftlog_kwargs or {}))(self(k, z=self.z, ignore_growth=True).T)
        default_params = dict(interp_s='log', interp_order_s=self.interp_order_k,
                              interp_order_z=self.interp_order_z, extrap_z=self.extrap_z, growth_factor_sq=self.growth_factor_sq)
        default_params.update(kwargs)
        return CorrelationFunctionInterpolator2D(s, z=self.z, xi=xi.T, **default_params)


class _BaseCorrelationFunctionInterpolator(BaseClass):

    """Base class for correlation function interpolators."""

    def params(self):
        """Return interpolator parameter dictionary."""
        return {name: getattr(self, name) for name in self.default_params}

    def as_dict(self):
        """Return interpolator as a dictionary."""
        state = self.params()
        for name in ['s', 'xi']:
            state[name] = getattr(self, name)
        return state

    def clone(self, **kwargs):
        """
        Clone interpolator, i.e. return a deepcopy with (possibly) other attributes in ``kwargs``
        (:meth:`clone` witout arguments is the same as :meth:`deepcopy`).

        See :meth:`deepcopy` doc for warning about interpolators built from callables.
        """
        return self.__class__(**{**self.as_dict(), **kwargs})

    def deepcopy(self):
        """
        Deep copy interpolator.

        If interpolator ``interp1`` is built from callable, requires its evaluation at ``k``, such that e.g.:
        >>> interp2 = interp1.clone()
        will not be provide exactly the same interpolated values as ``interp1`` (due to interpolation errors).
        """
        return self.__class__(**self.as_dict())

    @property
    def smin(self):
        """Minimum (interpolated) ``s`` value."""
        return self.s[0]

    @property
    def smax(self):
        """Maximum (interpolated) ``k`` value."""
        return self.s[-1]

    @property
    def extrap_smin(self):
        """Minimum (extrapolated) ``s`` value (same as minimum interpolated value)."""
        return self.s[0]

    @property
    def extrap_smax(self):
        """Maximum (extrapolated) ``s`` value (same as maximum interpolated value)."""
        return self.s[-1]


class CorrelationFunctionInterpolator1D(_BaseCorrelationFunctionInterpolator):

    """1D correlation funcion interpolator."""

    def __init__(self, s, xi, interp_s='log', interp_order_s=3):
        """
        Initialize :class:`CorrelationFunctionInterpolator1D`.

        Parameters
        ----------
        s : array_like
            Seperations.

        xi : array_like
            Correlation function to be interpolated.

        interp_s : string, default='log'
            If 'log', interpolation is performed in log-s coordinates.

        interp_order_s : int, default=3
            Interpolation order, i.e. degree of smoothing spline along ``s``.
        """
        self._rsigma8sq = 1.
        self.spline = GenericSpline(s, fun=xi, interp_x=interp_s, interp_order_x=interp_order_s)
        self.s = self.spline.x
        self.interp_s = self.spline.interp_x
        self.interp_order_s = self.spline.interp_order_x
        self.is_from_callable = False

        def interp(s, islogs=False, **kwargs):
            return self.spline(s, islogx=islogs, **kwargs) * self._rsigma8sq

        self.interp = interp

    default_params = _get_default_kwargs(__init__, start=3)

    @property
    def xi(self):
        """Return correlation function array (if interpolator built from callable, evaluate it), with normalisation."""
        if self.is_from_callable:
            return self(self.s)
        return self.spline.fun[:, 0] * self._rsigma8sq

    @classmethod
    def from_callable(cls, s=None, xi_callable=None):
        """
        Build :class:`CorrelationFunctionInterpolator1D` from callable.

        Parameters
        ----------
        s : array_like, default=None
            Array of separations where the provided ``xi_callable`` can be trusted.
            It will be used if ``:attr:xi`` is requested.
            Must be strictly increasing.

        xi_callable : callable
            Correlation function callable.

        Returns
        -------
        self : CorrelationFunctionInterpolator1D
        """
        if s is None: s = get_default_s_callable()
        self = cls.__new__(cls)
        self.__dict__.update(self.default_params)
        self._rsigma8sq = 1.
        self.s = np.atleast_1d(s)

        def interp(s, islogs=False, **kwargs):
            dtype = _bcast_dtype(s)
            s = np.asarray(s, dtype=dtype)
            if islogs: s = 10**s
            toret = xi_callable(s, **kwargs) * self._rsigma8sq
            return toret.astype(dtype=dtype, copy=False)

        self.interp = interp
        return self

    def __call__(self, s, islogs=False, **kwargs):
        """
        Evaluate correlation function at separations ``s``.

        Parameters
        ----------
        s : array_like
            Separations where to evaluate the correlation function.

        islogs : bool, default=False
            Whether input ``s`` is already in log10-space.
        """
        return self.interp(s, islogs=islogs, **kwargs)

    def sigma_d(self, **kwargs):
        """
        Return the r.m.s. of the displacement field by transforming correlation function into power spectrum.

        See :meth:`PowerSpectrumInterpolator1D.sigma_d` arguments.
        """
        return self.to_pk().sigma_d(**kwargs)

    def sigma_r(self, r, **kwargs):
        """
        Return the r.m.s. of perturbations in a sphere of :math:`r` by transforming correlation function into power spectrum.

        See :meth:`PowerSpectrumInterpolator1D.sigma_r` arguments.
        """
        return self.to_pk().sigma_r(r, **kwargs)

    def sigma8(self, **kwargs):
        """Return the r.m.s. of perturbations in a sphere of 8."""
        return self.sigma_r(8., **kwargs)

    def rescale_sigma8(self, sigma8=1.):
        """Rescale the correlation function to the provided ``sigma8`` normalisation."""
        self._rsigma8sq = 1.  # reset rsigma8 for interpolation
        self._rsigma8sq = sigma8**2 / self.sigma8()**2

    def to_pk(self, ns=1024, fftlog_kwargs=None, **kwargs):
        """
        Transform correlation function into power spectrum using :class:`FFTlog`.

        ns : int, default=1024
            Number of separations used in FFTlog transform.

        fftlog_kwargs : dict, default=None
            Arguments for :class:`FFTlog`.

        kwargs : dict
            Arguments for the new :class:`PowerSpectrumInterpolator1D` instance.

        Returns
        -------
        pk : PowerSpectrumInterpolator1D
        """
        s = np.geomspace(self.extrap_smin, self.extrap_smax, ns)
        k, pk = CorrelationToPower(s, complex=False, **(fftlog_kwargs or {}))(self(s))
        default_params = dict(interp_k='log', interp_order_k=self.interp_order_s)
        default_params.update(kwargs)
        return PowerSpectrumInterpolator1D(k, pk=pk, **default_params)


class CorrelationFunctionInterpolator2D(_BaseCorrelationFunctionInterpolator):

    """2D correlation function interpolator."""

    def __init__(self, s, z=0, xi=None, interp_s='log', interp_order_s=3, interp_order_z=None, extrap_z=None, growth_factor_sq=None):
        r"""
        Initialize :class:`CorrelationFunctionInterpolator2D`.

        ``growth_factor_sq`` is a callable that can be prodided to rescale the output of the base spline interpolation.
        Indeed, variations of :math:`z \rightarrow \xi(k,z)` are (mostly) :math:`s` scale independent, such that more accurate interpolation in ``z``
        can be achieved by providing the `z` variations separately in a well-sampled ``growth_factor_sq``.

        Parameters
        ----------
        s : array_like
            Separations.

        z : array_like, float, default=0
            Redshifts.

        xi : array_like
            Correlation function to be interpolated.
            If ``z`` is scalar, should be 1D; else 2D, with shape ``(s.size, z.size)``.

        interp_s : string, default='log'
            If 'log', interpolation is performed in log-s coordinates.

        interp_order_s : int, default=3
            Interpolation order, i.e. degree of smoothing spline along ``s``.

        interp_order_z : int, default=None
            Interpolation order, i.e. degree of smoothing spline along ``z``.
            If ``None``, the maximum order given ``z`` size (see :meth:`GenericSpline.min_spline_order`) is considered.

        extrap_z : bool, default=None
            If ``True``, clip out-of-bounds ``z`` input coordinates.
            If ``None``, and ``growth_factor_sq`` is provided, defaults to ``True``
            (hence assuming ``growth_factor_sq`` will provide the extrapolation to out-of-bounds ``z`` input coordinates).

        growth_factor_sq : callable, default=None
            Function that takes ``z`` as argument and returns the growth factor squared at that redshift.
            This will rescale the output of the base spline interpolation.
            Therefore, make sure that provided ``pk`` does not contain the redundant ``z`` variations.
        """
        self._rsigma8sq = 1.
        self.growth_factor_sq = growth_factor_sq
        if extrap_z is None: extrap_z = self.growth_factor_sq is not None
        self.spline = GenericSpline(s, y=z, fun=xi, interp_x=interp_s, interp_order_x=interp_order_s, interp_order_y=interp_order_z, extrap_y=extrap_z)
        self.s, self.z = self.spline.x, self.spline.y
        self.interp_s, self.extrap_z = self.spline.interp_x, self.spline.extrap_y
        self.interp_order_s, self.interp_order_z = self.spline.interp_order_x, self.spline.interp_order_y
        self.is_from_callable = False

        def interp(s, z=0, islogs=False, ignore_growth=False, **kwargs):
            toret = self.spline(s, z, islogx=islogs, **kwargs)
            if self.growth_factor_sq is not None and not ignore_growth:
                toret = toret * self.growth_factor_sq(z)
            return toret * self._rsigma8sq

        self.interp = interp

    default_params = _get_default_kwargs(__init__, start=4)

    def as_dict(self):
        """Return interpolator as a dictionary."""
        state = super(CorrelationFunctionInterpolator2D, self).as_dict()
        state['z'] = self.z
        return state

    @property
    def xi(self):
        """Return ``xi`` (if interpolator built from callable, evaluate it), without growth factor, but with normalisation."""
        if self.is_from_callable:
            growth_factor_sq = self.growth_factor_sq
            self.growth_factor_sq = lambda x: np.ones_like(x)  # to avoid using growth factor in following call
            toret = self(self.s, self.z)
            self.growth_factor_sq = growth_factor_sq
            return toret
        return self.spline.fun * self._rsigma8sq

    @property
    def zmin(self):
        """Minimum (spline-interpolated) redshift."""
        return self.z[0]

    @property
    def zmax(self):
        """Maximum (spline-interpolated) redshift."""
        return self.z[-1]

    def __call__(self, s, z=0, grid=True, islogs=False, **kwargs):
        """
        Evaluate correlation function at separations ``s`` and redshifts ``z``.

        Parameters
        ----------
        s : array_like
            Separations where to evaluate the correlation function.

        z : array_like, default=0
            Redshifts where to evaluate the correlation function.

        grid : bool, default=True
            Whether ``s``, ``z`` coordinates should be interpreted as a grid, in which case the output will be of shape ``(s.size, z.size)``.

        islogs : bool, default=False
            Whether input ``s`` is already in log10-space.

        ignore_growth : bool, default=False
            Whether to ignore multiplication by growth function (if provided).
        """
        return self.interp(s, z=z, grid=grid, islogs=islogs, **kwargs)

    @classmethod
    def from_callable(cls, s=None, z=None, xi_callable=None, growth_factor_sq=None):
        """
        Build :class:`CorrelationFunctionInterpolator2D` from callable.

        Parameters
        ----------
        s : array_like, default=None
            Array of separations where the provided ``xi_callable`` can be trusted.
            It will be used if ``:attr:xi`` is requested.
            Must be strictly increasing.

        z : array_like, default=None
            Array of redshifts where the provided ``xi_callable`` can be trusted.
            Same remark as for ``s``.

        xi_callable : callable, default=None
            Correlation function callable.
            If ``growth_factor_sq`` is not provided, should take ``s``, ``z``, ``grid`` as arguments (see :meth:`__call__`)
            else, should take ``s`` as arguments.

        growth_factor_sq : callable, default=None
            See remark above.

        Returns
        -------
        self : CorrelationFunctionInterpolator2D
        """
        if s is None: s = get_default_s_callable()
        if z is None: z = get_default_z_callable()
        self = cls.__new__(cls)
        self.__dict__.update(self.default_params)
        self._rsigma8sq = 1.
        self.s, self.z = (np.atleast_1d(xx) for xx in (s, z))
        self.__dict__.update(self.default_params)
        self.interp_order_z = GenericSpline.min_spline_order(self.z)
        self.growth_factor_sq = growth_factor_sq
        self.is_from_callable = True

        if self.growth_factor_sq is not None:

            def interp(s, z=0, grid=True, islogs=False, ignore_growth=False):
                dtype = _bcast_dtype(s, z)
                s, z = (np.asarray(xx, dtype=dtype) for xx in (s, z))
                if islogs: s = 10**s
                toret = xi_callable(s) * self._rsigma8sq
                if grid:
                    toret_shape = s.shape + z.shape
                else:
                    toret_shape = s.shape
                if not ignore_growth:
                    growth = self.growth_factor_sq(z)
                    if grid:
                        toret = toret[..., None] * growth.ravel()
                    else:
                        toret = toret * growth
                elif z.size:
                    toret = np.repeat(toret[..., None], z.size, axis=-1)
                toret.shape = toret_shape
                return toret.astype(dtype=dtype, copy=False)

        else:

            def interp(s, z=0, grid=True, islogs=False):
                dtype = _bcast_dtype(s)
                s = np.asarray(s, dtype=dtype)
                if islogs: s = 10**s
                toret = xi_callable(s, z=z, grid=grid) * self._rsigma8sq
                return toret.astype(dtype=dtype, copy=False)

        self.interp = interp
        return self

    def sigma_dz(self, z=0, **kwargs):
        """
        Return the r.m.s. of the displacement field by transforming correlation function into power spectrum.

        See :meth:`PowerSpectrumInterpolator2D.sigma_dz` arguments.
        """
        return self.to_pk().sigma_dz(z=z, **kwargs)

    def sigma_rz(self, r, z=0, **kwargs):
        """
        Return the r.m.s. of perturbations in a sphere of :math:`r` by transforming correlation function into power spectrum.

        See :meth:`PowerSpectrumInterpolator2D.sigma_rz` arguments.
        """
        return self.to_pk().sigma_rz(r, z=z, **kwargs)

    def sigma8_z(self, z=0, **kwargs):
        """
        Return the r.m.s. of perturbations in a sphere of 8 by transforming correlation function into power spectrum.

        See :meth:`PowerSpectrumInterpolator2D.sigma8_z` arguments.
        """
        return self.sigma_rz(8., z=z, **kwargs)

    def rescale_sigma8(self, sigma8=1.):
        """Rescale correlation function to the provided ``sigma8`` normalisation  at :math:`z = 0`."""
        self._rsigma8sq = 1.  # reset rsigma8 for interpolation
        self._rsigma8sq = sigma8**2 / self.sigma8_z(z=0)**2

    def growth_rate_rz(self, r, z=0, **kwargs):
        r"""
        Evaluate the growth rate at the log-derivative of perturbations in a sphere of :math:`r` by transforming correlation function into power spectrum.

        See :meth:`PowerSpectrumInterpolator2D.growth_rate_rz` arguments.
        """
        return self.to_pk().growth_rate_rz(r, z=z, **kwargs)

    def to_1d(self, z=0, **kwargs):
        """
        Return :class:`CorrelationFunctionInterpolator1D` instance corresponding to interpolator at ``z``.

        Parameters
        ----------
        z : float, default=0
            Redshift.

        kwargs : dict
            Arguments for the new :class:`CorrelationFunctionInterpolator1D` instance.
            By default, the new instance inherits the same ``s`` interpolattion settings.

        Returns
        -------
        interp : CorrelationFunctionInterpolator1D
        """
        if self.is_from_callable:
            return CorrelationFunctionInterpolator1D.from_callable(self.s, xi_callable=lambda s: self.interp(s, z=z))
        default_params = dict(interp_order_s=self.interp_order_s)
        default_params.update(kwargs)
        return CorrelationFunctionInterpolator1D(self.s, self(self.s, z=z), **default_params)

    def to_pk(self, ns=1024, fftlog_kwargs=None, **kwargs):
        """
        Transform correlation function into power spectrum using :class:`FFTlog`.

        ns : int, default=1024
            Number of separations used in FFTlog transform.

        fftlog_kwargs : dict, default=None
            Arguments for :class:`FFTlog`.

        kwargs : dict
            Arguments for the new :class:`PowerSpectrumInterpolator2D` instance.

        Returns
        -------
        pk : PowerSpectrumInterpolator2D
        """
        s = np.geomspace(self.extrap_smin, self.extrap_smax, ns)
        k, pk = CorrelationToPower(s, complex=False, **(fftlog_kwargs or {}))(self(s, self.z, ignore_growth=True).T)
        default_params = dict(interp_k='log', extrap_pk='log', interp_order_k=self.interp_order_s,
                              interp_order_z=self.interp_order_z, extrap_z=self.extrap_z, growth_factor_sq=self.growth_factor_sq)
        default_params.update(kwargs)
        return PowerSpectrumInterpolator2D(k, z=self.z, pk=pk.T, **default_params)
