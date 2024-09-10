"""
Utilities for power spectrum and correlation function interpolations.
Useful classes are :class:`PowerSpectrumInterpolator1D`, :class:`PowerSpectrumInterpolator2D`,
:class:`CorrelationFunctionInterpolator1D`, :class:`CorrelationFunctionInterpolator2D`.
"""

import inspect

import numpy as np
from scipy import integrate

from .utils import BaseClass, _bcast_dtype
from .fftlog import PowerToCorrelation, CorrelationToPower, TophatVariance
from .jax import numpy as jnp
from .jax import romberg, simpson, opmask, _mask_bounds, numpy_jax


def get_default_k_callable():
    # Taken from https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/training/spectra_generation_scripts/2_create_spectra.py
    k = np.concatenate([np.logspace(-5, -4, num=20, endpoint=False),
                        np.logspace(-4, -3, num=40, endpoint=False),
                        np.logspace(-3, -2, num=60, endpoint=False),
                        np.logspace(-2, -1, num=80, endpoint=False),
                        np.logspace(-1, 0, num=100, endpoint=False),
                        np.logspace(0, 2, num=240, endpoint=True)])
                        #np.logspace(0, 2, num=240, endpoint=True)])
    return k


def get_default_s_callable():
    return np.logspace(-6., 2., 500)


def get_default_z_callable():
    return np.linspace(0., 10.**0.5, 30)**2  # approximates default class z


_default_extrap_kmin = 1e-7
_default_extrap_kmax = 1e2


def _pad_log(k, pk, extrap_kmin=_default_extrap_kmin, extrap_kmax=_default_extrap_kmax):
    f"""
    Pad ``pk`` and ``k`` in log10-log10-space between ``extrap_kmin`` and ``k[0]`` and ``k[-1]`` and ``extrap_kmax``.

    Parameters
    ----------
    k : array_like
        Wavenumbers.

    pk : array_like
        Power spectrum.

    extrap_kmin : float, default={_default_extrap_kmin}
        Minimum wavenumber of extrapolation range.

    extrap_kmax : float, default={_default_extrap_kmax}
        Maximum wavenumber of extrapolation range.

    Returns
    -------
    logk : array
        log10 of wavenumbers.

    logpk : array
        log10 of power spectrum.
    """
    jnp = numpy_jax(k, pk)
    logk = jnp.log10(k)
    logpk = jnp.log10(pk)
    log_extrap_kmin = jnp.log10(jnp.minimum(extrap_kmin, k[0] * (1 - 1e-9)))
    log_extrap_kmax = jnp.log10(jnp.maximum(extrap_kmax, k[-1] * (1 + 1e-9)))
    dtype = logpk.dtype

    dlogpkdlogk = (logpk[-1] - logpk[-2]) / (logk[-1] - logk[-2])
    padhighk = jnp.array([logk[-1] * 0.1 + log_extrap_kmax * 0.9, log_extrap_kmax], dtype=dtype)
    delta = [dlogpkdlogk * (padhighk[0] - logk[-1]), dlogpkdlogk * (padhighk[1] - logk[-1])]
    padhighpk = jnp.array([logpk[-1] + delta[0], logpk[-1] + delta[1]], dtype=dtype)

    dlogpkdlogk = (logpk[1] - logpk[0]) / (logk[1] - logk[0])
    padlowk = jnp.array([log_extrap_kmin, logk[0] * 0.1 + log_extrap_kmin * 0.9], dtype=dtype)
    delta = [dlogpkdlogk * (padlowk[0] - logk[0]), dlogpkdlogk * (padlowk[1] - logk[0])]
    padlowpk = jnp.array([logpk[0] + delta[0], logpk[0] + delta[1]], dtype=dtype)

    logk = jnp.concatenate([padlowk, logk, padhighk], axis=0)
    logpk = jnp.concatenate([padlowpk, logpk, padhighpk], axis=0)
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
    jnp = numpy_jax(x)
    return 3. * (jnp.sin(x) - x * jnp.cos(x)) / x**3


def kernel_tophat2(x):
    """Non-vectorized tophat function."""
    jnp = numpy_jax(x)
    x = jnp.asarray(x)
    mask = x < 0.1
    if x.size:
        #x = x.copy()
        #x = opmask(x, mask, _kernel_tophat_lowx(x[mask]**2)**2)
        #x = opmask(x, ~mask, _kernel_tophat_highx(x[~mask])**2)
        x = jnp.where(mask, _kernel_tophat_lowx(x**2), _kernel_tophat_highx(x))**2
        return x
    if mask: return _kernel_tophat_lowx(x**2)**2
    return _kernel_tophat_highx(x)**2


def integrate_sigma_d2(pk, kmin=1e-7, kmax=1e2, method='simpson', epsabs=1e-5, epsrel=1e-5, nk=None):
    r"""
    Return the variance of the displacement field, i.e.:

    .. math::

        \sigma_{d}^{2} = \frac{1}{6 \pi^{2}} \int dk P(k)

    Parameters
    ----------
    pk : callable
        Power spectrum.

    kmin : float, default=1e-7
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
    p = pk(kmin)
    pshape = p.shape
    jnp = numpy_jax(p)

    dtype = _bcast_dtype(p)
    if not p.size:
        return jnp.zeros(pshape, dtype=dtype)

    def integrand(logk):
        k = jnp.exp(logk)
        p = pk(k).reshape(k.shape + (-1,))
        return k[:, None] * p  # extra k factor because log integration

    limits = (jnp.log(kmin * (1. + 1e-9)), jnp.log(kmax * (1. - 1e-9)))  # to avoid nan's at the extremities (numerical inaccuracy)

    if method == 'quad':
        jnp = np

        def integrand(logk, i=None):
            k = jnp.exp(logk)
            p = pk(k)
            if i is not None: p = p[:, i]
            return k * p  # extra k factor because log integration

        if pshape:
            psize = pshape[0]
            tmp = [integrate.quad(integrand, *limits, args=(i,), epsabs=epsabs, epsrel=epsrel)[0] for i in range(psize)]
        else:
            tmp = integrate.quad(integrand, *limits, epsabs=epsabs, epsrel=epsrel)[0]

    elif method == 'romberg':
        tmp = romberg(integrand, *limits, epsabs=epsabs, epsrel=epsrel)
    elif method == 'leggauss':  # not accurate
        if nk is None: nk = 100
        x, wx = np.polynomial.legendre.leggauss(nk)
        logk = (limits[1] - limits[0]) / 2. * (1. + x) + limits[0]
        y = integrand(logk)
        w = (limits[1] - limits[0]) / 2. * wx
        tmp = jnp.sum(y * w[:, None], axis=0)
    elif method == 'simpson':  # accurate
        if nk is None: nk = 1024
        logk = jnp.linspace(*limits, nk)
        y = integrand(logk)
        tmp = simpson(y, x=logk, axis=0)
    tmp = jnp.asarray(tmp).reshape(pshape)
    sigmad2 = 1. / (6. * jnp.pi**2) * tmp
    return sigmad2.astype(dtype)


def integrate_sigma_r2(r, pk, kmin=1e-7, kmax=1e2, method='fftlog', epsabs=1e-5, epsrel=1e-5, nk=None, kernel=kernel_tophat2):
    r"""
    Return the variance of perturbations smoothed by a kernel :math:`W` of radius :math:`r`, i.e.:

    .. math::

        \sigma_{r}^{2} = \frac{1}{2 \pi^{2}} \int dk k^{2} P(k) W^{2}(kr)

    Parameters
    ----------
    r : float, array
        Smoothing radius.

    pk : callable
        Power spectrum.

    kmin : float, default=1e-7
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
    p = pk(kmin)
    pshape = p.shape
    jnp = numpy_jax(p)

    def integrand(logk):
        k = jnp.exp(logk)
        p = pk(k).reshape(k.shape + (-1,))
        # shape (k, r, z)
        # extra k factor because log integration
        return kernel(k[:, None] * r)[:, :, None] * (k[:, None]**3 * p)[:, None, :]

    dtype = _bcast_dtype(r, p if p.shape else None)
    r = jnp.array(r)
    rshape = r.shape
    if not p.size:
        return jnp.zeros(rshape + pshape, dtype=dtype)
    r = r.ravel()

    limits = (jnp.log(kmin * (1. + 1e-9)), jnp.log(kmax * (1. - 1e-9)))

    if method == 'quad':
        jnp = np

        def integrand(logk, r, i=None):
            k = jnp.exp(logk)
            p = pk(k)
            if i is not None: p = p[:, i]
            # shape (k, r, z)
            # extra k factor because log integration
            return kernel(k * r) * (k**3 * p)
        tmp = []
        for rr in r:
            if pshape:
                psize = pshape[0]
                tt = jnp.array([integrate.quad(integrand, *limits, args=(rr, i), epsabs=epsabs, epsrel=epsrel)[0] for i in range(psize)])
            else:
                tt = integrate.quad(integrand, *limits, args=(rr,), epsabs=epsabs, epsrel=epsrel)[0]
            tmp.append(tt)
    elif method == 'romberg':
        tmp = romberg(integrand, *limits, epsabs=epsabs, epsrel=epsrel)
    elif method == 'leggauss':  # not accurate
        if nk is None: nk = 100
        x, wx = np.polynomial.legendre.leggauss(nk)
        logk = (limits[1] - limits[0])/ 2. * (1. + x) + limits[0]
        y = integrand(logk)
        w = (limits[1] - limits[0]) / 2. * wx
        tmp = jnp.sum(y * w[:, None, None], axis=0)
    elif method == 'simpson':  # accurate
        if nk is None: nk = 1024
        logk = jnp.linspace(*limits, nk)
        y = integrand(logk)
        tmp = simpson(y, x=logk, axis=0)
    elif method == 'fftlog':
        if nk is None: nk = 1024
        k = jnp.geomspace(kmin, kmax, nk)
        s, var = TophatVariance(k)(pk(k).reshape(k.shape + (-1,)).T)
        tmp = (2. * np.pi**2) * Interpolator1D(s, var.T, assume_sorted=True)(r)
    tmp = jnp.asarray(tmp).reshape(rshape + pshape)
    sigmar2 = 1. / (2. * jnp.pi**2) * tmp
    return sigmar2.astype(dtype)


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
        default_params.pop(rm)
    return default_params


class _BasePowerSpectrumInterpolator(BaseClass):

    """Base class for power spectrum interpolators."""

    def _prepare(self, k, pk, z=None, interp_k='log', extrap_pk='log', extrap_kmin=_default_extrap_kmin, extrap_kmax=_default_extrap_kmax):
        self._np = numpy_jax(k, pk)
        self.k = self._np.asarray(k, dtype='f8').ravel()
        self._pk = self._np.asarray(pk, dtype='f8')
        if self._pk.ndim > 1 or z is not None:
            self._pk = self._pk.reshape(self.k.shape + (-1,))
        ix = self._np.argsort(self.k)
        self.k, self._pk = (xx[ix] for xx in (self.k, self._pk))
        if z is not None:
            self.z = self._np.asarray(z, dtype='f8').ravel()
            ix = self._np.argsort(self.z)
            self.z, self._pk = self.z[ix], self._pk[:, ix]
        self.interp_k = str(interp_k)
        self.extrap_pk = str(extrap_pk)
        k, pk = self.k, self._pk
        self.extrap_kmin, self.extrap_kmax = k[0], k[-1]
        if self.extrap_pk == 'log':
            if self.interp_k != 'log':
                raise ValueError('log-log extrapolation requires log-x interpolation')
            self.extrap_kmin, self.extrap_kmax = extrap_kmin, extrap_kmax
            k, pk = _pad_log(k, pk, extrap_kmin=extrap_kmin, extrap_kmax=extrap_kmax)
            k, pk = 10**k, 10**pk
        return k, pk

    def params(self):
        """Return interpolator parameter dictionary."""
        return {name: getattr(self, name) for name in self.default_params}

    def as_dict(self):
        """Return interpolator as a dictionary."""
        state = self.params()
        for name in ['k', 'pk']:
            state[name] = getattr(self, name)
        if hasattr(self, 'z'):
            state['z'] = self.z
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


from .jax import Interpolator1D, Interpolator2D


class PowerSpectrumInterpolator1D(_BasePowerSpectrumInterpolator):
    """
    1D power spectrum interpolator, broadly adapted from CAMB's P(k) interpolator by Antony Lewis
    in https://github.com/cmbant/CAMB/blob/master/camb/results.py, providing extra useful methods,
    such as :meth:`sigma_r` or :meth:`to_xi`.
    """

    def __init__(self, k, pk, interp_k='log', extrap_pk='log', extrap_kmin=_default_extrap_kmin, extrap_kmax=_default_extrap_kmax, interp_order_k=3):
        f"""
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

        extrap_kmin : float, default={_default_extrap_kmin}
            Minimum extrapolation range in ``k``.

        extrap_kmax : float, default={_default_extrap_kmax}
            Maximum extrapolation range in ``k``.

        interp_order_k : int, default=3
            Interpolation order, i.e. degree of smoothing spline along ``k``.
        """
        self._rsigma8sq = 1.
        k, pk = self._prepare(k, pk, interp_k=interp_k, extrap_pk=extrap_pk, extrap_kmin=extrap_kmin, extrap_kmax=extrap_kmax)
        self.interp_order_k = int(interp_order_k)
        _interp = Interpolator1D(k, pk, k=self.interp_order_k, interp_x=self.interp_k, interp_fun=self.extrap_pk, assume_sorted=True)
        self._interp = _interp
        self.is_from_callable = False

    default_params = _get_default_kwargs(__init__, start=3)

    @property
    def pk(self):
        """Return power spectrum array (if interpolator built from callable, evaluate it), with normalisation."""
        if self.is_from_callable:
            return self(self.k)
        return self._pk * self._rsigma8sq

    @classmethod
    def from_callable(cls, k=None, pk_callable=None, extrap_kmin=_default_extrap_kmin, extrap_kmax=_default_extrap_kmax):
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
        jnp = numpy_jax(k)
        self.k = jnp.sort(jnp.asarray(k, dtype='f8').ravel())
        self._np = numpy_jax(pk_callable(k[:1]))
        self.extrap_kmin, self.extrap_kmax = extrap_kmin, extrap_kmax
        #self.extrap_kmin, self.extrap_kmax = self._np.minimum(extrap_kmin, self.k[0]), self._np.maximum(extrap_kmax, self.k[-1])
        #self.extrap_kmin, self.extrap_kmax = self.kmin, self.kmax
        self.is_from_callable = True

        def interp(k, bounds_error=False, **kwargs):
            dtype = _bcast_dtype(k)
            k = self._np.asarray(k, dtype=dtype)
            toret_shape = k.shape
            k = k.ravel()
            mask_k, = _mask_bounds([k], [(self.extrap_kmin, self.extrap_kmax)], bounds_error=bounds_error)
            toret = self._np.where(mask_k, pk_callable(k, **kwargs), self._np.nan)
            return toret.astype(dtype).reshape(toret_shape)

        self._interp = interp

        return self

    def __call__(self, k, **kwargs):
        """
        Evaluate power spectrum at wavenumbers ``k``.

        Parameters
        ----------
        k : array_like
            Wavenumbers where to evaluate the power spectrum.

        islogk : bool, default=False
            Whether input ``k`` is already in log10-space.
        """
        return self._interp(k, **kwargs) * self._rsigma8sq

    def sigma_d(self, **kwargs):
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
        toret = integrate_sigma_d2(self, kmin=self.extrap_kmin, kmax=self.extrap_kmax, **kwargs)**0.5
        return toret

    def sigma_r(self, r, **kwargs):
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
            Array of shape ``(r.size,)``.
        """
        toret = integrate_sigma_r2(r, self, kmin=self.extrap_kmin, kmax=self.extrap_kmax, **kwargs)**0.5
        return toret.astype(_bcast_dtype(r))

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
        k = self._np.geomspace(self.extrap_kmin, self.extrap_kmax, nk)
        s, xi = PowerToCorrelation(k, complex=False, **(fftlog_kwargs or {}))(self(k).T)
        default_params = dict(interp_s='log', interp_order_s=self.interp_order_k)
        default_params.update(kwargs)
        return CorrelationFunctionInterpolator1D(s, xi=xi.T, **default_params)


class PowerSpectrumInterpolator2D(_BasePowerSpectrumInterpolator):
    """
    2D power spectrum interpolator, broadly adapted from CAMB's P(k) interpolator by Antony Lewis
    in https://github.com/cmbant/CAMB/blob/master/camb/results.py, providing extra useful methods,
    such as :meth:`sigma_rz` or :meth:`to_xi`.
    """

    def __init__(self, k, z, pk, interp_k='log', extrap_pk='log', extrap_kmin=_default_extrap_kmin, extrap_kmax=_default_extrap_kmax,
                 interp_order_k=3, interp_order_z=3, growth_factor_sq=None):
        rf"""
        Initialize :class:`PowerSpectrumInterpolator2D`.

        ``growth_factor_sq`` is a callable that can be prodided to rescale the output of the base spline interpolation.
        Indeed, variations of :math:`z \rightarrow P(k, z)` are (mostly) :math:`k` scale independent, such that more accurate interpolation in ``z``
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

        extrap_kmin : float, default={_default_extrap_kmin}
            Minimum extrapolation range in ``k``.

        extrap_kmax : float, default={_default_extrap_kmax}
            Maximum extrapolation range in ``k``.

        interp_order_k : int, default=3
            Interpolation order, i.e. degree of smoothing spline along ``k``.

        interp_order_z : int, default=None
            Interpolation order, i.e. degree of smoothing spline along ``z``.
            If ``None``, the maximum order given ``z`` size (see :meth:`GenericSpline.min_spline_order`) is considered.

        growth_factor_sq : callable, default=None
            Function that takes ``z`` as argument and returns the growth factor squared at that redshift.
            This will rescale the output of the base spline interpolation.
            Therefore, make sure that provided ``pk`` does not contain the redundant ``z`` variations.
        """
        self._rsigma8sq = 1.
        self.growth_factor_sq = growth_factor_sq
        k, pk = self._prepare(k, pk, z=z, interp_k=interp_k, extrap_pk=extrap_pk, extrap_kmin=extrap_kmin, extrap_kmax=extrap_kmax)
        self.interp_order_k, self.interp_order_z = int(interp_order_k), int(interp_order_z)
        is2d = self._pk.shape[1] > 1
        if is2d:
            _interp = Interpolator2D(k, self.z, pk, kx=self.interp_order_k, ky=self.interp_order_z, interp_x=self.interp_k, interp_fun=self.extrap_pk, assume_sorted=True)
        else:
            if self.growth_factor_sq is None:
                raise ValueError('provide either 2D pk array or growth_factor_sq')
            _interp = Interpolator1D(k, pk[:, 0], k=self.interp_order_k, interp_x=self.interp_k, interp_fun=self.extrap_pk, assume_sorted=True)

        self.is_from_callable = False

        def interp(k, z, grid=True, ignore_growth=False, bounds_error=False, **kwargs):
            dtype = _bcast_dtype(k, z)
            k, z = (self._np.asarray(xx, dtype=dtype) for xx in (k, z))
            if grid:
                toret_shape = k.shape + z.shape
            else:
                toret_shape = k.shape
            k, z = (xx.ravel() for xx in (k, z))
            mask_k, mask_z = _mask_bounds([k, z], [(self.extrap_kmin, self.extrap_kmax), (self.zmin, self.zmax)], bounds_error=bounds_error)
            if not is2d: mask_z |= True  # ignore input z
            if grid: mask_k = mask_k[:, None] & mask_z
            else: mask_k = mask_k & mask_z
            if is2d:
                tmp = _interp(k, z, grid=grid, **kwargs)
            else:
                tmp = _interp(k, **kwargs)
                if grid:
                    tmp = self._np.repeat(tmp[:, None], z.size, axis=-1)
            if self.growth_factor_sq is not None and not ignore_growth:
                tmp = tmp * self.growth_factor_sq(z).astype(dtype)
            toret = self._np.where(mask_k, tmp, self._np.nan)
            return toret.astype(dtype).reshape(toret_shape)

        self._interp = interp

    default_params = _get_default_kwargs(__init__, start=4)

    @property
    def pk(self):
        """Return power spectrum array (if interpolator built from callable, evaluate it), without growth factor, but with normalisation."""
        if self.is_from_callable:
            kwargs = {'ignore_growth': True} if self.growth_factor_sq is not None else {}
            return self(self.k, self.z, **kwargs)
        return self._pk * self._rsigma8sq

    @property
    def zmin(self):
        """Minimum (spline-interpolated) redshift."""
        return self.z[0]

    @property
    def zmax(self):
        """Maximum (spline-interpolated) redshift."""
        return self.z[-1]

    @classmethod
    def from_callable(cls, k=None, z=None, pk_callable=None, growth_factor_sq=None, extrap_kmin=_default_extrap_kmin, extrap_kmax=_default_extrap_kmax):
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
        jnp = numpy_jax(k, z)
        self.k, self.z = (jnp.sort(jnp.asarray(xx, dtype='f8').ravel()) for xx in (k, z))
        self.growth_factor_sq = growth_factor_sq
        self._np = numpy_jax(pk_callable(k[:1], z[:1]) if self.growth_factor_sq is None else pk_callable(k[:1]) * self.growth_factor_sq(z[:1]))
        self.extrap_kmin, self.extrap_kmax = extrap_kmin, extrap_kmax
        #self.extrap_kmin, self.extrap_kmax = self._np.minimum(extrap_kmin, self.k[0]), self._np.maximum(extrap_kmax, self.k[-1])
        #self.extrap_kmin, self.extrap_kmax = self.kmin, self.kmax
        self.is_from_callable = True

        def interp(k, z, grid=True, ignore_growth=False, bounds_error=False):
            dtype = _bcast_dtype(k, z)
            k, z = (self._np.asarray(xx, dtype=dtype) for xx in (k, z))
            if grid:
                toret_shape = k.shape + z.shape
            else:
                toret_shape = k.shape
            k, z = (xx.ravel() for xx in (k, z))
            mask_k, mask_z = _mask_bounds([k, z], [(self.extrap_kmin, self.extrap_kmax), (self.zmin, self.zmax)], bounds_error=bounds_error)
            if grid: mask_k = mask_k[:, None] & mask_z
            else: mask_k = mask_k & mask_z
            if self.growth_factor_sq is not None:
                tmp = pk_callable(k).astype(dtype)
                if not ignore_growth:
                    growth = self.growth_factor_sq(z).astype(dtype)
                else:
                    growth = 1.
                if grid:
                    tmp = tmp[..., None] * growth
                else:
                    tmp = tmp * growth
            else:
                tmp = pk_callable(k, z, grid=grid)
            toret = self._np.where(mask_k, tmp, self._np.nan)
            return toret.astype(dtype).reshape(toret_shape)

        self._interp = interp
        return self

    def __call__(self, k, z, grid=True, **kwargs):
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
        return self._interp(k, z=z, grid=grid, **kwargs) * self._rsigma8sq

    def sigma_dz(self, z, **kwargs):
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
        toret = integrate_sigma_d2(lambda k: self(k, z), kmin=self.extrap_kmin, kmax=self.extrap_kmax, **kwargs)**0.5
        return toret.astype(_bcast_dtype(z))

    def sigma_rz(self, r, z, **kwargs):
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
            Array of shape ``(r.size, z.size)``.
        """
        toret = integrate_sigma_r2(r, lambda k: self(k, z), kmin=self.extrap_kmin, kmax=self.extrap_kmax, **kwargs)**0.5
        return toret.astype(_bcast_dtype(r, z))

    def sigma8_z(self, z=0, **kwargs):
        """Return the r.m.s. of perturbations in a sphere of 8."""
        return self.sigma_rz(8., z=z, **kwargs)

    def rescale_sigma8(self, sigma8=1.):
        """Rescale power spectrum to the provided ``sigma8`` normalisation  at :math:`z = 0`."""
        self._rsigma8sq = 1.  # reset rsigma8 for interpolation
        self._rsigma8sq = sigma8**2 / self.sigma8_z(z=0)**2

    def growth_rate_rz(self, r, z, dz=1e-3, **kwargs):
        r"""
        Evaluate the growth rate at the log-derivative of perturbations in a sphere of :math:`r`, i.e.:

        .. math:

            f(r, z) = \frac{d ln \sigma_r(z)}{d ln a}

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
        r, z = (self._np.asarray(xx, dtype=dtype) for xx in (r, z))
        toret_shape = r.shape + z.shape
        if not all(toret_shape):
            return self._np.zeros(toret_shape, dtype=dtype)
        z = z.ravel()

        def finite_difference(fun):
            feval = [feval.reshape(-1, z.size) for feval in [fun(z - dz), fun(z - hdz), fun(z), fun(z + hdz), fun(z + dz)]]
            toret = self._np.where(z < self.zmin + hdz, -feval[4] + 4 * feval[3] - 3 * feval[2], feval[3] - feval[1])
            toret = self._np.where(z > self.zmax - hdz, -(-feval[0] + 4 * feval[1] - 3 * feval[2]), toret)
            return toret / dz

        dsigdz = finite_difference(lambda z: self._np.log(self.sigma_rz(r, z, **kwargs)))
        # a = 1/(1 + z) => da = -1/(1+z)^2 dz => dln(a) = -1/(1 + z) dz
        dsigdlna = -dsigdz * (1 + z)
        return dsigdlna.astype(dtype).reshape(toret_shape)

    def to_1d(self, z, **kwargs):
        """
        Return :class:`PowerSpectrumInterpolator1D` instance corresponding to interpolator at ``z``.

        Parameters
        ----------
        z : float, array
            Redshift.

        kwargs : dict
            Arguments for the new :class:`PowerSpectrumInterpolator1D` instance.
            By default, the new instance inherits the same ``k`` interpolattion and extrapolation settings.

        Returns
        -------
        interp : PowerSpectrumInterpolator1D
        """
        if self.is_from_callable:

            def pk_callable(k, **kwargs):
                return self._interp(k, z=z, **kwargs) * self._rsigma8sq

            return PowerSpectrumInterpolator1D.from_callable(self.k, pk_callable=pk_callable, extrap_kmin=self.extrap_kmin, extrap_kmax=self.extrap_kmax)
        default_params = dict(extrap_pk=self.extrap_pk, extrap_kmin=self.extrap_kmin, extrap_kmax=self.extrap_kmax, interp_order_k=self.interp_order_k)
        default_params.update(kwargs)
        self.extrap_kmin, self.extrap_kmax = -np.inf, np.inf  # in case self.k > self.extrap_kmax
        pk = self(self.k, z=z)
        self.extrap_kmin, self.extrap_kmax = default_params['extrap_kmin'], default_params['extrap_kmax']
        return PowerSpectrumInterpolator1D(self.k, pk, **default_params)

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
        k = self._np.geomspace(self.extrap_kmin, self.extrap_kmax, nk)
        s, xi = PowerToCorrelation(k, complex=False, **(fftlog_kwargs or {}))(self(k, z=self.z, ignore_growth=True).T)
        default_params = dict(interp_s='log', interp_order_s=self.interp_order_k,
                              interp_order_z=self.interp_order_z, growth_factor_sq=self.growth_factor_sq)
        default_params.update(kwargs)
        return CorrelationFunctionInterpolator2D(s, z=self.z, xi=xi.T, **default_params)


class _BaseCorrelationFunctionInterpolator(BaseClass):

    """Base class for correlation function interpolators."""

    def _prepare(self, s, xi, z=None, interp_s='log'):
        self._np = numpy_jax(s, xi)
        self.s = self._np.asarray(s, dtype='f8').ravel()
        self._xi = self._np.asarray(xi, dtype='f8')
        if self._xi.ndim > 1: self._xi = self._xi.reshape(self.s.shape + (-1,))
        ix = self._np.argsort(self.s)
        self.s, self._xi = (xx[ix] for xx in (self.s, self._xi))
        if z is not None:
            self.z = self._np.asarray(z, dtype='f8').ravel()
            ix = self._np.argsort(self.z)
            self.z, self._xi = self.z[ix], self._xi[:, ix]
        self.interp_s = str(interp_s)
        return self.s, self._xi

    def params(self):
        """Return interpolator parameter dictionary."""
        return {name: getattr(self, name) for name in self.default_params}

    def as_dict(self):
        """Return interpolator as a dictionary."""
        state = self.params()
        for name in ['s', 'xi']:
            state[name] = getattr(self, name)
        if hasattr(self, 'z'):
            state['z'] = self.z
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
        s, xi = self._prepare(s, xi, interp_s=interp_s)
        self.interp_order_s = int(interp_order_s)
        _interp = Interpolator1D(s, xi, k=self.interp_order_s, interp_x=self.interp_s)
        self._interp = _interp
        self.is_from_callable = False

    default_params = _get_default_kwargs(__init__, start=3)

    @property
    def xi(self):
        """Return correlation function array (if interpolator built from callable, evaluate it), with normalisation."""
        if self.is_from_callable:
            return self(self.s)
        return self._xi * self._rsigma8sq

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
        jnp = numpy_jax(s) 
        self.s = jnp.sort(jnp.asarray(s, dtype='f8').ravel())
        self._np = numpy_jax(xi_callable(s[:1]))

        def interp(s, bounds_error=False, **kwargs):
            dtype = _bcast_dtype(s)
            s = self._np.asarray(s, dtype=dtype)
            toret_shape = s.shape
            s = s.ravel()
            mask_s, = _mask_bounds([s], [(self.smin, self.smax)], bounds_error=bounds_error)
            toret = self._np.where(mask_s, xi_callable(s, **kwargs), self._np.nan)
            return toret.astype(dtype).reshape(toret_shape)

        self._interp = interp
        return self

    def __call__(self, s, **kwargs):
        """
        Evaluate correlation function at separations ``s``.

        Parameters
        ----------
        s : array_like
            Separations where to evaluate the correlation function.

        islogs : bool, default=False
            Whether input ``s`` is already in log10-space.
        """
        return self._interp(s, **kwargs) * self._rsigma8sq

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
        s = self._np.geomspace(self.extrap_smin, self.extrap_smax, ns)
        k, pk = CorrelationToPower(s, complex=False, **(fftlog_kwargs or {}))(self(s))
        default_params = dict(interp_k='log', interp_order_k=self.interp_order_s)
        default_params.update(kwargs)
        return PowerSpectrumInterpolator1D(k, pk=pk, **default_params)


class CorrelationFunctionInterpolator2D(_BaseCorrelationFunctionInterpolator):

    """2D correlation function interpolator."""

    def __init__(self, s, z, xi=None, interp_s='log', interp_order_s=3, interp_order_z=None, growth_factor_sq=None):
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

        growth_factor_sq : callable, default=None
            Function that takes ``z`` as argument and returns the growth factor squared at that redshift.
            This will rescale the output of the base spline interpolation.
            Therefore, make sure that provided ``pk`` does not contain the redundant ``z`` variations.
        """
        self._rsigma8sq = 1.
        self.growth_factor_sq = growth_factor_sq
        s, xi = self._prepare(s, xi, z=z, interp_s=interp_s)

        self.interp_order_s, self.interp_order_z = int(interp_order_s), int(interp_order_z)
        is2d = self._xi.shape[1] > 1
        if is2d:
            _interp = Interpolator2D(s, self.z, xi, kx=self.interp_order_s, ky=self.interp_order_z, interp_x=self.interp_s, assume_sorted=True)
        else:
            if self.growth_factor_sq is None:
                raise ValueError('provide either 2D pk array or growth_factor_sq')
            _interp = Interpolator1D(s, xi[:, 0], k=self.interp_order_s, interp_x=self.interp_s, assume_sorted=True)

        self.is_from_callable = False

        def interp(s, z, grid=True, ignore_growth=False, bounds_error=False, **kwargs):
            dtype = _bcast_dtype(s, z)
            s, z = (self._np.asarray(xx, dtype=dtype) for xx in (s, z))
            if grid:
                toret_shape = s.shape + z.shape
            else:
                toret_shape = s.shape
            s, z = (xx.ravel() for xx in (s, z))
            mask_s, mask_z = _mask_bounds([s, z], [(self.smin, self.smax), (self.zmin, self.zmax)], bounds_error=bounds_error)
            if not is2d: mask_z |= True  # ignore input z
            if grid: mask_s = mask_s[:, None] & mask_z
            else: mask_s = mask_s & mask_z
            if is2d:
                tmp = _interp(s, z, grid=grid, **kwargs)
            else:
                tmp = _interp(s, **kwargs)
                if grid:
                    tmp = self._np.repeat(tmp[:, None], z.size, axis=-1)
            if self.growth_factor_sq is not None and not ignore_growth:
                tmp = tmp * self.growth_factor_sq(z).astype(dtype)
            toret = self._np.where(mask_s, tmp, self._np.nan)
            return toret.astype(dtype).reshape(toret_shape)

        self._interp = interp

    default_params = _get_default_kwargs(__init__, start=4)

    @property
    def xi(self):
        """Return ``xi`` (if interpolator built from callable, evaluate it), without growth factor, but with normalisation."""
        if self.is_from_callable:
            growth_factor_sq = self.growth_factor_sq
            self.growth_factor_sq = lambda x: jnp.ones_like(x)  # to avoid using growth factor in following call
            toret = self(self.s, self.z)
            self.growth_factor_sq = growth_factor_sq
            return toret
        return self._xi * self._rsigma8sq

    @property
    def zmin(self):
        """Minimum (spline-interpolated) redshift."""
        return self.z[0]

    @property
    def zmax(self):
        """Maximum (spline-interpolated) redshift."""
        return self.z[-1]

    def __call__(self, s, z, grid=True, **kwargs):
        """
        Evaluate correlation function at separations ``s`` and redshifts ``z``.

        Parameters
        ----------
        s : array_like
            Separations where to evaluate the correlation function.

        z : array_like
            Redshifts where to evaluate the correlation function.

        grid : bool, default=True
            Whether ``s``, ``z`` coordinates should be interpreted as a grid, in which case the output will be of shape ``(s.size, z.size)``.

        islogs : bool, default=False
            Whether input ``s`` is already in log10-space.

        ignore_growth : bool, default=False
            Whether to ignore multiplication by growth function (if provided).
        """
        return self._interp(s, z, grid=grid, **kwargs) * self._rsigma8sq

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
        jnp = numpy_jax(s, z)
        self.s, self.z = (jnp.sort(jnp.asarray(xx, dtype='f8').ravel()) for xx in (s, z))
        self.growth_factor_sq = growth_factor_sq
        self._np = numpy_jax(xi_callable(s[:1], z[:1]) if self.growth_factor_sq is None else xi_callable(s[:1]) * self.growth_factor_sq(z[:1]))
        self.is_from_callable = True

        def interp(s, z, grid=True, ignore_growth=False, bounds_error=False):
            dtype = _bcast_dtype(s, z)
            s, z = (self._np.asarray(xx, dtype=dtype) for xx in (s, z))
            if grid:
                toret_shape = s.shape + z.shape
            else:
                toret_shape = s.shape
            s, z = (xx.ravel() for xx in (s, z))
            mask_s, mask_z = _mask_bounds([s, z], [(self.smin, self.smax), (self.zmin, self.zmax)], bounds_error=bounds_error)
            if grid: mask_s = mask_s[:, None] & mask_z
            else: mask_s = mask_s & mask_z

            if self.growth_factor_sq is not None:
                tmp = xi_callable(s).astype(dtype)
                if not ignore_growth:
                    growth = self.growth_factor_sq(z).astype(dtype)
                else:
                    growth = 1.
                if grid:
                    tmp = tmp[..., None] * growth
                else:
                    tmp = tmp * growth
            else:
                tmp = xi_callable(s, z, grid=grid)
            toret = self._np.where(mask_s, tmp, self._np.nan)
            return toret.astype(dtype).reshape(toret_shape)

        self._interp = interp
        return self

    def sigma_dz(self, z, **kwargs):
        """
        Return the r.m.s. of the displacement field by transforming correlation function into power spectrum.

        See :meth:`PowerSpectrumInterpolator2D.sigma_dz` arguments.
        """
        return self.to_pk().sigma_dz(z=z, **kwargs)

    def sigma_rz(self, r, z, **kwargs):
        """
        Return the r.m.s. of perturbations in a sphere of :math:`r` by transforming correlation function into power spectrum.

        See :meth:`PowerSpectrumInterpolator2D.sigma_rz` arguments.
        """
        return self.to_pk().sigma_rz(r, z=z, **kwargs)

    def sigma8_z(self, z, **kwargs):
        """
        Return the r.m.s. of perturbations in a sphere of 8 by transforming correlation function into power spectrum.

        See :meth:`PowerSpectrumInterpolator2D.sigma8_z` arguments.
        """
        return self.sigma_rz(8., z=z, **kwargs)

    def rescale_sigma8(self, sigma8=1.):
        """Rescale correlation function to the provided ``sigma8`` normalisation  at :math:`z = 0`."""
        self._rsigma8sq = 1.  # reset rsigma8 for interpolation
        self._rsigma8sq = sigma8**2 / self.sigma8_z(z=0)**2

    def growth_rate_rz(self, r, z, **kwargs):
        r"""
        Evaluate the growth rate at the log-derivative of perturbations in a sphere of :math:`r` by transforming correlation function into power spectrum.

        See :meth:`PowerSpectrumInterpolator2D.growth_rate_rz` arguments.
        """
        return self.to_pk().growth_rate_rz(r, z=z, **kwargs)

    def to_1d(self, z, **kwargs):
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
            return CorrelationFunctionInterpolator1D.from_callable(self.s, xi_callable=lambda s: self._interp(s, z=z) * self._rsigma8sq)
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
        s = self._np.geomspace(self.extrap_smin, self.extrap_smax, ns)
        k, pk = CorrelationToPower(s, complex=False, **(fftlog_kwargs or {}))(self(s, self.z, ignore_growth=True).T)
        default_params = dict(interp_k='log', extrap_pk='log', interp_order_k=self.interp_order_s,
                              interp_order_z=self.interp_order_z, growth_factor_sq=self.growth_factor_sq)
        default_params.update(kwargs)
        return PowerSpectrumInterpolator2D(k, z=self.z, pk=pk.T, **default_params)
