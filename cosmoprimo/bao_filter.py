"""
Different techniques to extra BAO from the power spectrum of correlation function.
For the power spectrum, the most accurate one is :class:`Wallish2018PowerSpectrumBAOFilter`.
For the correlation function: :class:`Kirkby2013CorrelationFunctionBAOFilter`.
"""

import numpy as np

from .interpolator import PowerSpectrumInterpolator2D, CorrelationFunctionInterpolator2D
from .utils import BaseClass, SolveLeastSquares
from .cosmology import Cosmology, Fourier


class RegisteredPowerSpectrumBAOFilter(type(BaseClass)):

    """Metaclass registering :class:`BasePowerSpectrumBAOFilter`-derived classes."""

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls
        return cls


class BasePowerSpectrumBAOFilter(BaseClass, metaclass=RegisteredPowerSpectrumBAOFilter):

    """Base BAO filter for power spectrum."""
    name = 'base'

    def __init__(self, pk_interpolator, cosmo=None, **kwargs):
        """
        Run BAO filter.

        Parameters
        ----------
        pk_interpolator : PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D
            Input power spectrum to remove BAO wiggles from.

        cosmo : Cosmology, default=None
            Cosmology instance, which may be used to tune filter settings (depending on ``rs_drag``).

        kwargs : dict
            Arguments for :meth:`set_k`.
        """
        self.pk_interpolator = pk_interpolator
        self.is2d = isinstance(pk_interpolator, PowerSpectrumInterpolator2D)
        self._cosmo = cosmo
        self.set_k(**kwargs)
        if self.is2d:
            self.pk = self.pk_interpolator(self.k, self.pk_interpolator.z, ignore_growth=True)
        else:
            self.pk = self.pk_interpolator(self.k)
        self.compute()

    def set_k(self, nk=1024):
        """
        Set wavenumbers where to evaluate the power spectrum (between :attr:`pk_interpolator.extrap_kmin` and :attr:`pk_interpolator.extrap_kmax`).

        Parameters
        ----------
        nk : int, default=1024
            Number of wavenumbers.
        """
        self.k = np.geomspace(self.pk_interpolator.extrap_kmin, self.pk_interpolator.extrap_kmax, nk)

    @property
    def wiggles(self):
        """Extracted wiggles."""
        return self.pk / self.pknow

    def smooth_pk_interpolator(self, **kwargs):
        """
        Return smooth (i.e. no-wiggle) power spectrum.

        Parameters
        ----------
        kwargs : dict
            Override interpolation and extrapolation settings of :attr:`pk_interpolator`.

        Returns
        -------
        interp : PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D
            1D or 2D depending on :attr:`pk_interpolator`.
        """
        return self.pk_interpolator.clone(k=self.k, pk=self.pknow, **kwargs)

    def smooth_xi_interpolator(self, **kwargs):
        """
        Return smooth (i.e. no-peak) correlation function using :class:`FFTlog`.

        Parameters
        ----------
        kwargs : dict
            Override interpolation and extrapolation settings of returned correlation function interpolator.

        Returns
        -------
        interp : CorrelationFunctionInterpolator1D, CorrelationFunctionInterpolator2D
            1D or 2D depending on :attr:`pk_interpolator`.
        """
        return self.smooth_pk_interpolator().to_xi(**kwargs)

    @property
    def cosmo(self):
        """Cosmology."""
        if self._cosmo is None:
            self._cosmo = Cosmology()
        return self._cosmo

    def rs_drag_ratio(self):
        """If :attr:`cosmo` is provided, return the ratio of its ``rs_drag`` to the fiducial one (from ``Cosmology()``), else 1."""
        if self._cosmo is None:
            return 1.
        return self.cosmo.get_thermodynamics().rs_drag / 100.91463132327911


class Hinton2017PowerSpectrumBAOFilter(BasePowerSpectrumBAOFilter):
    """
    Power spectrum BAO filter consisting in fitting a high degree polynomial to the input power spectrum in log-log space.

    References
    ----------
    https://github.com/Samreay/Barry/blob/master/barry/cosmology/power_spectrum_smoothing.py
    """
    name = 'hinton2017'

    def __init__(self, pk_interpolator, degree=13, sigma=1, weight=0.5, **kwargs):
        """
        Run BAO filter.

        Parameters
        ----------
        pk_interpolator : PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D
            Input power spectrum to remove BAO wiggles from.

        degree : int, default=13
            Polynomial degree.

        sigma : float, default=1
            Standard deviation of the Gaussian kernel that downweights the maximum of the power spectrum relative to the edges.

        weight : float, default=0.5
            Normalisation of the Gaussian kernel.
        """
        self.degree = degree
        self.sigma = sigma
        self.weight = weight
        super(Hinton2017PowerSpectrumBAOFilter, self).__init__(pk_interpolator, **kwargs)

    def compute(self):
        """Run filter."""
        logk = np.log(self.k)
        logpk = np.log(self.pk)
        maxk = logk[np.argmax(self.pk, axis=0)].flat[0]  # here we take just the first one, approximation
        gauss = np.exp(-0.5 * ((logk - maxk) / self.sigma)**2)
        w = np.ones_like(self.k) - self.weight * gauss

        gradient = np.array([logk**i for i in range(self.degree)])
        sls = SolveLeastSquares(gradient, precision=1. / w**2)
        sls(logpk.T)
        self.pknow = np.exp(sls.model()).T
        # series = np.polynomial.polynomial.Polynomial.fit(logk, logpk, self.degree, w=w)
        # self.pknow = np.exp(series(logk))


class SavGolPowerSpectrumBAOFilter(BasePowerSpectrumBAOFilter):
    r"""
    BAO smoothing with Savitzky-Golay filter.

    References
    ----------
    Taken from https://github.com/sfschen/velocileptors/blob/master/velocileptors/EPT/ept_fullresum_fftw.py

    Note
    ----
    Contrary to the reference, we work in :math:`\log(k)` - :math:`\log(k P(k))` space.
    """
    name = 'savgol'

    def compute(self):
        """Run filter."""
        from scipy.signal import savgol_filter
        # empirical setting of https://github.com/sfschen/velocileptors/blob/master/velocileptors/EPT/cleft_kexpanded_resummed_fftw.py#L37
        nfilter = int(np.ceil(np.log(7) / np.log(self.k[-1] / self.k[-2])) // 2 * 2 + 1)  # filter length ~ log span of one oscillation from k = 0.01
        # self.pknow = np.exp(savgol_filter(np.log(self.pk),nfilter,polyorder=4,axis=0))
        self.pknow = (np.exp(savgol_filter(np.log(self.k * self.pk.T), nfilter, polyorder=4, axis=-1)) / self.k).T
        hnfilter = nfilter // 2
        self.pknow[-hnfilter:] = self.pk[-hnfilter:]


class EHNoWigglePolyPowerSpectrumBAOFilter(BasePowerSpectrumBAOFilter):

    """Remove BAO wiggles using the Eisenstein & Hu no-wiggle analytic formula, emulated with a 6-th order polynomial."""
    name = 'ehpoly'

    def __init__(self, pk_interpolator, kbox=(5e-3, 0.5), dampkbox=(1e-2, 0.4), dampsigma=10, rescale_kbox=True, cosmo=None, **kwargs):
        """
        Run BAO filter.

        Parameters
        ----------
        pk_interpolator : PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D
            Input power spectrum to remove BAO wiggles from.

        kbox : tuple, default=(5e-3, 0.5)
            k-range to fit the Eisenstein & Hu no-wiggle power spectrum to the input one :attr:`pk_interpolator`.

        dampkbox : tuple, default=(1e-2, 0.4)
            k-range to interpolate between the Eisenstein & Hu no-wiggle power spectrum and the input one :attr:`pk_interpolator`
            with an Gaussian damping factor.

        dampsigma : float, default=10
            Standard deviation of the Gaussian damping factor.

        rescale_kbox : bool, default=True
            Whether to rescale ``kbox`` and ``dampkbox`` by the ratio of ``rs_drag`` relative to the fiducial cosmology
            (may help robustify the procedure for cosmologies far from the fiducial one).

        cosmo : Cosmology, default=None
            Cosmology instance, used to compute the Eisenstein & Hu no-wiggle power spectrum.

        kwargs : dict
            Arguments for :meth:`set_k`.
        """
        self.kbox = kbox
        self.dampkbox = dampkbox
        self.dampsigma = dampsigma
        self.rescale_kbox = rescale_kbox
        super(EHNoWigglePolyPowerSpectrumBAOFilter, self).__init__(pk_interpolator, cosmo=cosmo, **kwargs)

    def compute(self):
        """Run filter."""
        kbox, dampkbox = np.asarray(self.kbox), np.asarray(self.dampkbox)
        if self.rescale_kbox:
            scale = self.rs_drag_ratio()
            kbox, dampkbox = kbox / scale, dampkbox / scale
        pknow = Fourier(self.cosmo, engine='eisenstein_hu_nowiggle', set_engine=False).pk_interpolator(ignore_norm=True)(self.k)
        ratio = self.pk.T / pknow
        mask = (self.k >= kbox[0]) & (self.k <= kbox[1])
        k = self.k[mask]

        def model(k):
            return np.array([k**(i - 1) for i in range(6)])

        sls = SolveLeastSquares(model(k), precision=1.)
        params = sls(ratio[..., mask])

        tophat = self._tophat(self.k, kmin=dampkbox[0], kmax=dampkbox[-1], scale=self.dampsigma)
        model = ratio / params.dot(model(self.k))
        wiggles = (model - 1.) * tophat + 1.
        self.pknow = self.pk / wiggles.T

    @staticmethod
    def _tophat(k, kmin=1e-3, kmax=1, scale=1):
        """Tophat Gaussian kernel."""
        tophat = np.ones_like(k)
        mask = k > kmax
        tophat[mask] *= np.exp(-scale**2 * (k[mask] / kmax - 1.)**2)
        mask = k < kmin
        tophat[mask] *= np.exp(-scale**2 * (kmin / k[mask] - 1.)**2)
        return tophat


class Wallish2018PowerSpectrumBAOFilter(BasePowerSpectrumBAOFilter):
    """
    Filter BAO wiggles by sine-transforming the power spectrum to real space (where the BAO is better localised),
    cutting the peak and interpolating with a spline.

    References
    ----------
    https://arxiv.org/pdf/1810.02800.pdf, Appendix D (thanks to Stephen Chen for the reference)

    Note
    ----
    We have hand-tune parameters w.r.t. to the reference.
    """
    name = 'wallish2018'

    def compute(self):
        """Run filter."""
        from scipy import fftpack, interpolate
        k = np.linspace(self.pk_interpolator.extrap_kmin, 2., 4096)
        if self.is2d:
            pk = self.pk_interpolator(k, self.pk_interpolator.z, ignore_growth=True)
        else:
            pk = self.pk_interpolator(k)
        kpk = np.log(k * pk.T).T
        kpkffted = fftpack.dst(kpk, type=2, axis=0, norm='ortho', overwrite_x=False)
        even = kpkffted[::2]
        odd = kpkffted[1::2]

        xeven, xodd = 1 + np.arange(even.shape[0]), 1 + np.arange(odd.shape[0])
        spline_even = interpolate.CubicSpline(xeven, even, axis=0, bc_type='clamped', extrapolate=False)
        # dd_even = ndimage.uniform_filter1d(spline_even(xeven,nu=2), 3, axis=0, mode='reflect')
        dd_even = spline_even(xeven, nu=2)
        spline_odd = interpolate.CubicSpline(xodd, odd, axis=0, bc_type='clamped', extrapolate=False)
        # dd_odd = ndimage.uniform_filter1d(spline_odd(xodd,nu=2), 3, axis=0, mode='reflect')
        dd_odd = spline_odd(xodd, nu=2)
        self._even = even  # in case one wants to check everything is ok
        self._odd = odd
        self._dd_even = dd_even
        self._dd_odd = dd_odd
        margin_first = 20
        margin_second = 5
        offset_even = offset_odd = (-10, 20)

        def smooth_even_odd(even, odd, dd_even, dd_odd):
            argmax_even = dd_even[margin_first:].argmax() + margin_first
            argmax_odd = dd_odd[margin_first:].argmax() + margin_first
            ibox_even = (argmax_even + offset_even[0], argmax_even + margin_second + dd_even[argmax_even + margin_first:].argmax() + offset_even[1])
            ibox_odd = (argmax_odd + offset_odd[0], argmax_odd + margin_second + dd_odd[argmax_odd + margin_second:].argmax() + offset_odd[1])
            mask_even = np.ones_like(even, dtype=np.bool_)
            mask_even[ibox_even[0]:ibox_even[1] + 1] = False
            mask_odd = np.ones_like(odd, dtype=np.bool_)
            mask_odd[ibox_odd[0]:ibox_odd[1] + 1] = False
            spline_even = interpolate.CubicSpline(xeven[mask_even], even[mask_even] * xeven[mask_even]**2, axis=-1, bc_type='clamped', extrapolate=False)
            spline_odd = interpolate.CubicSpline(xodd[mask_odd], odd[mask_odd] * xodd[mask_odd]**2, axis=-1, bc_type='clamped', extrapolate=False)
            return spline_even(xeven) / xeven**2, spline_odd(xodd) / xodd**2

        if self.is2d:
            for iz in range(self.pk.shape[-1]):
                even[:, iz], odd[:, iz] = smooth_even_odd(even[:, iz], odd[:, iz], dd_even[:, iz], dd_odd[:, iz])
        else:
            even, odd = smooth_even_odd(even, odd, dd_even, dd_odd)

        self._even_now = even
        self._odd_now = odd
        merged = np.empty_like(kpkffted)
        merged[::2] = even
        merged[1::2] = odd
        kpknow = fftpack.idst(merged, type=2, axis=0, norm='ortho', overwrite_x=False)
        pknow = (np.exp(kpknow).T / k).T

        mask = (k > 1e-2) & (k < 1.5)
        k, pknow = k[mask], pknow[mask]
        mask_left, mask_right = self.k < 5e-4, self.k > 2.
        k = np.concatenate([self.k[mask_left], k, self.k[mask_right]])
        pknow = np.concatenate([self.pk[mask_left], pknow, self.pk[mask_right]])
        pknow = interpolate.CubicSpline(k, pknow, axis=0, bc_type='clamped', extrapolate=False)(self.k)
        tophat = self._tophat(self.k, kmax=1., scale=20.)
        wiggles = (self.pk / pknow - 1.).T * tophat + 1.
        self.pknow = self.pk / wiggles.T

    @staticmethod
    def _tophat(k, kmax=1, scale=1):
        """Tophat Gaussian kernel."""
        tophat = np.ones_like(k)
        mask = k > kmax
        tophat[mask] *= np.exp(-scale**2 * (k[mask] / kmax - 1.)**2)
        return tophat


class RegisteredCorrelationFunctionBAOFilter(type(BaseClass)):

    """Metaclass registering :class:`BaseCorrelationFunctionBAOFilter`-derived classes."""

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls
        return cls


class BaseCorrelationFunctionBAOFilter(BaseClass, metaclass=RegisteredCorrelationFunctionBAOFilter):

    """Base BAO filter for correlation function."""
    name = 'base'

    def __init__(self, xi_interpolator, cosmo=None, **kwargs):
        """
        Run BAO filter.

        Parameters
        ----------
        xi_interpolator : CorrelationFunctionInterpolator1D, CorrelationFunctionInterpolator2D
            Input correlation function to remove BAO peak from.

        cosmo : Cosmology, default=None
            Cosmology instance, which may be used to tune filter settings (depending on ``rs_drag``).

        kwargs : dict
            Arguments for :meth:`set_s`.
        """
        self.xi_interpolator = xi_interpolator
        self.is2d = isinstance(xi_interpolator, CorrelationFunctionInterpolator2D)
        self._cosmo = cosmo
        self.set_s(**kwargs)
        if self.is2d:
            self.xi = self.xi_interpolator(self.s, self.xi_interpolator.z, ignore_growth=True)
        else:
            self.xi = self.xi_interpolator(self.s)
        self.compute()

    def set_s(self, ns=1024):
        """
        Set separations where to evaluate the correlation function (between :attr:`xi_interpolator.extrap_smin` and :attr:`xi_interpolator.extrap_smax`).

        Parameters
        ----------
        ns : int, default=1024
            Number of separations.
        """
        self.s = np.geomspace(self.xi_interpolator.extrap_smin, self.xi_interpolator.extrap_smax, ns)

    def smooth_xi_interpolator(self, **kwargs):
        """
        Return smooth (i.e. no-peak) correlation function.

        Parameters
        ----------
        kwargs : dict
            Override interpolation settings of :attr:`xi_interpolator`.

        Returns
        -------
        interp : CorrelationFunctionInterpolator1D, CorrelationFunctionInterpolator2D
            1D or 2D depending on :attr:`xi_interpolator`.
        """
        return self.xi_interpolator.clone(s=self.s, xi=self.xinow, **kwargs)

    def smooth_pk_interpolator(self, **kwargs):
        """
        Return smooth (i.e. no-wiggle) power spectrum using :class:`FFTlog`.

        Parameters
        ----------
        kwargs : dict
            Override interpolation and extrapolation settings of return power spectrum interpolator.

        Returns
        -------
        interp : PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D
            1D or 2D depending on :attr:`pk_interpolator`.
        """
        return self.smooth_xi_interpolator().to_pk(**kwargs)

    @property
    def cosmo(self):
        """Cosmology."""
        if self._cosmo is None:
            self._cosmo = Cosmology()
        return self._cosmo

    def rs_drag_ratio(self):
        """If :attr:`cosmo` is provided, return the ratio of its ``rs_drag`` to the fiducial one (from ``Cosmology()``), else 1."""
        if self._cosmo is None:
            return 1.
        return self.cosmo.get_thermodynamics().rs_drag / 100.91463132327911


class Kirkby2013CorrelationFunctionBAOFilter(BaseCorrelationFunctionBAOFilter):
    """
    Filter BAO peak by cutting the peak and interpolating with 5-order polynomial.

    References
    ----------
    https://arxiv.org/abs/1301.3456
    https://github.com/igmhub/picca/blob/master/bin/picca_compute_pk_pksb.py
    """
    name = 'kirkby2013'

    def __init__(self, xi_interpolator, sbox_left=(50., 82.), sbox_right=(150., 190.), rescale_sbox=True, cosmo=None, **kwargs):
        """
        Run BAO filter.

        Parameters
        ----------
        xi_interpolator : CorrelationFunctionInterpolator1D, CorrelationFunctionInterpolator2D
            Input correlation function to remove BAO peak from.

        sbox_left : tuple
            s-range to fit the polynomial on the left-hand side of the BAO peak.

        sbox_right : tuple
            s-range to fit the polynomial on the right-hand side of the BAO peak.

        cosmo : Cosmology
            Cosmology instance, which may be used to tune filter settings (depending on ``rs_drag``).

        rescale_sbox : bool
            Whether to rescale ``sbox_left`` and ``sbox_right`` by the ratio of ``rs_drag`` relative to the fiducial cosmology
            (may help robustify the procedure for cosmologies far from the fiducial one).

        cosmo : Cosmology
            Cosmology instance, used to compute the Eisenstein & Hu no-wiggle power spectrum.

        kwargs : dict
            Arguments for :meth:`set_s`.
        """
        self.sbox_left = sbox_left
        self.sbox_right = sbox_right
        self.rescale_sbox = rescale_sbox
        super(Kirkby2013CorrelationFunctionBAOFilter, self).__init__(xi_interpolator, cosmo=cosmo, **kwargs)

    def compute(self):
        """Run filter."""
        sbox_left, sbox_right = np.asarray(self.sbox_left), np.asarray(self.sbox_right)
        if self.rescale_sbox:
            scale = self.rs_drag_ratio()
            sbox_left, sbox_right = sbox_left * scale, sbox_right * scale

        mask = ((self.s >= sbox_left[0]) & (self.s <= sbox_left[1])) | ((self.s >= sbox_right[0]) & (self.s <= sbox_right[1]))

        def model(s):
            return np.array([s**(1 - i) for i in range(5)])

        sls = SolveLeastSquares(model(self.s[mask]), precision=1.)
        params = sls(self.xi[mask].T)
        mask = (self.s > sbox_left[1]) & (self.s < sbox_right[0])
        self.xinow = self.xi.copy()
        self.xinow[mask] = params.dot(model(self.s[mask])).T


def PowerSpectrumBAOFilter(pk_interpolator, engine='wallish2018', **kwargs):
    """Run power spectrum BAO filter corresponding to the provided engine."""

    engine = engine.lower()
    try:
        engine = BasePowerSpectrumBAOFilter._registry[engine]
    except KeyError:
        raise ValueError('Power spectrum BAO filter {} is unknown'.format(engine))

    return engine(pk_interpolator, **kwargs)


def CorrelationFunctionBAOFilter(xi_interpolator, engine='kirkby2013', **kwargs):
    """Run correlation function BAO filter corresponding to the provided engine."""

    engine = engine.lower()
    try:
        engine = BaseCorrelationFunctionBAOFilter._registry[engine]
    except KeyError:
        raise ValueError('Correlation function BAO filter {} is unknown'.format(engine))

    return engine(xi_interpolator, **kwargs)
