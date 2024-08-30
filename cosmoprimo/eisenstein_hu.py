import warnings

import numpy as np

from .cosmology import BaseEngine, BaseSection, DefaultBackground, CosmologyError
from .interpolator import PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D
from . import constants, utils
from .jax import exception


class EisensteinHuEngine(BaseEngine):
    """
    Implementation of Eisenstein & Hu analytic formulae.

    References
    ----------
    https://arxiv.org/abs/astro-ph/9709112
    """
    name = 'eisenstein_hu'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        def raise_error(N_ncdm, Omega_k, has_fld):
            if N_ncdm:
                warnings.warn('{} cannot cope with massive neutrinos'.format(self.__class__.__name__))
            if Omega_k != 0.:
                warnings.warn('{} cannot cope with non-zero curvature'.format(self.__class__.__name__))
            if has_fld:
                warnings.warn('{} cannot cope with non-constant dark energy'.format(self.__class__.__name__))
        exception(raise_error, self['N_ncdm'], self['Omega_k'], self._has_fld)
        self.compute()
        self._A_s = self._get_A_s_fid()

    def _set_rsdrag(self):
        """Set sound horizon at the drag epoch."""

        self.omega_b = self['omega_b']
        self.omega_m = self['omega_cdm'] + self['omega_b']
        self.frac_b = self.omega_b / self.omega_m
        self.theta_cmb = self['T_cmb'] / 2.7

        # redshift and wavenumber of equality
        # EH eq. 2 & 3
        self.z_eq = 2.5e4 * self.omega_m * self.theta_cmb ** (-4) - 1.  # this is z
        self.k_eq = 0.0746 * self.omega_m * self.theta_cmb ** (-2)  # units of 1/Mpc

        # sound horizon and k_silk
        # EH eq. 4
        z_drag_b1 = 0.313 * self.omega_m ** (-0.419) * (1 + 0.607 * self.omega_m ** 0.674)
        z_drag_b2 = 0.238 * self.omega_m ** 0.223
        # self.z_drag = 1291 * self.omega_m ** 0.251 / (1. + 0.659 * self.omega_m ** 0.828) * (1. + z_drag_b1 * self.omega_b ** z_drag_b2)
        # HS1996, arXiv 9510117, eq. E1 actually better match to CLASS
        self.z_drag = 1345 * self.omega_m ** 0.251 / (1. + 0.659 * self.omega_m ** 0.828) * (1. + z_drag_b1 * self.omega_b ** z_drag_b2)

        # EH eq. 5
        self.r_drag = 31.5 * self.omega_b * self.theta_cmb ** (-4) * (1000. / (1 + self.z_drag))
        self.r_eq = 31.5 * self.omega_b * self.theta_cmb ** (-4) * (1000. / (1 + self.z_eq))

        # EH eq. 6
        self.rs_drag = 2. / (3. * self.k_eq) * self._np.sqrt(6. / self.r_eq)\
                       * self._np.log((self._np.sqrt(1 + self.r_drag) + self._np.sqrt(self.r_drag + self.r_eq)) / (1 + self._np.sqrt(self.r_eq)))
        # self.rs_drag = 44.5 * np.log(9.83 / self.omega_m) / np.sqrt(1. + 10. * self.omega_b**0.75)

    def compute(self):
        """Precompute coefficients for the transfer function."""

        self._set_rsdrag()

        # EH eq. 7
        self.k_silk = 1.6 * self.omega_b ** 0.52 * self.omega_m ** 0.73 * (1 + (10.4 * self.omega_m) ** (-0.95))  # 1/Mpc

        # alpha_c
        # EH eq. 11
        alpha_c_a1 = (46.9 * self.omega_m) ** 0.670 * (1 + (32.1 * self.omega_m) ** (-0.532))
        alpha_c_a2 = (12.0 * self.omega_m) ** 0.424 * (1 + (45.0 * self.omega_m) ** (-0.582))
        self.alpha_c = alpha_c_a1 ** (-self.frac_b) * alpha_c_a2 ** (-self.frac_b**3)

        # beta_c
        # EH eq. 12
        beta_c_b1 = 0.944 / (1 + (458 * self.omega_m) ** (-0.708))
        beta_c_b2 = 0.395 * self.omega_m ** (-0.0266)
        self.beta_c = 1. / (1 + beta_c_b1 * ((1 - self.frac_b) ** beta_c_b2) - 1)

        y_drag = (1 + self.z_eq) / (1 + self.z_drag)
        # EH eq. 15
        alpha_b_G = y_drag * (-6. * self._np.sqrt(1 + y_drag) + (2. + 3. * y_drag) * self._np.log((self._np.sqrt(1 + y_drag) + 1) / (self._np.sqrt(1 + y_drag) - 1)))
        self.alpha_b = 2.07 * self.k_eq * self.rs_drag * (1 + self.r_drag)**(-0.75) * alpha_b_G

        # EH eq. 23
        self.beta_node = 8.41 * self.omega_m ** 0.435
        # EH eq. 24
        self.beta_b = 0.5 + self.frac_b + (3. - 2. * self.frac_b) * self._np.sqrt((17.2 * self.omega_m) ** 2 + 1)

    def _rescale_sigma8(self):
        """Rescale perturbative quantities to match input sigma8."""
        if hasattr(self, '_rsigma8'):
            return self._rsigma8
        self._rsigma8 = 1.
        if 'sigma8' in self._params:
            fo = self.get_fourier()
            self._rsigma8 = self['sigma8'] / fo.sigma8_m
            self._sections.clear()
        return self._rsigma8


class Background(DefaultBackground):
    """
    Background quantities.

    Note
    ----
    Does not treat neutrinos.
    """
    @utils.flatarray()
    def growth_factor(self, z, znorm=None):
        """
        Approximation of growth factor.

        Parameters
        ----------
        z : array_like
            Redshifts.

        znorm : float, default=None
            If provided, growth factor is normalized as ``(1 + znorm) / (1 + z)`` in the matter domination era.
            Else, growth_factor is normalized to 1 at ``z = 0``.

        References
        ----------
        https://arxiv.org/abs/astro-ph/9709112, eq. 4
        https://ui.adsabs.harvard.edu/abs/1992ARA%26A..30..499C/abstract, eq. 29
        """
        def growth(z):
            return 1. / (1 + z) * 5 * self.Omega_m(z) / 2. / (self.Omega_m(z)**(4. / 7.) - self.Omega_de(z) + (1. + self.Omega_m(z) / 2.) * (1 + self.Omega_de(z) / 70.))

        growthz = growth(z)
        if znorm is not None:
            return (1. + znorm) * growthz
        return growth(z) / growth(0.)

    @utils.flatarray()
    def growth_rate(self, z):
        """
        Approximation of growth rate.

        References
        ----------
        https://arxiv.org/abs/astro-ph/0507263
        """
        wz1 = self.w0_fld + (1. - 0.5) * self.wa_fld
        return self.Omega_m(z)**(0.55 + 0.05 * (1 + wz1))


@utils.addproperty('rs_drag', 'z_drag')
class Thermodynamics(BaseSection):

    def __init__(self, engine):
        """Initialize :class:`Thermodynamics`."""
        super().__init__(engine)
        self._rs_drag = self._engine.rs_drag * self._engine['h']
        self._z_drag = self._engine.z_drag


@utils.addproperty('n_s', 'alpha_s', 'beta_s')
class Primordial(BaseSection):

    def __init__(self, engine):
        """Initialize :class:`Primordial`."""
        super().__init__(engine)
        self._h = self._engine['h']
        self._A_s = self._engine._A_s
        self._n_s = self._engine['n_s']
        self._alpha_s = self._engine['alpha_s']
        self._beta_s = self._engine['beta_s']
        self._rsigma8 = self._engine._rescale_sigma8()

    @property
    def A_s(self):
        r"""Scalar amplitude of the primordial power spectrum at :math:`k_\mathrm{pivot}`, unitless."""
        return self._A_s * self._rsigma8**2

    @property
    def ln_1e10_A_s(self):
        r""":math:`\ln(10^{10}A_s)`, unitless."""
        return np.log(1e10 * self.A_s)

    @property
    def k_pivot(self):
        r"""Primordial power spectrum pivot scale, where the primordial power is equal to :math:`A_{s}`, in :math:`h/\mathrm{Mpc}`."""
        return self._engine['k_pivot'] / self._h

    def pk_k(self, k, mode='scalar'):
        r"""
        The primordial spectrum of curvature perturbations at ``k``, generated by inflation, in :math:`(\mathrm{Mpc}/h)^{3}`.
        For scalar perturbations this is e.g. defined as:

        .. math::

            \mathcal{P_R}(k) = A_s \left (\frac{k}{k_\mathrm{pivot}} \right )^{n_s - 1 + 1/2 \alpha_s \ln(k/k_\mathrm{pivot}) + 1/6 \beta_s \ln(k/k_\mathrm{pivot})^2}

        See also: eq. 2 of `this reference <https://arxiv.org/abs/1303.5076>`_.

        Parameters
        ----------
        k : array_like
            Wavenumbers, in :math:`h/\mathrm{Mpc}`.

        mode : string, default='scalar'
            'scalar' mode.

        Returns
        -------
        pk : array
            The primordial power spectrum.
        """
        index = ['scalar'].index(mode)
        lnkkp = self._np.log(k / self.k_pivot)
        return self._h**3 * self.A_s * (k / self.k_pivot) ** (self.n_s - 1. + 1. / 2. * self.alpha_s * lnkkp + 1. / 6. * self.beta_s * lnkkp**2)

    def pk_interpolator(self, mode='scalar'):
        """
        Return power spectrum interpolator.

        Parameters
        ----------
        mode : string, default='scalar'
            'scalar', 'vector' or 'tensor' mode.

        Returns
        -------
        interp : PowerSpectrumInterpolator1D
        """
        return PowerSpectrumInterpolator1D.from_callable(pk_callable=lambda k: self.pk_k(k, mode=mode))


class Transfer(BaseSection):

    def transfer_k(self, k):
        """
        Return matter transfer function.

        Parameters
        ----------
        k : array_like
            Wavenumbers.

        Returns
        -------
        transfer : array
        """
        k = self._np.asarray(k) * self._engine['h']  # now in 1/Mpc
        # EH eq. 10
        q = k / (13.41 * self._engine.k_eq)
        ks = k * self._engine.rs_drag

        T_c_ln_beta = self._np.log(np.e + 1.8 * self._engine.beta_c * q)
        T_c_ln_nobeta = self._np.log(np.e + 1.8 * q)
        T_c_C_alpha = 14.2 / self._engine.alpha_c + 386. / (1 + 69.9 * q ** 1.08)
        T_c_C_noalpha = 14.2 + 386. / (1 + 69.9 * q ** 1.08)

        # EH eq. 18
        T_c_f = 1. / (1. + (ks / 5.4) ** 4)

        def T0(a, b):
            return a / (a + b * q**2)

        T_c = T_c_f * T0(T_c_ln_beta, T_c_C_noalpha) + (1 - T_c_f) * T0(T_c_ln_beta, T_c_C_alpha)

        # EH eq. 22
        s_tilde = self._engine.rs_drag * (1 + (self._engine.beta_node / ks)**3) ** (-1. / 3.)
        ks_tilde = k * s_tilde

        # EH eq. 21
        T_b_T0 = T0(T_c_ln_nobeta, T_c_C_noalpha)
        T_b_1 = T_b_T0 / (1 + (ks / 5.2)**2)
        T_b_2 = self._engine.alpha_b / (1 + (self._engine.beta_b / ks)**3) * self._np.exp(-(k / self._engine.k_silk) ** 1.4)
        T_b = self._np.sinc(ks_tilde / np.pi) * (T_b_1 + T_b_2)

        # EH eq. 16
        return self._engine.frac_b * T_b + (1 - self._engine.frac_b) * T_c


class Fourier(BaseSection):

    def __init__(self, engine):
        super().__init__(engine)
        self.pm = self._engine.get_primordial()
        self.tr = self._engine.get_transfer()
        self.ba = self._engine.get_background()
        self._h = self._engine['h']

    def pk_interpolator(self, of='delta_m', **kwargs):
        """
        Return :class:`PowerSpectrumInterpolator2D` instance.

        Parameters
        ----------
        of : string, tuple
            Perturbed quantities: 'delta_m', 'theta_m'.
            No distinction is made between baryons and CDM.
            Requesting velocity divergence 'theta_xx' will rescale the power spectrum by the growth rate as a function of ``z``.

        kwargs : dict
            Arguments for :class:`PowerSpectrumInterpolator2D`.
        """
        transfer = self.tr.transfer_k

        if isinstance(of, str): of = (of,)
        of = list(of)
        of = of + [of[0]] * (2 - len(of))

        ntheta = sum(of_.startswith('theta_') for of_ in of)
        if ntheta:
            def growth_factor_sq(z):
                return self.ba.growth_factor(z, znorm=0.)**2 * self.ba.growth_rate(z)**ntheta
        else:
            def growth_factor_sq(z):
                return self.ba.growth_factor(z, znorm=0.)**2

        def pk_callable(k):
            potential_to_density = (3. * self.ba.Omega0_m * 100**2 / (2. * (constants.c / 1e3)**2 * k**2)) ** (-2)
            curvature_to_potential = 9. / 25. * 2. * np.pi**2 / k**3 / self._h ** 3
            return transfer(k) ** 2 * potential_to_density * curvature_to_potential * self.pm.pk_k(k)

        return PowerSpectrumInterpolator2D.from_callable(pk_callable=pk_callable, growth_factor_sq=growth_factor_sq, **kwargs)

    def sigma_rz(self, r, z, of='delta_m', **kwargs):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`r \mathrm{Mpc}/h`. No distinction is made between baryons and CDM."""
        return self.pk_interpolator(of=of, **kwargs).sigma_rz(r, z)

    def sigma8_z(self, z, of='delta_m'):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`8 \mathrm{Mpc}/h`. No distinction is made between baryons and CDM."""
        return self.sigma_rz(8., z, of=of)

    @property
    def sigma8_m(self):
        r"""Current r.m.s. of matter perturbations in a sphere of :math:`8 \mathrm{Mpc}/h`, unitless."""
        return self.sigma8_z(0., of='delta_m')
