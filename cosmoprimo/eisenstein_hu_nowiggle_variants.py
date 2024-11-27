import warnings

import numpy as np

from .cosmology import BaseSection, BaseEngine
from .eisenstein_hu import Background, Thermodynamics, Primordial, CosmologyError
from .eisenstein_hu import Fourier as EHFourier
from .interpolator import PowerSpectrumInterpolator2D
from . import constants
from .jax import exception


class EisensteinHuNoWiggleVariantsEngine(BaseEngine):
    """
    Implementation of Eisenstein & Hu & variants analytic formulae.

    References
    ----------
    https://arxiv.org/pdf/astro-ph/9710252
    """
    name = 'eisenstein_hu_nowiggle_variants'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        def raise_error(has_fld):
            if has_fld:
                warnings.warn('{} cannot cope with non-constant dark energy'.format(self.__class__.__name__))
        exception(raise_error, self._has_fld)
        self.compute()
        self._A_s = self._get_A_s_fid()

    def _set_rsdrag(self):
        """Set sound horizon at the drag epoch."""

        self.omega_b = self['omega_b']
        self.omega_m = self['omega_cdm'] + self['omega_b'] + self['omega_ncdm_tot'] - self['omega_pncdm_tot']
        self.frac_b = self.omega_b / self.omega_m
        self.frac_cdm = self['omega_cdm'] / self.omega_m
        self.frac_cb = self.frac_cdm + self.frac_b
        self.frac_ncdm = 1. - self.frac_cb
        self.N_ncdm = self['N_ncdm']
        self.theta_cmb = self['T_cmb'] / 2.7

        # redshift and wavenumber of equality
        # EH eq. 1
        self.z_eq = 2.5e4 * self.omega_m * self.theta_cmb ** (-4) - 1.  # this is z
        self.k_eq = 0.0746 * self.omega_m * self.theta_cmb ** (-2)  # units of 1/Mpc

        # sound horizon and k_silk
        # EH eq. 2
        z_drag_b1 = 0.313 * self.omega_m ** (-0.419) * (1 + 0.607 * self.omega_m ** 0.674)
        z_drag_b2 = 0.238 * self.omega_m ** 0.223
        self.z_drag = 1291 * self.omega_m ** 0.251 / (1. + 0.659 * self.omega_m ** 0.828) * (1. + z_drag_b1 * self.omega_b ** z_drag_b2)
        # HS1996, arXiv 9510117, eq. E1 actually better match to CLASS
        # self.z_drag = 1345 * self.omega_m ** 0.251 / (1. + 0.659 * self.omega_m ** 0.828) * (1. + z_drag_b1 * self.omega_b ** z_drag_b2)

        self.rs_drag = 44.5 * self._np.log(9.83 / self.omega_m) / self._np.sqrt(1. + 10. * self.omega_b ** 0.75)

    def compute(self):
        """Precompute coefficients for the transfer function."""
        self._set_rsdrag()
        frac_bncdm = self.frac_b + self.frac_ncdm
        # EH eq. 11
        self.p_c = (5. - self._np.sqrt(1 + 24 * self.frac_cdm)) / 4.
        self.p_cb = (5. - self._np.sqrt(1 + 24. * self.frac_cb)) / 4.
        y_drag = (1 + self.z_eq) / (1 + self.z_drag)
        # EH eq. 15
        alpha_ncdm = self.frac_cdm / self.frac_cb * (5. - 2. * (self.p_c + self.p_cb)) / (5. - 4. * self.p_cb) * (1 + y_drag) ** (self.p_cb - self.p_c)\
                     * (1 + frac_bncdm * (-0.553 + 0.126 * frac_bncdm ** 2))\
                     / (1 - 0.193 * self._np.sqrt(self.frac_ncdm * self.N_ncdm) + 0.169 * self.frac_ncdm * self.N_ncdm ** 0.2)\
                     * (1 + (self.p_c - self.p_cb) / 2 * (1 + 1 / (3. - 4. * self.p_c) / (7. - 4. * self.p_cb)) / (1 + y_drag))
        self.gamma_ncdm = self._np.sqrt(alpha_ncdm)
        self.beta_c = 1 / (1 - 0.949 * frac_bncdm)


class Transfer(BaseSection):

    def __init__(self, engine):
        super().__init__(engine)
        self._h = engine['h']
        for name in ['omega_m', 'theta_cmb', 'N_ncdm', 'frac_ncdm', 'z_eq', 'p_cb', 'gamma_ncdm', 'rs_drag', 'beta_c']:
            setattr(self, '_' + name, getattr(engine, name))
        self.ba = engine.get_background()

    def transfer_kz(self, k, z=0., of='delta_m', grid=True):
        """
        Return matter transfer function.

        Parameters
        ----------
        k : array_like
            Wavenumbers.

        z : array_like, default=0.
            Redshifts.

        of : string, default='delta_m'
            Perturbed quantity, 'delta_cb' or 'delta_m'.

        grid : bool, default=True
            Whether ``k``, ``z`` coordinates should be interpreted as a grid, in which case the output will be of shape ``k.shape + z.shape``.

        Returns
        -------
        transfer : array
        """
        z = self._np.asarray(z)
        k = self._np.asarray(k) * self._h  # now in 1/Mpc
        if grid:
            toret_shape = k.shape + z.shape
            k = k.reshape(k.shape + (1,) * z.ndim)
        q = k / self._omega_m * self._theta_cmb ** 2

        # Compute the scale-dependent growth functions
        # EH eq. 14
        if self._N_ncdm:
            growth_k0 = self.ba.growth_factor(z, znorm=self._z_eq)
            y_freestream = 17.2 * self._frac_ncdm * (1 + 0.488 * self._frac_ncdm ** (-7. / 6.)) * (self._N_ncdm * q / self._frac_ncdm) ** 2
            tmp1 = growth_k0 ** (1. - self._p_cb)
            tmp2 = (growth_k0 / (1 + y_freestream)) ** 0.7
            if of == 'delta_cb':
                # EH eq. 12
                growth = (1. + tmp2) ** (self._p_cb / 0.7) * tmp1
            elif of == 'delta_m':
                # EH eq. 13
                growth = (self._frac_cb ** (0.7 / self._p_cb) + tmp2) ** (self._p_cb / 0.7) * tmp1
            else:
                raise CosmologyError('No {} transfer function can be computed (choices are ["delta_cb", "delta_m"]).'.format(of))
        else:
            growth = growth_k0 = np.ones_like(z)

        # Compute the master function
        # EH eq. 16
        gamma_eff = self._omega_m * (self._gamma_ncdm + (1 - self._gamma_ncdm) / (1 + (k * self._rs_drag * 0.43) ** 4))
        q_eff = q * self._omega_m / gamma_eff

        # EH eq. 18
        T_sup_L = self._np.log(np.e + 1.84 * self._beta_c * self._gamma_ncdm * q_eff)
        T_sup_C = 14.4 + 325. / (1 + 60.5 * q_eff ** 1.08)
        T_sup = T_sup_L / (T_sup_L + T_sup_C * q_eff ** 2)

        # EH eq. 22 - 23
        if self._N_ncdm:
            q_ncdm = 3.92 * q * self._np.sqrt(self._N_ncdm / self._frac_ncdm)
            max_fs_correction = 1 + 1.24 * self._frac_ncdm ** 0.64 * self._N_ncdm ** (0.3 + 0.6 * self._frac_ncdm)\
                                / (q_ncdm ** (-1.6) + q_ncdm ** 0.8)
            T_sup *= max_fs_correction

        # Now compute the CDM + HDM + baryon transfer functions
        # EH eq. 6
        toret = T_sup * growth / growth_k0
        if grid:
            toret = toret.reshape(toret_shape)
        return toret


class Fourier(EHFourier):

    def pk_interpolator(self, of='delta_m', **kwargs):
        """
        Return :class:`PowerSpectrumInterpolator2D` instance.

        Parameters
        ----------
        of : string, tuple
            Perturbed quantities: 'delta_m', 'theta_m', 'delta_cb', 'theta_cb'.
            No difference made between 'theta_cb' and 'theta_m'.
            Requesting velocity divergence 'theta_xx' will rescale the power spectrum by the growth rate as a function of ``z``.

        kwargs : dict
            Arguments for :class:`PowerSpectrumInterpolator2D`.
        """
        if not isinstance(of, (tuple, list)):
            of = (of, of)
        ntheta = sum(of_.startswith('theta_') for of_ in of)
        of = tuple(of_.replace('theta_', 'delta_') for of_ in of)
        if ntheta:
            def growth_factor_sq(ba, z):
                return ba.growth_factor(z, znorm=0.)**2 * ba.growth_rate(z)**ntheta
        else:
            def growth_factor_sq(ba, z):
                return ba.growth_factor(z, znorm=0.)**2

        def pk_callable(ba, pm, tr, k, z=0, grid=True):
            tk = tr.transfer_kz(k, z=z, grid=grid, of=of[0])
            if of[1] == of[0]: tk **= 2
            else: tk *= tr.transfer_kz(k, z=z, grid=grid, of=of[1])
            potential_to_density = (3. * ba.Omega0_m * 100**2 / (2. * (constants.c / 1e3)**2 * k**2)) ** (-2)
            curvature_to_potential = 9. / 25. * 2. * np.pi**2 / k**3 / ba.h ** 3
            pdd = potential_to_density * curvature_to_potential * pm.pk_k(k)
            return tk * growth_factor_sq(ba, z) * pdd.reshape(pdd.shape + (1,) * (tk.ndim - pdd.ndim))

        from .jax import Partial

        return PowerSpectrumInterpolator2D.from_callable(pk_callable=Partial(pk_callable, self.ba, self.pm, self.tr), growth_factor_sq=None, **kwargs)
