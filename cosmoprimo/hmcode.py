"""
HMcode2020: the non-linear (dark-matter only) matter power spectrum of Mead et al. (2021), computed from any linear one,
as CAMB's ``mead2020``.

References
----------
Mead, Brieden, Tröster, Heymans (2021), https://arxiv.org/abs/2009.01858
Mead (2017), https://arxiv.org/abs/1606.05345
Mead et al. (2016), https://arxiv.org/abs/1602.02154
Mead et al. (2015), https://arxiv.org/abs/1505.07833
"""

import numpy as np

from .jax import numpy_jax, odeint, Interpolator1D


#: Critical density today, (Msun/h) / (Mpc/h)^3, CAMB's HMcode value.
rho_crit_over_Msunph_per_Mpcph3 = 2.775e11


class HMcode2020(object):
    """
    HMcode2020 (dark-matter only) non-linear matter power spectrum from linear spectra.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmology, for its background parameters; the linear spectra are passed at call.
    """
    #: Integration nodes in ln(kR) for sigma(R) and n_eff: kR from 1e-3 to 200 (Simpson; the window squared is ~ 4.5 / (kR)^4 beyond)
    _log_kr = np.linspace(np.log(1e-3), np.log(200.), 4001)
    #: Integration nodes in ln k [h/Mpc] for sigma_v
    _log_k_velocity = np.linspace(np.log(1e-7), np.log(1e4), 4001)
    #: Halo masses [Msun/h], as CAMB
    _mass = np.geomspace(1., 1e18, 256)
    #: Wavenumbers [h/Mpc] of the BAO wiggle extraction, as CAMB
    _k_wiggle = np.geomspace(5e-3, 5., 512)
    #: Scale factors of the growth tables (RK4 nodes)
    _a_growth = np.concatenate([np.geomspace(1e-4, 1e-2, 400, endpoint=False), np.linspace(1e-2, 1., 2000)])
    #: Wavenumbers [h/Mpc] of the internal linear table the integrals read, extrapolated in log-log beyond its ends (CAMB's rebinning:
    #: the extrapolation, which sets the haloes where nothing has collapsed at 1 Msun/h, then does not depend on the input k grid)
    _k_table = np.geomspace(1e-3, 1e2, 512)
    #: sigma(R) at the halo radii by one FFT correlation per redshift; False: the direct quadrature at each radius
    _use_sigma_fft = True
    #: The Omega_m range the FFT table's k extent is sized for (the radii scale as Omega_m^(-1/3)); beyond it the shift is clipped
    _Omega_m_range = (0.005, 2.)

    def __init__(self, cosmo):
        # background, as CAMB's HMcode: matter (massive neutrinos included) + dark energy + curvature; the radiation is ignored,
        # folded into the curvature so that Omega_de is the cosmology's
        self.Omega0_cdm, self.Omega0_b = cosmo['Omega_cdm'], cosmo['Omega_b']
        self.Omega0_m, self.Omega0_de = cosmo['Omega_m'], cosmo['Omega_de']
        self.Omega0_ncdm = self.Omega0_m - self.Omega0_b - self.Omega0_cdm
        self.Omega0_k = 1. - self.Omega0_m - self.Omega0_de
        self.h, self.n_s, self.w0_fld, self.wa_fld, self.T0_cmb = (cosmo[name] for name in ['h', 'n_s', 'w0_fld', 'wa_fld', 'T_cmb'])
        # number of massive neutrino species, the mass split evenly (cold-matter ratio only)
        self.N_ncdm = max(np.sum(np.atleast_1d(cosmo['m_ncdm']) > 0.), 1)
        self._np = jnp = numpy_jax(self.Omega0_cdm, self.Omega0_b, self.Omega0_m, self.Omega0_de, self.h, self.n_s,
                                   self.w0_fld, self.wa_fld)
        # massive-neutrino fraction of the matter
        self.f_ncdm = self.Omega0_ncdm / self.Omega0_m
        # mean matter density today, (Msun/h) / (Mpc/h)^3
        self.mean_density = rho_crit_over_Msunph_per_Mpcph3 * self.Omega0_m
        # Lagrangian radius of each halo mass
        self._lagrangian_radius = (3. * self._mass / (4. * np.pi * self.mean_density))**(1. / 3.)

        # Linear growth D(a), unnormalized (D -> a in matter domination), by RK4 on the scale-factor grid
        a_grid = self._a_growth
        a_initial = a_grid[0]
        # initial conditions D ~ a^(1 - 3 f_de / 5), with f_de the dark-energy fraction at a_initial
        f_de = 1. - self.Omega_m(1. / a_initial - 1.)
        initial = jnp.array([a_initial**(1. - 3. * f_de / 5.), (1. - 3. * f_de / 5.) * a_initial**(-3. * f_de / 5.)])

        def derivatives(state, a):
            # state = (D, dD/da); D'' + (2 + dln H / dln a) D' / a = 3/2 Omega_m(a) D / a^2
            z = 1. / a - 1.
            w_fld = self.w0_fld + (1. - a) * self.wa_fld
            # dark-energy density over its value today
            rho_fld = a**(-3. * (1. + self.w0_fld + self.wa_fld)) * jnp.exp(-3. * self.wa_fld * (1. - a))
            # dln H / dln a = acceleration / E^2 (curvature cancels)
            acceleration = -(self.Omega0_m * a**-3 + self.Omega0_de * (1. + 3. * w_fld) * rho_fld) / 2.
            return jnp.stack([state[1], 1.5 * self.Omega_m(z) * state[0] / a**2 - (2. + acceleration / self.efunc(z)**2) * state[1] / a])

        self._growth_table = odeint(derivatives, initial, a_grid, method='rk4')[:, 0]
        self._growth_today = self._growth_table[-1]
        # accumulated growth G(a) = int_0^a D / a' da': early part, D / a -> 1, a rectangle of height D(a_initial) / a_initial
        # over [0, a_initial]; then trapezoids
        integrand = self._growth_table / a_grid
        cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum((integrand[1:] + integrand[:-1]) / 2. * np.diff(a_grid))])
        self._accumulated_growth_table = self._growth_table[0] + cumulative
        # CAMB's growint, used by the Dolag correction: g_LCDM(a) = exp(-int_a^1 Omega_m(a')^0.55 / a' da'), flat LCDM, same Omega_m
        Omega_m_lcdm = self.Omega0_m * a_grid**-3 / (self.Omega0_m * a_grid**-3 + 1. - self.Omega0_m)
        integrand = Omega_m_lcdm**0.55 / a_grid
        cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum((integrand[1:] + integrand[:-1]) / 2. * np.diff(a_grid))])
        self._log_growth_lcdm_table = -(cumulative[-1] - cumulative)

    def efunc(self, z):
        r"""
        :math:`E(z) = H(z) / H_0`, as CAMB's HMcode: matter (massive neutrinos included) + dark energy + curvature, no radiation.

        Parameters
        ----------
        z : float, array
            Redshifts.

        Returns
        -------
        efunc : float, array
        """
        a = 1. / (1. + z)
        rho_fld = a**(-3. * (1. + self.w0_fld + self.wa_fld)) * self._np.exp(-3. * self.wa_fld * (1. - a))
        return self._np.sqrt(self.Omega0_m * a**-3 + self.Omega0_de * rho_fld + self.Omega0_k * a**-2)

    def Omega_m(self, z):
        """
        Matter density parameter, massive neutrinos included.

        Parameters
        ----------
        z : float, array
            Redshifts.

        Returns
        -------
        Omega_m : float, array
        """
        return self.Omega0_m * (1. + z)**3 / self.efunc(z)**2

    def growth_factor(self, z, znorm=0.):
        """
        Linear growth factor.

        Parameters
        ----------
        z : float, array
            Redshifts.
        znorm : float, default=0.
            Redshift where the growth factor is normalized to 1; if None, normalized to a in matter domination.

        Returns
        -------
        growth_factor : float, array
        """
        growth_factor = self._np.interp(1. / (1. + z), self._a_growth, self._growth_table)
        if znorm is None:
            return growth_factor
        return growth_factor / self._np.interp(1. / (1. + znorm), self._a_growth, self._growth_table)

    def accumulated_growth_factor(self, z):
        """
        Accumulated growth G(a) = int_0^a D(a') / a' da', with D normalized to a in matter domination.

        Parameters
        ----------
        z : float, array
            Redshifts.

        Returns
        -------
        accumulated_growth_factor : float, array
        """
        return self._np.interp(1. / (1. + z), self._a_growth, self._accumulated_growth_table)

    def delta_c(self, z):
        """
        Linear collapse threshold, Mead (2017) fit (Appendix A), with the massive-neutrino factor of Mead et al. (2016).

        Parameters
        ----------
        z : float, array
            Redshifts.

        Returns
        -------
        delta_c : float, array
        """
        # g / a and G / a, with g normalized to a in matter domination
        growth_over_a = self.growth_factor(z, znorm=None) * (1. + z)
        accumulated_growth_over_a = self.accumulated_growth_factor(z) * (1. + z)
        log_Omega_m = self._np.log10(self.Omega_m(z))
        # f1, f2 of Mead (2017), Eq. A3
        f1 = -0.0069 - 0.0208 * (1. - growth_over_a) + 0.0312 * (1. - growth_over_a)**2 + 0.0021 * (1. - accumulated_growth_over_a)
        f2 = 0.0001 - 0.0647 * (1. - growth_over_a) - 0.0417 * (1. - growth_over_a)**2 + 0.0646 * (1. - accumulated_growth_over_a)
        return (1. + f1 * log_Omega_m + f2) * (3. / 20.) * (12. * np.pi)**(2. / 3.) * (1. - 0.041 * self.f_ncdm)

    def Delta_v(self, z):
        """
        Virialized overdensity, Mead (2017) fit (Appendix A), with the massive-neutrino factor of Mead et al. (2016).

        Parameters
        ----------
        z : float, array
            Redshifts.

        Returns
        -------
        Delta_v : float, array
        """
        # g / a and G / a, with g normalized to a in matter domination
        growth_over_a = self.growth_factor(z, znorm=None) * (1. + z)
        accumulated_growth_over_a = self.accumulated_growth_factor(z) * (1. + z)
        log_Omega_m = self._np.log10(self.Omega_m(z))
        # f3, f4 of Mead (2017), Eq. A4
        f3 = -0.79 - 10.17 * (1. - growth_over_a) + 2.51 * (1. - growth_over_a)**2 + 6.51 * (1. - accumulated_growth_over_a)
        f4 = -1.89 + 0.38 * (1. - growth_over_a) + 18.8 * (1. - growth_over_a)**2 - 15.87 * (1. - accumulated_growth_over_a)
        return (1. + f3 * log_Omega_m + f4 * log_Omega_m**2) * 18. * np.pi**2 * (1. + 0.763 * self.f_ncdm)

    def __call__(self, k, z, pk, pk_cb=None, return_terms=False):
        r"""
        Non-linear matter power spectrum.

        Parameters
        ----------
        k : array (nk,)
            Wavenumbers [h/Mpc] of the linear spectra, also those of the output; the spectra are extrapolated in log-log beyond them.
        z : array (nz,)
            Redshifts.
        pk : array (nk, nz)
            Linear matter power spectrum [(Mpc/h)^3], massive neutrinos included.
        pk_cb : array (nk, nz), default=None
            Linear cold-matter (cdm + baryons) power spectrum; ``None``: from ``pk`` with the Eisenstein & Hu (1999) ratio, as CAMB.
        return_terms : bool, default=False
            Also return the two-halo and one-halo terms.

        Returns
        -------
        pk_nonlinear : array (nk, nz)
            Non-linear matter power spectrum [(Mpc/h)^3].
        two_halo, one_halo : array (nk, nz)
            Two-halo and one-halo terms [(Mpc/h)^3], if ``return_terms``.
        """
        jnp = numpy_jax(pk, pk_cb, self.Omega0_m, self.h)
        self._np = jnp
        # sine and cosine integrals of the backend, for the NFW window (jax.scipy's has a custom JVP)
        if jnp is np:
            from scipy.special import sici
        else:
            from jax.scipy.special import sici
        # API boundary: user inputs may be lists or scalars
        k = np.asarray(k, dtype='f8')
        z = np.atleast_1d(np.asarray(z, dtype='f8'))
        pk = jnp.asarray(pk, dtype='f8').reshape(k.size, z.size)
        if pk_cb is None:
            # Eisenstein & Hu (1999) approximation to T_cb / T_m, as CAMB's HMcode; 1 without massive neutrinos
            has_nu = self.f_ncdm > 0.
            f_nu = jnp.where(has_nu, self.f_ncdm, 0.5)  # any positive value where there are none: the branch is discarded
            # p_cb of Eisenstein & Hu (1999), Eq. 11
            exponent = (5. - jnp.sqrt(1. + 24. * (1. - f_nu))) / 4.
            theta_cmb = self.T0_cmb / 2.7
            z_equality = 2.5e4 * self.Omega0_m * self.h**2 * theta_cmb**-4
            growth = (1. + z_equality) * self.growth_factor(z[None, :], znorm=None)
            # q of Eisenstein & Hu (1999), Eq. 5, k in 1/Mpc
            q = k[:, None] * self.h * theta_cmb**2 / (self.Omega0_m * self.h**2)
            y_free_streaming = 17.2 * f_nu * (1. + 0.488 * f_nu**(-7. / 6.)) * (self.N_ncdm * q / f_nu)**2
            growth_cb = (1. + (growth / (1. + y_free_streaming))**0.7)**(exponent / 0.7)
            growth_cbnu = ((1. - f_nu)**(0.7 / exponent) + (growth / (1. + y_free_streaming))**0.7)**(exponent / 0.7)
            pk_cb = pk * jnp.where(has_nu, growth_cb / growth_cbnu, 1.)**2
        pk_cb = jnp.asarray(pk_cb, dtype='f8').reshape(k.size, z.size)
        # P(k) -> Delta^2(k) = k^3 P(k) / (2 pi^2)
        delta2_factor = k**3 / (2. * np.pi**2)
        mass, radius, k_wiggle, k_table = self._mass, self._lagrangian_radius, self._k_wiggle, self._k_table
        # linear interpolation in log-log, extrapolated as power laws from the two end points (CAMB's ``find`` with ``iorder=1``,
        # and its off-table rule); k assumed increasing
        log_log = dict(k=1, interp_x='log', interp_fun='log', extrap=True, assume_sorted=True)

        # Quadrature weights: Simpson's rule on the uniform grids
        def simpson_weights(size, step):
            weights = np.ones(size)
            weights[1:-1:2], weights[2:-1:2] = 4., 2.
            return weights * step / 3.

        # sigma(R), n_eff: in ln(kR), with the top-hat window W(kR) and its derivative W'(kR) (small-kR expansions below 1e-3)
        log_kr = self._log_kr
        step_kr = log_kr[1] - log_kr[0]
        kr = np.exp(log_kr)
        weights_kr = simpson_weights(kr.size, step_kr)
        small = np.abs(kr) < 1e-3
        kr_safe = np.where(small, 1., kr)  # avoid the division by zero in the discarded branch
        tophat = np.where(small, 1. - kr**2 / 10., 3. * (np.sin(kr_safe) - kr_safe * np.cos(kr_safe)) / kr_safe**3)
        tophat_derivative = np.where(small, -kr / 5. + kr**3 / 70.,
                                     3. / kr_safe**4 * ((kr_safe**2 - 3.) * np.sin(kr_safe) + 3. * kr_safe * np.cos(kr_safe)))
        weights_sigma = weights_kr * tophat**2

        def sigma_quadrature(radius, delta2):
            # sigma(R) by direct quadrature in ln(kR) at each radius
            return jnp.sqrt(delta2(kr[None, :] / radius[:, None]) @ weights_sigma)

        # sigma_v: in ln k, Delta^2(k) / k^2
        k_velocity = np.exp(self._log_k_velocity)
        weights_velocity = simpson_weights(k_velocity.size, self._log_k_velocity[1] - self._log_k_velocity[0]) / k_velocity**2

        # haloes: Lagrangian and formation (0.01 M) radii
        radius_all = np.concatenate([radius, radius * 0.01**(1. / 3.)])
        if self._use_sigma_fft:
            # sigma(R) at every radius from one FFT correlation of the tabulated Delta^2 with the quadrature weights:
            # on a ln k grid of the step of ln(kR), Delta^2(kR / R) is the table shifted by an amount set by R alone
            # ln k table, uniform with the step of ln(kR), covering ln(kR / R) for every radius a call can ask for
            # (the mass grid, its formation radii 0.01^(1/3) R, R = 8 Mpc/h) over _Omega_m_range
            density = rho_crit_over_Msunph_per_Mpcph3 * np.array(self._Omega_m_range)
            radius_min = min((3. * mass[0] / (4. * np.pi * density[1]))**(1. / 3.) * 0.01**(1. / 3.), 8.)
            radius_max = max((3. * mass[-1] / (4. * np.pi * density[0]))**(1. / 3.), 8.)
            log_k_min = log_kr[0] - np.log(radius_max) - 2. * step_kr
            size = int(np.ceil((log_kr[-1] - np.log(radius_min) + 2. * step_kr - log_k_min) / step_kr)) + 1
            log_k_fft = log_k_min + step_kr * np.arange(size)
            # FFT of the weights, zero-padded to a power of 2 (conjugated: correlation)
            size_fft = 1 << int(np.ceil(np.log2(log_k_fft.size + log_kr.size)))
            weights_sigma_fft = np.conj(np.fft.rfft(weights_sigma, size_fft))
            # real shift of each radius, read by linear interpolation between integer shifts
            shift = (log_kr[0] - jnp.log(radius_all) - log_k_fft[0]) / step_kr
            size_correlation = log_k_fft.size - log_kr.size + 1
            lower = jnp.clip(jnp.floor(shift).astype(int), 0, size_correlation - 2)
            fraction = shift - lower

        # BAO wiggle extraction: Gaussian smoothing in ln k of width 0.25 (CAMB's), points within 3 sigma of either edge left as they are
        log_k_wiggle = np.log(k_wiggle)
        smoothing_width = 0.25
        smoothing = np.exp(-(log_k_wiggle[:, None] - log_k_wiggle[None, :])**2 / (2. * smoothing_width**2))
        smoothing /= smoothing.sum(axis=-1)[:, None]
        edge = np.abs(log_k_wiggle - log_k_wiggle[0]) < 3. * smoothing_width
        edge |= np.abs(log_k_wiggle - log_k_wiggle[-1]) < 3. * smoothing_width
        smoothing[edge] = np.eye(k_wiggle.size)[edge]
        in_wiggle = (k >= k_wiggle[0]) & (k <= k_wiggle[-1])
        # unnormalized Eisenstein & Hu (1998) no-wiggle Delta^2(k), Eqs. 26, 31, 30, 28, 29
        omega_m, omega_b, baryon_fraction = self.Omega0_m * self.h**2, self.Omega0_b * self.h**2, self.Omega0_b / self.Omega0_m
        sound_horizon = 44.5 * jnp.log(9.83 / omega_m) / jnp.sqrt(1. + 10. * omega_b**0.75)
        alpha_gamma = 1. - 0.328 * jnp.log(431. * omega_m) * baryon_fraction + 0.38 * jnp.log(22.3 * omega_m) * baryon_fraction**2
        gamma_eff = self.Omega0_m * self.h * (alpha_gamma + (1. - alpha_gamma) / (1. + (0.43 * k_wiggle * sound_horizon * self.h)**4))
        q_nowiggle = k_wiggle * (self.T0_cmb / 2.7)**2 / gamma_eff
        log_term = jnp.log(2. * np.e + 1.8 * q_nowiggle)
        transfer = log_term / (log_term + (14.2 + 731. / (1. + 62.5 * q_nowiggle)) * q_nowiggle**2)
        nowiggle_shape = k_wiggle**(self.n_s + 3.) * transfer**2

        def one_redshift(pk_z, pk_cb_z, redshift):
            # the integrals read the linear spectra rebinned on the internal table, extrapolated in log-log beyond it
            delta2 = Interpolator1D(k_table, Interpolator1D(k, pk_z * delta2_factor, **log_log)(k_table), **log_log)
            delta2_cb = Interpolator1D(k_table, Interpolator1D(k, pk_cb_z * delta2_factor, **log_log)(k_table), **log_log)
            # halo-model parameters (Mead et al. 2021, Table 2), functions of the cold-matter sigma_8(z)
            sigma8_cb = sigma_quadrature(np.array([8.]), delta2_cb)[0]
            # linear velocity dispersion [Mpc/h] of all matter: sqrt(int Delta^2(k) / k^2 dln k / 3)
            sigma_v = jnp.sqrt(delta2(k_velocity) @ weights_velocity / 3.)
            delta_c, Delta_v = self.delta_c(redshift), self.Delta_v(redshift)
            k_star = 0.05618 * sigma8_cb**-1.013
            damping_fraction = jnp.clip(0.2696 * sigma8_cb**0.9403, 1e-3, 0.99)
            k_damping, n_damping = 0.05699 * sigma8_cb**-1.089, 2.85
            eta = 0.1281 * sigma8_cb**-0.3644
            # haloes: sigma at the Lagrangian and formation radii
            if self._use_sigma_fft:
                # correlation of the tabulated Delta^2 with the weights at every integer shift, read at each radius' shift
                table = delta2_cb(np.exp(log_k_fft))
                correlation = jnp.fft.irfft(jnp.fft.rfft(table, size_fft) * weights_sigma_fft, size_fft)[:size_correlation]
                sigmas = jnp.sqrt((1. - fraction) * correlation[lower] + fraction * correlation[lower + 1])
            else:
                sigmas = sigma_quadrature(radius_all, delta2_cb)
            sigma, sigma_formation = sigmas[:radius.size], sigmas[radius.size:]
            # peak height nu, virial radii
            nu = delta_c / sigma
            virial_radius = radius / Delta_v**(1. / 3.)
            # non-linear radius nu(R_nl) = 1 (the smallest radius if nothing has collapsed, as CAMB)
            nonlinear_radius = jnp.where(nu[0] > 1., radius[0], jnp.exp(jnp.interp(0., jnp.log(nu), jnp.log(radius))))
            # effective index n_eff = -3 - d ln sigma^2 / d ln R at R_nl, clipped to [n_s - 4, n_s]
            delta2_nonlinear = delta2_cb(kr / nonlinear_radius)
            sigma2_nonlinear = delta2_nonlinear @ weights_sigma
            dsigma2_nonlinear = 2. * (delta2_nonlinear @ (weights_kr * tophat * tophat_derivative * kr))
            effective_index = jnp.clip(-3. - dsigma2_nonlinear / sigma2_nonlinear, self.n_s - 4., self.n_s)
            # smoothing of the transition between the two terms
            alpha = jnp.clip(1.875 * 1.603**effective_index, 0.5, 2.)
            # Bullock concentrations: g(z_f) = delta_c g(z) / sigma(f M, z), z_f = z if that lies in the future; amplitude 5.196
            growth_z = self.growth_factor(redshift)
            target = delta_c * growth_z / sigma_formation
            a_formation = jnp.interp(target, self._growth_table / self._growth_today, self._a_growth)
            z_formation = jnp.where(target > growth_z, redshift, 1. / a_formation - 1.)
            concentration = 5.196 * (1. + z_formation) / (1. + redshift)
            # Dolag et al. (2004) dark-energy correction, normalized at high z (HMcode2020), below z = 10:
            # (g / g_LCDM)(z = 10) / (g / g_LCDM)(z)
            a_dolag = 1. / (1. + jnp.array([10., redshift]))
            growth_lcdm_10, growth_lcdm_z = jnp.exp(jnp.interp(a_dolag, self._a_growth, self._log_growth_lcdm_table))
            dolag = self.growth_factor(10.) / growth_lcdm_10 * growth_lcdm_z / growth_z
            concentration = concentration * jnp.where(redshift < 10., dolag, 1.)

            # two-halo term: linear, BAO wiggle damped by sigma_v, amplitude damped beyond k_damping
            delta2_wiggle = delta2(k_wiggle)
            # no-wiggle spectrum normalized to the linear one at k = 0.03 h/Mpc
            nowiggle = nowiggle_shape * delta2(0.03) / jnp.exp(jnp.interp(np.log(0.03), log_k_wiggle, jnp.log(nowiggle_shape)))
            # wiggle = linear - smoothed(linear / no-wiggle) * no-wiggle
            wiggle = delta2_wiggle - (smoothing @ (delta2_wiggle / nowiggle)) * nowiggle
            wiggle_at_k = jnp.where(in_wiggle, jnp.interp(np.log(k), log_k_wiggle, wiggle), 0.)
            dewiggled = pk_z * delta2_factor + (jnp.exp(-(k * sigma_v)**2) - 1.) * wiggle_at_k
            x_damping = (k / k_damping)**n_damping
            two_halo = jnp.maximum(dewiggled * (1. - damping_fraction * x_damping / (1. + x_damping)), 0.)

            # one-halo term: NFW windows bloated by nu^eta, integrated over nu with the Sheth & Tormen (1999) multiplicity function
            # g(nu) = A (1 + (q nu^2)^-p) exp(-q nu^2 / 2), normalized to int g(nu) dnu = 1
            st_p, st_q, st_amplitude = 0.3, 0.707, 0.21616
            sheth_tormen = st_amplitude * (1. + (st_q * nu**2)**(-st_p)) * jnp.exp(-st_q * nu**2 / 2.)
            # Fourier transform of the NFW profile truncated at the virial radius, normalized to W(k -> 0) = 1, clipped to [0, 1]:
            # k r_s with the scale radius r_s = R_v / c, and k (r_s + R_v)
            k_scale_radius = k[:, None] * nu[None, :]**eta * virial_radius[None, :] / concentration[None, :]
            k_outer = (1. + concentration[None, :]) * k_scale_radius
            si_inner, ci_inner = sici(k_scale_radius)
            si_outer, ci_outer = sici(k_outer)
            window = jnp.cos(k_scale_radius) * (ci_outer - ci_inner) + jnp.sin(k_scale_radius) * (si_outer - si_inner)
            window = window - jnp.sin(concentration[None, :] * k_scale_radius) / k_outer
            # normalized by the enclosed mass, ln(1 + c) - c / (1 + c)
            window = window / (jnp.log(1. + concentration) - concentration / (1. + concentration))[None, :]
            window = jnp.clip(window, 0., 1.)
            integrand = sheth_tormen[None, :] * window**2 * mass[None, :]
            # trapezoidal rule in nu; only the cold matter clusters in haloes
            one_halo = jnp.sum((integrand[:, 1:] + integrand[:, :-1]) / 2. * jnp.diff(nu)[None, :], axis=-1)
            one_halo = one_halo * (1. - self.f_ncdm)**2 / self.mean_density * delta2_factor
            # damped below k_star
            x_star = (k / k_star)**4
            one_halo = one_halo * x_star / (1. + x_star)

            pk_nonlinear = (two_halo**alpha + one_halo**alpha)**(1. / alpha)
            return pk_nonlinear / delta2_factor, two_halo / delta2_factor, one_halo / delta2_factor

        if jnp is np:
            terms = [one_redshift(pk[:, iz], pk_cb[:, iz], redshift) for iz, redshift in enumerate(z)]
            pk_nonlinear, two_halo, one_halo = (np.stack([term[iterm] for term in terms], axis=-1) for iterm in range(3))
        else:  # vmapped over the redshifts: one traced body, rather than z.size unrolled copies (compile time, per-operation overhead)
            import jax
            pk_nonlinear, two_halo, one_halo = (term.T for term in jax.vmap(one_redshift)(pk.T, pk_cb.T, jnp.asarray(z)))
        if return_terms:
            return pk_nonlinear, two_halo, one_halo
        return pk_nonlinear


def hmcode2020_ratio(fourier, engine):
    r"""
    The matter :math:`P_\mathrm{nl} / P_\mathrm{lin}` of HMcode2020 on a Fourier section's own linear tables (``table(non_linear=False)``
    of ``'delta_m'`` and ``'delta_cb'``), on their (k, z) grid: what an engine with ``non_linear='mead2020'`` multiplies its linear spectra
    by, as CAMB does for every variable.

    Parameters
    ----------
    fourier : Fourier section
        With ``table(non_linear=False, of=...)`` returning ``k`` [h/Mpc], ``z``, ``pk`` [(Mpc/h)^3] of shape (nk, nz).
    engine : BaseEngine
        For the background parameters.

    Returns
    -------
    ratio : array (nk, nz)
    """
    k, z, pk = fourier.table(non_linear=False, of='delta_m')
    pk_cb = fourier.table(non_linear=False, of='delta_cb')[-1]
    return HMcode2020(engine)(k, z, pk, pk_cb=pk_cb) / pk
