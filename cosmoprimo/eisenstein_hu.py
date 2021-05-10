import warnings

import numpy as np

from .cosmology import BaseEngine, BaseSection, CosmologyError
from .interpolator import PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D


class EisensteinHuEngine(BaseEngine):
    """
    Implementation of Eisenstein & Hu analytic formulae.

    References
    ----------
    https://arxiv.org/abs/astro-ph/9709112
    """
    def __init__(self, *args, **kwargs):
        super(EisensteinHuEngine,self).__init__(*args,**kwargs)
        if self['N_ncdm']:
            warnings.warn('{} cannot cope with massive neutrinos'.format(self.__class__.__name__))
        if self['Omega_k'] != 0.:
            warnings.warn('{} cannot cope with non-zero curvature'.format(self.__class__.__name__))
        self.compute()

    def _set_rsdrag(self):
        """Set sound horizon at the drag epoch."""

        self.omega_b = self['omega_b']
        self.omega_m = self['omega_cdm'] + self['omega_b']
        self.frac_baryon  = self.omega_b / self.omega_m
        self.theta_cmb = self['T_cmb'] / 2.7

        # redshift and wavenumber of equality
        # EH eq. 2 & 3
        self.z_eq = 2.5e4 * self.omega_m * self.theta_cmb ** (-4) - 1. # this is z
        self.k_eq = 0.0746 * self.omega_m * self.theta_cmb ** (-2) # units of 1/Mpc

        # sound horizon and k_silk
        # EH eq. 4
        z_drag_b1 = 0.313 * self.omega_m ** (-0.419) * (1 + 0.607 * self.omega_m ** 0.674)
        z_drag_b2 = 0.238 * self.omega_m ** 0.223
        #self.z_drag = 1291 * self.omega_m ** 0.251 / (1. + 0.659 * self.omega_m ** 0.828) * (1. + z_drag_b1 * self.omega_b ** z_drag_b2)
        # HS1996, arXiv 9510117, eq. E1 actually better match to CLASS
        self.z_drag = 1345 * self.omega_m ** 0.251 / (1. + 0.659 * self.omega_m ** 0.828) * (1. + z_drag_b1 * self.omega_b ** z_drag_b2)

        # EH eq. 5
        self.r_drag = 31.5 * self.omega_b * self.theta_cmb ** (-4) * (1000. / (1 + self.z_drag))
        self.r_eq   = 31.5 * self.omega_b * self.theta_cmb ** (-4) * (1000. / (1 + self.z_eq))

        # EH eq. 6
        self.rs_drag = 2. / (3.*self.k_eq) * np.sqrt(6. / self.r_eq) * \
                    np.log((np.sqrt(1 + self.r_drag) + np.sqrt(self.r_drag + self.r_eq)) / (1 + np.sqrt(self.r_eq)) )

    def compute(self):
        """Precompute coefficients for the transfer function."""

        self._set_rsdrag()

        # EH eq. 7
        self.k_silk = 1.6 * self.omega_b ** 0.52 * self.omega_m ** 0.73 * (1 + (10.4*self.omega_m) ** (-0.95)) # 1/Mpc

        # alpha_c
        # EH eq. 11
        alpha_c_a1 = (46.9*self.omega_m) ** 0.670 * (1 + (32.1*self.omega_m) ** (-0.532))
        alpha_c_a2 = (12.0*self.omega_m) ** 0.424 * (1 + (45.0*self.omega_m) ** (-0.582))
        self.alpha_c = alpha_c_a1 ** (-self.frac_baryon) * alpha_c_a2 ** (-self.frac_baryon**3)

        # beta_c
        # EH eq. 12
        beta_c_b1 = 0.944 / (1 + (458*self.omega_m) ** (-0.708))
        beta_c_b2 = 0.395 * self.omega_m ** (-0.0266)
        self.beta_c = 1. / (1 + beta_c_b1 * ((1-self.frac_baryon) ** beta_c_b2) - 1)

        y = (1 + self.z_eq) / (1 + self.z_drag)
        # EH eq. 15
        alpha_b_G = y * (-6.*np.sqrt(1+y) + (2. + 3.*y) * np.log((np.sqrt(1+y)+1) / (np.sqrt(1+y)-1)))
        self.alpha_b = 2.07 *  self.k_eq * self.rs_drag * (1+self.r_drag)**(-0.75) * alpha_b_G

        # EH eq. 23
        self.beta_node = 8.41 * self.omega_m ** 0.435
        # EH eq. 24
        self.beta_b = 0.5 + self.frac_baryon + (3. - 2.*self.frac_baryon) * np.sqrt( (17.2*self.omega_m) ** 2 + 1)


class Background(BaseSection):
    """
    Background quantities.

    Note
    ----
    Does not treat neutrinos.
    """
    def __init__(self, engine):
        self.engine = engine
        self.H0 = self.engine['H0']
        for name in ['cdm','b','k','g']:
            setattr(self,'Omega0_{}'.format(name),self.engine['Omega_{}'.format(name)])

    @property
    def Omega0_m(self):
        """Current density parameter of matter, unitless."""
        return self.Omega0_cdm + self.Omega0_b

    @property
    def Omega0_Lambda(self):
        """Current density parameter of cosmological constant, unitless."""
        return 1.0 - self.Omega0_m - self.Omega0_g - self.Omega0_k

    def hubble_function(self, z):
        """Hubble function, in :math:`\mathrm{km}/\mathrm{s}/\mathrm{Mpc}`."""
        return self.efunc(z) * self.H0

    def efunc(self, z):
        r"""Function giving :math:`E(z)`, where the Hubble parameter is defined as :math:`H(z) = H_{0} E(z)`, unitless."""
        return np.sqrt(self.Omega0_m * (1 + z)**3 + self.Omega0_g * (1 + z)**4 + self.Omega0_k * (1 + z)**2 + self.Omega0_Lambda)

    def Omega_m(self, z):
        """Density parameter of matter, unitless."""
        return self.Omega0_m * (1+z)**3 / self.efunc(z)**2

    def Omega_Lambda(self, z):
        """Density parameter of cosmological constant, unitless."""
        return self.Omega0_Lambda / self.efunc(z)**2

    def growth_factor(self, z):
        """
        Approximation of growth factor.

        References
        ----------
        https://arxiv.org/abs/astro-ph/9709112, eq. 4
        https://ui.adsabs.harvard.edu/abs/1992ARA%26A..30..499C/abstract, eq. 29
        """
        def growth(z):
            return 1./(1.+z)*5*self.Omega_m(z)/2./(self.Omega_m(z)**(4./7.) - self.Omega_Lambda(z) + (1.+self.Omega_m(z)/2.)*(1.+self.Omega_Lambda(z)/70.))

        return growth(z)/growth(0)

    def growth_rate(self, z):
        """
        Approximation of growth rate.

        References
        ----------
        https://arxiv.org/abs/astro-ph/0507263
        """
        return self.Omega_m(z)**0.55


class Thermodynamics(BaseSection):

    @property
    def rs_drag(self):
        r"""Comoving sound horizon at the baryon drag epoch, in :math:`\mathrm{Mpc}/h`."""
        return self.engine.rs_drag * self.engine['h']

    @property
    def z_drag(self):
        r"""Baryon drag redshift, unitless."""
        return self.engine.z_drag


class Primordial(BaseSection):

    def __init__(self, engine):
        """Initialise :class:`Primordial`."""
        self.engine = engine
        self.n_s = self.engine['n_s']


class Transfer(BaseSection):

    def transfer_k(self, k, frac_baryon=None):
        """
        Return matter transfer function.

        Parameters
        ----------
        k : array_like
            Wavenumbers.

        frac_baryon : float
            If not ``None``, scale the baryon transfer function w.r.t. to the CDM one.
            May be useful to remove BAO.

        Returns
        -------
        transfer : numpy.ndarray
        """
        k = np.asarray(k) * self.engine['h'] # now in 1/Mpc
        # EH eq. 10
        q = k / (13.41*self.engine.k_eq)
        ks = k*self.engine.rs_drag

        T_c_ln_beta = np.log(np.e + 1.8*self.engine.beta_c*q)
        T_c_ln_nobeta = np.log(np.e + 1.8*q);
        T_c_C_alpha = 14.2 / self.engine.alpha_c + 386. / (1 + 69.9 * q ** 1.08)
        T_c_C_noalpha = 14.2 + 386. / (1 + 69.9 * q ** 1.08)

        # EH eq. 18
        T_c_f = 1. / (1. + (ks/5.4) ** 4)
        T0 = lambda a, b : a / (a + b*q**2)
        T_c = T_c_f * T0(T_c_ln_beta, T_c_C_noalpha) + (1-T_c_f) * T0(T_c_ln_beta, T_c_C_alpha)

        # EH eq. 22
        s_tilde = self.engine.rs_drag * (1 + (self.engine.beta_node/ks)**3) ** (-1./3.)
        ks_tilde = k*s_tilde

        # EH eq. 21
        T_b_T0 = T0(T_c_ln_nobeta, T_c_C_noalpha)
        T_b_1 = T_b_T0 / (1 + (ks/5.2)**2 )
        T_b_2 = self.engine.alpha_b / (1 + (self.engine.beta_b/ks)**3 ) * np.exp(-(k/self.engine.k_silk) ** 1.4)
        T_b = np.sinc(ks_tilde/np.pi) * (T_b_1 + T_b_2)

        # EH eq. 16
        frac_baryon = self.engine.frac_baryon if frac_baryon is None else frac_baryon
        return frac_baryon*T_b + (1-frac_baryon)*T_c


class Fourier(BaseSection):

    def __init__(self, engine):
        self.engine = engine
        self.tr = self.engine.get_transfer()
        self.ba = self.engine.get_background()

    def pk_interpolator(self, of='delta_m', ignore_norm=False, **kwargs):
        """
        Return :class:`PowerSpectrumInterpolator2D` instance.

        Parameters
        ----------
        of : string, tuple
            Perturbed quantities: 'delta_m', 'theta_m'.
            No distinction is made between baryons and CDM.
            Requesting velocity divergence 'theta_xx' will rescale the power spectrum by the growth rate as a function of ``z``.

        ignore_norm : bool
            Whether to ignore power spectrum normalisation.
            If ``False``, ``sigma8`` should be provided as part of the parameters.

        kwargs : dict
            Arguments for :class:`PowerSpectrumInterpolator2D`.
        """
        transfer = self.tr.transfer_k

        if not isinstance(of,(tuple,list)):
            of = (of,of)
        ntheta = sum(of_.startswith('theta_') for of_ in of)
        if ntheta:
            def growth_factor_sq(z): return self.ba.growth_factor(z)**2*self.ba.growth_rate(z)**ntheta
        else:
            def growth_factor_sq(z): return self.ba.growth_factor(z)**2

        def pk_callable(k):
            return transfer(k)**2*k**self.engine['n_s']

        toret = PowerSpectrumInterpolator2D.from_callable(pk_callable=pk_callable,growth_factor_sq=growth_factor_sq,**kwargs)
        if not ignore_norm:
            if 'sigma8' in self.engine.params:
                toret.rescale_sigma8(self.engine['sigma8'])
            else:
                raise CosmologyError('A sigma8 value must be provided to normalise EH power spectrum.')
        return toret

    def sigma_rz(self, r, z, of='delta_m', **kwargs):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`r \mathrm{Mpc}/h`. No distinction is made between baryons and CDM."""
        return self.pk_interpolator(of=of,**kwargs).sigma_rz(r,z)

    def sigma8_z(self, z, of='delta_m'):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`8 \mathrm{Mpc}/h`. No distinction is made between baryons and CDM."""
        return self.sigma_rz(8.,z,of=of)

    @property
    def sigma8_m(self):
        r"""Current r.m.s. of matter perturbations in a sphere of :math:`8 \mathrm{Mpc}/h`, unitless."""
        return self.engine['sigma8']
