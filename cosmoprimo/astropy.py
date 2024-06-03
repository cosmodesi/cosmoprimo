import numpy as np
from astropy import units
from astropy import cosmology as astropy_cosmology

from .cosmology import BaseEngine, BaseBackground
from . import constants, utils


class AstropyEngine(BaseEngine):

    """Wrapper on astropy cosmology engine."""
    name = 'astropy'

    def __init__(self, *args, **kwargs):
        super(AstropyEngine, self).__init__(*args, **kwargs)
        N_eff = self['N_eff']
        m_nu = self['m_ncdm']
        m_nu = list(m_nu) + [0.] * (int(N_eff) - len(m_nu))
        kwargs = {'H0': self['H0'], 'Om0': self['Omega_b'] + self['Omega_cdm'],
                  'Tcmb0': self['T_cmb'], 'Neff': N_eff, 'm_nu': units.Quantity(m_nu, units.eV), 'Ob0': self['Omega_b']}
        name = 'CDM'
        if self['wa_fld'] != -1:
            name = 'wa{}'.format(name)
            kwargs['wa'] = self['wa_fld']
        if self['w0_fld'] != 0:
            kwargs['w0'] = self['w0_fld']
            if self['wa_fld'] != -1:
                name = 'w0{}'.format(name)  # w0wa model
            else:
                name = 'w{}'.format(name)  # w model
        if self['Omega_k'] == 0:
            name = 'Flat{}'.format(name)
        else:
            kwargs['Ode0'] = 1 - (self['Omega_b'] + self['Omega_cdm'])  # this is a first guess for OdeO because neutrino treatment...
            self._astropy = getattr(astropy_cosmology, name)(**kwargs)
            # now adjust Ode0 based on Omega_k
            kwargs['Ode0'] = 1.0 - self._astropy.Om0 - self['Omega_k'] - self._astropy.Ogamma0 - self._astropy.Onu0

        self._astropy = getattr(astropy_cosmology, name)(**kwargs)


class Background(BaseBackground):
    r"""
    Background quantities.

    Note
    ----
    In astropy, neutrinos (even massive) are treated like radiation, relative to the photon density,
    see Komatsu et al. 2011, eq 26.
    Then :math:`\Omega_{\nu}`, :math:`\Omega_{\mathrm{ncdm}}` and :math:`\Omega_{m}` do not match our definitions,
    hence we do not include them here.
    """
    def __init__(self, engine):
        super(Background, self).__init__(engine=engine)
        self.ba = self._engine._astropy

    @property
    def age(self):
        r"""The current age of the Universe, in :math:`\mathrm{Gy}`."""
        return self.ba.age(0.).value

    @utils.flatarray()
    def Omega_k(self, z):
        r"""Density parameter of curvature, unitless."""
        return self.ba.Ok(z)

    @utils.flatarray()
    def Omega_cdm(self, z):
        r"""Density parameter of cold dark matter, unitless."""
        return self.ba.Odm(z)

    @utils.flatarray()
    def Omega_b(self, z):
        r"""Density parameter of baryons, unitless."""
        return self.ba.Ob(z)

    @utils.flatarray()
    def Omega_g(self, z):
        r"""Density parameter of photons, unitless."""
        return self.ba.Ogamma(z)

    @utils.flatarray()
    def Omega_de(self, z):
        r"""Density of total dark energy, unitless."""
        return self.ba.Ode(z)

    @utils.flatarray()
    def rho_crit(self, z):
        r"""
        Comoving critical density excluding curvature :math:`\rho_{c}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`.

        This is defined as:

        .. math::

              \rho_{\mathrm{crit}}(z) = \frac{3 H(z)^{2}}{8 \pi G}.
        """
        # astropy in g/cm3
        return self.ba.critical_density(z).value * 1e3 / (1e10 * constants.msun) * constants.megaparsec**3 / self.h**2 / (1 + z)**3

    @utils.flatarray()
    def efunc(self, z):
        r"""Function giving :math:`E(z)`, where the Hubble parameter is defined as :math:`H(z) = H_{0} E(z)`, unitless."""
        return self.ba.efunc(z)

    @utils.flatarray()
    def time(self, z):
        r"""Proper time (age of universe), in :math:`\mathrm{Gy}`."""
        if z.size:  # required to avoid error in np.vectorize
            return self.ba.age(z).value
        return np.zeros_like(z)

    @utils.flatarray()
    def comoving_radial_distance(self, z):
        r"""
        Comoving radial distance, in :math:`mathrm{Mpc}/h`.

        See eq. 15 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_C(z)`.
        """
        if z.size:
            return self.ba.comoving_distance(z).value * self.h
        return np.zeros_like(z)

    @utils.flatarray()
    def luminosity_distance(self, z):
        r"""
        Luminosity distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 21 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{L}(z)`.
        """
        if z.size:
            return self.ba.luminosity_distance(z).value * self.h
        return np.zeros_like(z)

    @utils.flatarray()
    def angular_diameter_distance(self, z):
        r"""
        Proper angular diameter distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 18 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{A}(z)`.
        """
        if z.size:
            return self.ba.angular_diameter_distance(z).value * self.h
        return np.zeros_like(z)

    @utils.flatarray(iargs=[0, 1])
    def angular_diameter_distance_2(self, z1, z2):
        r"""
        Angular diameter distance of object at :math:`z_{2}` as seen by observer at :math:`z_{1}`,
        that is, :math:`S_{K}((\chi(z_{2}) - \chi(z_{1})) \sqrt{|K|}) / \sqrt{|K|} / (1 + z_{2})`,
        where :math:`S_{K}` is the identity if :math:`K = 0`, :math:`\sin` if :math:`K < 0`
        and :math:`\sinh` if :math:`K > 0`.
        """
        if z1.size:
            return self.ba.angular_diameter_distance_z1z2(z1, z2).value * self.h
        return np.zeros_like(z1)

    @utils.flatarray()
    def comoving_angular_distance(self, z):
        r"""
        Comoving angular distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 16 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{M}(z)`.
        """
        if z.size:
            return self.angular_diameter_distance(z) * (1. + z)
        return np.zeros_like(z)
