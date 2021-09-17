import warnings

import numpy as np
import astropy
from astropy import units
from astropy import cosmology as astropy_cosmology

from .cosmology import BaseEngine, BaseSection, BaseBackground, CosmologyError


class AstropyEngine(BaseEngine):

    """Wrapper on astropy cosmology engine."""

    def __init__(self, *args, **kwargs):
        super(AstropyEngine,self).__init__(*args,**kwargs)
        if self.params.get('Omega_Lambda',None) is not None:
            warnings.warn('{} cannot cope with dynamic dark energy + cosmological constant'.format(self.__class__.__name__))
        N_eff = self['N_eff']
        m_nu = self['m_ncdm']
        m_nu = m_nu + [0.]*(int(N_eff) - len(m_nu))
        kwargs = {'H0':self['H0'],'Om0':self['Omega_b']+self['Omega_cdm'],
                'Tcmb0':self['T_cmb'],'Neff':N_eff,'m_nu':units.Quantity(m_nu, units.eV),'Ob0':self['Omega_b']}
        name = 'CDM'
        if self['wa_fld'] != -1:
            name = 'wa{}'.format(name)
            kwargs['wa'] = self['wa_fld']
        if self['w0_fld'] != 0:
            kwargs['w0'] = self['w0_fld']
            if self['wa_fld'] != -1:
                name = 'w0{}'.format(name) # w0wa model
            else:
                name = 'w{}'.format(name) # w model
        if self['Omega_k'] == 0:
            name = 'Flat{}'.format(name)
        else:
            kwargs['Ode0'] = 1-(self['Omega_b']+self['Omega_cdm']) # this is a first guess for OdeO because neutrino treatment...
            self._astropy = getattr(astropy_cosmology,name)(**kwargs)
            # now adjust Ode0 based on Omega_k
            kwargs['Ode0'] = 1.0 - self._astropy.Om0 - self['Omega_k'] - self._astropy.Ogamma0 - self._astropy.Onu0

        self._astropy = getattr(astropy_cosmology,name)(**kwargs)
        #print(name,kwargs)


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
        super(Background,self).__init__(engine=engine)
        self.ba = self.engine._astropy

    def Omega_k(self, z):
        r"""Density parameter of curvature, unitless."""
        return self.ba.Ok(z)

    def Omega_cdm(self, z):
        r"""Density parameter of cold dark matter, unitless."""
        return self.ba.Odm(z)

    def Omega_b(self, z):
        r"""Density parameter of baryons, unitless."""
        return self.ba.Ob(z)

    def Omega_g(self, z):
        r"""Density parameter of photons, unitless."""
        return self.ba.Ogamma(z)

    def Omega_de(self, z):
        r"""Density of total dark energy, unitless."""
        return self.ba.Ode(z)

    def rho_crit(self, z):
        r"""
        Comoving critical density excluding curvature :math:`\rho_{c}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`.

        This is defined as:

        .. math::

              \rho_{\mathrm{crit}}(z) = \frac{3 H(z)^{2}}{8 \pi G}.
        """
        # astropy in g/cm3
        return self.ba.critical_density(z).value * 1e3 / (1e10*constants.msun) * constants.megaparsec**3

    def efunc(self, z):
        r"""Function giving :math:`E(z)`, where the Hubble parameter is defined as :math:`H(z) = H_{0} E(z)`, unitless."""
        return self.ba.efunc(z)

    def time(self, z):
        r"""Proper time (age of universe), in :math:`\mathrm{Gy}`."""
        return self.ba.age(z).value

    def comoving_radial_distance(self, z):
        r"""
        Comoving radial distance, in :math:`mathrm{Mpc}/h`.

        See eq. 15 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_C(z)`.
        """
        return self.ba.comoving_distance(z).value * self.h

    def luminosity_distance(self, z):
        r"""
        Luminosity distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 21 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{L}(z)`.
        """
        return self.ba.luminosity_distance(z).value * self.h

    def angular_diameter_distance(self, z):
        r"""
        Proper angular diameter distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 18 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{A}(z)`.
        """
        return self.ba.angular_diameter_distance(z).value * self.h

    def comoving_angular_distance(self, z):
        r"""
        Comoving angular distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 16 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{M}(z)`.
        """
        return self.angular_diameter_distance(z) * (1. + z)
