import warnings

import numpy as np

from .cosmology import BaseEngine, BaseSection
from .eisenstein_hu_nowiggle import Background, Primordial, Fourier
from .jax import exception


class BBKSEngine(BaseEngine):
    """
    Implementation of BBKS no-wiggle analytic formulae.

    References
    ----------
    https://ui.adsabs.harvard.edu/abs/1986ApJ...304...15B/abstract
    https://arxiv.org/abs/astro-ph/9412025
    https://arxiv.org/abs/1812.05995
    """
    name = 'bbks'

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

    def compute(self):
        """Precompute coefficients for the transfer function."""
        # 1812.05995 eq. 16
        self.gamma = self['omega_m'] * self._np.exp(-self['Omega_b'] * (1. + self._np.sqrt(2. * self['h']) / self['Omega_m']))


class Transfer(BaseSection):

    def __init__(self, engine):
        super().__init__(engine)
        self._h = engine['h']
        for name in ['gamma']:
            setattr(self, '_' + name, getattr(engine, name))

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
        q = self._np.asarray(k) * self._h / self._gamma
        x = 2.34 * q
        return self._np.log(1 + x) / x * (1. + 3.89 * q * (16.2 * q)**2 + (5.47 * q)**3 + (6.71 * q)**4)**(-0.25)
