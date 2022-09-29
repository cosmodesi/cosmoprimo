import warnings

import numpy as np

from .cosmology import BaseEngine, BaseSection
from .eisenstein_hu_nowiggle import Background, Primordial, Fourier


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
        super(BBKSEngine, self).__init__(*args, **kwargs)
        if self['N_ncdm']:
            warnings.warn('{} cannot cope with massive neutrinos'.format(self.__class__.__name__))
        if self['Omega_k'] != 0.:
            warnings.warn('{} cannot cope with non-zero curvature'.format(self.__class__.__name__))
        if self._has_fld:
            warnings.warn('{} cannot cope with non-constant dark energy'.format(self.__class__.__name__))
        self.compute()
        self._A_s = self._get_A_s_fid()

    def compute(self):
        """Precompute coefficients for the transfer function."""
        # 1812.05995 eq. 16
        self.gamma = self['omega_m'] * np.exp(-self['Omega_b'] * (1. + np.sqrt(2. * self['h']) / self['Omega_m']))


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
        q = np.asarray(k) * self._engine['h'] / self._engine.gamma
        x = 2.34 * q
        return np.log(1 + x) / x * (1. + 3.89 * q * (16.2 * q)**2 + (5.47 * q)**3 + (6.71 * q)**4)**(-0.25)
