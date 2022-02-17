import numpy as np

from .cosmology import BaseSection
from . import eisenstein_hu
from .eisenstein_hu import EisensteinHuEngine, Background, Thermodynamics, Primordial, Transfer, Fourier


class EisensteinHuNoWiggleEngine(EisensteinHuEngine):
    """
    Implementation of Eisenstein & Hu no-wiggle analytic formulae.

    References
    ----------
    https://arxiv.org/abs/astro-ph/9709112
    """

    def compute(self):
        """Precompute coefficients for the transfer function."""
        self._set_rsdrag()
        #self.rs_drag = 44.5 * np.log(9.83/self.omega_m) / np.sqrt(1. + 10.*self.omega_b**0.75)
        self.alpha_gamma = 1. - 0.328 * np.log(431.*self.omega_m) * self.frac_baryon + 0.38 * np.log(22.3*self.omega_m) * self.frac_baryon**2


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
        k = np.asarray(k) * self._engine['h'] # now in 1/Mpc
        ks = k * self._engine.rs_drag
        gamma_eff = self._engine.omega_m * (self._engine.alpha_gamma + (1 - self._engine.alpha_gamma) / (1 + (0.43*ks) ** 4))
        q = k * self._engine.theta_cmb**2 / gamma_eff
        L0 = np.log(2*np.e + 1.8 * q)
        C0 = 14.2 + 731.0 / (1 + 62.5 * q)
        return L0 / (L0 + C0 * q**2)
