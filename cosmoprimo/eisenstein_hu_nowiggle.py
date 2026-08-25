import numpy as np

from .cosmology import BaseSection
from .eisenstein_hu import EisensteinHuEngine, Background, Thermodynamics, Primordial, Fourier, CosmologyError


class EisensteinHuNoWiggleEngine(EisensteinHuEngine):
    """
    Implementation of Eisenstein & Hu no-wiggle analytic formulae.

    References
    ----------
    https://arxiv.org/abs/astro-ph/9709112
    """
    name = 'eisenstein_hu_nowiggle'

    def compute(self):
        """Precompute coefficients for the transfer function."""
        self._set_rsdrag()
        # self.rs_drag = 44.5 * np.log(9.83 / self.omega_m) / np.sqrt(1. + 10. * self.omega_b**0.75)
        self.alpha_gamma = 1. - 0.328 * self._np.log(431. * self.omega_m) * self.frac_b + 0.38 * self._np.log(22.3 * self.omega_m) * self.frac_b**2


class Transfer(BaseSection):

    def __init__(self, engine):
        super().__init__(engine)
        self._h = engine['h']
        for name in ['rs_drag', 'omega_m', 'alpha_gamma', 'theta_cmb']:
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
        k = self._np.asarray(k) * self._h  # now in 1/Mpc
        ks = k * self._rs_drag
        gamma_eff = self._omega_m * (self._alpha_gamma + (1 - self._alpha_gamma) / (1 + (0.43 * ks) ** 4))
        q = k * self._theta_cmb**2 / gamma_eff
        L0 = self._np.log(2 * np.e + 1.8 * q)
        C0 = 14.2 + 731.0 / (1 + 62.5 * q)
        return L0 / (L0 + C0 * q**2)
