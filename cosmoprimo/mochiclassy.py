"""Cosmological calculation with the Boltzmann code MochiCLASS."""

import numpy as np

from pyclass import mochiclass

from .cosmology import BaseEngine, CosmologyInputError, CosmologyComputationError
from . import classy


# Input parameters that switch on the scalar-modified-gravity (smg) sector.
_smg_parameters = ('gravity_model', 'expansion_model', 'parameters_smg', 'expansion_smg', 'Omega_smg')

# Verbosity at which mochi_class writes the alpha-functions, M_*^2 and the EFT
# combinations (cs2num, lambda_i) to the background table; required by
# :meth:`Background.h1`, :meth:`Background.h3` and :meth:`Background.h5`.
_output_background_smg = 3


class MochiClassEngine(classy.ClassEngine):

    """Engine for the Boltzmann code mochiclass."""

    name = 'mochiclass'

    _default_cosmological_parameters = dict()

    def _set_classy(self, params):

        if any(name in params for name in _smg_parameters):
            # Only raise verbosity, never lower a user-provided value.
            params = dict(params)
            params['output_background_smg'] = max(int(params.get('output_background_smg', 0)), _output_background_smg)

        class _ClassEngine(mochiclass.ClassEngine):

            def compute(self, tasks):
                try:
                    return super(_ClassEngine, self).compute(tasks)
                except mochiclass.ClassInputError as exc:
                    raise CosmologyInputError from exc
                except mochiclass.ClassComputationError as exc:
                    raise CosmologyComputationError from exc

        self.classy = _ClassEngine(params=params)


def _flatarray(func):
    """Decorator to make ``func(self, x)`` accept any array shape (and scalars)."""
    from functools import wraps

    @wraps(func)
    def wrapper(self, x, *args, **kwargs):
        x = np.asarray(x, dtype='f8')
        toret = func(self, x.ravel(), *args, **kwargs)
        if x.ndim == 0:
            return toret[0]
        return toret.reshape(x.shape)

    return wrapper


class Background(classy.BaseClassBackground, mochiclass.Background):

    r"""
    Background quantities, including the Horndeski / EFT-of-dark-energy functions
    :math:`h_{1}`, :math:`h_{3}` and :math:`h_{5}` of `arXiv:1902.06978
    <https://arxiv.org/abs/1902.06978>`_ (eqs. 64, 66, 68), which parameterize the
    quasi-static effective Newton constant

    .. math:: Y(k, z) = h_{1} \frac{1 + k^{2} h_{5}}{1 + k^{2} h_{3}}.

    :math:`h_{1}`, :math:`h_{3}` and :math:`h_{5}` are functions of
    :math:`\eta = \ln a = -\ln (1 + z)`, the time variable the underlying splines are
    tabulated in and the one Horndeski / EFT-of-dark-energy codes (e.g. fkptjax) integrate
    in.  :meth:`Y` keeps taking a redshift.

    These require the smg sector to be switched on, e.g.::

        cosmo = AbacusSummit(0, engine='mochiclass', Omega_Lambda=0, Omega_fld=0, Omega_smg=-1,
                             gravity_model='propto_omega', parameters_smg='1., 0.5, 0.3, 0., 1.',
                             expansion_model='wowa', expansion_smg='0.685, -1., 0.')
        cosmo.h1(eta), cosmo.h3(eta), cosmo.h5(eta), cosmo.Y(k, z)
    """

    def _eft_of_de(self):
        r"""
        Return dict of cubic splines, as a function of :math:`\ln a`, of the background
        quantities entering eqs. (62) - (69) of arXiv:1902.06978.

        All ingredients are taken directly from the mochi_class background table, so no
        finite differencing of the alpha-functions is involved. Using primes for
        :math:`d / d\ln a`, the required identities are:

        - :math:`\xi \equiv H^{\prime} / H = -\frac{3}{2} (\rho_\mathrm{tot} + p_\mathrm{tot}) / H^{2}`
        - :math:`\alpha_{2}` (eq. 63) is mochi_class' ``lambda_2``, and mochi_class'
          ``cs2num`` :math:`= (2 - \alpha_{B}) \alpha_{1} / 2 + \alpha_{2}` is exactly
          the numerator of :math:`h_{3}`;
        - :math:`2 \xi^{2} + \xi^{\prime} + \xi (3 + \alpha_{M}) = \xi \alpha_{M} - \frac{3}{2} p_\mathrm{tot}^{\prime} / H^{2}`,
          which removes the need for :math:`\xi^{\prime}` (hence :math:`H^{\prime\prime}`) in :math:`\mu^{2}` (eq. 69).
          The :math:`3/2` (rather than :math:`1/2`) is CLASS' ``(.)`` convention: the table stores
          :math:`8 \pi G p / 3`.  Note :math:`p_\mathrm{tot}^{\prime}` must include the smg fluid;
          see the comment on ``dp_over_H2`` below.
        """
        if getattr(self, '_eft_of_de_splines', None) is None:
            from scipy import interpolate

            table = self.table()
            names = table.dtype.names or ()
            for name in ('braiding_smg', 'cs2num', '(.)p_smg_prime'):
                if name not in names:
                    raise CosmologyInputError('mochi_class did not output "{}"; h1 / h3 / h5 require the smg sector, '
                                              'i.e. one of {} among the engine parameters'.format(name, _smg_parameters))
            z = table['z']
            a = 1. / (1. + z)
            H = table['H [1/Mpc]']  # H / c, in 1 / Mpc
            H2 = H**2
            di = {'M2': table['M*^2_smg'],
                  'alpha_B': table['braiding_smg'],
                  'alpha_M': table['Mpl_running_smg'],
                  'alpha_T': table['tensor_excess_smg'],
                  'cs2num': table['cs2num'],
                  # xi = H' / H
                  'xi': -1.5 * (table['(.)rho_tot'] + table['(.)p_tot']) / H2,
                  # p_tot' / H^2, with ' = d / dln a and (.)p_tot_prime = dp_tot / dtau.
                  # mochi_class' (.)p_tot_prime EXCLUDES the smg fluid: background_functions()
                  # accumulates dp_dloga species by species (source/background.c, l. 432 - 568)
                  # and the smg correction is commented out at source/background.c:2517, with
                  # the note "Never used anywhere in hiclass, because here only matter
                  # contributions without smg are considered in dp_dlna". The missing piece is
                  # tabulated separately as (.)p_smg_prime, in the same d / dtau convention
                  # (factor = a * H in gravity_smg/background_smg.c:2406), and mochi_class' own
                  # mu_p uses exactly this sum (background_smg.c:1249).
                  #
                  # It vanishes identically for w = -1, which is why h3 / h5 were right there
                  # and 50%-level wrong for any w != -1; with the term restored they match the
                  # 'heftcamb' engine to ~5e-3 % (see Projects/DESI/DESI-DR2-MG/FullShape/
                  # h1h3h5_HEFTCAMB_vs_mochiclass_v2.ipynb).
                  'dp_over_H2': (table['(.)p_tot_prime'] + table['(.)p_smg_prime']) / (a * H) / H2,
                  # a H / c, in h / Mpc
                  'aH': a * H / self.h}
            loga = np.log(a)
            argsort = np.argsort(loga)  # the table is tabulated in increasing ln(a), but let's be safe
            loga = loga[argsort]
            self._eft_of_de_splines = {name: interpolate.CubicSpline(loga, value[argsort], extrapolate=False)
                                       for name, value in di.items()}
        return self._eft_of_de_splines

    def _eft_of_de_at_eta(self, eta):
        r"""Evaluate :meth:`_eft_of_de` at :math:`\eta = \ln a`, and add the derived alpha_1, alpha_2, mu^2."""
        toret = {name: spline(eta) for name, spline in self._eft_of_de().items()}
        aB, aM, aT = toret['alpha_B'], toret['alpha_M'], toret['alpha_T']
        # eq. 62
        toret['alpha_1'] = alpha_1 = aB + (aB - 2.) * aT + 2. * aM
        # eq. 63, mochi_class' lambda_2 (equivalently, cs2num - (2 - alpha_B) alpha_1 / 2)
        toret['alpha_2'] = alpha_2 = toret['cs2num'] - (2. - aB) * alpha_1 / 2.
        # eq. 69
        toret['mu2'] = -3. * (aB * (toret['xi'] * aM - 1.5 * toret['dp_over_H2']) + toret['xi'] * alpha_2)
        return toret

    @_flatarray
    def h1(self, eta):
        r"""
        :math:`h_{1} = (1 + \alpha_{T}) / M_{\ast}^{2}`, eq. 64 of arXiv:1902.06978, unitless,
        as a function of :math:`\eta = \ln a`.
        """
        bg = self._eft_of_de_at_eta(eta)
        return (1. + bg['alpha_T']) / bg['M2']

    @_flatarray
    def h3(self, eta):
        r"""
        :math:`h_{3} = \left[(2 - \alpha_{B}) \alpha_{1} + 2 \alpha_{2}\right] / (2 a^{2} H^{2} \mu^{2})`,
        eq. 66 of arXiv:1902.06978, in :math:`(\mathrm{Mpc}/h)^{2}` (i.e. for :math:`k` in :math:`h/\mathrm{Mpc}`),
        as a function of :math:`\eta = \ln a`.
        """
        bg = self._eft_of_de_at_eta(eta)
        return bg['cs2num'] / (bg['aH']**2 * bg['mu2'])

    @_flatarray
    def h5(self, eta):
        r"""
        :math:`h_{5} = \left[\frac{1 + \alpha_{M}}{1 + \alpha_{T}} \alpha_{1} + \alpha_{2}\right] / (a^{2} H^{2} \mu^{2})`,
        eq. 68 of arXiv:1902.06978, in :math:`(\mathrm{Mpc}/h)^{2}` (i.e. for :math:`k` in :math:`h/\mathrm{Mpc}`),
        as a function of :math:`\eta = \ln a`.
        """
        bg = self._eft_of_de_at_eta(eta)
        numerator = (1. + bg['alpha_M']) / (1. + bg['alpha_T']) * bg['alpha_1'] + bg['alpha_2']
        return numerator / (bg['aH']**2 * bg['mu2'])

    def Y(self, k, z):
        r"""
        Quasi-static effective Newton constant :math:`Y = h_{1} (1 + k^{2} h_{5}) / (1 + k^{2} h_{3})`,
        eq. 24 of arXiv:1902.06978, unitless.

        Parameters
        ----------
        k : array_like
            Wavenumbers, in :math:`h/\mathrm{Mpc}`.

        z : array_like
            Redshifts.  Note :meth:`h1`, :meth:`h3` and :meth:`h5` themselves take
            :math:`\eta = \ln a`; the conversion is done here.

        Returns
        -------
        Y : array
            Array of shape ``(k.shape, z.shape)``.
        """
        k, z = np.asarray(k, dtype='f8'), np.asarray(z, dtype='f8')
        k2 = k.reshape(k.shape + (1,) * z.ndim)**2
        eta = -np.log(1. + z)
        return self.h1(eta) * (1. + k2 * self.h5(eta)) / (1. + k2 * self.h3(eta))


class Thermodynamics(classy.BaseClassThermodynamics, mochiclass.Thermodynamics):

    """Your modifications, if any."""


class Primordial(classy.BaseClassPrimordial, mochiclass.Primordial):

     """Your modifications, if any."""


class Perturbations(classy.BaseClassPerturbations, mochiclass.Perturbations):

     """Your modifications, if any."""


class Transfer(classy.BaseClassTransfer, mochiclass.Transfer):

     """Your modifications, if any."""


class Harmonic(classy.BaseClassHarmonic, mochiclass.Harmonic):
     """Your modifications, if any."""


class Fourier(classy.BaseClassFourier, mochiclass.Fourier):
     """Your modifications, if any."""
