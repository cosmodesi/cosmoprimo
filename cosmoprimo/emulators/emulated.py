"""Cosmological calculation with the emulator."""

import numpy as np
from cosmoprimo.interpolator import PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D, get_default_k_callable, get_default_z_callable

from cosmoprimo.cosmology import BaseEngine, BaseSection, BaseBackground, CosmologyInputError, CosmologyComputationError
from cosmoprimo.interpolator import PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D
from cosmoprimo import utils
from . import Emulator


class EmulatedEngine(BaseEngine):

    """Engine using emulator."""
    name = 'emulated'
    path = None

    def __init__(self, *args, **kwargs):
        super(EmulatedEngine, self).__init__(*args, **kwargs)
        emulator = getattr(self.__class__, '_emulator', None)
        if emulator is None:
            emulator = self.__class__._emulator = Emulator.load(self.path)
        self._state = emulator.predict(**{param: self[param] for param in self._emulator.params})


def get_section_state(state, section='background'):
    section = section + '.'
    toret = {}
    for name, value in state.items():
        if name.startswith(section):
            toret[name[len(section):]] = value
    return toret


class Background(BaseBackground):

    """Tabulated background quantities."""

    def __init__(self, engine):
        super(Background, self).__init__(engine=engine)
        self.__setstate__(get_section_state(engine._state, section='background'))

    @utils.flatarray(dtype=np.float64)
    def time(self, z):
        r"""Proper time (age of universe), in :math:`\mathrm{Gy}`."""
        return self._state['time'](z)

    @utils.flatarray(dtype=np.float64)
    def comoving_radial_distance(self, z):
        r"""
        Comoving radial distance, in :math:`mathrm{Mpc}/h`.

        See eq. 15 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_C(z)`.
        """
        return self._state['comoving_radial_distance'](z)

    @utils.flatarray(dtype=np.float64)
    def angular_diameter_distance(self, z):
        r"""
        Proper angular diameter distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 18 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{A}(z)`.
        """
        return self._state['angular_diameter_distance'](z)

    @utils.flatarray(iargs=[0, 1], dtype=np.float64)
    def angular_diameter_distance_2(self, z1, z2):
        r"""
        Angular diameter distance of object at :math:`z_{2}` as seen by observer at :math:`z_{1}`,
        that is, :math:`S_{K}((\chi(z_{2}) - \chi(z_{1})) \sqrt{|K|}) / \sqrt{|K|} / (1 + z_{2})`,
        where :math:`S_{K}` is the identity if :math:`K = 0`, :math:`\sin` if :math:`K < 0`
        and :math:`\sinh` if :math:`K > 0`.
        camb's ``angular_diameter_distance2(z1, z2)`` is not used as it returns 0 when z2 < z1.
        """
        if np.any(z2 < z1):
            import warnings
            warnings.warn(f"Second redshift(s) z2 ({z2}) is less than first redshift(s) z1 ({z1}).")
        chi1, chi2 = self.comoving_radial_distance(z1), self.comoving_radial_distance(z2)
        K = self.K  # in (h/Mpc)^2
        if K == 0:
            return (chi2 - chi1) / (1 + z2)
        elif K > 0:
            return np.sin(np.sqrt(K) * (chi2 - chi1)) / np.sqrt(K) / (1 + z2)
        return np.sinh(np.sqrt(-K) * (chi2 - chi1)) / np.sqrt(-K) / (1 + z2)

    @utils.flatarray(dtype=np.float64)
    def comoving_angular_distance(self, z):
        r"""
        Comoving angular distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 16 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{M}(z)`.
        """
        return self.angular_diameter_distance(z) * (1. + z)

    @utils.flatarray(dtype=np.float64)
    def luminosity_distance(self, z):
        r"""
        Luminosity distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 21 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{L}(z)`.
        """
        return self.angular_diameter_distance(z) * (1. + z)**2

    def __getstate__(self):
        state = {}
        state['z'] = z = get_default_z_callable()
        for name in ['time', 'comoving_radial_distance', 'angular_diameter_distance']:
            state[name] = getattr(self, name)(z)

    def __setstate__(self, state):
        from scipy import interpolate
        state = dict(state)
        z = state.pop('z')
        for name, value in state.items():
            state[name] = interpolate.interp1d(z, value, kind='cubic', bounds_error=True, assume_sorted=True)
        self._state = state


class Transfer(BaseSection):

    def __init__(self, engine):
        self.__setstate__(engine._state, section='transfer')
        self._engine = engine

    def table(self):
        r"""Return source functions (in array of shape (k.size, z.size))."""
        return self._table['table']

    def __getstate__(self):
        return {'state': self.table()}


@utils.addproperty('n_s', 'alpha_s', 'beta_s')
class Primordial(BaseSection):

    def __init__(self, engine):
        """Initialize :class:`Primordial`."""
        self._state = get_section_state(engine._state, section='primordial')
        self._engine = engine
        self._h = self._engine['h']
        self._A_s = self._engine._A_s
        self._n_s = self._engine['n_s']
        self._alpha_s = self._engine['alpha_s']
        self._beta_s = self._engine['beta_s']
        self._rsigma8 = self._engine._rescale_sigma8()

    @property
    def A_s(self):
        r"""Scalar amplitude of the primordial power spectrum at :math:`k_\mathrm{pivot}`, unitless."""
        return self._A_s * self._rsigma8**2

    @property
    def ln_1e10_A_s(self):
        r""":math:`\ln(10^{10}A_s)`, unitless."""
        return np.log(1e10 * self.A_s)

    @property
    def k_pivot(self):
        r"""Primordial power spectrum pivot scale, where the primordial power is equal to :math:`A_{s}`, in :math:`h/\mathrm{Mpc}`."""
        return self._engine['k_pivot'] / self._h

    def pk_k(self, k, mode='scalar'):
        r"""
        The primordial spectrum of curvature perturbations at ``k``, generated by inflation, in :math:`(\mathrm{Mpc}/h)^{3}`.
        For scalar perturbations this is e.g. defined as:

        .. math::

            \mathcal{P_R}(k) = A_s \left (\frac{k}{k_\mathrm{pivot}} \right )^{n_s - 1 + 1/2 \alpha_s \ln(k/k_\mathrm{pivot}) + 1/6 \beta_s \ln(k/k_\mathrm{pivot})^2}

        See also: eq. 2 of `this reference <https://arxiv.org/abs/1303.5076>`_.

        Parameters
        ----------
        k : array_like
            Wavenumbers, in :math:`h/\mathrm{Mpc}`.

        mode : string, default='scalar'
            'scalar' mode.

        Returns
        -------
        pk : array
            The primordial power spectrum.
        """
        index = ['scalar'].index(mode)
        lnkkp = np.log(k / self.k_pivot)
        return self._h**3 * self.A_s * (k / self.k_pivot) ** (self.n_s - 1. + 1. / 2. * self.alpha_s * lnkkp + 1. / 6. * self.beta_s * lnkkp**2)

    def pk_interpolator(self, mode='scalar'):
        """
        Return power spectrum interpolator.

        Parameters
        ----------
        mode : string, default='scalar'
            'scalar', 'vector' or 'tensor' mode.

        Returns
        -------
        interp : PowerSpectrumInterpolator1D
        """
        return PowerSpectrumInterpolator1D.from_callable(pk_callable=lambda k: self.pk_k(k, mode=mode))


class Harmonic(BaseSection):

    def __init__(self, engine):
        self.__setstate__(engine._state, section='harmonic')
        self._engine = engine
        self._rsigma8 = self._engine._rescale_sigma8()
        self.ellmax_cl = self._engine['ellmax_cl']

    def unlensed_cl(self, ellmax=-1):
        r"""Return unlensed :math:`C_{\ell}` ['tt', 'ee', 'bb', 'te'], unitless."""
        return self._state['unlensed_cl'][:ellmax] * self._rsigma8**2

    def lens_potential_cl(self, ellmax=-1):
        r"""Return potential :math:`C_{\ell}` ['pp', 'tp', 'ep'], unitless."""
        return self._state['lens_potential_cl'][:ellmax] * self._rsigma8**2

    def lensed_cl(self, ellmax=-1):
        r"""Return lensed :math:`C_{\ell}` ['tt', 'ee', 'bb', 'te'], unitless."""
        return self._state['lensed_cl'][:ellmax] * self._rsigma8**2

    def __getstate__(self):
        """Return this class' state dictionary."""
        state = {}
        for name in ['unlensed_cl', 'lens_potential_cl', 'lensed_cl']:
            table = getattr(self, name)(ellmax=self.ellmax_cl)
            for key in table.dtype.names:
                state['{}.{}'.format(name, key)] = table[key]
        return state

    def __setstate__(self, state):
        """Set this class' state dictionary."""
        self._state = dict(state)
        tables = {}
        for keyname, value in state.items():
            name, key = keyname.split('.')
            tables.setdefault(name, {})
            tables[name][key] = value
        for name, value in tables.items():
            names = list(value.keys())
            self._state[name] = table = np.empty(value[names[0]].shape[0], [('ell', np.int64)] + [(name, np.float64) for name in names])
            for name in names: table[name] = value[name]
            table['ell'] = np.arange(table.shape[0])


@utils.addproperty('sigma8_m')
class Fourier(BaseSection):

    def __init__(self, engine):
        self.__setstate__(engine._state, section='fourier')
        self._engine = engine
        self._h = self._engine['h']
        self._rsigma8 = self._engine._rescale_sigma8()
        self._sigma8_m = self.sigma8_z(0., of='delta_m')

    def sigma_rz(self, r, z, of='delta_m', **kwargs):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`r \mathrm{Mpc}/h`."""
        return self.pk_interpolator(non_linear=False, of=of, **kwargs).sigma_rz(r, z)

    def sigma8_z(self, z, of='delta_m'):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`8 \mathrm{Mpc}/h`."""
        return self.sigma_rz(8., z, of=of)

    def table(self, non_linear=False, of='m'):
        r"""
        Return power spectrum table, in :math:`(\mathrm{Mpc}/h)^{3}`.

        Parameters
        ----------
        non_linear : bool, default=False
            Whether to return the non_linear power spectrum (if requested in parameters, with 'non_linear': 'halofit' or 'mead').
            Computed only for of == 'delta_m' or 'delta_cb'.

        of : string, tuple, default='delta_m'
            Perturbed quantities.
            'delta_m' for matter perturbations, 'delta_cb' for cold dark matter + baryons, 'phi', 'psi' for Bardeen potentials, or 'phi_plus_psi' for Weyl potential.
            Provide a tuple, e.g. ('delta_m', 'theta_cb') for the cross matter density - cold dark matter + baryons velocity power spectra.

        Returns
        -------
        k : numpy.ndarray
            Wavenumbers.

        z : numpy.ndarray
            Redshifts.

        pk : numpy.ndarray
            Power spectrum array of shape (len(k), len(z)).
        """
        if isinstance(of, str): of = (of,)
        of = list(of)
        of = of + [of[0]] * (2 - len(of))
        of = tuple(sorted(of))
        return self._state['k'], self._state['z'], self._state['pk_non_linear' if non_linear else 'pk'][of] * self._rsigma8**2

    def pk_interpolator(self, non_linear=False, of='delta_m', **kwargs):
        r"""
        Return :class:`PowerSpectrumInterpolator2D` instance.

        Parameters
        ----------
        non_linear : bool, default=False
            Whether to return the non_linear power spectrum (if requested in parameters, with 'non_linear': 'halofit' or 'mead').
            Computed only for of == 'delta_m'.

        of : string, tuple, default='delta_m'
            Perturbed quantities.

        kwargs : dict
            Arguments for :class:`PowerSpectrumInterpolator2D`.
        """
        ka, za, pka = self.table(non_linear=non_linear, of=of)
        return PowerSpectrumInterpolator2D(ka, za, pka, **kwargs)

    def pk_kz(self, k, z, non_linear=False, of='delta_m'):
        r"""
        Return power spectrum, in :math:`(\mathrm{Mpc}/h)^{3}`.

        Parameters
        ----------
        k : array_like
            Wavenumbers, in :math:`h/\mathrm{Mpc}`.

        z : array_like
            Redshifts.

        non_linear : bool, default=False
            Whether to return the non_linear power spectrum (if requested in parameters, with 'non_linear': 'halofit' or 'mead').

        of : string, default='delta_m'
            Perturbed quantities.

        Returns
        -------
        pk : array
            Power spectrum array of shape (len(k),len(z)).
        """
        interp = self.pk_interpolator(non_linear=non_linear, of=of)
        return interp(k, z)

    def __getstate__(self):
        """Return this class' state dictionary."""
        state = {}
        state['k'] = k = get_default_k_callable()
        state['z'] = z = get_default_z_callable()
        try: state['pk_non_linear'] = {of: self.pk_interpolator(non_linear=True, of=of)(k, z) for of in [('delta_m', 'delta_m')]}
        except: pass
        list_of = []
        ofs = ['delta_cb', 'delta_m', 'theta_cb', 'theta_m', 'phi_plus_psi']
        for iof1, of1 in enumerate(ofs):
            for of2 in ofs[iof1:]:
                list_of.append(tuple(sorted((of1, of2))))
        state['pk'] = {}
        for of in list_of:
            try: state['pk'][of] = self.pk_interpolator(non_linear=False, of=of)(k, z)
            except: pass
        return state

    def __setstate__(self, state):
        """Set this class' state dictionary."""
        self._state = dict(state)