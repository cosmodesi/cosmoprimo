"""Emulated cosmological calculation."""

import os
from pathlib import Path

import numpy as np
from scipy import interpolate

from cosmoprimo.interpolator import PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D
from cosmoprimo.cosmology import BaseEngine, BaseSection, BaseBackground, CosmologyInputError, CosmologyComputationError, find_conflicts
from cosmoprimo.jax import Interpolator1D
from cosmoprimo.interpolator import PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D
from cosmoprimo import utils, Cosmology, CosmologyError


def get_default_k_callable():
    # Taken from https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/training/spectra_generation_scripts/2_create_spectra.py
    k = np.concatenate([np.array([1e-6]),
                        np.logspace(-5, -4, num=20, endpoint=False),
                        np.logspace(-4, -3, num=40, endpoint=False),
                        np.logspace(-3, -2, num=60, endpoint=False),
                        np.logspace(-2, -1, num=80, endpoint=False),
                        np.logspace(-1, 0, num=100, endpoint=False),
                        np.logspace(0, 1, num=120, endpoint=True),
                        np.array([1e2])])
    return k


def get_default_z_callable(key='fourier', non_linear=False):
    if 'background' in key:
        return 1. / np.logspace(-3, 0., 256)[::-1] - 1.
    z = np.linspace(0., 10.**0.5, 30)**2  # approximates default class z
    if non_linear:
        return z[z < 2.]
    return z


class EmulatedEngine(BaseEngine):

    """Engine using emulator."""
    name = 'emulated'
    path = None

    def __init__(self, *args, **kwargs):
        super(EmulatedEngine, self).__init__(*args, **kwargs)
        emulator = getattr(self.__class__, '_emulator', None)
        if emulator is None:
            from . import Emulator
            emulator = Emulator()
            if not isinstance(self.path, dict):
                self.path = {str(self.path): None}
            for path, url in self.path.items():
                if not os.path.exists(path):
                    from cosmoprimo.emulators.tools.utils import download
                    download(url, path)
                emulator.update(Emulator.load(path))
            self.__class__._emulator = emulator

        self._A_s = self._get_A_s_fid()
        self._sigma8 = self._get_sigma8_fid()

        self._needs_rescale = None
        params, requires = {}, []
        for engine in emulator.engines.values():
            for param in engine.params:
                if param == 'z':
                    requires.append(engine)
                    continue
                if param in params: continue
                try: params[param] = self[param]
                except CosmologyError as exc:
                    if param == 'sigma8':  # A_s provided by cosmology, emulator wants sigma8
                        params[param] = self._sigma8
                        self._needs_rescale = 'A_s'
                    elif 'A_s' in find_conflicts(param, conflicts=Cosmology._conflict_parameters):  # sigma8 provided by cosmology, emulator wants A_s
                        self._params['A_s'] = self._A_s
                        params[param] = self[param]
                        del self._params['A_s']
                        self._needs_rescale = 'sigma8'
                    #else:  # maybe default values
                    #    raise exc
        if 'm_ncdm' in params:  # FIXME
            params['m_ncdm'] = self['m_ncdm_tot']

        params = emulator.defaults | params

        for operation in emulator.xoperations:
            params = operation(params)

        def predict(section):
            fixed = {name: value for name, value in emulator.fixed.items() if name.startswith(section + '.')}
            base_predict = {}
            requires_predict = []  # where this section requires extra parameters than cosmo
            # For sections that do not require extra parameters, call predict
            for name, engine in emulator.engines.items():
                if name.startswith(section + '.'):
                    if engine in requires:
                        requires_predict.append(name)
                    else:
                        base_predict[name] = engine.predict(params)

            def finalize(predict):
                # Apply postprocessing
                predict = fixed | predict
                X = dict(self._params)
                kw_yoperation = {}  #'cosmo': self}
                for operation in emulator.yoperations[::-1]:
                    try: predict = operation.inverse(predict, X=X, **kw_yoperation)
                    except KeyError: pass
                return {name[len(section) + 1:]: value for name, value in predict.items()}

            if requires_predict:

                def predict(**requires):
                    requires = params | requires
                    for name in requires_predict:
                        base_predict[name] = emulator.engines[name].predict(requires)
                    return finalize(base_predict)

                return predict

            return finalize(base_predict)

        self._predict = predict

    @classmethod
    def load(cls, filename):
        """Load class from disk."""

        class EmulatedEngine(cls):

            path = filename

        return EmulatedEngine

    def _rescale_sigma8(self):
        """Rescale perturbative quantities to match input sigma8 or A_s."""
        if getattr(self, '_rsigma8', None) is not None:
            return self._rsigma8
        self._rsigma8 = 1.
        if self._needs_rescale == 'sigma8':  # sigma8 provided by cosmology, emulator wants A_s
            self._sections.clear()  # to remove fourier with potential _rsigma8 != 1
            self._rsigma8 = self._params['sigma8'] / self.get_fourier().sigma8_m
            # As we cannot rescale sigma8 for the non-linear power spectrum
            # we recompute the power spectra
            if any('fourier.pk_non_linear' in name for name in self._emulator.engines):
                self._params['A_s'] = self._A_s * self._rsigma8**2
                self._sections.clear()
                self.__init__(**self._params, extra_params=self._extra_params)
                del self._params['A_s']
                self._rsigma8 = 1.
                self._rsigma8 = self._params['sigma8'] / self.get_fourier().sigma8_m
            self._sections.clear()  # to reinitialize fourier with correct _rsigma8
        if self._needs_rescale == 'A_s':  # A_s provided by cosmology, emulator wants sigma8
            self._sections.clear()  # to remove fourier with potential _rsigma8 != 1
            self._rsigma8 = (self._params['A_s'] / self.get_primordial().A_s)**0.5
            # As we cannot rescale sigma8 for the non-linear power spectrum
            # we recompute the power spectra
            if any('fourier.pk_non_linear' in name for name in self._emulator.engines):
                self._params['sigma8'] = self._sigma8 * self._rsigma8**2
                self._sections.clear()
                self.__init__(**self._params)
                del self._params['sigma8']
                self._rsigma8 = 1.
                self._rsigma8 = (self._params['A_s'] / self.get_primordial().A_s)**0.5
            self._sections.clear()  # to reinitialize fourier with correct _rsigma8
        return self._rsigma8


class Background(BaseBackground):

    """Background quantities."""

    def __init__(self, engine):
        super().__init__(engine)
        self.__setstate__(engine._predict(section='background'))

    @utils.flatarray()
    def rho_ncdm(self, z, species=None):
        return self._state['rho_ncdm'](z).T[species if species is not None else slice(None)]

    @utils.flatarray()
    def p_ncdm(self, z, species=None):
        return self._state['p_ncdm'](z).T[species if species is not None else slice(None)]

    @utils.flatarray()
    def rho_fld(self, z):
        r"""Comoving density of dark energy fluid :math:`\rho_{\mathrm{fld}}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self._state['rho_fld'](z)

    #@utils.flatarray()
    #def rho_tot(self, z):
    #    r"""Comoving total density :math:`\rho_{\mathrm{tot}}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
    #    return self._state['rho_tot'](z)

    @utils.flatarray()
    def time(self, z):
        r"""Proper time (age of universe), in :math:`\mathrm{Gy}`."""
        return self._state['time'](z)

    @utils.flatarray()
    def comoving_radial_distance(self, z):
        r"""
        Comoving radial distance, in :math:`mathrm{Mpc}/h`.

        See eq. 15 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_C(z)`.
        """
        return self._state['comoving_radial_distance'](z)

    def __getstate__(self):
        state = {}
        state['z'] = z = get_default_z_callable('background')
        for name in ['rho_ncdm', 'p_ncdm', 'rho_fld', 'time', 'comoving_radial_distance']:
            state[name] = getattr(self, name)(z)
        return state

    def __setstate__(self, state):
        state = dict(state)
        z = state.pop('z')
        for name, value in state.items():
            #state[name] = interpolate.interp1d(z, value, kind='cubic', bounds_error=True, assume_sorted=True, axis=-1)
            state[name] = Interpolator1D(z, value.T, k=3, interp_x='lin', interp_fun='lin', extrap=False, assume_sorted=True)
        self._state = state


@utils.addproperty('rs_drag', 'z_drag', 'rs_star', 'z_star', 'YHe')
class Thermodynamics(BaseSection):

    def __init__(self, engine):
        super().__init__(engine)
        self.__setstate__(engine._predict(section='thermodynamics'))

    #def table(self):
    #    r"""Return thermodynamics table."""
    #    return self._table['table']

    def __getstate__(self):
        state = {}
        for name in ['rs_drag', 'z_drag', 'rs_star', 'z_star', 'YHe']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, '_' + name, value)

if False:
    class Transfer(BaseSection):

        def __init__(self, engine):
            self.__setstate__(engine._predict(section='transfer'))

        def table(self):
            r"""Return source functions (in array of shape (k.size, z.size))."""
            return self._state['table']

        def __getstate__(self):
            state = {}
            for name in ['table']:
                table = getattr(self, name)()
                for key in table.dtype.names:
                    state['{}.{}'.format(name, key)] = table[key]
            return state

        def __setstate__(self, state):
            self._state = dict(state)
            use_jax = self._np is not np
            tables = {}
            for keyname, value in state.items():
                name, key = keyname.split('.')
                tables.setdefault(name, {})
                tables[name][key] = value
            for name, value in tables.items():
                names = list(value.keys())
                if use_jax:
                    table = fake_nparray({name: value[name] for name in names})
                else:
                    table = np.empty(value[names[0]].shape[0], dtype=[(name, np.float64) for name in names])
                self._state[name] = table
                for name in names: table[name] = value[name]


@utils.addproperty('k_pivot', 'n_s', 'alpha_s', 'beta_s')
class Primordial(BaseSection):

    def __init__(self, engine):
        """Initialize :class:`Primordial`."""
        super().__init__(engine)
        self.__setstate__(engine._predict(section='primordial'))
        self._h = engine['h']
        self._n_s = engine['n_s']
        self._alpha_s = engine['alpha_s']
        self._beta_s = engine['beta_s']
        self._k_pivot = engine['k_pivot'] / self._h
        self._rsigma8 = engine._rescale_sigma8()

    @property
    def A_s(self):
        r"""Scalar amplitude of the primordial power spectrum at :math:`k_\mathrm{pivot}`, unitless."""
        return self._state['A_s'] * self._rsigma8**2

    @property
    def ln_1e10_A_s(self):
        r""":math:`\ln(10^{10}A_s)`, unitless."""
        return np.log(1e10 * self.A_s)

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

    def __getstate__(self):
        """Return this class' state dictionary."""
        return {name: getattr(self, name) for name in ['A_s']}

    def __setstate__(self, state):
        """Set this class' state dictionary."""
        self._state = dict(state)


class fake_nparray(dict):

    @property
    def size(self):
        for value in self.values():
            return value.size
        return 0

    def __getitem__(self, name):
        if isinstance(name, str):
            return super().__getitem__(name)
        return self.__class__({key: self[key][name] for key in self})


class Harmonic(BaseSection):

    def __init__(self, engine):
        super().__init__(engine)
        self._rsigma8 = engine._rescale_sigma8()
        self.__setstate__(engine._predict(section='harmonic'))
        self.ellmax_cl = engine['ellmax_cl']

    def unlensed_cl(self, ellmax=-1):
        r"""Return unlensed :math:`C_{\ell}` ['tt', 'ee', 'bb', 'te'], unitless."""
        if ellmax < 0:
            ellmax = self.ellmax_cl + 1 + ellmax
        return self._state['unlensed_cl'][:ellmax + 1]

    def lens_potential_cl(self, ellmax=-1):
        r"""Return potential :math:`C_{\ell}` ['pp', 'tp', 'ep'], unitless."""
        if ellmax < 0:
            ellmax = self.ellmax_cl + 1 + ellmax
        return self._state['lens_potential_cl'][:ellmax + 1]

    def lensed_cl(self, ellmax=-1):
        r"""Return lensed :math:`C_{\ell}` ['tt', 'ee', 'bb', 'te'], unitless."""
        if ellmax < 0:
            ellmax = self.ellmax_cl + 1 + ellmax
        return self._state['lensed_cl'][:ellmax + 1]

    def __getstate__(self):
        """Return this class' state dictionary."""
        state = {}
        for name in ['unlensed_cl', 'lens_potential_cl', 'lensed_cl']:
            try: table = getattr(self, name)()
            except: continue  # e.g. lensing=False
            for key in table.dtype.names:
                if key != 'ell':
                    state['{}.{}'.format(name, key)] = table[key]
        return state

    def __setstate__(self, state):
        """Set this class' state dictionary."""
        self._state = dict(state)
        use_jax = self._np is not np
        tables = {}
        for keyname, value in state.items():
            name, key = keyname.split('.')
            tables.setdefault(name, {})
            tables[name][key] = value
        for name, value in tables.items():
            names = [key for key in value if key != 'ell']
            if use_jax:
                table = fake_nparray({name: value[name] for name in ['ell'] + names})
            else:
                table = np.empty(value[names[0]].shape[0], dtype=[('ell', np.int64)] + [(name, np.float64) for name in names])
            self._state[name] = table
            for name in names:
                table[name] = value[name] * self._rsigma8**2
            table['ell'] = np.arange(table[names[0]].shape[0])


def _make_tuple(of, size=2):
    if isinstance(of, str): of = (of,)
    of = list(of)
    of = of + [of[0]] * (size - len(of))
    return tuple(sorted(of))


@utils.addproperty('sigma8_m')
class Fourier(BaseSection):

    def __init__(self, engine):
        super().__init__(engine)
        self._callable = False
        state = engine._predict(section='fourier')
        if callable(state):
            self._callable = state
        else:
            self.__setstate__(state)
        self._h = engine['h']
        self._rsigma8 = engine._rescale_sigma8()
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
        if self._callable:
            pk_interpolator = self.pk_interpolator(non_linear=non_linear, of=of)
            return pk_interpolator.k, pk_interpolator.z, pk_interpolator.pk
        of = _make_tuple(of)
        suffix = '_non_linear' if non_linear else ''
        return self._state['k'], self._state['z' + suffix], self._state['pk' + suffix][of] * self._rsigma8**2

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
        from cosmoprimo.interpolator import _bcast_dtype
        if self._callable:

            # TODO: if this is ever to be used, vectorize over z

            of = _make_tuple(of)
            suffix = '_non_linear' if non_linear else ''

            def pk_callable(k, z, grid=True):
                if not grid:
                    raise NotImplementedError('grid must be True')
                pk = []
                dtype = _bcast_dtype(k, z)
                k = self._np.asarray(k, dtype=dtype)
                z = self._np.asarray(z, dtype=dtype)
                zflat = z.ravel()
                if not zflat.size:
                    return np.empty(shape=k.shape + z.shape, dtype=dtype)
                for zz in zflat:
                    state = self._callable(z=zz)
                    self.__setstate__(state)
                    state = self._state
                    pk.append(state['pk' + suffix][of] * self._rsigma8**2)
                del self._state
                pk = self._np.column_stack(pk)
                return Interpolator1D(state['k'], pk, interp_x='log', interp_fun='log', extrap=True)(k)

            return PowerSpectrumInterpolator2D.from_callable(k=get_default_k_callable(), z=get_default_z_callable(non_linear=non_linear), pk_callable=pk_callable)

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
        z_non_linear = get_default_z_callable(non_linear=True)
        try:
            state['pk_non_linear.delta_m.delta_m'] = self.pk_interpolator(non_linear=True, of=('delta_m', 'delta_m'))(k, z_non_linear)
            state['z_non_linear'] = z_non_linear
        except:
            pass
        list_of = []
        ofs = ['delta_cb', 'delta_m', 'theta_cb', 'theta_m', 'phi_plus_psi']
        for iof1, of1 in enumerate(ofs):
            for of2 in ofs[iof1:]:
                list_of.append(tuple(sorted((of1, of2))))
        for of in list_of:
            try: state['pk.{}.{}'.format(*of)] = self.pk_interpolator(non_linear=False, of=of)(k, z)
            except: pass
        return state

    def __setstate__(self, state):
        """Set this class' state dictionary."""
        self._state = {}
        for keyname, value in state.items():
            if keyname.startswith('pk'):
                name, *keys = keyname.split('.')
                self._state.setdefault(name, {})
                self._state[name][tuple(keys)] = value
            else: # k, z
                self._state[keyname] = value