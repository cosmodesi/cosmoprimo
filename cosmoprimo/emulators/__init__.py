import sys

import numpy as np

from .tools import *
from . import tools
from .emulated import EmulatedEngine
from .hybrid import CAPSEEngine


def get_calculator(cosmo, section=None, emulated_engine=None):

    """
    Turn input cosmology into calculator:

    .. code-block:: python

        cosmo = Cosmology()
        calculator = get_calculator(cosmo)
        calculator(Omega_m=0.2)  # dictionary of arrays {'background.comoving_radial_distance': ..., 'fourier.pk.delta_cb.delta_cb': ...}
    """

    from cosmoprimo import Cosmology

    if not isinstance(cosmo, Cosmology): return cosmo

    section_names = tools.base.make_list(section if section is not None else list(cosmo.engine._Sections))
    sorted_section_names = ['background', 'thermodynamics', 'primordial', 'perturbations', 'transfer', 'fourier', 'harmonic'][::-1]
    for section_name in section_names:
        if section_name not in sorted_section_names:
            sorted_section_names.append(section_name)
    section_names = [section for section in sorted_section_names if section in section_names]
    if emulated_engine is None:
        emulated_engine = EmulatedEngine
    emulated_module = sys.modules[emulated_engine.__module__]

    def calculator(**params):
        from cosmoprimo import CosmologyError
        toret = {}
        try:
            clone = cosmo.clone(**params)
            for section_name in section_names:
                section = getattr(clone, 'get_{}'.format(section_name))()
                state = {}
                #getstate = getattr(section, '__getstate__', None)
                if False: #getstate is not None:  Python3.12 defines __getstate__()...
                    state = getstate()
                else:  # fallback to emulated' __getstate__
                    Section = getattr(emulated_module, section_name.capitalize(), None)
                    if Section is not None:
                        getstate = getattr(Section, '__getstate__', None)
                        if getstate is not None:
                            state = getstate(section)
                for name, value in state.items():
                    toret['{}.{}'.format(section_name, name)] = value
        except CosmologyError as exc:
            raise CalculatorComputationError from exc
        return toret

    return calculator


class Emulator(tools.Emulator):

    """Subclass :class:`tools.Emulator` to be able to provide a cosmology as calculator."""

    def _get_calculator(self, calculator, params=None):
        return super(Emulator, self)._get_calculator(get_calculator(calculator), params=params)


class BaseSampler(tools.samples.BaseSampler):

    """Subclass :class:`tools.samples.BaseSampler` to be able to provide a cosmology as calculator."""

    def set_calculator(self, calculator, params=None):
        super(BaseSampler, self).set_calculator(get_calculator(calculator), params=params)


class InputSampler(BaseSampler, tools.samples.InputSampler):

    """Subclass :class:`tools.samples.InputSampler` to be able to provide a cosmology as calculator."""


class GridSampler(BaseSampler, tools.samples.GridSampler):

    """Subclass :class:`tools.samples.GridSampler` to be able to provide a cosmology as calculator."""


class DiffSampler(BaseSampler, tools.samples.DiffSampler):

    """Subclass :class:`tools.samples.DiffSampler` to be able to provide a cosmology as calculator."""


class QMCSampler(BaseSampler, tools.samples.QMCSampler):

    """Subclass :class:`tools.samples.QMCSampler` to be able to provide a cosmology as calculator."""


def mask_subsample(size, factor=1., seed=42):
    # If factor <= 1., mask for this fraction of samples; else for factor number of samples
    rng = np.random.RandomState(seed=seed)
    mask = np.zeros(size, dtype='?')
    if factor <= 1.: factor = int(factor * size)
    mask[rng.choice(size, factor, replace=False)] = True
    return mask


from cosmoprimo.cosmology import Cosmology
from cosmoprimo.jax import Interpolator1D
from cosmoprimo.interpolator import PowerSpectrumInterpolator1D

from scipy.special import comb


def smoothstep(x, xmin=0, xmax=1, order=1):
    x = np.clip((x - xmin) / (xmax - xmin), 0, 1)
    result = 0
    for n in range(0, order + 1):
        result += comb(order + n, n) * comb(2 * order + 1, order - n) * (-x)**n
    result *= x**(order + 1)
    return result


class HarmonicNormOperation(Operation):

    name = 'harmonic_norm'

    def __init__(self, ref_theta_cosmomc=0.010409108133982346):  # DESI fiducial cosmology
        self.ref_theta_cosmomc = ref_theta_cosmomc

    def initialize(self, v, **kwargs):
        names = list(v.keys())
        cl_names = tools.utils.find_names(names, ['harmonic.*_cl.*'])
        self.ells, self.wells, self.windows, self.norm_cl_names = {}, {}, {}, {}
        wsize = 60
        oversampling = 1
        for keyname in cl_names:
            namespace, name, key = keyname.split('.')
            self.norm_cl_names.setdefault(name, [])
            self.norm_cl_names[name].append(keyname)
            size = v[keyname].shape[-1]
            self.ells[name] = ells = np.arange(size)
            #self.windows[name] = np.concatenate([np.logspace(-10., 0., wsize), np.ones(size - 2 * wsize, dtype='f8'), np.logspace(0., -10., wsize)], axis=0)
            smooth = smoothstep(np.linspace(0., 1., wsize * oversampling), xmin=0.2, xmax=0.8, order=3)
            self.windows[name] = np.concatenate([smooth, np.ones(size * oversampling - 3 * wsize * oversampling, dtype='f8'), smooth[::-1], np.zeros(wsize, dtype='f8')], axis=0)
            self.wells[name] = np.linspace(0., size, size * oversampling)

    def __call__(self, v, X=None, cosmo=None):
        if cosmo is None: cosmo = Cosmology(**X, engine='bbks')
        s = cosmo['theta_cosmomc'] / self.ref_theta_cosmomc
        A_s = 10**9 * cosmo['A_s']
        for namespace, cl_names in self.norm_cl_names.items():
            ell = self.ells[namespace]
            elli = self.wells[namespace] / (1. + self.windows[namespace] * s)
            for cl_name in cl_names:
                v[cl_name] = Interpolator1D(ell, v[cl_name] / A_s, extrap=True)(elli)
        return v

    def inverse(self, v, X=None, cosmo=None):
        if cosmo is None: cosmo = Cosmology(**X, engine='bbks')
        s = cosmo['theta_cosmomc'] / self.ref_theta_cosmomc
        A_s = 10**9 * cosmo['A_s']
        for namespace, cl_names in self.norm_cl_names.items():
            ell = self.wells[namespace] / (1. + self.windows[namespace] * s)
            elli = self.ells[namespace]
            for cl_name in cl_names:
                v[cl_name] = Interpolator1D(ell, v[cl_name] * A_s, extrap=True)(elli)
        return v

    def __getstate__(self):
        return {name: getattr(self, name) for name in ['name', 'ells', 'windows', 'norm_cl_names', 'ref_theta_cosmomc']}


class FourierNormOperation(Operation):
    """
    Operation to apply to dictionary of output fourier quantities:
    Divide all power spectra by ``ref_pk_name``, and seperate pure k-dependence at z = 0 for ``ref_pk_name``.
    """
    name = 'fourier_norm'

    def __init__(self, ref_pk_name='fourier.pk.delta_cb.delta_cb'):
        self.ref_pk_name = ref_pk_name

    def initialize(self, v, **kwargs):
        self.norm_pk_names = tools.utils.find_names(list(v.keys()), ['fourier.pk.*.*', 'fourier.pk_non_linear.*.*'])
        self.norm_pk_names = [pk_name for pk_name in self.norm_pk_names if pk_name != self.ref_pk_name]

    def __call__(self, v, X=None, cosmo=None):
        #if self.ref_pk_name not in v: return v
        k = v['fourier.k']
        z = v['fourier.z']
        if cosmo is None: cosmo = Cosmology(**X)
        h = cosmo['h']
        #prim = cosmo.get_primordial(engine='bbks').pk_k(k=k / h) / h**3
        prim = cosmo.get_fourier(engine='bbks').pk_interpolator(extrap_kmin=k[0] / 10., extrap_kmax=k[-1] * 10.)(k=k / h, z=z[0]) / h**3
        #h = prim = 1.
        for pk_name in [self.ref_pk_name] + self.norm_pk_names:
            v[pk_name] = PowerSpectrumInterpolator1D(k, v[pk_name], extrap_kmin=k[0] / 10., extrap_kmax=k[-1] * 10.)(k / h) / h**3
        pk_dd = v[self.ref_pk_name]
        for pk_name in self.norm_pk_names: v[pk_name] = v[pk_name] / pk_dd[..., :v[pk_name].shape[-1]]  # for pk_non_linear, stop before zmax
        v['fourier.pkz'] = v[self.ref_pk_name] / v[self.ref_pk_name][..., [0]]  # normalize at z = 0
        v[self.ref_pk_name] = v[self.ref_pk_name][..., 0] / prim
        return v

    def inverse(self, v, X=None, cosmo=None):
        #if self.ref_pk_name not in v: return v
        k = v['fourier.k']
        z = v['fourier.z']
        if cosmo is None: cosmo = Cosmology(**X)
        h = cosmo['h']
        #prim = cosmo.get_primordial(engine='bbks').pk_k(k=k / h) / h**3
        prim = cosmo.get_fourier(engine='bbks').pk_interpolator(extrap_kmin=k[0] / 10., extrap_kmax=k[-1] * 10.)(k=k / h, z=z[0]) / h**3
        #h = prim = 1.
        ref = v[self.ref_pk_name] * prim
        pk_dd = v[self.ref_pk_name] = ref[..., None] * v['fourier.pkz']
        for pk_name in self.norm_pk_names:
            v[pk_name] = v[pk_name] * pk_dd[..., :v[pk_name].shape[-1]]  # for pk_non_linear, stop before zmax
        for pk_name in [self.ref_pk_name] + self.norm_pk_names:
            v[pk_name] = PowerSpectrumInterpolator1D(k / h, v[pk_name] * h**3, extrap_kmin=k[0] / 10., extrap_kmax=k[-1] * 10.)(k)
        return v

    def __getstate__(self):
        return {name: getattr(self, name) for name in ['name', 'ref_pk_name', 'norm_pk_names']}
