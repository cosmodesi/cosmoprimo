from .tools import *
from . import tools, emulated
from .emulated import EmulatedEngine
from .hybrid import CAPSEEngine


def get_calculator(cosmo, xoperation=None, section=None):

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

    def calculator(**params):
        from cosmoprimo import CosmologyError
        toret = {}
        try:
            clone = cosmo.clone(**params)
            for section_name in section_names:
                section = getattr(clone, 'get_{}'.format(section_name))()
                getstate = getattr(section, '__getstate__', None)
                if getstate is not None:
                    state = getstate()
                else:  # fallback to emulated' __getstate__
                    getstate = getattr(getattr(emulated, section_name.capitalize(), None), '__getstate__', None)
                    if getstate is not None:
                        state = getstate(section)
                    else:
                        continue
                for name, value in state.items():
                    toret['{}.{}'.format(section_name, name)] = value
        except CosmologyError as exc:
            raise CalculatorComputationError from exc
        return toret

    return calculator


class Emulator(tools.Emulator):

    """Subclass :class:`tools.Emulator` to be able to provide a cosmology as calculator."""

    def set_calculator(self, calculator, params=None):
        super(Emulator, self).set_calculator(get_calculator(calculator), params=params)


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


import numpy as np
from matplotlib import pyplot as plt

from cosmoprimo import utils


def mask_subsample(size, factor=1., seed=42):
    # If factor <= 1., mask for this fraction of samples; else for factor number of samples
    rng = np.random.RandomState(seed=seed)
    mask = np.zeros(size, dtype='?')
    if factor <= 1.: factor = int(factor * size)
    mask[rng.choice(size, factor, replace=False)] = True
    return mask


def pale_colors(color, nlevels, pale_factor=0.6):
    """Make color paler. Same as GetDist."""
    from matplotlib.colors import colorConverter
    color = colorConverter.to_rgb(color)
    colors = [color]
    for _ in range(1, nlevels):
        colors.append([c * (1 - pale_factor) + pale_factor for c in colors[-1]])
    return colors


def plot_residual_background(ref_samples, emulated_samples, quantities=None, subsample=1., q=(0.68, 0.95, 0.99), color='C0', fn=None):
    """
    Plot residual of emulated background quantities against reference.

    Parameters
    ----------
    ref_samples : Samples
        Samples that contain true (reference) background quantities, e.g. 'background.comoving_radial_distance'.

    emulated_samples : Samples, cosmology
        Samples obtained with the emulated cosmology, or emulated cosmology.

    quantities : list, default=None
        Select a subset of background quantities to plot.
        Defaults to all background quantities in ``ref_samples``.

    subsample : float, int, default=1.
        Optionally, use a subset of input samples.
        If < 1., use this fraction of samples; else use ``subsampler`` number of samples.

    fn : str, Path, default=None
        If not ``None``, save figure to this location.
    """

    mask = mask_subsample(ref_samples.size, factor=subsample)
    ref_samples = ref_samples[mask]

    if isinstance(emulated_samples, Samples):
        emulated_samples = emulated_samples[mask]
    else:
        sampler = InputSampler(get_calculator(emulated_samples, section='background'), params=[name[2:] for name in ref_samples.columns('X.*')], samples=ref_samples)
        emulated_samples = sampler.run()

    namespace = 'Y.background.'

    if quantities is None:
        quantities = [name[len(namespace):] for name in ref_samples if name.startswith(namespace) and name not in [namespace + 'z']]

    fig, lax = plt.subplots(len(quantities), figsize=(6, 2 * len(quantities)), sharex=True, sharey=False, squeeze=False)
    fig.subplots_adjust(hspace=0.2)
    lax = lax.ravel()
    if namespace + 'z' in ref_samples: z = ref_samples[namespace + 'z'][0]
    else: z = ref_samples.attrs['fixed']['background.z']
    colors = pale_colors(color, len(q))
    for ax, name in zip(lax, quantities):
        mask = z > 0
        if not np.flatnonzero(emulated_samples[namespace + name]).any(): continue
        diff = np.abs(emulated_samples[namespace + name][..., mask] / ref_samples[namespace + name][..., mask] - 1.)
        diff = diff[np.isfinite(diff).all(axis=-1)]
        lims = np.quantile(diff, [0.] + list(q) + [1.], axis=0)
        for lim, color in list(zip(zip(lims[:-1], lims[1:]), colors))[::-1]:
            ax.fill_between(z[mask], lim[0], lim[1], color=color, linewidth=0.)
        ax.set_ylabel('|emulated/ref - 1|')
        #ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-6, 1e-1)
        ax.set_title(name)
        ax.grid(True)
    ax.set_xlabel('$z$')
    fig.align_ylabels()
    if fn is not None:
        utils.savefig(fn, fig=fig)
    return fig


def plot_residual_thermodynamics(ref_samples, emulated_samples, quantities=None, subsample=1., q=(0.68, 0.95, 0.99), color='C0', fn=None):
    """
    Plot residual of emulated thermodynamics quantities against reference.

    Parameters
    ----------
    ref_samples : Samples
        Samples that contain true (reference) thermodynamics quantities, e.g. 'thermodynamics.rs_drag'.

    emulated_samples : Samples, cosmology
        Samples obtained with the emulated cosmology, or emulated cosmology.

    quantities : list, default=None
        Select a subset of thermodynamics quantities to plot.
        Defaults to all thermodynamics quantities in ``ref_samples``.

    subsample : float, int, default=1.
        Optionally, use a subset of input samples.
        If < 1., use this fraction of samples; else use ``subsampler`` number of samples.

    fn : str, Path, default=None
        If not ``None``, save figure to this location.
    """
    mask = mask_subsample(ref_samples.size, factor=subsample)
    ref_samples = ref_samples[mask]

    if isinstance(emulated_samples, Samples):
        emulated_samples = emulated_samples[mask]
    else:
        sampler = InputSampler(get_calculator(emulated_samples, section='thermodynamics'), samples=ref_samples)
        emulated_samples = sampler.run()

    namespace = 'Y.thermodynamics.'

    if quantities is None:
        quantities = [name[len(namespace):] for name in ref_samples if name.startswith(namespace)]

    fig, ax = plt.subplots(sharex=True, sharey=False, squeeze=True)
    fig.subplots_adjust(hspace=0.1)
    idx = np.linspace(0., 1., len(quantities))
    colors = pale_colors(color, len(q))
    for iname, name in enumerate(quantities):
        diff = np.abs(emulated_samples[namespace + name] / ref_samples[namespace + name] - 1.)
        diff = diff[np.isfinite(diff)]
        lims = np.quantile(diff, [0.] + list(q) + [1.], axis=0)
        for lim, color in list(zip(zip(lims[:-1], lims[1:]), colors))[::-1]:
            mask = (diff >= lim[0]) & (diff <= lim[1])
            ax.plot(np.full(mask.sum(), idx[iname]), diff[mask], color=color, marker='.', alpha=0.1)
    ax.set_xticks(idx)
    ax.set_xticklabels(quantities, rotation=40, ha='right')
    ax.set_ylabel('|emulated/ref - 1|')
    ax.set_yscale('log')
    ax.set_ylim(1e-7, 1.)
    ax.grid(True)
    if fn is not None:
        utils.savefig(fn, fig=fig)
    return fig


def plot_residual_primordial(ref_samples, emulated_samples, quantities=None, subsample=1., fn=None):
    """
    Plot residual of emulated primordial quantities against reference.

    Parameters
    ----------
    ref_samples : Samples
        Samples that contain true (reference) primordial quantities, e.g. 'primordial.A_s'.

    emulated_samples : Samples, cosmology
        Samples obtained with the emulated cosmology, or emulated cosmology.

    quantities : list, default=None
        Select a subset of primordial quantities to plot.
        Defaults to all primordial quantities in ``ref_samples``.

    subsample : float, int, default=1.
        Optionally, use a subset of input samples.
        If < 1., use this fraction of samples; else use ``subsampler`` number of samples.

    fn : str, Path, default=None
        If not ``None``, save figure to this location.
    """
    mask = mask_subsample(ref_samples.size, factor=subsample)
    ref_samples = ref_samples[mask]

    if isinstance(emulated_samples, Samples):
        emulated_samples = emulated_samples[mask]
    else:
        sampler = InputSampler(get_calculator(emulated_samples, section='primordial'), samples=ref_samples)
        emulated_samples = sampler.run()

    namespace = 'Y.primordial.'

    if quantities is None:
        quantities = [name[len(namespace):] for name in ref_samples if name.startswith(namespace)]

    fig, ax = plt.subplots(sharex=True, sharey=False, squeeze=True)
    fig.subplots_adjust(hspace=0.1)
    idx = np.linspace(0., 1., len(quantities))
    for iname, name in enumerate(quantities):
        for ref, emulated in zip(ref_samples[namespace + name], emulated_samples[namespace + name]):
            ax.plot(idx[iname], np.abs(emulated / ref - 1.), color='k', marker='o')
    ax.set_xticks(idx)
    ax.set_xticklabels(quantities, rotation=40, ha='right')
    ax.set_ylabel('|emulated/ref - 1|')
    ax.set_yscale('log')
    ax.grid(True)
    if fn is not None:
        utils.savefig(fn, fig=fig)
    return fig


def plot_residual_harmonic(ref_samples, emulated_samples, quantities=None, fsky=1., subsample=1., fn=None):
    """
    Plot ratio of emulated harmonic quantities minus reference, divided by estimated error.

    Parameters
    ----------
    ref_samples : Samples
        Samples that contain true (reference) harmonic quantities, e.g. 'harmonic.cl.tt'.

    emulated_samples : Samples, cosmology
        Samples obtained with the emulated cosmology, or emulated cosmology.

    quantities : list, default=None
        Select a subset of harmonic quantities to plot.
        Defaults to all harmonic quantities in ``ref_samples``.

    fsky : fraction, default=1.
        Sky fraction to use in errors. No extra noise assumed.

    subsample : float, int, default=1.
        Optionally, use a subset of input samples.
        If < 1., use this fraction of samples; else use ``subsampler`` number of samples.

    fn : str, Path, default=None
        If not ``None``, save figure to this location.
    """
    mask = mask_subsample(ref_samples.size, factor=subsample)
    ref_samples = ref_samples[mask]

    if isinstance(emulated_samples, Samples):
        emulated_samples = emulated_samples[mask]
    else:
        sampler = InputSampler(get_calculator(emulated_samples, section='harmonic'), samples=ref_samples)
        emulated_samples = sampler.run()

    namespace = 'Y.harmonic.'

    if quantities is None:
        quantities = [name[len(namespace):] for name in ref_samples if name.startswith(namespace)]

    fig, lax = plt.subplots(len(quantities), figsize=(6, 2 * len(quantities)), sharex=True, sharey=False, squeeze=False)
    fig.subplots_adjust(hspace=0.3)
    lax = lax.ravel()
    for ax, name in zip(lax, quantities):
        for isample, (ref, emulated) in enumerate(zip(ref_samples[namespace + name], emulated_samples[namespace + name])):
            of = name[-2:]
            kcl12 = (namespace + name[:-2] + of[0] * 2, namespace + name[:-2] + of[1] * 2)
            kcl12 = {'lens_potential_cl.tp': (namespace + 'unlensed_cl.tt', namespace + 'lens_potential_cl.pp'), 'lens_potential_cl.ep': (namespace + 'unlensed_cl.ee', namespace + 'lens_potential_cl.pp')}.get(name, kcl12)
            cl1, cl2 = (ref_samples[kcl][isample] for kcl in kcl12)
            ells = np.arange(ref.size)
            prefac = 1. / np.sqrt(fsky * (2 * ells + 1))
            sigma = prefac * np.sqrt(emulated**2 + cl1 * cl2)
            mask = ells > 1
            ax.plot(ells[mask], np.abs(emulated[mask] - ref[mask]) / sigma[mask], color='k')
        ax.set_ylabel(r'$|\mathrm{emulated} - \mathrm{ref}| / \sigma$')
        #ax.set_yscale('log')
        ax.set_title(name)
        ax.grid(True)
    ax.set_xlabel(r'$\ell$')
    fig.align_ylabels()
    if fn is not None:
        utils.savefig(fn, fig=fig)
    return fig


def plot_residual_fourier(ref_samples, emulated_samples, quantities=None, iz=0, volume=1e9, kstep=5e-3, subsample=1., q=(0.68, 0.95, 0.99), color='C0', fn=None):
    """
    Plot ratio of emulated fourier quantities minus reference, divided by estimated error.

    Parameters
    ----------
    ref_samples : Samples
        Samples that contain true (reference) fourier quantities, e.g. 'fourier.pk.delta_cb.delta_cb'.

    emulated_samples : Samples, cosmology
        Samples obtained with the emulated cosmology, or emulated cosmology.

    quantities : list, default=None
        Select a subset of fourier quantities to plot.
        Defaults to all fourier quantities in ``ref_samples``.

    iz : int, default=0
        Select this redshift index.

    volume : float, default=1e9
        Volume, in power spectrum units ((Mpc/h)^3), to assume for errors.

    kstep : float, default=5e-3
        Width of k-bins, in power spectrum units (h/Mpc), to assume for errors.

    subsample : float, int, default=1.
        Optionally, use a subset of input samples.
        If < 1., use this fraction of samples; else use ``subsampler`` number of samples.

    fn : str, Path, default=None
        If not ``None``, save figure to this location.
    """
    mask = mask_subsample(ref_samples.size, factor=subsample)
    ref_samples = ref_samples[mask]

    if isinstance(emulated_samples, Samples):
        emulated_samples = emulated_samples[mask]
    else:
        sampler = InputSampler(get_calculator(emulated_samples, section='fourier'), samples=ref_samples)
        emulated_samples = sampler.run()

    namespace = 'Y.fourier.'

    if quantities is None:
        quantities = [name[len(namespace):] for name in ref_samples if name.startswith(namespace + 'pk')]

    fig, lax = plt.subplots(len(quantities), figsize=(6, 2 * len(quantities)), sharex=True, sharey=False, squeeze=False)
    fig.subplots_adjust(hspace=0.3)
    lax = lax.ravel()
    if namespace + 'k' in ref_samples:
        k = ref_samples[namespace + 'k'][0]
        if iz is not None: z = ref_samples[namespace + 'z'][0][iz]
    else:
        k = ref_samples.attrs['fixed']['fourier.k']
        if iz is not None: z = ref_samples.attrs['fixed']['fourier.z'][iz]
    colors = pale_colors(color, len(q))
    for ax, name in zip(lax, quantities):
        of = name.split('.')
        basename, of = namespace + '.'.join(of[:-2]) + '.', of[-2:]
        pk1 = ref_samples[basename + '.'.join([of[0]] * 2)][..., iz]
        pk2 = ref_samples[basename + '.'.join([of[1]] * 2)][..., iz]
        if volume is None: prefac = 1. / np.sqrt(2.)
        else: prefac = 1. / np.sqrt(k**2 * kstep * volume)
        ref = ref_samples[namespace + name][..., iz]
        emulated = emulated_samples[namespace + name][..., iz]
        sigma = prefac * np.sqrt(ref**2 + pk1 * pk2)
        diff = np.abs(emulated - ref) / sigma
        diff = diff[np.isfinite(diff).all(axis=-1)]
        lims = np.quantile(diff, [0.] + list(q) + [1.], axis=0)
        mask = k > 1e-6
        for lim, color in list(zip(zip(lims[:-1], lims[1:]), colors))[::-1]:
            ax.fill_between(k[mask], lim[0][mask], lim[1][mask], color=color, linewidth=0.)
        ax.set_ylabel(r'$|\mathrm{emulated} - \mathrm{ref}| / \sigma$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('{} @ z = {}'.format(name, z))
        ax.grid(True)
    ax.set_xlabel(r'$k$')
    fig.align_ylabels()
    if fn is not None:
        utils.savefig(fn, fig=fig)
    return fig


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