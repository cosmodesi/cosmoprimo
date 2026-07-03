"""Emulator residual plotting (requires matplotlib).

Install optional dependency: ``pip install 'cosmoprimo[plotting]'``.
"""
try:
    from matplotlib import pyplot as plt
    from matplotlib.colors import colorConverter
except ImportError as exc:
    raise ImportError(
        "cosmoprimo emulators plotting requires matplotlib. "
        "Install with: pip install 'cosmoprimo[plotting]'"
    ) from exc

import numpy as np

from cosmoprimo import utils
from cosmoprimo.emulators import get_calculator, mask_subsample
from cosmoprimo.emulators.tools.samples import InputSampler, Samples


def pale_colors(color, nlevels, pale_factor=0.6):
    """Make color paler. Same as GetDist."""
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


def plot_residual_harmonic(ref_samples, emulated_samples, quantities=None, fsky=1., subsample=1., q=(0.68, 0.95, 0.99), color='C0', fn=None):
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
    colors = pale_colors(color, len(q))
    fig, lax = plt.subplots(len(quantities), figsize=(6, 2 * len(quantities)), sharex=True, sharey=False, squeeze=False)
    fig.subplots_adjust(hspace=0.3)
    lax = lax.ravel()
    for ax, name in zip(lax, quantities):
        of = name[-2:]
        kcl12 = (namespace + name[:-2] + of[0] * 2, namespace + name[:-2] + of[1] * 2)
        kcl12 = {'lens_potential_cl.tp': (namespace + 'unlensed_cl.tt', namespace + 'lens_potential_cl.pp'), 'lens_potential_cl.ep': (namespace + 'unlensed_cl.ee', namespace + 'lens_potential_cl.pp')}.get(name, kcl12)
        cl1, cl2 = (ref_samples[kcl] for kcl in kcl12)
        emulated = emulated_samples[namespace + name]
        ref = ref_samples[namespace + name]
        ells = np.arange(ref.shape[-1])
        if fsky is not None:
            prefac = 1. / np.sqrt(fsky * (2 * ells + 1))
            sigma = prefac * np.sqrt(emulated**2 + cl1 * cl2)
            ylabel = r'$|\mathrm{emulated} - \mathrm{ref}| / \sigma$'
        else:
            sigma = 0.5 * np.sqrt(emulated**2 + cl1 * cl2)
            ylabel = r'|emulated/ref - 1|'
        mask = ells > 1
        diff = np.abs(emulated[..., mask] - ref[..., mask]) / sigma[..., mask]
        diff = diff[np.isfinite(diff).all(axis=-1)]
        ells = np.ones(diff.shape) * ells[mask]
        lims = np.quantile(diff, [0.] + list(q) + [1.], axis=0)
        for lim, color in list(zip(zip(lims[:-1], lims[1:]), colors))[::-1]:
            mask = (diff >= lim[0]) & (diff <= lim[1])
            ax.scatter(ells[mask], diff[mask], color=color, marker='.', alpha=0.1)
        ax.set_ylabel(ylabel)
        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1.)
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

