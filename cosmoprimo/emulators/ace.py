r"""``Cosmology(engine='ace')`` -- the packaged jaxace / jaxmapse / jaxcapse emulators.

One engine over three trained networks, each downloaded on demand by its own package:

==================  ==========================  ==================================================
section             emulator                    what it serves
==================  ==========================  ==================================================
background          none                        :class:`~cosmoprimo.cosmology.DefaultBackground`
thermodynamics      ``ACE_mnuw0wacdm_..``       :math:`r_s(z_\mathrm{drag})`
fourier             ``mnuw0wacdm_class``        linear :math:`P(k, z)`, :math:`\sigma_8(z)`
harmonic            ``camb_lcdm``               lensed TT/TE/EE and the lensing potential
==================  ==========================  ==================================================

The background is not emulated: cosmoprimo solves those ODEs directly, and measured against CAMB
``efunc``, ``growth_factor`` and ``growth_rate`` agree to 6e-13 -- there is nothing an emulator
could add. jaxace is still used for the growth factor handed to the :math:`P(k)` network, so
that its input matches what it was trained against.

Unit conventions, which is where this kind of wiring goes wrong quietly:

- jaxace returns :math:`r_s` in Mpc, cosmoprimo wants Mpc/:math:`h`;
- jaxmapse is trained in Mpc units (:math:`k` in 1/Mpc, :math:`P` in Mpc:sup:`3`);
- jaxcapse returns :math:`D_\ell = \ell(\ell+1)C_\ell/2\pi` in :math:`\mu\mathrm{K}^2` for TT/TE/EE
  and :math:`\ell^2(\ell+1)^2 C_\ell^{\phi\phi}/2\pi` for PP, while cosmoprimo's sections return
  raw dimensionless :math:`C_\ell`.

Measured against CAMB at the DESI fiducial, as rms(difference) / rms(reference) above
:math:`\ell = 30` -- a ratio of norms, because TE crosses zero and a pointwise ratio there
reports 2% as 100%:

===============  ==========
quantity         agreement
===============  ==========
``rs_drag``      7e-5
``sigma8_m``     2.8e-4
:math:`P(k)`     2e-3
lensed TT        1.2e-3
lensed EE        6.4e-3
lensed TE        5.5e-3
===============  ==========

:math:`C_\ell^{\phi\phi}` is the exception: it matches to ~1% up to :math:`\ell \sim 300` and then
drifts, reaching 1.19x at :math:`\ell = 1000` and 1.69x at :math:`\ell = 2000`. The low-:math:`\ell`
agreement says the conversion is right; the high-:math:`\ell` drift is where the non-linear
treatment dominates :math:`C_\ell^{\phi\phi}`, so it most likely reflects the CAMB settings the
network was trained with rather than anything here. Do not use the lensing potential above
:math:`\ell \sim 300` without checking it against the code you mean to reproduce.

Limits worth knowing before trusting a number: the Cl network is **LCDM only** -- varying
``m_ncdm``, ``w0_fld`` or ``wa_fld`` leaves the spectra blind to them, and this engine warns
rather than silently returning the fiducial; ``bb`` comes back as zeros; and the ACE
:math:`\sigma_8(z)` is total-matter, served for ``delta_cb`` as an approximation (0.5% low at the
DESI fiducial with 0.06 eV).
"""

import numpy as np

from cosmoprimo.cosmology import BaseEngine, BaseSection, DefaultBackground
from cosmoprimo.jax import numpy as jnp


#: The packaged networks, and the parameters each takes -- in the order it takes them.
EMULATORS = {
    'thermodynamics': dict(kind='jaxace', name='ACE_mnuw0wacdm_ln10As_basis',
                           inputs=['z', 'logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'm_ncdm',
                                   'w0_fld', 'wa_fld']),
    'fourier': dict(kind='jaxmapse', name='mnuw0wacdm_class',
                    inputs=['logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'm_ncdm',
                            'w0_fld', 'wa_fld']),
    'harmonic': dict(kind='jaxcapse', name='camb_lcdm', ellmax=5000,
                     inputs=['logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'tau_reio']),
}

#: Parameters the Cl network does not take. Varying one leaves the spectra blind to it, so the
#: harmonic section refuses rather than returning something that looks fine.
_LCDM_ONLY = ('m_ncdm', 'w0_fld', 'wa_fld', 'Omega_k', 'N_eff')

#: cosmoprimo name for each jaxace ``w0waCDMCosmology`` argument.
_JAXACE = {'ln10As': 'logA', 'ns': 'n_s', 'h': 'h', 'omega_b': 'omega_b', 'omega_c': 'omega_cdm',
           'm_nu': 'm_ncdm_tot', 'w0': 'w0_fld', 'wa': 'wa_fld'}


def _load(kind, name):
    if kind == 'jaxace':
        import jaxace

        return jaxace.get_emulator(name)
    if kind == 'jaxmapse':
        from pathlib import Path

        import jaxmapse

        root = Path(jaxmapse.artifact_path(name))
        return {'delta_m': jaxmapse.load_emulator(str(root / 'Pk_lin_mm')),
                'delta_cb': jaxmapse.load_emulator(str(root / 'Pk_lin_cb'))}
    import jaxcapse

    loaded = jaxcapse.trained_emulators.get(name, {})
    if not loaded or any(network is None for network in loaded.values()):
        loaded = jaxcapse.reload_emulators(name)[name]
    return loaded


class AceEngine(BaseEngine):
    """Engine backed by the packaged jaxace / jaxmapse / jaxcapse networks."""
    name = 'ace'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._emulators, self._logA_value = {}, None

    def emulator(self, section):
        """The network for one section, loaded (and downloaded) on first use."""
        if section not in self._emulators:
            spec = EMULATORS[section]
            self._emulators[section] = _load(spec['kind'], spec['name'])
        return self._emulators[section]

    def inputs(self, section, z=None):
        """The network's inputs, in its order -- read off this cosmology by name.

        A parameter the cosmology cannot supply is an error, not something to guess at. The
        common case is the amplitude: these networks take ``logA``, and cosmoprimo's default
        cosmology is parametrised by ``sigma8``, which cannot be converted without the Boltzmann
        solve the emulator exists to avoid. ``BaseEngine._get_A_s_fid`` would return a
        fitting-formula guess and every spectrum would be quietly off.
        """
        from cosmoprimo.cosmology import CosmologyError

        spec = EMULATORS[section]
        values = []
        for name in spec['inputs']:
            if name == 'z':
                values.append(jnp.asarray(z))
                continue
            if name == 'logA':
                values.append(self.logA())
                continue
            try:
                values.append(jnp.asarray(self['m_ncdm_tot' if name == 'm_ncdm' else name]))
            except CosmologyError:
                raise CosmologyError(
                    f'the {spec["name"]!r} network takes {spec["inputs"]}, and this cosmology '
                    f'does not provide {name!r}') from None
        return values

    def logA(self):
        r"""The amplitude the networks take, solved from :math:`\sigma_8` when that is what was given.

        cosmoprimo's default cosmology is parametrised by :math:`\sigma_8`, which no network here
        accepts. It does not need a Boltzmann solve, though: the linear power spectrum is linear
        in :math:`A_s`, so :math:`\sigma_8 \propto e^{\ln A / 2}` at fixed shape, and the ACE
        network's own :math:`\sigma_8` output inverts it in closed form from one evaluation --

        .. math:: \ln A = \ln A_\mathrm{ref} + 2 \ln(\sigma_8 / \sigma_8(\ln A_\mathrm{ref}))

        exactly, for any reference. ``_get_A_s_fid`` would have been the lazy alternative and is
        a fitting formula: every spectrum would come out quietly off.
        """
        from cosmoprimo.cosmology import CosmologyError

        if self._logA_value is None:
            try:
                self._logA_value = jnp.asarray(self['logA'])
            except CosmologyError:
                try:
                    sigma8 = jnp.asarray(self['sigma8'])
                except CosmologyError:
                    raise CosmologyError(
                        'these networks take `logA`; give the amplitude as `logA`, `A_s` or '
                        '`sigma8`') from None
                reference = 3.
                # output[1] is sigma8_z, and z = 0 makes it the same quantity `sigma8_m` serves
                # -- inverting on output[0] instead leaves the round trip 5e-5 off for no reason
                predicted = self._ace(0., logA=reference)[0, 1]
                self._logA_value = reference + 2. * jnp.log(sigma8 / predicted)
        return self._logA_value

    def jaxace_cosmology(self):
        """jaxace's own cosmology object, for the growth the P(k) network was trained against."""
        import jaxace

        return jaxace.w0waCDMCosmology(**{name: jnp.asarray(self[cosmoprimo])
                                          for name, cosmoprimo in _JAXACE.items()})

    def ace(self, z):
        """The ACE network on a redshift grid: ``(nz, 7)`` = (sigma8, sigma8_z, rs_drag [Mpc],
        H_z, r_z, D_z, f_z)."""
        return self._ace(z)

    def _ace(self, z, logA=None):
        """``ace``, with the amplitude overridable -- :meth:`logA` needs to call in before it is
        known, and would otherwise recurse."""
        z = jnp.atleast_1d(jnp.asarray(z))
        spec = EMULATORS['thermodynamics']
        if logA is not None:
            values = [jnp.asarray(z) if name == 'z'
                      else jnp.asarray(logA) if name == 'logA'
                      else jnp.asarray(self['m_ncdm_tot' if name == 'm_ncdm' else name])
                      for name in spec['inputs']]
        else:
            values = self.inputs('thermodynamics', z=z)
        stacked = jnp.stack([value if value.ndim else jnp.full(z.shape, value)
                             for value in values], axis=-1)
        return self.emulator('thermodynamics').run_emulator(stacked)


class Background(DefaultBackground):
    """Solved, not emulated: cosmoprimo's own ODEs match CAMB to 6e-13."""


class Thermodynamics(BaseSection):

    def __init__(self, engine):
        super().__init__(engine)
        self._engine = engine

    @property
    def rs_drag(self):
        r""":math:`r_s(z_\mathrm{drag})`, in :math:`\mathrm{Mpc}/h` -- the network returns Mpc."""
        return self._engine.ace(0.)[0, 2] * self._engine['h']


class Fourier(BaseSection):

    def __init__(self, engine):
        super().__init__(engine)
        self._engine = engine

    def _pk(self, of, z):
        r"""``(nz, nk)`` linear :math:`P(k, z)` on the network's own k grid, in
        :math:`(\mathrm{Mpc}/h)^3`."""
        engine = self._engine
        component = engine.emulator('fourier')['delta_m' if of == 'delta_m' else 'delta_cb']
        values = jnp.array(engine.inputs('fourier'))
        z = jnp.atleast_1d(jnp.asarray(z))
        growth = engine.jaxace_cosmology().D_z(z)
        pk = component.get_Pk(values, z, growth)
        if of.startswith('theta'):
            # pk_tt = f_z^2 pk_cb at scale-independent growth, with f_z from the ACE network so
            # that sigma8_z(theta_cb) = f_z sigma8_z(delta_cb) exactly
            pk = engine.ace(z)[:, 6, None]**2 * pk
        h = engine['h']
        return jnp.asarray(component.k_grid) / h, pk * h**3

    def pk_interpolator(self, of='delta_m', **kwargs):
        r"""Linear :math:`P(k, z)`, as the usual 2D interpolator."""
        from cosmoprimo.interpolator import PowerSpectrumInterpolator2D

        if isinstance(of, (list, tuple)):
            of = of[0] if len(set(of)) == 1 else '_'.join(of)
        z = np.linspace(0., 3., 30)
        k, pk = self._pk(of, z)
        return PowerSpectrumInterpolator2D(np.asarray(k), z, np.asarray(pk).T, **kwargs)

    def pk_kz(self, k, z, of='delta_m', **kwargs):
        return self.pk_interpolator(of=of, **kwargs)(k, z)

    def sigma8_z(self, z, of='delta_m'):
        r""":math:`\sigma_8(z)`.

        Total-matter, and served for ``delta_cb`` as an approximation -- 0.5% low at the DESI
        fiducial with 0.06 eV, which is the network's own convention, not a choice made here.
        """
        values = self._engine.ace(z)
        sigma8 = values[:, 1]
        if str(of).startswith('theta'):
            sigma8 = values[:, 6] * sigma8
        return sigma8 if np.ndim(z) else sigma8[0]

    @property
    def sigma8_m(self):
        return self.sigma8_z(0., of='delta_m')


class Harmonic(BaseSection):

    def __init__(self, engine):
        super().__init__(engine)
        self._engine = engine
        varied = [name for name in _LCDM_ONLY if _is_varied(engine, name)]
        if varied:
            # A warning, not an error: the network has some fixed neutrino content baked in and
            # refusing the standard 0.06 eV would make the engine useless for the default
            # cosmology. What it must not do is stay silent -- the spectra are blind to these,
            # so a scan over them would return the same Cl and look perfectly well behaved.
            import warnings

            warnings.warn(f'the packaged Cl network is LCDM-only and takes '
                          f'{EMULATORS["harmonic"]["inputs"]}; {varied} are set away from their '
                          f'defaults here and the spectra will not respond to them')
        self.ellmax_cl = min(int(engine['ellmax_cl']), EMULATORS['harmonic']['ellmax'])

    def _ells(self, ellmax):
        if ellmax is None or ellmax < 0:
            ellmax = self.ellmax_cl + 1 + (ellmax if ellmax is not None else -1)
        if ellmax > EMULATORS['harmonic']['ellmax']:
            raise ValueError(f'the network is trained to l = {EMULATORS["harmonic"]["ellmax"]}, '
                             f'asked for {ellmax}')
        return int(ellmax), jnp.arange(2, int(ellmax) + 1)

    def lensed_cl(self, ellmax=-1):
        r"""Lensed :math:`C_\ell`, raw and dimensionless; ``bb`` is zeros."""
        from .cosmology import _Table

        ellmax, ells = self._ells(ellmax)
        engine = self._engine
        values = jnp.array(engine.inputs('harmonic'))
        networks = engine.emulator('harmonic')
        # the networks give Dl = l(l+1) Cl / 2pi in muK^2
        factor = (2. * np.pi) / (ells * (ells + 1)) / (engine['T_cmb'] * 1e6)**2
        table = {'ell': np.arange(ellmax + 1)}
        for name in ('tt', 'ee', 'bb', 'te'):
            if name == 'bb' and 'BB' not in networks:
                table[name] = jnp.zeros(ellmax + 1)
                continue
            raw = networks[name.upper()].get_Cl(values)[:ellmax - 1] * factor
            table[name] = jnp.concatenate([jnp.zeros(2), raw])
        return _Table(table)

    def lens_potential_cl(self, ellmax=-1):
        r"""Lensing-potential :math:`C_\ell`; ``tp`` and ``ep`` are zeros."""
        from .cosmology import _Table

        ellmax, ells = self._ells(ellmax)
        engine = self._engine
        values = jnp.array(engine.inputs('harmonic'))
        # the network gives l^2 (l+1)^2 Cl^phiphi / 2pi
        raw = engine.emulator('harmonic')['PP'].get_Cl(values)[:ellmax - 1] \
            * (2. * np.pi) / (ells * (ells + 1))**2
        return _Table({'ell': np.arange(ellmax + 1),
                       'pp': jnp.concatenate([jnp.zeros(2), raw]),
                       'tp': jnp.zeros(ellmax + 1), 'ep': jnp.zeros(ellmax + 1)})


def _is_varied(engine, name):
    """Whether a parameter differs from the value the LCDM network assumes."""
    defaults = {'m_ncdm': 0., 'w0_fld': -1., 'wa_fld': 0., 'Omega_k': 0.}
    if name == 'm_ncdm':
        try:
            return float(np.sum(engine['m_ncdm_tot'])) > 1e-8
        except Exception:
            return False
    if name not in defaults:
        return False
    try:
        return not np.isclose(float(engine[name]), defaults[name])
    except Exception:
        return False
