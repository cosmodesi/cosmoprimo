"""Cosmological calculation with the Boltzmann code CAMB."""

import warnings

import numpy as np
import camb
from camb import CAMBdata, model, CAMBError

from .cosmology import BaseEngine, BaseSection, BaseBackground, CosmologyError
from .interpolator import PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D
from . import utils, constants


def enum(*sequential, **named):
    """Enumeration values to serve as ready flags."""
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


def _build_task_dependency(tasks):
    """
    Fill the task list with all the needed modules.

    Parameters
    ----------
    tasks : list
        list of strings, containing initially only the last module required.
        For instance, to recover all the modules, the input should be ``['fourier']``.

    Returns
    -------
    tasks : list
        Complete task list.
    """
    if not isinstance(tasks, (tuple, list)):
        tasks = [tasks]
    tasks = set(tasks)
    if 'thermodynamics' in tasks:
        tasks.discard('background')
    # if 'lensing' in tasks:
    #    tasks.add('harmonic')
    if 'harmonic' in tasks:
        tasks.add('fourier')
    if 'fourier' in tasks:
        tasks.add('transfer')
    return list(tasks)


class CambEngine(BaseEngine):

    """Engine for the Boltzmann code CAMB."""
    name = 'camb'

    def __init__(self, *args, **kwargs):
        # Big thanks to https://github.com/LSSTDESC/CCL/blob/master/pyccl/boltzmann.py!
        # Quantities in the synchronous gauge
        super(CambEngine, self).__init__(*args, **kwargs)
        if self._params.get('Omega_Lambda', None) is not None:
            warnings.warn('{} cannot cope with dynamic dark energy + cosmological constant'.format(self.__class__.__name__))
        self._camb_params = camb.CAMBparams()
        self._camb_params.set_cosmology(H0=self['H0'], ombh2=self['omega_b'], omch2=self['omega_cdm'], omk=self['Omega_k'],
                                        TCMB=self['T_cmb'], tau=self.get('tau_reio', None), zrei=self.get('z_reio', None), deltazrei=self['reionization_width'],
                                        Alens=self['A_L'])  # + neutrinos
        self._camb_params.InitPower.set_params(As=self._get_A_s_fid(), ns=self['n_s'], nrun=self['alpha_s'], pivot_scalar=self['k_pivot'],
                                               pivot_tensor=self['k_pivot'], parameterization='tensor_param_rpivot', r=self['r'])

        self._camb_params.share_delta_neff = False
        self._camb_params.omnuh2 = self['omega_ncdm'].sum()
        self._camb_params.num_nu_massless = self['N_ur']
        self._camb_params.num_nu_massive = self['N_ncdm']
        self._camb_params.nu_mass_eigenstates = self['N_ncdm']
        delta_neff = self['N_eff'] - constants.NEFF  # used for BBN YHe comps

        # CAMB defines a neutrino degeneracy factor as T_i = g^(1/4)*T_nu
        # where T_nu is the standard neutrino temperature from first order
        # computations
        # CLASS defines the temperature of each neutrino species to be
        # T_i_eff = TNCDM * T_cmb where TNCDM is a fudge factor to get the
        # total mass in terms of eV to match second-order computations of the
        # relationship between m_nu and Omega_nu.
        # We are trying to get both codes to use the same neutrino temperature.
        # thus we set T_i_eff = T_i = g^(1/4) * T_nu and solve for the right
        # value of g for CAMB. We get g = (TNCDM / (11/4)^(-1/3))^4
        g = np.array(self['T_ncdm_over_cmb'], dtype=np.float64)**4 * (4. / 11.)**(-4. / 3.)
        m_ncdm = np.array(self['m_ncdm'])
        self._camb_params.nu_mass_numbers = np.ones(self['N_ncdm'], dtype=np.int32)
        self._camb_params.nu_mass_fractions = m_ncdm / m_ncdm.sum()
        self._camb_params.nu_mass_degeneracies = g

        # get YHe from BBN
        self._camb_params.bbn_predictor = camb.bbn.get_predictor()
        self._camb_params.YHe = self._camb_params.bbn_predictor.Y_He(self._camb_params.ombh2 * (camb.constants.COBE_CMBTemp / self._camb_params.TCMB)**3, delta_neff)

        self._camb_params.set_classes(dark_energy_model=camb.dark_energy.DarkEnergyFluid)
        self._camb_params.DarkEnergy.set_params(w=self['w0_fld'], wa=self['wa_fld'])

        if self['non_linear']:
            self._camb_params.NonLinearModel = camb.nonlinear.Halofit()
            halofit_version = self.get('halofit_version', 'mead')
            options = {}
            for name in ['HMCode_A_baryon', 'HMCode_eta_baryon', 'HMCode_logT_AGN']:
                tmp = self.get(name, None)
                if tmp is not None: options[name] = tmp
            self._camb_params.NonLinearModel.set_params(halofit_version=halofit_version, **options)

        self._camb_params.DoLensing = self['lensing']
        self._camb_params.Want_CMB_lensing = self['lensing']
        self._camb_params.set_for_lmax(lmax=self['ellmax_cl'])
        self._camb_params.set_matter_power(redshifts=self['z_pk'], kmax=self['kmax_pk'] * self['h'], nonlinear=self['non_linear'], silent=True)

        if not self['non_linear']:
            assert self._camb_params.NonLinear == camb.model.NonLinear_none

        self._camb_params.WantScalars = 's' in self['modes']
        self._camb_params.WantVectors = 'v' in self['modes']
        self._camb_params.WantTensors = 't' in self['modes']
        for key, value in self.extra_params.items():
            if key == 'accuracy':
                self._camb_params.set_accuracy(self.extra_params['accuracy'])
            else:
                setattr(self._camb_params, key, value)

        self.ready = enum(ba=False, th=False, tr=False, le=False, hr=False, fo=False)

    def compute(self, tasks):
        """
        The main function, which executes the desired modules.

        Parameters
        ----------
        tasks : list, string
            Calculation to perform, in the following list:
            ['background', 'thermodynamics', 'transfer', 'harmonic', 'lensing', 'fourier']
        """

        tasks = _build_task_dependency(tasks)

        if 'background' in tasks and not self.ready.ba:
            self.ba = camb.get_background(self._camb_params, no_thermo=True)
            self.ready.ba = True

        if 'thermodynamics' in tasks and not self.ready.th:
            self.ba = self.th = camb.get_background(self._camb_params, no_thermo=False)
            self.ready.ba = self.ready.th = True

        if 'transfer' in tasks and not self.ready.tr:
            self.tr = camb.get_transfer_functions(self._camb_params)
            self.ready.tr = True

        if 'harmonic' in tasks and not self.ready.hr:
            # self._camb_params.Want_CMB = True
            # self._camb_params.DoLensing = self['lensing']
            # self._camb_params.Want_CMB_lensing = self['lensing']
            self.ready.hr = True
            self.ready.fo = False

        if 'lensing' in tasks and not self.ready.le:
            self._camb_params.DoLensing = True
            self._camb_params.Want_CMB_lensing = True
            self.ready.le = True
            self.tr = CAMBdata()
            self.tr.calc_power_spectra(self._camb_params)
            self.le = self.hr = self.fo = self.tr
            self.ready.fo = True

        if 'fourier' in tasks and not self.ready.fo:
            self.tr.calc_power_spectra()
            self.fo = self.hr = self.le = self.tr
            self.ready.fo = True


class Background(BaseBackground):

    def __init__(self, engine):
        super(Background, self).__init__(engine=engine)
        self._engine.compute('background')
        self.ba = self._engine.ba
        # convert RHO to 1e10 Msun/h
        # self._H0 = self.ba.Params.H0
        #self._h = self.H0 / 100
        # camb densities are 8 pi G a^4 rho in Mpc unit
        self._RH0_ = constants.rho_crit_Msunph_per_Mpcph3 * constants.c**2 / (self.H0 * 1e3)**2 / 3.
        # for name in ['m', 'ncdm_tot']:
        #     setattr(self,'_Omega0_{}'.format(name),getattr(self,'Omega_{}'.format(name))(0.))

    @utils.flatarray(dtype=np.float64)
    def Omega_k(self, z):
        r"""Density parameter of curvature, unitless."""
        return self.ba.get_Omega('K', z=z)

    @utils.flatarray(dtype=np.float64)
    def Omega_cdm(self, z):
        r"""Density parameter of cold dark matter, unitless."""
        return self.ba.get_Omega('cdm', z=z)

    @utils.flatarray(dtype=np.float64)
    def Omega_b(self, z):
        r"""Density parameter of baryons, unitless."""
        return self.ba.get_Omega('baryon', z=z)

    @utils.flatarray(dtype=np.float64)
    def Omega_g(self, z):
        r"""Density parameter of photons, unitless."""
        return self.ba.get_Omega('photon', z=z)

    @utils.flatarray(dtype=np.float64)
    def Omega_ur(self, z):
        r"""Density parameter of ultra relativistic neutrinos, unitless."""
        return self.ba.get_Omega('neutrino', z=z)

    @utils.flatarray(dtype=np.float64)
    def Omega_ncdm_tot(self, z):
        r"""Total density parameter of massive neutrinos, unitless."""
        return self.ba.get_Omega('nu', z=z)

    @utils.flatarray(dtype=np.float64)
    def Omega_de(self, z):
        r"""Total density of dark energy (fluid + cosmological constant), unitless."""
        return self.ba.get_Omega('de', z=z)

    @utils.flatarray(dtype=np.float64)
    def rho_k(self, z):
        r"""Comoving density of curvature :math:`\rho_{k}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.ba.get_background_densities(1. / (1 + z), vars=['K'])['K'] * self._RH0_ * (1 + z)

    @utils.flatarray(dtype=np.float64)
    def rho_cdm(self, z):
        r"""Comoving density of cold dark matter :math:`\rho_{cdm}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.ba.get_background_densities(1. / (1 + z), vars=['cdm'])['cdm'] * self._RH0_ * (1 + z)

    @utils.flatarray(dtype=np.float64)
    def rho_b(self, z):
        r"""Density parameter of baryons, unitless."""
        return self.ba.get_background_densities(1. / (1 + z), vars=['baryon'])['baryon'] * self._RH0_ * (1 + z)

    @utils.flatarray(dtype=np.float64)
    def rho_g(self, z):
        r"""Comoving density of photons :math:`\rho_{g}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.ba.get_background_densities(1. / (1 + z), vars=['photon'])['photon'] * self._RH0_ * (1 + z)

    @utils.flatarray(dtype=np.float64)
    def rho_ur(self, z):
        r"""Comoving density of ultra-relativistic radiation (massless neutrinos) :math:`\rho_{ur}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.ba.get_background_densities(1. / (1 + z), vars=['neutrino'])['neutrino'] * self._RH0_ * (1 + z)

    @utils.flatarray(dtype=np.float64)
    def rho_ncdm_tot(self, z):
        r"""Total comoving density of non-relativistic part of massive neutrinos :math:`\rho_{ncdm}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.ba.get_background_densities(1. / (1 + z), vars=['nu'])['nu'] * self._RH0_ * (1 + z)

    @utils.flatarray(dtype=np.float64)
    def rho_de(self, z):
        r"""Comoving total density of dark energy :math:`\rho_{\mathrm{de}}` (fluid + cosmological constant), in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.ba.get_background_densities(1. / (1 + z), vars=['de'])['de'] * self._RH0_ * (1 + z)

    @utils.flatarray(dtype=np.float64)
    def efunc(self, z):
        r"""Function giving :math:`E(z)`, where the Hubble parameter is defined as :math:`H(z) = H_{0} E(z)`, unitless."""
        return self.hubble_function(z) / (100. * self._h)

    @utils.flatarray(dtype=np.float64)
    def hubble_function(self, z):
        r"""Hubble function, in :math:`\mathrm{km}/\mathrm{s}/\mathrm{Mpc}`."""
        return self.ba.hubble_parameter(z)

    @utils.flatarray(dtype=np.float64)
    def time(self, z):
        r"""Proper time (age of universe), in :math:`\mathrm{Gy}`."""
        if z.size:
            return np.vectorize(self.ba.physical_time)(z)
        return np.zeros_like(z)

    @utils.flatarray(dtype=np.float64)
    def comoving_radial_distance(self, z):
        r"""
        Comoving radial distance, in :math:`mathrm{Mpc}/h`.

        See eq. 15 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_C(z)`.
        """
        return self.ba.comoving_radial_distance(z) * self._h

    @utils.flatarray(dtype=np.float64)
    def luminosity_distance(self, z):
        r"""
        Luminosity distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 21 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{L}(z)`.
        """
        return self.ba.luminosity_distance(z) * self._h

    @utils.flatarray(dtype=np.float64)
    def angular_diameter_distance(self, z):
        r"""
        Proper angular diameter distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 18 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{A}(z)`.
        """
        return self.ba.angular_diameter_distance(z) * self._h

    @utils.flatarray(dtype=np.float64)
    def comoving_angular_distance(self, z):
        r"""
        Comoving angular distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 16 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{M}(z)`.
        """
        return self.angular_diameter_distance(z) * (1. + z)


@utils.addproperty('rs_drag', 'z_drag', 'rs_star', 'z_star')
class Thermodynamics(BaseSection):

    def __init__(self, engine):
        self._engine = engine
        self._engine.compute('thermodynamics')
        self.th = self._engine.th
        # convert RHO to 1e10 Msun/h
        self._h = self.th.Params.H0 / 100

        derived = self.th.get_derived_params()
        self._rs_drag = derived['rdrag'] * self._h
        self._z_drag = derived['zdrag']
        self._rs_star = derived['rstar'] * self._h
        self._z_star = derived['zstar']

    @utils.flatarray(dtype=np.float64)
    def rs_z(self, z):
        """Comoving sound horizon."""
        return self.th.sound_horizon(z) * self._h


class Transfer(BaseSection):

    def __init__(self, engine):
        self._engine = engine
        self._engine.compute('transfer')
        self.tr = self._engine.tr

    def table(self):
        r"""
        Return source functions.

        Returns
        -------
        tk : dict
            Dictionary of perturbed quantities (in array of shape (k size, z size)).
        """
        data = self.tr.get_matter_transfer_data()
        dtype = [(name, np.float64) for name in model.transfer_names]
        # shape (k, z)
        toret = np.empty(data.transfer.shape[1:], dtype=dtype)
        for name in model.transfer_names:
            toret[name] = data.transfer_data[model.transfer_names.index(name)]
        return toret


class Primordial(BaseSection):

    def __init__(self, engine):
        self._engine = engine
        self.pm = self._engine._camb_params.InitPower
        self._h = self._engine._camb_params.h
        self._rsigma8 = self._engine._rescale_sigma8()

    @property
    def A_s(self):
        r"""Scalar amplitude of the primordial power spectrum at :math:`k_\mathrm{pivot}`, unitless."""
        return self.pm.As * self._rsigma8**2

    @property
    def ln_1e10_A_s(self):
        r""":math:`\ln(10^{10}A_s)`, unitless."""
        return np.log(1e10 * self.A_s)

    @property
    def n_s(self):
        r"""Power-law index i.e. tilt of the primordial power spectrum, unitless."""
        return self.pm.ns

    @property
    def alpha_s(self):
        r"""Running of the spectral index at :math:`k_\mathrm{pivot}`, unitless."""
        return self.pm.nrun

    @property
    def k_pivot(self):
        r"""Primordial power spectrum pivot scale, where the primordial power is equal to :math:`A_{s}`, in :math:`h/\mathrm{Mpc}`."""
        return self.pm.pivot_scalar / self._h

    def pk_k(self, k, mode='scalar'):
        r"""
        The primordial spectrum of curvature perturbations at ``k``, generated by inflation, in :math:`(\mathrm{Mpc}/h)^{3}`.
        For scalar perturbations this is e.g. defined as:

        .. math::

            \mathcal{P_R}(k) = A_s \left (\frac{k}{k_0} \right )^{n_s - 1 + 1/2 \ln(k/k_0) (dn_s / d\ln k)}

        See also: eq. 2 of `this reference <https://arxiv.org/abs/1303.5076>`_.

        Parameters
        ----------
        k : array_like
            Wavenumbers, in :math:`h/\mathrm{Mpc}`.

        mode : string, default='scalar'
            'scalar', 'vector' or 'tensor' mode.

        Returns
        -------
        pk : array
            The primordial power spectrum.
        """
        index = ['scalar', 'vector', 'tensor'].index(mode)
        return self._h**3 * self._engine._camb_params.primordial_power(k * self._h, index) * self._rsigma8**2

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
        self._engine = engine
        self._engine.compute('harmonic')
        self.hr = self._engine.hr
        self._rsigma8 = self._engine._rescale_sigma8()
        self.ellmax_cl = self._engine['ellmax_cl']

    def unlensed_cl(self, ellmax=-1):
        r"""Return unlensed :math:`C_{\ell}` ['tt','ee','bb','te'], unitless."""
        if ellmax < 0:
            ellmax = self.ellmax_cl + 1 + ellmax
        table = self.hr.get_unlensed_total_cls(lmax=ellmax, CMB_unit=None, raw_cl=True)
        names = ['tt', 'ee', 'bb', 'te']
        toret = np.empty(table.shape[0], [('ell', np.int64)] + [(name, np.float64) for name in names])
        for iname, name in enumerate(names): toret[name] = table[:, iname] * self._rsigma8**2
        toret['ell'] = np.arange(table.shape[0])
        return toret

    def lens_potential_cl(self, ellmax=-1):
        r"""Return potential :math:`C_{\ell}` ['pp','tp','ep'], unitless."""
        # self._engine.compute('lensing')
        if not self.hr.Params.DoLensing:
            raise CAMBError('You asked for potential cl, but they have not been calculated. Please set lensing = True.')
        self.hr = self._engine.hr
        if ellmax < 0:
            ellmax = self.ellmax_cl + 1 + ellmax
        table = self.hr.get_lens_potential_cls(lmax=ellmax, CMB_unit=None, raw_cl=True)
        names = ['pp', 'tp', 'ep']
        toret = np.empty(table.shape[0], [('ell', np.int64)] + [(name, np.float64) for name in names])
        for iname, name in enumerate(names): toret[name] = table[:, iname] * self._rsigma8**2
        toret['ell'] = np.arange(table.shape[0])
        return toret

    def lensed_cl(self, ellmax=-1):
        r"""Return lensed :math:`C_{\ell}` ['tt','ee','bb','te'], unitless."""
        if ellmax < 0:
            ellmax = self.ellmax_cl + 1 + ellmax
        if not self.hr.Params.DoLensing:
            raise CAMBError('You asked for lensed cl, but they have not been calculated. Please set lensing = True.')
        table = self.hr.get_total_cls(lmax=ellmax, CMB_unit=None, raw_cl=True)
        names = ['tt', 'ee', 'bb', 'te']
        toret = np.empty(table.shape[0], [('ell', np.int64)] + [(name, np.float64) for name in names])
        for iname, name in enumerate(names): toret[name] = table[:, iname] * self._rsigma8**2
        toret['ell'] = np.arange(table.shape[0])
        return toret


class Fourier(BaseSection):

    def __init__(self, engine):
        self._engine = engine
        self._engine.compute('fourier')
        self.fo = self._engine.fo
        self._rsigma8 = self._engine._rescale_sigma8()

    def _checkz(self, z):
        """Check that perturbations are calculated at several redshifts, else raise an error if ``z`` not close to requested redshift."""
        nz = len(self.fo.transfer_redshifts)
        if nz == 1:
            zcalc = self.fo.transfer_redshifts[0]
            if not np.allclose(z, zcalc):
                raise CosmologyError('Power spectrum computed for a single redshift z = {:.2g}, cannot interpolate to {:.2g}.'.format(zcalc, z))
        return nz

    def sigma_rz(self, r, z, of='delta_m', **kwargs):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`r \mathrm{Mpc}/h`."""
        return self.pk_interpolator(nonlinear=False, of=of, **kwargs).sigma_rz(r, z)

    def sigma8_z(self, z, of='delta_m'):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`8 \mathrm{Mpc}/h`."""
        return self.sigma_rz(8., z, of=of)

    @property
    def sigma8_m(self):
        r"""Current r.m.s. of matter perturbations in a sphere of :math:`8 \mathrm{Mpc}/h`, unitless."""
        return self.fo.get_sigma8_0() * self._rsigma8

    @staticmethod
    def _index_pk_of(of='delta_m'):
        """Convert to CAMB naming conventions."""
        return {'delta_m': 'delta_tot', 'delta_cb': 'delta_nonu', 'theta_cdm': 'v_newtonian_cdm', 'theta_b': 'v_newtonian_baryon'}[of]

    def table(self, nonlinear=False, of='m'):
        r"""
        Return power spectrum table, in :math:`(\mathrm{Mpc}/h)^{3}`.

        Parameters
        ----------
        nonlinear : bool, default=False
            Whether to return the nonlinear power spectrum (if requested in parameters, with 'nonlinear':'halofit' or 'HMcode').
            Computed only for of == 'delta_m' or 'delta_cb'.

        of : string, tuple, default='delta_m'
            Perturbed quantities.
            No difference made between 'theta_cb' and 'theta_m'.

        Returns
        -------
        k : array
            Wavenumbers.

        z : array
            Redshifts.

        pk : array
            Power spectrum array of shape (len(k),len(z)).
        """
        if not isinstance(of, (tuple, list)):
            of = (of, of)

        of = list(of)

        def get_pk(var1, var2):
            var1 = self._index_pk_of(var1)
            var2 = self._index_pk_of(var2)
            ka, za, pka = self.fo.get_linear_matter_power_spectrum(var1=var1, var2=var2, hubble_units=True, k_hunit=True, have_power_spectra=True, nonlinear=nonlinear)
            return ka, za, pka.T

        pka = None
        for iof, of_ in enumerate(of):
            if of_ == 'theta_cb':
                tmpof = of.copy()
                ba = Background(self._engine)
                Omegas = self._engine['Omega_cdm'], self._engine['Omega_b']
                Omega_tot = sum(Omegas)
                Omega_cdm, Omega_b = (Omega / Omega_tot for Omega in Omegas)
                if of[iof - 1] == of_:
                    pka_cdm = get_pk('theta_cdm', 'theta_cdm')[-1]
                    pka_cdm_b = get_pk('theta_cdm', 'theta_b')[-1]
                    ka, za, pka_b = get_pk('theta_b', 'theta_b')
                    pka = Omega_cdm**2 * pka_cdm + 2. * Omega_b * Omega_cdm * pka_cdm_b + Omega_b**2 * pka_b
                    # pka *= (ba.efunc(za) * 100 * self._engine['h'] * 1000 / (1+za) / constants.c)**2
                    break
                else:
                    tmpof[iof] = 'theta_cdm'
                    pka_cdm = get_pk(*tmpof)[-1]
                    tmpof[iof] = 'theta_b'
                    ka, za, pka_b = get_pk(*tmpof)
                    pka = Omega_cdm * pka_cdm + Omega_b * pka_b
                    break

        if pka is None:
            ka, za, pka = get_pk(*of)

        pka = pka * self._rsigma8**2
        return ka, za, pka

    def pk_interpolator(self, nonlinear=False, of='delta_m', **kwargs):
        r"""
        Return :class:`PowerSpectrumInterpolator2D` instance.

        Parameters
        ----------
        nonlinear : bool, default=False
            Whether to return the nonlinear power spectrum (if requested in parameters, with 'nonlinear':'halofit' or 'HMcode').
            Computed only for of == 'delta_m'.

        of : string, tuple, default='delta_m'
            Perturbed quantities.
            No difference made between 'theta_cb' and 'theta_m'.

        kwargs : dict
            Arguments for :class:`PowerSpectrumInterpolator2D`.
        """
        ka, za, pka = self.table(nonlinear=nonlinear, of=of)
        return PowerSpectrumInterpolator2D(ka, za, pka, **kwargs)

    def pk_kz(self, k, z, nonlinear=False, of='delta_m'):
        r"""
        Return power spectrum, in :math:`(\mathrm{Mpc}/h)^{3}`.

        Parameters
        ----------
        k : array_like
            Wavenumbers, in :math:`h/\mathrm{Mpc}`.

        z : array_like
            Redshifts.

        nonlinear : bool, default=False
            Whether to return the nonlinear power spectrum (if requested in parameters, with 'nonlinear':'halofit' or 'HMcode').

        of : string, default='delta_m'
            Perturbed quantities.
            No difference made between 'theta_cb' and 'theta_m'.

        Returns
        -------
        pk : array
            Power spectrum array of shape (len(k),len(z)).
        """
        self._checkz(z)
        interp = self.pk_interpolator(nonlinear=nonlinear, of=of)
        return interp(k, z)
