"""Cosmological calculation with the Boltzmann code CAMB."""

import warnings

import numpy as np

from .cosmology import BaseEngine, BaseSection, BaseBackground, CosmologyInputError, CosmologyComputationError
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
        self._set_camb()
        self._camb_params = self.camb.CAMBparams()

        try:

            kwargs = {name: self.get(name, value) for name, value in self.get_default_params().items()}
            YHe = None if self['YHe'] == 'BBN' else self['YHe']
            self._camb_params.set_cosmology(H0=self['H0'], ombh2=self['omega_b'], omch2=self['omega_cdm'], omk=self['Omega_k'],
                                            TCMB=self['T_cmb'], Alens=self['A_L'], YHe=YHe, **kwargs)  # + neutrinos
            # Let's do this by hand for backward-compatibility (for isitgr)
            tau_reio, z_reio, reionization_width = self.get('tau_reio', None), self.get('z_reio', None), self['reionization_width']
            if tau_reio is not None:
                self._camb_params.Reion.set_tau(tau_reio, delta_redshift=reionization_width)
            elif z_reio is not None:
                self._camb_params.Reion.set_zrei(z_reio, delta_redshift=reionization_width)
            self._camb_params.InitPower.set_params(As=self._get_A_s_fid(), ns=self['n_s'], nrun=self['alpha_s'], nrunrun=self['beta_s'], pivot_scalar=self['k_pivot'],
                                                   pivot_tensor=self['k_pivot'], parameterization='tensor_param_rpivot', r=self['r'], nt=self['n_t'], ntrun=self['alpha_t'])

            self._camb_params.share_delta_neff = False
            self._camb_params.omnuh2 = self['omega_ncdm'].sum()
            self._camb_params.num_nu_massless = self['N_ur']
            self._camb_params.num_nu_massive = self['N_ncdm']
            self._camb_params.nu_mass_eigenstates = self['N_ncdm']
            delta_neff = self['N_eff'] - constants.NEFF  # used for BBN YHe comps

            # CAMB defines a neutrino degeneracy factor as T_i = g^(1/4)*T_nu
            # where T_nu is the standard neutrino temperature from first order computations
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
            self._camb_params.bbn_predictor = self.camb.bbn.get_predictor()
            self._camb_params.YHe = self._camb_params.bbn_predictor.Y_He(self._camb_params.ombh2 * (self.camb.constants.COBE_CMBTemp / self._camb_params.TCMB)**3, delta_neff)

            if self._has_fld:
                self._camb_params.set_classes(dark_energy_model=self.camb.dark_energy.DarkEnergyPPF if self['use_ppf'] else self.camb.dark_energy.DarkEnergyFluid)
                self._camb_params.DarkEnergy.set_params(w=self['w0_fld'], wa=self['wa_fld'], cs2=self['cs2_fld'])

            self._camb_params.DoLensing = self['lensing']
            self._camb_params.Want_CMB_lensing = self['lensing']
            self._camb_params.set_for_lmax(lmax=self['ellmax_cl'])
            self._camb_params.set_matter_power(redshifts=self['z_pk'], kmax=self['kmax_pk'] * self['h'], nonlinear=self['non_linear'], silent=True)

            if self['non_linear']:
                self._camb_params.NonLinear = self.camb.model.NonLinear_both
                self._camb_params.NonLinearModel = self.camb.nonlinear.Halofit()

                non_linear = self['non_linear']
                if non_linear in ['mead', 'hmcode']:
                    halofit_version = 'mead'
                elif non_linear in ['halofit']:
                    halofit_version = 'original'
                else:
                    halofit_version = non_linear

                options = {}
                for name in ['HMCode_A_baryon', 'HMCode_eta_baryon', 'HMCode_logT_AGN']:
                    tmp = self.get(name, None)
                    if tmp is not None: options[name] = tmp
                self._camb_params.NonLinearModel.set_params(halofit_version=halofit_version, **options)

            if not self['non_linear']:
                assert self._camb_params.NonLinear == self.camb.model.NonLinear_none

            self._camb_params.WantScalars = 's' in self['modes']
            self._camb_params.WantVectors = 'v' in self['modes']
            self._camb_params.WantTensors = 't' in self['modes']
            for key, value in self._extra_params.items():
                if key == 'accuracy':
                    self._camb_params.set_accuracy(self._extra_params['accuracy'])
                else:
                    setattr(self._camb_params, key, value)

        except (self.camb.baseconfig.CAMBParamRangeError, self.camb.baseconfig.CAMBValueError, self.camb.baseconfig.CAMBError, self.camb.baseconfig.CAMBUnknownArgumentError) as exc:

            raise CosmologyInputError from exc

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
        try:

            if 'background' in tasks and not self.ready.ba:
                self.ba = self.camb.get_background(self._camb_params, no_thermo=True)
                self.ready.ba = True

            if 'thermodynamics' in tasks and not self.ready.th:
                self.ba = self.th = self.camb.get_background(self._camb_params, no_thermo=False)
                self.ready.ba = self.ready.th = True

            if 'transfer' in tasks and not self.ready.tr:
                self.tr = self.camb.get_transfer_functions(self._camb_params)
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
                self.tr = self.camb.CAMBdata()
                self.tr.calc_power_spectra(self._camb_params)
                self.le = self.hr = self.fo = self.tr
                self.ready.fo = True

            if 'fourier' in tasks and not self.ready.fo:
                self.tr.calc_power_spectra()
                self.fo = self.hr = self.le = self.tr
                self.ready.fo = True

        except self.camb.baseconfig.CAMBError as exc:

            raise CosmologyInputError from exc


    def _set_camb(self):
        import camb
        self.camb = camb


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

    @property
    def age(self):
        r"""The current age of the Universe, in :math:`\mathrm{Gy}`."""
        self._engine.compute('thermodynamics')
        return self._engine.th.get_derived_params()['age']

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

    @utils.flatarray(iargs=[0, 1], dtype=np.float64)
    def angular_diameter_distance_2(self, z1, z2):
        r"""
        Angular diameter distance of object at :math:`z_{2}` as seen by observer at :math:`z_{1}`,
        that is, :math:`S_{K}((\chi(z_{2}) - \chi(z_{1})) \sqrt{|K|}) / \sqrt{|K|} / (1 + z_{2})`,
        where :math:`S_{K}` is the identity if :math:`K = 0`, :math:`\sin` if :math:`K < 0`
        and :math:`\sinh` if :math:`K > 0`.
        camb's ``angular_diameter_distance2(z1, z2)`` is not used as it returns 0 when z2 < z1.
        """
        # return self.ba.angular_diameter_distance2(z1, z2) * self._h  # returns 0 when z2 < z1
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


@utils.addproperty('rs_drag', 'z_drag', 'rs_star', 'z_star', 'YHe')
class Thermodynamics(BaseSection):

    def __init__(self, engine):
        self._engine = engine
        self._engine.compute('thermodynamics')
        self.th = self._engine.th
        self.ba = self._engine.ba
        # convert RHO to 1e10 Msun/h
        self._h = self.th.Params.H0 / 100

        derived = self.th.get_derived_params()
        self._rs_drag = derived['rdrag'] * self._h
        self._z_drag = derived['zdrag']
        self._rs_star = derived['rstar'] * self._h
        self._z_star = derived['zstar']
        self._YHe = self._engine._camb_params.YHe

    @utils.flatarray(dtype=np.float64)
    def rs_z(self, z):
        """Comoving sound horizon."""
        return self.th.sound_horizon(z) * self._h

    @property
    def theta_cosmomc(self):
        return self.th.cosmomc_theta()

    @property
    def theta_star(self):
        return self.rs_star / (self.ba.angular_diameter_distance(self.z_star) * self._h) / (1 + self.z_star)


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
        dtype = [(name, np.float64) for name in self._engine.camb.model.transfer_names]
        # shape (k, z)
        toret = np.empty(data.transfer.shape[1:], dtype=dtype)
        for name in self.camb.model.transfer_names:
            toret[name] = data.transfer_data[self._engine.camb.model.transfer_names.index(name)]
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
        r"""Power-law scalar index i.e. tilt of the primordial scalar power spectrum, unitless."""
        return self.pm.ns

    @property
    def alpha_s(self):
        r"""Running of the scalar spectral index at :math:`k_\mathrm{pivot}`, unitless."""
        return self.pm.nrun

    @property
    def beta_s(self):
        r"""Running of the running of the scalar spectral index at :math:`k_\mathrm{pivot}`, unitless."""
        return self.pm.nrunrun

    @property
    def r(self):
        r"""Tensor-to-scalar power spectrum ratio at :math:`k_\mathrm{pivot}`, unitless."""
        return self.pm.r

    @property
    def n_t(self):
        r"""Power-law tensor index i.e. tilt of the tensor primordial power spectrum, unitless."""
        return self.pm.nt

    @property
    def alpha_t(self):
        r"""Running of the tensor spectral index at :math:`k_\mathrm{pivot}`, unitless."""
        return self.pm.ntrun

    @property
    def k_pivot(self):
        r"""Primordial power spectrum pivot scale, where the primordial power is equal to :math:`A_{s}`, in :math:`h/\mathrm{Mpc}`."""
        return self.pm.pivot_scalar / self._h

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
        r"""Return unlensed :math:`C_{\ell}` ['tt', 'ee', 'bb', 'te'], unitless."""
        if ellmax < 0:
            ellmax = self.ellmax_cl + 1 + ellmax
        table = self.hr.get_unlensed_total_cls(lmax=ellmax, CMB_unit=None, raw_cl=True)
        names = ['tt', 'ee', 'bb', 'te']
        toret = np.empty(table.shape[0], [('ell', np.int64)] + [(name, np.float64) for name in names])
        for iname, name in enumerate(names): toret[name] = table[:, iname] * self._rsigma8**2
        toret['ell'] = np.arange(table.shape[0])
        return toret

    def lens_potential_cl(self, ellmax=-1):
        r"""Return potential :math:`C_{\ell}` ['pp', 'tp', 'ep'], unitless."""
        # self._engine.compute('lensing')
        if not self.hr.Params.DoLensing:
            raise self._engine.camb.CAMBError('You asked for potential cl, but they have not been calculated. Please set lensing = True.')
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
        r"""Return lensed :math:`C_{\ell}` ['tt', 'ee', 'bb', 'te'], unitless."""
        if ellmax < 0:
            ellmax = self.ellmax_cl + 1 + ellmax
        if not self.hr.Params.DoLensing:
            raise self._engine.camb.CAMBError('You asked for lensed cl, but they have not been calculated. Please set lensing = True.')
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
        self._h = self._engine._camb_params.h

    def _checkz(self, z):
        """Check that perturbations are calculated at several redshifts, else raise an error if ``z`` not close to requested redshift."""
        nz = len(self.fo.transfer_redshifts)
        if nz == 1:
            zcalc = self.fo.transfer_redshifts[0]
            if not np.allclose(z, zcalc):
                raise CosmologyInputError('Power spectrum computed for a single redshift z = {:.2g}, cannot interpolate to {:.2g}.'.format(zcalc, z))
        return nz

    def sigma_rz(self, r, z, of='delta_m', **kwargs):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`r \mathrm{Mpc}/h`."""
        return self.pk_interpolator(non_linear=False, of=of, **kwargs).sigma_rz(r, z)

    def sigma8_z(self, z, of='delta_m'):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`8 \mathrm{Mpc}/h`."""
        return self.sigma_rz(8., z, of=of)

    @property
    def sigma8_m(self):
        r"""Current r.m.s. of matter perturbations in a sphere of :math:`8 \mathrm{Mpc}/h`, unitless."""
        return self.fo.get_sigma8()[-1] * self._rsigma8

    @staticmethod
    def _index_pk_of(of='delta_m'):
        """Convert to CAMB naming conventions."""
        return {'delta_m': 'delta_tot', 'delta_cb': 'delta_nonu', 'theta_cdm': 'v_newtonian_cdm', 'theta_b': 'v_newtonian_baryon', 'phi_plus_psi': 'Weyl'}[of]

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

        kpow, factor = 0, self._rsigma8**2
        for iof, of_ in enumerate(of):
            if of_ == 'theta_cb':
                Omegas = self._engine['Omega_cdm'], self._engine['Omega_b']
                Omega_tot = sum(Omegas)
                Omega_cdm, Omega_b = (Omega / Omega_tot for Omega in Omegas)
                tmpof = of.copy()
                tmpof[iof] = 'theta_cdm'
                pka_cdm = self.table(non_linear=non_linear, of=tmpof)[-1]
                tmpof[iof] = 'theta_b'
                ka, za, pka_b = self.table(non_linear=non_linear, of=tmpof)
                pka = Omega_cdm * pka_cdm + Omega_b * pka_b
                return ka, za, pka
            if of_ == 'phi_plus_psi':  # we use Weyl ~ k^2 * (phi + psi) / 2
                factor *= 2
                kpow -= 2

        var1, var2 = [self._index_pk_of(of_) for of_ in of]
        # Do the hubble_units, k_hunits conversion manually as it is incorrect for Weyl ~ k^2 (phi + psi) / 2
        ka, za, pka = self.fo.get_linear_matter_power_spectrum(var1=var1, var2=var2, hubble_units=False, k_hunit=False, have_power_spectra=True, nonlinear=non_linear)
        pka = pka.T
        pka = pka * ka[:, None]**kpow * factor
        h = self._h
        return ka / h, za, pka * h**3

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
        self._checkz(z)
        interp = self.pk_interpolator(non_linear=non_linear, of=of)
        return interp(k, z)
