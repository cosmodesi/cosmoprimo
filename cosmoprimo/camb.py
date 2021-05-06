"""Cosmological calculation with the Boltzmann code CAMB."""

import functools

import numpy as np
from scipy import interpolate
import camb
from camb import CAMBdata, model, CAMBError

from .cosmology import BaseEngine, BaseSection, CosmologyError
from .interpolator import PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D
from . import constants


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
    if not isinstance(tasks,list):
        tasks = [tasks]
    tasks = set(tasks)
    if 'thermodynamics' in tasks:
        tasks.discard('background')
    #if 'lensing' in tasks:
    #    tasks.add('harmonic')
    if 'harmonic' in tasks:
        tasks.add('fourier')
    if 'fourier' in tasks:
        tasks.add('transfer')
    return list(tasks)


class CambEngine(BaseEngine):

    """Engine for the Boltzmann code CAMB."""

    def __init__(self, *args, **kwargs):
        # Big thanks to https://github.com/LSSTDESC/CCL/blob/master/pyccl/boltzmann.py!
        super(CambEngine,self).__init__(*args,**kwargs)
        self.camb_params = camb.CAMBparams()
        self.camb_params.set_cosmology(H0=self['H0'],ombh2=self['omega_b'],omch2=self['omega_cdm'],omk=self['Omega_k'],
                                        TCMB=self['T_cmb'],tau=self.get('tau_reio',None),zrei=self.get('z_reio',None),deltazrei=self['reionization_width'],
                                        Alens=self['A_L']) # + neutrinos
        self.camb_params.InitPower.set_params(As=self._get_A_s_fid(),ns=self['n_s'],nrun=self['alpha_s'],pivot_scalar=self['k_pivot'],
                                        pivot_tensor=self['k_pivot'],parameterization='tensor_param_rpivot',r=self['r'])

        self.camb_params.share_delta_neff = False
        self.camb_params.omnuh2 = self['omega_ncdm']
        self.camb_params.num_nu_massless = self['N_ur']
        self.camb_params.num_nu_massive = self['N_ncdm']
        self.camb_params.nu_mass_eigenstates = self['N_ncdm']
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
        g = (constants.TNCDM * (11./4)**(1./3.)) ** 4
        m_ncdm = np.array(self['m_ncdm'])
        self.camb_params.nu_mass_numbers = np.ones(self['N_ncdm'], dtype='i4')
        self.camb_params.nu_mass_fractions = m_ncdm/m_ncdm.sum()
        self.camb_params.nu_mass_degeneracies = np.full(self['N_ncdm'], g, dtype='f8')

        # get YHe from BBN
        self.camb_params.bbn_predictor = camb.bbn.get_predictor()
        self.camb_params.YHe = self.camb_params.bbn_predictor.Y_He(self.camb_params.ombh2*(camb.constants.COBE_CMBTemp / self.camb_params.TCMB)**3,delta_neff)

        self.camb_params.set_classes(dark_energy_model=camb.dark_energy.DarkEnergyFluid)
        self.camb_params.DarkEnergy.set_params(w=self['w0_fld'],wa=self['wa_fld'],cs2=self['cs2_fld'])

        if self['non_linear']:
            self.camb_params.NonLinearModel = camb.nonlinear.Halofit()
            halofit_version = self.get('halofit_version','mead')
            options = {}
            for name in ['HMCode_A_baryon','HMCode_eta_baryon','HMCode_logT_AGN']:
                tmp = self.get(name,None)
                if tmp is not None: options[name] = tmp
            self.camb_params.NonLinearModel.set_params(halofit_version=halofit_version, **options)

        self.camb_params.DoLensing = self['lensing']
        self.camb_params.Want_CMB_lensing = self['lensing']
        self.camb_params.set_for_lmax(lmax=self['ellmax_cl'])
        self.camb_params.set_matter_power(redshifts=self['z_pk'],kmax=self['kmax_pk']*self['h'],nonlinear=self['non_linear'],silent=True)

        if not self['non_linear']:
            assert self.camb_params.NonLinear == camb.model.NonLinear_none

        self.camb_params.WantScalars = 's' in self['modes']
        self.camb_params.WantVectors = 'v' in self['modes']
        self.camb_params.WantTensors = 't' in self['modes']
        for key,value in self.extra_params:
            if key == 'accuracy':
                self.camb_params.set_accuracy(self.extra_params['accuracy'])
            else:
                setattr(self.camb_params,key,value)

        self.ready = enum(ba=False,th=False,tr=False,le=False,hr=False,fo=False)


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
            self.ba = camb.get_background(self.camb_params,no_thermo=True)
            self.ready.ba = True

        if 'thermodynamics' in tasks and not self.ready.th:
            self.ba = self.th = camb.get_background(self.camb_params,no_thermo=False)
            self.ready.ba = self.ready.th = True

        if 'transfer' in tasks and not self.ready.tr:
            self.tr = camb.get_transfer_functions(self.camb_params)
            self.ready.tr = True

        if 'harmonic' in tasks and not self.ready.hr:
            #self.camb_params.Want_CMB = True
            #self.camb_params.DoLensing = self['lensing']
            #self.camb_params.Want_CMB_lensing = self['lensing']
            self.ready.hr = True
            self.ready.fo = False

        if 'lensing' in tasks and not self.ready.le:
            self.camb_params.DoLensing = True
            self.camb_params.Want_CMB_lensing = True
            self.ready.le = True
            self.tr = CAMBdata()
            self.tr.calc_power_spectra(self.camb_params)
            self.le = self.hr = self.fo = self.tr
            self.ready.fo = True

        if 'fourier' in tasks and not self.ready.fo:
            self.tr.calc_power_spectra()
            self.fo = self.hr = self.le = self.tr
            self.ready.fo = True

    def _rescale_sigma8(self):
        """Rescale perturbative quantities to match input sigma8."""
        if hasattr(self,'_rsigma8'):
            return self._rsigma8
        self._rsigma8 = 1.
        if 'sigma8' in self.params:
            self.compute('fourier')
            self._rsigma8 = self['sigma8']/self.fo.get_sigma8_0()
        return self._rsigma8


def makescalar(func):

    @functools.wraps(func)
    def wrapper(self, z):
        toret = func(self, z)
        if np.isscalar(z):
            return toret[0]
        return toret

    return wrapper


class Background(BaseSection):

    def __init__(self, engine):
        self.engine = engine
        self.engine.compute('background')
        self.ba = self.engine.ba
        # convert RHO to 1e10 Msun/h
        self.H0 = self.ba.Params.H0
        self.h = self.H0 / 100
        self._RH0_ = constants.rho_crit_Msunph_per_Mpcph3 * constants.c**2 / (self.H0*1e3)**2 / 3.
        for name in ['k','cdm','b','g','ur','ncdm','de']:
            setattr(self,'Omega0_{}'.format(name),getattr(self,'Omega_{}'.format(name))(0.))

    def Omega_k(self, z):
        r"""Density parameter of curvature, unitless."""
        return self.ba.get_Omega('K',z=z)

    def Omega_cdm(self, z):
        r"""Density parameter of cold dark matter, unitless."""
        return self.ba.get_Omega('cdm',z=z)

    def Omega_b(self, z):
        r"""Density parameter of baryons, unitless."""
        return self.ba.get_Omega('baryon',z=z)

    def Omega_g(self, z):
        r"""Density parameter of photons, unitless."""
        return self.ba.get_Omega('photon',z=z)

    def Omega_ur(self, z):
        r"""Density parameter of ultra relativistic neutrinos, unitless."""
        return self.ba.get_Omega('neutrino',z=z)

    def Omega_ncdm(self, z):
        r"""Density parameter of massive neutrinos, unitless."""
        return self.ba.get_Omega('nu',z=z)

    def Omega_de(self, z):
        r"""Density of total dark energy (fluid + cosmological constant), unitless."""
        return self.ba.get_Omega('de',z=z)

    @makescalar
    def rho_k(self, z):
        r"""
        Comoving density of curvature :math:`\rho_{k}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`.

        This is defined such that:

        .. math::

            \rho_{\mathrm{crit}} = \rho_\mathrm{tot} + \rho_k
        """
        return self.ba.get_background_densities(1./(1+z),vars=['K'])['K'] * self._RH0_ * (1 + z)

    @makescalar
    def rho_cdm(self, z):
        r"""Comoving density of cold dark matter :math:`\rho_{cdm}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.ba.get_background_densities(1./(1+z),vars=['cdm'])['cdm'] * self._RH0_ * (1 + z)

    @makescalar
    def rho_b(self, z):
        r"""Density parameter of baryons, unitless."""
        return self.ba.get_background_densities(1./(1+z),vars=['baryon'])['baryon'] * self._RH0_ * (1 + z)

    @makescalar
    def rho_g(self, z):
        r"""Comoving density of photons :math:`\rho_{g}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.ba.get_background_densities(1./(1+z),vars=['photon'])['photon'] * self._RH0_ * (1 + z)

    @makescalar
    def rho_ur(self, z):
        r"""Comoving density of ultra-relativistic radiation (massless neutrinos) :math:`\rho_{ur}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.ba.get_background_densities(1./(1+z),vars=['neutrino'])['neutrino'] * self._RH0_ * (1 + z)

    @makescalar
    def rho_ncdm(self, z):
        r"""Comoving density of non-relativistic part of massive neutrinos :math:`\rho_{ncdm}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.ba.get_background_densities(1./(1+z),vars=['nu'])['nu'] * self._RH0_ * (1 + z)

    @makescalar
    def rho_de(self, z):
        r"""Comoving total density of dark energy :math:`\rho_{\mathrm{de}}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`.

        This is defined as:

        .. math::

              \rho_{\mathrm{de}}(z) = \rho_{\mathrm{fld}}(z) + \rho_{\mathrm{\Lambda}}(z).
        """
        return self.ba.get_background_densities(1./(1+z),vars=['de'])['de'] * self._RH0_ * (1 + z)

    def time(self, z):
        r"""Proper time (age of universe), in :math:`\mathrm{Gy}`."""
        return np.vectorize(self.ba.physical_time)(z)

    def hubble_function(self, z):
        r"""Hubble function, in :math:`\mathrm{km}/\mathrm{s}/\mathrm{Mpc}`."""
        return self.ba.hubble_parameter(z)

    def efunc(self, z):
        r"""Function giving :math:`E(z)`, where the Hubble parameter is defined as :math:`H(z) = H_{0} E(z)`, unitless."""
        return self.hubble_function(z) / (100. * self.h)

    def comoving_radial_distance(self, z):
        r"""
        Comoving radial distance, in :math:`mathrm{Mpc}/h`.

        See eq. 15 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_C(z)`.
        """
        return self.ba.comoving_radial_distance(z) * self.h

    def luminosity_distance(self, z):
        r"""
        Luminosity distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 21 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{L}(z)`.
        """
        return self.ba.luminosity_distance(z) * self.h

    def angular_diameter_distance(self, z):
        r"""
        Proper angular diameter distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 18 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{A}(z)`.
        """
        return self.ba.angular_diameter_distance(z) * self.h

    def comoving_angular_distance(self, z):
        r"""
        Comoving angular distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 16 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{M}(z)`.
        """
        return self.angular_diameter_distance(z) * (1. + z)


class Thermodynamics(BaseSection):

    def __init__(self, engine):
        self.engine = engine
        self.engine.compute('thermodynamics')
        self.th = self.engine.th
        # convert RHO to 1e10 Msun/h
        self.h = self.th.Params.H0 / 100

        derived = self.th.get_derived_params()
        self.rs_drag = derived['rdrag'] * self.h
        self.z_drag = derived['zdrag']
        self.rs_star = derived['rstar'] * self.h
        self.z_star = derived['zstar']

    def rs_z(self, z):
        """Comoving sound horizon."""
        return self.th.sound_horizon(z) * self.h


class Transfer(BaseSection):

    def __init__(self, engine):
        self.engine = engine
        self.engine.compute('transfer')
        self.tr = self.engine.tr

    def table(self):
        r"""
        Return source functions.

        Returns
        -------
        tk : dict
            Dictionary of perturbed quantities (in array of shape (k size, z size)).
        """
        data = self.tr.get_matter_transfer_data()
        dtype = [(name,'f8') for name in model.transfer_names]
        # shape (k, z)
        toret = np.empty(data.transfer.shape[1:],dtype=dtype)
        for name in model.transfer_names:
            toret[name] = data.transfer_data[model.transfer_names.index(name)]
        return toret


class Primordial(BaseSection):

    def __init__(self, engine):
        self.engine = engine
        self.pr = self.engine.camb_params.InitPower
        self.h = self.engine.camb_params.h
        self._rsigma8 = self.engine._rescale_sigma8()

    @property
    def A_s(self):
        r"""Scalar amplitude of the primordial power spectrum at :math:`k_\mathrm{pivot}`, unitless."""
        return self.pr.As * self._rsigma8**2

    @property
    def ln_1e10_A_s(self):
        r""":math:`\ln(10^{10}A_s)`, unitless."""
        return np.log(1e10*self.A_s)

    @property
    def n_s(self):
        r"""Power-law index i.e. tilt of the primordial power spectrum, unitless."""
        return self.pr.ns

    @property
    def k_pivot(self):
        r"""Primordial power spectrum pivot scale, where the primordial power is equal to :math:`A_{s}`, in :math:`h/\mathrm{Mpc}`."""
        return self.pr.pivot_scalar  / self.h

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
        pk : numpy.ndarray
            The primordial power spectrum.
        """
        index = ['scalar','vector','tensor'].index(mode)
        return self.h**3*self.engine.camb_params.primordial_power(k*self.h, index) * self._rsigma8**2

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
        return PowerSpectrumInterpolator1D.from_callable(pk_callable=lambda k: self.pk_k(k,mode=mode))


class Harmonic(BaseSection):

    def __init__(self, engine):
        self.engine = engine
        self.engine.compute('harmonic')
        self.hr = self.engine.hr
        self._rsigma8 = self.engine._rescale_sigma8()
        self.ellmax_cl = self.engine['ellmax_cl']

    def unlensed_cl(self, ellmax=-1):
        """Return unlensed :math:`C_{\ell}` ['tt','ee','bb','te'], unitless."""
        if ellmax < 0:
            ellmax = self.ellmax_cl + 1 + ellmax
        table = self.hr.get_unlensed_total_cls(lmax=ellmax,CMB_unit=None,raw_cl=True)
        names = ['tt', 'ee', 'bb', 'te']
        toret = np.empty(table.shape[0],[('ell','i8')] + [(name,'f8') for name in names])
        for iname,name in enumerate(names): toret[name] = table[:,iname] * self._rsigma8**2
        toret['ell'] = np.arange(table.shape[0])
        return toret

    def lens_potential_cl(self, ellmax=-1):
        """Return potential :math:`C_{\ell}` ['pp','tp','ep'], unitless."""
        #self.engine.compute('lensing')
        if not self.hr.Params.DoLensing:
            raise CAMBError('You asked for potential cl, but they have not been calculated. Please set lensing = True.')
        self.hr = self.engine.hr
        if ellmax < 0:
            ellmax = self.ellmax_cl + 1 + ellmax
        table = self.hr.get_lens_potential_cls(lmax=ellmax,CMB_unit=None,raw_cl=True)
        names = ['pp','tp','ep']
        toret = np.empty(table.shape[0],[('ell','i8')] + [(name,'f8') for name in names])
        for iname,name in enumerate(names): toret[name] = table[:,iname] * self._rsigma8**2
        toret['ell'] = np.arange(table.shape[0])
        return toret

    def lensed_cl(self, ellmax=-1):
        """Return lensed :math:`C_{\ell}` ['tt','ee','bb','te'], unitless."""
        if ellmax < 0:
            ellmax = self.ellmax_cl + 1 + ellmax
        if not self.hr.Params.DoLensing:
            raise CAMBError('You asked for lensed cl, but they have not been calculated. Please set lensing = True.')
        table = self.hr.get_total_cls(lmax=ellmax,CMB_unit=None,raw_cl=True)
        names = ['tt', 'ee', 'bb', 'te']
        toret = np.empty(table.shape[0],[('ell','i8')] + [(name,'f8') for name in names])
        for iname,name in enumerate(names): toret[name] = table[:,iname] * self._rsigma8**2
        toret['ell'] = np.arange(table.shape[0])
        return toret


class Fourier(BaseSection):

    def __init__(self, engine):
        self.engine = engine
        self.engine.compute('fourier')
        self.fo = self.engine.fo
        self._rsigma8 = self.engine._rescale_sigma8()

    def _checkz(self, z):
        """Check that perturbations are calculated at several redshifts, else raise an error if ``z`` not close to requested redshift."""
        nz = len(self.fo.transfer_redshifts)
        if nz == 1:
            zcalc = self.fo.transfer_redshifts[0]
            if not np.allclose(z,zcalc):
                raise CosmologyError('Power spectrum computed for a single redshift z = {:.2g}, cannot interpolate to {:.2g}.'.format(zcalc,z))
        return nz

    def sigma_rz(self, r, z, of='delta_m', **kwargs):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`r \mathrm{Mpc}/h`."""
        return self.pk_interpolator(nonlinear=False,of=of,**kwargs).sigma_rz(r,z)

    def sigma8_z(self, z, of='delta_m'):
        """Return the r.m.s. of `of` perturbations in sphere of :math:`8 \mathrm{Mpc}/h`."""
        return self.sigma_rz(8.,z,of=of)

    @property
    def sigma8_m(self):
        r"""Current r.m.s. of matter perturbations in a sphere of :math:`8 \mathrm{Mpc}/h`, unitless."""
        return self.fo.get_sigma8_0()

    @staticmethod
    def _index_pk_of(of='delta_m'):
        """Convert to CAMB naming conventions."""
        return {'delta_m':'delta_tot','delta_cb':'delta_nonu','theta_cdm':'v_newtonian_cdm','theta_b':'v_newtonian_baryon'}[of]

    def table(self, nonlinear=False, of='m'):
        """
        Return power spectrum table, in :math:`(\mathrm{Mpc}/h)^{3}`.

        Parameters
        ----------
        nonlinear : bool, default=False
            Whether to return the nonlinear power spectrum (if requested in parameters, with 'nonlinear':'halofit' or 'HMcode').
            Computed only for of == 'delta_m' or 'delta_cb'.

        of : string, tuple, default='delta_m'
            Perturbed quantities.
            Does not make difference between 'theta_cb' and 'theta_m'.

        Returns
        -------
        k : numpy.ndarray
            Wavenumbers.

        z : numpy.ndarray
            Redshifts.

        pk : numpy.ndarray
            Power spectrum array of shape (len(k),len(z)).
        """
        if not isinstance(of,(tuple,list)):
            of = (of,of)

        of = list(of)

        def get_pk(var1, var2):
            var1 = self._index_pk_of(var1)
            var2 = self._index_pk_of(var2)
            ka, za, pka = self.fo.get_linear_matter_power_spectrum(var1=var1,var2=var2,hubble_units=True,k_hunit=True,have_power_spectra=True,nonlinear=nonlinear)
            return ka, za, pka.T

        pka = None
        for iof,of_ in enumerate(of):
            if of_ == 'theta_cb':
                tmpof = of.copy()
                ba = Background(self.engine)
                Omegas = self.engine['Omega_cdm'],self.engine['Omega_b']
                Omega_tot = sum(Omegas)
                Omega_cdm,Omega_b = (Omega/Omega_tot for Omega in Omegas)
                if of[iof-1] == of_:
                    pka_cdm = get_pk('theta_cdm','theta_cdm')[-1]
                    pka_cdm_b = get_pk('theta_cdm','theta_b')[-1]
                    ka,za,pka_b = get_pk('theta_b','theta_b')
                    pka = Omega_cdm**2 * pka_cdm + 2.*Omega_b*Omega_cdm * pka_cdm_b + Omega_b**2 * pka_b
                    #pka *= (ba.efunc(za) * 100 * self.engine['h'] * 1000 / (1+za) / constants.c)**2
                    break
                else:
                    tmpof[iof] = 'theta_cdm'
                    pka_cdm = get_pk(*tmpof)[-1]
                    tmpof[iof] = 'theta_b'
                    ka,za,pka_b = get_pk(*tmpof)
                    pka = Omega_cdm * pka_cdm + Omega_b * pka_b
                    break

        if pka is None:
            ka,za,pka = get_pk(*of)

        pka *= self._rsigma8**2
        return ka,za,pka

    def pk_interpolator(self, nonlinear=False, of='delta_m', **kwargs):
        """
        Return :class:`PowerSpectrumInterpolator2D` instance.

        Parameters
        ----------
        nonlinear : bool, default=False
            Whether to return the nonlinear power spectrum (if requested in parameters, with 'nonlinear':'halofit' or 'HMcode').
            Computed only for of == 'delta_m'.

        of : string, tuple, default='delta_m'
            Perturbed quantities.
            Does not make difference between 'theta_cb' and 'theta_m'.

        kwargs : dict
            Arguments for :class:`PowerSpectrumInterpolator2D`.
        """
        ka,za,pka = self.table(nonlinear=nonlinear,of=of)
        return PowerSpectrumInterpolator2D(ka,za,pka,**kwargs)

    def pk_kz(self, k, z, nonlinear=False, of='delta_m'):
        """
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
            Does not make difference between 'theta_cb' and 'theta_m'.

        Returns
        -------
        pk : numpy.ndarray
            Power spectrum array of shape (len(k),len(z)).
        """
        self._checkz(z)
        interp = self.pk_interpolator(nonlinear=nonlinear,of=of)
        return interp(k,z)