import sys
import traceback
import warnings

import numpy as np

from cosmosis.datablock import names, option_section

# These are pre-defined strings we use as datablock
# section names
cosmo = names.cosmological_parameters


def setup(options):

    # Read options from the ini file which are fixed across
    # the length of the chain
    config = {'lmax': options.get_int(option_section, 'lmax', default=2000),
              'zmax': options.get_double(option_section, 'zmax', default=3.0),
              'kmax': options.get_double(option_section, 'kmax', default=50.0),
              'debug': options.get_bool(option_section, 'debug', default=False),
              'lensing': options.get_bool(option_section, 'lensing', default=True),
              'cmb': options.get_bool(option_section, 'cmb', default=True),
              'mpk': options.get_bool(option_section, 'mpk', default=True)}


    for _, key in options.keys(option_section):
        if key.startswith('cosmoprimo_'):
            config[key] = options[option_section, key]

    # Return all this config information
    return config


def get_cosmoprimo_inputs(block, config):

    # Get parameters from block and give them the
    # names and form that cosmoprimo expects
    nnu = block.get_double(names.cosmological_parameters, 'nnu', default=3.046)
    nmassive = block.get_int(names.cosmological_parameters, 'num_massive_neutrinos', default=0)
    m_ncdm = block.get_double(names.cosmological_parameters, 'mnu', default=0.06)
    if not nmassive:
        m_ncdm = []
    else:
        m_ncdm = [m_ncdm / nmassive] * nmassive

    params = {'lensing': bool(config['lensing']),
              'A_s': block[names.cosmological_parameters, 'A_s'],
              'n_s': block[names.cosmological_parameters, 'n_s'],
              'H0': 100 * block[names.cosmological_parameters, 'h0'],
              'omega_b': block[names.cosmological_parameters, 'ombh2'],
              'omega_cdm': block[names.cosmological_parameters, 'omch2'],
              'tau_reio': block[names.cosmological_parameters, 'tau'],
              'T_cmb': block.get_double(names.cosmological_parameters, 'TCMB', default=2.726),
              'N_eff': nnu,
              'm_ncdm': m_ncdm,
              'engine': config.get('engine', 'class')}

    if config["cmb"] or config["lensing"]:
        params.update({
          'ellmax_cl': config["lmax"],
        })

    if config["mpk"]:
        params.update({
            'kmax_pk':  config["kmax"],
            'z_pk': np.arange(0.0, config['zmax'], 0.01),
        })

    if block.has_value(names.cosmological_parameters, "massless_nu"):
        warnings.warn("Parameter massless_nu is being ignored. Set nnu, the effective number of relativistic species in the early Universe.")

    if (block.has_value(names.cosmological_parameters, "omega_nu") or block.has_value(names.cosmological_parameters, "omnuh2")) and not (block.has_value(names.cosmological_parameters, "mnu")):
        warnings.warn("Parameter omega_nu and omnuh2 are being ignored. Set mnu and num_massive_neutrinos instead.")

    for key, val in config.items():
        if key.startswith('cosmoprimo_'):
            params[key[6:]] = val

    return params


def get_cosmoprimo_outputs(block, c, config):

    # Define z we want to sample
    dz = 0.01
    z = np.arange(0.0, config["zmax"] + dz, dz)
    #k = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    nz = len(z)

    ba = c.get_background()

    # Extract (interpolate) P(k,z) at the requested
    # sample points.
    if config["mpk"]:
        fo = c.get_fourier()
        block[names.cosmological_parameters, 'sigma_8'] = fo.sigma8_m

        section_names = {'matter_power_lin': 'delta_m', 'cdm_baryon_power_lin': 'delta_cb'}

        for section_name, of in section_names.items():
            pk_interpolator = fo.pk_interpolator(of=of)
            block.put_grid(section_name, "k_h", pk_interpolator.k, "z", pk_interpolator.z, "p_k", pk_interpolator.pk)

        if c['non_linear']:
            pk_interpolator = fo.pk_interpolator(of='delta_m', non_linear=True)
            block.put_grid("matter_power_nl", "k_h", pk_interpolator.k, "z", pk_interpolator.z, "p_k", pk_interpolator.pk)

        # Get growth rates and sigma_8
        sigma_8, fsigma_8 = fo.sigma8_z(z, of='delta_m'), fo.sigma8_z(z, of='theta_m')
        d_z = sigma_8 / fo.sigma8_z(0., of='delta_m')
        f_z = fsigma_8 / sigma_8
        block[names.growth_parameters, "z"] = z
        block[names.growth_parameters, "sigma_8"] = sigma_8
        block[names.growth_parameters, "fsigma_8"] = fsigma_8
        block[names.growth_parameters, "d_z"] = d_z
        block[names.growth_parameters, "f_z"] = f_z
        block[names.growth_parameters, "a"] = 1 / (1 + z)

        block[names.cosmological_parameters, "sigma_8"] = sigma_8[0]
        # sigma12 and S_8 - other variants of sigma_8
        block[names.cosmological_parameters, "sigma_12"] = fo.sigma_rz(12. / ba.h, 0.)
        block[names.cosmological_parameters, "S_8"] = sigma_8[0] * np.sqrt(ba.Omega0_m / 0.3)

    ##
    # Distances and related quantities
    ##

    # save redshifts of samples
    block[names.distances, 'z'] = z
    block[names.distances, 'nz'] = nz

    # Save distance samples
    D_L = ba.luminosity_distance(z)
    D_A = ba.angular_diameter_distance(z)
    D_M = D_A * (1 + z)
    from cosmoprimo import constants
    H = (100. * ba.efunc(z)) / (constants.c / 1e3)
    D_V = (z * D_M**2 / H)**(1./3.)
    block[names.distances, 'D_L'] = D_L
    block[names.distances, 'D_A'] = D_A
    block[names.distances, 'D_M'] = D_M
    block[names.distances, 'D_V'] = D_V
    mu = np.full_like(D_L, -np.inf)
    mask = D_L > 0
    mu[mask] = 5. * np.log10(D_L[mask]) + 25.
    block[names.distances, 'mu'] = mu

    # Save some auxiliary related parameters
    block[names.distances, 'age'] = ba.age

    th = c.get_thermodynamics()
    block[names.distances, 'rs_zdrag'] = th.rs_drag

    rs_DV = th.rs_drag * D_V
    F_AP = D_M * H
    block[names.distances, 'rs_DV'] = rs_DV
    block[names.distances, 'F_AP'] = F_AP

    ##
    # Now the CMB C_ell
    ##
    if config["cmb"]:
        c_ell_data = c.lensed_cl() if config['lensing'] else c.unlensed_cl()
        ell = c_ell_data['ell']
        ell = ell[2:]

        # Save the ell range
        block[names.cmb_cl, "ell"] = ell

        # t_cmb is in K, convert to mu_K, and add ell(ell+1) factor
        tcmb_muk = block[names.cosmological_parameters, 'tcmb'] * 1e6
        factor = ell * (ell + 1.0) / 2 / np.pi * tcmb_muk**2

        # Save each of the four spectra
        for s in ['tt', 'ee', 'te', 'bb']:
            block[names.cmb_cl, s] = c_ell_data[s][2:] * factor


def execute(block, config):
    from cosmoprimo import Cosmology, CosmologyError

    try:
        # Set input parameters
        params = get_cosmoprimo_inputs(block, config)
        c = Cosmology(**params)

        # Extract outputs
        get_cosmoprimo_outputs(block, c, config)
    except CosmologyError as error:
        if config['debug']:
            sys.stderr.write("Error in cosmoprimo. You set debug=T so here is more debug info:\n")
            traceback.print_exc(file=sys.stderr)
        else:
            sys.stderr.write("Error in cosmoprimo. Set debug=T for info: {}\n".format(error))
        return 1
    return 0


def cleanup(config):
    pass