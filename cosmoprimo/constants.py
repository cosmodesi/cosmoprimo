"""Useful constants for cosmology."""

import numpy as np

from scipy import constants
from scipy.constants import *


electronvolt_over_joule = 1.602176634e-19
# NOTE: using here up to scipy accuracy; replace by e.g. CLASS accuracy?
megaparsec_over_m = 1e6 * constants.parsec  # m
msun_over_kg = 1.98847 * 1e30  # kg
# h^2 * kg/m^3
rho_crit_over_kgph_per_mph3 = 3.0 * (100. * 1e3 / megaparsec_over_m)**2 / (8 * constants.pi * constants.gravitational_constant)
# h^2 * kg/m^3 / msun / Mpc^3 = Msun/h / (Mpc/h)^3
rho_crit_over_Msunph_per_Mpcph3 = rho_crit_over_kgph_per_mph3 / (1e10 * msun_over_kg) * megaparsec_over_m**3
# T_ncdm, N_ur as taken from CLASS, explanatory.ini
TNCDM_OVER_CMB = 0.71611
NEFF = 3.044
TCMB = 2.7255
gigayear_over_megaparsec = 3.06601394e2  # conversion factor from megaparsecs to gigayears
