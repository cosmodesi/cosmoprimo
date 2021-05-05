"""Useful constants for cosmology."""

import numpy as np
from scipy.constants import *

electronvolt = 1.602176634e-19
# NOTE: using here up to scipy accuracy; replace by e.g. CLASS accuracy?
megaparsec = 1e6*constants.parsec # m
msun = 1.98847 * 1e30 # kg
# h^2 * kg/m^3
rho_crit_kgph_per_mph3 = 3.0 * (100.*1e3/megaparsec)**2 / (8 * constants.pi * constants.gravitational_constant)
# h^2 * kg/m^3 / msun / Mpc^3 = Msun/h / (Mpc/h)^3
rho_crit_Msunph_per_Mpcph3 = rho_crit_kgph_per_mph3 / (10**10*msun) * megaparsec**3
# T_ncdm, as taken from CLASS, explanatory.ini
TNCDM = 0.71611
NEFF = 3.046
