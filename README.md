# cosmoprimo

**cosmoprimo** is a Python package for primordial cosmology, i.e. background and perturbation quantities typically predicted by Boltzmann codes
[CAMB](https://github.com/cmbant/CAMB) or [CLASS](https://github.com/lesgourg/class_public).
**cosmoprimo** provides a coherent interface (and parameter names!) to *CLASS* and *CAMB*, ensuring same units and same definitions
(e.g. fsigma8 as the amplitude of velocity perturbations in spheres of 8 Mpc/h).
It also includes approximations such as [EH1998](https://arxiv.org/abs/astro-ph/9709112), [BBKS](https://ui.adsabs.harvard.edu/abs/1986ApJ...304...15B/abstract),
together with routines to manipulate these results:
- automatic sigma8 rescaling
- the FFTlog algorithm (using numpy.fft or [pyfftw](https://github.com/pyFFTW/pyFFTW) engines)
- interpolate power spectra, correlation functions (as a function of k and k, z)
- compute sigma(r,z)
- filters to remove BAO features from the power spectrum/correlation function.

## Documentation

Documentation is hosted on Read the Docs, [cosmoprimo docs](https://cosmoprimo.readthedocs.io/).

# Requirements

Only strict requirements are:
- numpy
- scipy

If one wants to use CLASS:
- pyclass

If one wants to use CAMB:
- camb

For faster FFTs in the FFTlog algorithm:
- pyfftw

## Installation

See [cosmoprimo building](https://cosmoprimo.readthedocs.io/en/latest/user/building.html).

## Acknowledgments

Much inspired by [pyccl](https://github.com/LSSTDESC/CCL/tree/master/pyccl) and [nbodykit's cosmology class](https://github.com/bccp/nbodykit/blob/master/nbodykit/cosmology/cosmology.py).
Help from [mcfit](https://github.com/eelregit/mcfit), [hankl](https://github.com/minaskar/hankl), [barry](https://github.com/Samreay/Barry).

## License

**cosmoprimo** is free software distributed under a GPLv3 license. For details see the [LICENSE](https://github.com/adematti/cosmoprimo/blob/main/LICENSE).
