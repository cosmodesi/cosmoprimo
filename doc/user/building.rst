.. _user-building:

Building
========

Requirements
------------
Only strict requirements are:

  - numpy
  - scipy

Extra requirements are:

  - `pyclass <https://github.com/adematti/pyclass>`_ (for CLASS)
  - `CAMB <https://github.com/cmbant/CAMB>`_ (for CAMB)
  - `DISCO-EB <https://github.com/dforero0896/DISCO-EB>`_ (for the differentiable
    Einstein-Boltzmann solver; pulls in jax, diffrax, equinox, jaxtyping, jax_cosmo and lineax)
  - `pyfftw <https://github.com/pyFFTW/pyFFTW>`_ (for faster FFTs)

pip
---
To install **cosmoprimo**, simply run::

  python -m pip install git+https://github.com/cosmodesi/cosmoprimo

If you want to install extra requirements as well (pyclass, CAMB, astropy, pyfftw), run::

  python -m pip install git+https://github.com/cosmodesi/cosmoprimo#egg=cosmoprimo[class,camb,astropy,extras]

pyclass can be installed independently through::

  python -m pip install git+https://github.com/adematti/pyclass

DISCO-EB can be installed independently through::

  python -m pip install git+https://github.com/dforero0896/DISCO-EB.git@master

.. note::
   The ``discoeb`` extra points at this DISCO-EB fork, whose ``master`` carries a fix (correct
   batched-solver output for more than one output redshift) that is not upstream yet. Once the
   DISCO-EB pull request is merged upstream, ``git+https://github.com/ohahn/DISCO-EB`` can be used
   instead.