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
  - `pyfftw <https://github.com/pyFFTW/pyFFTW>`_ (for faster FFTs)

pip
---
To install **cosmoprimo**, simply run::

  python -m pip install git+https://github.com/cosmodesi/cosmoprimo

If you want to install extra requirements as well (pyclass, CAMB, astropy, pyfftw), run::

  python -m pip install git+https://github.com/cosmodesi/cosmoprimo#egg=cosmoprimo[class,camb,astropy,extras]

pyclass can be installed independently through::

  python -m pip install git+https://github.com/adematti/pyclass