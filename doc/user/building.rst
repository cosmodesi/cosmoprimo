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

PIP
---
To install **cosmoprimo**, simply run::

  python -m pip install git+https://github.com/adematti/cosmoprimo

If you want to install extra requirements as well (pyclass, CAMB, pyfftw), run::

  pip install git+https://github.com/adematti/cosmoprimo#egg=cosmoprimo[extras]
