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

If you want to install extra requirements as well (pyclass, CAMB, pyfftw), run::

  pip install git+https://github.com/cosmodesi/cosmoprimo#egg=cosmoprimo[extras]

pyclass with Mac OS
--------------------
Boltzmann code `CLASS <http://class-code.net>`_  will be compiled at installation time.

If you wish to use clang compiler (instead of gcc), you may encounter an error related to ``-fopenmp`` flag.
In this case, you can try to export:

.. code:: bash

  export CC=clang

Before installing **pyclass**. This will set clang OpenMP flags for compilation (see https://github.com/lesgourg/class_public/issues/405).
Note that with Mac OS "gcc" may sometimes point to clang.
