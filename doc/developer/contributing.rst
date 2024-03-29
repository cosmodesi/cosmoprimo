.. _developer-contributing:

Contributing
============

Contributions to **cosmoprimo** are more than welcome!
Please follow these guidelines before filing a pull request:

* Please abide by `PEP8`_ as much as possible in your code writing, add docstrings and tests for each new functionality.

* Check documentation compiles, with the expected result; see :ref:`developer-documentation`.

* Submit your pull request.

Adding a new engine
===================

**cosmoprimo** delegates the actual cosmological calculations to engine, such as *class*, *camb*, etc.
To add another engine (e.g. for a modified *camb* version), just add a new file called :mod:`yourengine.py`.
This module should define :class:`YourEngine` (inheriting from :class:`cosmoprimo.cosmology.BaseEngine`), and
the :class:`BaseSection` "sections" (which take a :class:`YourEngine` instance and to which cosmology attributes / methods are attached)
:class:`Background`, :class:`Thermodynamics`, :class:`Primordial`, :class:`Transfer`, :class:`Harmonic` and :class:`Fourier`.
Take a look at :mod:`cosmoprimo.camb`, and the engine for the modified *camb* :class:`isitgr`.

For modified *class* versions, it is best to implement the corresponding cython wrapper :mod:`yourengine` in **pyclass**
(following the guidelines in its `README <https://github.com/adematti/pyclass/blob/main/README.rst>`_),
and extend the engine and sections of :mod:`classy.py` in :mod:`yourengine.py` as e.g.:

.. code-block:: python

    from pyclass import yourengine

    from .cosmology import BaseEngine
    from . import classy


    class YourEngine(classy.ClassEngine):

        name = 'yourengine'

        def _set_classy(self, params):

            class _ClassEngine(base.ClassEngine):

                def compute(self, tasks):
                    try:
                        return super(_ClassEngine, self).compute(tasks)
                    except base.ClassInputError as exc:
                        raise CosmologyInputError from exc
                    except base.ClassComputationError as exc:
                        raise CosmologyComputationError from exc

            self.classy = _ClassEngine(params=params)


    class Background(classy.BaseClassBackground, yourengine.Background):

        """Your modifications, if any."""


    class Thermodynamics(classy.BaseClassThermodynamics, yourengine.Thermodynamics):

        """Your modifications, if any."""


    """Same for :class:`Primordial`, :class:`Perturbations`, :class:`Transfer`, :class:`Harmonic` and :class:`Fourier`."""


Finally, this new engine can be trivially added to :func:`cosmology.get_engine`,
and one will be able to invoke it as:

.. code-block:: python

    from cosmoprimo import Cosmology

    cosmo = Comology(engine='yourengine')


References
----------

.. target-notes::

.. _`prospector`: http://prospector.landscape.io/en/master/

.. _`PEP8`: https://www.python.org/dev/peps/pep-0008/

.. _`Codacy`: https://app.codacy.com/
