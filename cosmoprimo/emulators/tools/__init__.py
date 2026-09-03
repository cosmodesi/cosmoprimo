"""Emulation, with no cosmology in it.

Everything here works on a plain ``target(params) -> dict`` of named arrays::

    from cosmoprimo.emulators.tools import Emulator, Space

    emu = Emulator(target, Space(samples=chain))
    emu.train(budget=4)                # or engine='taylor', order=3
    emu.predict(h=0.68, omega_cdm=0.12)

Four engines, and the choice is about the node set rather than the accuracy: ``'chebyshev'``
tiles the box with a sparse grid and interpolates, ``'taylor'`` expands about its centre,
``'polynomial'`` fits a declared basis to scattered points, and ``'mlp'`` trains a network on a
pile of them. The first two need every node they plan; the last two do not, which is what lets
them cover a box part of which the calculator refuses.

What a particular calculator knows about itself -- that ``Cl ~ A_s``, that a full-shape spectrum
dilates -- belongs on its target, not here. :mod:`cosmoprimo.emulators` builds those for
cosmologies; ``desilike.emulators`` builds them for its calculators; both call this.
"""

from .space import Space
from .emulate import Emulator, CoverageError, NotTrained, StateVersionError
from .training import TrainingSet, NodeEvaluationError
from .validation import Validation, validate
from .io import write_state, read_state
