import os
import inspect

import numpy as np

from .jax import numpy as jnp
from . import mpi
from .samples import Samples

from .utils import BaseClass
from . import utils


def find_uniques(li):
    toret = []
    for el in li:
        if el not in toret:
            toret.append(el)
    return toret


class EmulatedCalculator(object):

    @classmethod
    def load(cls, filename):
        return Emulator.load(filename).to_calculator()

    def save(self, fn):
        return self.emulator.save(fn)


class Emulator(BaseClass):

    """
    Class to emulate a :class:`BaseEngine` instance.

    For a :class:`BaseEngine` to be emulated, it must implement, for each section:

    - __getstate__(self): a method returning ``state`` a dictionary of attributes as basic python types and numpy arrays
    - __setstate__(self, state): a method setting section's state
    """

    def __init__(self, calculator, params, engine='mlp', mpicomm=mpi.COMM_WORLD):
        """
        Initialize calculator.

        Parameters
        ----------
        calculator : BaseCalculator
            Input calculator.

        engine : str, dict, BaseEmulatorEngine, default='mlp'
            A dictionary mapping calculator's derived attribute names (including wildcard) to emulator engine,
            which can be a :class:`BaseEmulatorEngine` (type or instance) or one of ['mlp'].
            A single emulator engine can be provided, and used for all calculator's derived attributes.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator.
        """
        self.mpicomm = mpicomm
        self.set_calculator(calculator, params)

        self.classify_varied()
        if self.mpicomm.rank == 0:
            self.log_info('Varied parameters: {}.'.format(list(self.params.keys())))
            self.log_info('Found varying {} and fixed {} outputs.'.format(self.varied, list(self.fixed.keys())))
        if not self.varied:
            raise ValueError('Found no varying quantity in provided calculator')

        if not hasattr(engine, 'items'):
            engine = {'*': engine}
        for engine in engine.values():
            engine = get_engine(engine)
        self.engines = utils.expand_dict(engine, self.varied)
        for name, engine in self.engines.items():
            if engine is None:
                raise ValueError('Engine not specified for varying attribute {}'.format(name))
            engine.initialize(params=self.params, mpicomm=self.mpicomm)

        self.varied_shape = {name: -1 for name in self.engines}
        self.samples = {}

    def set_calculator(self, calculator, params):
        self.calculator = calculator
        sig = inspect.signature(calculator)
        self.defaults = {}
        for param in sig.parameters.values():
            if param.kind == param.POSITIONAL_OR_KEYWORD:
                self.defaults[param.name] = param.default
        self.params = dict(params)

    def classify_varied(self, niterations=3, seed=42):
        self.varied, self.fixed = [], {}
        state = {}
        rng = np.random.RandomState(seed=seed)
        for i in range(niterations):
            params = {param: rng.uniform(*limits) for param, limits in self.params.items()}
            for name, value in self.calculator(**params).items():
                state[name] = state.get(name, []) + [value]
        for name, values in state.items():
            try:
                eq = all(utils.deep_eq(value, values[0]) for value in values)
            except Exception as exc:
                raise ValueError('Unable to check equality of {} (type: {})'.format(name, type(values[0]))) from exc
            if eq:
                self.fixed[name] = values[0]
            else:
                self.varied.append(name)
    @property
    def mpicomm(self):
        return getattr(self, '_mpicomm', mpi.COMM_WORLD)

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = mpicomm
        try:
            for engine in self.engines.values():
                engine.mpicomm = mpicomm
        except AttributeError:
            pass

    def set_samples(self, name=None, samples=None, **kwargs):
        """
        Set samples for :meth:`fit`.

        Parameters
        ----------
        name : str, default=None
            Name of calculator's derived attribute(s) (of :attr:`varied`) these samples apply to.
            If ``None``, samples are set for all attributes.

        samples : Samples, default=None
            Samples containing ``calculator.params`` and calculator's derived attributes :attr:`varied`.
            If ``None``, samples will be generated using engines' :meth:`BaseEmulatorEngine.get_default_samples` methods.

        **kwargs : dict
            If ``samples`` is ``None``, optional arguments for :meth:`BaseEmulatorEngine.get_default_samples`.
        """
        if name is None:
            unique_engines = find_uniques(self.engines.values())
            if len(unique_engines) == 1:
                engine = unique_engines[0]
            else:
                raise ValueError('Provide either attribute name or engine')
        elif isinstance(name, str):
            engine = self.engines[name]
        else:
            engine = name

        def calculator(**params):
            state = self.calculator(**params)
            return {name: state[name] for name in self.varied}

        if self.mpicomm.bcast(samples is None, root=0):
            samples = engine.get_default_samples(calculator, **kwargs)
        elif self.mpicomm.rank == 0:
            samples = samples if isinstance(samples, Samples) else Samples.load(samples)
        for name, eng in self.engines.items():
            if eng is engine:
                self.samples[name] = samples

    def fit(self, name=None, **kwargs):
        """
        Fit :class:`BaseEmulatorEngine` to samples.

        Parameters
        ----------
        name : str, default=None
            Name of calculator's derived attribute(s) (of :attr:`varied`) these samples apply to.
            If ``None``, fits will be performed for all calculator's derived attributes.

        **kwargs : dict
            Optional arguments for :meth:`BaseEmulatorEngine.fit`.
        """
        def _get_X_Y(samples, yname):
            X, Y, attrs = None, None, None
            if self.mpicomm.rank == 0:
                nsamples = samples.size
                X = np.concatenate([samples['X.' + name].reshape(nsamples, 1) for name in self.params], axis=-1)
                Y = samples['Y.' + yname]
                yshape = Y.shape[1:]
                Y = Y.reshape(nsamples, -1)
                self.varied_shape[yname] = yshape
                attrs = dict(samples.attrs)
            self.varied_shape[yname] = self.mpicomm.bcast(self.varied_shape[yname], root=0)
            return X, Y, attrs

        if name is None:
            name = list(self.engines.keys())
        if not utils.is_sequence(name):
            name = [name]
        names = name

        for name in names:
            self.engines[name] = engine = self.engines[name].copy()
            engine.fit(*_get_X_Y(self.samples[name], name, **kwargs))

    def predict(self, **params):
        params = {**self.defaults, **params}
        X = jnp.array([params[name] for name in self.params])
        return {**self.fixed, **{name: engine.predict(X).reshape(self.varied_shape[name]) for name, engine in self.engines.items()}}

    def to_calculator(self):

        def calculator(**params):
            return self.predict(**params)

        return calculator

    def __getstate__(self):
        state = {'engines': {}}
        for name, engine in self.engines.items():
            state['engines'][name] = {'name': engine.name, **engine.__getstate__()}
        for name in ['params', 'defaults', 'fixed', 'varied_shape']:
            state[name] = getattr(self, name)
        return state

    def save(self, filename):
        state = self.__getstate__()
        if self.mpicomm.rank == 0:
            self.log_info('Saving {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
            np.save(filename, state, allow_pickle=True)

    def __setstate__(self, state):
        super(Emulator, self).__setstate__(state)
        for name, state in self.engines.items():
            state = state.copy()
            self.engines[name] = get_engine(state.pop('name')).from_state(state)


class RegisteredEmulatorEngine(type(BaseClass)):

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls
        return cls


def get_engine(engine):
    """
    Return engine (class) for emulation.

    Parameters
    ----------
    engine : type, BaseEmulatorEngine, str
        Engine (type or instance) or one of ['taylor', 'mlp'].

    Returns
    -------
    engine : BaseEmulatorEngine
    """
    if isinstance(engine, str):
        engine = engine.lower()
        if engine == 'taylor':
            from . import taylor
        elif engine == 'mlp':
            from . import mlp

        try:
            engine = BaseEmulatorEngine._registry[engine]()
        except KeyError:
            raise ValueError('Unknown engine {}.'.format(engine))

    if isinstance(engine, type):
        engine = engine()
    return engine


class BaseEmulatorEngine(BaseClass, metaclass=RegisteredEmulatorEngine):

    name = 'base'

    def initialize(self, params, mpicomm=mpi.COMM_WORLD):
        self.params = dict(params)
        self.mpicomm = mpicomm

    def get_default_samples(self, calculator):
        raise NotImplementedError

    def fit(self, X, Y, attrs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class PointEmulatorEngine(BaseEmulatorEngine):

    """Basic emulator that returns constant prediction."""
    name = 'point'

    def get_default_samples(self, calculator):
        from .samples import GridSampler
        sampler = GridSampler(calculator, self.params)
        sampler.run()
        return sampler.samples

    def fit(self, X, Y, attrs):
        self.point = np.asarray(self.mpicomm.bcast(Y[0] if self.mpicomm.rank == 0 else None, root=0))

    def predict(self, X):
        # Dumb prediction
        return self.point

    def __getstate__(self):
        state = {}
        for name in ['point']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state