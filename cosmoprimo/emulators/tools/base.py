import os
import inspect
from typing import Any

import numpy as np
from numpy.core.numeric import normalize_axis_index

from cosmoprimo.jax import jit, vmap
from cosmoprimo.jax import numpy as jnp
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


def make_list(li):
    if li is None:
        return []
    if not utils.is_sequence(li):
        li = [li]
    return list(li)


def sort_varied_fixed(samples, subsample=None):
    """
    Sort varied and fixed arrays in input samples (dictionary of arrays).
    ``subsample`` samples can be selected randomly.
    """
    varied, fixed = {}, {}
    if not samples:
        return varied, fixed
    index = slice(None)
    if subsample is not None:
        rng = np.random.RandomState(seed=42)
        for name, values in samples.items():
            size = values.shape[0]
            break
        index = rng.choice(size, subsample, replace=False)
    for name, values in samples.items():
        values = np.asarray(values)[index]
        try:
            eq = all(utils.deep_eq(value, values[0]) for value in values)
        except Exception as exc:
            raise ValueError('Unable to check equality of {} (type: {})'.format(name, type(values[0]))) from exc
        if eq:
            fixed[name] = values[0]
        else:
            varied[name] = values[0].shape
    return varied, fixed


class EmulatedCalculator(object):

    """
    Load the emulator as a calculator:

    .. code-block:: python

        calculator = EmulatedCalculator.load(fn)
        calculator(**params)  # dictionary of arrays

    """

    @classmethod
    def load(cls, filename):
        return Emulator.load(filename).to_calculator()

    def save(self, fn):
        return self.emulator.save(fn)


class Emulator(BaseClass):

    """
    Class to emulate an input calculator.

    Given the calculator:

    .. code-block::

        def calculator(a=0, b=0):
            x = np.linspace(0., 1., 10)
            return {'x': x, 'y': a * x + b}

    First way: samples computed on-the-fly.

    .. code-block:: python

        emulator = Emulator(calculator, params={'a': (0.8, 1.2), 'b': (0.8, 1.2)}, engine=TaylorEmulatorEngine(order=3))
        emulator.set_samples()  # samples computed on-the-fly
        emulator.fit()
        emulator.save(fn)

        calculator = EmulatedCalculator.load(fn)
        calculator(a=1.1, b=1.1)  # return {'x': ..., 'y': ...}

    Second way: precomputed samples.

    .. code-block:: python

        sampler = QMCSampler(calculator, params={'a': (0.8, 1.2), 'b': (0.8, 1.2)}, save_fn=samples_fn)
        sampler.run(niterations=10000)  # samples save to samples_fn
        sampler.samples  # samples on MPI rank 0

        emulator = Emulator()
        emulator.set_samples(samples=samples_fn, engine=MLPEmulatorEngine(nhidden=(10, 10)))
        emulator.fit(verbose=1)

        calculator = calculator.to_calculator()
        calculator(a=1.1, b=1.1)  # return {'x': ..., 'y': ...}

    """

    def __init__(self, calculator=None, params=None, samples=None, engine=None, xoperation=None, yoperation=None, mpicomm=mpi.COMM_WORLD):
        """
        Initialize emulator.

        Parameters
        ----------
        calculator : BaseCalculator, default=None
            Input calculator, a function ``calculator(**x) -> y``, with ``x`` a dictionary of parameters and ``y`` a dictionary containing output arrays.
            Optional if ``samples`` are directly provided.

        params : list, dict, default=None
            Dictionary of {parameter name: parameter limits} for input ``calculator``.
            Optional if ``samples`` are directly provided.

        samples : Samples, default=None
            Input samples, typically obtained with one of :class:`BaseSampler` subclasses, e.g. :class:`QMCSampler`.
            Can be provided on MPI rank 0 only.

        engine : str, dict, BaseEmulatorEngine, default=None
            A dictionary mapping calculator's output names (including wildcard) to emulator engine,
            which can be a :class:`BaseEmulatorEngine` (type or instance) or one of ['taylor', 'mlp'].
            A single emulator engine can be provided, and used for all calculator's ouputs.

        xoperation : str, BaseOperation, default=None
            Optionally, operation to apply to input 'x' dictionary of parameters, before it is fed to the emulator engine(s).

        yoperation : str, BaseOperation, default=None
            Optionally, operation to apply to output 'y' dictionary of parameters, before it is fed to the emulator engine(s).

        mpicomm : MPI communicator, default=mpi.COMM_WORLD
            Optionally, the MPI communicator.
        """
        self.mpicomm = mpicomm

        self.xoperations = [get_operation(operation) for operation in make_list(xoperation)]
        self.yoperations = [get_operation(operation) for operation in make_list(yoperation)]

        self.engines, self.defaults, self.fixed = {}, {}, {}
        if engine is not None:
            self.set_engine(engine)
        if calculator is not None:
            self.set_calculator(calculator, params)
        if self.mpicomm.bcast(samples is not None, root=0):
            self.set_samples(samples=samples)

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

    def set_engine(self, engine, update=True):
        """Set input (str, dict, BaseEmulatorEngine) engine. Not called directly (use :meth:`__init__` or :meth:`set_samples` instead). ``update=True`` to add to existing engines."""
        if not hasattr(engine, 'items'):
            engine = {'*': engine}
        engines = {key: get_engine(engine) for key, engine in engine.items()}
        if not hasattr(self, '_input_engines'): self._input_engines = {}
        if not hasattr(self, '_init_engines'): self._init_engines = {}
        if update:
            self._input_engines.update(engines)
        else:
            self._input_engines = engines

    def set_calculator(self, calculator, params):
        """Set calculator and parameter limits for sampling. Not called directly (use :meth:`__init__` or :meth:`set_samples` instead)."""
        params = dict(params)
        sig = inspect.signature(calculator)
        self.defaults = {}
        for param in sig.parameters.values():
            if param.kind == param.POSITIONAL_OR_KEYWORD:
                self.defaults[param.name] = param.default

        def classify_varied(niterations=3, seed=42):
            state = {}
            rng = np.random.RandomState(seed=seed)
            for i in range(niterations):
                p = {param: rng.uniform(*limits) for param, limits in params.items()}
                for name, value in calculator(**p).items():
                    state[name] = state.get(name, []) + [value]
            return sort_varied_fixed(state)

        varied, fixed = self.mpicomm.bcast(classify_varied(), root=0)

        if self.mpicomm.rank == 0:
            self.log_info('Varied parameters: {}.'.format(list(params)))
            self.log_info('Found varying {} and fixed {} outputs.'.format(list(varied), list(fixed)))
        if not varied:
            raise ValueError('Found no varying quantity in provided calculator')
        self._calculator, self._params, self._varied, self._fixed = calculator, params, varied, fixed

    def _get_engine_X_Y(self, samples, params=None, varied=None, fixed=None):
        # Internal method to apply :attr:`xoperation` and attr:`yoperation` to input samples to engines
        if params is None:
            params = [name[2:] for name in samples.columns('X.*')]
        if varied is None:
            varied = [name[2:] for name in samples.columns('Y.*')]
        X = {name: samples['X.' + name].copy() for name in params}
        Y = dict(fixed or {})
        Y.update({name: samples['Y.' + name].copy() for name in varied})
        for operation in self.yoperations:
            try:
                operation.initialize(Y, X=X)
                Y = operation(Y, X=X)
            except KeyError:
                pass
        for operation in self.xoperations:
            try:
                operation.initialize(X)
                X = operation(X)
            except KeyError:
                pass
        return X, Y

    def set_samples(self, engine=None, samples=None, calculator=None, params=None, **kwargs):
        """
        Set samples for :meth:`fit`.

        Parameters
        ----------
        engine : str, dict, BaseEmulatorEngine, default=None
            A dictionary mapping calculator's output names (including wildcard) to emulator engine,
            which can be a :class:`BaseEmulatorEngine` (type or instance) or one of ['taylor', 'mlp'].
            A single emulator engine can be provided, and used for all calculator's ouputs.
            Optional if already provided when instantiating (:meth:`__init__`).

        samples : Samples, default=None
            Input samples, typically obtained with one of :class:`BaseSampler` subclasses, e.g. :class:`QMCSampler`.
            Optional if already provided when instantiating (:meth:`__init__`).
            Can be provided on MPI rank 0 only.

        calculator : BaseCalculator, default=None
            Input calculator, a function ``calculator(**x) -> y``, with ``x`` a dictionary of parameters and ``y`` a dictionary containing output arrays.
            Optional if ``samples`` are directly provided.
            Optional if already provided when instantiating (:meth:`__init__`).

        params : list, dict, default=None
            Dictionary of {parameter name: parameter limits} for input ``calculator``.
            Optional if already provided when instantiating (:meth:`__init__`).

        **kwargs : dict
            If ``samples`` is ``None``, optional arguments for :meth:`BaseEmulatorEngine.get_default_samples`.
            Used only if ``samples`` is not provided.
        """
        if engine is not None:
            self.set_engine(engine)

        def initialize_engines(params, varied):
            engines = utils.expand_dict(self._input_engines, list(varied))
            for name, engine in engines.items():
                if engine is None:
                    raise ValueError('Engine not specified for varying attribute {}'.format(name))
                engine.initialize(params=params, mpicomm=self.mpicomm)
                #engine.xshape = (len(params),)
                #engine.yshape = varied[name]
            return engines

        if self.mpicomm.bcast(samples is None, root=0):
            if calculator is not None:
                self.set_calculator(calculator, params)

            params, varied, fixed = self._params, self._varied, self._fixed
            this_engines = initialize_engines(params, varied)

            def calculator(**params):
                state = self._calculator(**params)
                return {name: state[name] for name in varied}

            for engine in this_engines.values():
                samples = engine.get_default_samples(calculator, params, **kwargs)
                if samples is not None: samples.attrs['fixed'] = dict(fixed)  # only on rank 0
                break

        else:
            if self.mpicomm.rank == 0:
                samples = samples if isinstance(samples, Samples) else Samples.load(samples)

                if params is None:
                    params = {name[2:]: None for name in samples.columns('X.*')}
                params = dict(params)
                varied, fixed = sort_varied_fixed({name[2:]: samples[name] for name in samples.columns('Y.*')}, subsample=min(samples.size, 10))
            else:
                samples = None

        import warnings
        if self.mpicomm.rank == 0:
            notfinite = [name for name, value in samples.items() if not np.isfinite(value).all()]
            if notfinite:
                warnings.warn('{} are not finite'.format(notfinite))
            X, Y = self._get_engine_X_Y(samples, params=params, varied=varied, fixed=fixed)
            for name in fixed: Y.pop(name)
            varied, _fixed = sort_varied_fixed(Y, subsample=min(samples.size, 10))
            fixed.update(_fixed)
            params = list(X)

        params, varied, fixed = self.mpicomm.bcast((params, varied, fixed) if self.mpicomm.rank == 0 else None, root=0)
        self.fixed.update(fixed)
        if not hasattr(self, 'samples'): self.samples = {}

        this_engines = initialize_engines(params, varied)
        for name, engine in this_engines.items():
            self.samples[name] = samples
            self._init_engines[name] = engine

    def update(self, other=None, **kwargs):
        """
        Update current emulator with another.
        Useful to parallelize fitting.
        """
        if other is not None:
            self.yoperations += other.yoperations
            self.engines.update(other.engines)
            self.defaults.update(other.defaults)
            self.fixed.update(other.fixed)
        self.__dict__.update(kwargs)

    @property
    def params(self):
        """All input parameters."""
        params = []
        for engine in self.engines.values():
            params += [param for param in engine.params if param not in params]
        return params

    def fit(self, name=None, **kwargs):
        """
        Fit :class:`BaseEmulatorEngine` to samples.

        Parameters
        ----------
        name : str, default=None
            Name of calculator's calculator's outputs these samples apply to.
            If ``None``, fits will be performed for all calculator's outputs.

        **kwargs : dict
            Optional arguments for :meth:`BaseEmulatorEngine.fit`.
        """
        def _get_X_Y(samples, yname, params):
            X, Y, attrs = None, None, None
            if self.mpicomm.rank == 0:
                X, Y = self._get_engine_X_Y(samples, params=params, fixed=self.fixed)
                X = np.column_stack([X[name] for name in self.engines[yname].params])
                Y = Y[yname]
                attrs = dict(samples.attrs)
            if not self.mpicomm.bcast(np.isfinite(X).all() if self.mpicomm.rank == 0 else None, root=0):
                raise ValueError('X is not finite')
            if not self.mpicomm.bcast(np.isfinite(Y).all() if self.mpicomm.rank == 0 else None, root=0):
                raise ValueError('{} is not finite'.format(yname))
            return X, Y, attrs  # operations may yield jax arrays

        if name is None:
            name = list(self.samples.keys())
        if not utils.is_sequence(name):
            name = [name]
        names = utils.find_names(list(self.samples.keys()), name)

        for name in names:
            self.engines[name] = engine = self._init_engines[name].copy()
            if self.mpicomm.rank == 0:
                self.log_info('Fitting {}.'.format(name))
            engine.fit(*_get_X_Y(self.samples[name], name, params=engine.params), **kwargs)

    def predict(self, params):
        """Return emulated calculator output 'y' given input params."""
        params = X = {**self.defaults, **params}
        for operation in self.xoperations:
            params = operation(params)
        predict = dict(self.fixed)
        predict.update({name: engine.predict(params) for name, engine in self.engines.items()})
        for operation in self.yoperations[::-1]:
            predict = operation.inverse(predict, X=X)
        return predict

    def to_calculator(self):
        """Return emulated calculator."""

        def calculator(**params):
            return self.predict(params)

        return calculator

    def __getstate__(self):
        state = {'engines': {}, 'xoperations': [], 'yoperations': []}
        for name, engine in self.engines.items():
            #if hasattr(engine, 'yshape'):  # else, not fit yet
            state['engines'][name] = engine.__getstate__()
        for operation in self.xoperations:
            state['xoperations'].append(operation.__getstate__())
        for operation in self.yoperations:
            state['yoperations'].append(operation.__getstate__())
        for name in ['defaults', 'fixed']:
            state[name] = getattr(self, name)
        return state

    def save(self, filename):
        """Save emulator to disk."""
        state = self.__getstate__()
        if self.mpicomm.rank == 0:
            self.log_info('Saving {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
            np.save(filename, state, allow_pickle=True)

    def __setstate__(self, state):
        super(Emulator, self).__setstate__(state)
        for name, state in self.engines.items():
            self.engines[name] = BaseEmulatorEngine.from_state(state)
        for ii, state in enumerate(self.xoperations):
            self.xoperations[ii] = Operation.from_state(state)
        for ii, state in enumerate(self.yoperations):
            self.yoperations[ii] = Operation.from_state(state)


class RegisteredEmulatorEngine(type(BaseClass)):

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls
        return cls


def get_engine(engine):
    """
    Return engine for emulation.

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
    """
    Base class for emulator engine.
    Subclasses should implement :meth:`_fit_no_operation`, :meth:`_predict_no_operation`,
    and inherit :meth:`__init__`, :meth:`__getstate__`, :meth:`__setstate__`.
    """
    name = 'base'

    def __init__(self, xoperation=None, yoperation=None):
        self.xoperations = [get_operation(operation) for operation in make_list(xoperation)]
        self.yoperations = [get_operation(operation) for operation in make_list(yoperation)]

    def initialize(self, params, mpicomm=mpi.COMM_WORLD):
        self.params = list(params)
        self.mpicomm = mpicomm

    def get_default_samples(self, calculator, params):
        raise NotImplementedError

    def fit(self, X, Y, attrs, **kwargs):
        # print('pre', Y.shape)
        if self.mpicomm.rank == 0:
            for operation in self.xoperations:
                operation.initialize(X)
                X = vmap(operation)(X)
            for operation in self.yoperations:
                operation.initialize(Y)
                Y = vmap(operation)(Y)
            xshape, yshape = X.shape[1:], Y.shape[1:]
            X, Y = np.asarray(X).reshape(len(X), -1), np.asarray(Y).reshape(len(Y), -1)
        self.xshape, self.yshape = self.mpicomm.bcast((xshape, yshape) if self.mpicomm.rank == 0 else None, root=0)
        self._fit_no_operation(X, Y, attrs, **kwargs)

    def _fit_no_operation(self, X, Y, attrs):
        raise NotImplementedError

    #@jit(static_argnums=[0])
    def predict(self, params):
        X = jnp.column_stack([params[name] for name in self.params])
        for operation in self.xoperations:
            X = operation(X)
        Y = self._predict_no_operation(X.reshape(-1)).reshape(self.yshape)
        for operation in self.yoperations[::-1]:
            Y = operation.inverse(Y, X=params)
        return Y

    def _predict_no_operation(self, X):
        raise NotImplementedError

    def __getstate__(self):
        state = {}
        for name in ['name', 'params', 'xshape', 'yshape']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        state.update({'xoperations': [], 'yoperations': []})
        for operation in self.xoperations:
            state['xoperations'].append(operation.__getstate__())
        for operation in self.yoperations:
            state['yoperations'].append(operation.__getstate__())
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        for ii, state in enumerate(self.xoperations):
            self.xoperations[ii] = Operation.from_state(state)
        for ii, state in enumerate(self.yoperations):
            self.yoperations[ii] = Operation.from_state(state)

    @classmethod
    def from_state(cls, state):
        state = dict(state)
        name = state.pop('name')
        cls = BaseEmulatorEngine._registry[name]
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def __copy__(self, *args, **kwargs):
        new = super().__copy__(*args, **kwargs)
        new.xoperations = [operation.copy() for operation in self.xoperations]
        new.yoperations = [operation.copy() for operation in self.yoperations]
        return new


class PointEmulatorEngine(BaseEmulatorEngine):

    """Basic emulator that returns constant prediction."""

    name = 'point'

    def get_default_samples(self, calculator, params):
        from .samples import GridSampler
        sampler = GridSampler(calculator, params)
        sampler.run()
        return sampler.samples

    def _fit_no_operation(self, X, Y, attrs):
        self.point = np.asarray(self.mpicomm.bcast(Y[0] if self.mpicomm.rank == 0 else None, root=0))

    def _predict_no_operation(self, X):
        # Dumb prediction
        return self.point

    def __getstate__(self):
        state = super(PointEmulatorEngine, self).__getstate__()
        for name in ['point']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class RegisteredOperation(type(BaseClass)):

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls
        return cls


def get_operation(operation):
    """
    Return operation for emulation.

    Parameters
    ----------
    operation : type, Operation, str
        Operation (type or instance) or one of ['base', 'pca'].

    Returns
    -------
    operation : Operation
    """
    if isinstance(operation, str):
        operation = operation.lower()
        try:
            operation = Operation._registry[operation]()
        except KeyError:
            raise ValueError('Unknown operation {}.'.format(operation))

    if isinstance(operation, type):
        operation = operation()
    return operation


class Operation(BaseClass, metaclass=RegisteredOperation):

    """Base class for operation to apply to calculator input parameters 'x' or output 'y'."""

    name = 'base'
    _locals = {}
    _direct = 'v'
    _inverse = None
    verbose = False

    def __init__(self, direct, inverse=None, locals=None):
        self._locals = dict(locals or {})
        self._direct = str(direct)
        self._inverse = str(inverse) if inverse is not None else None

    def initialize(self, v, **kwargs):
        return

    def __call__(self, v, **kwargs):
        """From values in samples (parameters 'x' and output 'y') to quantities fed to emulators."""
        return utils.evaluate(self._direct, locals={**self._locals, 'v': v, **kwargs}, verbose=self.verbose)

    def inverse(self, v, **kwargs):
        """From emulated quantities to calculator output 'y'."""
        return utils.evaluate(self._inverse, locals={**self._locals, 'v': v, **kwargs}, verbose=self.verbose)

    def __getstate__(self):
        return {name: getattr(self, name) for name in ['name', '_direct', '_inverse', '_locals']}

    @classmethod
    def from_state(cls, state):
        state = dict(state)
        name = state.pop('name')
        cls = Operation._registry[name]
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new


class Log10Operation(Operation):

    """Apply log10 (inverse: 10^x)."""

    name = 'log10'

    def __init__(self):
        super().__init__('jnp.log10(v)',
                         inverse='10**v',
                         locals={})


class ArcsinhOperation(Operation):

    """Apply arcsinh (inverse: sinh)."""

    name = 'arcsinh'

    def __init__(self):
        super().__init__('jnp.arcsinh(v)',
                         inverse='jnp.sinh(v)',
                         locals={})


class ScaleOperation(Operation):

    """Apply rescaling by limits. Use input ``limits`` if provided, else take min / max from samples."""

    name = 'scale'

    def __init__(self, limits=None):
        self.limits = list(limits) if limits else [None] * 2

    def initialize(self, values):
        values = np.asarray(values)
        if self.limits[0] is None:
            self.limits[0] = np.min(values, axis=0)
        if self.limits[1] is None:
            self.limits[1] = np.max(values, axis=0)
        mask = self.limits[1] == self.limits[0]
        self.limits[0] = np.where(mask, 0., self.limits[0])
        self.limits[1] = np.where(mask, 1., self.limits[1])
        super().__init__('(v - limits[0]) / (limits[1] - limits[0])',
                         inverse='v * (limits[1] - limits[0]) + limits[0]',
                         locals={'limits': self.limits})


class NormOperation(Operation):

    """Apply rescaling by mean / std from samples."""

    name = 'norm'

    def __init__(self):
        return

    def initialize(self, v):
        v = np.asarray(v)
        mean, sigma = np.mean(v, axis=0), np.std(v, ddof=1, axis=0)
        sigma = np.where(sigma == 0., 1., sigma)
        super().__init__('(v - mean) / sigma',
                         inverse='v * sigma + mean',
                         locals={'mean': mean, 'sigma': sigma})


class PCAOperation(Operation):

    """Dot values with PCA using ``npcs`` eigenvectors."""

    name = 'pca'

    def __init__(self, npcs=1):
        self.npcs = npcs

    def initialize(self, v, **kwargs):
        v = np.asarray(v)
        self.mean, self.sigma = np.mean(v, axis=0), np.std(v, ddof=1, axis=0)
        self.sigma[self.sigma == 0.] = 1.
        self.eigenvectors = utils.subspace((v - self.mean) / self.sigma, npcs=self.npcs)
        self.eigenvectors = self.eigenvectors.T.reshape((-1,) + self.mean.shape)

    def __call__(self, v):
        return jnp.sum(jnp.expand_dims((v - self.mean) / self.sigma, axis=0) * self.eigenvectors, axis=tuple(range(1, self.eigenvectors.ndim)))

    def inverse(self, v):
        return jnp.sum(jnp.expand_dims(v, axis=tuple(range(1, self.eigenvectors.ndim))) * self.eigenvectors, axis=0) * self.sigma + self.mean

    def __getstate__(self):
        return {name: getattr(self, name) for name in ['name', 'mean', 'sigma', 'eigenvectors'] if hasattr(self, name)}


class ChebyshevOperation(Operation):

    """
    Dot values with Chebyshev basis up to order ``order`` along ``axis``.
    Some numerical unaccuracy if ``order > shape[axis] / 2``.
    """

    name = 'chebyshev'

    def __init__(self, order, axis=-1):
        self.order = int(order)
        self.axis = int(axis)

    def initialize(self, v, **kwargs):
        from scipy import special
        poly = []
        size = v.shape[1:][self.axis]
        ndim = v.ndim - 1
        shape = [1] * ndim
        self.axis = normalize_axis_index(self.axis, ndim)
        shape.insert(self.axis, size)
        if self.order > size // 2:
            import warnings
            warnings.warn('order = {:d} for size = {:d} is unstable.'.format(self.order, size))
        for n in range(self.order + 1):
            x = np.linspace(-1., 1., size).reshape(shape)
            poly.append(special.chebyt(n)(x))
        # self.axis = input axis, self.axis + 1 = output axis
        self.poly = np.concatenate(poly, axis=self.axis + 1)  # shape = (1,..., size, self.order + 1, 1,...)
        flatpoly = np.reshape(self.poly, (size, -1))
        self.proj = flatpoly.dot(np.linalg.inv(flatpoly.T.dot(flatpoly))).reshape(self.poly.shape)  # projector, for C = eye(size)

    def __call__(self, v):
        return jnp.sum(jnp.expand_dims(v, self.axis + 1) * self.poly, axis=self.axis)  # shape is (v.shape[0], ..., self.order + 1, ..., v.shape[-1])

    def inverse(self, v):
        return jnp.sum(jnp.expand_dims(v, self.axis) * self.proj, axis=self.axis + 1)

    def __getstate__(self):
        return {name: getattr(self, name) for name in ['name', 'proj', 'poly', 'axis'] if hasattr(self, name)}