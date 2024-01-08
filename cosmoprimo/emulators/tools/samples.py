import os
from collections import UserDict

import numpy as np
from scipy.stats import qmc
from scipy.stats.qmc import Sobol, Halton, LatinHypercube


from . import mpi
from . import utils
from .utils import BaseClass


class BaseMetaClass(type(UserDict), type(BaseClass)): pass


class Samples(UserDict, BaseClass, metaclass=BaseMetaClass):

    """
    Class representing samples obtained from a calculator.
    Essentially a dictionary of arrays, plus optional attributes :attr:`attrs`.

    .. code-block::

        size = 10
        samples = Samples({'xa': np.linspace(0., 1., size), 'xb': np.linspace(0., 1., size), 'c': np.linspace(0., 1., size)})
        samples.columns('x*')  ['a', 'b']
        samples.select(['xa', 'c'])  # samples restricted to columns ['xa', 'c']
        samples['c']  # access column 'c'
        assert samples.size == size
        assert samples[:3].size == 3  # global slicing

    """

    def __init__(self, samples=None, attrs=None):
        """
        Initialize samples.

        Parameters
        ----------
        samples : dict, default=None
            Dictionary of arrays.

        attrs : dict, default=None
            Optionally, attributes to be stored in :attr:`attrs`.
        """
        self.data = dict(samples or {})
        self.attrs = dict(attrs or {})

    def __getstate__(self):
        return {'data': self.data, 'attrs': self.attrs}

    def __setstate__(self, state):
        super(Samples, self).__setstate__(state)

    def save(self, filename):
        """
        Save to ``filename``.
        If filename ends with '.npy', save as a unique file.
        Else, save in this directory, with a '.npy' file for attrs and each name: array.
        """
        filename = str(filename)
        in_dir = not filename.endswith('.npy')
        state = self.__getstate__()
        if in_dir:
            self.log_info('Saving to directory {}.'.format(filename))
            attrs_fn = os.path.join(filename, 'attrs.npy')
            data_dir = os.path.join(filename, 'data')
            utils.mkdir(filename)
            utils.mkdir(data_dir)
            np.save(attrs_fn, state['attrs'], allow_pickle=True)
            for name, value in state['data'].items():
                np.save(os.path.join(data_dir, name + '.npy'), value)
        else:
            self.log_info('Saving {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
            np.save(filename, state, allow_pickle=True)

    @classmethod
    def load(cls, filename):
        """Load samples."""
        filename = str(filename)
        in_dir = not filename.endswith('.npy')
        if in_dir:
            cls.log_info('Loading from directory {}.'.format(filename))
            attrs_fn = os.path.join(filename, 'attrs.npy')
            data_dir = os.path.join(filename, 'data')
            state = {}
            state['attrs'] = np.load(attrs_fn, allow_pickle=True)[()]
            state['data'] = {}
            for basename in os.listdir(data_dir):
                fn = os.path.join(data_dir, basename)
                if os.path.isfile(fn):
                    state['data'][os.path.splitext(basename)[0]] = np.load(fn)
        else:
            cls.log_info('Loading {}.'.format(filename))
            state = np.load(filename, allow_pickle=True)[()]
        new = cls.from_state(state)
        return new

    def __copy__(self):
        """Shallow copy."""
        new = super().__copy__()
        new.attrs = dict(self.attrs)
        return new

    def deepcopy(self):
        """Deep copy."""
        import copy
        return copy.deepcopy(self)

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        new = self.copy()
        for name in self:
            new[name] = self[name][key]
        return new

    @classmethod
    def concatenate(cls, *others, intersection=False):
        """
        Concatenate input samples, which requires all samples to hold same parameters,
        except if ``intersection == True``, in which case common parameters are selected.
        """
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        if not others: return cls()
        new = cls(attrs=dict(others[0].attrs))
        new_params = []
        if intersection:
            for other in others:
                new_params += [param for param in other if param not in new_params]
        else:
            new_params = list(others[0].keys())
            for other in others:
                other_params = list(other.keys())
                if set(other_params) != set(new_params):
                    raise ValueError('cannot concatenate values as parameters do not match: {} != {}.'.format(new_params, other_params))
        for param in new_params:
            try:
                value = np.concatenate([np.atleast_1d(other[param]) for other in others], axis=0)
            except ValueError as exc:
                raise ValueError('error while concatenating array for parameter {}'.format(param)) from exc
            new[param] = value
        return new

    @classmethod
    def scatter(cls, samples, mpicomm=mpi.COMM_WORLD, mpiroot=0):
        """Scatter accross this MPI communicator."""
        samples = samples or {}
        params, attrs = mpicomm.bcast((list(samples.keys()), samples.attrs) if mpicomm.rank == mpiroot else None, root=0)
        toret = cls(attrs=attrs)
        for param in params:
            toret[param] = mpi.scatter(samples[param] if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, mpiroot=mpiroot)
        return toret

    @classmethod
    def gather(cls, samples, mpicomm=mpi.COMM_WORLD, mpiroot=0):
        """Gather from this MPI communicator."""
        params, attrs = mpicomm.bcast((list(samples.keys()), samples.attrs) if mpicomm.rank == mpiroot else None, root=mpiroot)
        toret = cls(attrs=attrs)
        for param in params:
            toret[param] = mpi.gather(samples[param], mpicomm=mpicomm, mpiroot=mpiroot)
        return toret

    @property
    def shape(self):
        """Samples shape ``(size, )``."""
        for array in self.values():
            return array.shape[:1]
        return tuple()

    @property
    def size(self):
        """Samples size."""
        shape = self.shape
        if shape:
            s = 1
            for ss in shape:
                s *= ss
            return s
        return 0

    def columns(self, include=None, exclude=None):
        """
        Return selected columns.

        Parameters
        ----------
        include : str, list, default=None
            (List of) column names to select including wildcard, e.g. '*' to select all columns,
            ['X.*', 'a'] to select all columns starting with 'X.' and column 'a' (if exists).

        exclude : str, list, default=None
            Same as ``include``, but to exclude columns.

        Returns
        -------
        columns : list
            List of selected columns.
        """
        columns = list(self.keys())
        if include is not None:
            columns = utils.find_names(columns, include)
        if exclude is not None:
            columns = [column for column in columns if column not in utils.find_names(columns, exclude)]
        return columns

    def select(self, include=None, exclude=None):
        """
        Select input columns.

        Parameters
        ----------
        include : str, list, default=None
            (List of) column names to select including wildcard, e.g. '*' to select all columns,
            ['X.*', 'a'] to select all columns starting with 'X.' and column 'a' (if exists).

        exclude : str, list, default=None
            Same as ``include``, but to exclude columns.

        Returns
        -------
        new : Samples
            Samples with list of selected columns.
        """
        new = self.copy()
        new.data = {name: self[name] for name in self.columns(include=include, exclude=exclude)}
        return new


class RQuasiRandomSequence(qmc.QMCEngine):

    def __init__(self, d, seed=0.5):
        super().__init__(d=d)
        self.seed = float(seed)
        phi = 1.0
        # This is the Newton's method, solving phi**(d + 1) - phi - 1 = 0
        eq_check = phi**(self.d + 1) - phi - 1
        while np.abs(eq_check) > 1e-12:
            phi -= (phi**(self.d + 1) - phi - 1) / ((self.d + 1) * phi**self.d - 1)
            eq_check = phi**(self.d + 1) - phi - 1
        self.inv_phi = [phi**(-(1 + d)) for d in range(self.d)]

    def _random(self, n=1, *, workers=1):
        toret = (self.seed + np.arange(self.num_generated + 1, self.num_generated + n + 1)[:, None] * self.inv_phi) % 1.
        self.num_generated += n
        return toret

    def reset(self):
        self.num_generated = 0
        return self

    def fast_forward(self, n):
        self.num_generated += n
        return self


if not hasattr(qmc.QMCEngine, '_random'):  # old scipy version <= 1.8.1
    RQuasiRandomSequence.random = RQuasiRandomSequence._random
    del RQuasiRandomSequence._randoms


def get_qmc_engine(engine):

    return {'sobol': Sobol, 'halton': Halton, 'lhs': LatinHypercube, 'rqrs': RQuasiRandomSequence}.get(engine, engine)


class CalculatorComputationError(Exception):

    """Exception raised by calculator to be caught up with NaNs."""


class BaseSampler(BaseClass):

    """
    Base sampler class.
    Subclasses should implement :meth:`points`.
    Produced samples have columns starting with 'X.', which are input parameters, and the others starting with 'Y.', for calculator outputs.

    .. code-block:: python

        sampler = QMCSampler(calculator, params={'a': (0.8, 1.2), 'b': (0.8, 1.2)}, save_fn=samples_fn)
        sampler.run(niterations=10000)  # samples save to samples_fn
        sampler.samples  # samples on MPI rank 0
    """

    def __init__(self, calculator, params=None, mpicomm=mpi.COMM_WORLD, save_fn=None, samples=None):
        """
        Initialize base sampler.

        Parameters
        ----------
        calculator : callable
            Input calculator.

        params : dict
            Dictionary of {parameter name: parameter limits} for input ``calculator``.

        mpicomm : MPI communicator, default= mpi.COMM_WORLD
            MPI communicator.

        save_fn : str, Path, default=None
            If not ``None``, save samples to this location.
        """
        self.mpicomm = mpicomm
        self.set_calculator(calculator, params)
        if not len(self.params):
            raise ValueError('Provide at least one parameter')
        self.save_fn = save_fn
        self.samples = None
        if self.mpicomm.rank == 0 and samples is not None:
            self.samples = samples if isinstance(samples, Samples) else Samples.load(samples)

    def set_calculator(self, calculator, params):
        """Set calculator and parameters."""
        self.calculator = calculator
        self.params = dict(params)

    def run(self, save_every=20, **kwargs):
        """
        Run sampling. Sampling can be interrupted anytime, and resumed by providing
        the path to the saved samples in ``samples`` argument of :meth:`__init__`.

        Parameters
        ----------
        niterations : int, default=300
            Number of samples to draw.

        save_every : int, default=20
            Save every ``save_every`` iterations.
        """
        if self.mpicomm.rank == 0:
            samples = self.points(**kwargs)
            default_params = {}
            for name in list(samples.keys()):
                samples['X.' + name] = samples.pop(name)
                default_params[name] = np.median(samples['X.' + name], axis=0)
        if self.mpicomm.rank == 0:
            save_every = save_every * self.mpicomm.size
            self.log_info('Running for {:d} iterations (saving after {:d}) on {:d} rank(s).'.format(samples.size, save_every, self.mpicomm.size))
            nsplits = (samples.size + save_every - 1) // save_every
        nsplits = self.mpicomm.bcast(nsplits if self.mpicomm.rank == 0 else None, root=0)
        default_params = self.mpicomm.bcast(default_params if self.mpicomm.rank == 0 else None, root=0)

        try:
            default_state = self.calculator(**default_params)
        except Exception as exc:
            raise ValueError('error when running calculator with params {}, could not obtain default state'.format(default_params)) from exc
        default_state = {name: np.full_like(value, np.nan) for name, value in default_state.items()}
        for isplit in range(nsplits):
            isample_min, isample_max = self.mpicomm.bcast((isplit * samples.size // nsplits, (isplit + 1) * samples.size // nsplits) if self.mpicomm.rank == 0 else None, root=0)
            scatter_samples = Samples.scatter(samples[isample_min:isample_max] if self.mpicomm.rank == 0 else None, mpicomm=self.mpicomm, mpiroot=0)
            for name, value in default_state.items():
                scatter_samples['Y.' + name] = np.repeat(value[None, ...], scatter_samples.size, axis=0)
            for ivalue in range(scatter_samples.size):
                try:
                    state = self.calculator(**{param: scatter_samples['X.' + param][ivalue] for param in self.params})
                except CalculatorComputationError:
                    continue
                for name, value in state.items():
                    scatter_samples['Y.' + name][ivalue] = value
            gather_samples = Samples.gather(scatter_samples, mpicomm=self.mpicomm, mpiroot=0)
            if self.mpicomm.rank == 0:
                gather_samples.attrs['params'] = dict(self.params)
                self.log_info('Done {:d} / {:d}.'.format(isample_max, samples.size))
                if self.samples is None:
                    self.samples = gather_samples
                else:
                    self.samples = Samples.concatenate(self.samples, gather_samples)
                if self.save_fn is not None:
                    self.samples.save(self.save_fn)
            else:
                self.samples = None
        return self.samples

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def points(self, **kwargs):
        # Return Samples instance containing points to use evaluate calculator against.
        raise NotImplementedError


class InputSampler(BaseSampler):

    """Input sampler, i.e. sampler that evaluate calculator on input samples (points)."""
    name = 'input'

    def __init__(self, calculator, samples, params=None, mpicomm=mpi.COMM_WORLD, save_fn=None):
        """
        Initialize input sampler.

        Parameters
        ----------
        calculator : callable
            Input calculator.

        samples : str, Path, Samples
            Input samples. Input parameters to evaluate calculator with should start with 'X.'.

        params : dict
            Dictionary of {parameter name: parameter limits} for input ``calculator``.

        mpicomm : MPI communicator, default= mpi.COMM_WORLD
            MPI communicator.

        save_fn : str, Path, default=None
            If not ``None``, save samples to this location.
        """
        self.mpicomm = mpicomm
        params, self._points = None, None
        if self.mpicomm.rank == 0:
            if params is None:
                params = dict.fromkeys([name[2:] for name in samples if name.startswith('X.')])
            self._points = Samples({name: samples['X.' + name] for name in params})
        params = self.mpicomm.bcast(params, root=0)
        super(InputSampler, self).__init__(calculator, params=params, mpicomm=mpicomm, save_fn=save_fn, samples=None)

    def points(self):
        return self._points


class GridSampler(BaseSampler):

    """Grid sampler, i.e. evaluate calculator on a grid of parameters."""
    name = 'grid'

    def __init__(self, calculator, params=None, size=1, grid=None, mpicomm=mpi.COMM_WORLD, save_fn=None, samples=None):
        """
        Initialize grid sampler.

        Parameters
        ----------
        calculator : callable
            Input calculator.

        params : dict
            Dictionary of {parameter name: parameter limits} for input ``calculator``.

        size : int, dict, default=1
            A dictionary mapping parameter name to grid size for this parameter.
            Can be a single value, used for all parameters.

        grid : array, dict, default=None
            A dictionary mapping parameter name (including wildcard) to values.
            If provided, ``size`` is ignored.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator.

        save_fn : str, Path, default=None
            If not ``None``, save samples to this location.

        samples : str, Path, Samples
            Path to or samples to resume from.
        """
        super(GridSampler, self).__init__(calculator, params=params, mpicomm=mpicomm, save_fn=save_fn, samples=samples)
        self.grids = utils.expand_dict(grid, list(self.params.keys()))
        self.size = utils.expand_dict(size, list(self.params.keys()))
        for param, limits in self.params.items():
            grid, size = self.grids[param], self.size[param]
            if grid is None:
                if size is None:
                    raise ValueError('size (and grid) not specified for parameter {}'.format(param))
                size = int(size)
                if size < 1:
                    raise ValueError('size is {} < 1 for parameter {}'.format(size, param))
                if size == 1:
                    grid = [np.mean(limits)]
                else:
                    grid = np.linspace(*limits, size)
                if self.mpicomm.rank == 0:
                    self.log_info('{} grid is {}.'.format(param, grid))
            else:
                grid = np.sort(np.ravel(grid))
            self.grids[param] = grid
        self.grids = list(self.grids.values())
        del self.size

    def points(self, size=1):
        grid = np.meshgrid(*self.grids, indexing='ij')
        samples = Samples({param: value.ravel() for param, value in zip(self.params, grid)})
        nsamples = len(self.samples) if self.samples is not None else 0
        return samples[nsamples:]


class DiffSampler(BaseSampler):

    """Sample points for finite differentiation (emulator engine :class:`TaylorEmulatorEngine`)."""

    def __init__(self, calculator, params=None, order=1, accuracy=2, mpicomm=mpi.COMM_WORLD, save_fn=None, samples=None):
        """
        Initialize diff sampler.

        Parameters
        ----------
        calculator : callable
            Input calculator.

        params : dict
            Dictionary of {parameter name: parameter limits} for input ``calculator``.

        order : int, dict, default=1
            A dictionary mapping parameter name (including wildcard) to maximum derivative order.
            If a single value is provided, applies to all varied parameters.

        accuracy : int, dict, default=2
            A dictionary mapping parameter name (including wildcard) to derivative accuracy (number of points used to estimate it).
            If a single value is provided, applies to all varied parameters.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator.

        save_fn : str, Path, default=None
            If not ``None``, save samples to this location.

        samples : str, Path, Samples
            Path to or samples to resume from.
        """
        super(DiffSampler, self).__init__(calculator, params=params, mpicomm=mpicomm, save_fn=save_fn, samples=samples)
        from .taylor import deriv_ncoeffs

        for name, item in zip(['order', 'accuracy'], [order, accuracy]):
            setattr(self, name, utils.expand_dict(item, list(self.params.keys())))

        for param, value in self.order.items():
            if value is None: value = 0
            self.order[param] = int(value)

        for param in params:
            if self.order[param] == 0: continue
            value = self.accuracy[param]
            if value is None:
                raise ValueError('accuracy not specified for parameter {}'.format(param))
            value = int(value)
            if value < 1:
                raise ValueError('accuracy is {} < 1 for parameter {}'.format(value, param))
            if value % 2:
                raise ValueError('accuracy is {} for parameter {}, but it must be a positive EVEN integer'.format(value, param))
            self.accuracy[param] = value

        self.grid_center, grids = {}, []
        for param, limits in self.params.items():
            if self.order[param]:
                size = deriv_ncoeffs(self.order[param], acc=self.accuracy[param])
                grid = np.linspace(*limits, size)
                hsize = size // 2
                cindex = hsize
                order = np.zeros(len(grid), dtype='i')
                for ord in range(self.order[param], 0, -1):
                    s = deriv_ncoeffs(ord, acc=self.accuracy[param])
                    order[cindex - s // 2:cindex + s // 2 + 1] = ord
                order[cindex] = 0
                center = grid[hsize]
                grid = (grid, order, self.order[param])
                if mpicomm.rank == 0:
                    self.log_info('{} grid is {}.'.format(param, grid[0]))
            else:
                center = np.mean(limits)
                grid = (np.array([center]), np.array([0]), 0)
            self.grid_center[param] = center
            grids.append(grid)
        self.grids = grids

    def points(self):
        from .taylor import deriv_grid
        samples = np.array(deriv_grid(self.grids)).T
        samples = Samples({param: value for param, value in zip(self.params, samples)})
        cidx = True
        for array, grid in zip(samples.values(), self.grids):
            grid = grid[0]
            center = grid[len(grid) // 2]
            atol = 0.
            cidx &= np.isclose(array, center, rtol=0., atol=atol)
        cidx = tuple(np.flatnonzero(cidx))
        assert len(cidx) == 1
        self.log_info('Differentiation will evaluate {:d} points.'.format(len(array)))
        samples.attrs['cidx'] = cidx
        samples.attrs['order'] = self.order
        samples.attrs['accuracy'] = self.accuracy
        nsamples = len(self.samples) if self.samples is not None else 0
        return samples[nsamples:]


class QMCSampler(BaseSampler):

    """Quasi Monte-Carlo sequences, using :mod:`scipy.qmc` (+ RQuasiRandomSequence)."""
    name = 'qmc'

    def __init__(self, calculator, params=None, engine='rqrs', mpicomm=mpi.COMM_WORLD, save_fn=None, samples=None, **kwargs):
        """
        Initialize QMC sampler.

        Parameters
        ----------
        calculator : callable
            Input calculator.

        params : dict
            Dictionary of {parameter name: parameter limits} for input ``calculator``.

        engine : str, default='rqrs'
            QMC engine, to choose from ['sobol', 'halton', 'lhs', 'rqrs'].

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator. If ``None``, defaults to ``calculator``'s :attr:`BaseCalculator.mpicomm`.

        save_fn : str, Path, default=None
            If not ``None``, save samples to this location.

        **kwargs : dict
            Optional engine-specific arguments, e.g. random seed ``seed``.
        """
        super(QMCSampler, self).__init__(calculator, params=params, mpicomm=mpicomm, save_fn=save_fn, samples=samples)
        self.engine = get_qmc_engine(engine)(d=len(self.params), **kwargs)

    def points(self, niterations=300):
        lower, upper = [], []
        for limits in self.params.values():
            lower.append(limits[0])
            upper.append(limits[1])
        self.engine.reset()
        nsamples = len(self.samples) if self.samples is not None else 0
        self.engine.fast_forward(nsamples)
        samples = qmc.scale(self.engine.random(n=niterations), lower, upper).T
        return Samples({param: value for param, value in zip(self.params, samples)})