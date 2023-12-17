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

    def __init__(self, samples=None, attrs=None):
        self.data = dict(samples or {})
        self.attrs = dict(attrs or {})

    def __getstate__(self):
        return {'data': self.data, 'attrs': self.attrs}

    def __setstate__(self, state):
        super(Samples, self).__setstate__(state)

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def save(self, filename):
        """Save to ``filename``."""
        self.log_info('Saving {}.'.format(filename))
        utils.mkdir(os.path.dirname(filename))
        np.save(filename, self.__getstate__(), allow_pickle=True)

    @classmethod
    def load(cls, filename):
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.from_state(state)
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
        new = cls()
        new_params = []
        others = list(others[:1]) + [other for other in others[1:] if other]
        if intersection:
            for other in others:
                new_params += [param for param in other if param not in new_params]
        else:
            new_params = list(other.keys())
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
        """Scatter."""
        samples = samples or {}
        params, attrs = mpicomm.bcast((list(samples.keys()), samples.attrs) if mpicomm.rank == mpiroot else None, root=0)
        toret = cls(attrs=attrs)
        for param in params:
            toret[param] = mpi.scatter(samples[param] if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, mpiroot=mpiroot)
        return toret

    @classmethod
    def gather(cls, samples, mpicomm=mpi.COMM_WORLD, mpiroot=0):
        """Gather."""
        params, attrs = mpicomm.bcast((list(samples.keys()), samples.attrs) if mpicomm.rank == mpiroot else None, root=0)
        toret = cls(attrs=attrs)
        for param in params:
            toret[param] = mpi.gather(samples[param] if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, mpiroot=mpiroot)
        return toret

    @property
    def shape(self):
        for array in self.values():
            return array.shape
        return tuple()

    @property
    def size(self):
        shape = self.shape
        if shape:
            s = 1
            for ss in shape:
                s *= ss
            return s
        return 0


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


class BaseSampler(BaseClass):

    """Base sampler class."""

    def __init__(self, calculator, params=None, mpicomm=mpi.COMM_WORLD, save_fn=None, **kwargs):
        """
        Initialize base sampler.

        Parameters
        ----------
        calculator : callable
            Input calculator.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator.

        save_fn : str, Path, default=None
            If not ``None``, save samples to this location.

        **kwargs : dict
            Optional engine-specific arguments.
        """
        self.mpicomm = mpicomm
        self.set_calculator(calculator, params)
        if not len(self.params):
            raise ValueError('Provide at least one parameter')
        self.save_fn = save_fn
        self.samples = None

    def set_calculator(self, calculator, params):
        """Set calculator and parameters."""
        self.calculator = calculator
        self.params = dict(params)

    def run(self, **kwargs):
        """
        Run sampling. Sampling can be interrupted anytime, and resumed by providing
        the path to the saved samples in ``samples`` argument of :meth:`__init__`.

        Parameters
        ----------
        niterations : int, default=300
            Number of samples to draw.
        """
        samples = None
        if self.mpicomm.rank == 0:
            samples = self.points(**kwargs)
            for name in list(samples.keys()):
                samples['X.' + name] = samples.pop(name)
        local_samples = Samples.scatter(samples, mpicomm=self.mpicomm, mpiroot=0)
        for ivalue in range(local_samples.size):
            state = self.calculator(**{param: local_samples['X.' + param][ivalue] for param in self.params})
            for name, value in state.items():
                name = 'Y.' + name
                local_samples.setdefault(name, [])
                local_samples[name].append(value)
        samples = Samples.gather(local_samples, mpicomm=self.mpicomm, mpiroot=0)
        if self.mpicomm.rank == 0:
            if self.samples is None:
                self.samples = samples
            else:
                self.samples = Samples.concatenate(self.samples, samples)
            if self.save_fn is not None:
                self.samples.save(self.save_fn)
        else:
            self.samples = None
        return self.samples

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


class GridSampler(BaseSampler):

    """Grid sampler."""
    name = 'grid'

    def __init__(self, calculator, params=None, mpicomm=mpi.COMM_WORLD, save_fn=None, size=1, grid=None):
        """
        Initialize grid sampler.

        Parameters
        ----------
        calculator : callable
            Input calculator.

        samples : str, Path, Samples
            Path to or samples to resume from.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator.

        save_fn : str, Path, default=None
            If not ``None``, save samples to this location.

        size : int, dict, default=1
            A dictionary mapping parameter name to grid size for this parameter.
            Can be a single value, used for all parameters.

        grid : array, dict, default=None
            A dictionary mapping parameter name (including wildcard) to values.
            If provided, ``size`` and ``ref_scale`` are ignored.
        """
        super(GridSampler, self).__init__(calculator, params=params, mpicomm=mpicomm, save_fn=save_fn)
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
        return Samples({param: value.ravel() for param, value in zip(self.params, grid)})


class DiffSampler(BaseSampler):

    """Sample points for finite differentiation."""

    def __init__(self, calculator, params=None, order=1, accuracy=2, mpicomm=mpi.COMM_WORLD, save_fn=None, **kwargs):
        """
        Initialize diff sampler.

        calculator : callable
            Input calculator.

        samples : str, Path, Samples
            Path to or samples to resume from.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator.

        save_fn : str, Path, default=None
            If not ``None``, save samples to this location.
        """
        super(DiffSampler, self).__init__(calculator, params=params, mpicomm=mpicomm, save_fn=save_fn)
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
        return samples


class QMCSampler(BaseSampler):

    """Quasi Monte-Carlo sequences, using :mod:`scipy.qmc` (+ RQuasiRandomSequence)."""
    name = 'qmc'

    def __init__(self, calculator, params=None, samples=None, mpicomm=mpi.COMM_WORLD, engine='rqrs', save_fn=None, **kwargs):
        """
        Initialize QMC sampler.

        Parameters
        ----------
        calculator : callable
            Input calculator.

        samples : str, Path, Samples
            Path to or samples to resume from.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator. If ``None``, defaults to ``calculator``'s :attr:`BaseCalculator.mpicomm`.

        engine : str, default='rqrs'
            QMC engine, to choose from ['sobol', 'halton', 'lhs', 'rqrs'].

        save_fn : str, Path, default=None
            If not ``None``, save samples to this location.

        seed : int, default=None
            Random seed.

        **kwargs : dict
            Optional engine-specific arguments.
        """
        super(QMCSampler, self).__init__(calculator, params=params, mpicomm=mpicomm, save_fn=save_fn)
        self.engine = get_qmc_engine(engine)(d=len(self.params), **kwargs)
        self.samples = None
        if self.mpicomm.rank == 0 and samples is not None:
            self.samples = samples if isinstance(samples, Samples) else Samples.load(samples)

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