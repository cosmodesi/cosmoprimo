import os
import re
import sys
import time
import logging
import traceback

import numpy as np


"""A few utilities."""


logger = logging.getLogger('Utils')


def is_sequence(item):
    """Whether input item is a tuple or list."""
    return isinstance(item, (list, tuple))


def exception_handler(exc_type, exc_value, exc_traceback):
    """Print exception with a logger."""
    # Do not print traceback if the exception has been handled and logged
    _logger_name = 'Exception'
    log = logging.getLogger(_logger_name)
    line = '=' * 100
    # log.critical(line[len(_logger_name) + 5:] + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    log.critical('\n' + line + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    if exc_type is KeyboardInterrupt:
        log.critical('Interrupted by the user.')
    else:
        log.critical('An error occured.')


def mkdir(dirname):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname)  # MPI...
    except OSError:
        return


def setup_logging(level=logging.INFO, stream=sys.stdout, filename=None, filemode='w', **kwargs):
    """
    Set up logging.

    Parameters
    ----------
    level : string, int, default=logging.INFO
        Logging level.

    stream : _io.TextIOWrapper, default=sys.stdout
        Where to stream.

    filename : string, default=None
        If not ``None`` stream to file name.

    filemode : string, default='w'
        Mode to open file, only used if filename is not ``None``.

    kwargs : dict
        Other arguments for :func:`logging.basicConfig`.
    """
    # Cannot provide stream and filename kwargs at the same time to logging.basicConfig, so handle different cases
    # Thanks to https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    if isinstance(level, str):
        level = {'info': logging.INFO, 'debug': logging.DEBUG, 'warning': logging.WARNING}[level.lower()]
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    t0 = time.time()

    class MyFormatter(logging.Formatter):

        def format(self, record):
            self._style._fmt = '[%09.2f] ' % (time.time() - t0) + ' %(asctime)s %(name)-28s %(levelname)-8s %(message)s'
            return super(MyFormatter, self).format(record)

    fmt = MyFormatter(datefmt='%m-%d %H:%M ')
    if filename is not None:
        mkdir(os.path.dirname(filename))
        handler = logging.FileHandler(filename, mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)
    logging.basicConfig(level=level, handlers=[handler], **kwargs)
    sys.excepthook = exception_handler


class BaseMetaClass(type):

    """Metaclass to add logging attributes to :class:`BaseClass` derived classes."""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        cls.set_logger()
        return cls

    def set_logger(cls):
        """
        Add attributes for logging:

        - logger
        - methods log_debug, log_info, log_warning, log_error, log_critical
        """
        cls.logger = logging.getLogger(cls.__name__)

        def make_logger(level):

            @classmethod
            def logger(cls, *args, **kwargs):
                return getattr(cls.logger, level)(*args, **kwargs)

            return logger

        for level in ['debug', 'info', 'warning', 'error', 'critical']:
            setattr(cls, 'log_{}'.format(level), make_logger(level))


class BaseClass(object, metaclass=BaseMetaClass):
    """
    Base class that implements :meth:`copy`.
    To be used throughout this package.
    """
    def __copy__(self, *args, **kwargs):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def copy(self, *args, **kwargs):
        return self.__copy__(*args, **kwargs)

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def save(self, filename):
        """Save to ``filename``."""
        self.log_info('Saving {}.'.format(filename))
        mkdir(os.path.dirname(filename))
        np.save(filename, self.__getstate__(), allow_pickle=True)

    @classmethod
    def load(cls, filename):
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.from_state(state)
        return new


def find_names(allnames, name):
    """
    Search parameter name ``name`` in list of names ``allnames``,
    matching template forms ``[::]``;
    return corresponding parameter names.
    Contrary to :func:`find_names_latex`, it does not handle latex strings,
    but can take a list of parameter names as ``name``
    (thus returning the concatenated list of matching names in ``allnames``).

    >>> find_names(['a_1', 'a_2', 'b_1', 'c_2'], ['a_[:]', 'b_[:]'])
    ['a_1', 'a_2', 'b_1']

    Parameters
    ----------
    allnames : list
        List of parameter names (strings).

    name : list, str
        List of parameter name(s) to match in ``allnames``.

    Returns
    -------
    toret : list
        List of parameter names (strings).
    """
    if not is_sequence(allnames):
        allnames = [allnames]

    if is_sequence(name):
        toret = []
        for nn in name:
            for n in find_names(allnames, nn):
                if n not in toret:
                    toret.append(n)
        return toret

    if isinstance(name, re.Pattern):
        pattern = name
    else:
        #name = fnmatch.translate(name)  # does weird things to -
        pattern = name.replace('*', '.*?') + '$'  # ? for non-greedy, $ to match end of string
    toret = []
    for paramname in allnames:
        match = re.match(pattern, paramname)
        if match:
            toret.append(paramname)
    return toret


def expand_dict(di, names):
    """
    Expand input dictionary, taking care of wildcards, e.g.:

    >>> expand_dict({'*': 2}, ['a', 'b'])
    {'a': 2, 'b': 2}
    >>> expand_dict({'a*': 2, 'b': 1}, ['a1', 'a2', 'b'])
    {'a1': 2, 'a2': 2, 'b': 1}
    """
    toret = dict.fromkeys(names)
    if is_sequence(di):
        di = dict(zip(names, di))
    if not hasattr(di, 'items'):
        di = {'*': di}
    for template, value in di.items():
        for tmpname in find_names(names, template):
            toret[tmpname] = value
    return toret


def deep_eq(obj1, obj2, equal_nan=True):
    """(Recursively) test equality between ``obj1`` and ``obj2``."""
    from cosmoprimo import jax
    if type(obj2) is type(obj1):
        if isinstance(obj1, dict):
            if obj2.keys() == obj1.keys():
                return all(deep_eq(obj1[name], obj2[name]) for name in obj1)
        elif isinstance(obj1, (tuple, list)):
            if len(obj2) == len(obj1):
                return all(deep_eq(o1, o2) for o1, o2 in zip(obj1, obj2))
        elif isinstance(obj1, (np.ndarray,) + jax.array_types):
            return np.array_equal(obj2, obj1, equal_nan=equal_nan)
        else:
            return obj2 == obj1
    return False


def subspace(X, precision=None, npcs=None, chi2min=None, fweights=None, aweights=None):
    r"""
    Project input values ``X`` to a subspace.
    See https://arxiv.org/pdf/2009.03311.pdf

    Parameters
    ----------
    X : array
        Array of shape (number of samples, ndim).

    precision : array, default=None
        Optionally, precision matrix, to normalize ``X``.

    npcs : int, default=None
        Optionally, number (<= ndim) of principal components to keep.
        If ``None``, number of components to be kept is fixed by ``chi2min``.

    chi2min : int, default=None
        In case ``npcs`` is provided, threshold for the maximum difference in :math:`\chi^{2}`
        w.r.t. keeping all components. If ``None``, all components are kept.

    fweights : array, default=None
        Optionally, integer frequency weights, of shape (number of samples,).

    aweights : array, default=None
        Optionally, observation weights.

    Returns
    -------
    eigenvectors : array of shape (ndim, npcs)
        Eigenvectors.
    """
    X = np.asarray(X)
    X = X.reshape(X.shape[0], -1)
    if precision is None:
        L = np.array(1.)
    else:
        L = np.linalg.cholesky(precision)
    X = X.dot(L)
    cov = np.cov(X, rowvar=False, ddof=0, fweights=fweights, aweights=aweights)
    size = cov.shape[0]
    if npcs is not None:
        if npcs > size:
            raise ValueError('Number of requested components is {0:d}, but dimension is {1:d} < {0:d}.'.format(npcs, size))
        import scipy as sp
        eigenvalues, eigenvectors = sp.linalg.eigh(cov, subset_by_index=[size - 1 - npcs, size - 1])
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if npcs is None:
        if chi2min is None:
            npcs = size
        else:
            npcs = size - np.sum(np.cumsum(eigenvalues) < chi2min)
        eigenvectors = eigenvectors[..., -npcs:]
    return L.dot(eigenvectors)


import ast


def evaluate(value, type=None, locals=None, verbose=True):
    """
    Evaluate several lines of input, returning the result of the last line.

    Reference
    ---------
    https://stackoverflow.com/questions/12698028/why-is-pythons-eval-rejecting-this-multiline-string-and-how-can-i-fix-it

    Parameters
    ----------
    value : str, any type
        If value is string, call ``eval``, with input ``locals`` (dictionary of local objects).
        "np", "sp", "jnp", "jsp" are recognized as numpy, scipy, jax.numpy, jax.scipy (if jax is installed).

    type : type, default=None
        If not ``None``, cast output ``value`` with ``type``.

    locals : dict, default=None
        Dictionary of local objects to use when calling ``eval``.

    Returns
    -------
    value : evaluated value.

    """
    import numpy as np
    import scipy as sp
    if isinstance(value, str):
        from cosmoprimo.jax import numpy as jnp
        from cosmoprimo.jax import scipy as jsp
        locals = dict(locals or {})
        globals = locals | {'np': np, 'sp': sp, 'jnp': jnp, 'jsp': jsp}  # FIXME: hack for nested loops
        tree = ast.parse(value)
        eval_expr = ast.Expression(tree.body[-1].value)
        exec_expr = ast.Module(tree.body[:-1], type_ignores=[])
        try:
            exec(compile(exec_expr, 'file', 'exec'), globals, locals)
            value = eval(compile(eval_expr, 'file', 'eval'), globals, locals)
        except Exception as exc:
            if verbose:
                raise Exception('unable to evaluate {} with locals = {} and globals = {}'.format(value, locals, globals)) from exc
            raise exc
    if type is not None:
        value = type(value)
    return value


def download(url, target, authorization=None, size=None):
    """
    Download file from input ``url``.

    Parameters
    ----------
    url : str, Path
        url to download file from.

    target : str, Path
        Path where to save the file, on disk.

    size : int, default=None
        Expected file size, in bytes, used to show progression bar.
        If not provided, taken from header (if the file is larger than a couple of GBs,
        it may be wrong due to integer overflow).
        If a sensible file size is obtained, a progression bar is printed.
    """
    # Adapted from https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    print('Downloading {} to {}.'.format(url, target))
    mkdir(os.path.dirname(target))
    import requests
    # See https://stackoverflow.com/questions/61991164/python-requests-missing-content-length-response
    headers = {}
    if authorization:
        headers.update({'Authorization': authorization})
    if size is None:
        size = requests.head(url, headers={**headers, 'Accept-Encoding': None}).headers.get('content-length')
    try:
        r = requests.get(url, headers=headers, allow_redirects=True, stream=True)
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        return False

    with open(target, 'wb') as file:
        if size is None or int(size) < 0:  # no content length header
            file.write(r.content)
        else:
            import shutil
            width = shutil.get_terminal_size((80, 20))[0] - 9  # pass fallback
            dl, size, current = 0, int(size), 0
            for data in r.iter_content(chunk_size=2048):
                dl += len(data)
                file.write(data)
                if size:
                    frac = min(dl / size, 1.)
                    done = int(width * frac)
                    if done > current:  # it seems, when content-length is not set iter_content does not care about chunk_size
                        print('\r[{}{}] [{:3.0%}]'.format('#' * done, ' ' * (width - done), frac), end='', flush=True)
                        current = done
            print('')
    return True