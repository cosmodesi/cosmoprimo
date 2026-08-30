"""Reading and writing an emulator's state.

HDF5 by default, because a trained emulator outlives the session that made it: an ``.h5`` file is
readable by anything, browsable with ``h5ls``, and does not execute code when opened. ``.npy``
still works, but it is a pickle -- it can only be read back by a compatible Python, and reading
one is running whatever it contains.

The state is ordinary nested Python -- dicts, tuples, strings, ``None``, arrays -- so the mapping
is explicit rather than clever: every value becomes a group carrying a ``type`` attribute, and the
structure of the file mirrors the structure of the state. A dict becomes a group of named
subgroups, so ``h5ls -r`` shows the parameter names and output names directly.
"""

import numpy as np


HDF5 = ('.h5', '.hdf5')
PICKLE = ('.npy',)


def _write(group, value):
    if isinstance(value, dict):
        group.attrs['type'] = 'dict'
        for key, item in value.items():
            key = str(key)
            if '/' in key:
                raise ValueError(f'cannot write the key {key!r}: "/" is HDF5\'s group separator')
            _write(group.create_group(key), item)
    elif isinstance(value, (list, tuple)):
        # a group per element, not one array: the state holds heterogeneous tuples, such as
        # (engine state, output shape), which no single dataset could carry
        group.attrs['type'] = 'tuple' if isinstance(value, tuple) else 'list'
        group.attrs['length'] = len(value)
        for index, item in enumerate(value):
            _write(group.create_group(str(index)), item)
    elif value is None:
        group.attrs['type'] = 'none'
    elif isinstance(value, str):
        group.attrs['type'] = 'str'
        group.attrs['value'] = value
    elif isinstance(value, (bool, np.bool_)):
        group.attrs['type'] = 'bool'
        group.attrs['value'] = bool(value)
    else:
        array = np.asarray(value)
        if array.dtype == object:
            raise TypeError(f'cannot write {type(value).__name__} to HDF5: object arrays have no '
                            f'representation. Save to .npy if you really need a pickle.')
        # remember whether this was a python scalar, so `budget` comes back an int rather than a
        # 0-d array that later fails an `int()`-free comparison
        group.attrs['type'] = 'scalar' if np.ndim(value) == 0 else 'array'
        group.create_dataset('value', data=array)


def _read(group):
    kind = group.attrs['type']
    if kind == 'dict':
        return {key: _read(group[key]) for key in group}
    if kind in ('tuple', 'list'):
        values = [_read(group[str(index)]) for index in range(int(group.attrs['length']))]
        return tuple(values) if kind == 'tuple' else values
    if kind == 'none':
        return None
    if kind == 'str':
        return str(group.attrs['value'])
    if kind == 'bool':
        return bool(group.attrs['value'])
    array = group['value'][()]
    return array.item() if kind == 'scalar' else np.asarray(array)


def write_state(path, state):
    """Write ``state`` to ``path``; HDF5 unless the name says ``.npy``."""
    import os

    path = str(path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if path.endswith(PICKLE):
        np.save(path, state, allow_pickle=True)
        return path
    if not path.endswith(HDF5):
        path = path + HDF5[0]
    import h5py

    # Write-then-rename, so a failed write (an unserializable value found halfway through the
    # state) never leaves a truncated file where a cache lookup will trust it: measured, a
    # half-written emulator read back as KeyError deep inside _read, one session later.
    partial = path + '.partial'
    try:
        with h5py.File(partial, 'w') as file:
            _write(file.create_group('emulator'), state)
    except BaseException:
        import contextlib
        with contextlib.suppress(OSError):
            os.remove(partial)
        raise
    os.replace(partial, path)
    return path


def read_state(path):
    """Read back what :func:`write_state` wrote."""
    path = str(path)
    if path.endswith(PICKLE):
        return np.load(path, allow_pickle=True)[()]
    import h5py

    with h5py.File(path, 'r') as file:
        return _read(file['emulator'])
