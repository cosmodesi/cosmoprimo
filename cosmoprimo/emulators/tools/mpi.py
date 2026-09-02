"""Collectives that do not go through pickle.

``mpicomm.allgather(obj)`` pickles: every array is serialised to bytes, sent, and deserialised.
For a training round that is fatal twice over --

- **the 2 GB limit.** mpi4py's lowercase calls cap a single message at 2 GB. One node of a
  monomials emulator is ~145 MB of tables, so a handful of nodes per rank per round overflows
  it, and the failure is an opaque pickle/overflow error rather than "your message is too big".
- **the memory.** Pickling holds the serialised copy, the received copy and the reconstructed
  arrays at once, so the peak is several times the data. That is what turns a 6 GB training set
  into an OOM at 16 ranks.

``gather`` below uses ``Gatherv``/``Allgatherv`` into a preallocated buffer instead, with a
custom contiguous datatype so the byte count per element -- not the element count -- is what MPI
sees, which lifts the 2 GB cap as well. Taken from mpytools (``mpytools.core.gather``), itself
from nbodykit's ``nbodykit.utils.GatherArray``; kept here so the emulator layer does not gain a
dependency on either.
"""

import numpy as np


def gather(data, mpiroot=0, mpicomm=None):
    """Gather ``data`` from all ranks along its first axis.

    Parameters
    ----------
    data : array_like
        This rank's share. ``shape[1:]`` and ``dtype`` must agree across ranks; the leading
        length may differ (including zero).
    mpiroot : int, Ellipsis or None, default=0
        Rank to gather to. ``None`` (or ``Ellipsis``) gathers to every rank -- the allgather.
    mpicomm : MPI communicator

    Returns
    -------
    array or None
        The concatenation in rank order on ``mpiroot`` (on every rank when it is ``None``),
        and ``None`` elsewhere.
    """
    from mpi4py import MPI

    if mpiroot is None:
        mpiroot = Ellipsis

    if all(mpicomm.allgather(np.isscalar(data))):
        if mpiroot is Ellipsis:
            return np.array(mpicomm.allgather(data))
        gathered = mpicomm.gather(data, root=mpiroot)
        if mpicomm.rank == mpiroot:
            return np.array(gathered)
        return None

    data = np.ascontiguousarray(data)   # Gatherv reads the buffer directly, so C order is required
    local_length = data.shape[0]

    shapes = mpicomm.allgather(data.shape)
    dtypes = mpicomm.allgather(data.dtype)

    if dtypes[0].char == 'V':
        names = set(dtypes[0].names)
        if any(set(dtype.names) != names for dtype in dtypes[1:]):
            raise ValueError('mismatch between data type fields in structured data')
        if any(dtypes[0][name] == 'O' for name in dtypes[0].names):
            raise ValueError('object data types ("O") not allowed in structured data in gather')
        newshape = list(data.shape)
        newshape[0] = mpicomm.allreduce(local_length)
        recvbuffer = (np.empty(newshape, dtype=dtypes[0], order='C')
                      if mpiroot is Ellipsis or mpicomm.rank == mpiroot else None)
        for name in dtypes[0].names:
            gathered = gather(data[name], mpiroot=mpiroot, mpicomm=mpicomm)
            if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
                recvbuffer[name] = gathered
        return recvbuffer

    if dtypes[0] == 'O':
        raise ValueError('object data types ("O") not allowed in gather')

    if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
        bad_shape = any(shape[1:] != shapes[0][1:] for shape in shapes[1:])
        bad_dtype = any(dtype != dtypes[0] for dtype in dtypes[1:])
    else:
        bad_shape, bad_dtype = None, None
    if mpiroot is not Ellipsis:
        bad_shape, bad_dtype = mpicomm.bcast((bad_shape, bad_dtype), root=mpiroot)
    if bad_shape:
        raise ValueError(f'mismatch between shape[1:] across ranks in gather: {shapes}')
    if bad_dtype:
        raise ValueError(f'mismatch between dtypes across ranks in gather: {dtypes}')

    newshape = list(data.shape)
    newshape[0] = mpicomm.allreduce(local_length)
    recvbuffer = (np.empty(newshape, dtype=data.dtype, order='C')
                  if mpiroot is Ellipsis or mpicomm.rank == mpiroot else None)

    counts = np.array(mpicomm.allgather(local_length), order='C')
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # One MPI element per row, sized in bytes: the element count then stays small whatever the
    # row size, which is what lifts the 2 GB cap.
    duplicity = np.prod(data.shape[1:], dtype='intp')
    datatype = MPI.BYTE.Create_contiguous(duplicity * data.dtype.itemsize)
    datatype.Commit()
    if mpiroot is Ellipsis:
        mpicomm.Allgatherv([data, datatype], [recvbuffer, (counts, offsets), datatype])
    else:
        mpicomm.Gatherv([data, datatype], [recvbuffer, (counts, offsets), datatype], root=mpiroot)
    datatype.Free()

    return recvbuffer
