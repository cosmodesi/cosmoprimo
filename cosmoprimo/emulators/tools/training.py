"""Evaluating the node set: resumable by default, chunked by default.

Training an emulator on an expensive calculator is a campaign, not a function call. A CMB node
set at CLASS precision is hours of compute; a full-shape one is worse. Three properties follow,
and all three were learned by losing work without them:

- **Resumable.** A run that cannot resume loses everything to one kill, one oom, one timeout.
  Checkpoint incrementally and skip what is done.
- **Chunked.** Long monolithic jobs monopolise a machine and remove the chance to intervene.
  Training stops cleanly at a time budget and reports how far it got.
- **Progress guaranteed.** At least one node per run, always: a budget shorter than a single
  evaluation would otherwise make every run a no-op and resumption would never terminate.

The node sets are nested, so raising the budget reuses every evaluation already made: refining an
emulator costs only the new nodes.
"""

import os
import logging
import time

import numpy as np


def _seconds(value):
    """Accept 1800, '30min', '2h', '45s'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    for suffix, factor in (('min', 60.), ('h', 3600.), ('s', 1.)):
        if text.endswith(suffix):
            return float(text[:-len(suffix)]) * factor
    return float(text)


class NodeEvaluationError(Exception):
    """A node the calculator could not evaluate.

    Never swallowed: a sparse-grid fit needs every node of its combination, so a missing one is
    fatal and must be reported where it happens, not discovered later as a confusing KeyError.
    """


class TrainingSet(object):
    """The training set: a target evaluated on a node set, resumably.

    It builds the data an engine is then fitted to, and stops there -- the fitting is
    :meth:`~.engines.BaseEngine.fit`, and choosing the nodes is :meth:`~.engines.BaseEngine.nodes`.
    What is expensive is neither of those; it is this.

    Parameters
    ----------
    target : callable
        ``target(**params) -> dict`` of named arrays.
    nodes : array
        ``(n_nodes, n_params)`` in training coordinates.
    params : list
        Names of the training coordinates, ordered as the columns of ``nodes``.
    batch_size : int, default=None
        ``None`` calls ``target`` with scalar parameters, one node at a time. An integer calls it
        with a dict of arrays of at most that length, expecting arrays back with a leading node
        axis -- worth it whenever the calculator can batch.

        Every batch has exactly this many nodes. The last one is padded with a repeated node
        and the extra results dropped, so a jitted target sees one shape and is traced once: a
        ragged final batch would recompile the whole pipeline to evaluate a handful of nodes,
        which costs far more than re-evaluating a duplicate.

        A size rather than a flag, because "all of them at once" is rarely what a calculator
        wants: batching trades memory for calls, and the whole node set may not fit. It is also
        the chunking granularity -- ``chunk`` can only stop between batches, so a batch that is
        too large is a run that cannot be interrupted.
    mpicomm : default=None
        An MPI communicator; nodes are split across ranks and the results gathered. Only rank 0
        writes the checkpoint.
    fixed : dict, default=None
        Parameters held at a reference value during sampling -- in particular those a scaling
        removes from the grid, which the calculator still needs a value for.
    checkpoint : str, default=None
        ``.npz`` path; resumption reads it and skips finished nodes.
    chunk : str, float, default=None
        Wall-clock budget for one run, e.g. ``'30min'``. The training stops cleanly and reports
        ``partial``; rerun to continue.
    save_every : int, default=50
        Checkpoint cadence, in nodes.
    """
    def __init__(self, target, nodes, params, fixed=None, checkpoint=None, chunk=None,
                 save_every=50, batch_size=None, mpicomm=None, drop_non_finite=False):
        self.target, self.params = target, list(params)
        self.nodes = np.atleast_2d(np.asarray(nodes, dtype='f8'))
        if self.nodes.shape[1] != len(self.params):
            raise ValueError(f'nodes have {self.nodes.shape[1]} columns for '
                             f'{len(self.params)} parameters')
        self.batch_size = None if batch_size is None else int(batch_size)
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError(f'batch_size must be at least 1, got {batch_size}')
        self.mpicomm = mpicomm
        # Parameters held at a reference value while sampling. A parameter removed by a scaling
        # still has to be given a value: the calculator cannot be evaluated without it, and the
        # scaling then divides that value out. Any value works -- that is what "exact" means.
        self.fixed = dict(fixed or {})
        self.checkpoint, self.chunk = checkpoint, _seconds(chunk)
        self.save_every = int(save_every)
        self.keys, self.values = [], {}
        # Only a regression engine may set this; see _call. Counted rather than silent, because
        # a box that loses many nodes is a box in the wrong place, whatever the engine.
        self.drop_non_finite = bool(drop_non_finite)
        self.n_non_finite = 0
        #: shapes of the first successful call, so a later failure can be turned into NaNs
        self._output_template = {}

    # ── state ──────────────────────────────────────────────────────────────────
    def _load(self):
        if not (self.checkpoint and os.path.exists(self.checkpoint)):
            return set()
        stored = dict(np.load(self.checkpoint, allow_pickle=True))
        self.keys = [tuple(row) for row in stored['nodes']]
        self.values = {name: list(stored[name]) for name in stored if name != 'nodes'}
        return {tuple(np.round(row, 12)) for row in stored['nodes']}

    def _save(self):
        if self.checkpoint:
            np.savez(self.checkpoint, nodes=np.array(self.keys),
                     **{name: np.array(value) for name, value in self.values.items()})

    @property
    def done(self):
        return len(self.keys)

    @property
    def complete(self):
        return self.done >= len(self.nodes)

    # ── run ────────────────────────────────────────────────────────────────────
    logger = logging.getLogger('TrainingSet')

    def run(self):
        """Evaluate what remains, within the chunk budget. Returns True when complete.

        ``chunk`` can only stop between batches, so with a large ``batch_size`` the time budget
        is checked correspondingly rarely.
        """
        finished = self._load()
        todo = [row for row in self.nodes if tuple(np.round(row, 12)) not in finished]
        rank = self.mpicomm.rank if self.mpicomm is not None else 0
        started, index = time.time(), 0

        while index < len(todo):
            # always evaluate at least one node per run: a budget shorter than a single
            # evaluation would otherwise make every run a no-op, and resumption would never
            # terminate. Guaranteed forward progress is the point of chunking.
            if index and self.chunk is not None and time.time() - started > self.chunk:
                if rank == 0:
                    self.logger.info(f'time budget reached at {self.done}/{len(self.nodes)} '
                                     f'nodes; rerun to continue')
                break
            # One row per round leaves every rank but one idle under MPI: the rank-split
            # happens inside _evaluate, so a round must carry at least one row per rank.
            # batch_size (the array-call convention) is untouched -- with it None, each rank
            # still evaluates its rows one at a time.
            rows_per_round = (self.batch_size if self.batch_size is not None
                              else (self.mpicomm.size if self.mpicomm is not None else 1))
            batch = todo[index:index + rows_per_round]
            names, values = self._evaluate(batch)
            for name, value in zip(names, values):
                self.values.setdefault(name, []).extend(value)
            self.keys.extend(tuple(row) for row in batch)
            index += len(batch)
            if rank == 0 and self.done % self.save_every < len(batch):
                self._save()
                self.logger.info(f'{self.done}/{len(self.nodes)} nodes')

        if rank == 0:
            self._save()
        if rank == 0:
            self.logger.info(f'training {"complete" if self.complete else "partial"} '
                             f'({self.done}/{len(self.nodes)} nodes)')
        return self.complete

    def _evaluate(self, batch):
        """Evaluate ``batch`` rows, split across MPI ranks when there is a communicator.

        One :func:`~.mpi.gather` per output name, and not one ``allgather`` of the whole
        ``(names, values)`` structure. The lowercase call pickles, which caps a message at 2 GB
        and holds the serialised copy, the received copy and the reconstructed arrays at once --
        a monomials node is ~145 MB of tables, so a round of a few nodes per rank overflows the
        cap, and the peak memory is what turned a 6 GB training set into an OOM at 16 ranks.
        ``Allgatherv`` writes straight into one preallocated buffer instead.
        """
        rows = list(batch)
        if self.mpicomm is None or self.mpicomm.size <= 1:
            return self._evaluate_local(rows)

        from .mpi import gather

        size = self.mpicomm.size
        mine = rows[self.mpicomm.rank::size]
        # A rank that raises here must not leave the collective below: the other ranks would
        # block in it forever, and the job holds its whole allocation until someone notices and
        # kills it by hand. Measured, repeatedly, on this training. So every rank reaches the
        # same exchange, failure or not, and they all raise together afterwards -- which also
        # gives every rank a traceback rather than hiding it on whichever one happened to fail.
        try:
            names, values = self._evaluate_local(mine)
            failure = None
        except Exception as exc:
            names, values, failure = [], [], f'{type(exc).__name__}: {exc}'
        failures = self.mpicomm.allgather(failure)
        if any(failure is not None for failure in failures):
            reported = [(rank, failure) for rank, failure in enumerate(failures)
                        if failure is not None]
            rank, first = reported[0]
            raise NodeEvaluationError(
                f'{len(reported)}/{size} ranks failed to evaluate their nodes; first was rank '
                f'{rank}: {first}')
        # A rank with no rows returns no names and no arrays, but Allgatherv still needs a buffer
        # of the right trailing shape and dtype from it -- so agree on the layout first. These
        # are a handful of tuples, so the pickling collective is the right tool here.
        layouts = self.mpicomm.allgather(
            [(name, np.asarray(value[0]).shape, np.asarray(value[0]).dtype.str)
             for name, value in zip(names, values) if len(value)])
        layout = next((entry for entry in layouts if entry), None)
        if layout is None:
            raise NodeEvaluationError('no rank produced any value for this batch')
        mine_values = dict(zip(names, values))

        merged, counts = {}, [len(rows[rank::size]) for rank in range(size)]
        offsets = np.cumsum([0] + counts[:-1])
        for name, shape, dtype in layout:
            local = mine_values.get(name)
            local = (np.asarray(local) if local else np.empty((0,) + tuple(shape), dtype=dtype))
            stacked = gather(local, mpiroot=None, mpicomm=self.mpicomm)
            # back into node order: rank r held rows r, r + size, r + 2*size, ..., and `stacked`
            # holds each rank's share contiguously
            merged[name] = [stacked[offsets[index % size] + index // size]
                            for index in range(len(rows))]
        return list(merged), [merged[name] for name in merged]

    def _evaluate_local(self, rows):
        """(names, values) for these rows, on this rank.

        Every value is turned into a numpy array at the node itself. That is the device-to-host move:
        the target is a jax pipeline, so what it returns lives wherever jax put it, and holding
        a node set of those keeps the whole training set resident on the accelerator. Host
        memory is the right home for it -- it is written to a checkpoint and fitted with numpy,
        never used on device again.
        """
        collected = {}
        if self.batch_size is None:
            for row in rows:
                params = {**self.fixed, **dict(zip(self.params, row))}
                for name, value in self._call(params).items():
                    collected.setdefault(name, []).append(np.asarray(value))
        else:
            # MPI has already split `rows` across ranks, so re-chunk here rather than trusting the
            # caller's slice: this rank's share is not the caller's batch
            for start in range(0, len(rows), self.batch_size):
                chunk = rows[start:start + self.batch_size]
                wanted = len(chunk)
                if wanted < self.batch_size:
                    # pad to a constant batch shape and drop the extras below: under jax a ragged
                    # final batch is a second shape, and retracing the whole pipeline costs far
                    # more than re-evaluating one duplicated node
                    chunk = list(chunk) + [chunk[-1]] * (self.batch_size - wanted)
                params = {**{name: np.array([value] * self.batch_size)
                             for name, value in self.fixed.items()},
                          **{name: np.array([row[index] for row in chunk])
                             for index, name in enumerate(self.params)}}
                for name, value in self._call(params).items():
                    value = np.asarray(value)
                    if len(value) != self.batch_size:
                        raise NodeEvaluationError(
                            f'the target returned {len(value)} values of {name!r} for a batch of '
                            f'{self.batch_size} nodes; with batch_size set it must return arrays '
                            f'with a leading node axis')
                    collected.setdefault(name, []).extend(list(value[:wanted]))
        return list(collected), [collected[name] for name in collected]

    def _call(self, params):
        """The target takes a dict of parameters, and returns what should be fitted -- any
        transform it wants to apply is its own business, done inside.

        Non-finite outputs are refused as loudly as exceptions: an interpolating engine mixes
        every node into every coefficient, so a single NaN node silently poisons the whole emulator --
        measured, an emulated posterior came back -inf at its own box centre, one full session
        after the node that caused it.

        With ``drop_non_finite`` the values are kept as they came instead, NaNs and all, and the
        node is counted in :attr:`n_non_finite`. That is only correct for a regression engine,
        where the row is simply left out of the least-squares fit; the caller decides, from
        ``engine.requires_all_nodes``. The node still enters the checkpoint, so resumption and
        the done/complete accounting are unaffected -- the filtering happens at fit time."""
        try:
            values = self.target(params)
        except Exception as exc:
            # A node can fail two ways: return non-finite values, or raise. Both are the same
            # event for a caller that can drop it -- a Boltzmann code refusing a cosmology
            # sometimes returns NaN and sometimes throws, and which one it does is not the
            # caller's business. Without a successful call there is no shape to return, so the
            # first success is remembered as a template.
            if self.drop_non_finite and not self._output_template and self.values:
                # No successful call on this rank yet -- but the checkpoint already holds
                # results, and their shapes are the same. Without this a rank whose first node
                # is the bad one has no template and raises, which is how a tolerated failure
                # still killed the run.
                self._output_template = {name: np.shape(np.asarray(stored[0]))
                                         for name, stored in self.values.items() if len(stored)}
            if not (self.drop_non_finite and self._output_template):
                raise NodeEvaluationError(f'node {params} failed: '
                                          f'{type(exc).__name__}: {exc}') from exc
            self.n_non_finite += 1
            return {name: np.full(shape, np.nan) for name, shape in self._output_template.items()}
        if self.drop_non_finite and not self._output_template:
            self._output_template = {name: np.shape(np.asarray(value))
                                     for name, value in values.items()}
        bad = {name: int((~np.isfinite(np.asarray(value))).sum()) for name, value in values.items()
               if not np.isfinite(np.asarray(value)).all()}
        if bad:
            if not self.drop_non_finite:
                raise NodeEvaluationError(f'node {params} returned non-finite values in '
                                          f'{bad} (output name -> count). One such node poisons '
                                          f'every coefficient of the fit; shrink the Space to '
                                          f'where the calculator is finite, or use a regression '
                                          f'engine, which can drop it.')
            self.n_non_finite += 1
        return values

    def inputs(self):
        """The node coordinates actually evaluated, ``(n_nodes, n_params)``.

        (Distinct from :attr:`nodes`, which is the set planned -- they differ mid-run.)
        """
        self._complete_or_raise()
        return np.array(self.keys)

    def outputs(self):
        """What the calculator returned, ``{name: (n_nodes, ...)}``."""
        self._complete_or_raise()
        return {name: np.array(value) for name, value in self.values.items()}

    def _complete_or_raise(self):
        if not self.complete:
            raise ValueError(f'training is incomplete ({self.done}/{len(self.nodes)}); a sparse '
                             f'grid needs every node of its combination')
