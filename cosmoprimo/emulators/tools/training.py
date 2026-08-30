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


class Training(object):
    """Evaluate a target on a node set, resumably.

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
                 save_every=50, batch_size=None, mpicomm=None):
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
    logger = logging.getLogger('Training')

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
            batch = ([todo[index]] if self.batch_size is None
                     else todo[index:index + self.batch_size])
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
        """Evaluate ``batch`` rows, split across MPI ranks when there is a communicator."""
        rows = list(batch)
        if self.mpicomm is not None and self.mpicomm.size > 1:
            mine = rows[self.mpicomm.rank::self.mpicomm.size]
            gathered = self.mpicomm.allgather(self._evaluate_local(mine))
            names = gathered[0][0]
            merged = {name: [] for name in names}
            # interleave back into node order: rank r held rows r, r+size, r+2*size, ...
            per_rank = [dict(zip(chunk_names, chunk_values))
                        for chunk_names, chunk_values in gathered]
            for index in range(len(rows)):
                source = per_rank[index % self.mpicomm.size]
                position = index // self.mpicomm.size
                for name in names:
                    merged[name].append(source[name][position])
            return list(merged), [merged[name] for name in merged]
        return self._evaluate_local(rows)

    def _evaluate_local(self, rows):
        """(names, values) for these rows, on this rank."""
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

        Non-finite outputs are refused as loudly as exceptions: every engine mixes every node
        into every coefficient, so ONE NaN node silently poisons the whole emulator -- measured,
        an emulated posterior came back -inf at its own box centre, one full session after the
        node that caused it."""
        try:
            values = self.target(params)
        except Exception as exc:
            raise NodeEvaluationError(f'node {params} failed: '
                                      f'{type(exc).__name__}: {exc}') from exc
        bad = {name: int((~np.isfinite(np.asarray(value))).sum()) for name, value in values.items()
               if not np.isfinite(np.asarray(value)).all()}
        if bad:
            raise NodeEvaluationError(f'node {params} returned non-finite values in '
                                      f'{bad} (output name -> count). One such node poisons '
                                      f'every coefficient of the fit; shrink the Space to where '
                                      f'the calculator is finite.')
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
