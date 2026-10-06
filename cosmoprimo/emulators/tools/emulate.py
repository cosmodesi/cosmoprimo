"""Emulating something, in four steps you can see.

    emu = Emulator(target, Space(samples=chain))   # what, and where it must be accurate
    emu.params                                     # what the interpolant will expand
    emu.train(budget=4)                            # sample and fit
    emu.predict(h=0.68, omega_cdm=0.12)            # use it

The target is a plain callable, ``target(params) -> dict`` of named arrays. Nothing else is
assumed of it: no methods to implement, no protocol to satisfy, no base class. A function, a
bound method, a lambda around a Boltzmann code -- anything.

Everything a particular calculator knows about itself is a subclass of this class. :class:`Emulator`
is a template: three hooks, each of which does nothing by default, each overridable on its own::

    class HarmonicEmulator(Emulator):

        def select_params(self, names):
            # which of the space's parameters the interpolant expands. What you leave out is
            # handled exactly by the pair below, and costs no nodes at all.
            return [name for name in names if name != 'A_s']

        def transform(self, values, params):
            # applied to the target's output before fitting. Divide out what you know:
            # the flatter the interpolant's job, the fewer nodes it takes.
            return {name: value / params['A_s'] for name, value in values.items()}

        def inverse_transform(self, values, params):
            # ... and put back at prediction. Must invert `transform` exactly.
            return {name: value * params['A_s'] for name, value in values.items()}

A subclass may also give back the thing the user started with, rather than a dict of arrays --
``to_cosmology`` in cosmoprimo, ``to_calculator`` in desilike. That is deliberately not part of
this API: what a trained emulator turns back into is a statement about the calculator's own
world, and this layer has no notion of one. :meth:`predict` is what it offers.

``transform`` is applied after training, to the collected values, never before they are stored:
the checkpoint holds physical outputs, so changing what you divide out costs a refit, not another
run of the Boltzmann code.

A fourth hook, :meth:`Emulator.select_node_params`, says which of the exact parameters the NODES
should vary anyway. By default an exact parameter is held at the centre of the box while the nodes
are evaluated; a subclass may instead have it sampled alongside the expanded ones, so that a node
set (and its checkpoint) drawn before the parameter was made exact is reused as it stands -- the
transform divides the parameter out of every node, and the fit sees the same nodes projected onto
fewer parameters.
"""

import logging

import numpy as np

from cosmoprimo.jax import numpy_jax, use_jax

from .training import TrainingSet, NodeEvaluationError
from .space import Space
from .validation import validate as _validate
from .constraints import constraint_violations, clip_to_constraints


def _relative_rms(prediction, reference):
    """Worst over outputs of ``rms(difference) / rms(reference)``.

    A ratio of norms, not a pointwise ratio: TE, tp and ep cross zero, and dividing by them there
    manufactures infinities that have nothing to do with emulator accuracy.
    """
    worst = 0.
    for name, truth in reference.items():
        truth = np.asarray(truth)
        norm = np.sqrt(np.mean(truth**2))
        if not norm:
            continue
        error = np.sqrt(np.mean((np.asarray(prediction[name]) - truth)**2))
        worst = max(worst, error / norm)
    return worst


class CoverageError(Exception):
    """A prediction was requested outside the trained box.

    Raised, not clipped: measured, one clipped draw gave dchi2 2e4 where every draw inside was
    below 0.2. Silent clipping turns an obvious failure into a plausible wrong answer.
    """


_VIOLATION = ('raise', 'warn', 'nan', 'clip', 'ignore')


def _check_violation(value):
    if value not in _VIOLATION:
        raise ValueError(f'violation must be one of {_VIOLATION}, not {value!r}')
    return value


class NotTrained(Exception):
    """The emulator has not been trained yet."""


class StateVersionError(Exception):
    """A saved emulator was written by an incompatible version of this package."""


class Emulator(object):
    """Emulate ``target`` over ``space``. Subclass to teach it what your calculator knows.

    Parameters
    ----------
    target : callable
        ``target(params) -> dict`` of named arrays. Nothing else is assumed.
    space : Space
        Where accuracy is required, in the user's own parameters.
    engine : str, default='chebyshev'
        Which engine fits the nodes: ``'chebyshev'`` (sparse-grid interpolation),
        ``'taylor'`` (a local expansion, see :mod:`taylor`), ``'polynomial'`` (least-squares
        regression over a declared basis, see :mod:`polynomial` -- the one to reach for when part
        of the box is a region the calculator refuses, since it needs no complete node set) or
        ``'mlp'``.
    constraints : list, default=None
        Declarative constraints of the user's own parameters, on top of the built-in ones (the
        trained box and the node cloud, see :meth:`constraints`): :class:`~.constraints.LinearConstraint`
        instances, their text form (``'w0_fld + wa_fld < 0'``) or their saved state.  Saved with the emulator.
    violation : str, default='raise'
        What :meth:`predict` does where a constraint is violated: ``'raise'`` (eager; NaN when
        traced), ``'warn'`` (eager; NaN), ``'nan'``, ``'clip'`` (predict at the point clipped into
        the constraints that clip -- the box and the node cloud -- and leave enforcement to the
        caller, see :meth:`violations`) or ``'ignore'``.
    options : dict
        Passed to the engine (``levels``, ``budget``, ...).
    """
    logger = logging.getLogger('Emulator')

    #: Layout of the saved state. Bump it whenever a change would make an old file read back
    #: wrong rather than fail loudly -- a renamed key, a changed convention, a different
    #: normalisation. A silently misread emulator is the worst outcome available here: it
    #: predicts confidently and is wrong everywhere.
    version = 1

    def __init__(self, target, space, engine='chebyshev', constraints=None, violation='raise', **options):
        from .constraints import as_constraint
        if not callable(target):
            raise TypeError(f'target must be callable, `target(params) -> dict`; got '
                            f'{type(target).__name__}')
        self.target, self.space = target, space
        self.engine_name, self.options = engine, dict(options)
        self.violation = _check_violation(violation)
        self._constraints = [as_constraint(constraint) for constraint in (constraints or [])]
        self._engines = {}
        self.training = self.training_space()
        names = list(self.training.params)
        expanded = list(self.select_params(names))
        unknown = [name for name in expanded if name not in names]
        if unknown:
            raise ValueError(f'{type(self).__name__}.select_params returned {unknown}, not in the '
                             f'training space ({names})')
        if not expanded:
            raise ValueError(f'{type(self).__name__}.select_params left nothing to expand')
        self.params = expanded
        # Which parameters the NODES vary. The expanded ones, unless a subclass asks for some of
        # the exact parameters to be sampled as well (`select_node_params`): those are then node
        # coordinates the fit does not see -- `transform` has taken them out.
        sampled = list(self.select_node_params(names))
        unknown = [name for name in sampled if name not in names]
        if unknown:
            raise ValueError(f'{type(self).__name__}.select_node_params returned {unknown}, not in '
                             f'the training space ({names})')
        self.node_params = self._node_params(names, expanded, sampled)

    # ── the hooks: override any, ignore the rest ───────────────────────────────
    def output_coordinates(self, name):
        """The coordinates output *name* is fitted in, or ``None`` for the emulator's own.

        ``(space, params, to_training)``: the :class:`Space` the engine takes its geometry from,
        the parameters it expands, and a callable turning a point in the emulator's own training
        parameters into that output's. The node set is shared whatever this returns -- what
        changes is the coordinates each fit sees.

        Override it when one output responds simply to a reparametrisation the others do not
        want: :math:`C_\ell` are nearly a translation along :math:`\ell` in :math:`h` and nearly
        stationary in :math:`\theta_\mathrm{MC}`, while a power spectrum wants :math:`h` itself,
        so a cosmology emulating both fits them in different variables over the same nodes.
        """
        return None

    def training_space(self):
        """The :class:`Space` the interpolant actually works in. The user's own, by default.

        Override it when the parameters accuracy is required over are not the ones worth
        expanding in. For the CMB they are not: a chain runs in ``Omega_m``, while the spectra
        respond simply to the physical density ``omega_cdm``, and that map mixes in ``h`` -- so it
        is not a rescaling, and whitening, being linear, cannot absorb it.

        Whatever this returns must be paired with :meth:`to_training`, and :meth:`Space.map` is
        the way to build it, so the two describe the same region.
        """
        return self.space

    def to_training(self, params):
        """User parameters -> training parameters. Identity by default.

        Applied at every prediction, so it should be cheap; a cosmology basis change costs about
        0.6 ms.
        """
        return params

    def from_training(self, params):
        """Training parameters -> user parameters: the inverse of :meth:`to_training`.

        Needed because the nodes are laid out in the training basis but the calculator is called
        with them, and it need not accept that basis. Where the change of variables is between two
        quantities the calculator already understands (``Omega_m`` and ``omega_cdm``, say) the
        identity here is right; where the expansion variable is not an input at all -- ``w0 + wa``
        is not -- this is what keeps that fact inside the emulator instead of leaking a special
        case into the calculator.
        """
        return params

    def select_params(self, names):
        """Which of the space's parameters the interpolant expands. All of them, by default.

        ``names`` are the training parameters (see :meth:`training_space`), which are the user's
        own unless a subclass says otherwise.

        What you leave out must be handled exactly by :meth:`transform` and
        :meth:`inverse_transform` -- it then costs no nodes at all, and is unbounded, since its
        dependence is not interpolated.
        """
        return list(names)

    def select_node_params(self, names):
        """Which of the EXACT parameters the nodes should vary anyway. None, by default.

        A parameter left out of :meth:`select_params` is normally held at the centre of the box
        while the nodes are evaluated: its dependence is divided out by :meth:`transform`, so
        sampling it would only spend nodes on a direction the fit never sees. Returning some of
        those parameters here samples them at the nodes regardless. The transform still takes them
        out, node by node, and the fit is made over the node set PROJECTED onto the expanded
        parameters -- as many nodes as before, over fewer inputs.

        Two reasons to want that. A node set drawn while the parameter was still expanded -- and
        the checkpoint holding its evaluations -- is then reused as it stands when the parameter is
        later made exact, instead of being evaluated again with the parameter pinned: the
        checkpoint matches nodes by their coordinates, and the coordinates are unchanged. And it is
        a check of the transform itself: a dependence the transform failed to remove is, in the
        projected set, a scatter no fit over the remaining parameters can absorb, and shows in the
        fit residual rather than silently in every prediction.

        ``names`` are the training parameters, as :meth:`select_params` receives them. The names
        returned must be among them and not among the expanded ones (an expanded parameter is
        sampled anyway).
        """
        return []

    @staticmethod
    def _node_params(names, expanded, sampled):
        """The node coordinates, in a FIXED order: the training order whenever an exact parameter
        is sampled, the expanded order otherwise.

        The order is the column order of the node array, and so of every checkpoint, so it must not
        depend on how a subclass happens to list its selections: a node set drawn when every
        parameter was expanded (``expanded == names``) has to come out identical once one of them
        becomes exact-but-sampled, or the whole point of sampling it is lost. Without a sampled
        parameter the expanded order is kept as it was, so files and checkpoints written before
        this hook existed are unaffected.
        """
        if not sampled:
            return list(expanded)
        wanted = set(expanded) | set(sampled)
        return [name for name in names if name in wanted]

    def transform(self, values, params):
        """Applied to the target's output before fitting. Identity by default."""
        return values

    def inverse_transform(self, values, params):
        """Undone at prediction; must invert :meth:`transform` exactly. Identity by default."""
        return values

    # ── machinery that consumes the hooks (not itself one) ─────────────────
    def _evaluate_target(self, params):
        """The calculator, called in its own parameters whatever basis the nodes are in."""
        return self.target(self.from_training(params))

    # ── state ─────────────────────────────────────────────────────────────────
    @property
    def exact_params(self):
        """Handled exactly by :meth:`transform` / :meth:`inverse_transform`, at zero grid cost --
        and unbounded: their dependence is not interpolated, so they may be varied outside the
        trained box."""
        return [name for name in self.training.params if name not in self.params]

    @property
    def trained(self):
        """One fitted engine per output, so having any is what being trained means -- no
        separate flag to fall out of step with them."""
        return bool(self._engines)

    # ── train ─────────────────────────────────────────────────────────────────
    def nodes(self, budget=None, **kwargs):
        """The parameter values the calculator will be evaluated at.

        Exposed so a training can be sized -- or handed to an external batch system -- before
        paying for it. The grid's levels are nested, so raising ``budget`` later reuses every
        evaluation already made; the Taylor engine's stencils are not, so there the order is
        worth choosing before paying (see :class:`~.taylor.TaylorEngine`).
        """
        return self._engine(budget=budget, params=self.node_params, **kwargs).nodes()

    def _engine(self, budget=None, space=None, params=None, geometry=None, **kwargs):
        from .engines import ChebyshevEngine
        from .mlp import MLPEngine
        from .polynomial import PolynomialEngine
        from .taylor import TaylorEngine

        classes = {cls.name: cls
                   for cls in (ChebyshevEngine, TaylorEngine, PolynomialEngine, MLPEngine)}
        # `space`/`params` for an output fitted in coordinates of its own
        # (:meth:`output_coordinates`); the emulator's own otherwise. `params` is also the node
        # parameters for the engine that DRAWS the nodes, which differ from the expanded ones when
        # an exact parameter is sampled at the nodes (see `select_node_params`).
        space = self.training if space is None else space
        params = self.params if params is None else list(params)
        subspace = space.marginal(params) if len(params) < len(space.params) else space
        # `geometry` overrides the space's own box: the engine that draws the extra nodes of an
        # augmented training (see :meth:`_augmented_nodes`) is the same class with the same
        # options over a narrower box, and nothing else about it differs.
        if geometry is None:
            geometry = subspace.geometry()
        options = {**self.options, **kwargs}
        # `budget` may arrive twice -- once at construction (kept in `options`) and once from
        # `train` -- and passing both to the engine is a TypeError. An explicit one wins; the
        # constructor's is the fallback.
        if budget is None:
            budget = options.pop('budget', None)
        else:
            options.pop('budget', None)
        if self.engine_name not in classes:
            raise ValueError(f'unknown engine {self.engine_name!r}; available '
                             f'{sorted(classes)}')
        # the box, whitening included, comes from the space itself (see :meth:`Space.geometry`):
        # what stays here is only what the space has no say in -- which engine, and how big.
        cls = classes[self.engine_name]
        # The chain, for an engine that places its nodes rather than deriving them from the box.
        # Not part of `geometry()`, which describes the region and deliberately hands over plain
        # arrays -- mean, rotation, scale -- and not the samples behind them; this is the one
        # engine that wants the samples themselves, so it asks for them by name.
        if cls.wants_samples and 'samples' not in options and subspace.samples is not None:
            options = {**options, 'samples': subspace.samples}
        return cls(**geometry, budget=budget, **options)

    def _augmented_nodes(self, augment, budget=None, **kwargs):
        """The extra nodes of an augmented training: ``(n, nparams)`` in physical parameters.

        ``augment`` is one specification or a list of them, each ``{'bounds': {name: (low,
        high)}, 'nsamples': n, ...}``: the engine's own node draw (Sobol' through ``valid``, for
        the scattered engines) over the training box cut down to ``bounds`` on the named axes,
        the others left at their full range. Any other key overrides an engine option for that
        draw alone -- ``seed`` most usefully; it defaults to the engine's seed plus the
        specification's index so a sub-box is not the base draw rescaled.

        Bounds are given in the user's parameters, like a :class:`Space`'s; they are mapped
        through the axis transforms into the engine's expansion variable here, as the space
        does for its own bounds. Only for a box-shaped space: a whitened engine draws on the
        posterior's principal axes and a cut on one physical axis is not a face of that box.
        """
        specs = [augment] if isinstance(augment, dict) else list(augment or [])
        if not specs:
            return np.empty((0, len(self.node_params)))
        from .space import _ranges
        # over the node parameters, like the base draw: the extra nodes join the same array
        subspace = self.training.marginal(self.node_params) \
            if len(self.node_params) < len(self.training.params) else self.training
        geometry = subspace.geometry()
        if 'covariance' in geometry:
            raise ValueError('augment needs a box-shaped Space (bounds only); this one is '
                             'whitened, so a cut on a physical axis would not be a face of its box')
        extra = []
        for index, spec in enumerate(specs):
            spec = dict(spec)
            bounds = dict(spec.pop('bounds', None) or {})
            unknown = [name for name in bounds if name not in geometry['params']]
            if unknown:
                raise ValueError(f'augment bounds name {unknown}, not among the emulated '
                                 f'parameters {geometry["params"]}')
            narrowed = _ranges(bounds, geometry['transform'])
            limits = dict(geometry['limits'])
            for name, (low, high) in narrowed.items():
                low, high = max(limits[name][0], low), min(limits[name][1], high)
                if not low < high:
                    raise ValueError(f'augment bounds for {name!r} ({bounds[name]}) do not overlap '
                                     f'the training box {limits[name]}')
                limits[name] = (low, high)
            options = {**self.options, **kwargs, **spec}
            options['seed'] = int(spec.get('seed', int(options.get('seed', 42)) + 1 + index))
            engine = self._engine(budget=budget, geometry={**geometry, 'limits': limits,
                                                           'bounds': {**geometry['bounds'], **narrowed}},
                                  params=self.node_params,
                                  **{name: value for name, value in options.items() if name != 'budget'})
            nodes = np.atleast_2d(np.asarray(engine.nodes(), dtype='f8'))
            self.logger.info(f'augmenting with {len(nodes)} nodes over '
                             + ', '.join(f'{name} [{low:.5g}, {high:.5g}]' for name, (low, high) in bounds.items()))
            extra.append(nodes)
        return np.concatenate(extra, axis=0)

    def train(self, engine=None, budget=None, checkpoint=None, chunk=None, batch_size=None,
              mpicomm=None, per_output=None, max_non_finite=0.05, method='auto',
              basis_budget=None, drop_non_finite=None, fit=True, rows_per_rank=None,
              outlier_factor=None, outlier_factor_low=None, outlier_on='transformed', augment=None,
              **kwargs):
        """Evaluate the calculator on the node set and fit.

        ``augment`` adds nodes where accuracy is wanted most: one or several
        ``{'bounds': {name: (low, high)}, 'nsamples': n}`` specifications, each a draw of the
        engine's own kind (Sobol' through ``valid``) over the training box narrowed on the named
        axes, appended to the base node set (see :meth:`_augmented_nodes`). A box-uniform draw
        spends its nodes in proportion to volume, and the region a chain settles in is a small
        fraction of it -- the near-GR corner of the EFT-of-dark-energy box held 28 of 65536 nodes
        within 0.05 of GR, and the error there was 10x the box median (2026-09-10). Extra nodes
        go through the same outlier cuts as the rest, which matters: 8% of the gated near-GR
        draw was still absurd. The checkpoint keys nodes by coordinates, so a checkpoint of the
        un-augmented training, copied under the augmented name, is resumed with only the extra
        nodes to evaluate.

        Resumable and chunked: pass ``checkpoint`` and ``chunk='30min'`` for anything expensive,
        then rerun until it reports complete. A kill then costs one node, not the training.
        ``batch_size`` calls the target with dicts of arrays of that length instead of one node
        at a time; ``mpicomm`` splits the nodes across ranks.

        ``rows_per_rank`` (MPI only) is how many nodes each rank evaluates between two
        exchanges; see :class:`~.training.TrainingSet`.

        ``outlier_factor`` (regression engines only) drops, before the fit, every node at which
        some output component exceeds that factor times the component's median |value| over
        the nodes. A stability gate says where a calculator *runs*, not where its answer is
        sane: over the EFT-of-dark-energy box, 8% of the gate-passing nodes returned one-loop
        tables 1e4 to 1e87 times their typical size (large c_M with w0 > -0.5), and a network
        fitted with them lost the sane 92% -- its asinh scale followed the maximum. Dropped
        nodes are logged; the fit then extrapolates there, finite but meaningless, which is
        what a chain that never visits those models can live with.

        ``outlier_factor_low`` is the mirror cut: a node is dropped when some sign-definite
        component (one that keeps the same sign over every node -- a component that crosses
        zero is legitimately tiny near the crossing) falls below the median divided by that
        factor. The same absurd region has a collapsed face: models whose growth is switched off
        return tables and growth scalars 1e-3 to 1e-5 times typical, and in the transformed
        space the network fits they dominate the mean-squared loss by orders of magnitude, so
        the fit of the sane 96% is spent on them (measured: the growth scalar's median error
        stayed at 0.6% whatever the schedule until they were removed).

        ``outlier_on`` says which values the two cuts look at: ``'transformed'`` (the default, what
        the engines are fitted to), ``'raw'`` (the calculator's outputs as the checkpoint holds
        them), or ``'both'`` (a node must pass both). They differ whenever :meth:`transform` changes
        the dynamic range: dividing the one-loop tables of the EFT-of-dark-energy emulator by the
        primordial amplitude squared brought 1446 nodes under the ``1e4 x median`` cut that the raw
        tables had failed, and those were the odd-growth models the cut existed for (sigma8 at fixed
        amplitude at half the median, f(k) shapes deviating twice as much); fitted with them, the
        growth-rate row and the sigma^2 scalars came out 2.5-3.5x worse in the likelihood (2026-09-11).
        ``'both'`` keeps the raw cut's node set and adds whatever the transformed outputs reject.

        ``fit=False`` stops once every node is evaluated and checkpointed, leaving the emulator
        untrained: the two stages want different machines (the node evaluations are Boltzmann
        and perturbation-theory calls, many CPU ranks; a network fit is one GPU), so a first job
        evaluates and a second, with the same ``checkpoint``, finds the set complete and fits.

        ``per_output`` overrides the engine options for named outputs, e.g.
        ``per_output={'pk': dict(budget=2)}``. A key matches an output name, or the part of it
        before the first dot -- so ``{'background': dict(budget=2)}`` reaches every output a
        cosmology composite's background section contributes. Only ever downward: every output is
        fitted from the same node set, so a lower budget uses a nested subset of it, while a
        higher one would need evaluations that were never made. Use it when one output is much
        smoother than the rest and does not deserve the same number of terms.
        """
        per_output = dict(per_output or {})
        if engine is not None:
            self.engine_name = engine
        # The engine that draws the nodes: over the node parameters, which include any exact
        # parameter a subclass asked to have sampled (`select_node_params`). The engines that are
        # FITTED, below, are over the expanded parameters alone.
        built = self._engine(budget=budget, params=self.node_params, **kwargs)
        sampled = [name for name in self.node_params if name not in self.params]
        # The node set is drawn once, on rank 0, and broadcast: it is deterministic, but an
        # engine with a `valid` predicate filters a candidate pool that can be a million points
        # (the EFT-of-DE gate keeps 9%), five minutes of work that 255 other ranks would
        # otherwise repeat -- and a set every rank must agree on exactly is safer sent than
        # recomputed.
        def draw():
            nodes = np.atleast_2d(np.asarray(built.nodes(), dtype='f8'))
            if augment:
                # appended, not merged: the base draw keeps its order, so a checkpoint of the
                # un-augmented training is a prefix of this one and resumes with the extra
                # nodes alone
                nodes = np.concatenate([nodes, self._augmented_nodes(augment, budget=budget, **kwargs)])
            return nodes

        if mpicomm is not None and mpicomm.size > 1:
            nodes = mpicomm.bcast(draw() if mpicomm.rank == 0 else None, root=0)
        else:
            nodes = draw()
        whitened = getattr(built, 'whitened', False)
        self.logger.info(f'training on {len(nodes)} nodes over {len(self.params)} parameters'
                         + (f' (whitened, condition number {built.condition_number():.1f})'
                            if whitened else '')
                         + f'; {len(self.exact_params)} handled exactly'
                         + (f', of which {sampled} sampled at the nodes and divided out'
                            if sampled else ''))
        # The node box in physical parameters. Cheap, and it makes a whole class of bug
        # visible at a glance: a declared transform that never reaches the engine, or is
        # not inverted on the way out, hands the calculator the expansion variable and
        # the nodes silently land outside the region the box describes.
        expansions = {name: spec for name, spec in getattr(built, 'transforms', {}).items() if spec}
        self.logger.info(
            'node box: ' + ', '.join(
                f'{name} [{nodes[:, index].min():.5g}, {nodes[:, index].max():.5g}]'
                for index, name in enumerate(built.params))
            + (f'; expansion variables {expansions}' if expansions else ''))

        # a parameter handled exactly leaves the grid, but the calculator still needs a value for
        # it: hold it at the space centre while sampling, and let `transform` take it out -- unless
        # it is one the subclass wants sampled, in which case it is a node coordinate like the rest
        centers = self.training.center
        fixed = {name: centers[name] for name in self.exact_params if name not in self.node_params}
        training = TrainingSet(self._evaluate_target, nodes, self.node_params, fixed=fixed,
                               checkpoint=checkpoint, chunk=chunk, batch_size=batch_size,
                               mpicomm=mpicomm,
                               **({} if rows_per_rank is None else {'rows_per_rank': rows_per_rank}),
                               # An interpolating engine cannot absorb a hole on its own -- but it
                               # can if the caller also lowers `basis_budget`, which buys the
                               # redundancy by giving up polynomial degree. So the tolerance is
                               # requestable, not merely a property of the engine class.
                               drop_non_finite=(not built.requires_all_nodes)
                               if drop_non_finite is None else bool(drop_non_finite))
        if not training.run():
            raise RuntimeError(f'training incomplete ({training.done}/{len(nodes)}); rerun to '
                               f'continue -- the checkpoint holds what is done')
        if not fit:
            self.logger.info(f'every node evaluated ({training.done}/{len(nodes)}); fit=False, '
                             f'so the emulator stays untrained -- rerun with the same checkpoint to fit')
            return self

        # transform after collection, node by node: the checkpoint holds physical outputs, so
        # changing what is divided out costs a refit, not another run of the Boltzmann code
        inputs, outputs = training.inputs(), training.outputs()
        transformed = {}
        for index, row in enumerate(inputs):
            params = {**fixed, **dict(zip(self.node_params, row))}
            values = self.transform({name: value[index] for name, value in outputs.items()},
                                    params)
            for name, value in values.items():
                transformed.setdefault(name, []).append(np.asarray(value))

        prefixes = {name.split('.', 1)[0] for name in transformed}
        unknown = [name for name in per_output
                   if name not in transformed and name not in prefixes]
        if unknown:
            raise ValueError(f'per_output names {unknown} are not outputs; '
                             f'have {sorted(transformed)}')

        if (not built.requires_all_nodes) or drop_non_finite:
            # A regression engine can be fitted on the survivors; an interpolating one never
            # reaches here, because TrainingSet refuses the node outright. Loud, and bounded: a box
            # that loses a large share of its nodes is a box in the wrong place, and a fit over
            # what is left would be extrapolating into the hole rather than covering it.
            finite = np.ones(len(inputs), dtype='?')
            for values in transformed.values():
                stacked = np.asarray(values).reshape(len(values), -1)
                finite &= np.isfinite(stacked).all(axis=1)
            lost = int((~finite).sum())
            if lost:
                fraction = lost / len(finite)
                self.logger.info(f'dropping {lost}/{len(finite)} nodes ({fraction:.1%}) whose '
                                 f'outputs were non-finite; fitting on the rest')
                if fraction > max_non_finite:
                    raise NodeEvaluationError(
                        f'{lost}/{len(finite)} nodes ({fraction:.1%}) returned non-finite '
                        f'values, above max_non_finite={max_non_finite:.1%}. The box covers a '
                        f'region the calculator cannot evaluate; move or shrink it rather than '
                        f'fitting around the hole.')
                inputs = np.asarray(inputs)[finite]
                transformed = {name: [value for value, keep in zip(values, finite) if keep]
                               for name, values in transformed.items()}
            if outlier_factor is not None or outlier_factor_low is not None:
                if outlier_on not in ('transformed', 'raw', 'both'):
                    raise ValueError(f"outlier_on must be 'transformed', 'raw' or 'both'; got {outlier_on!r}")
                sane = np.ones(len(inputs), dtype='?')
                if outlier_on in ('raw', 'both'):
                    # the calculator's own outputs, on the rows that survived the finite mask (the
                    # transformed values decide finiteness: a log or a ratio can fail where the
                    # raw value is fine, never the other way round)
                    # `rows` rather than a sliced copy of the whole set: every rank holds the raw
                    # set already (~2 GB for the EFT-of-DE training), and a second copy per rank
                    # OOM-killed 32 ranks on a 241 GB node (2026-09-11)
                    sane &= self._outlier_mask(outputs, outlier_factor, outlier_factor_low,
                                               'the raw outputs', rows=finite if lost else None)
                if outlier_on in ('transformed', 'both'):
                    sane &= self._outlier_mask(transformed, outlier_factor, outlier_factor_low,
                                               'the transformed outputs')
                lost = int((~sane).sum())
                if lost:
                    self.logger.info(f'dropping {lost}/{len(sane)} nodes ({lost / len(sane):.1%}) in all '
                                     f'(outlier_on={outlier_on!r}); fitting on the rest')
                    inputs = np.asarray(inputs)[sane]
                    transformed = {name: [value for value, keep in zip(values, sane) if keep]
                                   for name, values in transformed.items()}

        # one engine per output, all sharing the node set -- and, unless `output_coordinates`
        # says otherwise, the coordinates too. Under MPI the outputs are dealt
        # round-robin across the ranks and the fitted engines gathered back, so every rank ends
        # with the same complete set: with the plain loop every rank fitted every output --
        # 16 ranks doing 16 identical copies of the work. Measured on an 11-parameter, 78-output
        # MLP emulator with 5632 samples: the node evaluations took 45 min on 16 ranks and the
        # redundant per-rank fit then ran for more than 5 h, well past the evaluations. The
        # Chebyshev fit is a linear solve and never noticed; the network training is what this
        # is for. States travel through pickle (allgather): an engine's state is a few arrays.
        names = list(transformed)
        # The nodes as training-parameter points, for an output fitted in coordinates of its own
        # (`output_coordinates`): every node coordinate -- a sampled exact parameter included --
        # with the other exact parameters at their fixed values. Taken before the projection
        # below, and only when some output asks for it.
        node_rows = [dict(zip(self.node_params, row)) for row in inputs] \
            if any(self.output_coordinates(name) is not None for name in names) else None
        # The fit sees the expanded parameters only: a sampled exact parameter was a node
        # coordinate up to here, and `transform` has taken its dependence out of every output, so
        # its column is dropped and the nodes stand projected onto the remaining ones.
        if sampled:
            columns = [self.node_params.index(name) for name in self.params]
            inputs = np.asarray(inputs)[:, columns]
        rank, size = (mpicomm.rank, mpicomm.size) if mpicomm is not None else (0, 1)
        # Rank-local reduction before the fits. Up to here every rank holds the whole raw set
        # (`outputs`, and the TrainingSet's own copy of it) plus every transformed output, and
        # nothing below reads any of it but this rank's own outputs: ~6 GB per rank on the
        # 65536-node EFT-of-DE training, which bounded a 241 GB node to 26 ranks and OOM-killed
        # 32 of them in the final allgather AFTER all 64 networks had been fitted (2026-09-12).
        # Keeping only what this rank fits, as stacked arrays, leaves a few hundred MB.
        transformed = {name: np.asarray(transformed[name]) for name in names[rank::size]}
        del outputs
        training.release()
        # ... provided the memory actually goes back to the system. The per-node values are
        # millions of small arrays, allocated through malloc's heap rather than mmap, and glibc
        # keeps freed heap for the process: measured, RSS stayed at 199 GB across 32 ranks after
        # the release (2026-09-12). Freed heap is reused by this process, so the fits and the
        # gather can run in it, but a co-scheduled process (or the cgroup accounting) does not
        # see it -- trim explicitly where libc offers it, and silently do nothing elsewhere.
        try:
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except (OSError, AttributeError):
            pass
        mine = {}
        coordinates = {}      # cached by parameter tuple: several outputs share one basis
        for name in names[rank::size]:
            values = transformed[name]
            options = {'budget': budget, **kwargs,
                       **per_output.get(name, per_output.get(name.split('.', 1)[0], {}))}
            own = self.output_coordinates(name)
            if own is None:
                fit_params, fit_inputs = None, inputs
            else:
                space, fit_params, to_training = own
                fit_params = list(fit_params)
                key = tuple(fit_params)
                if key not in coordinates:
                    mapped = [dict(to_training({**fixed, **row})) for row in node_rows]
                    coordinates[key] = np.array([[point[param] for param in fit_params]
                                                 for point in mapped])
                fit_inputs = coordinates[key]
                options = {**options, 'space': space, 'params': fit_params}
            fit = self._engine(**options)
            fit.fit(fit_inputs, values.reshape(len(values), -1), method=method,
                    basis_budget=basis_budget) \
                if fit.name == 'chebyshev' else fit.fit(fit_inputs,
                                                        values.reshape(len(values), -1))
            if hasattr(fit, 'validation_loss'):
                # one line per output, so a job log can be read per quantity (which networks
                # early-stopped, which are the worst) rather than per anonymous rank
                self.logger.info(f'output {name!r}: {getattr(fit, "epochs_run", "?")}/{fit.epochs} epochs, '
                                 f'validation loss {fit.validation_loss:.3e}')
            mine[name] = (fit.__getstate__(), tuple(values.shape[1:]), fit_params)
        from .engines import engine_from_state
        if size > 1:
            gathered = {}
            for part in mpicomm.allgather(mine):
                gathered.update(part)
        else:
            gathered = mine
        self._engines = {name: (engine_from_state(gathered[name][0]), gathered[name][1], gathered[name][2])
                         for name in names}
        if not self._engines:
            raise RuntimeError('the target returned no outputs, so there is nothing to fit')
        return self

    def _outlier_mask(self, values, outlier_factor, outlier_factor_low, what, rows=None):
        """The nodes the two outlier cuts keep, over ``{name: (nnodes, ...) or list}``; logged.
        ``rows`` (a boolean mask) restricts every output to those nodes first, one output at a
        time, so the whole set is never copied.

        The high cut: some component of some output above ``outlier_factor`` times that
        component's median |value| over the nodes. The low cut: some SIGN-DEFINITE component (one
        that keeps its sign over every node) below the median divided by ``outlier_factor_low``.
        See :meth:`train` for why either exists.
        """
        sane, worst, lowest = None, {}, {}
        for name, value in values.items():
            signed = np.asarray(value)
            if rows is not None:
                signed = signed[rows]
            signed = signed.reshape(len(signed), -1)
            if sane is None:
                sane = np.ones(len(signed), dtype='?')
            stacked = np.abs(signed)
            median = np.median(stacked, axis=0)
            if outlier_factor is not None:
                over = (stacked / np.where(median > 0., median, np.inf)).max(axis=1)
                sane &= over <= outlier_factor
                worst[name] = float(over.max())
            if outlier_factor_low is not None:
                definite = ((signed.min(axis=0) > 0.) | (signed.max(axis=0) < 0.)) & (median > 0.)
                if definite.any():
                    under = (median[definite] / stacked[:, definite]).max(axis=1)
                    sane &= under <= outlier_factor_low
                    lowest[name] = float(under.max())
        lost = int((~sane).sum())
        if lost:
            top = sorted(worst.items(), key=lambda item: -item[1])[:3]
            bottom = sorted(lowest.items(), key=lambda item: -item[1])[:3]
            self.logger.info(
                f'{what}: {lost}/{len(sane)} nodes ({lost / len(sane):.1%}) have an output '
                + (f'above {outlier_factor:g} x its median size (worst: ' + ', '.join(f'{name} at {value:.2g}x' for name, value in top) + ')' if worst else '')
                + (' or ' if worst and lowest else '')
                + (f'below its median / {outlier_factor_low:g} (lowest: ' + ', '.join(f'{name} at 1/{value:.2g}' for name, value in bottom) + ')' if lowest else ''))
        return sane

    # ── use ───────────────────────────────────────────────────────────────────
    # ── constraints ───────────────────────────────────────────────────────────
    def constraints(self):
        """Every constraint the emulator answers under: the trained box, the node cloud (when the
        nodes were whitened along a covariance -- otherwise they fill the box and the two would be
        the same constraint counted twice), then the user's declarative ones.  See :mod:`.constraints`."""
        from .constraints import BoxConstraint, NodeConstraint
        located = self._node_engine({name: 0. for name in self.training.params}) if self._engines else None
        whitened = located is not None and located[0].whitened
        return [BoxConstraint()] + ([NodeConstraint()] if whitened else []) + list(self._constraints)

    def violations(self, **params):
        """``{constraint name: distance}`` at *params* (the user's own), 0 where satisfied. Traceable."""
        return constraint_violations(self.constraints(), self, dict(params), dict(self.to_training(dict(params))))

    def _check(self, given, params):
        """``given``: what the user passed. ``params``: the same, in training coordinates.

        The names are always checked; the constraints only when the values are concrete. Inside a
        jax trace a parameter has no value to compare, so the check is skipped rather than raising
        a TracerBoolConversionError -- but :meth:`predict` still enforces it on the output (NaN),
        following cosmoprimo's usual "raise in eager, NaN inside jax" contract (`exception_or_nan`).
        Returning a silent extrapolation was the old behaviour and it is the dangerous one: a
        sampler cannot tell a wrong number from a right one, while a NaN maps to -inf.
        """
        missing = [name for name in self.space.params if name not in given]
        if missing:
            raise ValueError(f'missing parameters {missing}')
        if self.violation in ('ignore', 'clip', 'nan') or use_jax(*params.values()):
            return
        violated = [constraint for constraint in self.constraints()
                    if float(np.asarray(constraint.violation(self, given, params))) > 0.]
        if not violated:
            return
        reasons = '; '.join(constraint.describe(self, given, params) for constraint in violated)
        converted = ('' if self.training is self.space else
                     f' (the training basis; you gave {dict(given)})')
        message = (f'{reasons}{converted}. Extrapolation here is catastrophic, not gradual -- '
                   f'widen the Space and retrain (nested nodes mean the existing evaluations are '
                   f'reused), or pass violation="ignore" (or "clip", and enforce the constraints yourself).')
        if self.violation == 'raise':
            raise CoverageError(message)
        import warnings
        warnings.warn(message)

    def _node_engine(self, training):
        """``(engine, values, fit_params)``: the engine that answers for the node cloud, and *training*
        in its coordinates; ``None`` when nothing is fitted.

        Delegated to an engine, which owns the geometry the nodes were laid out with -- the
        transforms, the whitening rotation, the per-axis domain (see
        :meth:`~.engines.BaseEngine.distance`). One engine answers for all, since they share the
        node set, but it has to be asked in its own coordinates: an output fitted through
        :meth:`output_coordinates` has others. Preferring an engine that uses the emulator's own
        coordinates is not cosmetic -- which engine comes first is dict order, and that differs
        between a freshly trained emulator and the same one read back from a file.
        """
        if not self._engines:
            return None
        xnp = numpy_jax(*training.values())
        for name, (engine, _, fit_params) in self._engines.items():
            if fit_params is None:
                return engine, xnp.stack([xnp.asarray(training[param]) for param in self.params]), None
        name, (engine, _, fit_params) = next(iter(self._engines.items()))
        mapped = dict(self.output_coordinates(name)[2](training))
        return engine, xnp.stack([xnp.asarray(mapped[param]) for param in fit_params]), fit_params

    def predict(self, **params):
        if not self.trained:
            raise NotTrained('call train() first')
        training = dict(self.to_training(dict(params)))
        self._check(params, training)
        if self.violation == 'clip':
            training = clip_to_constraints(self.constraints(), self, training)
        out = self._evaluate(training)
        if self.violation not in ('ignore', 'clip'):
            # Enforce the constraints on the output. Eager calls already raised in `_check`; this
            # is what makes the guard survive a jit, where the check itself cannot run. NaN
            # propagates to -inf in a posterior, so a violating point is rejected rather than
            # silently extrapolated -- the engines' own words: "catastrophic, not gradual".
            mask = None
            for violation in constraint_violations(self.constraints(), self, dict(params), training).values():
                mask = (violation > 0.) if mask is None else (mask | (violation > 0.))
            if mask is not None:
                # the outputs as well as the parameters: inside someone else's jit, an operation
                # on constant inputs is still staged out, so a prediction made at concrete
                # parameters comes back as a tracer -- which is exactly what happens when a
                # pipeline jits over one parameter and holds this emulator's own fixed. Choosing
                # the numpy from the parameters alone then picks plain numpy and the write of a
                # nan into a traced array raises.
                xnp = numpy_jax(*training.values(), *out.values(), mask)
                out = {name: xnp.where(xnp.reshape(mask, mask.shape + (1,) * (xnp.ndim(value) - xnp.ndim(mask)))
                                       if xnp.ndim(value) > xnp.ndim(mask) else mask,
                                       xnp.nan, value)
                       for name, value in out.items()}
        return out

    def predict_in_box(self, **params):
        """Prediction at *params* clipped into the constraints that clip (the trained box,
        then the node cloud), and ``{constraint name: distance}`` at *params* itself.

        Whatever :attr:`violation` says: for a caller that turns the distances into constraints
        of its own -- desilike's ``Constraint``, a hard wall for a sampler and a soft one for an
        optimiser.  Nothing is silent here, the distances come back with the prediction; inside
        every constraint the prediction is :meth:`predict`'s.  Declarative constraints
        (:class:`~.constraints.LinearConstraint`) are reported, not clipped.
        """
        if not self.trained:
            raise NotTrained('call train() first')
        missing = [name for name in self.space.params if name not in params]
        if missing:
            raise ValueError(f'missing parameters {missing}')
        training = dict(self.to_training(dict(params)))
        constraints = self.constraints()
        violations = constraint_violations(constraints, self, dict(params), training)
        return self._evaluate(clip_to_constraints(constraints, self, training)), violations

    def _evaluate(self, training):
        """The engines' prediction at *training* parameters, with no check."""
        # `xnp` so a traced parameter stays traced: np.array() on a tracer raises, and the whole
        # point of the engines being jax-friendly is that a likelihood can jit through this
        xnp = numpy_jax(*training.values())
        values = xnp.stack([xnp.asarray(training[name]) for name in self.params])
        # an output fitted in coordinates of its own (:meth:`output_coordinates`) is evaluated in
        # them; the map is cached by parameter tuple, since sections sharing a basis share the work
        coordinates, predicted = {}, {}
        for name, (engine, shape, fit_params) in self._engines.items():
            if fit_params is None:
                point = values
            else:
                key = tuple(fit_params)
                if key not in coordinates:
                    own = self.output_coordinates(name)
                    mapped = dict(own[2](training))
                    coordinates[key] = xnp.stack([xnp.asarray(mapped[param])
                                                  for param in fit_params])
                point = coordinates[key]
            predicted[name] = xnp.reshape(engine.predict(point), shape)
        # `transform` saw training parameters at fit time, so its inverse must see them too
        return self.inverse_transform(predicted, training)

    __call__ = predict

    def contract(self, name, matrix):
        """Fold a fixed linear ``matrix`` into one output, exactly and permanently.

        The motivating case is a window matrix: a theory computed on a fine grid can be emulated
        grid-agnostically and then contracted onto the handful of data bins a likelihood actually
        uses -- once, here, instead of on every evaluation. The fine-grid coefficients then exist
        only while fitting.

        Exact, not an approximation: every engine is linear in the coefficients it contracts, so
        ``matrix @ predict(x)`` and ``contract(matrix).predict(x)`` are the same function.

        Only meaningful for outputs that reach the user unchanged -- an output that
        :meth:`inverse_transform` still rescales per prediction is fine, since that is elementwise,
        but one it mixes is not, and this does not check.
        """
        if not self.trained:
            raise NotTrained('call train() first')
        if name not in self._engines:
            raise ValueError(f'no output {name!r}; have {sorted(self._engines)}')
        engine, shape, fit_params = self._engines[name]
        matrix = np.asarray(matrix, dtype='f8')
        if matrix.ndim != 2:
            raise ValueError(f'matrix must be 2-d, got {matrix.ndim}-d')
        if int(np.prod(shape)) != matrix.shape[1]:
            raise ValueError(f'output {name!r} has shape {shape} ({int(np.prod(shape))} values), '
                             f'and the matrix acts on {matrix.shape[1]}')
        self._engines[name] = (engine.contract(matrix), (matrix.shape[0],), fit_params)
        return self

    def validate(self, truth=None, points=None, metric=None, npoints=100, seed=42,
                 metric_name=None, **kwargs):
        """Compare against a reference -- the target itself, by default.

        Leads with sigma, not the mean: a constant offset cancels under importance reweighting,
        and only the scatter costs sample size.

        The default ``metric`` is the worst over outputs of ``rms(prediction - reference) /
        rms(reference)`` -- a ratio of norms, never a pointwise ratio, which would divide by zero
        wherever a cross-spectrum changes sign. Pass a chi2 against a real covariance when you
        have one; that is the number that actually matters.
        """
        points = points if points is not None else self.space.draw(size=npoints, seed=seed)
        if metric is None:
            metric, metric_name = _relative_rms, metric_name or 'relative rms'
        return _validate(predict=lambda params: self.predict(**params),
                         truth=truth if truth is not None else self.target,
                         points=points, metric=metric, space=self.space,
                         metric_name=metric_name or 'dchi2', **kwargs)

    # ── state ─────────────────────────────────────────────────────────────────
    def __getstate__(self):
        """Enough to predict, not to retrain.

        The target is a plain callable and may be a lambda over a Boltzmann code, so it is not
        saved; a subclass that can rebuild its own target restores it in :meth:`__setstate__`.
        Saying so is better than pickling a closure that would break on the next import.
        """
        if not self.trained:
            raise NotTrained('nothing to write; call train() first')
        import cosmoprimo

        return {'version': int(self.version),
                'cosmoprimo_version': str(getattr(cosmoprimo, '__version__', 'unknown')),
                'cls': f'{type(self).__module__}.{type(self).__name__}',
                'space': self.space.__getstate__(),
                'training': self.training.__getstate__(),
                'params': list(self.params), 'node_params': list(self.node_params),
                'engine_name': self.engine_name,
                'violation': self.violation,
                # for a reader older than `violation`, which knows this key only
                'coverage': self.violation if self.violation in ('raise', 'warn', 'ignore') else 'raise',
                'constraints': [constraint.__getstate__() for constraint in self._constraints],
                'options': dict(self.options),
                'engines': {name: (engine.__getstate__(), tuple(shape),
                                   None if fit_params is None else list(fit_params))
                            for name, (engine, shape, fit_params) in self._engines.items()}}

    def __setstate__(self, state):
        from .engines import engine_from_state

        version = int(state.get('version', 0))
        if version != self.version:
            raise StateVersionError(
                f'this file was written at state version {version}, and '
                f'{type(self).__name__} reads version {self.version}. Retrain, or check out the '
                f'version of cosmoprimo that wrote it '
                f'({state.get("cosmoprimo_version", "unknown")}).')
        self.space = Space.__new__(Space)
        self.space.__setstate__(state['space'])
        self.training = Space.__new__(Space)
        self.training.__setstate__(state['training'])
        self.params, self.engine_name = list(state['params']), state['engine_name']
        # files written before exact parameters could be sampled at the nodes have no entry: the
        # nodes then varied the expanded parameters and nothing else
        self.node_params = list(state.get('node_params', self.params))
        # `coverage`: the name before `violation`, the only one a file written then carries
        from .constraints import constraint_from_state
        self.violation = _check_violation(state.get('violation', state.get('coverage', 'raise')))
        self._constraints = [constraint_from_state(constraint) for constraint in state.get('constraints', [])]
        self.options = dict(state['options'])
        # An entry is `(engine, shape, params)` -- the coordinates that output was fitted in,
        # `None` for the emulator's own. A file written before those existed holds a pair, and a
        # pair says exactly one thing, so it is read rather than refused: these files are hours of
        # Boltzmann code, and no version of the emulator ever wrote a pair meaning anything else.
        self._engines = {}
        for name, entry in state['engines'].items():
            engine, shape, fit_params = entry if len(entry) == 3 else (*entry, None)
            self._engines[name] = (engine_from_state(engine), tuple(shape),
                                   None if fit_params is None else list(fit_params))
        self.target = None

    def write(self, path):
        """Write the trained emulator to ``path``. Read it back with :meth:`read`.

        HDF5 unless the name ends in ``.npy``; a bare name gets ``.h5``. Returns the path
        actually written, which is the one to hand to ``Cosmology(engine=...)``.
        """
        from .io import write_state

        return write_state(path, self.__getstate__())

    @classmethod
    def from_state(cls, state):
        """Rebuild whichever subclass wrote this state.

        Separate from :meth:`read` so a state can be nested inside another emulator's -- an
        emulator that trains a helper emulator of its own has to carry it along, or reading it
        back gives something that cannot predict.
        """
        import importlib

        module, name = state['cls'].rsplit('.', 1)
        saved = getattr(importlib.import_module(module), name)
        if not issubclass(saved, Emulator):
            raise TypeError(f'{state["cls"]} is not an Emulator')
        new = saved.__new__(saved)
        new.__setstate__(state)
        return new

    @classmethod
    def read(cls, path):
        """Read a trained emulator, of whatever subclass wrote it."""
        from .io import read_state

        return cls.from_state(read_state(path))

    def __repr__(self):
        return (f'{type(self).__name__}({len(self.params)} expanded, '
                f'{len(self.exact_params)} exact, engine={self.engine_name!r}, '
                f'trained={self.trained})')
