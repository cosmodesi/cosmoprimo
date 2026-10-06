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
        return self._engine(budget=budget, **kwargs).nodes()

    def _engine(self, budget=None, space=None, params=None, **kwargs):
        from .engines import ChebyshevEngine
        from .mlp import MLPEngine
        from .polynomial import PolynomialEngine
        from .taylor import TaylorEngine

        classes = {cls.name: cls
                   for cls in (ChebyshevEngine, TaylorEngine, PolynomialEngine, MLPEngine)}
        # `space`/`params` for an output fitted in coordinates of its own
        # (:meth:`output_coordinates`); the emulator's own otherwise
        space = self.training if space is None else space
        params = self.params if params is None else list(params)
        subspace = space.marginal(params) if len(params) < len(space.params) else space
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
        return cls(**subspace.geometry(), budget=budget, **options)

    def train(self, engine=None, budget=None, checkpoint=None, chunk=None, batch_size=None,
              mpicomm=None, per_output=None, max_non_finite=0.05, method='auto',
              basis_budget=None, drop_non_finite=None, **kwargs):
        """Evaluate the calculator on the node set and fit.

        Resumable and chunked: pass ``checkpoint`` and ``chunk='30min'`` for anything expensive,
        then rerun until it reports complete. A kill then costs one node, not the training.
        ``batch_size`` calls the target with dicts of arrays of that length instead of one node
        at a time; ``mpicomm`` splits the nodes across ranks.

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
        built = self._engine(budget=budget, **kwargs)
        nodes = built.nodes()
        whitened = getattr(built, 'whitened', False)
        self.logger.info(f'training on {len(nodes)} nodes over {len(self.params)} parameters'
                         + (f' (whitened, condition number {built.condition_number():.1f})'
                            if whitened else '')
                         + f'; {len(self.exact_params)} handled exactly')
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
        # it: hold it at the space centre while sampling, and let `transform` take it out
        centers = self.training.center
        fixed = {name: centers[name] for name in self.exact_params}
        training = TrainingSet(self._evaluate_target, nodes, self.params, fixed=fixed,
                               checkpoint=checkpoint, chunk=chunk, batch_size=batch_size,
                               mpicomm=mpicomm,
                               # An interpolating engine cannot absorb a hole on its own -- but it
                               # can if the caller also lowers `basis_budget`, which buys the
                               # redundancy by giving up polynomial degree. So the tolerance is
                               # requestable, not merely a property of the engine class.
                               drop_non_finite=(not built.requires_all_nodes)
                               if drop_non_finite is None else bool(drop_non_finite))
        if not training.run():
            raise RuntimeError(f'training incomplete ({training.done}/{len(nodes)}); rerun to '
                               f'continue -- the checkpoint holds what is done')

        # transform after collection, node by node: the checkpoint holds physical outputs, so
        # changing what is divided out costs a refit, not another run of the Boltzmann code
        inputs, outputs = training.inputs(), training.outputs()
        transformed = {}
        for index, row in enumerate(inputs):
            params = {**fixed, **dict(zip(self.params, row))}
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

        # one engine per output, all sharing the node set -- and, unless `output_coordinates`
        # says otherwise, the coordinates too
        self._engines = {}
        rows = [{**fixed, **dict(zip(self.params, row))} for row in inputs]
        coordinates = {}      # cached by parameter tuple: several outputs share one basis
        for name, values in transformed.items():
            values = np.asarray(values)
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
                    mapped = [dict(to_training(row)) for row in rows]
                    coordinates[key] = np.array([[point[param] for param in fit_params]
                                                 for point in mapped])
                fit_inputs = coordinates[key]
                options = {**options, 'space': space, 'params': fit_params}
            fit = self._engine(**options)
            fit.fit(fit_inputs, values.reshape(len(values), -1), method=method,
                    basis_budget=basis_budget) \
                if fit.name == 'chebyshev' else fit.fit(fit_inputs,
                                                        values.reshape(len(values), -1))
            self._engines[name] = (fit, values.shape[1:], fit_params)
        if not self._engines:
            raise RuntimeError('the target returned no outputs, so there is nothing to fit')
        return self

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
                'params': list(self.params), 'engine_name': self.engine_name,
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
