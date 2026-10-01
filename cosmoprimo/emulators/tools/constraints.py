"""The region an emulator answers over, as a list of constraints with distances.

:meth:`~.emulate.Emulator.constraints` returns them; :meth:`~.emulate.Emulator._check` and
:meth:`~.emulate.Emulator.predict` enforce them according to ``violation``; a caller that
enforces them itself (desilike, as a hard wall for a sampler and a soft one for an optimiser)
reads their distances from :meth:`~.emulate.Emulator.violations` or
:meth:`~.emulate.Emulator.predict_in_box`.

Each constraint's :meth:`~Constraint.violation` is a traceable distance, 0 where it is satisfied.
Two are built in and rebuilt from the emulator itself, so they are never saved:

* :class:`BoxConstraint` -- the trained box, one low/high pair per parameter, in the expansion
  variable of the training basis, in units of each box width;
* :class:`NodeConstraint` -- the band the nodes fill when they were whitened along a covariance:
  inside every parameter's own range, a point can still be off it (measured: 70.6% of the
  rectangle for a pair at correlation -0.95), and an interpolant answers there from coefficients
  nothing constrained. Distance in the engine's whitened coordinates, in units of each axis's width.

A user adds declarative ones, saved with the emulator: :class:`LinearConstraint`, a bounded linear
combination of the user's own parameters (``w0 + wa < 0``, a positive density, ...).
"""

import re

import numpy as np

from cosmoprimo.jax import numpy_jax


def _excess(value, low, high, xnp):
    """``max(low - value, 0) + max(value - high, 0)``, either side optional (``None``)."""
    total = 0.
    if low is not None:
        total = total + xnp.maximum(low - value, 0.)
    if high is not None:
        total = total + xnp.maximum(value - high, 0.)
    return total


class Constraint(object):
    """One condition on the parameters, with a traceable distance.

    Attributes
    ----------
    name : str
        A plain identifier (a desilike Variable basename): no dots.
    scale : float
        Width of a soft penalty, in the units of :meth:`violation`.
    """
    kind = None
    #: Whether :meth:`clip` moves a point to the nearest one satisfying the constraint.
    clips = False

    def __init__(self, name, scale=0.01):
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', str(name)):
            raise ValueError(f'constraint name must be an identifier, not {name!r}')
        self.name, self.scale = str(name), float(scale)

    def violation(self, emulator, params, training):
        """Distance outside the constraint (0 inside), from the user's *params* and the *training* ones."""
        raise NotImplementedError

    def clip(self, emulator, training):
        """*training* parameters clipped to the nearest point satisfying the constraint, when :attr:`clips`."""
        return training

    def describe(self, emulator, params, training):
        """One line for an error message, at a violating point."""
        return f'{self.name} (distance {float(self.violation(emulator, params, training)):.3g})'

    def __getstate__(self):
        raise TypeError(f'{type(self).__name__} is built from the emulator and is not saved')

    def __repr__(self):
        return f'{type(self).__name__}({self.name!r})'


class BoxConstraint(Constraint):
    """The trained box (see the module docstring)."""
    kind = 'box'
    clips = True

    def __init__(self, scale=0.01):
        super().__init__('box', scale=scale)

    def _expansion(self, emulator, training):
        return dict(emulator.training.forward(training))

    def violation(self, emulator, params, training):
        xnp = numpy_jax(*training.values())
        expansion = self._expansion(emulator, training)
        total = xnp.zeros(())
        for name in emulator.params:
            low, high = emulator.training.limits[name]
            total = total + _excess(xnp.asarray(expansion[name]), low, high, xnp) / (high - low)
        return total

    def outside(self, emulator, training):
        """``{parameter: value}`` of the user's training parameters outside their own range (eager)."""
        expansion = self._expansion(emulator, training)
        return {name: training[name] for name in emulator.params
                if not (emulator.training.limits[name][0] <= expansion[name] <= emulator.training.limits[name][1])}

    def clip(self, emulator, training):
        xnp = numpy_jax(*training.values())
        expansion = self._expansion(emulator, training)
        for name in emulator.params:
            low, high = emulator.training.limits[name]
            expansion[name] = xnp.clip(xnp.asarray(expansion[name]), low, high)
        return {**training, **dict(emulator.training.inverse(expansion))}

    def describe(self, emulator, params, training):
        return f'outside the trained box: {self.outside(emulator, training)}'


class NodeConstraint(Constraint):
    """The band the whitened nodes fill (see the module docstring)."""
    kind = 'nodes'
    clips = True

    def __init__(self, scale=0.01):
        super().__init__('nodes', scale=scale)

    def violation(self, emulator, params, training):
        located = emulator._node_engine(training)
        if located is None:
            return numpy_jax(*training.values()).zeros(())
        engine, values, _ = located
        return engine.distance(values)

    def clip(self, emulator, training):
        located = emulator._node_engine(training)
        if located is None:
            return training
        engine, values, fit_params = located
        if fit_params is not None:
            # an engine fitted in coordinates of its own cannot hand the emulator's back
            return training
        clipped = engine.clip_to_domain(values)
        return {**training, **{name: clipped[index] for index, name in enumerate(emulator.params)}}

    def describe(self, emulator, params, training):
        return 'inside the box but off the node cloud it was fitted on'


class LinearConstraint(Constraint):
    """``lower <= sum_i coefficients[i] * params[i] <= upper``, in the user's own parameters.

    Declarative, so it is saved with the emulator.  Either bound may be ``None``.  The distance is
    the excess over the bound, in the units of the combination.

    Parameters
    ----------
    name : str
        An identifier.
    coefficients : dict
        ``{parameter: coefficient}``.
    lower, upper : float, optional
    scale : float, default=0.01
    """
    kind = 'linear'

    def __init__(self, name, coefficients, lower=None, upper=None, scale=0.01):
        super().__init__(name, scale=scale)
        if lower is None and upper is None:
            raise ValueError(f'constraint {name!r} needs a lower or an upper bound')
        self.coefficients = {str(param): float(value) for param, value in dict(coefficients).items()}
        self.lower = None if lower is None else float(lower)
        self.upper = None if upper is None else float(upper)

    @classmethod
    def from_string(cls, text, name=None, aliases=None, scale=0.01):
        """``'w0 + wa < -0.5'``, ``'2 omega_b - omega_cdm >= 0'``, ``'-1 < w0 < 0'``: a linear
        combination of parameter names with numeric coefficients, bounded by ``<``/``<=`` or
        ``>``/``>=``.  *aliases* maps the names written to the parameters' own (``{'w0': 'w0_fld'}``)."""
        aliases = dict(aliases or {})
        parts = re.split(r'\s*(<=|>=|<|>)\s*', str(text).strip())
        if len(parts) not in (3, 5) or any(not part for part in parts[::2]):
            raise ValueError(f'cannot read {text!r} as a bounded linear combination')

        def number(token):
            try:
                return float(token)
            except ValueError:
                return None

        def combination(token):
            coefficients = {}
            for sign, term in re.findall(r'([+-]?)\s*([^+-]+)', token.replace(' ', '')):
                match = re.fullmatch(r'([0-9.]*(?:[eE][-+]?[0-9]+)?)\*?([A-Za-z_][A-Za-z0-9_]*)', term)
                if match is None:
                    raise ValueError(f'cannot read the term {term!r} of {text!r}')
                factor = float(match.group(1)) if match.group(1) else 1.
                param = aliases.get(match.group(2), match.group(2))
                coefficients[param] = coefficients.get(param, 0.) + (-factor if sign == '-' else factor)
            return coefficients

        lower = upper = None
        if len(parts) == 5:
            low, op1, expression, op2, high = parts
            if op1[0] != op2[0] or number(low) is None or number(high) is None:
                raise ValueError(f'cannot read {text!r} as a bounded linear combination')
            lower, upper = (number(low), number(high)) if op1[0] == '<' else (number(high), number(low))
            coefficients = combination(expression)
        else:
            left, op, right = parts
            if number(right) is not None:
                coefficients, bound, less = combination(left), number(right), op[0] == '<'
            elif number(left) is not None:
                coefficients, bound, less = combination(right), number(left), op[0] == '>'
            else:
                raise ValueError(f'one side of {text!r} must be a number')
            lower, upper = (None, bound) if less else (bound, None)
        if name is None:
            name = '_'.join(param for param in coefficients)
            name = re.sub(r'[^A-Za-z0-9_]', '_', name) + ('_upper' if upper is not None and lower is None else '_bound')
        return cls(name, coefficients, lower=lower, upper=upper, scale=scale)

    def value(self, params, xnp=np):
        return sum(coefficient * xnp.asarray(params[param]) for param, coefficient in self.coefficients.items())

    def violation(self, emulator, params, training):
        xnp = numpy_jax(*params.values())
        return xnp.asarray(_excess(self.value(params, xnp=xnp), self.lower, self.upper, xnp))

    def describe(self, emulator, params, training):
        terms = ' + '.join(f'{coefficient:g} {param}' for param, coefficient in self.coefficients.items())
        bounds = ' and '.join(text for text in [None if self.lower is None else f'>= {self.lower:g}',
                                                None if self.upper is None else f'<= {self.upper:g}'] if text)
        return f'{self.name}: {terms} = {float(self.value(params)):.6g}, required {bounds}'

    def __getstate__(self):
        return {'kind': self.kind, 'name': self.name, 'coefficients': dict(self.coefficients),
                'lower': self.lower, 'upper': self.upper, 'scale': self.scale}

    @classmethod
    def from_state(cls, state):
        state = dict(state)
        state.pop('kind', None)
        return cls(**state)

    def __eq__(self, other):
        return isinstance(other, LinearConstraint) and self.__getstate__() == other.__getstate__()

    def __repr__(self):
        return f'LinearConstraint({self.name!r}, {self.coefficients}, lower={self.lower}, upper={self.upper})'


def constraint_violations(constraints, emulator, params, training):
    """``{constraint name: distance}`` of *constraints* for *emulator* at the user's *params*,
    whose training-basis values are *training*; 0 where a constraint is satisfied. Traceable."""
    return {constraint.name: constraint.violation(emulator, params, training) for constraint in constraints}


def clip_to_constraints(constraints, emulator, training):
    """*training* parameters clipped by each of *constraints* that clips, in order (the box
    first, then the node cloud); the others (declarative, :class:`LinearConstraint`) leave it as is."""
    for constraint in constraints:
        if constraint.clips:
            training = constraint.clip(emulator, training)
    return training


def constraint_from_state(state):
    """Read back a saved (declarative) constraint."""
    kinds = {LinearConstraint.kind: LinearConstraint}
    if state.get('kind') not in kinds:
        raise ValueError(f"unknown constraint kind {state.get('kind')!r}")
    return kinds[state['kind']].from_state(state)


def as_constraint(value):
    """A :class:`LinearConstraint`, its text form, or its saved state."""
    if isinstance(value, Constraint):
        return value
    if isinstance(value, str):
        return LinearConstraint.from_string(value)
    if isinstance(value, dict):
        return constraint_from_state(value)
    raise TypeError(f'cannot read a constraint from {type(value).__name__}')
