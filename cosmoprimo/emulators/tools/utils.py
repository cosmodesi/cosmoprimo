"""Shared polynomial-basis / stencil mathematics for emulators.

Single home for the node, weight and index-set constructions used by both the cosmoprimo
emulator engines and desilike's graph-level emulators:

- uniform centered finite-difference stencils (:func:`fd_stencil`);
- polynomial-interpolation (Fornberg-type) derivative weights at an arbitrary point from
  arbitrary nodes (:func:`interpolation_weights`) -- the correct replacement for shifted
  uniform stencils near prior boundaries (a shifted stencil evaluated with centered weights
  silently returns the derivative at the shifted point, not the requested one);
- Chebyshev machinery: values (:func:`chebyshev_values`), Lobatto nodes
  (:func:`chebyshev_lobatto_nodes`) and the per-dimension change of basis
  (:func:`chebyshev_vandermonde_inverse`);
- named expansion-variable transforms (:data:`TRANSFORMS`), e.g. ``'sqrt'`` -- the natural
  variable for the neutrino mass (free-streaming scale ~ sqrt(m));
- anisotropic Smolyak sparse grids: nested Chebyshev-Lobatto levels
  (:func:`nested_level_nodes`) and the admissible level set with combination-technique
  coefficients (:func:`smolyak_combination`);
- per-parameter option dicts with a ``'*'`` default (:func:`expand_dict`);
- free multi-index sets (:func:`multi_index_set`) and the tensor-product basis they address
  (:func:`tensor_basis`), which is what a *regression* over a polynomial basis needs: a
  collocation grid ties its index set to its node set, a regression does not.

Static (fit-time) constructions use plain numpy; value-dependent functions dispatch through
:func:`cosmoprimo.jax.numpy_jax`, so they work both eagerly and inside jax traces.
"""

import itertools
import math

import numpy as np

from cosmoprimo.jax import numpy as jnp, numpy_jax


def fd_stencil(order, acc=2):
    """
    Uniform centered finite-difference stencil for the order-th derivative at accuracy acc.

    Returns (offsets, coeffs): integer offsets and weights such that
    ``f^(order)(x) ~ sum(coeffs[i] * f(x + offsets[i] * h)) / h^order``.
    Zero-weight points (e.g. the center for odd orders) are omitted.
    """
    nside = (order + acc - 1) // 2
    offsets = np.arange(-nside, nside + 1)
    # Vandermonde system: sum_j c_j * j^k = order! * delta(k, order)
    matrix = np.array([[float(offset) ** k for offset in offsets] for k in range(len(offsets))])
    rhs = np.zeros(len(offsets))
    rhs[order] = float(math.factorial(order))
    coeffs = np.linalg.solve(matrix, rhs)
    mask = np.abs(coeffs) > 1e-12
    return offsets[mask], coeffs[mask]


def interpolation_weights(nodes, x0, order, scale=None):
    """
    Polynomial-interpolation weights for the order-th derivative at *x0* from *nodes*.

    Solves the Vandermonde system ``sum_j w_j u_j^r = r! delta(r, order)`` in the scaled
    positions ``u_j = (nodes_j - x0) / scale``; the derivative is then
    ``f^(order)(x0) = sum_j w_j f(nodes_j)`` (the returned weights include the
    ``1 / scale^order`` factor). Exact for any polynomial of degree < len(nodes); reduces
    to the classical centered weights on a symmetric uniform grid.

    *x0* may carry batch dimensions (shape ``B``); *nodes* is 1D of length ``n``; the
    returned weights have shape ``(*B, n)``. Dispatches on the input types, so *x0* may be
    a jax tracer.
    """
    jnp = numpy_jax(x0)
    nodes = np.asarray(nodes, dtype='f8')
    nnodes = len(nodes)
    if order >= nnodes:
        raise ValueError(f'derivative order {order} needs more than {nnodes} nodes')
    if scale is None:
        scale = 0.5 * (nodes.max() - nodes.min()) or 1.
    rhs = np.zeros(nnodes)
    rhs[order] = float(math.factorial(order))
    x0 = jnp.asarray(x0)
    scaled = (nodes - x0[..., None]) / scale  # (*B, n)
    rows = [jnp.ones_like(scaled)]
    for _ in range(nnodes - 1):
        rows.append(rows[-1] * scaled)
    matrix = jnp.stack(rows, axis=-2)  # (*B, n, n)
    rhs_b = jnp.broadcast_to(jnp.asarray(rhs), matrix.shape[:-1])
    return jnp.linalg.solve(matrix, rhs_b[..., None])[..., 0] / scale ** order


def chebyshev_values(t, nmax):
    """Chebyshev polynomials ``T_0..T_{nmax}`` at *t* (scalar or array), shape ``(nmax + 1, *t.shape)``."""
    jnp = numpy_jax(t)
    t = jnp.asarray(t)
    values = [jnp.ones_like(t), t]
    for _ in range(nmax - 1):
        values.append(2. * t * values[-1] - values[-2])
    return jnp.stack(values[:nmax + 1])


def legendre_values(t, nmax):
    """Legendre polynomials ``P_0..P_{nmax}`` at *t*, shape ``(nmax + 1, *t.shape)``.

    The companion of :func:`chebyshev_values`, and the right family when the samples are drawn
    uniformly over the box rather than from the arcsine measure: a least-squares fit is best
    conditioned when its basis is orthonormal under the measure the points came from, and
    Legendre is the family orthogonal under the uniform one.
    """
    jnp = numpy_jax(t)
    t = jnp.asarray(t)
    values = [jnp.ones_like(t), t]
    for degree in range(1, nmax):
        values.append(((2 * degree + 1) * t * values[-1] - degree * values[-2]) / (degree + 1))
    return jnp.stack(values[:nmax + 1])


#: The orthogonal families a tensor basis can be built from, by name.
BASES = {'chebyshev': chebyshev_values, 'legendre': legendre_values}


def tensor_basis(values, powers, domains, basis='chebyshev'):
    r"""Evaluate the tensor-product basis addressed by *powers*.

    .. math:: \phi_k(x) = \prod_d Q_{\alpha_{kd}}\!\left(s_d(x_d)\right),

    with ``Q`` the family named by *basis* and ``s_d`` the affine map taking ``domains[d]`` onto
    ``[-1, 1]``. One function serves three callers that have to agree exactly, or the fit and the
    evaluation are in different bases: an interpolant's ``predict``, the design matrix of a
    least-squares fit, and the candidate pool a point selection scores.

    Parameters
    ----------
    values : array
        ``(nparams,)`` for one point, or ``(nparams, npoints)`` for many -- parameters first,
        which is the layout :meth:`~.engines.BaseEngine._traced` returns, and what lets one point
        and a whole design matrix share this code.
    powers : array
        ``(nterms, nparams)`` multi-indices, e.g. from :func:`multi_index_set`.
    domains : array
        ``(nparams, 2)`` low/high per axis, in the same coordinates as *values*.
    basis : str, default='chebyshev'
        A key of :data:`BASES`.

    Returns
    -------
    phi : array
        ``(nterms,)`` or ``(nterms, npoints)``.

    Dispatches through :func:`cosmoprimo.jax.numpy_jax`, so it is traceable.
    """
    if basis not in BASES:
        raise ValueError(f'unknown basis {basis!r}; available {sorted(BASES)}')
    evaluate = BASES[basis]
    xnp = numpy_jax(values)
    values = xnp.asarray(values)
    powers = np.asarray(powers)
    domains = np.asarray(domains, dtype='f8')
    factors = []
    for index in range(powers.shape[1]):
        low, high = domains[index]
        scaled = (2. * values[index] - low - high) / (high - low)
        table = evaluate(scaled, int(powers[:, index].max()))
        factors.append(table[powers[:, index]])
    return xnp.prod(xnp.stack(factors), axis=0)


def expand_dict(value, names, label=''):
    """``{name: value}`` from an int or a dict, whose ``'*'`` key, if any, is the default."""
    if not isinstance(value, dict):
        return {name: value for name in names}
    default = value.get('*', None)
    unknown = [name for name in value if name != '*' and name not in names]
    if unknown:
        raise ValueError(f'{label} names unknown parameters {unknown}; have {list(names)}')
    if default is None:
        missing = [name for name in names if name not in value]
        if missing:
            raise ValueError(f'{label} is missing {missing}; give them, or a "*" default')
    return {name: value.get(name, default) for name in names}


def multi_index_set(orders, budget=None, interaction='total'):
    """The multi-indices a polynomial basis keeps: per-axis degree caps, cut by an interaction rule.

    A collocation grid has no such choice -- its index set is whatever makes its node set
    unisolvent. A regression does, and that choice is where the dimensional scaling is won or
    lost: at 6 parameters and degree 3, the full tensor product holds 4096 terms, total degree
    holds 84, and the hyperbolic cross holds 34.

    Parameters
    ----------
    orders : sequence of int
        Maximum degree per axis. Anisotropic, so an axis the output barely depends on can be
        given 1, and every term above that disappears along with the samples only it needed.
    budget : int, default=None
        The interaction cut, ``max(orders)`` by default -- which for ``'total'`` is the full
        polynomial of that degree. Lowering it drops mixed terms and leaves the pure ones alone.
    interaction : str, default='total'
        - ``'total'``: total degree, ``sum(alpha) <= budget``.
        - ``'hyperbolic'``: the hyperbolic cross, ``prod(alpha + 1) <= budget + 1``. Keeps every
          pure term the caps allow -- a degree-``budget`` term along one axis costs exactly its
          budget -- while a term mixing several axes is charged the product, so high-order
          interactions are what vanishes first. This is the sparse rule, and it encodes the same
          premise a Smolyak grid is built on: a smooth function's mixed high derivatives are far
          smaller than its pure ones.
        - ``'tensor'``: no cut, the full product of the per-axis caps.

    Returns
    -------
    powers : array
        ``(nterms, nparams)`` int, lexicographically sorted -- the ordering
        :meth:`~.engines.ChebyshevEngine.fit` also gives its sparse coefficients, so the two
        engines' ``powers`` mean the same thing.
    """
    if interaction not in ('total', 'hyperbolic', 'tensor'):
        raise ValueError(f"interaction must be 'total', 'hyperbolic' or 'tensor'; "
                         f'got {interaction!r}')
    orders = [int(order) for order in orders]
    if any(order < 0 for order in orders):
        raise ValueError(f'orders must be non-negative; got {orders}')
    ndim = len(orders)
    budget = (max(orders) if orders else 0) if budget is None else int(budget)
    alpha, indices = [0] * ndim, []

    def cost(depth):
        """The rule's cost of the first *depth* entries. Non-decreasing in every entry, which is
        what lets the enumeration prune rather than filter a full tensor product it could never
        afford to build in the first place."""
        if interaction == 'total':
            return sum(alpha[:depth])
        if interaction == 'hyperbolic':
            product = 1
            for value in alpha[:depth]:
                product *= value + 1
            return product - 1
        return 0

    def recurse(depth):
        if depth == ndim:
            indices.append(tuple(alpha))
            return
        for value in range(orders[depth] + 1):
            alpha[depth] = value
            if cost(depth + 1) > budget:
                break               # monotone in `value`: nothing larger can fit either
            recurse(depth + 1)
        alpha[depth] = 0

    recurse(0)
    return np.array(sorted(indices), dtype='i4').reshape(-1, ndim)


def chebyshev_lobatto_nodes(nnodes, limits=(-1., 1.)):
    """*nnodes* Chebyshev-Lobatto nodes spanning *limits* (endpoints included), sorted ascending."""
    lo, hi = (float(lim) for lim in limits)
    if nnodes == 1:
        return np.array([0.5 * (lo + hi)])
    angles = np.pi * np.arange(nnodes) / (nnodes - 1)
    return np.sort(0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(angles))


def chebyshev_vandermonde_inverse(nodes, limits=(-1., 1.)):
    """Inverse Chebyshev Vandermonde at *nodes* (in *limits*), mapping values at the nodes to
    Chebyshev coefficients of degree ``len(nodes) - 1``: ``coeffs = inverse @ values``."""
    lo, hi = (float(lim) for lim in limits)
    t_nodes = (2. * np.asarray(nodes, dtype='f8') - lo - hi) / (hi - lo)
    return np.linalg.inv(np.polynomial.chebyshev.chebvander(t_nodes, len(t_nodes) - 1))


def _sqrt_forward(x):
    jnp = numpy_jax(x)
    return jnp.sqrt(jnp.maximum(x, 0.))


# Named expansion-variable transforms: name -> (forward, inverse), forward monotone
# increasing. With a transform, step sizes / anchors / collocation ranges are in
# transformed units, and derivatives are w.r.t. the transformed variable.
TRANSFORMS = {'sqrt': (_sqrt_forward, lambda u: u * u)}


def nested_level_nodes(level, limits=(-1., 1.)):
    """Nodes of the nested Chebyshev-Lobatto rule at *level*: 1 node at level 0,
    ``2^level + 1`` nodes otherwise; each level's nodes contain the previous level's."""
    return chebyshev_lobatto_nodes(1 if level == 0 else 2 ** level + 1, limits=limits)


def smolyak_combination(max_levels, budget):
    """
    Anisotropic Smolyak level set and combination-technique coefficients.

    Parameters
    ----------
    max_levels : sequence of int
        Maximum 1-D level per dimension.
    budget : int
        Total level budget: the admissible set is ``{l : l_i <= max_levels[i], sum l_i <= budget}``.

    Returns
    -------
    dict
        ``{level_vector: coefficient}`` restricted to non-zero combination coefficients
        ``c_l = sum_{z in {0,1}^d} (-1)^{|z|} [l + z admissible]``.
    """
    max_levels = [int(level) for level in max_levels]
    ndims = len(max_levels)
    level_vectors = [lv for lv in itertools.product(*[range(level + 1) for level in max_levels])
                     if sum(lv) <= budget]
    level_set = set(level_vectors)
    combination = {}
    for lv in level_vectors:
        coeff = 0
        for z in itertools.product((0, 1), repeat=ndims):
            if tuple(l + dz for l, dz in zip(lv, z)) in level_set:
                coeff += (-1) ** sum(z)
        if coeff != 0:
            combination[lv] = coeff
    return combination


def cardinal_cubic_weights(nodes, x):
    """Dense local-cubic-Lagrange weights of *x* on uniform *nodes*, shape ``(nnodes,)``.

    Same 4-node bracketing scheme as folps' fog_collocation_weights (fourth-order accurate,
    weights sum to one), jax-traceable in *x*.
    """
    nodes = jnp.asarray(nodes)
    nnodes = nodes.shape[0]
    step = nodes[1] - nodes[0]
    position = (jnp.clip(x, nodes[0], nodes[-1]) - nodes[0]) / step
    index = jnp.clip(jnp.floor(position) - 1, 0, nnodes - 4)
    offset = position - (index + 1.)
    lagrange = [-offset * (offset - 1.) * (offset - 2.) / 6.,
                (offset + 1.) * (offset - 1.) * (offset - 2.) / 2.,
                -(offset + 1.) * offset * (offset - 2.) / 2.,
                (offset + 1.) * offset * (offset - 1.) / 6.]
    node_index = jnp.arange(nnodes)
    weights = 0.
    for shift, weight in enumerate(lagrange):
        weights = weights + weight * (node_index == index + shift)
    return weights


def lagrange_weights(nodes, x):
    """Dense global-Lagrange weights of *x* on *nodes* (exact for polynomials of degree
    ``nnodes - 1``), jax-traceable in *x*."""
    nodes = jnp.asarray(nodes)
    nnodes = nodes.shape[0]
    weights = []
    for node_index in range(nnodes):
        others = jnp.delete(nodes, node_index, assume_unique_indices=True)
        weights.append(jnp.prod((x - others) / (nodes[node_index] - others)))
    return jnp.stack(weights)


def logit_transform(low, high):
    """``(forward, inverse)`` for a variable confined to the open interval ``(low, high)``.

    ``forward = log((x - low) / (high - x))`` maps that interval onto the whole real line, so an
    expansion variable built with it can never leave it -- which is what lets a Chebyshev box
    sit against a hard bound (e.g. ``w0 + wa < 0``) without a single node crossing it. A Smolyak
    grid is unisolvent, so one node past the bound is not a smaller problem, it is a singular one.

    The inverse is written as a sigmoid rather than ``(low + high e^u) / (1 + e^u)``: the latter
    overflows to ``inf / inf = nan`` for large ``u``, where this saturates cleanly at a bound
    (past ``|u| ~ 37`` in float64, far outside any box).
    """
    low, high = float(low), float(high)

    def forward(value):
        # the dispatching numpy, not the plain one: this runs inside a jit at every prediction, where plain
        # numpy raises on a tracer.
        return numpy_jax(value).log((value - low) / (high - value))

    def inverse(value):
        xnp = numpy_jax(value)
        return low + (high - low) / (1. + xnp.exp(-value))

    return forward, inverse


#: The dark-energy bound as an expansion variable: ``w0 + wa`` confined to (-5, 0). 0 is CAMB's
#: PPF limit ("w + wa > 0 gives w>0 at high redshift"), stricter than cosmoprimo's own 1/3 check;
#: -5 is a floor below any plausible posterior (a CMB-only w0waCDM chain reaches -4.45) and is not
#: cosmetic -- placed too far below, the logit is strongly right-skewed and a symmetric box cannot
#: cover both tails. Registered under a name so a trained emulator's geometry stays serialisable.
TRANSFORMS['logit_w0pwa'] = logit_transform(-5., 0.)
