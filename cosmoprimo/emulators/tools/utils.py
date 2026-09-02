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
  coefficients (:func:`smolyak_combination`).

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
    n_nodes = len(nodes)
    if order >= n_nodes:
        raise ValueError(f'derivative order {order} needs more than {n_nodes} nodes')
    if scale is None:
        scale = 0.5 * (nodes.max() - nodes.min()) or 1.
    rhs = np.zeros(n_nodes)
    rhs[order] = float(math.factorial(order))
    x0 = jnp.asarray(x0)
    scaled = (nodes - x0[..., None]) / scale  # (*B, n)
    rows = [jnp.ones_like(scaled)]
    for _ in range(n_nodes - 1):
        rows.append(rows[-1] * scaled)
    matrix = jnp.stack(rows, axis=-2)  # (*B, n, n)
    rhs_b = jnp.broadcast_to(jnp.asarray(rhs), matrix.shape[:-1])
    return jnp.linalg.solve(matrix, rhs_b[..., None])[..., 0] / scale ** order


def chebyshev_values(t, n_max):
    """Chebyshev polynomials ``T_0..T_{n_max}`` at *t* (scalar or array), shape ``(n_max + 1, *t.shape)``."""
    jnp = numpy_jax(t)
    t = jnp.asarray(t)
    values = [jnp.ones_like(t), t]
    for _ in range(n_max - 1):
        values.append(2. * t * values[-1] - values[-2])
    return jnp.stack(values[:n_max + 1])


def chebyshev_lobatto_nodes(n_nodes, limits=(-1., 1.)):
    """*n_nodes* Chebyshev-Lobatto nodes spanning *limits* (endpoints included), sorted ascending."""
    lo, hi = (float(lim) for lim in limits)
    if n_nodes == 1:
        return np.array([0.5 * (lo + hi)])
    angles = np.pi * np.arange(n_nodes) / (n_nodes - 1)
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
    n_dims = len(max_levels)
    level_vectors = [lv for lv in itertools.product(*[range(level + 1) for level in max_levels])
                     if sum(lv) <= budget]
    level_set = set(level_vectors)
    combination = {}
    for lv in level_vectors:
        coeff = 0
        for z in itertools.product((0, 1), repeat=n_dims):
            if tuple(l + dz for l, dz in zip(lv, z)) in level_set:
                coeff += (-1) ** sum(z)
        if coeff != 0:
            combination[lv] = coeff
    return combination


def cardinal_cubic_weights(nodes, x):
    """Dense local-cubic-Lagrange weights of *x* on uniform *nodes*, shape ``(n_nodes,)``.

    Same 4-node bracketing scheme as folps' fog_collocation_weights (fourth-order accurate,
    weights sum to one), jax-traceable in *x*.
    """
    nodes = jnp.asarray(nodes)
    n_nodes = nodes.shape[0]
    step = nodes[1] - nodes[0]
    position = (jnp.clip(x, nodes[0], nodes[-1]) - nodes[0]) / step
    index = jnp.clip(jnp.floor(position) - 1, 0, n_nodes - 4)
    offset = position - (index + 1.)
    lagrange = [-offset * (offset - 1.) * (offset - 2.) / 6.,
                (offset + 1.) * (offset - 1.) * (offset - 2.) / 2.,
                -(offset + 1.) * offset * (offset - 2.) / 2.,
                (offset + 1.) * offset * (offset - 1.) / 6.]
    node_index = jnp.arange(n_nodes)
    weights = 0.
    for shift, weight in enumerate(lagrange):
        weights = weights + weight * (node_index == index + shift)
    return weights


def lagrange_weights(nodes, x):
    """Dense global-Lagrange weights of *x* on *nodes* (exact for polynomials of degree
    ``n_nodes - 1``), jax-traceable in *x*."""
    nodes = jnp.asarray(nodes)
    n_nodes = nodes.shape[0]
    weights = []
    for node_index in range(n_nodes):
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
