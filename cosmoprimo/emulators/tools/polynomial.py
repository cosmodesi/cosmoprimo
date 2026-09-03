"""A polynomial *regression* engine: a small basis, scattered points, a least-squares solve.

The sparse grid in :mod:`engines` and this share a functional form -- both predict
``coefficients @ phi(x)`` over a tensor-product Chebyshev basis, so both evaluate in one
contraction and differentiate for free. What they do not share is where the coefficients come
from, and that is the whole difference::

    grid:        index set == node set, square Vandermonde, exact at the nodes
    regression:  index set chosen, node set chosen, overdetermined, exact nowhere

Three things follow, and each is a reason to reach for this engine.

**A missing evaluation is one missing row.** A Smolyak grid is unisolvent -- exactly as many
nodes as basis terms -- so one node the Boltzmann code refuses makes the system singular, and a
NaN there mixes into every coefficient. Here the row is simply struck from the design matrix and
the remaining ones still determine the basis, provided there are more of them than terms. That is
what ``oversampling`` buys, and it is why this engine can cover a box containing a region the
truth does not exist in -- the ``w0 + wa > 0`` corner of a w0waCDM prior, roughly a fifth of it,
which no collocation grid can be laid over.

**The index set is free.** A grid's terms are whatever its nodes make solvable. Here the basis is
declared -- total degree, or a hyperbolic cross (see :func:`~.utils.multi_index_set`) -- and the
node count follows from it rather than the other way round. At 6 parameters and degree 3 that is
84 terms for total degree, or 34 for the cross, against the tensor product's 4096. Which of the
two is not a free choice: measured, total degree is worth 6-7x in error at matched cost until the
parameter count makes it unaffordable, because it keeps the low-order interactions the cross
trades away.

**The points can be chosen inside the valid region.** Nothing requires a lattice, so the
candidates are drawn cheaply, filtered by a ``valid`` predicate that costs nothing to evaluate,
and only the survivors are ever handed to the calculator. Ten thousand candidates, fifty CLASS
calls, and none of them spent where there is no answer. Which of the survivors to buy is then a
second question, and a smaller one than it looks: pivoted QR on the Vandermonde picks the subset
that conditions the basis best, which is worth having only while conditioning is what binds --
see :func:`fekete_selection`, where the crossover is measured, and the ``selection`` argument of
:class:`PolynomialEngine`, which sits on the right side of it by default.

Against :class:`~.taylor.TaylorEngine`, whose cost is comparable: a Taylor expansion is anchored
at one point and degrades away from it, while this is fitted over the whole box and its error is
spread across it. Against :class:`~.mlp.MLPEngine`, which is also a regression and also tolerates
holes: the network needs orders of magnitude more samples and gives up exact derivatives and a
deterministic fit, so it is the answer at many more parameters than this basis can carry, not at
these.

What it does not do is guarantee anything at its nodes. An interpolant is exact there and that is
a real property to give up; take the grid whenever the grid fits.
"""

import logging

import numpy as np

from .engines import LinearBasisEngine
from .utils import BASES, expand_dict, multi_index_set, tensor_basis


#: How the candidate pool is drawn, by name -- the *sampling measure*, not the basis. A
#: least-squares fit over an orthogonal family is best conditioned when its points are drawn from
#: the measure that family is orthogonal under, so the two are paired by :data:`MEASURE_FOR_BASIS`
#: unless the caller says otherwise.
MEASURES = ('chebyshev', 'uniform', 'samples')

#: The measure each basis is orthogonal under.
MEASURE_FOR_BASIS = {'chebyshev': 'chebyshev', 'legendre': 'uniform'}


def draw_measure(unit, domains, measure):
    """Map points from the unit cube onto the box, under *measure*.

    ``'uniform'`` is the affine map. ``'chebyshev'`` is the inverse arcsine CDF,
    ``x = -cos(pi u)``, which piles points up near the ends of each interval in exactly the
    density the Chebyshev polynomials are orthogonal under -- the same density the Lobatto nodes
    of a collocation grid have, and for the same reason: it is what keeps the Lebesgue constant,
    and hence the fitted coefficients, from blowing up near the edges of the box.

    ``'samples'`` is not a map of the cube at all and never reaches here: it draws the pool from
    the distribution the Space was measured from. See :class:`PolynomialEngine`.
    """
    if measure not in ('chebyshev', 'uniform'):
        raise ValueError(f'draw_measure fills the box; {measure!r} does not come from it')
    unit = np.asarray(unit, dtype='f8')
    low, high = domains[:, 0], domains[:, 1]
    if measure == 'uniform':
        return low + unit * (high - low)
    return 0.5 * (low + high) - 0.5 * (high - low) * np.cos(np.pi * unit)


#: Samples per basis term below which :meth:`PolynomialEngine.nodes` selects by pivoted QR rather
#: than taking the pool in order, under ``selection='auto'``. The crossover is measured, and it is
#: a real trade in both directions -- see :func:`fekete_selection`.
FEKETE_BELOW = 1.25


def fekete_selection(design, nsamples):
    """Indices of the ``nsamples`` rows of *design* that condition the basis best.

    Pivoted QR on ``design.T`` orders the rows by how much each adds to the volume spanned by
    those already chosen -- the greedy maximum-volume, or approximate Fekete, selection. It picks
    the points that make the basis observable, which for a Chebyshev basis over a box means
    something close to a Lobatto grid, and over a box with a corner cut out of it the best
    available substitute for one.

    A single pass yields at most ``nterms`` useful pivots: past that the residual is numerically
    zero and the ordering is arbitrary. So oversampling is done in *rounds* -- each round runs the
    same selection over the points not yet taken -- which gives every extra sample the same claim
    to being well placed as the first ``nterms``, rather than whatever LAPACK happened to leave
    at the tail of one permutation.

    Worth using only where conditioning is what binds, and the measurement is sharp about where
    that is. On a 6-parameter box with a fifth of it removed, against the same valid pool taken in
    Sobol order (median over 8 seeds, error as median / 90th of the max relative error):

    ==========  =====  ==================  ==================
    basis       x      pivoted QR (cond)   Sobol order (cond)
    ==========  =====  ==================  ==================
    84 terms     1.05  9.3e-4      (43.6)  9.9e-4     (148.3)
    84 terms     1.20  8.2e-4      (37.8)  5.4e-4      (48.1)
    84 terms     2.00  5.4e-4      (21.6)  3.2e-4      (19.9)
    210 terms    1.05  1.5e-4     (153.3)  1.4e-4     (238.5)
    210 terms    2.00  9.2e-5      (55.9)  5.4e-5      (49.2)
    ==========  =====  ==================  ==================

    The conditioning gain is real and large where the fit is barely determined -- 3.4x at 1.05
    samples per term -- and it has vanished by 1.5, where extra samples condition the design on
    their own. The error tells the other half: maximum volume drives points towards the boundary
    of the region, which is what conditions a basis and is *not* where the error should be spent
    once there are samples to spare. Past the crossover it costs a consistent 1.6x.
    """
    from scipy.linalg import qr

    design = np.asarray(design, dtype='f8')
    npoints, nterms = design.shape
    nsamples = min(int(nsamples), npoints)
    remaining, chosen = np.arange(npoints), []
    while len(chosen) < nsamples and len(remaining):
        take = min(nsamples - len(chosen), nterms, len(remaining))
        _, _, pivots = qr(design[remaining].T, pivoting=True, mode='economic')
        picked = remaining[pivots[:take]]
        chosen.extend(int(index) for index in picked)
        remaining = np.setdiff1d(remaining, picked)
    return np.array(chosen, dtype='i8')


class PolynomialEngine(LinearBasisEngine):
    """Least-squares polynomial regression over a declared basis and chosen points.

    Parameters
    ----------
    order : int, dict, default=3
        Maximum degree per parameter; a dict may carry a ``'*'`` default. Anisotropic, and this
        is the cheapest knob there is: an axis given 1 loses every higher term and the samples
        those terms alone needed.
    budget : int, default=None
        The interaction cut, ``max(order)`` by default. See :func:`~.utils.multi_index_set`; with
        ``interaction='hyperbolic'`` this is what makes the basis sparse in the parameter count.
    interaction : str, default='total'
        ``'total'``, ``'hyperbolic'`` or ``'tensor'``. Total degree by default, on measurement
        rather than on the dimensional argument: on a 6-parameter box it beat the hyperbolic cross
        by 6-7x in median error at matched cost (7.9e-4 against 4.8e-3 at ~100 calls, 5.0e-4
        against 3.7e-3 at ~160), at comparable term counts. The cross spends its budget on high
        pure degrees and drops the low-order interactions; a cosmology target has those -- the
        growth couples ``w0`` and ``wa``, the transfer function couples ``omega_cdm``,
        ``omega_b``, ``h`` and ``n_s`` -- so dropping them is what costs. Reach for
        ``'hyperbolic'`` when total degree stops being payable, which at degree 3 is past roughly
        a dozen parameters (286 terms at 10, 816 at 15, against 76 and 151).
    basis : str, default='chebyshev'
        A key of :data:`~.utils.BASES`. ``'legendre'`` with ``measure='uniform'`` is the
        equivalent pairing; neither is more accurate than the other for a smooth target, and the
        default is Chebyshev so that a fitted engine is the same object the grid produces.
    oversampling : float, default=2.
        Samples per basis term, and the knob that buys tolerance to a hole: at 1 the fit is
        interpolatory and a lost sample is again fatal, while at 2 it takes half the node set to
        make it underdetermined. It is bought with accuracy, not for free -- against a fixed
        evaluation budget, a larger basis at 1.2 beats a smaller one at 3 by ~4x (1.2e-4 at 252
        calls against 4.4e-4 at the same). So raise it for a box whose corners the calculator
        refuses, and lower it for one it does not. Ignored when ``nsamples`` is given.
    nsamples : int, default=None
        The sample count outright, overriding ``oversampling`` -- for a fixed evaluation budget.
    ridge : float, default=0.
        Tikhonov regularisation, *relative*: the solve is damped by ``ridge * s_max**2``, so it is
        scale free and ``0`` is a plain least-squares solve. Small values (``1e-10`` to ``1e-6``)
        buy a well-behaved answer where the surviving points do not quite determine every term --
        at the cost of a biased one, which is why it is off by default and why a rank deficiency
        is reported rather than silently absorbed.
    measure : str, default=None
        How the candidate pool is drawn, a member of :data:`MEASURES`. ``'chebyshev'`` and
        ``'uniform'`` fill the box; they default to whichever ``basis`` is orthogonal under (see
        :data:`MEASURE_FOR_BASIS`), the pairing that conditions a least-squares fit best, and the
        choice between them was not measurable -- they landed within 25% of each other, inside
        the spread across seeds.

        ``'samples'`` is the one that matters, and it draws the pool from :attr:`samples` -- the
        chain the Space was measured from -- rather than from the box. Prefer it whenever a chain
        is available. Filling a box is the wrong instinct in more than about four dimensions:
        almost all of its volume is in the corners, so a low-discrepancy fill spends nearly every
        evaluation where the posterior has no support, and the fit trades away the centre to
        serve them. A sparse grid over the same box does not have this problem -- most of its
        sub-grids are level 0 along most axes, so its nodes pile up on the centre and the axes,
        which is a real part of why it wins where it wins.

        Measured against the production 8-parameter CMB emulator -- a budget-3 Chebyshev grid on
        a chain-derived box, 817 CAMB calls -- as the worst over TT/EE/TE of
        ``rms(dCl) / rms(Cl)``, median over 150 draws:

        ==================================  =====  ==============  ============
        fit                                 calls  on chain draws  on box draws
        ==================================  =====  ==============  ============
        chebyshev grid, budget 3            817    4.5e-4          8.1e-4
        polynomial, 45 terms, box-drawn      90    2.3e-3          3.0e-3
        polynomial, 45 terms, chain-drawn    90    1.3e-4          7.9e-4
        polynomial, 165 terms, chain-drawn  330    7.3e-5          9.3e-4
        ==================================  =====  ==============  ============

        Nine times fewer calls than the grid for 3.6x the accuracy where the posterior is, and no
        worse out in the box -- which is the column that had to be checked, since :meth:`outside`
        still declares the whole box. Note also that box-drawn is worse than chain-drawn even on
        box draws: filling that volume is not merely wasteful, it is a bad design.
    samples : array, default=None
        ``(nsamples, nparams)`` in the expansion variable, as a
        :class:`~.space.Space` stores them -- required by ``measure='samples'``, and supplied
        automatically by :class:`~.emulate.Emulator` when the Space carries a chain. Not
        serialised: like ``valid``, it shapes the node set and has no part in a prediction.
    selection : str, default='auto'
        Which of the valid candidates to spend the evaluations on. ``'fekete'`` picks them by
        pivoted QR on the Vandermonde (approximate Fekete points, best conditioning); ``'pool'``
        takes the pool in its own low-discrepancy order. ``'auto'`` chooses by
        :data:`FEKETE_BELOW`: Fekete while the fit is near-interpolatory and conditioning is what
        binds, the pool once there are samples enough to condition the design on their own.
        :func:`fekete_selection` carries the measurement the crossover comes from -- the trade
        runs in both directions, and neither choice is right everywhere.
    candidates : int, default=None
        Size of the candidate pool, ``20`` per basis term by default (at least 2000, at most
        200000). Candidates are free -- the pool is scored, not evaluated -- so this only needs
        to be large enough that the selection has something to choose between, and generous
        enough that a rejected region still leaves a well-conditioned subset.
    valid : callable, default=None
        ``valid(**params) -> bool`` over the *physical* parameters, filtering the pool before any
        point is selected. Called once with arrays, and per row if that raises or returns the
        wrong shape. This is the cheap physicality check -- ``w0_fld + wa_fld < 0``, a positive
        neutrino mass, whatever the calculator refuses -- and running it here rather than
        discovering the refusal after the fact is what turns a fifth of the box from wasted
        evaluations into no evaluations at all.

        Not serialised: it shapes the node set and has no part in a prediction. Through
        :class:`~.emulate.Emulator`, pass it to :meth:`~.emulate.Emulator.train` (or
        :meth:`~.emulate.Emulator.nodes`) rather than to the constructor -- the constructor's
        options are kept and written into the saved state, and a Python callable has no HDF5
        representation, so a trained emulator would refuse to be written.
    seed : int, default=42
        Seeds the Sobol pool. The selection itself is deterministic given the pool.
    levels : dict, default=None
        Accepted and ignored: the sparse grid's knob, passed through by the emulator so that
        swapping engines needs no other change.

    See :class:`~.engines.BaseEngine` for the geometry arguments.

    Notes
    -----
    Coverage is the box and no more -- :meth:`~.engines.BaseEngine.outside` is inherited as it
    stands, and ``valid`` does not enter it. That is deliberate: ``valid`` is a property of the
    calculator, not of the fit, so a point it rejects is one where the truth does not exist,
    nothing downstream has a reason to ask, and the likelihood that would consume the answer
    rejects it for itself. Making it part of the guard would also mean carrying an arbitrary
    Python predicate through the saved state, and a reloaded emulator either answering differently
    or dropping the guard in silence.

    What is worth knowing is that the fit *is* extrapolating across that region -- a polynomial
    has no notion of a hole and answers confidently over one. Keep the hole a region nothing asks
    about; do not read this engine as having covered it.
    """
    name = 'polynomial'
    logger = logging.getLogger('PolynomialEngine')

    #: A fit, not an interpolant: a sample the calculator could not evaluate is dropped from the
    #: design matrix and costs that sample alone. The premise of the whole engine.
    requires_all_nodes = False

    #: ``measure='samples'`` draws its pool from the chain, so the Emulator hands it over.
    wants_samples = True

    def __init__(self, params, limits, order=3, budget=None, interaction='total',
                 basis='chebyshev', oversampling=2., nsamples=None, ridge=0., measure=None,
                 selection='auto', candidates=None, valid=None, samples=None, seed=42,
                 levels=None, **kwargs):
        super().__init__(params, limits, **kwargs)
        self.order = {name: int(value) for name, value
                      in expand_dict(order, self.params, 'order').items()}
        if any(value < 0 for value in self.order.values()):
            raise ValueError(f'order must be non-negative; got {self.order}')
        self.budget = None if budget is None else int(budget)
        self.interaction = str(interaction)
        # Built once here so a typo is a construction error rather than a surprise after the node
        # set has been generated, or -- worse -- after the calculator has been paid for.
        multi_index_set([1] * len(self.params), interaction=self.interaction)
        if basis not in BASES:
            raise ValueError(f'unknown basis {basis!r}; available {sorted(BASES)}')
        self.basis = basis
        self.measure = MEASURE_FOR_BASIS[basis] if measure is None else str(measure)
        if self.measure not in MEASURES:
            raise ValueError(f'unknown measure {self.measure!r}; available {list(MEASURES)}')
        if selection not in ('auto', 'fekete', 'pool'):
            raise ValueError(f"selection must be 'auto', 'fekete' or 'pool'; got {selection!r}")
        self.selection = selection
        self.oversampling = float(oversampling)
        self.nsamples = None if nsamples is None else int(nsamples)
        self.ridge = float(ridge)
        self.candidates = None if candidates is None else int(candidates)
        self.valid = valid
        self.samples = None if samples is None else np.asarray(samples, dtype='f8')
        if self.measure == 'samples' and self.samples is None:
            raise ValueError("measure='samples' needs the samples the space was measured from; "
                             "pass `samples=`, or give the Emulator a Space carrying a chain.")
        self.seed = int(seed)
        self.powers = self.coefficients = self.domains = None
        # Set by `fit`, and reported: how many of the planned samples came back usable, and how
        # far the fit sits from the values it was given. A regression has no exactness to check,
        # so this residual is the only thing that says the basis was rich enough.
        self.ndropped, self.residual = 0, None

    @property
    def nterms(self):
        return len(multi_index_set([self.order[name] for name in self.params],
                                   budget=self.budget, interaction=self.interaction))

    # ── nodes ─────────────────────────────────────────────────────────────────
    def _valid_mask(self, physical):
        """Which rows of *physical* the ``valid`` predicate keeps.

        Vectorised first -- ``lambda w0_fld, wa_fld: w0_fld + wa_fld < 0`` is already an array
        expression, and the pool is large enough that a Python loop over it is the slow part of
        building an emulator that hasn't run the Boltzmann code yet. Falls back to a row loop when
        the predicate is not written that way, rather than making the caller declare which it is.
        """
        if self.valid is None:
            return np.ones(len(physical), dtype='?')
        columns = {name: physical[:, index] for index, name in enumerate(self.params)}
        try:
            mask = np.asarray(self.valid(**columns), dtype='?')
            if mask.shape != (len(physical),):
                raise ValueError
        except Exception:
            mask = np.array([bool(self.valid(**dict(zip(self.params, row))))
                             for row in physical], dtype='?')
        return mask

    def nodes(self):
        """The points to evaluate the calculator at: an ``(nsamples, nparams)`` array, physical.

        Pool, filter, select -- in that order, which is the order that keeps the expensive step
        last. The pool is quasi-random (Sobol, mapped through :func:`draw_measure`) rather than a
        lattice, so the filter can remove any part of it without leaving a hole anything depends
        on, and only the survivors are ever handed to the calculator. What :attr:`selection` then
        decides is which of them to spend the evaluations on -- see :func:`fekete_selection` for
        why that is a trade rather than an improvement.
        """
        from scipy.stats import qmc

        # The multi-index set comes from the declared knobs alone, no node values, so `nodes`
        # and `fit` each build it rather than sharing state through the instance -- they run on
        # *different* instances, `Emulator.train` making one engine for the node set and a fresh
        # one per output to fit it. `box` is (nparams, 2), in the coordinates the basis is
        # evaluated in.
        powers = multi_index_set([self.order[name] for name in self.params],
                                 budget=self.budget, interaction=self.interaction)
        box = np.array([self._domain(name) for name in self.params])
        nterms = len(powers)
        # A declared `nsamples` wins; otherwise `oversampling` sets it, never below one sample
        # per basis term, which is where the least-squares problem stops being determined.
        nsamples = max(self.nsamples, 1) if self.nsamples is not None \
            else max(int(np.ceil(self.oversampling * nterms)), nterms)
        npool = self.candidates if self.candidates is not None \
            else int(np.clip(20 * nterms, 2000, 200000))
        # Rounded up to a power of two: Sobol' is only balanced on such counts, and warns
        # otherwise. Free to honour, since a candidate is scored and never evaluated.
        npool = 2 ** int(np.ceil(np.log2(max(npool, nsamples, 1))))

        if self.measure == 'samples':
            # A sample is in the expansion variable: transformed already, and not whitened. So
            # the two coordinates it has to reach are each half of the usual round trip --
            # whiten it for the interpolant's own coordinate, invert the transform for the
            # calculator's parameter -- and `_internal` / `_physical`, which each do both
            # halves, are the wrong maps here. Putting a chain sample through `_physical`
            # unwhitens a point that was never whitened: measured on the production CMB box,
            # that alone took the fit from 2e-3 to a median error of 159.
            # Shuffled, because a chain's rows are serially correlated -- taking the first
            # `npool` of them is one stretch of one walker, not a sample of the posterior.
            take = min(int(npool), len(self.samples))
            index = np.random.default_rng(self.seed).choice(len(self.samples), take, replace=False)
            rows = self.samples[index]
            pairs = [self._transform_pair(name) for name in self.params]
            physical = np.array([[float(pair[1](value)) if pair else float(value)
                                  for value, pair in zip(row, pairs)] for row in rows])
            internal = np.array([self.whiten(row) for row in rows]) if self.whitened else rows
        else:
            unit = qmc.Sobol(d=len(self.params), scramble=True, seed=self.seed).random(npool)
            internal = draw_measure(unit, box, self.measure)
            physical = np.array([self._physical(row) for row in internal])
        npool = len(internal)

        keep = self._valid_mask(physical)
        nvalid = int(keep.sum())
        if nvalid < nterms:
            raise ValueError(
                f'`valid` keeps {nvalid} of {npool} candidates, fewer than the {nterms} basis '
                f'terms, so no choice of points can determine the fit. Raise `candidates`, or -- '
                f'if the predicate is rejecting most of the box -- move the box rather than '
                f'fitting a basis over a region that is mostly not there.')
        if nvalid < npool:
            self.logger.info(f'`valid` keeps {nvalid}/{npool} candidates ({nvalid / npool:.1%}); '
                             f'selecting {nsamples} of them')
        internal, physical = internal[keep], physical[keep]

        selection = self.selection
        if selection == 'auto':
            selection = 'fekete' if nsamples < FEKETE_BELOW * nterms else 'pool'
        if selection == 'pool':
            # The pool's own order, which is a low-discrepancy sequence and so already spreads
            # over whatever survived the filter. Truncating it is the whole operation.
            return physical[:nsamples]
        design = np.asarray(tensor_basis(internal.T, powers, box, basis=self.basis)).T
        chosen = fekete_selection(design, nsamples)
        if len(chosen) < nsamples:
            self.logger.warning(f'only {len(chosen)} of {nsamples} points could be selected')
        return physical[chosen]

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, inputs, outputs, weights=None):
        """``inputs``: (nsamples, nparams), physical. ``outputs``: (nsamples, noutputs).

        Takes the points as given: they need not be the ones :meth:`nodes` proposed, need not be
        all of them, and need not lie on anything. Rows whose output is not finite are dropped
        here as well as upstream, so an engine used directly behaves like one used through
        :class:`~.emulate.Emulator`.

        ``weights`` scales the residual of each sample, for the case where some are known better
        than others.
        """
        inputs = np.asarray(inputs, dtype='f8')
        outputs = np.asarray(outputs, dtype='f8')
        # A column, not a row: `atleast_2d` on a one-output-per-sample array would make it
        # (1, nsamples) and the fit would silently be over one sample of nsamples outputs.
        if outputs.ndim == 1:
            outputs = outputs[:, None]
        if len(inputs) != len(outputs):
            raise ValueError(f'{len(inputs)} inputs against {len(outputs)} outputs')
        powers = multi_index_set([self.order[name] for name in self.params],
                                 budget=self.budget, interaction=self.interaction)
        box = np.array([self._domain(name) for name in self.params])
        nterms = len(powers)

        internal = np.array([self._internal(row) for row in inputs])
        design = np.asarray(tensor_basis(internal.T, powers, box, basis=self.basis)).T

        finite = np.isfinite(outputs).all(axis=1) & np.isfinite(design).all(axis=1)
        self.ndropped = int((~finite).sum())
        design, outputs = design[finite], outputs[finite]
        if weights is not None:
            weights = np.asarray(weights, dtype='f8')[finite]
            if np.any(weights < 0.):
                raise ValueError('weights must be non-negative')
            root = np.sqrt(weights)[:, None]
            design, outputs = design * root, outputs * root

        nsamples = len(design)
        if nsamples < nterms:
            raise ValueError(
                f'{nsamples} usable samples against {nterms} basis terms: the fit is '
                f'underdetermined. Lower `order` or `budget` so the basis is smaller, or raise '
                f'`oversampling` so more points are evaluated -- with this many free directions a '
                f'solve would invent structure rather than measure it.')

        # SVD rather than `lstsq`: the rank is wanted as a diagnostic in its own right, and the
        # ridge is a filter on the singular values, which is where it belongs.
        left, singular, right = np.linalg.svd(design, full_matrices=False)
        largest = float(singular.max()) if singular.size else 0.
        rank = int((singular > largest * max(nsamples, nterms) * np.finfo('f8').eps).sum())
        damping = self.ridge * largest ** 2
        if not damping and rank < nterms:
            raise ValueError(
                f'the usable samples span only {rank} of {nterms} basis directions '
                f'({(nterms - rank) / nterms:.1%} unconstrained). Lower `order` or `budget` so '
                f'the basis is smaller, raise `oversampling`, or set a small `ridge` if the '
                f'unconstrained directions are ones you accept damping to zero.')
        filtered = singular / (singular ** 2 + damping)
        self.coefficients = right.T @ (filtered[:, None] * (left.T @ outputs))
        self.powers, self.domains = powers, box

        # The residual is the whole quality statement a regression can make: it is exact nowhere,
        # so a fit that misses its own samples badly has a basis too small for the target, and
        # nothing further downstream will say so.
        predicted = design @ self.coefficients
        scale = float(np.abs(outputs).max())
        self.residual = float(np.sqrt(np.mean((predicted - outputs) ** 2)) / (scale or 1.))
        condition = largest / float(singular.min()) if singular.min() > 0 else np.inf
        self.logger.info(
            f'least-squares fit on {nsamples}'
            + (f' (+{self.ndropped} dropped)' if self.ndropped else '')
            + f' samples, {nterms} terms, condition number {condition:.1f}, '
              f'relative rms residual {self.residual:.2e}')
        return self

    # ── state ─────────────────────────────────────────────────────────────────
    def __getstate__(self):
        state = self._geometry_state()
        state.update({'order': dict(self.order), 'budget': self.budget,
                      'interaction': self.interaction, 'basis': self.basis,
                      'measure': self.measure, 'selection': self.selection,
                      'oversampling': self.oversampling,
                      'nsamples': self.nsamples, 'ridge': self.ridge,
                      'candidates': self.candidates, 'seed': self.seed,
                      'ndropped': self.ndropped, 'residual': self.residual,
                      'powers': self.powers, 'coefficients': self.coefficients,
                      'domains': self.domains})
        return state

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new._set_geometry(state)
        new.order, new.budget = dict(state['order']), state['budget']
        new.interaction, new.basis = state['interaction'], state['basis']
        new.measure, new.oversampling = state['measure'], state['oversampling']
        new.selection = state['selection']
        new.nsamples, new.ridge = state['nsamples'], state['ridge']
        new.candidates, new.seed = state['candidates'], state['seed']
        # `.get` with the old spelling: this counter was `n_dropped` in states written before the
        # rename, and a saved emulator outlives the code that wrote it. Diagnostics only -- it
        # records how many samples the fit dropped and has no part in a prediction -- so an older
        # file reads back correctly rather than dying on a key that changes nothing.
        new.ndropped = state.get('ndropped', state.get('n_dropped', 0))
        new.residual = state['residual']
        new.powers, new.coefficients = state['powers'], state['coefficients']
        new.domains = state['domains']
        # Not serialisable, and not needed to predict: it only ever shaped the node set.
        new.valid = None
        return new
