"""A multi-layer perceptron engine, for when the sparse grid runs out of room.

The Chebyshev engine is the better choice whenever it fits: its node count is deterministic, its
levels are nested so raising the budget reuses every evaluation, and it needs no training beyond a
linear solve. What it cannot do is many parameters -- a sparse grid still grows with dimension,
and past roughly a dozen axes the node count stops being payable. An MLP does not care about
dimension in the same way: it takes a fixed pile of quasi-random samples and fits.

So the trade is: grid for few parameters and exactness, network for many parameters and a
stochastic fit. Both present the same interface, so ``train(engine='mlp')`` is the only change.

Written directly in JAX and optax -- no flax. The network is a plain list of ``(weight, bias)``
arrays, which is what makes the state a handful of arrays that HDF5 can hold and ``predict`` a
few matrix products that jit like anything else.
"""

import logging

import numpy as np

from cosmoprimo.jax import numpy_jax

from .engines import BaseEngine


ACTIVATIONS = ('silu', 'tanh', 'relu')


def _activate(x, name, xnp):
    if name == 'silu':
        return x / (1. + xnp.exp(-x))
    if name == 'tanh':
        return xnp.tanh(x)
    if name == 'relu':
        return xnp.maximum(x, 0.)
    raise ValueError(f'unknown activation {name!r}; available {list(ACTIVATIONS)}')


class MLPEngine(BaseEngine):
    """Dense network over quasi-random samples of the box.

    Parameters
    ----------
    nsamples : int, default=None
        How many points to evaluate the calculator at. ``512 * nparams`` by default, which is
        a starting point rather than a recommendation -- unlike the grid, there is no node count
        at which this becomes exact, so validate.
    nhidden : tuple, default=(64, 64, 64)
        Hidden layer widths.
    activation : str, default='silu'
        One of :data:`ACTIVATIONS`.
    epochs, patience, learning_rate, batch_size, validation_frac, optimizer, seed
        Training schedule. ``patience`` stops early once the validation loss has not improved for
        that many epochs, which is what keeps a long ``epochs`` from being wasted.
    lr_decay : float, default=1.
        Ratio of the learning rate at the last epoch to ``learning_rate``: the rate decays
        exponentially over ``epochs`` (all of them, whether or not early stopping cuts the run
        short), and 1 keeps it constant. A fixed rate leaves the fit bouncing around the optimum
        at a size set by that rate; 1e-2 or so is what takes a percent-level fit to its last
        factor of a few.
    valid : callable, default=None
        A predicate over the physical parameters, called by name
        (``valid(w0_fld=..., wa_fld=...)``, vectorised or not), that says where the calculator
        can be evaluated. The Sobol pool is filtered through it before anything is evaluated,
        so no sample is spent where the truth does not exist -- and, more to the point, the
        node set is then a sample of the *valid* region rather than of the box. Dropping the
        failures afterwards (the engine's tolerance for non-finite nodes) is the right tool for
        a stray refusal; it is the wrong one for a box that is 70% hole, since
        :meth:`~.emulate.Emulator.train` refuses a training that loses more than
        ``max_non_finite`` of its nodes, and it should. As for the polynomial engine, ``valid``
        is a property of the calculator and not of the fit: it is not saved with the state, and
        the fit *is* extrapolating across the region it excludes -- keep that region one nothing
        asks about (a prior that vetoes it).
    asinh_quantile : float, default=0.5
        Which quantile of |y| over the samples sets the asinh scale of each component. asinh is
        linear below its scale and logarithmic above, so the scale is where relative accuracy
        stops being resolved: with the median (the default), every sample below the median is
        fitted to an absolute error, and an output spanning six decades (a growth scalar from
        1e-4 to 300 over the EFT-of-DE box) is unresolved on its lower half. A low quantile
        (0.01) puts 99% of the samples in the logarithmic regime, at the price of an asinh range
        as wide as the output's dynamic range in decades. Applied to sign-definite components
        only; a component that changes sign over the samples keeps the median as its scale.
    output_transform : str, default='none'
        ``'asinh'`` fits ``asinh(y / s)`` per output component instead of ``y``, with ``s`` the
        median |y| of that component over the samples (its typical size), and puts ``sinh`` back
        at prediction. Sign-preserving and linear for |y| << s, logarithmic beyond: it is what
        lets one network cover an output whose value spans many decades across the box. The
        plain standardisation cannot: it makes the network fit ``y`` to a fixed fraction of the
        component's SPREAD, and where that spread is dominated by a few huge samples the small
        ones are lost entirely. Measured on the one-loop tables of a full-shape emulator over an
        11-parameter box (sigma8 from 0.04 to > 1): raw columns spanning 1e18 fitted to 3% of
        their standard deviation, i.e. relative errors of 1e7 and worse at the small end;
        see the tests. Not compatible with :meth:`contract` (no longer affine after the last
        layer).
    candidates : int, default=None
        Initial size of the pool ``valid`` filters, ``4 * nsamples`` by default (rounded up to a
        power of two, on which Sobol' is balanced); it doubles on its own until ``nsamples``
        survive, up to :attr:`MAX_CANDIDATES`.

    ``levels`` and ``budget`` are accepted and ignored: they are the sparse grid's knobs, and are
    passed through by the emulator so that swapping engines needs no other change.
    """
    name = 'mlp'
    logger = logging.getLogger('MLPEngine')

    #: A fit, not an interpolant: a sample the calculator could not evaluate can simply be left
    #: out, costing that sample alone. This is what lets an MLP cover a box containing a region
    #: where the truth does not exist -- the packaged ACE networks span w0_fld in (-3, 0.5) and
    #: wa_fld in (-3, 2), roughly 18% of which is the unphysical w0 + wa > 0 corner, and no
    #: collocation grid can be laid over that.
    requires_all_nodes = False

    #: Cap on the candidate pool `valid` filters (see `nodes`): 2^22, about four million.
    MAX_CANDIDATES = 2 ** 22

    def __init__(self, params, limits, levels=None, budget=None, nsamples=None,
                 nhidden=(64, 64, 64), activation='silu', epochs=2000, patience=200,
                 learning_rate=1e-3, batch_size=64, validation_frac=0.1, optimizer='adam',
                 seed=42, valid=None, candidates=None, output_transform='none', lr_decay=1.,
                 asinh_quantile=0.5, fit_seed=None, loss='mse', huber_delta=1., **kwargs):
        super().__init__(params, limits, **kwargs)
        if output_transform not in ('none', 'asinh'):
            raise ValueError(f"output_transform must be 'none' or 'asinh'; got {output_transform!r}")
        self.output_transform = str(output_transform)
        self.asinh_quantile = float(asinh_quantile)
        if not 0. < self.asinh_quantile <= 1.:
            raise ValueError(f'asinh_quantile must be in (0, 1]; got {asinh_quantile}')
        self._asinh_scale = None
        self.nsamples = int(nsamples) if nsamples is not None else 512 * len(self.params)
        self.valid = valid
        self.candidates = None if candidates is None else int(candidates)
        self.nhidden = tuple(int(width) for width in nhidden)
        if activation not in ACTIVATIONS:
            raise ValueError(f'unknown activation {activation!r}; available {list(ACTIVATIONS)}')
        self.activation = activation
        self.epochs, self.patience = int(epochs), int(patience)
        self.learning_rate, self.batch_size = float(learning_rate), int(batch_size)
        self.lr_decay = float(lr_decay)
        if not 0. < self.lr_decay <= 1.:
            raise ValueError(f'lr_decay is the ratio of the final to the initial learning rate, in (0, 1]; got {lr_decay}')
        self.validation_frac, self.optimizer = float(validation_frac), str(optimizer)
        self.seed = int(seed)
        # `seed` draws the nodes AND seeds the fit (the validation split, the initial weights,
        # the batch order), so changing it changes the training set. `fit_seed` reseeds the fit
        # alone: the same nodes, refitted from different weights -- which is how the fit-to-fit
        # variance of a recipe is measured, and without that number no comparison between two
        # recipes fitted once each means anything (two 4x512 fits of the EFT-of-DE emulator on
        # node sets differing by 3% landed 2-5x apart in every output group, 2026-09-12).
        self.fit_seed = None if fit_seed is None else int(fit_seed)
        # The training loss, on the standardised (and asinh-transformed, if asked) targets.
        # 'mse' is the plain mean square. 'huber' is quadratic up to `huber_delta` standard
        # deviations of residual and linear beyond, so a node the network cannot fit stops
        # dominating the gradient. Measured on the EFT-of-dark-energy emulator (2026-09-12, two
        # seeds of one 4x512 recipe): the worst 1% of the held-out nodes carried 60-99% of each
        # network's squared error, the held-out rms was 3-4x the training rms, and the median
        # error of the sigma8 network differed 5x between the seeds -- the fit is steered by a
        # tail of wild models (large c_M, w0 > 0) at the expense of the bulk, and which way it
        # is steered is a draw.
        if loss not in ('mse', 'huber'):
            raise ValueError(f"loss must be 'mse' or 'huber'; got {loss!r}")
        self.loss, self.huber_delta = str(loss), float(huber_delta)
        if not self.huber_delta > 0.:
            raise ValueError(f'huber_delta must be positive; got {huber_delta}')
        self.layers = None
        self._output_mean = self._output_scale = None

    # ── nodes ─────────────────────────────────────────────────────────────────
    def nodes(self):
        """Quasi-random samples of the box, in physical parameters.

        Sobol rather than uniform random: a low-discrepancy sequence covers the box far more
        evenly at the same count, and the count here is the entire cost. not nested the way the
        grid's levels are -- raising ``nsamples`` means evaluating a fresh set, so pick it once.

        With ``valid``, a larger pool is drawn, filtered, and the first ``nsamples`` survivors
        kept -- the pool's own order is low-discrepancy, so truncating it is the whole
        operation, as in the polynomial engine. Without it the pool *is* the node set, so
        nothing changes for an engine built without a predicate.
        """
        from scipy.stats import qmc
        from .engines import valid_mask

        dimension = len(self.params)

        def draw(npool):
            unit = qmc.Sobol(d=dimension, scramble=True, seed=self.seed).random(npool)
            if self.whitened:
                internal = (2. * unit - 1.) * self.nsigma
                return np.array([self.unwhiten(row) for row in internal])
            low = np.array([self._domain(name)[0] for name in self.params])
            high = np.array([self._domain(name)[1] for name in self.params])
            internal = low + unit * (high - low)
            return np.array([self._physical(row) for row in internal])

        if self.valid is None:
            return draw(self.nsamples)
        # The pool grows until enough survive, doubling from `candidates` (4 x nsamples by
        # default) up to MAX_CANDIDATES. A Sobol' sequence with a fixed seed is nested -- a
        # larger draw starts with the same points -- so the survivors of a larger pool begin with
        # those of a smaller one and the node set is stable under the growth. Measured need: the
        # EFT-of-dark-energy box is 11% valid, so 5632 samples take ~ 65000 candidates; the
        # predicate costs ~0.06 ms per candidate, so a million of them is a minute.
        npool = self.candidates if self.candidates is not None else 4 * self.nsamples
        npool = 2 ** int(np.ceil(np.log2(max(npool, self.nsamples, 1))))
        while True:
            physical = draw(npool)
            keep = valid_mask(self.valid, self.params, physical)
            nvalid = int(keep.sum())
            if nvalid >= self.nsamples or npool >= self.MAX_CANDIDATES:
                break
            self.logger.info(f'`valid` keeps {nvalid}/{npool} candidates, fewer than the {self.nsamples} '
                             f'samples asked for; doubling the pool')
            npool *= 2
        if nvalid < self.nsamples:
            raise ValueError(
                f'`valid` keeps {nvalid} of {npool} candidates, fewer than the {self.nsamples} '
                f'samples asked for, and the pool is at its cap ({self.MAX_CANDIDATES}). The '
                f'predicate is rejecting almost all of the box: move the box rather than fitting '
                f'over a region that is mostly not there.')
        self.logger.info(f'`valid` keeps {nvalid}/{npool} candidates ({nvalid / npool:.1%}); '
                         f'taking the first {self.nsamples}')
        return physical[keep][:self.nsamples]

    # ── fit ───────────────────────────────────────────────────────────────────
    def _standardise(self, outputs):
        """Zero mean, unit scale per output component.

        Not cosmetic: a network fits a residual of order one. Cl span decades across ell, and
        without this the loss is dominated by whichever component happens to be largest and the
        rest is never fitted at all.
        """
        mean = outputs.mean(axis=0)
        scale = outputs.std(axis=0)
        scale = np.where(scale == 0., 1., scale)     # a component that never varies
        return mean, scale

    def fit(self, inputs, outputs):
        """``inputs``: (nnodes, nparams), physical. ``outputs``: (nnodes, noutputs)."""
        import jax
        from jax import numpy as jnp
        import optax

        inputs = np.asarray(inputs, dtype='f8')
        outputs = np.asarray(outputs, dtype='f8')
        if len(inputs) != len(outputs):
            raise ValueError(f'{len(inputs)} inputs against {len(outputs)} outputs')
        internal = np.array([self._internal(row) for row in inputs])
        if self.output_transform == 'asinh':
            # The component's typical size: its median |y|. A component that is zero in more
            # than half the samples but not all falls back to its maximum, and one that is zero
            # everywhere to 1. The scale must NOT be tied to the maximum otherwise: a first
            # version floored it at 1e-12 x max|y|, meant for all-zero columns, and on a
            # training set holding a few absurd nodes (|y| up to 1e75 against a median of 50,
            # models the stability gate let through) that floor became the scale for 50 of 78
            # outputs -- asinh then linear over the whole sane range, the network fitted
            # y / 1e13 and predicted 1e68 (2026-09-10). Outliers are the training set's
            # problem to drop (see Emulator.train); the transform keeps the sane range.
            # Per component: a sign-definite one (never crossing zero over the samples) takes the
            # requested quantile, so a low quantile puts it in the logarithmic regime over most
            # of its range; a sign-changing one keeps the median. Near a zero crossing |y| is
            # legitimately tiny, a low quantile then makes the scale tiny too and the transform
            # nearly log|y| with a sign flip, which a network cannot fit: measured on a one-loop
            # table column, 3% of the row maximum at the 1st percentile against ~0.1% at the
            # median (2026-09-10).
            absolute = np.abs(outputs)
            definite = (outputs.min(axis=0) > 0.) | (outputs.max(axis=0) < 0.)
            quantile = np.where(definite, self.asinh_quantile, max(self.asinh_quantile, 0.5))
            scale = np.array([np.quantile(absolute[:, index], quantile[index]) for index in range(absolute.shape[1])])
            fallback = absolute.max(axis=0)
            self._asinh_scale = np.where(scale > 0., scale, np.where(fallback > 0., fallback, 1.))
            outputs = np.arcsinh(outputs / self._asinh_scale)
        self._output_mean, self._output_scale = self._standardise(outputs)
        targets = (outputs - self._output_mean) / self._output_scale

        seed = self.seed if self.fit_seed is None else self.fit_seed
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(inputs))
        nvalidation = max(int(len(inputs) * self.validation_frac + 0.5), 1)
        if nvalidation >= len(inputs):
            raise ValueError(f'{nvalidation} validation samples out of {len(inputs)}: train on '
                             f'more nodes, or lower validation_frac')
        validation, training = order[:nvalidation], order[nvalidation:]

        widths = (internal.shape[1],) + self.nhidden + (targets.shape[1],)
        key = jax.random.PRNGKey(seed)
        layers = []
        for index in range(len(widths) - 1):
            key, subkey = jax.random.split(key)
            # Glorot: keeps the activation variance from collapsing or blowing up with depth
            bound = np.sqrt(6. / (widths[index] + widths[index + 1]))
            layers.append((jax.random.uniform(subkey, (widths[index], widths[index + 1]),
                                              minval=-bound, maxval=bound, dtype=jnp.float64),
                           jnp.zeros(widths[index + 1], dtype=jnp.float64)))

        activation = self.activation

        def forward(layers, x):
            for weight, bias in layers[:-1]:
                x = _activate(x @ weight + bias, activation, jnp)
            weight, bias = layers[-1]
            return x @ weight + bias

        loss_name, delta = self.loss, self.huber_delta

        def loss_fn(layers, x, y):
            residual = forward(layers, x) - y
            if loss_name == 'huber':
                # 0.5 r^2 inside |r| <= delta, delta (|r| - 0.5 delta) outside; scaled by 2 so it
                # coincides with the mean square where every residual is small
                magnitude = jnp.abs(residual)
                quadratic = jnp.minimum(magnitude, delta)
                return jnp.mean(quadratic**2 + 2. * delta * (magnitude - quadratic))
            return jnp.mean(residual**2)

        x_train = jnp.asarray(internal[training])
        y_train = jnp.asarray(targets[training])
        x_validation = jnp.asarray(internal[validation])
        y_validation = jnp.asarray(targets[validation])
        batch = min(self.batch_size, len(training))
        # whole batches only: the ragged tail of a shuffled epoch is a different subset each
        # time, so nothing is systematically left out, and a fixed batch count is what lets the
        # whole epoch be one traced loop below
        nbatches = len(training) // batch
        schedule = self.learning_rate
        if self.lr_decay < 1.:
            schedule = optax.exponential_decay(self.learning_rate, transition_steps=nbatches * self.epochs,
                                               decay_rate=self.lr_decay)
        tx = getattr(optax, self.optimizer)(schedule)
        state = tx.init(layers)

        # One traced call per epoch: the gradient, the optimiser update and the parameter update
        # for every batch, scanned. The alternative -- a Python loop dispatching one jitted
        # gradient per batch, with the optax update eager between them -- costs a few ms of
        # overhead per batch whatever the device, so a GPU ran it no faster than a CPU core and a
        # 78-output emulator over 5632 samples took more than 5 h to fit. The data are arguments
        # of the jitted function, not constants closed over, so a large training set is not
        # baked into the executable (`step` closes over the traced arguments, which is fine).
        @jax.jit
        def epoch(layers, state, order, x_train, y_train, x_validation, y_validation):

            def step(carry, index):
                layers, state = carry
                grads = jax.grad(loss_fn)(layers, x_train[index], y_train[index])
                updates, state = tx.update(grads, state, layers)
                return (optax.apply_updates(layers, updates), state), None

            (layers, state), _ = jax.lax.scan(step, (layers, state), order)
            # the validation metric is the mean square whatever the training loss, so that the
            # early stopping, the best-state choice and the logged number mean the same thing
            # across recipes
            return layers, state, jnp.mean((forward(layers, x_validation) - y_validation)**2)

        best, best_loss, waited = layers, np.inf, 0
        iepoch = -1
        for iepoch in range(self.epochs):
            order = jnp.asarray(rng.permutation(len(training))[:nbatches * batch].reshape(nbatches, batch))
            layers, state, loss = epoch(layers, state, order, x_train, y_train, x_validation, y_validation)
            loss = float(loss)
            if not np.isfinite(loss):
                # A divergence (a non-finite loss) never recovers: the parameters are non-finite
                # from here on and every later epoch is wasted while `best` stays where it was.
                # Measured on a growth scalar at a 1e-3 rate decayed only 1e-4 over 8000 epochs:
                # the best epoch was ~300 and the remaining 7700 changed nothing. Stop, keep best.
                self.logger.warning(f'non-finite validation loss at epoch {iepoch + 1}; stopping with the best state '
                                    f'(epoch of best loss {best_loss:.3e})')
                break
            if loss < best_loss:
                # keep the best validation loss, not the last one: past the optimum the training
                # loss keeps falling while the prediction gets worse
                best, best_loss, waited = layers, loss, 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        self.layers = [(np.asarray(weight), np.asarray(bias)) for weight, bias in best]
        self.validation_loss = best_loss
        self.epochs_run = iepoch + 1
        # The one line that says whether the fit converged or stopped early, and how far it got:
        # a standardised validation MSE of 1e-4 is a relative error of ~1% on an asinh-transformed
        # output, 1e-6 is ~0.1%. Without it a 78-network job log said nothing about the fit.
        self.logger.info(f'{targets.shape[1]} components, {len(training)} samples: {self.epochs_run}/{self.epochs} epochs, '
                         f'best validation loss {best_loss:.3e} (standardised MSE)')
        return self

    # ── predict ───────────────────────────────────────────────────────────────
    def predict(self, values):
        """``values``: physical parameters, in :attr:`params` order."""
        if self.layers is None:
            raise ValueError('not fitted')
        xnp = numpy_jax(values)
        x = self._traced(values)
        for weight, bias in self.layers[:-1]:
            x = _activate(x @ xnp.asarray(weight) + xnp.asarray(bias), self.activation, xnp)
        weight, bias = self.layers[-1]
        x = x @ xnp.asarray(weight) + xnp.asarray(bias)
        x = x * xnp.asarray(self._output_scale) + xnp.asarray(self._output_mean)
        if self.output_transform == 'asinh':
            x = xnp.sinh(x) * xnp.asarray(self._asinh_scale)
        return x

    def contract(self, matrix):
        """Left-multiply the output by a fixed ``matrix``, exactly.

        The network is non-linear, but everything after its last layer is affine -- the last
        matmul and the output standardisation -- so ``M`` folds into it. The standardisation is
        absorbed at the same time (and reset), because ``M @ (y * scale + mean)`` is affine in
        the last layer's output but not a rescaling of ``scale`` alone.
        """
        if self.layers is None:
            raise ValueError('not fitted')
        if self.output_transform != 'none':
            raise ValueError(f'contract is exact only for an affine output; output_transform={self.output_transform!r} is not')
        matrix = np.asarray(matrix, dtype='f8')
        weight, bias = self.layers[-1]
        if matrix.shape[1] != weight.shape[1]:
            raise ValueError(f'matrix is {matrix.shape}, cannot act on an output of '
                             f'{weight.shape[1]}')
        scale, mean = self._output_scale, self._output_mean
        self.layers[-1] = ((weight * scale) @ matrix.T, (bias * scale + mean) @ matrix.T)
        self._output_scale = np.ones(matrix.shape[0])
        self._output_mean = np.zeros(matrix.shape[0])
        return self

    # ── state ─────────────────────────────────────────────────────────────────
    def __getstate__(self):
        state = self._geometry_state()
        state.update({'nsamples': self.nsamples, 'nhidden': self.nhidden,
                      'activation': self.activation, 'epochs': self.epochs,
                      'patience': self.patience, 'learning_rate': self.learning_rate,
                      'batch_size': self.batch_size, 'validation_frac': self.validation_frac,
                      'optimizer': self.optimizer, 'seed': self.seed, 'lr_decay': self.lr_decay,
                      'fit_seed': -1 if self.fit_seed is None else self.fit_seed,
                      'loss': self.loss, 'huber_delta': self.huber_delta,
                      'output_transform': self.output_transform, 'asinh_quantile': self.asinh_quantile,
                      'asinh_scale': self._asinh_scale if self._asinh_scale is not None else np.zeros(0),
                      'output_mean': self._output_mean, 'output_scale': self._output_scale,
                      'nlayers': 0 if self.layers is None else len(self.layers)})
        for index, (weight, bias) in enumerate(self.layers or []):
            state[f'weight.{index}'], state[f'bias.{index}'] = weight, bias
        return state

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new._set_geometry(state)
        for name in ('nsamples', 'nhidden', 'activation', 'epochs', 'patience', 'learning_rate',
                     'batch_size', 'validation_frac', 'optimizer', 'seed'):
            setattr(new, name, state[name])
        new.nhidden = tuple(new.nhidden)
        new.lr_decay = float(state.get('lr_decay', 1.))
        fit_seed = int(state.get('fit_seed', -1))
        new.fit_seed = None if fit_seed < 0 else fit_seed
        new.loss, new.huber_delta = str(state.get('loss', 'mse')), float(state.get('huber_delta', 1.))
        new.output_transform = str(state.get('output_transform', 'none'))
        new.asinh_quantile = float(state.get('asinh_quantile', 0.5))
        scale = state.get('asinh_scale', None)
        new._asinh_scale = None if scale is None or np.size(scale) == 0 else np.asarray(scale)
        new._output_mean, new._output_scale = state['output_mean'], state['output_scale']
        nlayers = int(state['nlayers'])
        new.layers = [(state[f'weight.{index}'], state[f'bias.{index}'])
                      for index in range(nlayers)] or None
        return new
