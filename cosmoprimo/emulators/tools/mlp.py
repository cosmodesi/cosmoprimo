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
        How many points to evaluate the calculator at. ``512 * n_params`` by default, which is
        a starting point rather than a recommendation -- unlike the grid, there is no node count
        at which this becomes exact, so validate.
    nhidden : tuple, default=(64, 64, 64)
        Hidden layer widths.
    activation : str, default='silu'
        One of :data:`ACTIVATIONS`.
    epochs, patience, learning_rate, batch_size, validation_frac, optimizer, seed
        Training schedule. ``patience`` stops early once the validation loss has not improved for
        that many epochs, which is what keeps a long ``epochs`` from being wasted.

    ``levels`` and ``budget`` are accepted and ignored: they are the sparse grid's knobs, and are
    passed through by the emulator so that swapping engines needs no other change.
    """
    name = 'mlp'

    def __init__(self, params, limits, levels=None, budget=None, nsamples=None,
                 nhidden=(64, 64, 64), activation='silu', epochs=2000, patience=200,
                 learning_rate=1e-3, batch_size=64, validation_frac=0.1, optimizer='adam',
                 seed=42, **kwargs):
        super().__init__(params, limits, **kwargs)
        self.nsamples = int(nsamples) if nsamples is not None else 512 * len(self.params)
        self.nhidden = tuple(int(width) for width in nhidden)
        if activation not in ACTIVATIONS:
            raise ValueError(f'unknown activation {activation!r}; available {list(ACTIVATIONS)}')
        self.activation = activation
        self.epochs, self.patience = int(epochs), int(patience)
        self.learning_rate, self.batch_size = float(learning_rate), int(batch_size)
        self.validation_frac, self.optimizer = float(validation_frac), str(optimizer)
        self.seed = int(seed)
        self.layers = None
        self._output_mean = self._output_scale = None

    # ── nodes ─────────────────────────────────────────────────────────────────
    def nodes(self):
        """Quasi-random samples of the box, in physical parameters.

        Sobol rather than uniform random: a low-discrepancy sequence covers the box far more
        evenly at the same count, and the count here is the entire cost. not nested the way the
        grid's levels are -- raising ``nsamples`` means evaluating a fresh set, so pick it once.
        """
        from scipy.stats import qmc

        dimension = len(self.params)
        unit = qmc.Sobol(d=dimension, scramble=True, seed=self.seed).random(self.nsamples)
        if self.whitened:
            internal = (2. * unit - 1.) * self.nsigma
            return np.array([self.unwhiten(row) for row in internal])
        low = np.array([self._domain(name)[0] for name in self.params])
        high = np.array([self._domain(name)[1] for name in self.params])
        internal = low + unit * (high - low)
        return np.array([self._physical(row) for row in internal])

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
        """``inputs``: (n_nodes, n_params), physical. ``outputs``: (n_nodes, n_outputs)."""
        import jax
        from jax import numpy as jnp
        import optax

        inputs = np.asarray(inputs, dtype='f8')
        outputs = np.asarray(outputs, dtype='f8')
        if len(inputs) != len(outputs):
            raise ValueError(f'{len(inputs)} inputs against {len(outputs)} outputs')
        internal = np.array([self._internal(row) for row in inputs])
        self._output_mean, self._output_scale = self._standardise(outputs)
        targets = (outputs - self._output_mean) / self._output_scale

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(inputs))
        nvalidation = max(int(len(inputs) * self.validation_frac + 0.5), 1)
        if nvalidation >= len(inputs):
            raise ValueError(f'{nvalidation} validation samples out of {len(inputs)}: train on '
                             f'more nodes, or lower validation_frac')
        validation, training = order[:nvalidation], order[nvalidation:]

        widths = (internal.shape[1],) + self.nhidden + (targets.shape[1],)
        key = jax.random.PRNGKey(self.seed)
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

        def loss_fn(layers, x, y):
            return jnp.mean((forward(layers, x) - y)**2)

        gradient = jax.jit(jax.value_and_grad(loss_fn))
        evaluate = jax.jit(loss_fn)
        tx = getattr(optax, self.optimizer)(self.learning_rate)
        state = tx.init(layers)

        x_train = jnp.asarray(internal[training])
        y_train = jnp.asarray(targets[training])
        x_validation = jnp.asarray(internal[validation])
        y_validation = jnp.asarray(targets[validation])
        batch = min(self.batch_size, len(training))

        best, best_loss, waited = layers, np.inf, 0
        for epoch in range(self.epochs):
            shuffled = rng.permutation(len(training))
            for start in range(0, len(training), batch):
                index = shuffled[start:start + batch]
                _, grads = gradient(layers, x_train[index], y_train[index])
                updates, state = tx.update(grads, state, layers)
                layers = optax.apply_updates(layers, updates)
            loss = float(evaluate(layers, x_validation, y_validation))
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
        return x * xnp.asarray(self._output_scale) + xnp.asarray(self._output_mean)

    def contract(self, matrix):
        """Left-multiply the output by a fixed ``matrix``, exactly.

        The network is non-linear, but everything after its last layer is affine -- the last
        matmul and the output standardisation -- so ``M`` folds into it. The standardisation is
        absorbed at the same time (and reset), because ``M @ (y * scale + mean)`` is affine in
        the last layer's output but not a rescaling of ``scale`` alone.
        """
        if self.layers is None:
            raise ValueError('not fitted')
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
                      'optimizer': self.optimizer, 'seed': self.seed,
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
        new._output_mean, new._output_scale = state['output_mean'], state['output_scale']
        nlayers = int(state['nlayers'])
        new.layers = [(state[f'weight.{index}'], state[f'bias.{index}'])
                      for index in range(nlayers)] or None
        return new
