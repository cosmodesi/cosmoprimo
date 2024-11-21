import numpy as np

from .base import BaseEmulatorEngine, Operation, NormOperation, ScaleOperation
from . import mpi


def create_learning_rate_fn(base_learning_rate, num_epochs, steps_per_epoch):
    """
    Create learning rate schedule.
    Taken from https://flax.readthedocs.io/en/latest/guides/training_techniques/lr_schedule.html.
    """
    import optax
    warmup_epochs = int(0.1 * num_epochs + 0.5)
    warmup_fn = optax.linear_schedule(
      init_value=0., end_value=base_learning_rate,
      transition_steps=warmup_epochs * steps_per_epoch)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch)
    schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn


def _make_tuple(obj, length=None):
    # Return tuple from ``obj``.
    if np.ndim(obj) == 0:
        obj = (obj,)
        if length is not None:
            obj *= length
    return tuple(obj)


class MLPEmulatorEngine(BaseEmulatorEngine):
    """
    Multi-layer perceptron emulator. Based on Joe DeRose and Stephen Chen's EmulateLSS code:
    https://github.com/sfschen/EmulateLSS
    Or cosmopower:
    https://github.com/alessiospuriomancini/cosmopower

    TODO
    ----
    Flax or not flax?

    Parameters
    ----------
    nhidden : tuple, default=(32, 32, 32)
        Size of hidden layers.

    loss : str, callable, default='mse'
        Loss function for training.
    """
    name = 'mlp'
    dtype = 'float64'

    def __init__(self, *args, nhidden=(32, 32, 32), activation='silu', loss='mse', model_yoperation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.nhidden = tuple(nhidden)
        self.loss = loss
        self.activation = _make_tuple(activation, length=len(self.nhidden))
        from .base import make_list, get_operation
        self.model_yoperations = [get_operation(operation) for operation in make_list(model_yoperation)]
        for operations in [self.xoperations, self.yoperations]:
            if len(operations) == 0 or operations[-1].name not in ['scale', 'norm', 'pca']:
                operations.append(ScaleOperation())

    def get_default_samples(self, calculator, params, engine='rqrs', niterations=int(1e4)):
        """
        Returns samples.

        Parameters
        ----------
        order : int, dict, default=3
            A dictionary mapping parameter name (including wildcard) to maximum derivative order.
            If a single value is provided, applies to all varied parameters.

        engine : str, default='rqrs'
            QMC engine, to choose from ['sobol', 'halton', 'lhs', 'rqrs'].

        niterations : int, default=1000
            Number of samples to draw.
        """
        from .samples import QMCSampler
        sampler = QMCSampler(calculator, params, engine=engine, mpicomm=self.mpicomm)
        sampler.run(niterations=niterations)
        return sampler.samples

    def _fit_no_operation(self, X, Y, attrs, validation_frac=0.1, optimizer='adam', loss=None, batch_frac=(0.1, 0.3, 1.), epochs=1000, learning_rate=(1e-2, 1e-3, 1e-5), learning_rate_scheduling=False, batch_norm=False, patience=100, seed=42):
        """
        Fit.

        Parameters
        ----------
        validation_frac : float, default=0.2
            Fraction of the training sample to use for validation.

        optimizer : str, default='adam'
            :mod:`optax` optimizer to use.

        loss : str, callable, default=None
            Override loss function for training provided at initialization (:meth:`__init__`).

        batch_frac : tuple, list, default=(0.1, 0.3, 1.)
            Optimization batch sizes, in units of total sample size.

        epochs : int, tuple, list, default=1000
            Number of optimization epochs or a list of such number for each batch.

        learning_rate : float, tuple, list, default=(1e-2, 1e-3, 1e-5)
            Learning rate, a float or a list of such float for each batch.

        learning_rate_scheduling : bool, callable, default=False
            If ``True``, use learning rate scheduling (cosine scheduler).
            If ``callable``, provide a function with same signature as :func:`create_learning_rate_fn`.
            See https://flax.readthedocs.io/en/latest/guides/training_techniques/lr_schedule.html.

        batch_norm : bool, default=False
            If ``True``, apply batch normalization.
            See https://flax.readthedocs.io/en/latest/guides/training_techniques/batch_norm.html.

        patience : int, tuple, list, default=100
            Wait for this number of epochs without loss improvement before stopping the optimization.

        seed : int, default=None
            Random seed.
        """
        if loss is None:
            loss = self.loss

        if isinstance(learning_rate_scheduling, bool) and learning_rate_scheduling:
            learning_rate_scheduling = create_learning_rate_fn

        import jax
        from jax import numpy as jnp
        import optax
        from flax import linen as nn
        from flax.training import train_state
        from flax.serialization import to_state_dict

        validation_frac = float(validation_frac)
        list_batch_frac = _make_tuple(batch_frac, length=1)
        list_epochs = _make_tuple(epochs, length=len(list_batch_frac))
        list_learning_rate = _make_tuple(learning_rate, length=len(list_batch_frac))
        list_patience = _make_tuple(patience, length=len(list_batch_frac))
        rng = np.random.RandomState(seed=seed)

        self.model_operations = None
        for operation in self.model_yoperations:
            operation.initialize(Y)

        class ExplicitMLP(nn.Module):

            features: tuple
            activation: str
            batch_norm: bool
            yoperations: list
            dtype: str = 'f8'

            @property
            def nlayers(self):
                return len(self.features)

            @nn.compact
            def __call__(self, inputs, train=False):
                x = inputs
                for ilayer, feat in enumerate(self.features):
                    # linear network operation
                    if self.batch_norm and ilayer > 0:
                        x = nn.BatchNorm(use_running_average=not train, name=f'batch_{ilayer}', dtype=self.dtype, epsilon=1e-5)(x)
                    x = nn.Dense(feat, name=f'layer_{ilayer}', dtype=self.dtype)(x)
                    # non-linear activation function
                    if ilayer < self.nlayers - 1:
                        activation = self.activation[ilayer]
                        if activation == 'identity-silu':
                            beta = self.param(f'beta_{ilayer}', nn.initializers.zeros_init(), (), self.dtype)
                            alpha = self.param(f'alpha_{ilayer}', nn.initializers.zeros_init(), (), self.dtype)
                            x = ((1. - beta) + beta / (1 + jnp.exp(-alpha * x))) * x
                        elif activation == 'silu':
                            x = x / (1 + jnp.exp(-x))
                        elif activation == 'relu':
                            x = jnp.maximum(x, 0.)
                        elif activation == 'tanh':
                            x = jnp.tanh(x)
                        else:
                            raise ValueError('unknown activation {}'.format(activation))
                for operation in self.yoperations:
                    x = operation(x)
                return x

            def operations(self, params, batch_stats):
                operations = []
                for ilayer in range(self.nlayers):
                    # linear network operation
                    if self.batch_norm and ilayer > 0:
                        pbatch = params['batch_{:d}'.format(ilayer)]
                        sbatch = batch_stats['batch_{:d}'.format(ilayer)]
                        scale = np.asarray(pbatch['scale'] / jnp.sqrt(sbatch['var'] + 1e-5))
                        mean = np.asarray(sbatch['mean'])
                        bias = np.asarray(pbatch['bias'])
                        operations.append(Operation('scale * (v - mean) + bias', locals={'scale': scale, 'mean': mean, 'bias': bias}))
                    player = params['layer_{:d}'.format(ilayer)]
                    operations.append(Operation('v @ kernel + bias', locals={name: np.asarray(player[name]) for name in ['kernel', 'bias']}))
                    # non-linear activation function
                    if ilayer < self.nlayers - 1:
                        activation = self.activation[ilayer]
                        if activation == 'identity-silu':
                            operations.append(Operation('((1 - beta) + beta / (1 + jnp.exp(-alpha * v))) * v', locals={'beta': np.asarray(params['beta_{:d}'.format(ilayer)]), 'alpha': np.asarray(params['alpha_{:d}'.format(ilayer)])}))
                        elif activation == 'silu':
                            operations.append(Operation('v / (1 + jnp.exp(-v))', locals={}))
                        elif activation == 'relu':
                            operations.append(Operation('jnp.maximum(v, 0.)', locals={}))
                        elif activation == 'tanh':
                            operations.append(Operation('jnp.tanh(v)', locals={}))
                return operations

        nsamples = self.mpicomm.bcast(len(X) if self.mpicomm.rank == 0 else None)
        nvalidation = int(nsamples * validation_frac + 0.5)
        if nvalidation >= nsamples:
            raise ValueError('Cannot use {:d} validation samples (>= {:d} total samples)'.format(nvalidation, nsamples))

        compute_loss = loss
        if isinstance(loss, str) and loss == 'mse':

            def compute_loss(y_true, y_pred):
                return jnp.mean((y_true - y_pred)**2)

        def compute_metrics(y_true, y_pred):
            return {'distance': jnp.mean((y_true - y_pred)**2)**0.5}

        @jax.jit
        def eval_step(state, batch):
            x, y_true = batch
            y_pred = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, x, train=False)
            loss = compute_loss(y_true, y_pred)
            metrics = compute_metrics(y_true, y_pred)
            metrics['learning_rate'] = learning_rate_fn
            if learning_rate_scheduling:
                metrics['learning_rate'] = learning_rate_fn(state.step)
            return loss, metrics

        from typing import Any

        class TrainState(train_state.TrainState):
            batch_stats: dict

        if self.mpicomm.rank == 0:
            y = Y[0]
            for operation in self.model_yoperations: y = operation(y)
            model = ExplicitMLP(features=self.nhidden + y.shape, activation=self.activation, batch_norm=batch_norm, yoperations=[jax.vmap(operation.inverse) for operation in self.model_yoperations], dtype=self.dtype)
            best_params = {}
            best_batch_stats = {}

            # train using cooling/heating schedule for lr/batch-size
            for batch_frac, epochs, learning_rate, patience in zip(list_batch_frac, list_epochs, list_learning_rate, list_patience):

                # split into validation and training sub-sets
                samples = {'X': X, 'Y': Y}
                index1 = rng.choice(nsamples, size=nvalidation, replace=False)
                index2 = rng.choice(nsamples, size=nsamples, replace=False)
                index2 = index2[~np.isin(index2, index1)]

                assert index1.size + index2.size == nsamples
                for name, value in list(samples.items()):
                    samples['{}_validation'.format(name)] = value[index1].astype(self.dtype)
                    samples['{}_training'.format(name)] = value[index2].astype(self.dtype)

                ntraining = nsamples - nvalidation
                batch_size = max(int(ntraining * min(batch_frac, 1.) + 0.5), 1)
                self.log_info('Using (batch size, epochs, learning rate) = ({:d}, {:d}, {:.2e})'.format(batch_size, epochs, learning_rate))

                training_data = []
                for i in range(ntraining // batch_size):
                    sl = slice(i * batch_size, (i + 1) * batch_size)
                    training_data.append((samples['X_training'][sl], samples['Y_training'][sl]))

                #def reshape(samples):
                #    nbatch = ntraining // batch_size
                #    total_size = nbatch * batch_size
                #    return samples[:total_size].reshape(nbatch, batch_size, -1)

                #training_data = reshape(samples['X_training']), reshape(samples['Y_training'])

                random_key = jax.random.PRNGKey(seed)
                variables = model.init(random_key, jnp.ones(training_data[0][0].shape), train=False)
                # Create the optimizer
                learning_rate_fn = learning_rate
                if learning_rate_scheduling:
                    learning_rate_fn = learning_rate_scheduling(learning_rate, epochs, steps_per_epoch=len(training_data))
                tx = getattr(optax, optimizer)(learning_rate_fn)
                # Create a state
                best_state = state = TrainState.create(apply_fn=model.apply, tx=tx, params=best_params if best_params else variables['params'], batch_stats=best_batch_stats if best_batch_stats or not batch_norm else variables['batch_stats'])

                import functools
                @functools.partial(jax.jit, static_argnums=2)
                def train_step(state, batch, learning_rate_fn):
                    x, y_true = batch

                    def loss_fn(params):
                        y_pred, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, x, train=True, mutable=['batch_stats'])
                        loss = compute_loss(y_true, y_pred)
                        return loss, (y_pred, updates)

                    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (_, (y_pred, updates)), grads = gradient_fn(state.params)
                    state = state.apply_gradients(grads=grads)
                    state = state.replace(batch_stats=updates['batch_stats'])
                    metrics = compute_metrics(y_true, y_pred)
                    metrics['learning_rate'] = learning_rate_fn
                    if learning_rate_scheduling:
                        metrics['learning_rate'] = learning_rate_fn(state.step)
                    return state, metrics

                state = best_state
                best_loss = np.infty
                best_metrics = {}
                early_stopping_counter = 0

                # loop over epochs
                from tqdm import trange
                with trange(epochs) as t:
                    for epoch in t:
                        # loop over batches
                        #train_batch_metrics = []
                        for batch in training_data:
                            state, metrics = train_step(state, batch, learning_rate_fn)
                            #train_batch_metrics.append(metrics)
                        #state, metrics = jax.lax.scan(lambda state, batch: train_step(state, batch, learning_rate_fn), state, training_data)
                        # compute validation loss at the end of the epoch
                        loss, metrics = eval_step(state, (samples['X_validation'], samples['Y_validation']))

                        # update the progressbar
                        t.set_postfix(loss=loss)

                        # early stopping condition
                        if loss < best_loss:
                            best_state, best_loss, best_metrics = state, loss, metrics
                            early_stopping_counter = 0
                        else:
                            early_stopping_counter += 1
                        if early_stopping_counter >= patience:
                            break
                    #assert np.allclose(best_metrics['distance'], best_loss**0.5)
                    self.log_info(', '.join(['{} = {:.3e}'.format(name, value) for name, value in {'validation loss': best_loss, **best_metrics}.items()]))
                best_params, best_batch_stats = best_state.params, best_state.batch_stats

            self.model_operations = model.operations(best_params, best_batch_stats)
            #x = samples['X']
            #from cosmoprimo.jax import vmap
            #y_pred = vmap(self._predict_no_operation)(x)
            #print(samples['Y'][:3], y_pred[:3], compute_loss(samples['Y'], y_pred))
            #y_pred2 = best_state.apply_fn({'params': best_state.params, 'batch_stats': best_state.batch_stats}, x, train=False)
            #print(y_pred)
            #exit()

        mpi.barrier_idle(self.mpicomm)  # we rely on keras parallelisation; here we make MPI processes idle

        self.model_operations = self.mpicomm.bcast(self.model_operations, root=0)

    def _predict_no_operation(self, X):
        x = X
        for operation in self.model_operations:
            x = operation(x)
        for operation in self.model_yoperations:
            x = operation.inverse(x)
        return x

    def __getstate__(self):
        state = super().__getstate__()
        for name in ['nhidden']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        #try: state['tfmodel'] = self.tfmodel.__getstate__()
        #except AttributeError: pass
        for name in ['model_operations', 'model_yoperations']:
            try: state[name] = [operation.__getstate__() for operation in getattr(self, name)]
            except AttributeError: pass
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        for name in ['model_operations', 'model_yoperations']:
            try: setattr(self, name, [Operation.from_state(state) for state in getattr(self, name)])
            except AttributeError: pass