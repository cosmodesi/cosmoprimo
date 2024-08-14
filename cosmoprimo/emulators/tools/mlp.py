import numpy as np

from .base import BaseEmulatorEngine, Operation, NormOperation, ScaleOperation
from . import mpi


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
    dtype = 'float32'

    def __init__(self, *args, nhidden=(32, 32, 32), activation='identity-silu', loss='mse', **kwargs):
        super().__init__(*args, **kwargs)
        self.nhidden = tuple(nhidden)
        self.loss = loss
        self.activation = _make_tuple(activation, length=len(self.nhidden))
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

    def _fit_no_operation(self, X, Y, attrs, validation_frac=0.1, optimizer='adam', loss=None, batch_frac=(0.1, 0.3, 1.), epochs=1000, learning_rate=(1e-2, 1e-3, 1e-5), patience=100, verbose=0, seed=42):
        """
        Fit.

        Parameters
        ----------
        validation_frac : float, default=0.2
            Fraction of the training sample to use for validation.

        loss : str, callable, default=None
            Override loss function for training provided at initialization (:meth:`__init__`).

        optimizer : str, default='adam'
            Tensorflow optimizer to use.

        batch_frac : tuple, list, default=(0.1, 0.3, 1.)
            Optimization batch sizes, in units of total sample size.

        epochs : int, tuple, list, default=1000
            Number of optimization epochs or a list of such number for each batch.

        learning_rate : float, tuple, list, default=(1e-2, 1e-3, 1e-5)
            Learning rate, a float or a list of such float for each batch.

        verbose : int, default=0
            Tensorflow verbosity.

        seed : int, default=None
            Random seed.
        """
        if loss is None:
            loss = self.loss
        if loss in ['mse']:
            import tensorflow as tf
            def loss(y_true, y_pred):
                return tf.reduce_mean(tf.square(y_true - y_pred))
        optimizer = str(optimizer)
        validation_frac = float(validation_frac)
        list_batch_frac = _make_tuple(batch_frac)
        list_learning_rate = _make_tuple(learning_rate)
        list_batch_frac = _make_tuple(batch_frac, length=max(len(list_batch_frac), len(list_learning_rate)))
        list_learning_rate = _make_tuple(learning_rate, length=len(list_batch_frac))
        list_epochs = _make_tuple(epochs, length=len(list_batch_frac))
        list_patience = _make_tuple(patience, length=len(list_batch_frac))
        rng = np.random.RandomState(seed=seed)

        self.model_operations = None

        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping

        tf.keras.backend.set_floatx(self.dtype)

        class TFModel(tf.keras.Model):

            def __init__(self, architecture, activation):
                super(TFModel, self).__init__()
                self.architecture = architecture
                self.activation = activation
                self.nlayers = len(self.architecture) - 1

                self.W, self.b, self.alpha, self.beta = [], [], [], []
                for i in range(self.nlayers):
                    #self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i + 1]], 0., np.sqrt(2. / self.architecture[0])), name='W_{:d}'.format(i), trainable=True))
                    self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i + 1]], 0., 1e-3, dtype=self.dtype), name='W_{:d}'.format(i), trainable=True))
                    #self.W.append(tf.Variable(tf.zeros([self.architecture[i], self.architecture[i + 1]], dtype=self.dtype), name='W_{:d}'.format(i), trainable=True))
                    self.b.append(tf.Variable(tf.zeros([self.architecture[i + 1]], dtype=self.dtype), name='b_{:d}'.format(i), trainable=True))
                for i in range(self.nlayers - 1):
                    if self.activation[i] == 'identity-silu':
                        self.alpha.append(tf.Variable(tf.random.normal([self.architecture[i + 1]], dtype=self.dtype), name='alpha_{:d}'.format(i), trainable=True))
                        self.beta.append(tf.Variable(tf.random.normal([self.architecture[i + 1]], dtype=self.dtype), name='beta_{:d}'.format(i), trainable=True))

            @tf.function
            def call(self, x):
                for i in range(self.nlayers):
                    # linear network operation
                    x = tf.add(tf.matmul(x, self.W[i]), self.b[i])
                    # non-linear activation function
                    if i < self.nlayers - 1:
                        activation = self.activation[i]
                        if activation == 'identity-silu':
                            x = tf.multiply(tf.add(self.beta[i], tf.multiply(tf.sigmoid(tf.multiply(self.alpha[i], x)), tf.subtract(1., self.beta[i]))), x)
                        elif activation == 'tanh':
                            x = tf.tanh(x)
                        else:
                            raise ValueError('unknown activation {}'.format(activation))
                return x

            def operations(self):
                operations = []
                for i in range(self.nlayers):
                    # linear network operation
                    operations.append(Operation('v @ W + b', locals={'W': self.W[i].numpy(), 'b': self.b[i].numpy()}))
                    # non-linear activation function
                    if i < self.nlayers - 1:
                        activation = self.activation[i]
                        if activation == 'identity-silu':
                            operations.append(Operation('(beta + (1 - beta) / (1 + jnp.exp(-alpha * v))) * v', locals={'alpha': self.alpha[i].numpy(), 'beta': self.beta[i].numpy()}))
                        elif activation == 'tanh':
                            operations.append(Operation('jnp.tanh(v)', locals={}))
                return operations

            def __getstate__(self):
                state = {}
                for name in ['W', 'b', 'alpha', 'beta']:
                    state[name] = [value.numpy() for value in getattr(self, name) if hasattr(self, name)]
                return state

            def __setstate__(self, state):
                for name in ['W', 'b', 'alpha', 'beta']:
                    if hasattr(self, name):
                        for tfvalue, npvalue in zip(getattr(self, name), state[name]):
                            tfvalue.assign(npvalue)

        nsamples = self.mpicomm.bcast(len(X) if self.mpicomm.rank == 0 else None)
        nvalidation = int(nsamples * validation_frac + 0.5)
        if nvalidation >= nsamples:
            raise ValueError('Cannot use {:d} validation samples (>= {:d} total samples)'.format(nvalidation, nsamples))

        if self.mpicomm.rank == 0:
            architecture = [X.shape[-1]] + list(self.nhidden) + [Y.shape[-1]]
            tfmodel = TFModel(architecture, self.activation)
            state = getattr(self, 'tfmodel', None)
            if state is not None:
                if not isinstance(state, dict):
                    state = state.__getstate__()
                tfmodel.__setstate__(state)
            self.tfmodel = tfmodel
            self.tfmodel.compile(optimizer=optimizer, loss=loss)

            for batch_frac, epochs, learning_rate, patience in zip(list_batch_frac, list_epochs, list_learning_rate, list_patience):
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=patience)
                batch_size = max(int(nsamples * batch_frac + 0.5), 1)
                if learning_rate is None:
                    learning_rate = self.tfmodel.optimizer.lr.numpy()
                else:
                    self.tfmodel.optimizer.lr.assign(learning_rate)
                self.log_info('Using (batch size, epochs, learning rate) = ({:d}, {:d}, {:.2e})'.format(batch_size, epochs, learning_rate))
                samples = {'X': X, 'Y': Y}
                index1 = rng.choice(nsamples, size=nvalidation, replace=False)
                index2 = rng.choice(nsamples, size=nsamples, replace=False)
                index2 = index2[~np.isin(index2, index1)]
                print(X)

                assert index1.size + index2.size == nsamples
                for name, value in list(samples.items()):
                    samples['{}_validation'.format(name)] = value[index1].astype(self.dtype)
                    samples['{}_training'.format(name)] = value[index2].astype(self.dtype)

                self.tfmodel.fit(samples['X_training'], samples['Y_training'], batch_size=batch_size, epochs=epochs,
                                 validation_data=(samples['X_validation'], samples['Y_validation']), callbacks=[es], verbose=verbose)
                val_loss = loss(self.tfmodel.call(samples['X_validation']), samples['Y_validation'])
                self.log_info('Validation loss = {:.3e}.'.format(val_loss))

            self.model_operations = self.tfmodel.operations()

        mpi.barrier_idle(self.mpicomm)  # we rely on keras parallelisation; here we make MPI processes idle

        self.model_operations = self.mpicomm.bcast(self.model_operations, root=0)


    def _fit_no_operation2(self, X, Y, attrs, validation_frac=0.1, optimizer='adam', loss=None, batch_frac=(0.1, 0.3, 1.), epochs=1000, learning_rate=(1e-2, 1e-3, 1e-5), patience=100, verbose=0, seed=42):
        """
        Fit.

        Parameters
        ----------
        validation_frac : float, default=0.2
            Fraction of the training sample to use for validation.

        loss : str, callable, default=None
            Override loss function for training provided at initialization (:meth:`__init__`).

        optimizer : str, default='adam'
            Tensorflow optimizer to use.

        batch_frac : tuple, list, default=(0.1, 0.3, 1.)
            Optimization batch sizes, in units of total sample size.

        epochs : int, tuple, list, default=1000
            Number of optimization epochs or a list of such number for each batch.

        learning_rate : float, tuple, list, default=(1e-2, 1e-3, 1e-5)
            Learning rate, a float or a list of such float for each batch.

        verbose : int, default=0
            Tensorflow verbosity.

        seed : int, default=None
            Random seed.
        """
        if loss is None:
            loss = self.loss

        validation_frac = float(validation_frac)
        list_batch_frac = _make_tuple(batch_frac, length=1)
        list_epochs = _make_tuple(epochs, length=len(list_batch_frac))
        list_learning_rate = _make_tuple(learning_rate, length=len(list_batch_frac))
        list_patience = _make_tuple(patience, length=len(list_batch_frac))
        rng = np.random.RandomState(seed=seed)

        self.model_operations = None

        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping

        optimizer = tf.keras.optimizers.get(str(optimizer))
        tf.random.set_seed(seed=seed)
        tf.keras.backend.set_floatx(self.dtype)

        class TFModel(tf.keras.Model):

            def __init__(self, architecture, activation):
                super(TFModel, self).__init__()
                self.architecture = architecture
                self.activation = activation
                self.nlayers = len(self.architecture) - 1

                self.W, self.b, self.alpha, self.beta = [], [], [], []
                for i in range(self.nlayers):
                    #self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i + 1]], 0., np.sqrt(2. / self.architecture[0])), name='W_{:d}'.format(i), trainable=True))
                    self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i + 1]], 0., 1e-3, dtype=self.dtype), name='W_{:d}'.format(i), trainable=True))
                    self.b.append(tf.Variable(tf.zeros([self.architecture[i + 1]], dtype=self.dtype), name='b_{:d}'.format(i), trainable=True))
                for i in range(self.nlayers - 1):
                    if self.activation[i] == 'identity-silu':
                        self.alpha.append(tf.Variable(tf.random.normal([self.architecture[i + 1]], dtype=self.dtype), name='alpha_{:d}'.format(i), trainable=True))
                        self.beta.append(tf.Variable(tf.random.normal([self.architecture[i + 1]], dtype=self.dtype), name='beta_{:d}'.format(i), trainable=True))

            @tf.function
            def call(self, x):
                for i in range(self.nlayers):
                    # linear network operation
                    x = tf.add(tf.matmul(x, self.W[i]), self.b[i])
                    # non-linear activation function
                    if i < self.nlayers - 1:
                        activation = self.activation[i]
                        if activation == 'identity-silu':
                            x = tf.multiply(tf.add(self.beta[i], tf.multiply(tf.sigmoid(tf.multiply(self.alpha[i], x)), tf.subtract(1., self.beta[i]))), x)
                        elif activation == 'tanh':
                            x = tf.tanh(x)
                        else:
                            raise ValueError('unknown activation {}'.format(activation))
                return x

            def operations(self):
                operations = []
                for i in range(self.nlayers):
                    # linear network operation
                    operations.append(Operation('v @ W + b', locals={'W': self.W[i].numpy(), 'b': self.b[i].numpy()}))
                    # non-linear activation function
                    if i < self.nlayers - 1:
                        activation = self.activation[i]
                        if activation == 'identity-silu':
                            operations.append(Operation('(beta + (1 - beta) / (1 + jnp.exp(-alpha * v))) * v', locals={'alpha': self.alpha[i].numpy(), 'beta': self.beta[i].numpy()}))
                        elif activation == 'tanh':
                            operations.append(Operation('jnp.tanh(v)', locals={}))
                return operations

            def __getstate__(self):
                state = {}
                for name in ['W', 'b', 'alpha', 'beta']:
                    state[name] = [value.numpy() for value in getattr(self, name) if hasattr(self, name)]
                return state

            def __setstate__(self, state):
                for name in ['W', 'b', 'alpha', 'beta']:
                    if hasattr(self, name):
                        for tfvalue, npvalue in zip(getattr(self, name), state[name]):
                            tfvalue.assign(npvalue)

            @tf.function
            def compute_loss(self, X, Y):
                return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.call(X), Y)))

            @tf.function
            def compute_loss_and_gradients(self, X, Y):
                # compute loss on the tape
                with tf.GradientTape() as tape:
                    # loss
                    #loss = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.call(X), Y)))
                    loss = self.compute_loss(X, Y)

                # compute gradients
                gradients = tape.gradient(loss, self.trainable_variables)
                return loss, gradients

        nsamples = self.mpicomm.bcast(len(X) if self.mpicomm.rank == 0 else None)
        nvalidation = int(nsamples * validation_frac + 0.5)
        if nvalidation >= nsamples:
            raise ValueError('Cannot use {:d} validation samples (>= {:d} total samples)'.format(nvalidation, nsamples))

        if self.mpicomm.rank == 0:
            architecture = [X.shape[-1]] + list(self.nhidden) + [Y.shape[-1]]
            tfmodel = TFModel(architecture, self.activation)
            state = getattr(self, 'tfmodel', None)
            if state is not None:
                if not isinstance(state, dict):
                    state = state.__getstate__()
                tfmodel.__setstate__(state)
            self.tfmodel = tfmodel

            # train using cooling/heating schedule for lr/batch-size
            for batch_frac, epochs, learning_rate, patience in zip(list_batch_frac, list_epochs, list_learning_rate, list_patience):
                batch_size = max(int(nsamples * batch_frac + 0.5), 1)
                if learning_rate is None:
                    learning_rate = optimizer.lr.numpy()
                else:
                    optimizer.lr.assign(learning_rate)
                self.log_info('Using (batch size, epochs, learning rate) = ({:d}, {:d}, {:.2e})'.format(batch_size, epochs, learning_rate))
                # split into validation and training sub-sets
                samples = {'X': X, 'Y': Y}
                mask = np.zeros(nsamples, dtype='?')
                mask[rng.choice(nsamples, size=nvalidation, replace=False)] = True
                for name, value in list(samples.items()):
                    samples['{}_validation'.format(name)] = value[mask].astype(self.dtype)
                    samples['{}_training'.format(name)] = value[~mask].astype(self.dtype)

                # create iterable dataset (given batch size)
                ntraining = nsamples - nvalidation
                training_data = tf.data.Dataset.from_tensor_slices((samples['X_training'], samples['Y_training'])).shuffle(ntraining).batch(batch_size)

                # set up training loss
                training_loss = [np.infty]
                validation_loss = [np.infty]
                best_loss = np.infty
                early_stopping_counter = 0

                @tf.function
                def apply_gradients(gradients, tv):
                    optimizer.apply_gradients(zip(gradients, tv))

                # loop over epochs
                from tqdm import trange
                with trange(epochs) as t:
                    for epoch in t:
                        # loop over batches
                        for x, y in training_data:
                            loss, gradients = self.tfmodel.compute_loss_and_gradients(x, y)
                            # apply gradients
                            apply_gradients(gradients, self.tfmodel.trainable_variables)

                        # compute validation loss at the end of the epoch
                        validation_loss.append(self.tfmodel.compute_loss(samples['X_validation'], samples['Y_validation']).numpy())

                        # update the progressbar
                        t.set_postfix(loss=validation_loss[-1])

                        # early stopping condition
                        if validation_loss[-1] < best_loss:
                            best_loss = validation_loss[-1]
                            early_stopping_counter = 0
                        else:
                            early_stopping_counter += 1
                        if early_stopping_counter >= patience:
                            break
                    self.log_info('Validation loss = {:.3e}.'.format(best_loss))

            self.model_operations = self.tfmodel.operations()

        mpi.barrier_idle(self.mpicomm)  # we rely on keras parallelisation; here we make MPI processes idle

        self.model_operations = self.mpicomm.bcast(self.model_operations, root=0)

    def _predict_no_operation(self, X):
        x = X
        for operation in self.model_operations:
            x = operation(x)
        return x

    def __getstate__(self):
        state = super().__getstate__()
        for name in ['nhidden', 'sampler_options']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        try: state['tfmodel'] = self.tfmodel.__getstate__()
        except AttributeError: pass
        try: state['model_operations'] = [operation.__getstate__() for operation in self.model_operations]
        except AttributeError: pass
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        try: self.model_operations = [Operation.from_state(state) for state in self.model_operations]
        except AttributeError: pass