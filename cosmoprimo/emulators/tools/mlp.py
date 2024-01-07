import numpy as np

from .jax import numpy as jnp
from .base import BaseEmulatorEngine, Operation, NormOperation
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

    def __init__(self, *args, nhidden=(32, 32, 32), loss='mse', **kwargs):
        super().__init__(*args, **kwargs)
        self.nhidden = tuple(nhidden)
        self.loss = loss
        for operations in [self.xoperations, self.yoperations]:
            if len(operations) == 0 or operations[-1].name not in ['scale', 'norm', 'pca']:
                operations.append(NormOperation())

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

    def _fit_no_operation(self, X, Y, attrs, validation_frac=0.2, optimizer='adam', loss=None, batch_frac=(0.1, 0.3, 1.), epochs=1000, learning_rate=(1e-2, 1e-3, 1e-5), verbose=0, seed=None):
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
        optimizer = str(optimizer)
        validation_frac = float(validation_frac)
        list_batch_frac = _make_tuple(batch_frac, length=1)
        list_epochs = _make_tuple(epochs, length=len(list_batch_frac))
        list_learning_rate = _make_tuple(learning_rate, length=len(list_batch_frac))
        rng = np.random.RandomState(seed=seed)

        self.model_operations = None

        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping

        class TFModel(tf.keras.Model):

            def __init__(self, architecture):
                super(TFModel, self).__init__()
                self.architecture = architecture
                self.nlayers = len(self.architecture) - 1

                self.W, self.b, self.alpha, self.beta = [], [], [], []
                for i in range(self.nlayers):
                    self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i + 1]], 0., np.sqrt(2. / self.architecture[0])), name='W_{:d}'.format(i), trainable=True))
                    self.b.append(tf.Variable(tf.zeros([self.architecture[i + 1]]), name='b_{:d}'.format(i), trainable=True))
                for i in range(self.nlayers - 1):
                    self.alpha.append(tf.Variable(tf.random.normal([self.architecture[i + 1]]), name='alpha_{:d}'.format(i), trainable=True))
                    self.beta.append(tf.Variable(tf.random.normal([self.architecture[i + 1]]), name='beta_{:d}'.format(i), trainable=True))

            @tf.function
            def call(self, x):
                for i in range(self.nlayers):
                    # linear network operation
                    x = tf.add(tf.matmul(x, self.W[i]), self.b[i])
                    # non-linear activation function
                    if i < self.nlayers - 1:
                        x = tf.multiply(tf.add(self.beta[i], tf.multiply(tf.sigmoid(tf.multiply(self.alpha[i], x)), tf.subtract(1., self.beta[i]))), x)
                return x

            def operations(self):
                operations = []
                for i in range(self.nlayers):
                    # linear network operation
                    operations.append(Operation('v @ W + b', locals={'W': self.W[i].numpy(), 'b': self.b[i].numpy()}))
                    # non-linear activation function
                    if i < self.nlayers - 1:
                        operations.append(Operation('(beta + (1 - beta) / (1 + jnp.exp(-alpha * v))) * v', locals={'alpha': self.alpha[i].numpy(), 'beta': self.beta[i].numpy()}))
                return operations

            def __getstate__(self):
                state = {}
                for name in ['W', 'b', 'alpha', 'beta']:
                    state[name] = [value.numpy() for value in getattr(self, name)]
                return state

            def __setstate__(self, state):
                for name in ['W', 'b', 'alpha', 'beta']:
                    for tfvalue, npvalue in zip(getattr(self, name), state[name]):
                        tfvalue.assign(npvalue)

        nsamples = self.mpicomm.bcast(len(X) if self.mpicomm.rank == 0 else None)
        nvalidation = int(nsamples * validation_frac + 0.5)
        if nvalidation >= nsamples:
            raise ValueError('Cannot use {:d} validation samples (>= {:d} total samples)'.format(nvalidation, nsamples))

        if self.mpicomm.rank == 0:
            samples = {'X': X, 'Y': Y}
            mask = np.zeros(nsamples, dtype='?')
            mask[rng.choice(nsamples, size=nvalidation, replace=False)] = True
            for name, value in list(samples.items()):
                samples['{}_validation'.format(name)] = value[mask]
                samples['{}_training'.format(name)] = value[~mask]
            architecture = [X.shape[-1]] + list(self.nhidden) + [Y.shape[-1]]
            tfmodel = TFModel(architecture)
            state = getattr(self, 'tfmodel', None)
            if state is not None:
                if not isinstance(state, dict):
                    state = state.__getstate__()
                tfmodel.__setstate__(state)
            self.tfmodel = tfmodel
            self.tfmodel.compile(optimizer=optimizer, loss=loss, metrics=['mse'])
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=50)

            for batch_frac, epochs, learning_rate in zip(list_batch_frac, list_epochs, list_learning_rate):
                batch_size = max(int(nsamples * batch_frac + 0.5), 1)
                if learning_rate is None:
                    learning_rate = self.tfmodel.optimizer.lr.numpy()
                else:
                    self.tfmodel.optimizer.lr.assign(learning_rate)
                self.log_info('Using (batch size, epochs, learning rate) = ({:d}, {:d}, {:.2e})'.format(batch_size, epochs, learning_rate))
                self.tfmodel.fit(samples['X_training'], samples['Y_training'], batch_size=batch_size, epochs=epochs,
                                 validation_data=(samples['X_validation'], samples['Y_validation']), callbacks=[es], verbose=verbose)
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