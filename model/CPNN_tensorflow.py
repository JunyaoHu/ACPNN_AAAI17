from abc import abstractmethod, ABC

import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from sklearn.preprocessing import OneHotEncoder

from PyLDL.pyldl.metrics import score, sort_loss

class _Base(ABC):

    def __init__(self, random_state=None):
        if not random_state is None:
            np.random.seed(random_state)

        self._n_features = None
        self._n_outputs = None

    @abstractmethod
    def fit(self, X, y=None):
        self._X = X
        self._y = y
        self._n_features = self._X.shape[1]
        if not self._y is None:
            self._n_outputs = self._y.shape[1]

    def _not_been_fit(self):
        raise ValueError("The model has not yet been fit. "
                         "Try to call 'fit()' first with some training data.")

    @property
    def n_features(self):
        if self._n_features is None:
            self._not_been_fit()
        return self._n_features

    @property
    def n_outputs(self):
        if self._n_outputs is None:
            self._not_been_fit()
        return self._n_outputs

    def __str__(self):
        return self.__class__.__name__

class _BaseDeep(keras.Model):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        keras.Model.__init__(self)
        if not random_state is None:
            tf.random.set_seed(random_state)
        self._n_latent = n_latent
        self._n_hidden = n_hidden

    @staticmethod
    @tf.function
    def _l2_reg(model):
        reg = 0.
        for i in model.trainable_variables:
            reg += tf.reduce_sum(tf.square(i))
        return reg / 2.
    
class _BaseLDL(_Base):

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, X, y, metrics=None):
        if metrics is None:
            metrics = ["chebyshev", "clark", "canberra", "kl_divergence",
                       "cosine", "intersection"]
        return score(y, self.predict(X), metrics=metrics)

class BaseDeepLDL(_BaseLDL, _BaseDeep):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        _BaseLDL.__init__(self, random_state)
        _BaseDeep.__init__(self, n_hidden, n_latent, random_state)

    def fit(self, X, y):
        _BaseLDL.fit(self, X, y)
        self._X = tf.cast(self._X, dtype=tf.float32)
        self._y = tf.cast(self._y, dtype=tf.float32)

class CPNN(BaseDeepLDL):

    def _not_proper_mode(self):
        raise ValueError("The argument 'mode' can only be 'none', 'binary' or 'augment'.")

    def __init__(self, mode='none', v=5, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)
        if mode == 'none' or mode == 'binary' or mode == 'augment':
            self._mode = mode
        else:
            self._not_proper_mode()
        self._v = v

    def fit(self, X, y, learning_rate=5e-3, epochs=3000):
        super().fit(X, y)

        self._optimizer = RProp(init_alpha=learning_rate)

        if self._n_hidden is None:
            self._n_hidden = self._n_features * 3 // 2

        if self._mode == 'augment':
            one_hot = tf.one_hot(tf.math.argmax(self._y, axis=1), self.n_outputs)
            self._X = tf.repeat(self._X, self._v, axis=0)
            self._y = tf.repeat(self._y, self._v, axis=0)
            one_hot = tf.repeat(one_hot, self._v, axis=0)
            v = tf.reshape(tf.tile([1 / (i + 1) for i in range(self._v)], [X.shape[0]]), (-1, 1))
            self._y += self._y * one_hot * v

        input_shape = (self._n_features + (1 if self._mode == 'none' else self._n_outputs),)
        self._model = keras.Sequential([keras.layers.InputLayer(input_shape=input_shape),
                                        keras.layers.Dense(self._n_hidden, activation='sigmoid'),
                                        keras.layers.Dense(1, activation=None)])

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                loss = self._loss(self._X, self._y)
            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.get_updates(self.trainable_variables, gradients)

    def _make_inputs(self, X):
        temp = tf.reshape(tf.tile([i + 1 for i in range(self._n_outputs)], [X.shape[0]]), (-1, 1))
        if self._mode != 'none':
            temp = OneHotEncoder(sparse=False).fit_transform(temp)
        return tf.concat([tf.cast(tf.repeat(X, self._n_outputs, axis=0), dtype=tf.float32),
                          tf.cast(temp, dtype=tf.float32)],
                          axis=1)

    def _call(self, X):
        inputs = self._make_inputs(X)
        outputs = self._model(inputs)
        results = tf.reshape(outputs, (X.shape[0], self._n_outputs))
        b = tf.reshape(-tf.math.log(tf.math.reduce_sum(tf.math.exp(results), axis=1)), (-1, 1))
        return tf.math.exp(b + results)

    def _loss(self, X, y):
        return tf.math.reduce_mean(keras.losses.kl_divergence(y, self._call(X)))

    def predict(self, X):
        return self._call(X)


class BCPNN(CPNN):

    def __init__(self, **params):
        super().__init__(mode='binary', **params)


class ACPNN(CPNN):

    def __init__(self, **params):
        super().__init__(mode='augment', **params)
        
class RProp(keras.optimizers.Optimizer):
    
    def __init__(self, init_alpha=1e-3, scale_up=1.2, scale_down=0.5, min_alpha=1e-6, max_alpha=50., **kwargs):
        super(RProp, self).__init__(name='rprop', **kwargs)
        self.init_alpha = K.variable(init_alpha, name='init_alpha')
        self.scale_up = K.variable(scale_up, name='scale_up')
        self.scale_down = K.variable(scale_down, name='scale_down')
        self.min_alpha = K.variable(min_alpha, name='min_alpha')
        self.max_alpha = K.variable(max_alpha, name='max_alpha')

    def get_updates(self, params, gradients):
        grads = gradients
        shapes = [K.int_shape(p) for p in params]
        alphas = [K.variable(np.ones(shape) * self.init_alpha) for shape in shapes]
        old_grads = [K.zeros(shape) for shape in shapes]
        prev_weight_deltas = [K.zeros(shape) for shape in shapes]
        self.updates = []

        for param, grad, old_grad, prev_weight_delta, alpha in zip(params, grads,
                                                                   old_grads, prev_weight_deltas,
                                                                   alphas):

            new_alpha = K.switch(
                K.greater(grad * old_grad, 0),
                K.minimum(alpha * self.scale_up, self.max_alpha),
                K.switch(K.less(grad * old_grad, 0), K.maximum(alpha * self.scale_down, self.min_alpha), alpha)
            )

            new_delta = K.switch(K.greater(grad, 0),
                                 -new_alpha,
                                 K.switch(K.less(grad, 0),
                                          new_alpha,
                                          K.zeros_like(new_alpha)))

            weight_delta = K.switch(K.less(grad*old_grad, 0), -prev_weight_delta, new_delta)

            new_param = param + weight_delta

            grad = K.switch(K.less(grad*old_grad, 0), K.zeros_like(grad), grad)

            self.updates.append(K.update(param, new_param))
            self.updates.append(K.update(alpha, new_alpha))
            self.updates.append(K.update(old_grad, grad))
            self.updates.append(K.update(prev_weight_delta, weight_delta))

        return self.updates

    def get_config(self):
        config = {
            'init_alpha': float(K.get_value(self.init_alpha)),
            'scale_up': float(K.get_value(self.scale_up)),
            'scale_down': float(K.get_value(self.scale_down)),
            'min_alpha': float(K.get_value(self.min_alpha)),
            'max_alpha': float(K.get_value(self.max_alpha)),
        }
        base_config = super(RProp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))