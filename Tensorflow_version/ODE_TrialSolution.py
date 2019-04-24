import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ODE_TrialSolution(tf.keras.models.Model):
    def __init__(self, conditions, n_i, n_h, n_o=1, activation='sigmoid', call_method=None):
        super(ODE_TrialSolution, self).__init__()

        # Dimensions of the network
        self.n_i = n_i
        self.n_h = n_h
        self.n_o = n_o

        # Boundary conditions
        self.conditions = conditions

        self.call_method = call_method

        # Shallow network
        self.hidden_layer = tf.keras.layers.Dense(
            units=self.n_h, activation=activation)
        self.output_layer = tf.keras.layers.Dense(
            units=self.n_o, activation='linear')


    def call(self, X):
        X = tf.convert_to_tensor(X)
        if not self.call_method is None:
            return self.call_method(self, X)
        response = self.hidden_layer(X)
        response = self.output_layer(response)

        # Automatic conditions incorporation including Neumann BCs
        # It should be used to generate the *call* method instead of calculating it every damned time

        boundary_value = tf.constant(
            0., dtype='float64', shape=response.get_shape())

        for condition in self.conditions:
            vanishing = tf.constant(1., dtype='float64',
                                    shape=response.get_shape())
            temp_bc = 0
            if condition['type'] == 'dirichlet':
                temp_bc = tf.reshape(condition['function'](
                    X), shape=boundary_value.shape)
                for vanisher in self.conditions:
                    if vanisher['variable'] != condition['variable'] and vanisher['value'] != condition['value']:
                        if vanisher['type'] == 'dirichlet':
                            vanishing *= (X[:, vanisher['variable']]
                                            - tf.constant(vanisher['value'], dtype='float64', shape=boundary_value.shape))
                        elif vanisher['type'] == 'neumann':
                            vanishing *= (X[:, vanisher['variable']]
                                            - tf.constant(vanisher['value'], dtype='float64', shape=boundary_value.shape))
                boundary_value += temp_bc * vanishing
                response *= (tf.constant(condition['value'], dtype='float64', shape=boundary_value.shape)
                                - tf.reshape(X[:, condition['variable']], shape=boundary_value.shape))
            elif condition['type'] == 'neumann':
                temp_bc = (tf.reshape(condition['function'](X), shape=boundary_value.shape)
                            * tf.reshape(X[:, condition['variable']], shape=boundary_value.shape))
                boundary_value = temp_bc
                response *= (tf.constant(condition['value'], dtype='float64', shape=boundary_value.shape)
                                - tf.reshape(X[:, condition['variable']], shape=boundary_value.shape))
        response += boundary_value
        return response
