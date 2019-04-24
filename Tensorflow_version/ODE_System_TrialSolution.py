import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ODE_System_TrialSolution(tf.keras.models.Model):
    def __init__(self, ODE_number, conditions, n_i, n_h, n_o=1, activation='sigmoid', call_method=None):
        super(ODE_System_TrialSolution, self).__init__()

        # Dimensions of the network
        self.n_i = n_i
        self.n_h = n_h
        self.n_o = n_o

        # Boundary conditions
        self.conditions = conditions

        self.ODE_number = ODE_number

        self.call_method = call_method

        self.hidden_layers = []
        for _ in range(self.ODE_number):
            self.hidden_layers.append(tf.keras.layers.Dense(
                units=self.n_h, activation=activation))
        self.output_layers = []
        for _ in range(self.ODE_number):
            self.output_layers.append(tf.keras.layers.Dense(
                units=self.n_o, activation='linear'))

    def call(self, X):
        X = tf.convert_to_tensor(X)
        if not self.call_method is None:
            return self.call_method(self, X)
        responses = []
        for i in range(self.ODE_number):
            resp = self.hidden_layers[i](X)
            responses.append(self.output_layers[i](resp))

        for ode in range(self.ODE_number):
            boundary_value = tf.constant(
                0., dtype='float64', shape=responses[ode].get_shape())

            for condition in self.conditions:
                if condition['ode_number'] == ode+1:
                    vanishing = tf.constant(
                        1., dtype='float64', shape=responses[ode].get_shape())
                    temp_bc = 0
                    if condition['type'] == 'dirichlet':
                        temp_bc = tf.reshape(condition['function'](
                            X), shape=boundary_value.shape)
                        for vanisher in self.conditions:
                            if (vanisher['ode_number'] == ode+1
                                and vanisher['variable'] != condition['variable']
                                and vanisher['value'] != condition['value']):
                                if vanisher['type'] == 'dirichlet':
                                    vanishing *= (X[:, vanisher['variable']]
                                                  - tf.constant(vanisher['value'], dtype='float64', shape=boundary_value.shape))
                                elif vanisher['type'] == 'neumann':
                                    vanishing *= (X[:, vanisher['variable']]
                                                  - tf.constant(vanisher['value'], dtype='float64', shape=boundary_value.shape))
                        boundary_value += temp_bc * vanishing
                        responses[ode] *= (tf.constant(condition['value'], dtype='float64', shape=boundary_value.shape)
                                           - tf.reshape(X[:, condition['variable']], shape=boundary_value.shape))
                    elif condition['type'] == 'neumann':
                        temp_bc = (tf.reshape(condition['function'](X), shape=boundary_value.shape)
                                   * tf.reshape(X[:, condition['variable']], shape=boundary_value.shape))
                        boundary_value = temp_bc
                        responses[ode] *= (tf.constant(condition['value'], dtype='float64', shape=boundary_value.shape)
                                           - tf.reshape(X[:, condition['variable']], shape=boundary_value.shape))
            responses[ode] += boundary_value
        response = tf.concat(responses, axis=1)
        return responses
