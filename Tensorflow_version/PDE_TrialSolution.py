import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class PDE_TrialSolution(tf.keras.models.Model):
    def __init__(self, conditions, n_i, n_h, n_o=1, activation='sigmoid', call_method=None):
        super(PDE_TrialSolution, self).__init__()

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

        if X.shape[1] != 2:
            raise Exception('PDEs are supported only in two dimensions.')
        for condition in self.conditions:
            if condition['type'] != 'dirichlet':
                raise Exception(
                    'Only Dirichlet boundary conditions are supported.')
        boundary_value = tf.constant(
            0., dtype='float64', shape=(response.shape[0], 1))
        for condition in self.conditions:
            if condition['variable'] == 0:
                temp_boundary_value = condition['function'](X)

                for condition2 in self.conditions:
                    if condition2['variable'] == condition['variable'] and condition2['value'] != condition['value']:
                        if condition2['value'] != 0:
                            temp_boundary_value *= (tf.constant(condition2['value'], dtype='float64', shape=(response.shape[0], 1))
                                                    - tf.reshape(X[:, condition2['variable']], shape=(X.shape[0], 1)))
                        else:
                            temp_boundary_value *= tf.reshape(
                                X[:, condition2['variable']], shape=(X.shape[0], 1))
                boundary_value += temp_boundary_value

            elif condition['variable'] == 1:
                temp_boundary_value = condition['function'](X)

                for condition2 in self.conditions:
                    if condition2['variable'] != condition['variable']:
                        for condition3 in self.conditions:
                            if not condition2 is condition3 and condition3['variable']==condition2['variable']:
                                if condition2['value'] != 0:
                                    temp_boundary_value -= ((tf.constant(condition2['value'], dtype='float64', shape=(response.shape[0], 1))
                                                            - tf.reshape(X[:, condition2['variable']], shape=(X.shape[0], 1)))
                                                            * condition['function'](tf.constant(condition3['value'],
                                                                                                dtype='float64', shape=(response.shape[0], 2))))
                                else:
                                    temp_boundary_value -= (tf.reshape(X[:, condition2['variable']], shape=(X.shape[0], 1))
                                                            * condition['function'](tf.constant(condition3['value'],
                                                                                                dtype='float64', shape=(response.shape[0], 2))))
                for condition2 in self.conditions:
                    if condition2['variable'] == condition['variable'] and condition2['value'] != condition['value']:
                        if condition2['value'] != 0:
                            temp_boundary_value *= (tf.constant(condition2['value'], dtype='float64', shape=(response.shape[0], 1))
                                                -tf.reshape(X[:, condition2['variable']], shape=(X.shape[0], 1)))
                        else:
                            temp_boundary_value *= tf.reshape(X[:, condition2['variable']], shape=(X.shape[0], 1))
                boundary_value += temp_boundary_value
            if condition['value'] > 0:
                response *= (tf.constant(condition['value'], dtype='float64', shape=(response.shape[0], 1))
                            - tf.reshape(X[:, condition['variable']], shape=(response.shape[0], 1)))
            else:
                response *= tf.reshape(X[:, condition['variable']], shape=(response.shape[0], 1))

        response += boundary_value
        return response
