import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ODE_TrialSolution import ODE_TrialSolution
from PDE_TrialSolution import PDE_TrialSolution
from ODE_System_TrialSolution import ODE_System_TrialSolution


class TrialSolution(tf.keras.models.Model):
    def __init__(self, conditions, n_i, n_h, n_o=1, activation='sigmoid', equation_type='ODE', ODE_number=None, call_method=None):
        super(TrialSolution, self).__init__()

        # Type of the problem
        self.equation_type = equation_type

        # Dimensions of the network
        self.n_i = n_i
        self.n_h = n_h
        self.n_o = n_o

        # Boundary conditions
        self.conditions = conditions

        if self.equation_type == 'ODE':
            self.trial_solution = ODE_TrialSolution(conditions=conditions, n_i=n_i, n_h=n_h, n_o=n_o,
                                                    activation=activation, call_method=call_method)
        elif self.equation_type == 'PDE':
            self.trial_solution = PDE_TrialSolution(conditions=conditions, n_i=n_i, n_h=n_h, n_o=n_o,
                                                    activation=activation, call_method=call_method)
        elif self.equation_type == 'ODE_system':
            if ODE_number is None or (ODE_number < 1 and not isinstance(ODE_number, int)):
                raise Exception(
                    'For ODE system provide positive integer parameter ODE_number.')
            self.ODE_number = ODE_number
            self.trial_solution = ODE_System_TrialSolution(
                conditions=conditions, ODE_number=ODE_number, n_i=n_i, n_h=n_h, n_o=n_o,
                activation=activation, call_method=call_method)

    def call(self, X):
        return self.trial_solution(X)

    def train(self, X, diff_loss, epochs, verbose=True, message_frequency=1, learning_rate=0.1, optimizer_name='Adam'):
        if not isinstance(epochs, int) or epochs < 1:
            raise Exception('epochs parameter should be a positive integer.')
        if not isinstance(message_frequency, int) or message_frequency < 1:
            raise Exception(
                'message_frequency parameter should be a positive integer.')
        optimizer = None
        if optimizer_name == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise Exception(
                'Chosen optimizer is not supported.')
        # train_loss = tf.keras.metrics.Mean('train')
        @tf.function
        def train_step(X):
            # print('pre loss')
            with tf.GradientTape() as tape:
                loss = diff_loss(self, X)
            # print('post tape')
            gradients = tape.gradient(
                loss, self.trial_solution.trainable_variables)
            # print('post gradient')
            optimizer.apply_gradients(
                zip(gradients, self.trial_solution.trainable_variables))
            # print('post apply')
        # print('Before epoch loop')
        for epoch in range(epochs):
            # print(X.shape)
            for x in X:
                # print(x.shape)
                # print(X.shape[1])
                x_tensor = tf.reshape(x, shape=(1, X.shape[1]))
                # print(x_tensor.shape)
                train_step(x_tensor)
            # train_loss(diff_loss(self,X))
            if verbose and ((epoch+1) % message_frequency == 0):
                # print(f'Epoch: {epoch+1} Loss: {train_loss.result().numpy()}')
                print(f'Epoch: {epoch+1} Loss: {diff_loss(self, X)}')
