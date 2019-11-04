import numpy as np
from random import shuffle
from utilities import sigmoid, linear
from denselayer import Dense_Layer
import optimizers

# ---------------------------------------------------------------------------------------------------------
# -----Shallow Network Class-------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------


class ShallowNetwork:
    '''
    Class representing a neural network with a single densly connected layer.
    It consits of the following fields:
    1. input_dim - dimension of the input vector,
    2. hidden_dim - number of units in the hidden layer,
    3. output_dim - dimension of the output,

    4. input_layer - zero_bias, identity, linear activation Dense_Layer,

    5. hidden_layer - regular Dense_Layer with full parameters and given activation function
        by activation_function_hidden property,
    5. output_layer - Dense_Layer with wights matrix, but zero bias and linear activation function,/
    It consists of the following methods described below ath their declaration:
    1. initialization,
    2. forward_pass,
    3. loss_function,
    4. gradient_descent_single_epoch
    5. train
    '''

    # --------------------------------------------------------------------------------
    def __init__(self, input_dim=1, hidden_dim=1, output_dim=1, activation_function_hidden=sigmoid):
        '''
        Initialization of the class:
        @params:
        1. input_dim - positive int, default 1,
        2. hidden_dim - positive int, default 1,
        3. output_dim - positive int, default 1,
        4. activation_function_hidden - callable, default sigmoid, passed to the hidden layer.
        '''
        # Dimensions of the layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.compiled = False
        self.layers = []
        # Input layer
        self.layers.append(Dense_Layer(
            input_dim=input_dim,
            neuron_number=input_dim,
            activation_function=linear,
            zero_bias=True,
            identity=True,
        ))
        # Hidden layer
        self.layers.append(Dense_Layer(
            input_dim=input_dim,
            neuron_number=self.hidden_dim,
            activation_function=activation_function_hidden))

        # Output layer
        self.layers.append(Dense_Layer(
            input_dim=hidden_dim,
            neuron_number=self.output_dim,
            activation_function=linear,
            zero_bias=False
        ))

    # --------------------------------------------------------------------------------
    # Forward pass

    def __call__(self, x, n=0):
        '''
        Passes the input vector through the network N(X)
        @params:
            1. x - numpy array, input vector,
            2. n - non-negative int, degree of the derivative
        @returns: numpy array
        '''
        if isinstance(n, int) and n >= 0:
            response = x
            if n == 0:
                for layer in self.layers:
                    response = layer(response)
                return response
            # else:
            #     linear = self.hidden_layer.linear_response(input_result)
            #     network_derivative = self.visible_layer.weights @ (
            #         self.hidden_layer.weights ** n *
            #         self.hidden_layer.activation_function(linear, n)
            #     )
            #     return network_derivative
        else:
            raise Exception(f'Parameter n has to be a positive integer.')
    # --------------------------------------------------------------------------------
    # -----Learning methods-----------------------------------------------------------
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Multiple epoch training
    def fit(self, X, epochs, batch_size, learning_rate, verbose=True,
            message_frequency=1, **kwargs):
        '''
        Performs the given number of training epochs and prints the current loss function. 
        @params: 
        1. X - numpy array, set of input vectors,
        3. epochs - positive integer, number of steps in the training
        4. verbose, boolean, wether to print training messages,
        5. learning_rate - float, learning rate,
        6. batch_size - non-negative int, number of samples taken into one batch.
        '''
        if not self.compiled:
            raise Exception('The model is not compiled!')
        if not isinstance(X, np.ndarray):
            raise Exception('X should be a numpy array.')
        if not isinstance(batch_size, int) or batch_size < -1 or batch_size == 0:
            raise Exception(
                'batch_size parameter should ba a non-negative integer or -1')
        if not isinstance(learning_rate, float):
            raise Exception(
                'learning_rate should be a floating point parameter')
        if isinstance(epochs, int) and epochs > 0:
            for epoch in range(epochs):
                if batch_size == -1:
                    batch_size = X.shape[0]
                batched_counter = 0
                while batched_counter < X.shape[0]:
                    if batched_counter + batch_size <= X.shape[0]:
                        self.optimizer.train(network=self, X=X[batched_counter:batched_counter+batch_size, :],
                                             learning_rate=learning_rate, **kwargs)
                    else:
                        self.optimizer.train(network=self, X=X[batched_counter:, :],
                                             learning_rate=learning_rate, **kwargs)
                    batched_counter += batch_size
                if verbose and epoch % message_frequency == 0:
                    print(
                        f'Epoch: {epoch+1} Loss function: {self.loss(samples=X, **kwargs)}')
            print(
                f'Final loss function: {self.loss(samples=X, **kwargs)}')
        else:
            raise Exception(f'Parameter epochs has to be a positive integer.')

    # Setting loss and optimizer
    def compile(self, loss, optimizer):
        '''
        Sets loss and optimizer for the training procedure. 
        @params: 
        1. loss_name - string, name of the loss, so far: mse, rmse, cross_entropy, chi_squared,
        2. optimizer - string, name of the optimizer, so far: GD (gradient descent) and Adagrad.
        '''
        self.loss_function_to_wrap = loss
        if optimizer == 'GD':
            self.optimizer = optimizers.GradientDescent()
        elif optimizer == 'Adagrad':
            self.optimizer = optimizers.Adagrad(self.layers)
        else:
            raise Exception(
                'Only SGD and Adagrad are accepted as the optimizers.')
        self.compiled = True

    def loss(self, samples, derivative=False, **kwargs):
        '''
        Wrapper for the supplied loss function.
        @params: 
        1. samples - numpy array, set of input vectors,
        2. labels - numpy array, in case of supervised,
        3. derivative - boolean, for training purposes.
        '''
        return self.loss_function_to_wrap(self, samples, derivative, **kwargs)
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
