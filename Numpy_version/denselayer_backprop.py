import numpy as np
from utilities import *
# ---------------------------------------------------------------------------------------------------------
# -----Dense Layer Class-----------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------


class Dense_Layer:
    '''
    Class representing a single layer in a densly connected feedforward network.
    It consits of the following fields:
    1. input_dim - dimension of the input vector,
    2. neuron_number - number of units in the layer,
    3. activation_function - activation function for all of the units in the layer,
    4. weights - matrix containing all the weights among the previous layer and the current layer,
        its of dimension neuron_number times input_dim,
    5. bias - bias vector added to the weighted input before activation function, its dimension is neuron_number times 1.
    It consists of the following methods described below ath their declaration:
    1. initialization,
    2. linear_response,
    3. forward_pass.
    '''
    # ---------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        input_dim,
        neuron_number,
        activation_function=sigmoid,
        zero_bias=False,
        identity=False,
    ):
        '''
        Initialization of the class:
        @params:
        1. input_dim - positive int,
        2. neuron number - positive int,
        3. activation_function - function or a callable instance of a class,
        4. zero_bias - boolean, zero bias vs bias sampled from [-0.25, 0.25]^neuron_number,
        5. identity - boolea, if weights should be and identity matrix or sampled from [-1,1] intervals.
        '''
        if isinstance(input_dim, int) and input_dim > 0:
            self.input_dim = input_dim
        else:
            raise Exception('Input dimension has to be a positive integer.')
        if isinstance(neuron_number, int) and neuron_number > 0:
            self.neuron_number = neuron_number
        else:
            raise Exception('Neuron number has to be a positive integer.')
        if callable(activation_function):
            self.activation_function = activation_function
        else:
            raise Exception('Activation function has to be callable.')
        if isinstance(zero_bias, bool):
            self.zero_bias = zero_bias
            if not zero_bias:
                self.bias = (
                    0.5 * np.random.random(size=(self.neuron_number, 1)).astype(dtype="float64") - 0.25)
            else:
                self.bias = np.zeros(
                    shape=(self.neuron_number, 1)).astype(dtype="float64")
        else:
            raise Exception('zero_bias property has to be boolean.')
        if isinstance(identity, bool):
            self.identity = identity
            if not identity:
                self.weights = (2 * np.random.random(size=(neuron_number,
                                                           self.input_dim)).astype(dtype="float64") - 1)
            else:
                self.weights = np.eye(
                    neuron_number, self.input_dim).astype(dtype="float64")
        else:
            raise Exception('identity property has to be boolean.')

    def __call__(self, x, n=0):
        '''
        Passes the input vector through the layer by using the activation function on the linear response: f(u(x))
        @params: x - numpy array, input vector
        @returns: numpy array,
        '''
        return self.activation_function(self.linear_response(x), n=n)

    def linear_response(self, x):
        '''
        Performs the linear transformation: u=Wx+b
        @params: x - float, input vector
        @returns: float
        '''
        
        if isinstance(x, np.ndarray) and len(x.shape) == 1 and self.input_dim == 1:
            x = x.reshape(-1, 1)
        elif isinstance(x, np.ndarray) and len(x.shape) > 1 and x.shape[1] == self.input_dim:
            return x @ self.weights.T + self.bias.T
        else:
            raise Exception(
                f'The argument should be a numpy array of the input dimension.')

    def derivative(self, x):
        return self.activation_function(self.linear_response(x), 1)

    def update_parameters(self, weights_change, bias_change):
        if not self.identity:
            self.weights += weights_change
        if not self.zero_bias:
            self.bias += bias_change
