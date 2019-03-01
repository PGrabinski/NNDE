import numpy as np
from random import shuffle
# ---------------------------------------------------------------------------------------------------------
# -----Activation Functions--------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------
# Sigmoid activation function
# ---------------------------------------------------------------------------------------------------------


def sigmoid(x, n):
    '''
    Sigmoid activation function and its first three derivatives.
    @params:
    1. x - float, the argument,
    2. n - non-negative integer, n-th derivative
    @returns: float
    '''
    x = x.astype('float64')
    temp_sig = np.exp((-x).round(10), dtype="float64")
    temp_sig = 1 / (temp_sig + 1)
    if n == 0:
        return temp_sig
    elif n == 1:
        return temp_sig * (1 - temp_sig)
    elif n == 2:
        return temp_sig * (1 - temp_sig) * (1 - 2 * temp_sig)
    elif n == 3:
        return temp_sig * (1 - temp_sig) * (1 - 6 * temp_sig * (1 - temp_sig))
    else:
        raise Exception(
            f'Parameter n should be non-negative integer, but n = {n}')
# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------
# Linear activation function
# ---------------------------------------------------------------------------------------------------------


def linear(x, n):
    '''
    Linear activation function and its derivatives.
    @params:
    1. x - float, the argument,
    2. n - non-negative integer, n-th derivative
    @returns: float
    '''
    if n == 0:
        return x
    elif n == 1:
        return 1
    elif n > 1:
        return 0
    else:
        raise Exception(
            f'Parameter n should be non-negative integer, but n = {n}')
# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------
# ReLu activation function
# ---------------------------------------------------------------------------------------------------------


def ReLu(x, n):
    '''
    Relu function f(x)=max(0, x) and its derivatives.
    @params:
    1. x - float, the argument,
    2. n - non-negative integer, n-th derivative
    @returns: float
    '''
    if n == 0:
        return np.max(0, x)
    elif n == 1:
        return 0 if x < 0 else 1
    elif n > 1:
        return 0
    else:
        raise Exception(
            f'Parameter n should be non-negative integer, but n = {n}')
# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
# Kronecker delta function
# ---------------------------------------------------------------------------------------------------------
def kronecker_delta(i, j):
    '''
    Kronecker delta function.
    @params:
    i, j - non-negative integers
    @returns: boolean
    '''
    if isinstance(i, int) and isinstance(j, int) and i >= 0 and j >= 0:
        return i == j
    else:
        raise Exception(
            f'The arguments should be non-negative integers, but i = {i} and j = {j}.')
# ---------------------------------------------------------------------------------------------------------


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
            if not zero_bias:
                self.bias = (
                    0.5
                    * np.random.random(size=(self.neuron_number, 1)).astype(dtype="float64")
                    - 0.25
                )
            else:
                self.bias = np.zeros(
                    shape=(self.neuron_number, 1)).astype(dtype="float64")
        else:
            raise Exception('zero_bias property has to be boolean.')
        if isinstance(identity, bool):
            if not identity:
                self.weights = (
                    2
                    * np.random.random(size=(neuron_number, self.input_dim)).astype(
                        dtype="float64"
                    )
                    - 1
                )
            else:
                self.weights = np.eye(
                    neuron_number, self.input_dim).astype(dtype="float64")
        else:
            raise Exception('identity property has to be boolean.')

    def forward_pass(self, x):
        '''
        Passes the input vector through the layer by using the activation function on the linear response: f(u(x))
        @params: x - numpy array, input vector
        @returns: numpy array,
        '''
        return self.activation_function(self.linear_response(x), 0)

    def linear_response(self, x):
        '''
        Performs the linear transformation: u=Wx+b
        @params: x - float, input vector
        @returns: float
        '''
        resp = 0
        if isinstance(x, np.ndarray) and x.shape[0] == self.input_dim and x.shape[0] > 1:
            return (self.weights @ x).reshape((self.weights.shape[0], 1)) + self.bias
        elif isinstance(x, np.ndarray) and x.shape[0] == self.input_dim and x.shape[0] == 1:
            return self.weights * x + self.bias
        else:
            raise Exception(
                f'The argument should be of the layer input dimension.')

# ---------------------------------------------------------------------------------------------------------
# -----Shallow Network Class-------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------


class ShallowNetwork:
    '''
    Class representing a neural network with a single densly connected layer.
    It consits of the following fields:
    1. input_dim - dimension of the input vector,
    2. hidden_dim - number of units in the hidden layer,
    3. visible_dim - dimension of the output,
    4. input_layer - zero_bias, identity, linear activation Dense_Layer,
    5. hidden_layer - regular Dense_Layer with full parameters and given activation function
        by activation_function_hidden property,
    5. visible_layer - Dense_Layer with wights matrix, but zero bias and linear activation function,
    6. learning_rate - learning rate used in the parameters update, could be changed into a function of epochs and loss function,
    7. momentum - momentum of the training, could be change into a function of epochs and loss function,
    8. unsupervised - whether network will be used for unsupervised tasks.
    It consists of the following methods described below ath their declaration:
    1. initialization,
    2. forward_pass,
    3. single_epoch_training,
    4. network_derivative_bias,
    5. network_derivative_hidden_weights,
    6. network_derivative_visible_weights,
    7.
    '''

    # --------------------------------------------------------------------------------
    def __init__(self, loss_function, loss_function_single_point, bias_change,
                 hidden_weights_change, visible_weights_change,
                 input_dim=1, hidden_dim=1, visible_dim=1,
                 activation_function_hidden=sigmoid,
                 learning_rate=1e-1, momentum=1e-1,
                 unsupervised=False):
        '''
        Initialization of the class:
        @params:
        1. input_dim - positive int, default 1,
        2. hidden_dim - positive int, default 1,
        3. visible_dim - positive int, default 1,
        4. activation_function_hidden - callable, default sigmoid, passed to the hidden layer,
        5. learning_rate - float from interval (0,1], default 1e-1,
        6. momentum - float from interval [0, 1), default 1e-1,
        7. loss_function - callable, loss function to minimize for the whole dataset,
        8. loss_function_single_point - callable, loss function for a single point,
        9. bias_change - callable, update rule for bias,
        10. hidden_weights_change - callable, update rule for hidden weights,
        11. visible_weights_change - callable, update rule for visible weights
        12. unsupervised - boolean, default True. 
        '''
        # Supervised vs Unsupervised
        self.unsupervised = unsupervised
        # Dimensions of the layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.visible_dim = visible_dim
        # Input layer
        self.input_layer = Dense_Layer(
            input_dim=input_dim,
            neuron_number=input_dim,
            activation_function=linear,
            zero_bias=True,
            identity=True,
        )
        # Hidden layer
        self.hidden_layer = Dense_Layer(
            input_dim=input_dim, neuron_number=self.hidden_dim, activation_function=activation_function_hidden)
        # Visible (output) layer
        self.visible_layer = Dense_Layer(
            input_dim=hidden_dim,
            neuron_number=self.visible_dim,
            activation_function=linear,
            zero_bias=True,
        )
        # Learning parameters
        self.trainings_done = 0
        self.learning_rate = learning_rate
        self.learning_rate_initial = learning_rate
        self.momentum = momentum
        # Loss function
        self.loss_function_all = loss_function
        self.loss_function_single_point = loss_function_single_point
        # Parameters changes
        self.bias_change = bias_change
        self.hidden_weights_change = hidden_weights_change
        self.visible_weights_change = visible_weights_change

    # --------------------------------------------------------------------------------
    # Forward pass
    def forward_pass(self, x, n):
        '''
        Applies the n-th derivative of the Neural Network function on given vector x: N^(n)(X)
        @params: 
        1. x - numpy array, input vector
        2. n - non-negative int, degree of the derivative
        @returns: numpy array
        '''
        if isinstance(n, int) and n >= 0:
            input_result = self.input_layer.forward_pass(x)
            if n == 0:
                hidden_result = self.hidden_layer.forward_pass(input_result)
                return self.visible_layer.forward_pass(hidden_result)
            else:
                linear = self.input_layer.linear_response(input_result)
                return self.visible_layer.weights @ (
                    self.hidden_layer.weights ** n *
                    self.hidden_layer.activation_function(linear, n)
                )
        else:
            raise Exception(f'Parameter n has to be a positive integer.')

    # --------------------------------------------------------------------------------
    # -----Network derivatives in respect of the parameters---------------------------
    # --------------------------------------------------------------------------------
    # Derivative in respect of the hidden bias
    def network_derivative_bias(self, point, n):
        '''
        Computes the derivative in respect of the hidden bias of the n-th derivative of the neural network in respect of input. 
        @params:
        1. point - numpy array, input vector,
        2. n - non-negative int, degree of the derivative
        @returns: numpy array
        '''
        if isinstance(n, int) and n >= 0:
            hidden_activation_deriv_np1 = self.hidden_layer.activation_function(self.hidden_layer.linear_response(
                point), n+1)
            db_DnN = np.zeros(
                (self.visible_dim, self.input_dim, self.hidden_dim))
            for j in range(self.visible_dim):
                for i in range(self.input_dim):
                    for m in range(self.hidden_dim):
                        db_DnN[j, i, m] += (
                            self.visible_layer.weights[j, m]
                            * self.hidden_layer.weights[m, i] ** n
                            * hidden_activation_deriv_np1[m]
                        )
            return db_DnN
        else:
            raise Exception(f'Parameter n has to be a positive integer.')
    # --------------------------------------------------------------------------------
    # Derivative in respect of the hidden weights

    def network_derivative_hidden_weights(self, point, n):
        '''
        Computes the derivative in respect of the hidden weights of the n-th derivative of the neural network in respect of input. 
        @params:
        1. point - numpy array, input vector,
        2. n - non-negative int, degree of the derivative
        @returns: numpy array
        '''
        if isinstance(n, int) and n >= 0:
            dH_DnN = np.zeros((self.visible_dim, self.input_dim,
                               self.hidden_dim, self.input_dim))
            hidden_activation_deriv_n = self.hidden_layer.activation_function(self.hidden_layer.linear_response(
                point), n)
            hidden_activation_deriv_np1 = self.hidden_layer.activation_function(self.hidden_layer.linear_response(
                point), n+1)
            for j in range(self.visible_dim):
                for i in range(self.input_dim):
                    for m in range(self.hidden_dim):
                        for p in range(self.input_dim):
                            dH_DnN[j, i, m, p] = (self.visible_layer.weights[j, m]
                                                  * self.hidden_layer.weights[m, i] ** n
                                                  * hidden_activation_deriv_np1[m]
                                                  * point[p]
                                                  + self.visible_layer.weights[j, m]
                                                  * self.hidden_layer.weights[m, i] ** (n - 1)
                                                  * n
                                                  * hidden_activation_deriv_n[m]
                                                  * kronecker_delta(i, p)
                                                  )
            return dH_DnN
        else:
            raise Exception(f'Parameter n has to be a positive integer.')
    # --------------------------------------------------------------------------------
    # Derivative in respect of the visible weights

    def network_derivative_visible_weights(self, point, n):
        '''
        Computes the derivative in respect of the hidden weights of the n-th derivative of the neural network in respect of input. 
        @params:
        1. point - numpy array, input vector,
        2. n - non-negative int, degree of the derivative
        @returns: numpy array
        '''
        if isinstance(n, int) and n >= 0:
            dH_DnN = np.zeros((self.visible_dim, self.input_dim,
                               self.visible_dim, self.hidden_dim))
            hidden_activation_deriv_n = self.hidden_layer.activation_function(self.hidden_layer.linear_response(
                point), n)
            for j in range(self.visible_dim):
                for i in range(self.input_dim):
                    for m in range(self.visible_dim):
                        for p in range(self.hidden_dim):
                            dH_DnN[j, i, m, p] = (kronecker_delta(j, m)
                                                  * self.hidden_layer.weights[p, i] ** n
                                                  * hidden_activation_deriv_n[p]
                                                  )
            return dH_DnN
        else:
            raise Exception(f'Parameter n has to be a positive integer.')

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    def single_epoch_training(self, X, labels=None):
        '''
        Performs a single epoch training and updates the weights of the network
        according to the simple online gradient descent backpropagation. 
        @params: 
        1. X - numpy array, set of input vectors,
        2. labels - numpy array, in case of supervised
        '''
        if not self.unsupervised and labels is None:
            raise Exception('For supervised task, supply labels.')

        # Arrays for previous point step change
        prev_bias_change = np.zeros(
            shape=(self.hidden_dim, 1)).astype(dtype="float64")
        prev_hidden_weights_change = np.zeros(
            shape=(self.hidden_dim, self.input_dim)
        ).astype(dtype="float64")
        prev_visible_weights_change = np.zeros(
            shape=(self.visible_dim, self.hidden_dim)
        ).astype(dtype="float64")

        # Arrays for current point step change
        curr_bias_change = np.zeros(
            shape=(self.hidden_dim, 1)).astype(dtype="float64")
        curr_hidden_weights_change = np.zeros(
            shape=(self.hidden_dim, self.input_dim)
        ).astype(dtype="float64")
        curr_visible_weights_change = np.zeros(
            shape=(self.visible_dim, self.hidden_dim)
        ).astype(dtype="float64")
        label = None

        points = [i for i in range(X.shape[0])]
        shuffle(points)
        for i in points:
            # --------------------------------------------------------------------------------
            # Input vector
            point = X[i, :].reshape((self.input_dim, 1))
            if not self.unsupervised:
                label = labels[i]
            # --------------------------------------------------------------------------------
            # Derivatives of the network output
            # --------------------------------------------------------------------------------
            # Bias change for given point
            curr_bias_change = self.bias_change(self, point, label)
            # Linear combination: - learning_rate (1 - momentum) curr_change + momentum *prev_change
            curr_bias_change = (
                -curr_bias_change *
                self.learning_rate * (1 - self.momentum)
                + prev_bias_change * self.momentum)
            # Update the bias
            self.hidden_layer.bias += curr_bias_change
            # Save change for the next step
            prev_bias_change = curr_bias_change
            # --------------------------------------------------------------------------------
            # Hidden weights change for the given point
            curr_hidden_weights_change = self.hidden_weights_change(self, 
                point, label)
            # Linear combination: - learning_rate (1 - momentum) curr_change + momentum *prev_change
            curr_hidden_weights_change = (
                -curr_hidden_weights_change
                * self.learning_rate
                * (1 - self.momentum)
                + prev_hidden_weights_change * self.momentum)
            # Update the weights
            self.hidden_layer.weights += curr_hidden_weights_change
            # Save change for the next step
            prev_hidden_weights_change = curr_hidden_weights_change
            # --------------------------------------------------------------------------------
            # Visible weights change for the given point
            curr_visible_weights_change = self.visible_weights_change(self, 
                point, label)
            # Linear combination: - learning_rate (1 - momentum) curr_change + momentum *prev_change
            curr_visible_weights_change = (
                -curr_visible_weights_change
                * self.learning_rate
                * (1 - self.momentum)
                + prev_visible_weights_change * self.momentum)
            # Update the weights
            self.visible_layer.weights += curr_visible_weights_change
            # Save change for the next step
            prev_visible_weights_change = curr_visible_weights_change
        # --------------------------------------------------------------------------------
        self.trainings_done += 1
        self.update_learning_rate()

    def update_learning_rate(self):
        self.learning_rate = (
            self.learning_rate_initial * 0.1
            + 0.9 * self.learning_rate_initial *
            (np.exp(-self.trainings_done/50))
        )

    def loss_function(self, samples, labels):
        return self.loss_function_all(self, samples, labels)
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
