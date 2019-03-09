import numpy as np
from random import shuffle
from utilities import *
from denselayer import Dense_Layer
import rmse
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
    7. loss_function
    8. update_learning_rate
    9. train_single_epoch
    10. train
    '''

    # --------------------------------------------------------------------------------
    def __init__(self, loss_function=None, loss_function_single_point=None, bias_change=None,
                 hidden_weights_change=None, visible_weights_change=None,
                 input_dim=1, hidden_dim=1, visible_dim=1,
                 activation_function_hidden=sigmoid,
                 learning_rate=1e-1, momentum=1e-1,
                 unsupervised=False, predefined_loss=None):
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
        12. unsupervised - boolean, default True,
        13. predefined_loss - string, default None.
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
        self.learning_rate_decay = 50
        if predefined_loss is None:
            if (loss_function is None or loss_function_single_point is None or
                bias_change is None or hidden_weights_change is None
                    or visible_weights_change is None):
                raise Exception(
                    'Either define loss function or use a predefined one.')
            # Loss function
            self.loss_function_all = loss_function
            self.loss_function_single_point = loss_function_single_point
            # Parameters changes
            self.bias_change = bias_change
            self.hidden_weights_change = hidden_weights_change
            self.visible_weights_change = visible_weights_change
        elif predefined_loss == 'rmse':
            # Loss function
            self.loss_function_all = rmse.loss_function_all
            self.loss_function_single_point = rmse.loss_function_single_point
            # Parameters changes
            self.bias_change = rmse.bias_change_point
            self.hidden_weights_change = rmse.hidden_weights_change_point
            self.visible_weights_change = rmse.visible_weights_change_point

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
            input_result = x  # self.input_layer.forward_pass(x)
            if n == 0:
                hidden_result = self.hidden_layer.forward_pass(input_result)
                return self.visible_layer.forward_pass(hidden_result)
            else:
                linear = self.hidden_layer.linear_response(input_result)
                network_derivative = self.visible_layer.weights @ (
                    self.hidden_layer.weights ** n *
                    self.hidden_layer.activation_function(linear, n)
                )
                return network_derivative
        else:
            raise Exception(f'Parameter n has to be a positive integer.')

    # --------------------------------------------------------------------------------
    #  Weights P vector for an arbitrary derivative

    def derivative_P_vector(self, degrees):
        if isinstance(degrees, np.ndarray) and degrees.shape[0] == self.input_dim:
            P = np.ones((self.hidden_dim, 1))
            for i in range(self.input_dim):
                P *= self.hidden_layer.weights[:,
                                               i].reshape((self.hidden_dim, 1)) ** degrees[i]
            return P
        else:
            raise Exception(
                'The degrees parameter should be a 1d numpy array of the input dimension length.')

    # --------------------------------------------------------------------------------
    # Forward pass for arbitrary derivative
    def forward_pass_arbitrary_derivative(self, x, degrees):
        '''
        Applies an arbitrary mixed derivatives to the Neural Network function
        and evaluates it on given vector x: tilde(N) = partials N^(n)(X)
        @params:
        1. x - numpy array, input vector
        2. degrees - numpy array, degrees of the derivatives
        @returns: numpy array
        '''
        n = degrees.sum()
        if n - np.floor(n) == 0 and n > 0:
            input_result = x
            linear = self.hidden_layer.linear_response(input_result)
            network_derivative = self.visible_layer.weights @ (
                self.derivative_P_vector(degrees) *
                self.hidden_layer.activation_function(linear, int(n))
            )
            return network_derivative
        else:
            raise Exception(
                f'The degrees should sum up to a positive integer.')

    # --------------------------------------------------------------------------------
    # Arbitrary derivative in respect of the hidden bias

    def arbitrary_network_derivative_bias(self, point, degrees):
        '''
        Computes the derivative in respect of the hidden bias of the n-th derivative of the neural network in respect of input.
        @params:
        1. point - numpy array, input vector,
        2. n - non-negative int, degree of the derivative
        @returns: numpy array
        '''
        n = degrees.sum()
        if n - np.floor(n) == 0 and n > 0:
            hidden_activation_deriv_np1 = self.hidden_layer.activation_function(self.hidden_layer.linear_response(
                point), int(n+1))
            db_DnN = np.zeros((self.visible_dim, self.hidden_dim))
            P = self.derivative_P_vector(degrees)
            for j in range(self.visible_dim):
                for m in range(self.hidden_dim):
                    db_DnN[j, m] += (
                        self.visible_layer.weights[j, m]
                        * P[m]
                        * hidden_activation_deriv_np1[m]
                    )
            return db_DnN
        else:
            raise Exception(f'Parameter n has to be a positive integer.')

    # --------------------------------------------------------------------------------
    # Arbitrary derivative in respect of the hidden weights

    def arbitrary_network_derivative_hidden_weights(self, point, degrees):
        '''
        Computes the derivative in respect of the hidden weights of the n-th derivative of the neural network in respect of input.
        @params:
        1. point - numpy array, input vector,
        2. n - non-negative int, degree of the derivative
        @returns: numpy array
        '''
        n = degrees.sum()
        if n - np.floor(n) == 0 and n > 0:
            dH_DnN = np.zeros(
                (self.visible_dim, self.hidden_dim, self.input_dim))
            hidden_activation_deriv_n = self.hidden_layer.activation_function(self.hidden_layer.linear_response(
                point), int(n))
            hidden_activation_deriv_np1 = self.hidden_layer.activation_function(self.hidden_layer.linear_response(
                point), int(n)+1)
            P = self.derivative_P_vector(degrees)
            for j in range(self.visible_dim):
                for m in range(self.hidden_dim):
                    for p in range(self.input_dim):
                        dH_DnN[j, m, p] = (
                            self.visible_layer.weights[j, m]
                            * P[m]
                            * hidden_activation_deriv_np1[m]
                            * point[p]
                            + self.visible_layer.weights[j, m]
                            * P[m]
                            * (self.hidden_layer.weights[m, p]**-1)
                            * degrees[p]
                            * hidden_activation_deriv_n[m]
                        )
            return dH_DnN
        else:
            raise Exception(f'Parameter n has to be a positive integer.')

   # Arbitrary derivative in respect of the visible weights

    def arbitrary_network_derivative_visible_weights(self, point, degrees):
        '''
        Computes the derivative in respect of the visible weights of the n-th derivative of the neural network in respect of input.
        @params:
        1. point - numpy array, input vector,
        2. n - non-negative int, degree of the derivative
        @returns: numpy array
        '''
        n = degrees.sum()
        if n - np.floor(n) == 0 and n > 0:
            dV_DnN = np.zeros(
                (self.visible_dim, self.visible_dim, self.hidden_dim))
            hidden_activation_deriv_n = self.hidden_layer.activation_function(self.hidden_layer.linear_response(
                point), int(n))
            P = self.derivative_P_vector(degrees)
            for j in range(self.visible_dim):
                for m in range(self.visible_dim):
                    for p in range(self.hidden_dim):
                        dV_DnN[j, m, p] = (kronecker_delta(j, m)
                                           * P[p]
                                           * hidden_activation_deriv_n[p]
                                           )
            return dV_DnN
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
            dV_DnN = np.zeros((self.visible_dim, self.input_dim,
                               self.visible_dim, self.hidden_dim))
            hidden_activation_deriv_n = self.hidden_layer.activation_function(self.hidden_layer.linear_response(
                point), n)
            for j in range(self.visible_dim):
                for i in range(self.input_dim):
                    for m in range(self.visible_dim):
                        for p in range(self.hidden_dim):
                            dV_DnN[j, i, m, p] = (kronecker_delta(j, m)
                                                  * self.hidden_layer.weights[p, i] ** n
                                                  * hidden_activation_deriv_n[p]
                                                  )
            return dV_DnN
        else:
            raise Exception(f'Parameter n has to be a positive integer.')

    # --------------------------------------------------------------------------------
    # -----Learning methods-----------------------------------------------------------
    # --------------------------------------------------------------------------------

    # Updating the parameters in a single epochs
    def single_epoch_training(self, X, labels=None):
        '''
        Performs a single epoch training on all samples and updates the weights of the network
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
        # Update learning parameters
        self.trainings_done += 1
        self.update_learning_rate()

    # --------------------------------------------------------------------------------
    # Many epoch training
    def train(self, samples, epochs, labels=None):
        '''
        Performs the given number of training epochs and prints the current loss function. 
        @params: 
        1. X - numpy array, set of input vectors,
        2. labels - numpy array, in case of supervised,
        3. epochs - positive integer.
        '''
        if isinstance(epochs, int) and epochs > 0:
            self.learning_rate_decay = 0.2 * epochs
            for i in range(epochs):
                if labels is None:
                    self.single_epoch_training(X=samples)
                else:
                    self.single_epoch_training(X=samples, labels=labels)
                print(
                    f'Epoch: {i+1} Loss function: {self.loss_function(samples=samples, labels=labels)}')
        else:
            raise Exception(f'Parameter epochs has to be a positive integer.')

    def update_learning_rate(self):
        '''
        Updates the learning rate according basing on the supplied decay period.
        '''
        self.learning_rate = (
            self.learning_rate_initial * 0.1
            + 0.9 * self.learning_rate_initial *
            (np.exp(-self.trainings_done/self.learning_rate_decay))
        )

    def loss_function(self, samples, labels):
        '''
        Wrapper for the supplied loss function.
        @params: 
        1. X - numpy array, set of input vectors,
        2. labels - numpy array, in case of supervised.
        '''
        return self.loss_function_all(self, samples, labels)
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
