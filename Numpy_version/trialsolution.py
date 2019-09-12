from shallownetwork import ShallowNetwork
from utilities import sigmoid
import numpy as np

class TrialSolution:
    '''
    Class representing a trial solution Psi(x)= A(x) + B(x) N(X) used to solve differential equations,
    where A(x) is a function fulfilling the boundary/initial conditions,
    B(x) is a function vanishing on the boundaries, 
    and N(x) is a shallow neural network.
    It consits of the following fields:
    1. input_dim - dimension of the input vector,
    2. hidden_dim - number of units in the hidden layer,
    3. visible_dim - dimension of the output,
    4. boundary_condition_value_function,
    5. boundary_vanishing_function,
    6. network.
    It consists of the following methods described below ath their declaration:
    1. initialization,
    2. predict,
    3. train.
    '''
    # def __init__(self, loss_function, loss_function_single_point,
    #              bias_change, hidden_weights_change, visible_weights_change,
    #              boundary_condition_value_function, boundary_vanishing_function,
    #              input_dim=1, hidden_dim=1, output_dim=1, momentum=0, learning_rate=0.1, activation_function=sigmoid):
    def __init__(self, loss_function,
                 boundary_condition_value_function, boundary_vanishing_function,
                 input_dim=1, hidden_dim=1, output_dim=1, momentum=0, learning_rate=0.1, activation_function=sigmoid, optimizer='GD'):
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
        11. visible_weights_change - callable, update rule for visible weights,
        12. boundary_condition_value_function - callable, function fulfilling the boundary/initial conditions,
        13. boundary_vanishing_function - callable, a function vanishing on the boundary.
        '''
        # Dimensions of the Shallow Network
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Defining all of the three parts of the trial solution f(X)=A(X)+B(X)N(X)
        # A(X) is a function fullfiling the boundary or initial conditions
        self.boundary_condition_value_function = boundary_condition_value_function
        # B(X) is a function vanishing on the boundary or in the initial moment
        self.boundary_vanishing_function = boundary_vanishing_function
        # N(X) is the Shallow Network
        # self.network = ShallowNetwork(
        #     input_dim=self.input_dim, hidden_dim=self.hidden_dim, visible_dim=self.output_dim,
        #     momentum=momentum, learning_rate=learning_rate, loss_function=loss_function,
        #     loss_function_single_point=loss_function_single_point, bias_change=bias_change,
        #     hidden_weights_change=hidden_weights_change, visible_weights_change=visible_weights_change,
        #     unsupervised=True, activation_function_hidden=activation_function)
        self.network = ShallowNetwork(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            activation_function_hidden=activation_function)
        self.network.compile(loss=loss_function, optimizer=optimizer)

    def __call__(self, X):
        '''
        Returns the value of the solution of the differential equation with the trial function Psi(x)=A(x)+B(x)N(x)
        @params: X - numpy array, input vector
        @returns: numpy array,
        '''
        return self.boundary_condition_value_function(X) + self.boundary_vanishing_function(X) * self.network.forward_pass(X, 0)

    def train(self, samples, epochs, batch_size, learning_rate):
        '''
        Trains the network used in the trial solution to fit it to the supplied differential equation. 
        @params:
        1. samples - numpy array, training points
        2. epochs - positive integer (secured in the ShallowNetwork class training method), number of training epochs.
        '''
        self.network.fit(X=samples, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate)
