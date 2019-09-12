import numpy as np
'''
Optimizer interface and its implementations:
1. Gradient Descent
2. Adagrad
'''


class Optimizer:
    def update_layer(self, layer, further_signal, incoming_signal, learning_rate, **kwargs):
        raise Exception(
            'This is just a template class and should not be used as an optimizer!')
        return 0

    def train(self, network, X, learning_rate, **kwargs):
        '''
        Performs a single epoch training on all samples and updates the weights of the network
        according to the supplied optimization strategy (update_layer method) via backpropagation. 
        @params: 
        1. X - numpy array, samples,
        3. learning_rate - float, learning rate.
        '''
        responses = [X]

        for layer in network.layers:
            responses.append(layer(responses[-1]))

        layers_to_train = [network.layers[len(network.layers) - i - 1]
                           for i in range(len(network.layers))]

        previous_delta = network.loss(X, Y, derivative=True, **kwargs)

        for id, layer in enumerate(layers_to_train):
            previous_delta = self.update_layer(layer=layer, further_signal=previous_delta,
                                               incoming_signal=responses[len(responses) - id - 2], learning_rate=learning_rate, id=id)


class GradientDescent(Optimizer):
    '''
    Updates according to the formula:
    \Delta \theta_{i+1} = \Delta \theta_{i} - \eta \partial_\theta Loss(\theta)
    where \eta is the learning rate, and \theta represents the parameters of the network (layer).
    '''

    def update_layer(self, layer, further_signal, incoming_signal, learning_rate, **kwargs):
        derivative = layer.derivative(incoming_signal)

        delta_error = further_signal * derivative
        weight_change = -learning_rate * delta_error.T @ incoming_signal
        bias_change = -learning_rate * \
            delta_error.T @ np.ones(shape=(incoming_signal.shape[0], 1))
        layer.update_parameters(weight_change, bias_change)
        return delta_error @ layer.weights


class Adagrad(Optimizer):
    '''
    Updates according to the formula:
    \Delta \theta_{i+1} = \Delta \theta_{i} - \eta \partial_\theta Loss(\theta) / \sqrt(G+\epsilon)
    where \eta is the learning rate, and \theta represents the parameters of the network (layer),
    G is sum of all consecutive gradients from all steps, and epsilon=1e-8.
    '''

    def __init__(self, layers):
        '''
        Initialization of the gradient squares sums placeholders.
        '''
        self.weight_gradients_history = []
        self.bias_gradients_history = []
        for layer in layers:
            self.weight_gradients_history.append(
                np.ones(shape=layer.weights.shape, dtype='float64'))
            self.bias_gradients_history.append(
                np.ones(shape=layer.bias.shape, dtype='float64'))

    def update_layer(self, layer, further_signal, incoming_signal, learning_rate, **kwargs):
        derivative = layer.derivative(incoming_signal)

        id = kwargs['id']
        weight_factor = self.weight_gradients_history[len(
            self.weight_gradients_history) - id - 1]
        bias_factor = self.bias_gradients_history[len(
            self.bias_gradients_history) - id - 1]

        delta_error = further_signal * derivative

        weight_gradient = delta_error.T @ incoming_signal
        weight_mod = np.power(weight_factor + 1e-8, -0.5)
        weight_change = -learning_rate * weight_gradient * weight_mod

        bias_gradient = delta_error.T @ np.ones(
            shape=(incoming_signal.shape[0], 1))
        bias_mod = np.power(bias_factor + 1e-8, -0.5)
        bias_change = -learning_rate * bias_gradient * bias_mod

        layer.update_parameters(weight_change, bias_change)
        self.weight_gradients_history[len(
            self.weight_gradients_history) - id - 1] += weight_gradient ** 2
        self.bias_gradients_history[len(
            self.bias_gradients_history) - id - 1] += bias_gradient ** 2

        return delta_error @ layer.weights
