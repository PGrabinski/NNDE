import numpy as np
import matplotlib.pyplot as plt
import TrialSolution
import tensorflow as tf
##################################################################################################################
# This code is a total mess as this version of the differential equation package does not support backpropagation,
# but I have already implemented it, so it may get better and the explicit gradients will be erased from here.
##################################################################################################################


##################################################################################################################
# Common
##################################################################################################################

def train_test_dataset(a=0, b=1, n_total=1000, n_train=10):
    search = lambda A, B: np.array([i for i in A.flatten().tolist() if i not in B.flatten().tolist()])
    Xe1_interpolation = np.linspace(a,b,num=n_total)
    Xe1_interpolation = Xe1_interpolation.reshape(-1,1)
    Xe1 = Xe1_interpolation[::int(n_total/n_train)].reshape(-1,1)
    Xe1_interpolation = search(Xe1_interpolation, Xe1)
    Xe1_interpolation = Xe1_interpolation.reshape(-1,1)
    return Xe1, Xe1_interpolation

# def total_loss_function(self, samples, *kwargs):
#     loss = 0
#     for i in range(samples.shape[0]):
#         loss += self.loss_function_single_point(self, samples[i])
#     return loss/samples.shape[0]

def measure_accuracy(a, b, loss,
                        inits,
                        exact_function,
                        learning_rate=1e-1,
                        epochs=1000, verbose=True,
                        equation_type='ODE'): 
    train, interpolation = train_test_dataset(a, b)
    
    solution = TrialSolution.TrialSolution(inits, n_i=1, n_h=10, n_o=1, equation_type=equation_type)
    solution.train(X=train, epochs=epochs, diff_loss=loss, optimizer_name='SGD', learning_rate=1e-1, verbose=False)
    
    predict_train = solution(train)
    predict_interpolation =  solution(interpolation)
    ground_truth_train = exact_function(train)
    ground_truth_interpolation = exact_function(interpolation)
    train_abs = np.abs(predict_train - ground_truth_train).mean()
    interpolation_abs = np.abs(predict_interpolation - ground_truth_interpolation).mean()

    return train_abs, interpolation_abs


##################################################################################################################
# Example 1
##################################################################################################################

def example1_loss(network, inputs):
    # Compute the gradients
    with tf.GradientTape() as tape:
        inputs = tf.convert_to_tensor(inputs)
        tape.watch(inputs)
        response = network(inputs)
    grads = tape.gradient(response, inputs)
    X = inputs
  
    # Compute the loss
    loss = tf.square(grads + tf.multiply(X + (1 + 3*X**2)/(1+X+X**3), response)
            - X**3 -2*X - X**2*(1 + 3*X**2)/(1+X+X**3))
    return loss

example1_inits = [{'variable':0, 'value':0, 'type':'dirichlet',
        'function':lambda X: tf.constant(1., dtype='float64', shape=(X.shape[0],1))}]

psi_e1 = lambda x:  np.exp(-0.5*x**2)/(1+x+x**3) + x**2

##################################################################################################################
# Example 2
##################################################################################################################
 
def example2_loss(network, inputs):
    with tf.GradientTape() as tape:
        inputs = tf.convert_to_tensor(inputs)
        tape.watch(inputs)
        response = network(inputs)
    grads = tape.gradient(response, inputs)
    loss = tf.square(grads + tf.multiply(tf.constant(0.2, dtype='float64'), response)
          - tf.multiply(tf.exp( tf.multiply(tf.constant(-0.2, dtype='float64'), inputs)), tf.cos(inputs)))
    return loss

example2_inits = [{'variable':0, 'value':0, 'type':'dirichlet',
        'function':lambda X: tf.constant(0., dtype='float64', shape=X.shape)}]

psi_e2 = lambda x: np.exp(-0.2*x) * np.sin(x)

##################################################################################################################
# Example 3
##################################################################################################################    

def example3_loss(network, inputs):
    # Compute the gradients
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape:
            inputs = tf.convert_to_tensor(inputs)
            tape.watch(inputs)
            tape2.watch(inputs)
            response = network(inputs)  
        grads = tape.gradient(response, inputs)
    laplace = tape2.gradient(grads, inputs)
  
    # Compute the loss
    loss = tf.square(laplace + tf.constant(0.2, dtype='float64')*grads + response
            + tf.constant(0.2, dtype='float64')*tf.exp( tf.constant(-0.2, dtype='float64') * inputs)
                    * tf.cos(inputs))
    return loss

example3_inits = [{'variable':0, 'value':0, 'type':'dirichlet',
        'function':lambda X: tf.constant(0., dtype='float64', shape=(X.shape[0],1))},
        {'variable':0, 'value':0, 'type':'neumann',
        'function':lambda X: tf.constant(1., dtype='float64', shape=(X.shape[0],1))}]

psi_e3 = lambda x: np.exp(-0.2*x) * np.sin(x)