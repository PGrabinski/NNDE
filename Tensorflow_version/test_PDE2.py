from TrialSolution import TrialSolution
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

bcs = [{'variable':0, 'value':0, 'type':'dirichlet',
        'function':lambda X: tf.reshape(X[:,1]**3, shape=(X.shape[0],1))},
        {'variable':0, 'value':1, 'type':'dirichlet',
        'function':lambda X: tf.reshape((tf.constant(1., dtype='float64', shape=(X.shape[0],1))+
        tf.reshape(X[:,1]**3, shape=(X.shape[0],1)))*tf.constant(np.exp(-1), dtype='float64', shape=(X.shape[0], 1)), shape=(X.shape[0], 1))},
        {'variable':1, 'value':0, 'type':'dirichlet',
        'function':lambda X: tf.reshape(X[:,0]*tf.exp(-X[:,0]), shape=(X.shape[0],1))},
        {'variable':1, 'value':1, 'type':'dirichlet',
        'function':lambda X: tf.reshape(
            (tf.reshape(X[:,0], shape=(X.shape[0],1))+tf.constant(1., dtype='float64', shape=(X.shape[0],1)))
            * tf.reshape(tf.exp(-X[:,0]), shape=(X.shape[0],1))
            , shape=(X.shape[0],1))}]

n_samples = 10
X_p = np.linspace(0, 1, n_samples)
Y_p = np.linspace(0, 1, n_samples)
X_p, Y_p = np.meshgrid(X_p, Y_p)
X_p = X_p.flatten()
Y_p = Y_p.flatten()
samples = np.array([X_p, Y_p]).T

def call(self, X):
    response = self.hidden_layer(X)
    response = self.output_layer(response)
    x = tf.reshape(X[:,0], shape=(response.shape[0],1))
    y = tf.reshape(X[:,1], shape=(response.shape[0],1))
    e_1 = tf.constant(np.exp(-1.), dtype='float64', shape=(X.shape[0],1))
    one = tf.constant(1., dtype='float64', shape=(response.shape[0],1))
    response *= x * (one - x)
    response *= y * (one - y)
    response += (one - x) * y **3
    response += x*(one + y**3)*e_1
    response += (one - y) * x * (tf.exp(-x)-e_1)
    response += y *((one + x) * tf.exp(-x) - one + x
                    - tf.constant(2*np.exp(-1), dtype='float64', shape=(X.shape[0],1)) * x )
    return response

ts = TrialSolution(conditions=bcs, n_i=2, n_h=10, n_o=1,
                   ODE_number=2, equation_type='PDE', call_method=call)
ts2 = TrialSolution(conditions=bcs, n_i=2, n_h=10, n_o=1,
                   ODE_number=2, equation_type='PDE')

def diff_loss(network, inputs):
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape:
            inputs = tf.convert_to_tensor(inputs)
            tape.watch(inputs)
            tape2.watch(inputs)
            response = network(inputs)  
        grads = tape.gradient(response, inputs)
    laplace = tape2.gradient(grads, inputs)
    two = tf.constant(2, dtype='float64')
    loss = tf.square(laplace[:,0] + laplace[:,1]
                    - tf.exp(-inputs[:,0])*(inputs[:,0] - two  + inputs[:,1]**3 + inputs[:,1]))
    return loss

# (ts(tf.convert_to_tensor(samples)))   

ts2.train(X=samples, diff_loss=diff_loss, epochs=100000, message_frequency=100, optimizer_name='Adam', learning_rate=0.1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
new_shape = int(np.sqrt(samples.shape[0]))
pred = ts(samples)
pred2 = ts2(samples)
# print(pred)
Ze5sol = tf.reshape(pred2, shape=(samples.shape[0], 1)).numpy()
ax.plot_surface(X=samples[:,0].reshape((new_shape, new_shape)), Y=samples[:,1].reshape((new_shape, new_shape)), Z=Ze5sol.reshape((new_shape, new_shape)), label='Numerical - Training')
plt.show()
