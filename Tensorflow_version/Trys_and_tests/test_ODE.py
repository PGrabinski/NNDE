from TrialSolution import TrialSolution
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

inits = [{'variable':0, 'value':0, 'type':'dirichlet',
        'function':lambda X: tf.constant(0., dtype='float64', shape=(X.shape[0],1))},
        {'variable':0, 'value':0, 'type':'neumann',
        'function':lambda X: tf.constant(1., dtype='float64', shape=(X.shape[0],1))}]

X_train = np.arange(0, 2., 0.2) + 1e-8
X_train = X_train.reshape(-1,1)
X_test = np.arange(0, 2., 0.01) + 1e-8
X_test = X_test.reshape(-1,1) 

ts = TrialSolution(conditions=inits, n_i=1, n_h=10, n_o=1, equation_type='ODE')


def diff_loss(network, inputs):
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

print(ts(tf.convert_to_tensor(X_train)))   

ts.train(X=X_train, diff_loss=diff_loss, epochs=10000, message_frequency=1000, optimizer_name='Adam', learning_rate=0.1)

pred_train = ts.call(tf.convert_to_tensor(X_train, dtype='float64')).numpy()
pred_test = ts(tf.convert_to_tensor(X_test, dtype='float64')).numpy()
plt.scatter(X_train, pred_train, c='r', label='Numerical - Training', marker='+', s=30)
plt.plot(X_test, pred_test, c='g', label='Numerical - Test')
plt.plot(X_test, np.exp(-0.2*X_test)*np.sin(X_test), c='b', label='Analytic')
plt.legend()
plt.show()