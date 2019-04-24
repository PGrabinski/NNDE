from TrialSolution import TrialSolution
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

inits = [{'variable': 0, 'value': 0, 'type': 'dirichlet', 'ode_number': 1,
        'function': lambda X: tf.constant(0., dtype='float64', shape=(X.shape[0], 1))},
        {'variable': 0, 'value': 0, 'type': 'dirichlet', 'ode_number': 2,
        'function': lambda X: tf.constant(1., dtype='float64', shape=(X.shape[0], 1))}]

Xs = np.arange(0, 3, 0.2) + 1e-6
Xs = Xs.reshape(-1, 1)

ts = TrialSolution(conditions=inits, n_i=1, n_h=10, n_o=1,
                   ODE_number=2, equation_type='ODE_system')


def diff_loss(network, inputs):
    with tf.GradientTape() as tape1:
        inputs = tf.convert_to_tensor(inputs)
        tape1.watch(inputs)
        response1 = network(inputs)[0]
    grads1 = tape1.gradient(response1, inputs)  
    with tf.GradientTape() as tape2:
        inputs = tf.convert_to_tensor(inputs)
        tape2.watch(inputs)
        response2 = network(inputs)[1]
    grads2 = tape2.gradient(response2, inputs)
    loss1 = (grads1 - tf.cos(inputs) - response1**2 - response2 + tf.constant(1, dtype='float64')
            + inputs ** 2 + tf.sin(inputs) ** 2)
    loss2 = (grads2 - tf.constant(2., dtype='float64') * inputs 
            + (tf.constant(1., dtype='float64') + inputs**2) * tf.sin(inputs) - response1*response2)
    loss = tf.square(loss1) + tf.square(loss2)
    return loss

print(ts(tf.convert_to_tensor(Xs)))   

ts.train(X=Xs, diff_loss=diff_loss, epochs=10000, message_frequency=1000, optimizer_name='Adam', learning_rate=0.1)

pred1 = ts.call(tf.convert_to_tensor(Xs, dtype='float64'))[0].numpy()
pred2 = ts.call(tf.convert_to_tensor(Xs, dtype='float64'))[1].numpy()
plt.plot(Xs, pred1, label='prediction 1')
plt.plot(Xs, pred2, label='prediciton 2')
plt.plot(Xs, np.sin(Xs), label='true 1')
plt.plot(Xs, 1+Xs**2, label='true2')
plt.show()
