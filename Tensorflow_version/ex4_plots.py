import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from TrialSolution import TrialSolution

def const_function(c):
    def function(X):
        return tf.constant(c, dtype='float64', shape=(X.shape[0],1))
    return function

inits = [{'variable':0, 'value':0, 'type':'dirichlet', 'ode_number':1,
        'function': const_function(0.) },
        {'variable':0, 'value':0, 'type':'dirichlet', 'ode_number':2,
        'function': const_function(1.)}]

X_train = np.arange(0, 3, 0.2) + 1e-8
X_train = X_train.reshape(-1,1)
X_test = np.arange(0, 3, 0.01) + 1e-8
X_test = X_test.reshape(-1,1)

def diff_loss(network, inputs):
    with tf.GradientTape() as tape1: 
        with tf.GradientTape() as tape2:
            inputs = tf.convert_to_tensor(inputs)
            tape1.watch(inputs)
            tape2.watch(inputs)
            response = network(inputs)
            psi1 = tf.reshape(response[:, 0], shape=(inputs.shape[0],1))
            psi2 = tf.reshape(response[:, 1], shape=(inputs.shape[0],1))
    grads1 = tape1.gradient(psi1, inputs)
    grads2 = tape2.gradient(psi2, inputs)
    loss1 = (grads1 - tf.cos(inputs) - psi1**2 - psi2 + tf.constant(1, dtype='float64')
           + inputs ** 2 + tf.sin(inputs) ** 2)
    loss2 = (grads2 - 2. * inputs 
           + (1. + inputs**2) * tf.sin(inputs) - psi1*psi2)
    loss = tf.square(loss1) + tf.square(loss2)
    loss = tf.math.reduce_sum(loss)
    return loss

def custom_call(self, X):
    X = tf.convert_to_tensor(X)
    responses = []
    for i in range(self.ODE_number):
        resp = self.hidden_layers[i](X)
        responses.append(self.output_layers[i](resp))
    responses[0]*=X
    responses[0]+=0.
    responses[1]*=X
    responses[1]+=1.
    response = tf.concat(responses, axis=1)
    return response

for k in range(10, 110):
    ts = TrialSolution(conditions=inits, n_i=1, n_h=10, n_o=1, ODE_number=2, equation_type='ODE_system', call_method=custom_call)

    ts.train(X=X_train, diff_loss=diff_loss, epochs=10000, message_frequency=1000, optimizer_name='SGD', learning_rate=1e-3, verbose=False)

    plt.clf()
    pred1 = ts(tf.convert_to_tensor(X_train, dtype='float64'))[:, 0].numpy()
    pred2 = ts(tf.convert_to_tensor(X_train, dtype='float64'))[:, 1].numpy()
    pred1_test = ts(tf.convert_to_tensor(X_test, dtype='float64'))[:, 0].numpy()
    pred2_test = ts(tf.convert_to_tensor(X_test, dtype='float64'))[:, 1].numpy()
    plt.scatter(X_train, pred1, label='Numerical - Train - 1', marker='x', s=500, c='red')
    plt.scatter(X_train, pred2, label='Numerical - Train - 2', marker='o', s=500, c='orange')
    plt.plot(X_test, pred1_test, label='Numerical - Test - 1', c='xkcd:sky blue', linewidth=5)
    plt.plot(X_test, pred2_test, label='Numerical - Test - 2', c='blue', linewidth=5)
    plt.plot(X_test, np.sin(X_test), label='Analytic - 1', c='xkcd:goldenrod', linewidth=5)
    plt.plot(X_test, 1+X_test**2, label='Analytic - 2', c='tab:olive', linewidth=5)
    plt.xlim((-0.2,3.2))
    plt.ylim((-0.5,10.5))
    plt.xlabel(r'$x$', fontsize='50')
    plt.ylabel(r'$y$', fontsize='50')
    plt.title('Example 4', fontsize='60')
    plt.gcf().set_size_inches(30, 22.5)
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.legend(borderpad=1, fontsize='35')
    plt.savefig(f'plots/example4_mc_{k}.jpg')
    # plt.show()