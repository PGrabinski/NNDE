import numpy as np
'''
Utilities for the RMSE loss function
'''

def loss_function_single_point(self, point, ground_truth):
        N = self.forward_pass(point, 0)
        loss = np.sqrt(((N - ground_truth) ** 2).sum())
        return loss

def loss_function_all(self, samples, labels):
        loss = 0
        n_inv = len(samples) ** -1
        for i in range(samples.shape[0]):
            loss += self.loss_function_single_point(
                self, samples[i], labels[i]) ** 2
        loss *= n_inv
        loss = np.sqrt(loss)
        return loss

def bias_change_point(self, point, label):
    db = np.zeros((self.hidden_dim, 1)).astype(dtype="float64")
    change = self.forward_pass(point, 0) - label
    loss = self.loss_function_single_point(self, point, label)
    db_N = self.network_derivative_bias(point, 0)
    for m in range(self.hidden_dim):
        for j in range(self.visible_dim):
            db[m] += change[j] * db_N[j, 0, m]
    db /= loss
    return db   

def hidden_weights_change_point(self, point, label):
    dH = np.zeros((self.hidden_dim, self.input_dim)).astype(dtype="float64")
    change = self.forward_pass(point, 0) - label
    loss = self.loss_function_single_point(self, point, label)
    dH_N = self.network_derivative_hidden_weights(point, 0)
    for m in range(self.hidden_dim):
        for p in range(self.input_dim):
            for j in range(self.visible_dim):
                dH[m, p] += change[j] * dH_N[j, 0, m, p]
    dH /= loss
    return dH

def visible_weights_change_point(self, point, label):
    dV = np.zeros((self.visible_dim, self.hidden_dim)).astype(dtype="float64")
    change = self.forward_pass(point, 0) - label
    loss = self.loss_function_single_point(self, point, label)
    dV_N = self.network_derivative_visible_weights(point, 0)
    for m in range(self.visible_dim):
        for p in range(self.hidden_dim):
            for j in range(self.visible_dim):
                dV[m, p] += change[j] * dV_N[j, 0, m, p]
    dV /= loss
    return dV