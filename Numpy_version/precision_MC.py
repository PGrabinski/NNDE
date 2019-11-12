import numpy as np
import matplotlib.pyplot as plt
import nnde

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
    Xe1_interpolation = Xe1_interpolation.reshape((Xe1_interpolation.shape[0],1,1))
    Xe1 = Xe1_interpolation[::int(n_total/n_train)]
    Xe1_interpolation = search(Xe1_interpolation, Xe1)
    Xe1_interpolation = Xe1_interpolation.reshape((Xe1_interpolation.shape[0],1,1))
    return Xe1, Xe1_interpolation

def total_loss_function(self, samples, *kwargs):
    loss = 0
    for i in range(samples.shape[0]):
        loss += self.loss_function_single_point(self, samples[i])
    return loss/samples.shape[0]


def measure_accuracy(a, b, loss,
                        bias_derivative,
                        hidden_derivative,
                        visible_derivative,
                        initial_value,
                        vanishing_function,
                        exact_function, learning_rate=1e-1,
                        epochs=1000, verbose=True): 
    train, interpolation = train_test_dataset(a, b)
    
    solution = nnde.TrialSolution(loss_function=total_loss_function,
                                        loss_function_single_point=loss,
                                        bias_change=bias_derivative,
                                        hidden_weights_change=hidden_derivative,
                                        visible_weights_change=visible_derivative,
                                        boundary_condition_value_function=initial_value,
                                        boundary_vanishing_function=vanishing_function,
                                        input_dim=1, hidden_dim=10, output_dim=1,
                                        learning_rate=learning_rate, momentum=1e-1, verbose=verbose)
    solution.train(train, epochs)
    
    predict_train = np.array([solution.predict(train[i]) for i in range(train.shape[0])]).reshape((train.shape[0],))
    predict_interpolation =  np.array([solution.predict(interpolation[i]) for i in range(interpolation.shape[0])]).reshape((interpolation.shape[0],))
    ground_truth_train = np.array([exact_function(train[i]) for i in range(train.shape[0])]).flatten()
    ground_truth_interpolation = np.array([exact_function(interpolation[i]) for i in range(interpolation.shape[0])]).flatten()
    train_abs = np.abs(predict_train - ground_truth_train).mean()
    interpolation_abs = np.abs(predict_interpolation - ground_truth_interpolation).mean()

    return train_abs, interpolation_abs


##################################################################################################################
# Example 1
##################################################################################################################

def example1_loss_function_single_point(self, point, non_squared=False, *kwargs):
    N = self.forward_pass(point, 0)
    dN = self.forward_pass(point, 1)
    loss = (
        point * dN + N + (point + (1 + 3 * point ** 2)/(1 + point + point ** 3)) * (1 + point * N) 
        - point ** 3 - 2 * point - point ** 2 *(1 + 3 * point ** 2)/(1 + point + point ** 3)
        )
    if not non_squared:
        loss = loss ** 2
    return loss[0,0]

def example1_bias_change(self, point, label, *kwargs):
    db = np.zeros((self.hidden_dim, 1)).astype(dtype="float64")
    loss_sqrt = self.loss_function_single_point(self, point, non_squared=True)
    db_N = self.network_derivative_bias(point, 0)
    db_DN = self.network_derivative_bias(point, 1)
    point = point.reshape((1,))
    for m in range(self.hidden_dim):
        db[m] += 2 * loss_sqrt * (
        point * db_DN[0, 0, m] + db_N[0, 0, m] + (point + (1 + 3 * point ** 2)/(1 + point + point ** 3)) * point * db_N[0, 0, m])
    return db

def example1_hidden_weights_change(self, point, *kwargs):
    dH = np.zeros((self.hidden_dim, self.input_dim)).astype(dtype="float64")
    loss_sqrt = self.loss_function_single_point(self, point, non_squared=True)
    dH_N = self.network_derivative_hidden_weights(point, 0)
    dH_DN = self.network_derivative_hidden_weights(point, 1)
    for m in range(self.hidden_dim):
        for p in range(self.input_dim):
            dH[m, p] += 2 * loss_sqrt * (
                point * dH_DN[0, 0, m, p] + dH_N[0, 0, m, p] + (point + (1 + 3 * point ** 2)/(1 + point + point ** 3)) * point * dH_N[0, 0, m, p])
    return dH

def example1_visible_weights_change(self, point, *kwargs):
    dV = np.zeros((self.visible_dim, self.hidden_dim)).astype(dtype="float64")
    loss_sqrt = self.loss_function_single_point(self, point, non_squared=True)
    dV_N = self.network_derivative_visible_weights(point, 0)
    dV_DN = self.network_derivative_visible_weights(point, 1)
    for m in range(self.visible_dim):
        for p in range(self.hidden_dim):
            dV[m, p] += 2 * loss_sqrt * (
                point * dV_DN[0, 0, m, p] + dV_N[0, 0, m, p] + (point + (1 + 3 * point ** 2)/(1 + point + point ** 3)) * point * dV_N[0, 0, m, p])
    return dV

def example1_initial_value(x):
    return 1

def example1_boundary_vanishing(x):
    return x

psi_e1 = lambda x:  np.exp(-0.5*x**2)/(1+x+x**3) + x**2


##################################################################################################################
# Example 2
##################################################################################################################

def example2_loss_function_single_point(self, point, non_squared=False, *kwargs):
    N = self.forward_pass(point, 0)
    dN = self.forward_pass(point, 1)
    loss = (
        point * dN + N + 0.2 * point * N - np.exp(-0.2*point)*np.cos(point)
        )
    if not non_squared:
        loss = loss ** 2
    return loss[0,0]

def example2_bias_change(self, point, label, *kwargs):
    db = np.zeros((self.hidden_dim, 1)).astype(dtype="float64")
    loss_sqrt = self.loss_function_single_point(self, point, non_squared=True)
    db_N = self.network_derivative_bias(point, 0)
    db_DN = self.network_derivative_bias(point, 1)
    point = point.reshape((1,))
    for m in range(self.hidden_dim):
        db[m] += 2 * loss_sqrt * ( point * db_DN[0, 0, m] + db_N[0, 0, m] + 0.2 * point * db_N[0, 0, m])
    return db

def example2_hidden_weights_change(self, point, *kwargs):
    dH = np.zeros((self.hidden_dim, self.input_dim)).astype(dtype="float64")
    loss_sqrt = self.loss_function_single_point(self, point, non_squared=True)
    dH_N = self.network_derivative_hidden_weights(point, 0)
    dH_DN = self.network_derivative_hidden_weights(point, 1)
    for m in range(self.hidden_dim):
        for p in range(self.input_dim):
            dH[m, p] += 2 * loss_sqrt * ( point * dH_DN[0, 0, m, p] + dH_N[0, 0, m, p] + 0.2 * point * dH_N[0, 0, m, p])
    return dH

def example2_visible_weights_change(self, point, *kwargs):
    dV = np.zeros((self.visible_dim, self.hidden_dim)).astype(dtype="float64")
    loss_sqrt = self.loss_function_single_point(self, point, non_squared=True)
    dV_N = self.network_derivative_visible_weights(point, 0)
    dV_DN = self.network_derivative_visible_weights(point, 1)
    for m in range(self.visible_dim):
        for p in range(self.hidden_dim):
            dV[m, p] += 2 * loss_sqrt * (point * dV_DN[0, 0, m, p] + dV_N[0, 0, m, p] + 0.2 * point * dV_N[0, 0, m, p])
    return dV
    
psi_e2 = lambda x: np.exp(-0.2*x) * np.sin(x)

def example2_initial_value(x):
    return 0

def example2_boundary_vanishing(x):
    return x

##################################################################################################################
# Example 3
##################################################################################################################    

def example3_loss_function_single_point(self, point, non_squared=False, *kwargs):
    N = self.forward_pass(point, 0)
    dN = self.forward_pass(point, 1)
    d2N = self.forward_pass(point, 2)
    loss = ( 2 * N + 4 * point * dN + point ** 2 * d2N + 0.2 * (1 + 2 * point * N + point ** 2 * dN)
        + point + point ** 2 * N + 0.2 * np.exp(-0.2*point)*np.cos(point)
        )
    if not non_squared:
        loss = loss ** 2
    return loss[0,0]

def example3_bias_change(self, point, label, *kwargs):
    db = np.zeros((self.hidden_dim, 1)).astype(dtype="float64")
    loss_sqrt = self.loss_function_single_point(self, point, non_squared=True)
    db_N = self.network_derivative_bias(point, 0)
    db_DN = self.network_derivative_bias(point, 1)
    db_D2N = self.network_derivative_bias(point, 2)
    point = point.reshape((1,))
    for m in range(self.hidden_dim):
        db[m] += 2 * loss_sqrt * ( 2 * db_N[0, 0, m] + 4 * point * db_DN[0, 0, m] + point ** 2 * db_D2N[0, 0, m]
                                + 0.2 * (2 * point * db_N[0, 0, m] + point ** 2 * db_DN[0, 0, m])
                                + point ** 2 * db_N[0, 0, m] 
        )
    return db

def example3_hidden_weights_change(self, point, *kwargs):
    dH = np.zeros((self.hidden_dim, self.input_dim)).astype(dtype="float64")
    loss_sqrt = self.loss_function_single_point(self, point, non_squared=True)
    dH_N = self.network_derivative_hidden_weights(point, 0)
    dH_DN = self.network_derivative_hidden_weights(point, 1)
    dH_D2N = self.network_derivative_hidden_weights(point, 2)
    for m in range(self.hidden_dim):
        for p in range(self.input_dim):
            dH[m, p] += 2 * loss_sqrt * (2 * dH_N[0, 0, m, p] + 4 * point * dH_DN[0, 0, m, p] + point ** 2 * dH_D2N[0, 0, m, p]
                                + 0.2 * (2 * point * dH_N[0, 0, m, p] + point ** 2 * dH_DN[0, 0, m, p])
                                + point ** 2 * dH_N[0, 0, m, p]
        )
    return dH

def example3_visible_weights_change(self, point, *kwargs):
    dV = np.zeros((self.visible_dim, self.hidden_dim)).astype(dtype="float64")
    loss_sqrt = self.loss_function_single_point(self, point, non_squared=True)
    dV_N = self.network_derivative_visible_weights(point, 0)
    dV_DN = self.network_derivative_visible_weights(point, 1)
    dV_D2N = self.network_derivative_visible_weights(point, 2)
    for m in range(self.visible_dim):
        for p in range(self.hidden_dim):
            dV[m, p] += 2 * loss_sqrt * (2 * dV_N[0, 0, m, p] + 4 * point * dV_DN[0, 0, m, p] + point ** 2 * dV_D2N[0, 0, m, p]
                                + 0.2 * (2 * point * dV_N[0, 0, m, p] + point ** 2 * dV_DN[0, 0, m, p])
                                + point ** 2 * dV_N[0, 0, m, p]   
        )
    return dV

psi_e3 = lambda x: np.exp(-0.2*x) * np.sin(x)

def example3_initial_value(x):
    return x

def example3_boundary_vanishing(x):
    return x ** 2