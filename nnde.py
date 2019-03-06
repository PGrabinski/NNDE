import numpy as np
from denselayer import Dense_Layer
from shallownetwork import ShallowNetwork
from utilities import kronecker_delta, ReLu, sigmoid, linear
from trialsolution import TrialSolution
from rmse import loss_function_all, loss_function_single_point, bias_change_point, hidden_weights_change_point, visible_weights_change_point
# Just gathering all the imports.
