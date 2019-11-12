import pandas as pd
import precision_MC

train_values_ex1 = []
interpolation_values_ex1 = []

for k in range(10):
    train, interpolation = precision_MC.measure_accuracy(0., 1., loss=precision_MC.example1_loss_function_single_point, bias_derivative=precision_MC.example1_bias_change,
                        visible_derivative=precision_MC.example1_visible_weights_change, hidden_derivative=precision_MC.example1_hidden_weights_change,
                        initial_value=precision_MC.example1_initial_value, vanishing_function=precision_MC.example1_boundary_vanishing, exact_function=precision_MC.psi_e1,
                        epochs=10000, verbose=False)
    train_values_ex1.append(train)
    interpolation_values_ex1.append(interpolation)

data_ex1 = {'Train': train_values_ex1, 'Interpolation': interpolation_values_ex1}

errors_ex1 = pd.DataFrame(data_ex1)
errors_ex1.to_csv('example1_mc_results_e10k.csv', index=False, sep='\t')