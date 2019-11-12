import pandas as pd
import precision_MC

train_values_ex3 = []
interpolation_values_ex3 = []

for k in range(250):
    train, interpolation = precision_MC.measure_accuracy(0., 2., loss=precision_MC.example3_loss_function_single_point, bias_derivative=precision_MC.example3_bias_change,
                        visible_derivative=precision_MC.example3_visible_weights_change, hidden_derivative=precision_MC.example3_hidden_weights_change,
                        initial_value=precision_MC.example3_initial_value, vanishing_function=precision_MC.example3_boundary_vanishing, exact_function=precision_MC.psi_e3,
                        epochs=1000, verbose=False, learning_rate=1e-1)
    train_values_ex3.append(train)
    interpolation_values_ex3.append(interpolation)

data_ex3 = {'Train': train_values_ex3, 'Interpolation': interpolation_values_ex3}

errors_ex3 = pd.DataFrame(data_ex3)
errors_ex3.to_csv('example3_mc_results.csv', index=False, sep='\t')