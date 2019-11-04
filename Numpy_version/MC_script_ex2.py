import pandas as pd
import precision_MC

train_values_ex2 = []
interpolation_values_ex2 = []

for k in range(2):
    train, interpolation = precision_MC.measure_accuracy(0., 2., loss=precision_MC.example2_loss_function_single_point, bias_derivative=precision_MC.example2_bias_change,
                        visible_derivative=precision_MC.example2_visible_weights_change, hidden_derivative=precision_MC.example2_hidden_weights_change,
                        initial_value=precision_MC.example2_initial_value, vanishing_function=precision_MC.example2_boundary_vanishing, exact_function=precision_MC.psi_e2,
                        epochs=1000, verbose=False)
    train_values_ex2.append(train)
    interpolation_values_ex2.append(interpolation)

data_ex2 = {'Train': train_values_ex2, 'Interpolation': interpolation_values_ex2}

errors_ex2 = pd.DataFrame(data_ex2)
errors_ex2.to_csv('example2_mc_resuls.csv', index=False, sep='\t')