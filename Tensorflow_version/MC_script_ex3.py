import pandas as pd
import precision_MC as pmc

train_values_ex3 = []
interpolation_values_ex3 = []

for k in range(10):
    train, interpolation = pmc.measure_accuracy(a=0, b=1, inits=pmc.example3_inits, loss=pmc.example3_loss, exact_function=pmc.psi_e3, epochs=10000)
    train_values_ex3.append(train)
    interpolation_values_ex3.append(interpolation)

data_ex3 = {'Train': train_values_ex3, 'Interpolation': interpolation_values_ex3}

errors_ex3 = pd.DataFrame(data_ex3)
errors_ex3.to_csv('example3_mc_results_e10k_v2.csv', index=False, sep='\t')