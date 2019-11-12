import pandas as pd
import precision_MC as pmc

train_values_ex2 = []
interpolation_values_ex2 = []

for k in range(10):
    train, interpolation = pmc.measure_accuracy(a=0, b=1, inits=pmc.example2_inits, loss=pmc.example2_loss, exact_function=pmc.psi_e2, epochs=10000)
    train_values_ex2.append(train)
    interpolation_values_ex2.append(interpolation)

data_ex2 = {'Train': train_values_ex2, 'Interpolation': interpolation_values_ex2}

errors_ex2 = pd.DataFrame(data_ex2)
errors_ex2.to_csv('example2_mc_results_e10k.csv', index=False, sep='\t')