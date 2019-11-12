import pandas as pd
import precision_MC as pmc

train_values_ex1 = []
interpolation_values_ex1 = []

for k in range(10):
    train, interpolation = pmc.measure_accuracy(a=0, b=1, inits=pmc.example1_inits, loss=pmc.example1_loss, exact_function=pmc.psi_e1, epochs=10000)
    train_values_ex1.append(train)
    interpolation_values_ex1.append(interpolation)

data_ex1 = {'Train': train_values_ex1, 'Interpolation': interpolation_values_ex1}

errors_ex1 = pd.DataFrame(data_ex1)
errors_ex1.to_csv('example1_mc_results_e10k.csv', index=False, sep='\t')