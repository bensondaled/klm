##

'''
This is an example script for visualizing the results from results.csv

Everything here can be done in excel too.
'''

import os
import matplotlib.pyplot as pl

input_folder = '/Volumes/KLM2/tdp43_examples/'
results_file = os.path.join(input_folder, 'results_colocalization.csv')

results = pd.read_csv(results_file)

#conditions = ['FUS_1', 'FUS_2', 'FUS_3', 'FUS_4', 'FUS_5']
conditions = [f'KLM3114_{i}' for i in [1,2,3,4,5]]

for ci,cond in enumerate(conditions):
    rcond = results[results.img.str.startswith(cond)]
    vals = rcond.colocalization_index.values
    vals = vals[~np.isnan(vals)]
    frac = np.mean(vals>.8)
    pl.plot(ci, frac, marker='o')
    #pl.boxplot(vals, positions=[ci])

pl.xlim([-.5,5])

##
