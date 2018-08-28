
# Parameters
label_file = '/Users/ben/Desktop/km_data/label_log.txt'

#####
import numpy as np
import pandas as pd
import os

out_dir = os.path.split(label_file)[0]
out_path = os.path.join(out_dir, 'summary.csv')

with open(label_file, 'r') as f:
    data = f.readlines()

data = [d.strip().split(',') for d in data]
data = np.array(data)

result = pd.DataFrame(columns=['name','label'])

result.loc[:,'name'] = data[:,0]
result.loc[:,'label'] = data[:,1]

result.sort_values('name', inplace=True)
result.reset_index(inplace=True, drop=True)
result.to_csv(out_path)
