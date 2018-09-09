'''
Copy labels into a spreadsheet for further analysis.
'''

# Parameters
label_file = '/Users/kmack/Desktop/completed images/asyn/11.14.14/tifs and npys/label_log.txt'

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
