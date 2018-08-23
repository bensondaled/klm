import pandas as pd, numpy as np
import os
from Tkinter import Tk
from tkFileDialog import askopenfilename

############# STUFF TO EDIT ##############

rows_to_skip = 1
rename_columns = True


#######################################


Tk().withdraw()
filepath = askopenfilename()

path,filename = os.path.split(filepath)
basename = os.path.splitext(filename)[0]

data = pd.read_excel(filename, skiprows=rows_to_skip, header=None)
empty_cols = np.squeeze(np.where(data.isnull().mean()))
idxs = np.concatenate([[-1],empty_cols,[None]])
data = [data.iloc[:,i0+1:i1] for idx,i0,i1 in zip(range(len(idxs)-1),idxs[:-1],idxs[1:])]
if rename_columns:
    for d in data:
        d.columns = np.arange(d.shape[1])
data = pd.concat(data)
means = data.groupby(data.index).mean()
means -= means.iloc[0]
means.plot()
means.to_csv(os.path.join(path,'{}_result.csv'.format(basename)))
