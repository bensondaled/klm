##

import os, sys, shutil, random

path = '/Users/ben/cloud/for_others/korrie/datasets/20180405/'
tifs = sorted([os.path.join(path,f) for f in os.listdir(path) if f.endswith('.tif')])

##

random.shuffle(tifs)

key = []

for i,t in enumerate(tifs):
    root,name = os.path.split(t)
    name,ext = os.path.splitext(name)
    newname = '{}'.format(i)
    newpath = os.path.join(root, newname+'.tif')
    key.append((t, newpath, name, newname))
    print('Moving {} to {}'.format(t, newpath))
    shutil.move(t, newpath)

k = pd.DataFrame(key, columns=['original_path','new_path','original_name','new_name'])
k.to_csv(os.path.join(root,'key.csv'))

##
