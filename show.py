'''
Display selections performed in selections.py for visual inspection
'''

# Parameters
data_path = '/Users/kmack/Desktop/images'
###

import matplotlib.pyplot as pl
import os
import numpy as np
from skimage.io import imread
from skimage.exposure import equalize_adapthist as clahe
pl.ion()

fs = os.listdir(data_path)
fs = sorted([os.path.join(data_path,f) for f in fs if f.endswith('.tif')])
fnames = [os.path.splitext(f)[0] for f in fs]
outs = [f+'.npy' for f in fnames]

for f,o in zip(fs,outs):
    if os.path.exists(o):
        fig = pl.figure(num=f)
        ax = fig.add_axes([0,0,1,1])
        ax.axis('off')
        im = imread(f).squeeze()
        if im.ndim>2:
            im = im.sum(axis=-1)
        im = clahe(im)
        roi = np.load(o)
        ax.imshow(im, cmap=pl.cm.Greys_r)
        for r in roi:
            circ = pl.Circle(xy=(r[0],r[1]), radius=r[2], facecolor='orange', alpha=.5)
            ax.add_patch(circ)
