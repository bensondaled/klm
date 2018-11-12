'''
Compute colocalization scores based on manual selections
'''

##

# Parameters
data_path = '/Users/kmack/Desktop/images/'
channels_to_compare = [0,2] # list of two channel indices to compute colocalization across
save_images = True # save out individual images and their scores for inspection

# include_list : optional, path to label file containing only cells to use; use value None if want to include all cells
include_list = '/Users/kmack/Desktop/images/label_log.csv'

##

import os
import numpy as np
from skimage.io import imread
from collections import OrderedDict
import matplotlib.pyplot as pl

tif_filenames = sorted([t for t in os.listdir(data_path) if t.endswith('.tif')])
tif_paths = [os.path.join(data_path, tf) for tf in tif_filenames]
npy_paths = [os.path.splitext(t)[0]+'.npy' for t in tif_paths]

tif_paths = [tp for tp,nyp in zip(tif_paths,npy_paths) if os.path.exists(nyp)]
npy_paths = [nyp for nyp in npy_paths if os.path.exists(nyp)]
names = [os.path.splitext(os.path.split(f)[-1])[0] for f in tif_paths]

if save_images:
    ind_imgs_path = os.path.join(data_path, 'inspection_images')
    os.makedirs(ind_imgs_path, exist_ok=True)

# list all possible cells
cells = OrderedDict()
for nyp,tpath,nm in zip(npy_paths,tif_paths,names):
    for l in range(len(np.load(nyp))):
        ustr = f'{nm}_cell{l}'
        cells[ustr] = dict(npy_path=nyp, tif_path=tpath)

# include only the requested ones if given a list
if include_list is not None:
    with open(include_list, 'r') as f:
        data = f.readlines()
    data = [d.strip().split(',') for d in data]
    data = np.array(data)
    include = data[:,0]

    for key in list(cells.keys()):
        if key not in include:
            del cells[key]
##

# compute colocalization for each cell
coloc_path = os.path.join(data_path, 'coloc.csv')

def get_bbox(cell, shape):
    x,y,r = cell
    r = int(round(r))
    x0 = max(x-r,0)
    y0 = max(y-r,0)
    x1 = min(x+r,shape[1])
    y1 = min(y+r,shape[0])
    return [int(int(i)) for i in [y0,y1,x0,x1]]

for cellid in cells.keys():
    filename,celli = cellid.split('_cell')
    celli = int(celli)

    tif_path = os.path.join(data_path, filename+'.tif')
    npy_path = os.path.join(data_path, filename+'.npy')
    assert tif_path in tif_paths and npy_path in npy_paths

    im = imread(tif_path)
    assert im.ndim==3, 'Channels not found; greyscale image.'
    chandim = np.argmin(im.shape)
    im = np.rollaxis(im, chandim)
    #assert im.shape[0]==2, '2 channels not found.'
    sel = np.load(npy_path)
    
    cell = sel[celli]
    y0,y1,x0,x1 = get_bbox(cell, im.shape[1:])
    box = im[:,y0:y1,x0:x1]

    chani0,chani1 = channels_to_compare
    
    r = np.corrcoef(box[chani0].ravel(), box[chani1].ravel())[0,1]
    
    with open(coloc_path, 'a') as f:
        f.write(f'{cellid},{r}\n')
        
    if save_images:
        sim_path = os.path.join(ind_imgs_path, f'{cellid}.png')
        fig,axs = pl.subplots(1,2)
        axs[0].imshow(box[chani0])
        axs[1].imshow(box[chani1])
        axs[1].set_title(f'{r:0.3f}')
        for ax in axs:
            ax.axis('off')
        fig.savefig(sim_path)
        pl.close(fig)
    
##
