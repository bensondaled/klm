##

# Parameters
data_path = '/Users/ben/Desktop/km_data'
allowed_answers = ['y','n']

##

import os
import numpy as np
from skimage.io import imread
from skimage.exposure import equalize_adapthist as clahe
from collections import OrderedDict
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as pl
pl.ion()

tif_filenames = sorted([t for t in os.listdir(data_path) if t.endswith('.tif')])
tif_paths = [os.path.join(data_path, tf) for tf in tif_filenames]
npy_paths = [os.path.splitext(t)[0]+'.npy' for t in tif_paths]

tif_paths = [tp for tp,nyp in zip(tif_paths,npy_paths) if os.path.exists(nyp)]
npy_paths = [nyp for nyp in npy_paths if os.path.exists(nyp)]
names = [os.path.splitext(os.path.split(f)[-1])[0] for f in tif_paths]

# list all possible cells
cells = OrderedDict()
for nyp,tpath,nm in zip(npy_paths,tif_paths,names):
    for l in range(len(np.load(nyp))):
        ustr = f'{nm}_cell{l}'
        cells[ustr] = dict(npy_path=nyp, tif_path=tpath)

out_filename = 'label_log.txt'
out_path = os.path.join(data_path, out_filename)

if not os.path.exists(out_path):
    with open(out_path,'a') as f:
        pass

# remove all already-labelled cells from selection pool
with open(out_path, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        iid,val = line.split(',')
        if iid in cells:
            del cells[iid]

# main loop: randomly select a cell, display it, and store result
pad = 5
def press(evt):
    global pressed
    pressed = evt.key
fig,axs = pl.subplots(2,2); axs=axs.ravel()
fig.canvas.mpl_connect('key_press_event', press)
while True:
    if len(cells) == 0:
        break

    k = np.random.choice(list(cells.keys()))
    info = cells[k]

    last_us_idx = k.rfind('_')
    cellstr = k[last_us_idx+1:]
    assert cellstr[0:4]=='cell', 'Filename string not parsable'
    cellid = int(cellstr[4:])
    
    selection_data = np.load(info['npy_path'])
    img = imread(info['tif_path'])
    img_clahe = clahe(img)
    cell_params = selection_data[cellid]
    x,y,r = cell_params
    y0 = int(max(y-r-pad, 0))
    x0 = int(max(x-r-pad, 0))
    y1 = int(min(y+r+pad, img.shape[0]))
    x1 = int(min(x+r+pad, img.shape[1]))
    cell_img = img[y0:y1, x0:x1]
    cell_img_clahe = img_clahe[y0:y1, x0:x1]

    for ax in axs:
        ax.clear()

    axs[0].set_title(f'{len(cells)} cells remaining')
    axs[1].set_title('Allowed responses:\n'+str(allowed_answers))

    axs[0].imshow(cell_img, cmap=pl.cm.Greys_r)
    axs[1].imshow(cell_img_clahe, cmap=pl.cm.Greys_r, vmin=img_clahe.min(), vmax=img_clahe.max())
    axs[2].imshow(cell_img, cmap=pl.cm.viridis)
    axs[3].imshow(cell_img_clahe, cmap=pl.cm.viridis, vmin=img_clahe.min(), vmax=img_clahe.max())
    for ax in axs:
        ax.axis('off')
    
    print(k)
    global pressed
    pressed = ''
    while pressed not in allowed_answers:
        pl.waitforbuttonpress()

    print(pressed)
    print()

    with open(out_path, 'a') as f:
        f.write(f'{k},{pressed}\n')

    del cells[k]
    pressed = ''

pl.close()
print('Done, no more cells.')
##
