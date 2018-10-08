'''
Manually label individual cells, displayed randomly one by one
'''

##

# Parameters
data_path = '/Users/ben/Desktop/images/'
allowed_answers = ['y','n','i']
multi_channel = True  # must be False or True
ignore_channels = [1] # a list of channels to eliminate/ignore, ex. [1,2]

##

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as pl
import os
import numpy as np
from skimage.io import imread
from skimage.exposure import equalize_adapthist as clahe
from collections import OrderedDict
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

out_filename = 'label_log.csv'
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
fig,axs = pl.subplots(2,2); axs=axs.ravel()
global chan_idx
chan_idx = 0
imgs,imgs_clahe,clims_clahe = [],[],[]
_im0=_im1=_im2=_im3=None
def press(evt):
    global pressed
    global chan_idx
    pressed = evt.key
    
    if (multi_channel is True) and (pressed == 't'):
        chan_idx += 1
        if chan_idx >= len(imgs):
            chan_idx = 0
        _im0.set_data(imgs[chan_idx])
        _im0.set_clim(vmin=imgs[chan_idx].min(), vmax=imgs[chan_idx].max())
        _im1.set_data(imgs_clahe[chan_idx])
        _im1.set_clim(vmin=clims_clahe[chan_idx][0], vmax=clims_clahe[chan_idx][1])
        _im2.set_data(imgs[chan_idx])
        _im2.set_clim(vmin=imgs[chan_idx].min(), vmax=imgs[chan_idx].max())
        _im3.set_data(imgs_clahe[chan_idx])
        _im3.set_clim(vmin=clims_clahe[chan_idx][0], vmax=clims_clahe[chan_idx][1])
        fig.canvas.draw()
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

    if (multi_channel is False) and (img.ndim > 2):
        chandim = np.argmin(img.shape)
        img = img.sum(axis=chandim)
        img = (img-img.min())/(img.max()-img.min())
    elif (multi_channel is True) and img.ndim==2:
        multi_channel = False
    
    cell_params = selection_data[cellid]
    x,y,r = cell_params
    
    if multi_channel is False:
        img_clahe = clahe(img)
        cvmin,cvmax = img_clahe.min(), img_clahe.max()
        y0 = int(max(y-r-pad, 0))
        x0 = int(max(x-r-pad, 0))
        y1 = int(min(y+r+pad, img.shape[0]))
        x1 = int(min(x+r+pad, img.shape[1]))
        cell_img = img[y0:y1, x0:x1]
        cell_img_clahe = img_clahe[y0:y1, x0:x1]
    else:
        imgs,imgs_clahe = [],[]
        chandim = np.argmin(img.shape)
        for i in range(img.shape[chandim]):
            if i in ignore_channels:
                continue
            sls = [slice(None,None) for i in range(img.ndim)]
            sls[chandim] = i
            img_ch = img[sls]
            img_ch_norm = (img_ch-img_ch.min())/(img_ch.max()-img_ch.min())
            img_clahe_ch = clahe(img_ch_norm)
            cvmin,cvmax = img_clahe_ch.min(), img_clahe_ch.max()
            y0 = int(max(y-r-pad, 0))
            x0 = int(max(x-r-pad, 0))
            y1 = int(min(y+r+pad, img_ch.shape[0]))
            x1 = int(min(x+r+pad, img_ch.shape[1]))
            cell_img = img_ch[y0:y1, x0:x1]
            cell_img_clahe = img_clahe_ch[y0:y1, x0:x1]

            imgs.append(cell_img)
            imgs_clahe.append(cell_img_clahe)
            clims_clahe.append((cvmin,cvmax))

        cell_img = imgs[chan_idx]
        cell_img_clahe = imgs_clahe[chan_idx]
        cvmin,cvmax = clims_clahe[chan_idx]

    for ax in axs:
        ax.clear()

    axs[0].set_title(f'{len(cells)} cells remaining')
    axs[1].set_title('Allowed responses:\n'+str(allowed_answers))

    _im0 = axs[0].imshow(cell_img, cmap=pl.cm.Greys_r)
    _im1 = axs[1].imshow(cell_img_clahe, cmap=pl.cm.Greys_r, vmin=cvmin, vmax=cvmax)
    _im2 = axs[2].imshow(cell_img, cmap=pl.cm.viridis)
    _im3 = axs[3].imshow(cell_img_clahe, cmap=pl.cm.viridis, vmin=cvmin, vmax=cvmax)
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
