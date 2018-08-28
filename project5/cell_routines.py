import numpy as np
import pandas as pd
import os
import h5py 
from skimage.io import imread
from scipy.ndimage import label

class Analysis():
    def __init__(self, folder, kind, **kwargs):
        
        self.kind = kind
        self.cells_iter = cells_generator(folder)
        self.kwargs = kwargs

        self.save_path = os.path.join(folder, f'results_{kind}.csv')
        self.result = None

    def run(self):

        if self.kind == 'foci':
            fxn = compute_foci_idx
            label = 'foci_index'
        elif self.kind == 'colocalization':
            fxn = compute_coloc
            label = 'colocalization_index'
        elif self.kind == 'membrane':
            fxn = compute_membrane_localization
            label = 'membrane_localization_index'

        else:
            print(f'Task type {self.kind} not recognized. Aborting.')
            fxn = None

        if fxn is not None:
            self.result = pd.DataFrame()
            for idx,cell in enumerate(self.cells_iter):
                value = fxn(cell, **self.kwargs)
                row_dict = {
                        'img':cell['img_name'],
                        'cell_idx':cell['cell_idx'],
                        label:value,
                        }
                self.result = self.result.append(row_dict, ignore_index=True)

    def save(self):
        if self.result is not None:
            self.result.to_csv(self.save_path)
            print(f'Result saved to {self.save_path}.')

def compute_foci_idx(cell, channel=0, method=np.std, normalize=True):
    '''
    method: probably should be np.var or np.std
    '''
    pixels = cell['pixels']
    if pixels.shape[-1]==1:
        channel=0
    pix = pixels[:,channel]
    if normalize:
        pix = (pix-pix.min())/(pix.max()-pix.min())
    return method(pix, ddof=1)

def compute_coloc(cell, channel_a, channel_b):
    '''Colocalization between two channels, defined simply as pearson's r
    '''
    pixels = cell['pixels']
    return np.corrcoef(pixels[:,channel_a], pixels[:,channel_b])[0,1]

def compute_membrane_localization(cell, channel=0):
    '''Membrane localization, computed as center of mass of pixel intensity divided by cell radius
    
    img: image containing cell
    mask: image with same size with 1's as "is cell" and 0's otherwise
    '''
    mask = cell['mask']
    img = cell['full_img']
    
    if img.shape[-1]==1:
        channel=0

    cell_coords = np.argwhere(mask)
    cell_coords_w = np.where(mask)
    cell_center = cell_coords.mean(axis=0)
    cell_area = np.sum(mask)
    cell_radius = np.sqrt(cell_area / np.pi)
    intensities = img[...,channel][cell_coords_w]
    dists = np.array([dist(cell_center, c) for c in cell_coords])
    com = np.mean(intensities * dists[:,None,None])
    ml = com / cell_radius
    return ml

def dist(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sqrt((np.sum(a-b))**2)

def cells_generator(folder):
    ''' using the files inside folder, which should have pairs of original images and cellprofiler-derived mask images, make a generator to run through all cells in all images

    Parameters
    ----------
    folder : str
        path to input folder containing images and masks

    '''

    file_names = sorted([f for f in os.listdir(folder) if f.endswith('.tif') or f.endswith('.tiff')])
    img_names = [os.path.splitext(f)[0] for f in file_names if 'mask' not in f]

    for img_name in img_names:
        img_path = os.path.join(folder, img_name+'.tif')
        mask_path = os.path.join(folder, img_name+'_mask.tiff')

        if not os.path.exists(mask_path):
            print(f'Skipping {img_name} because mask not found.')
            continue
        
        print(f'Image {img_name}')

        # read in original image and masks from cellprofiler
        img = imread(img_path).squeeze()
        mask = imread(mask_path).squeeze()
        if img.ndim>2:
            channel_dim = np.argmin(img.shape)
            if channel_dim != img.ndim-1:
                if channel_dim != 0:
                    raise Exception('Cannot determine channel dimension.')
                img = np.rollaxis(img, 0, img.ndim)
                mask = np.rollaxis(mask, 0, mask.ndim)
        elif img.ndim==2:
            img = img[...,None]
        labels,nlab = label(mask)
            
        # iterate through cells
        for cell_idx in range(1, nlab+1):
            #print(f'\tcell #{cell_idx}')
            cell = img[labels==cell_idx]
            mask = labels==cell_idx
            yield dict(img_name=img_name, img_path=img_path, cell_idx=cell_idx, pixels=cell, full_img=img, mask=mask)
