'''
Select cells from images using a simple click interface
'''

# Parameters
data_path = '/Users/ben/Desktop/images'
multi_channel = True  # must be False or True
###

import matplotlib
matplotlib.use('tkagg')
import os
import numpy as np
import matplotlib.pyplot as pl
pl.ion()
from scipy.spatial.distance import euclidean as dist
from skimage.exposure import equalize_adapthist as clahe
from skimage.io import imread
import warnings
import sys

warnings.filterwarnings("ignore")

class UI():
    def __init__(self, path, rois=None, multi_channel=False):

        self.path = path
        #self.name = os.path.split(self.path)[-1]
        self.name = self.path
        self.name = os.path.splitext(self.name)[0]
        self.multi_channel = multi_channel
        img = imread(self.path).squeeze()

        if (self.multi_channel is False) and (img.ndim > 2):
            chandim = np.argmin(img.shape)
            img = img.sum(axis=chandim)
            img = (img-img.min())/(img.max()-img.min())
        elif (self.multi_channel is True) and img.ndim==2:
            self.multi_channel = False
    
        self.fig = pl.figure()#num=self.name)
        self.ax = self.fig.add_axes([0.01,0.01,.99,.99])
        self.ax.axis('off')
        self.imshow_kw = dict(cmap=pl.cm.Greys_r)

        # dragging params
        self.r_init = 3
        self.r_per_pix = 1

        # runtime
        self.drag_patch = None
        self.drag_pos0 = (0,0)
        self._hiding = False
        if rois is None:
            self.rois = []
            self.patches = []
        else:
            self.rois = [tuple(r) for r in rois]
            self.patches = [pl.Circle((r[0],r[1]), radius=r[2], color='coral', alpha=.5, picker=5) for r in self.rois]
            for p in self.patches:
                self.ax.add_patch(p)
            self.update_patches()
        
        if self.multi_channel is False:
            img = clahe(img)
            img0 = img
        else:
            self.imgs = []
            chandim = np.argmin(img.shape)
            for i in range(img.shape[chandim]):
                sls = [slice(None,None) for i in range(img.ndim)]
                sls[chandim] = i
                img_s = img[sls]
                img_s = (img_s-img_s.min())/(img_s.max()-img_s.min())
                imgi = clahe(img_s)
                self.imgs.append(imgi)

            self.chan_idx = 0
            img0 = self.imgs[self.chan_idx]

        self._im = self.ax.imshow(img0, vmin=0, vmax=1, **self.imshow_kw)

        self._mode = 'select'

        self.fig.canvas.mpl_connect('button_press_event', self.evt_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.evt_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.evt_key)
        self.fig.canvas.mpl_connect('pick_event', self.evt_pick)
        self.fig.canvas.mpl_connect('close_event', self.evt_close)

    def reset(self):
        if self.drag_patch is not None:
            self.drag_patch.remove()
            self.drag_patch = None
        self._mode = 'select'
        self.update_patches()
        self.fig.canvas.draw()

    def update_patches(self):
        for p in self.patches:
            p.set_alpha(.5)
            p.set_color('coral')

    def evt_click(self, evt):
        if self._mode not in ['select','dragging']:
            return
        if evt.inaxes != self.ax:
            return
        
        if self._mode == 'select':
            pt = [round(evt.xdata), round(evt.ydata)]
            self._mode = 'dragging'

        elif self._mode == 'dragging':
            if self.drag_patch is None: # didn't drag at all
                self.reset()
                return
            cent = self.drag_patch.center
            self.rois.append((cent[0],cent[1],self.drag_patch.radius)) # x,y,r
            patch = pl.Circle(cent, radius=self.drag_patch.radius, color='coral', alpha=.5, picker=5)
            self.ax.add_patch(patch)
            self.patches.append(patch)
            self.reset()

        self.fig.canvas.draw()

    def evt_motion(self, evt):
        if self._mode == 'dragging':

            if evt.inaxes != self.ax:
                self.reset()
                return

            x,y = evt.xdata,evt.ydata

            if self.drag_patch is None:
                self.drag_patch = pl.Circle((x,y), radius=self.r_init, edgecolor=(1,.6,.2,.8), facecolor=(1,1,1,.2), lw=1.5)
                self.drag_pos0 = (x,y)
                self.ax.add_patch(self.drag_patch)

            else:
                dx = x-self.drag_pos0[0]
                dy = y-self.drag_pos0[1]
                dp = np.sqrt(dx**2 + dy**2)
                #dp = [dx,dy][np.argmax(np.abs([dx,dy]))]
                new_r = max(0, self.r_init+dp*self.r_per_pix)
                self.drag_patch.set_radius(new_r)

        elif self._mode == 'remove':

            if evt.inaxes != self.ax:
                return
            x,y = evt.xdata,evt.ydata

            if self.rois is None or len(self.rois) == 0:
                return
            centers = [p.center for p in self.patches]
            best = np.argmin([dist((x,y), c) for c in centers])

            self.update_patches()
            self.patches[best].set_color('red')
            self.patches[best].set_alpha(1.)
            self.fig.canvas.draw()

        else:
            return

        self.fig.canvas.draw()

    def evt_key(self, evt):
        if evt.key == 'z':
            self.remove_roi(-1)

        elif evt.key == 'x':
            if self._mode == 'remove':
                self.reset()
            else:
                self._mode = 'remove'
        elif evt.key == 'v':
            self.evt_hideshow()
        
        elif evt.key == '1':
            vmin,vmax = self._im.get_clim()
            vmin = vmin - .03
            vmin = max(vmin, 0)
            self._im.set_clim(vmin=vmin, vmax=vmax)
            
        elif evt.key == '2':
            vmin,vmax = self._im.get_clim()
            vmin = vmin + .03
            vmin = min(vmin, vmax-.001)
            self._im.set_clim(vmin=vmin, vmax=vmax)
            
        elif evt.key == '9':
            vmin,vmax = self._im.get_clim()
            vmax = vmax - .03
            vmax = max(vmax, vmin+.001)
            self._im.set_clim(vmin=vmin, vmax=vmax)
            
        elif evt.key == '0':
            vmin,vmax = self._im.get_clim()
            vmax = vmax + .03
            vmax = min(vmax, 1)
            self._im.set_clim(vmin=vmin, vmax=vmax)

        elif evt.key == 't':
            # toggle channel
            if self.multi_channel:
                self.chan_idx += 1
                if self.chan_idx >= len(self.imgs):
                    self.chan_idx = 0
                self._im.set_data(self.imgs[self.chan_idx])
            
        if evt.key in ['1','2','9','0','t']:
            self.fig.canvas.draw()

    def evt_pick(self, evt):
        if self._mode != 'remove':
            return
        obj = evt.artist
        idx = self.patches.index(obj)
        self.remove_roi(idx)
    
    def remove_roi(self, idx):
        if self.rois is None or len(self.rois)==0:
            return
        self.patches[idx].remove()
        del self.rois[idx]
        del self.patches[idx]

        self.update_patches()
        self.fig.canvas.draw()
    
    def evt_hideshow(self, *args):
        if self._hiding:
            self._hiding = False
            for p in self.patches:
                p.set_visible(True)
        elif self._hiding == False:
            self._hiding = True
            for p in self.patches:
                p.set_visible(False)
        self.fig.canvas.draw()

    def evt_close(self, evt):
        roi = np.array(self.rois)
        np.save(self.name, roi)
        pl.close(self.fig)

if __name__ == '__main__':

    print('\nKey controls:\n\n1: decrease minimum\n2: increase minimum\n9: decrease maximum\n0: increase maximum\nz: undo\nx: delete mode\nv: hide/show selections\nt: toggle channel\n\n')

    try:
        flag = sys.argv[1]
    except:
        flag = ''

    fs = os.listdir(data_path)
    fs = sorted([os.path.join(data_path,f) for f in fs if f.endswith('.tif')])
    fnames = [os.path.splitext(f)[0] for f in fs]
    outs = [f+'.npy' for f in fnames]

    if flag == 'list':
        for i,f in enumerate(fnames):
            print('{}\t{}'.format(i,f))

    else:
        for i,(f,o) in enumerate(zip(fs,outs)):
            if os.path.exists(o) and flag!=str(i):
                print('{} already complete.'.format(f))
            else:
                if os.path.exists(o) and flag==str(i):
                    roi = np.load(o)
                else:
                    roi = None
                print('Running {}.'.format(f))
                ui = UI(f, rois=roi, multi_channel=multi_channel)
                break
