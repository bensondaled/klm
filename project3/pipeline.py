##
import sys
import warnings
from soup.classic import *
from skimage.exposure import equalize_adapthist as clahe
from skimage.feature import blob_dog
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.filters import gaussian
from sklearn.cluster import AffinityPropagation, MeanShift
from matplotlib.widgets import Slider
    
def yxr2mask(y,x,r,shape):
    """Given y,x coords and radius of a blob, generate a binary mask for it

    Parameters
    ----------
    y,x : int
        center coordinates
    r : int
        radius
    shape : tuple
        (y,x) shape of mask array

    Returns
    -------
    mask : np.ndarray, mask array
    """
    h,w = shape

    iy,ix = np.ogrid[-y:h-y, -x:w-x]
    return ix*ix + iy*iy <= r*r

def compute_clusterness(img_path, visualize=False, interactive=False):
    
    # read image and basic preprocessing
    im = imread(img_path).astype(np.uint16)
    im = np.squeeze(im).sum(axis=-1)
    im = clahe(im)
    im_disp = im.copy()
    im_prefilt = im.copy()
    im = gaussian(im, 3) 

    return np.array([np.std(im[im>0.05])])

    if visualize:
        fig,axs = pl.subplots(2,2,sharex=True,sharey=True)
        axs = axs.ravel()
        [ax.axis('off') for ax in axs]
        axs[0].imshow(im_disp)
   
    def compute_pts(upper=0.007):
        # hough transform to find circles
        lower = 1e-5
        sigma_canny = 3.
        edges = canny(im, sigma=sigma_canny, low_threshold=lower, high_threshold=upper)

        hough_radii = np.arange(15, 40, 2) # strictly for size constraints on cells
        hough_res = hough_circle(edges, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, threshold=.4*np.max(hough_res), min_xdistance=25, min_ydistance=25)

        pts = np.array([cy,cx]).T
        return pts,radii

    if interactive:
        def end_pause(*args):
            global paused
            paused = False

        def update_pts(*args):
            upper = slider.val
            pts,radii = compute_pts(upper)
            global _pts
            if _pts is not None:
                _pts.remove()
            _pts = axi.scatter(*pts.T[::-1], color='red')

        fig,(axi,ax_sl) = pl.subplots(2,1,num='Interactive window',gridspec_kw=dict(height_ratios=[10,1]))
        global _pts, paused
        _pts = None
        update_pts()
        slider = Slider(ax_sl, '', 0.0001, 0.1, valinit=0.005, valfmt='%0.5f')
        slider.on_changed(update_pts)
        fig.canvas.mpl_connect('close_event', end_pause)
        #fig.canvas.mpl_connect('button_press_event', self.on_press)
        axi.imshow(im)
        axi.axis('off')
        paused = True

    else:
        pts,radii = compute_pts()

    # merge bundles of hough results to converge on single cells
    ms = MeanShift(bandwidth=35)
    ms.fit(pts)
    c = ms.cluster_centers_
    labs = ms.labels_
    r = np.array([np.mean(radii[labs==ci]) for ci in np.arange(len(c))])
    circs = np.concatenate([c,r[:,None]], axis=1)

    if visualize:
        axs[1].imshow(im_disp)
        for cy,cx,ri in circs:
            circ = pl.Circle((cx,cy), ri, facecolor='none', edgecolor='k', lw=.5)
            axs[1].add_patch(circ)

    # characterize the contents of each cell
    masks = np.array([yxr2mask(*c,shape=im.shape) for c in circs])
    clusteriness = np.array([np.std(im_prefilt[m]) for m in masks])

    if visualize:
        disp = np.zeros_like(im)
        for v,m in zip(clusteriness,masks):
            disp[m] = v
        axs[2].imshow(disp)

    ret = clusteriness
    if visualize:
        ret = (ret, fig)
    return ret

##
if __name__ == '__main__':
    # how to use:
    # give the data path to a folder that contains one folder per condition
    # then just run this main, and it will save the images and values out to outputs

    #data_path = '111214_asyn/'
    data_path = sys.argv[1]

    out_path = os.path.join('outputs',data_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    conditions = [i for i in os.listdir(data_path) if i[0]!='.']

    results = []

    for cond in conditions:
        print(cond)
        cpath = os.path.join(data_path, cond)
        tfiles = [t for t in os.listdir(cpath) if t.endswith('.tif')]
        tpaths = [os.path.join(cpath,t) for t in tfiles]

        for tpath in tpaths:
            print('\t',tpath)
            t_name = os.path.splitext(os.path.split(tpath)[-1])[0]
            result = compute_clusterness(tpath, visualize=False)
            #result,fig = compute_clusterness(tpath, visualize=True)
            #figpath = os.path.join(out_path, '{}-{}'.format(cond,t_name))
            #fig.savefig(figpath)
            #pl.close(fig)
            
            for r in result:
                results.append(dict(    cond=cond,
                                        path=tpath,
                                        filename=t_name,
                                        value=r,
                    ))
    results = pd.DataFrame(results)
    out = os.path.join(out_path, 'all_results.csv')
    results.to_csv(out)
##
