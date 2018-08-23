"""
v3:
    sliding max
    threshs: 2., 2.5

v4: 
    alignment window = range(100,401,100)
    alignment mode: max
    threshs: 2., 2.5
"""

##
from soup import *
import sys
import warnings, pickle
import ipdb
from skimage.filters import median
from skimage.exposure import equalize_adapthist as clahe
from skimage.morphology import disk
from scipy.stats import chi2_contingency as chi
from scipy.signal import correlate2d as c2d
from matplotlib.patches import Circle

##

def align(im1, im2=None, vals=None, max_shift=0.25, return_vals=False, verbose=False):
    """Compute template-matching-based shift to align channels

    Aligns im2 to im1
    return_vals : give the vals to use to correct another image in the same way
    if im2 is none and vals is there, skip to correction
    """
    max_shift = int(np.min(im1.shape)*max_shift)
    
    if im2 is not None:
        ms = max_shift
        im1,im2 = im1.astype(np.float32),im2.astype(np.float32)
        template_matching_method = cv2.TM_CCORR_NORMED
        res = cv2.matchTemplate(im2, im1[ms:-ms,ms:-ms], template_matching_method)
        met = np.max(res)

        top_left = cv2.minMaxLoc(res)[3]
        sh_x,sh_y = top_left
        sh_x = -(sh_x - ms)
        sh_y = -(sh_y - ms)
        inp = im2
    elif vals is not None:
        sh_x,sh_y,met = vals
        inp = im1

    dtype = inp.dtype

    M = np.float32([[1,0,sh_x],[0,1,sh_y]])             
    result = cv2.warpAffine(inp,M,(inp.shape[1],inp.shape[0]),flags=cv2.INTER_LINEAR)
    if verbose:
        print('\t\tAlignment shift: {},\tmetric: {:0.4f}'.format([sh_x,sh_y], met))

    if not return_vals:
        return result.astype(dtype)
    elif return_vals:
        return result.astype(dtype), (sh_x,sh_y,met)

def manders(a,b):
    """Manders overlap coefficient between a and b
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074624/

    Parameters
    ----------
    a : np.ndarray
        array 1
    b : np.ndarray
        array 2

    Returns
    -------
    manders overlap coefficient
    """
    m = np.dot(a,b)/np.sqrt(np.dot(a,a) * np.dot(b,b))
    if m<0:
        print(m<0)
        print(np.any(a<0))
        print(np.any(b<0))
        print()
    return m

def corr(a,b):
    return np.corrcoef(a,b)[0,1]
    #return manders(a,b)

def preprocess(im):
    """Preprocess an image by applying median filter, histogram equalization, and normalization

    Parameters
    ----------
    im : np.ndarray
        image to process

    Returns
    -------
    im : processed image
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # median filter
        im = median(im, disk(5))

        # histogram equalization
        im = clahe(im, clip_limit=0.01, kernel_size=200)

        # normalization
        im = (im-im.min())/(im.max()-im.min())
        im = (255*im).astype(np.uint8)
        im = 255-im

    return im

def find_blobs(im, **kwargs):
    """Find circular blobs using OpenCV's blob detector

    Parameters
    ----------
    im : np.ndarray
        image

    Returns:
    blobs: pd.DataFrame with (y,x) coords of blob centers, and radii
    """
    default_params = dict(
                minThreshold = 3,
                maxThreshold = 5000,
                filterByArea = True,
                minArea = 180,
                maxArea = 4500,
                filterByColor=False,
                filterByConvexity=True,
                minConvexity = 0.6,
                maxConvexity = 1e10,
                filterByInertia=True,
                filterByCircularity = True,
                minCircularity = 0.0001,
                minDistBetweenBlobs = 3,
            )

    default_params.update(kwargs)

    o = im.copy()
    params = cv2.SimpleBlobDetector_Params()
    for k,v in default_params.items():
        setattr(params, k, v)
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(o)
    blobs = pd.DataFrame([(kp.pt[1], kp.pt[0], kp.size/2.) for kp in keypoints], columns=['y','x','r'])

    return blobs

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

def filter_blobs(blobs, shape, pad=7):
    """Given some blobs, remove those that are too close to the border
    """
    ok = []
    y,x = shape
    bad = np.array([b.x-b.r<pad or b.x+b.r>x-pad or b.y-b.r<pad or b.y+b.r>y-pad for idx,(_,b) in enumerate(blobs.iterrows())])
    return blobs[~bad]

def process(filename, ch1_thresh=2., ch2_thresh=2.5, align_window=range(100,401,100), show=True, ex=None):
    """Process a single tif file

    (0) preprocess
    (1) find blobs
    (2) convert blobs to masks
    (3) correlate masks of two channels

    Parameters
    ----------
    filename : str
        path to tif file to process
    ch[1,2]_thresh : float
        number of stds above mean that at least one pixel must exhibit in channel X for a blob to be kept (blobs are found from channel 1)
    align_window : int
        pixels in each direction to pull out windows to locally align, because mismatch occurs in imaging. supply a list. each value will be tried and the one that yields the maximum template matching score will be used to proceed. this is implemented because sometimes small windows are better (non-uniform shifts as you increase window size, hence the motivation for this local analysis), but sometimes large is better (when there's not enough intensity content in the window to be conclusive). do not approach the size of cells because then it will match foci within the cell and void the whole point of the analysis
    ex : list
        used internally by batch process when examples are being saved

    Returns
    -------
    masks: (n,y,x) array, where n is the number of blobs, and values are the correlation metric

    Importantly, mask is only valid for inspecting ch1. ch2 is dynamically aligned to channel 1 in windows so the mask will not a priori line up with it upon inspection in all cases
    """
    if ex is not None:
        ex_every_n, ex_idx, ex_key, ex_dir, ex_pad = ex

    # read in image
    if not filename.endswith('.tif'):
        filename += '.tif'
    ch1_orig,ch2_orig = imread(filename)

    # preprocessing
    ch1 = preprocess(ch1_orig)
    ch2 = preprocess(ch2_orig)

    # experimental alignment:
    #X,vals = align(ch1,ch2,return_vals=True) # this or other one, not both
    """
    fig,axs = pl.subplots(1,2,num='New file: {}, val={:0.4f}'.format(filename, vals[-1]), sharex=True, sharey=True)
    axs[0].imshow(ch1)
    axs[1].imshow(ch2) # not the aligned one
    pl.title(str(vals))
    axs[0].set_title(np.corrcoef(ch1.ravel(),ch2.ravel())[0,1])
    pl.waitforbuttonpress()
    """

    blobs = find_blobs(ch1)
    blobs = filter_blobs(blobs, ch1.shape)
    
    masks_ = np.array([yxr2mask(*blob,shape=ch1.shape) for blob in blobs.values])
    print('\t{} cells found.'.format(len(masks_)))

    corrs,alignments,poss,blobskept = [],[],[],[]
    db1,db2 = [],[]
    if show:
        masks = []
    for mindex,(m,blob) in enumerate(zip(masks_,blobs.iterrows())):
        # build a window for local alignment
        center = np.mean(np.argwhere(m), axis=0)

        if align_window is not None:
            # ALIGNMENT:
            # find the best alignment window size
            print('\tAlignment for mask {}'.format(mindex))
            alw_vals = []
            for alw in align_window:
                (y0,x0),(y1,x1) = center-alw, center+alw
                y0,x0 = max(0,y0),max(0,x0)
                y1,x1 = min(ch1.shape[0],y1),min(ch1.shape[1],x1)
                y0,x0,y1,x1 = map(int,[y0,x0,y1,x1])
                ch1_win = ch1[y0:y1,x0:x1]
                ch2_win,vals = align(ch1_win, ch2[y0:y1,x0:x1], return_vals=True, verbose=False)
                alw_vals.append(vals)
            alw_vals = np.array(alw_vals)

            mode = 'best' # best / agg

            if mode == 'best':
                best = align_window[np.argmax(alw_vals[:,-1])]
                # apply that window (quite repetitive)
                alw = best
                (y0,x0),(y1,x1) = center-alw, center+alw
                y0,x0 = max(0,y0),max(0,x0)
                y1,x1 = min(ch1.shape[0],y1),min(ch1.shape[1],x1)
                y0,x0,y1,x1 = map(int,[y0,x0,y1,x1])
                ch1_win = ch1[y0:y1,x0:x1]
                ch1o_win = ch1_orig[y0:y1,x0:x1]
                ch2_win,vals = align(ch1_win, ch2[y0:y1,x0:x1], return_vals=True, verbose=True)
                ch2o_win = align(ch2_orig[y0:y1,x0:x1], vals=vals, verbose=False)
            elif mode == 'agg':
                y0,y1,x0,x1 = 0,None,0,None
                vals = np.median(alw_vals, axis=0)
                ch1_win = ch1
                ch1o_win = ch1_orig
                ch2_win = align(ch2, vals=vals, verbose=True)
                ch2o_win = align(ch2_orig, vals=vals, verbose=False)

        elif align_window is None:
            WIN = 2.
            WIN = int(np.round(WIN*blob[1].r))
            # NO ALIGNMENT:
            #y0,y1,x0,x1 = 0,None,0,None
            # more efficient: just make small bounding box
            y0,y1 = int(center[0]-WIN),int(center[0]+WIN)
            x0,x1 = int(center[1]-WIN),int(center[1]+WIN)
            y0,x0 = max(0,y0),max(0,x0)
            y1,x1 = min(ch1.shape[0],y1),min(ch1.shape[1],x1)

            ch2o_win = ch2_orig[y0:y1,x0:x1]
            ch1o_win = ch1_orig[y0:y1,x0:x1]
            ch1_win = ch1[y0:y1,x0:x1]
            ch2_win = ch2[y0:y1,x0:x1]
            vals = [None,None,None]

        m_win = m[y0:y1,x0:x1]

        # filter masks for brightness in ch2
        # pct thresh is hardcoded here for now***
        if not np.mean(ch2o_win[m_win] > ch2_orig.mean()+ch2_thresh*ch2_orig.std())>0.1:
            continue
        if not np.mean(ch1o_win[m_win] > ch1_orig.mean()+ch1_thresh*ch1_orig.std())>0.1:
            continue

        corr_mode = 'basic' # basic or slidingmax

        if corr_mode == 'basic':
            corrs.append(corr(ch1_win[m_win],ch2_win[m_win]))
        elif corr_mode == 'slidingmax':
            PAD = 5
            ms = int(np.round(blob[1].r)) + PAD
            template_matching_method = cv2.TM_CCOEFF_NORMED
            cc2 = ch2_win
            cc1 = ch1_win
            ycent = int(cc1.shape[0]/2)
            xcent = int(cc1.shape[1]/2)
            cc1cut = cc1[ycent-ms:ycent+ms,xcent-ms:xcent+ms]
            cc2cut = cc2[ycent-ms:ycent+ms,xcent-ms:xcent+ms]
            res = cv2.matchTemplate(cc2, cc1cut, template_matching_method)
            cc = np.max(res)
            #cc = np.mean(res)
            #cc = np.percentile(res.ravel(), 90)
            cc_other = np.corrcoef(cc1cut.ravel(), cc2cut.ravel())[0,1]

            """
            print(ycent)
            print(ms)
            print(cc1cut.shape)
            print(cc2.shape, cc1.shape)
            fig,axs = pl.subplots(2,2); axs=axs.ravel()
            axs[0].imshow(cc1cut)
            axs[1].imshow(cc2)
            axs[2].imshow(res)
            axs[2].set_title(cc_other)
            axs[3].hist(res.ravel(), bins=100)
            axs[3].set_title(cc)
            print(cc)
            input()
            pl.close()
            """
            
            corrs.append(cc)
            db1.append(cc)
            db2.append(cc_other)
        alignments.append(vals)
        poss.append(center)
        blobskept.append(blobs.iloc[mindex].values)
        if show:
            masks.append(m)
        
        if ex is not None: 
            ex_idx += 1
            if ex_idx % ex_every_n==0:
                print('--Saving example.')
                ex_key = ex_key.append(dict(file=filename, yx=center, r=corrs[-1], ex_idx=ex_idx), ignore_index=True)
                pl.ioff()
                fig,axs = pl.subplots(1,2)
                axs[0].imshow(ch1_win, cmap=pl.cm.Greens_r)
                axs[1].imshow(ch2_win, cmap=pl.cm.Blues_r)
                ecent = np.mean(np.argwhere(m_win), axis=0)# effective center of roi
                for ax in axs:
                    ax.set_xlim(ecent[1]-80, ecent[1]+80)
                    ax.set_ylim(ecent[0]-80, ecent[0]+80)
                    ax.axis('off')
                    circ = Circle(ecent[::-1], blobs.iloc[mindex].r+8, edgecolor='orange', facecolor='none',alpha=0.7)
                    ax.add_patch(circ)
                pl.savefig(os.path.join(ex_dir, '{}.pdf'.format(ex_idx)))
                pl.close(fig)
                pl.ion()
    corrs = np.array(corrs)
    alignments = np.array(alignments)
    poss = np.array(poss)
    blobskept = np.array(blobskept)

    #pl.scatter(db1,db2)
    #input()

    if show:
        masks = np.array(masks) * corrs[:,None,None]
    
    print('\t{}/{} kept after channel filtering.'.format(len(corrs), len(masks_)))
    
    # exclude inf and nans
    is_bad = (np.isnan(corrs)) | (np.isinf(corrs))
    print('\t{} nans/infs found and excluded.'.format(np.sum(is_bad)))
    corrs = corrs[~is_bad]
    alignments = alignments[~is_bad]
    poss = poss[~is_bad]
    blobskept = blobskept[~is_bad]
    if show:
        masks = masks[~is_bad]

    if show:
        fig,axs = pl.subplots(2,2, sharex=True, sharey=True, gridspec_kw=dict(left=0,right=1,bottom=0,top=1)); axs=axs.ravel()
        axs[0].imshow(ch1, cmap=pl.cm.Greens_r)
        axs[1].imshow(ch2, cmap=pl.cm.Blues_r)
        im_ = axs[2].imshow(np.nanmax(masks,axis=0), pl.cm.inferno, vmin=np.min(corrs), vmax=np.max(corrs))
        pl.colorbar(im_, ax=axs[3])
        [ax.axis('off') for ax in axs]

    if ex is not None:
        ex = ex_every_n, ex_idx, ex_key, ex_dir, ex_pad

    return corrs,alignments,poss,blobskept,ch1_orig,ch2_orig,ex

def process_batch(files, name='batch', verbose=True, save_examples=False, **kwargs):
    """Process batch of files using process()

    kwargs passed to process()
    save_examples : used to save out random examples for manual labelling
    """
    # save example params
    if save_examples:
        ex_every_n = 80
        ex_idx = -1
        ex_key = pd.DataFrame(columns=['file','blob_idx','r','ex_idx'])
        ex_dir = 'outputs/examples/' + name+'_examples'
        ex_pad = 100
        if not os.path.exists(ex_dir):
            os.mkdir(ex_dir)
        _ex = [ex_every_n, ex_idx, ex_key, ex_dir, ex_pad]
    else:
        _ex = None

    results = {}

    for fidx,f in enumerate(files):
        if not (f.endswith('.tif') or f.endswith('.tiff')):
            warnings.warn('Non-tif file skipped.')
            continue

        if verbose:
            print('\n{:0.2f}% \t\t File: {}'.format(100*fidx/len(files),f))

        cs,al,poss,blobs,ch1,ch2,_ex = process(f, ex=_ex, **kwargs)
        results[f] = dict(rs=cs,alignments=al,positions=poss,blobs=blobs)

    with open(name+'_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    if save_examples:
        _, _, ex_key, _, _ = _ex
        pd.to_pickle(ex_key, os.path.join(ex_dir,'key.pd'))

##
if __name__ == '__main__':

    src = sys.argv[1]
    name = os.path.split(src)[-1]
    files = [os.path.join(d,f) for d,_,fs in os.walk(src) for f in fs]

    process_batch(files, name=name, show=False, save_examples=True)#, align_window=None)

