##

from statsmodels.stats.proportion import proportion_confint as conf
from scipy.stats import skew
import scipy.ndimage as ndi

path = '/Users/ben/cloud/for_others/korrie/datasets/'
dirs = [os.path.join(path,d) for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
tifs = sorted([os.path.join(dr,f) for dr in dirs for f in os.listdir(dr) if f.endswith('.tif')])

npys = [os.path.splitext(t)[0]+'.npy' for t in tifs]

keyfiles = {d:os.path.join(d,'key.csv') for d in dirs}
keys = {d:pd.read_csv(kf) if os.path.exists(kf) else None for d,kf in keyfiles.items()}
keys = {d:dict(k[['new_name','original_name']].astype(str).values) if k is not None else None for d,k in keys.items()}

##

def xyr2mask(x,y,r,shape):
    """Given y,x coords and radius of a blob, generate a binary mask for it

    Parameters
    ----------
    x,y : int
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

def zoom_on(x,y,r,img,pad=1,blank=False):
    x = int(round(x))
    y = int(round(y))
    r = int(round(r))

    if blank:
        msk = xyr2mask(x,y,r,shape=img.shape)
        img = img.copy().astype(float)
        img[~msk] = np.nan

    r += pad
    y0 = max(0, y-r)
    y1 = min(y+r, img.shape[0])
    x0 = max(0, x-r)
    x1 = min(x+r, img.shape[1])
    im = img[y0:y1, x0:x1]

    return im

def generate_img(row, expand_mask=0):
    x = row.x
    y = row.y
    r = row.r
    full = imread(row.path).squeeze().sum(axis=-1)
    img = zoom_on(x, y, r+expand_mask, img=full)
    img = (img-full.min())/(full.max()-full.min())
    return img

def metric(img, mask, zoom, n_std=1.5):

    #thresh = img.mean() + img.std()*n_std
    pix = img[mask]

    # normalize within a cell
    #pix = (pix-pix.min())/(pix.max()-pix.min())
    # normalize to the source image
    #pix = (pix-img.min())/(img.max()-img.min())
    zoom = (zoom-zoom.min())/(zoom.max()-zoom.min())

    #met = pix.max()-pix.min()
    #met = np.std(pix, ddof=1)
    #met = np.mean(pix>pix.mean()+2*pix.std(ddof=1))
    #met = skew(pix)
    #met = np.sum(pix[pix > thresh])
    #met = np.mean(pix)
    #met = np.average(np.abs(ndi.filters.laplace(zoom)))

    labs,nlab = ndi.label(zoom > zoom.mean()+zoom.std()*2.5)
    sizes = np.array([np.sum(labs==l) for l in range(0,nlab+1)])
    aggs = np.arange(0,nlab+1)[(sizes<zoom.size*.5) & (sizes>zoom.size*.01)]
    met = np.sum([np.sum(labs==a) for a in aggs]) / zoom.size
    #fig,axs = pl.subplots(1,2)
    #axs[0].imshow(zoom)
    #if len(aggs):
    #    axs[1].imshow(np.sum([labs==a for a in aggs], axis=0))
    #pl.waitforbuttonpress()
    #pl.close()

    return met

## compute and aggregate metrics

results = pd.DataFrame(columns=['path','idx','x','y','r','metric','img'])
expand_mask = 0

for idx,(tif,npy) in enumerate(zip(tifs, npys)):

    print('{}/{}'.format(idx,len(tifs)))

    # load images and selections
    img = imread(tif).squeeze().sum(axis=-1)
    roi = np.load(npy)

    # create masks and compute metrics
    mask = np.array([xyr2mask(*r[:2], r[2]+expand_mask, shape=img.shape) for r in roi])
    metrics = np.array([metric(img,m,zoom_on(*r,img,pad=0)) for r,m in zip(roi,mask)])
    cell_imgs = [None]*len(roi)

    # store in dataframe
    for idx,(ci,met,r) in enumerate(zip(cell_imgs, metrics, roi)):
        results = results.append(dict(path=tif, idx=idx, x=r[0], y=r[1], r=r[2], metric=met, img=ci), ignore_index=True)

## infer fields
results.loc[:,'name'] = [os.path.splitext(os.path.split(p)[-1])[0] for p in results.path.values]
results.loc[:,'dir'] = [os.path.split(p)[0] for p in results.path.values]
for rowi in range(len(results)):
    row = results.iloc[rowi]
    k = keys[row.dir]
    if k is not None:
        nn = k[row['name']]
        # fix inconsistent SLT name labelling
        if nn.startswith('SLT') and 'SLT_' in nn:
            nn = nn.replace('_','',1)
        results.loc[results.index[rowi],'name'] = nn

    # fix in unblinded ones
    if row['name'].startswith('KLM'):
        nn = '_'.join(row['name'].split('_')[1:])
        results.loc[results.index[rowi],'name'] = nn


results.loc[:,'condition'] = [n.split('_')[0] for n in results['name'].values]

## summary stats

# thresholds inspection:
minn,maxx = np.percentile(results.metric.values, [10,90])
threshs = np.linspace(minn, maxx, 6)
cols = pl.cm.copper(np.linspace(0,1,len(threshs)))
ucond = sorted(results.condition.unique())
for thresh,col in zip(threshs,cols):
    means = []
    for ci,cond in enumerate(ucond):
        ri = results[results.condition==cond]
        is_agg = ri.metric.values > thresh
        mean = is_agg.mean()
        means.append(mean)
    pl.plot(np.arange(len(means))+np.random.normal(0,.05), means, label='thresh={:0.2f}'.format(thresh), marker='o', color=col)
pl.legend(ncol=3, loc=(.5,1))
pl.xticks(np.arange(len(ucond)), ucond)
pl.ylabel('Fraction of cells with aggregates')

# continuous version:
pl.figure()
meds = []
for ci,cond in enumerate(ucond):
    med = results[results.condition==cond].metric.median()
    meds.append(med)
pl.plot(meds, color='k', marker='o')
pl.xticks(np.arange(len(ucond)), ucond)
pl.ylabel('Median aggregation score')

# specific threshold:
pl.figure()
ax = pl.gca()
thresh = .01#0.1#0.03#1.5
for ci,cond in enumerate(ucond):
    ri = results[results.condition==cond]
    is_agg = ri.metric.values > thresh
    ax.bar(ci, is_agg.mean(), width=.2, color='k')
ax.set_xticks(range(len(ucond)))
ax.set_xticklabels(ucond)
pl.ylabel('Fraction of cells with aggregates')
pl.title('Threshold = {:0.2f}'.format(thresh))

## image of examples by specific threshold
for ag in [False,True]:
    fig,axs = pl.subplots(4,4,gridspec_kw=dict(left=0,right=1,bottom=0,top=1,hspace=0,wspace=0)); axs=axs.ravel()
    if ag is True:
        samp = results[results.metric>thresh].sample(len(axs), replace=False)
    elif ag is False:
        samp = results[results.metric<=thresh].sample(len(axs), replace=False)
    imgs = [generate_img(row, expand_mask=expand_mask) for _,row in samp.iterrows()]
    for ax,s in zip(axs,imgs):
        ax.imshow(s, vmin=0, vmax=1, cmap=pl.cm.Greys_r)
        ax.axis('off')

## image of all aggregation scores
mv = results.metric.values
arg = np.argsort(mv)
fig,axs = pl.subplots(11,17,gridspec_kw=dict(left=0,right=1,bottom=0,top=1,hspace=0.01,wspace=0))
axs = axs.ravel()
marked = False
skip = len(results)//len(axs)
for ax,ar,m in zip(axs,arg[::skip],mv[arg][::skip]):
    ax.axis('off')
    img = generate_img(results.iloc[ar], expand_mask=expand_mask)
    if m>=thresh and not marked:
        marked=True
        continue
    #ax.imshow(img, vmin=0, vmax=1,)# cmap=pl.cm.Greys_r)
    ax.imshow(img,)# cmap=pl.cm.Greys_r)

## random inspections

cond = 'DMSO'
fig,axs = pl.subplots(4,5,gridspec_kw=dict(hspace=.2)); axs=axs.ravel()
r_ = results[results.condition==cond]
iis = np.random.randint(len(r_), size=len(axs))
for i,ax in zip(iis,axs):
    r = r_.iloc[i]
    im = generate_img(r)
    ax.imshow(im, vmin=0, vmax=1)#, cmap=pl.cm.Greys_r)
    #ax.imshow(im, cmap=pl.cm.Greys_r)
    ax.set_title('{:0.3f}'.format(r.metric))
    ax.axis('off')


##
