##
from skimage.filters import gaussian
from skimage.filters import threshold_otsu as otsu
from skimage.exposure import equalize_adapthist as clahe
from scipy import ndimage as ndi

img_path = '111214_asyn/asyn6/ASYN_6_8.tif'
    
im = imread(img_path).astype(np.uint16)
im_ = np.squeeze(im).sum(axis=-1)

im = clahe(im_)
im = gaussian(im, 5)

# v1
tval = otsu(im)
imt = im>tval
distance = ndi.distance_transform_edt(imt)
fig,axs = pl.subplots(2,2, sharex=True, sharey=True); axs=axs.ravel()
axs[0].imshow(im_)
axs[1].imshow(im)
axs[2].imshow(imt)
axs[3].imshow(distance)

# v2
lower = 1e-5
upper = 0.007
sigma_canny = 3.
edges = canny(im, sigma=sigma_canny, low_threshold=lower, high_threshold=upper)

hough_radii = np.arange(15, 40, 2) # strictly for size constraints on cells
hough_res = hough_circle(edges, hough_radii)
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, threshold=.4*np.max(hough_res), min_xdistance=25, min_ydistance=25)

pts = np.array([cy,cx]).T

##
