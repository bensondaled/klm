"""
1. download tools from this site: http://www.farsight-toolkit.org/wiki/FARSIGHT_Tutorials/Bio-Formats
2. run this command for each lif file:
    sh /path/to/bftools/bfconvert <filename.lif> <desired_prefix>_%s_C%c.tif
3. group into directories as desired
4. for every image with a C0 and C1, merge using this script
"""

##
import os
import numpy as np
from skimage.io import imread
##

dirname = '52616/t4' # a directory containing a bunch of C0 and C1 files
fs = [os.path.join(dirname,f) for f in os.listdir(dirname) if not f.startswith('.')]

##

for i in range(len(fs)//2):
    f0 = fs[i*2]
    f1 = fs[i*2+1]
    assert f0.endswith('_C0.tif')
    assert f1.endswith('_C1.tif')
    assert f0[:f0.index('C0')] == f1[:f1.index('C1')]
    t0 = imread(f0)[...,1]
    t1 = imread(f1)[...,2]
    assert t0.dtype==np.uint16
    t = np.array([t0,t1])

    prefix = f0[:f0.index('_C0')]
    imsave(prefix+'.tif', t)

##
