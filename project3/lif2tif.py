##
import os, sys, subprocess as sp

##
dirs = [o for o in os.listdir() if os.path.isdir(o)]
fs = [(d,f) for d in dirs for f in os.listdir(d) if f.endswith('.lif')]

##

bfconvert_path = '/Users/ben/code/klm/project2/lif2tif/bftools/bfconvert'

for d,f in fs:
    rootname = os.path.splitext(f)[0]
    inpath = os.path.join(d,f)
    outname = '{}_%s.tif'.format(rootname)
    outpath = os.path.join(d, outname)
    print(inpath, outpath)
    sp.call(["sh", bfconvert_path, inpath, outpath])

##
