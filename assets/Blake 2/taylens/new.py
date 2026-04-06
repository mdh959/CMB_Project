#import numpy as np
import pyfits
#import astLib
#import matplotlib
import healpy
from flipper import *
#import flipper.fftTools as fft

file1 = "/users/boryanah/Blake/taylens/out/lcmb000_3.fits"
map1 = liteMap.liteMapFromFits(file1)
ft1 = fftTools.fftFromLiteMap(map1)
'''
hdulist = pyfits.open("/users/boryanah/Blake/taylens/out/lcmb000_3.fits")
header = hdulist[0].header
print header
 #   flTrace.issue('flipper.liteMap',3,"Map header \n %s"%header)

#map1 = healpy.read_map(file1)

'''
