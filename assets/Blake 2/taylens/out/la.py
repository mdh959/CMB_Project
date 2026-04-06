from flipper import *
#import liteMap
import healpy
import pyfits
import numpy as np

n = np.arange(100.0)

hdu = pyfits.PrimaryHDU([n,n])

hdulist = pyfits.HDUList([hdu])
hdulist.writeto('template.fits') 

#read in HEALPix map
hpm = healpy.read_map('ucmb000_3.fits')
#read in the liteMap
flatMap = liteMap.liteMapFromFits('template.fits')
#load the data from hpm into flatMap using bilinear interpolation
flatMap.loadDataFromHealpixMap(hpm, interpolate = True)
#write out the flatMap
