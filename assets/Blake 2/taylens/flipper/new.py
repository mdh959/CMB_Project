#import numpy as np
#import pyfits
#import astLib
#import matplotlib
#import healpy
from flipper import *
import flipper.fftTools as fft

map1 = "../out/ucmb000_3.fits"
ft1 = fft.fftFromLiteMap(map1)
