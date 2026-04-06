import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flipper import *
import astLib.astWCS
import astLib.astCoords
import astLib.astCalc

print "AZI BUKI VEDI"

kappa = liteMap.liteMapFromFits('kappa.fits')
kappa_est = liteMap.liteMapFromFits('kappa_est.fits')

kappa.plot(show=False,saveFig = '../lala7.png')
plt.close()
kappa_est.plot(show=False,saveFig = '../lala8.png')
plt.close()

