import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flipper import *
import astLib.astWCS
import astLib.astCoords
import astLib.astCalc

kappa = liteMap.liteMapFromFits('noise_0_01/kappa.fits')
kappa_est = liteMap.liteMapFromFits('noise_0_01/kappa_est.fits')

Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr<6000.] = 0.
filt[0] = 0.
kappa = kappa.filterFromList([Larr,filt])
kappa_est = kappa_est.filterFromList([Larr,filt])

kappa.plot(show=False,saveFig = 'lala7.png')
plt.close()
kappa_est.plot(show=False,saveFig = 'lala8.png')
plt.close()

