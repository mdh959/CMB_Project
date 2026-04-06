import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from flipper import *
from lensingTools import *
import astLib.astWCS
import astLib.astCoords
import astLib.astCalc

phi_new1 = liteMap.liteMapFromFits('sims/kappa_est_S4_0.fits')
phi_new2 = liteMap.liteMapFromFits('sims/kappa_est_S4_1.fits')
phi_new3 = liteMap.liteMapFromFits('sims/kappa_S4_0.fits')
phi_new4 = liteMap.liteMapFromFits('sims/kappa_S4_1.fits')
#phi_new5 = liteMap.liteMapFromFits('sims/Tlens_S4_0.fits')
#phi_new6 = liteMap.liteMapFromFits('sims/Tlens_S4_1.fits')
phi_new5 = liteMap.liteMapFromFits('sims/T_map_0.fits')
phi_new6 = liteMap.liteMapFromFits('sims/T_map_1.fits')

phi_new1 = kappaToPhi(phi_new1)
phi_new2 = kappaToPhi(phi_new2)
phi_new3 = kappaToPhi(phi_new3)
phi_new4 = kappaToPhi(phi_new4)

Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr<5000.] = 0.
filt[Larr<5000.] = 0.
#filt[0] = 0.

phi_new1 = phi_new1.filterFromList([Larr,filt])
phi_new2 = phi_new2.filterFromList([Larr,filt])
phi_new3 = phi_new3.filterFromList([Larr,filt])
phi_new4 = phi_new4.filterFromList([Larr,filt])

phi_new1.plot(show=False,saveFig = 'est_S4_0.png',valueRange=[-1.e-8,1.e-8])
plt.close()
phi_new2.plot(show=False,saveFig = 'est_S4_1.png',valueRange=[-1.e-8,1.e-8])
plt.close()
phi_new3.plot(show=False,saveFig = 'real_S4_0.png',valueRange=[-1.e-8,1.e-8])
plt.close()
phi_new4.plot(show=False,saveFig = 'real_S4_1.png',valueRange=[-1.e-8,1.e-8])
plt.close()
phi_new5.plot(show=False,saveFig = 'Tl_S4_0.png')#,valueRange=[-1.e-4,1.e-4])
plt.close()
phi_new6.plot(show=False,saveFig = 'Tl_S4_1.png')#,valueRange=[-10,10])
plt.close()

