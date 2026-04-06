import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flipper import *
import astLib.astWCS
import astLib.astCoords
import astLib.astCalc

T_map = liteMap.liteMapFromFits('T_map.fits')
#T_map = liteMap.liteMapFromFits('Tlens_noise_1_0.fits')
phi_new = liteMap.liteMapFromFits('phi_new.fits')
Tlens = liteMap.liteMapFromFits('Tlens.fits')

T_map.plot(show=False,saveFig = 'lala1.png')
plt.close()
Tlens.plot(show=False,saveFig = 'lala2.png')
plt.close()
phi_new.plot(show=False,saveFig = 'lala3.png')
plt.close()
