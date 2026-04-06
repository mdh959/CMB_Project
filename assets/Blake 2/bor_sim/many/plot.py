import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flipper import *
import astLib.astWCS
import astLib.astCoords
import astLib.astCalc

T_map1 = liteMap.liteMapFromFits('sims/T_map_12.fits')
T_map2 = liteMap.liteMapFromFits('sims/T_map_25.fits')
T_map3 = liteMap.liteMapFromFits('sims/T_map_20.fits')
T_map4 = liteMap.liteMapFromFits('sims/T_map_23.fits')
T_map5 = liteMap.liteMapFromFits('sims/T_map_26.fits')
T_map6 = liteMap.liteMapFromFits('sims/T_map_29.fits')

T_map1.plot(show=False,saveFig = '../lala0.png')
plt.close()
T_map2.plot(show=False,saveFig = '../lala1.png')
plt.close()
T_map3.plot(show=False,saveFig = '../lala2.png')
plt.close()
T_map4.plot(show=False,saveFig = '../lala3.png')
plt.close()
T_map5.plot(show=False,saveFig = '../lala4.png')
plt.close()
T_map6.plot(show=False,saveFig = '../lala9.png')
plt.close()

