import matplotlib
matplotlib.use('Agg')
import astLib.astWCS
import astLib.astCoords
import astLib.astCalc
import numpy as np
import matplotlib.pyplot as plt
from flipper import *
from lensingTools import *

# binning
bin = 'BIN_200_LOG'#'BIN_100_LOG''BIN_250_1000'

# load phi maps
dirT = '/users/boryanah/Blake/bor_sim/constGrad/'
nameT = 'lensedT_constGrad.fits'

# load maps                                          
dirphi = '/users/boryanah/Blake/bor_sim/'
namep = 'phi.fits'
phi = liteMap.liteMapFromFits(dirphi+namep)

# noise params
thetaFWHMarcmin = 0.3
noiseUkArcmin = 1.
thetaFWHM = thetaFWHMarcmin*numpy.pi/(180.*60.)
deltaT = noiseUkArcmin/thetaFWHMarcmin
noiseFactor = 1

scal = 'Aug6_highAcc_CDM_scalCls_new.dat'
theoryPower = np.loadtxt(scal)

tcmb = 2.7255e6
lphi = theoryPower[:,0]
clphi = theoryPower[:,4]/(lphi**4)/tcmb**2

# do we divide by anything tt ps and what is p
phi = phi.copy()

# OPTION 1:
#phi.fillWithGaussianRandomField(lphi,clphi)#,bufferFactor=1)  
#phi.writeFits(dirT+namep, overWrite=True)  

# OPTION 2:
phi = liteMap.liteMapFromFits(dirT+namep)
print phi.info()

# CONSTANT GRADIENT
# setting temperature map
Tulens = phi.copy()
Tulens.data[:,:] = np.array([np.linspace(-250,250,2048),]*2048)

# OPTION 1:
#Tlens, Qlens, Ulens  =  lensMaps(phi,Tulens, Tulens, Tulens)
#Tlens.writeFits(dirT+nameT, overWrite=True)  

# OPTION 2:
Tlens = liteMap.liteMapFromFits(dirT+nameT)

# add noise
# noise ps
NlT = (deltaT*thetaFWHM)**2*np.exp(lphi*(lphi+1.)*thetaFWHM**2/(8.*np.log(2.)))*noiseFactor**2.
noise = Tlens.copy()
noise.fillWithGaussianRandomField(lphi,NlT)#,bufferFactor=1)  

#Tlens.data[:,:]  +=  noise.data[:,:]

savef = 'constGrad_1_0.png'
savefn = 'power_noise_n1_0.png'

area = Tulens.area
totalArea = 41252.96
fsky = area/totalArea

# to take lx and ly
Fphi = liteMap.fftFromLiteMap(phi)

# gradient of unlensed temperature with killed small modes
Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr>2000.] = 0.
filt[0] = 0.

'''
# here we get more reali-STICK
Tlens_filt = Tlens.filterFromList([Larr,filt])
gradTu = Tlens_filt.takeGradient()
gradTux = gradTu.gradX
gradTuy = gradTu.gradY
'''

# taking the gradient
gradTu = Tulens.takeGradient()
gradTux = gradTu.gradX
gradTuy = gradTu.gradY
gradTu.gradX.plot(show=False,saveFig = '../lala4.png');plt.close()

gradTu.gradY.plot(show=False,saveFig = '../lala3.png');plt.close()
Tlens.plot(show=False,saveFig = '../lala5.png')

# lensed temperature power spectrum
p2d = fftTools.powerFromLiteMap(Tlens)
l, l, ll, ClT_l_u, l, l = p2d.binInAnnuli('BIN_200_LOG')


# cutting maps edge of Tlens is f'ed up
x0 = 181.65 #181.25
y0 = -1.74
x1 = 178.35 #178.75
y1 = 1.74 
Tulens = Tulens.selectSubMap(x0,x1,y0,y1)
Tlens = Tlens.selectSubMap(x0,x1,y0,y1)
phi = phi.selectSubMap(x0,x1,y0,y1)
gradTux = gradTux.selectSubMap(x0,x1,y0,y1)
gradTuy = gradTuy.selectSubMap(x0,x1,y0,y1)

gradTux.plot(show=False,saveFig = '../lala2.png');plt.close()
gradTuy.plot(show=False,saveFig = '../lala1.png');plt.close()
'''
Tulens.plot()
Tulens_filt.plot()
Tlens.plot()
phi.plot()
'''


# lensed temperature power spectrum
p2d = fftTools.powerFromLiteMap(Tlens)
l, l, ll, ClT_l_c, l, l = p2d.binInAnnuli('BIN_200_LOG')

# lensing potential power spectrum
ps2d = fftTools.powerFromFFT(Fphi)
l, l, ll, Clphi_uncut, l, l = ps2d.binInAnnuli('BIN_200_LOG')


Tlens.plot(show=False,saveFig = '../lala0.png')
plt.close()


Fphi = liteMap.fftFromLiteMap(phi)


ps2d_cut = fftTools.powerFromFFT(Fphi)
l, l, ll, Clphi_cut, l, l = ps2d_cut.binInAnnuli(bin)

'''
#plt.loglog(ll,Clphi,label='cut')
#plt.legend()
#plt.show()
'''

lx = Fphi.lx
ly = Fphi.ly
Nx = Fphi.Nx
Ny = Fphi.Ny


# ROOT MEAN SQUARE
rms = np.sqrt(np.mean(gradTux.data**2+gradTuy.data**2))
print rms

FTlens = liteMap.fftFromLiteMap(Tlens)

# TEST seems ok
#Tlens.data[:,:] = FTlens.mapFromFFT()
#Tlens.plot(show=False,saveFig = '../lala6.png')

FTulens = liteMap.fftFromLiteMap(Tulens)


lx[0]=1.
'''
i_end = 300
l_ar = lx[:]
l_ar[i_end+1:-i_end]=l_ar[i_end+1]

# HERE
lx = l_ar
'''
# filter out small scales
Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr>20000.] = 0.
filt[0] = 0.
Tlens = Tlens.filterFromList([Larr,filt])

Fphi_e = Fphi.copy()
#Fphi_e.kMap[:,:] = (FTlens.kMap[:,:]-FTulens.kMap[:,:])/(1j*(rms*lx))

Fphi_e.kMap[:,:] = (FTlens.kMap[:,:])/(1j*(rms*lx))

Fphi_e.kMap[:,0] = 0. # because lx[0] is set to 1

# weighting
#W = (rms*lx)**2/np.interp(lx,ll,ClT_u)

phi_e = phi.copy()
phi_e.data[:,:] = Fphi_e.mapFromFFT()

# SECOND CUTTING
# cutting maps edge of Tlens is f'ed up
x0 = 181.25
y0 = -1.5
x1 = 178.75
y1 = 1.5 
phi_e = phi_e.selectSubMap(x0,x1,y0,y1)
phi = phi.selectSubMap(x0,x1,y0,y1)


ps2d_e = fftTools.powerFromLiteMap(phi_e)
l, l, ll, Clphi_e, l, l = ps2d_e.binInAnnuli(bin)

ps2d_c = fftTools.powerFromLiteMap(phi_e,phi) 
l, l, ll, Clphi_c, l, l = ps2d_c.binInAnnuli(bin)

diff = np.diff(ll)
diff = np.append(diff,diff[-1])
yerr = np.sqrt(Clphi_e*Clphi_cut)/np.sqrt(ll*diff*fsky)

f1 = open('out.txt', 'w+')
for i in range(len(ll)):
    print >> f1, ('% 8.1f' % ll[i]), ('% 18.8e' % Clphi_uncut[i]), ('% 18.8e' % Clphi_e[i]), ('% 18.8e' % Clphi_c[i]), ('% 18.8e' % yerr[i])

'''
plt.errorbar(ll,Clphi_c/Clphi_uncut,yerr=yerr/Clphi_uncut, fmt='o',label='cross/real uncut')
#plt.errorbar(ll[np.where(Clphi_c>0.)],Clphi_c[np.where(Clphi_c>0.)]/Clphi[np.where(Clphi_c>0.)],yerr=yerr[np.where(Clphi_c>0.)]/Clphi[np.where(Clphi_c>0.)], fmt='o',label='c+')
plt.plot(ll,np.ones(len(ll)),label='ones')
#plt.errorbar(ll[np.where(Clphi_c<0.)],-Clphi_c[np.where(Clphi_c<0.)]/Clphi[np.where(Clphi_c>0.)],yerr=yerr[np.where(Clphi_c<0.)]/Clphi[np.where(Clphi_c>0.)], fmt='o',label='c-')
plt.xscale("log")
plt.ylim([-5,5])
plt.legend()
plt.savefig(savef)
plt.close()
'''

Clphi_e *= ll**4/(2*np.pi)
Clphi_c *= ll**4/(2*np.pi)
Clphi_uncut *= ll**4/(2*np.pi)
Clphi_cut *= ll**4/(2*np.pi)
clphi *= lphi**4/(2*np.pi)
plt.plot(ll,Clphi_e,label='auto')
plt.plot(lphi,clphi,label='theory')
plt.plot(ll,Clphi_uncut,label='real uncut')
plt.plot(ll,Clphi_cut,label='real cut')
#plt.errorbar(ll[np.where(Clphi_c>0.)],Clphi_c[np.where(Clphi_c>0.)],yerr=yerr[np.where(Clphi_c>0.)], fmt='o',label='cross pos')
#plt.errorbar(ll[np.where(Clphi_c<0.)],-Clphi_c[np.where(Clphi_c<0.)],yerr=yerr[np.where(Clphi_c<0.)], fmt='o',label='cross neg')
plt.plot(ll[np.where(Clphi_c>0.)],Clphi_c[np.where(Clphi_c>0.)],'o',label='cross pos')
plt.plot(ll[np.where(Clphi_c<0.)],-Clphi_c[np.where(Clphi_c<0.)], 'o',label='cross neg')
#plt.xscale("log")
plt.yscale("log")
plt.xlim([1,20000])
plt.legend()
plt.savefig(savefn)
plt.close()

plt.plot(ll,ClT_l_u,label='real uncut')
plt.plot(ll,ClT_l_c,label='real cut')
plt.yscale("log")
plt.xlim([1,20000])
plt.legend()
plt.savefig('power_noise_n1_0.pdf')
plt.close()
