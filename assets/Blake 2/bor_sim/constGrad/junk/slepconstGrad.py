import matplotlib
matplotlib.use('Agg')
import astLib.astWCS
import astLib.astCoords
import astLib.astCalc
import numpy as np
import matplotlib.pyplot as plt
from flipper import *
from flipperPol import *
from lensingTools import *

# binning
bin = 'BIN_200_LOG'#'BIN_100_LOG''BIN_250_1000'

# load phi maps
dirT = '/users/boryanah/Blake/bor_sim/constGrad/'
nameT = 'lensedT_constGrad.fits'

# load random phi map                                          
dirphi = '/users/boryanah/Blake/bor_sim/'
namep = 'phi.fits'
phi = liteMap.liteMapFromFits(dirphi+namep)

# noise params
thetaFWHMarcmin = 0.3
noiseUkArcmin = 1.
thetaFWHM = thetaFWHMarcmin*numpy.pi/(180.*60.)
deltaT = noiseUkArcmin/thetaFWHMarcmin
noiseFactor = .01

scal = 'Aug6_highAcc_CDM_scalCls_new.dat'
theoryPower = np.loadtxt(scal)

tcmb = 2.7255e6
lphi = theoryPower[:,0]
clphi = theoryPower[:,4]/(lphi**4)/tcmb**2

# OPTION 1:
#phi.fillWithGaussianRandomField(lphi,clphi)#,bufferFactor=1)
#mean = np.mean(phi.data)
#phi.data[:,:] = phi.data[:,:] - mean  
#phi.writeFits(dirT+namep, overWrite=True)  

# OPTION 2:
phi = liteMap.liteMapFromFits(dirT+namep)
#print phi.info()

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

Tlens.data[:,:]  +=  noise.data[:,:]

savef = 'constGrad_1_0.png'
savefn = 'power_noise_n1_0.pdf'

area = Tulens.area
totalArea = 41252.96
fsky = area/totalArea

# to take lx and ly
Fphi = liteMap.fftFromLiteMap(phi)

# gradient of unlensed temperature with killed small modes
#Larr = np.arange(40000.)
#filt = np.ones(len(Larr))
#filt[Larr>2000.] = 0.
#filt[0] = 0.

# taking the gradient
gradTu = Tulens.takeGradient()
gradTux = gradTu.gradX
gradTuy = gradTu.gradY
gradTu.gradX.plot(show=False,saveFig = '../lala4.png');plt.close()
gradTu.gradY.plot(show=False,saveFig = '../lala3.png');plt.close()


# lensing potential power spectrum
ps2d = fftTools.powerFromFFT(Fphi)
l, l, ll, Clphi_uncut, l, l = ps2d.binInAnnuli('BIN_200_LOG')

'''
win = liteMapPol.initializeCosineWindow(phi,100,0)
exp = phi.copy()
exp.data[:,:] = win.data*phi.data
print "exp:"
print exp.info()
mean = np.mean(exp.data)
exp.data[:,:] = exp.data[:,:]-mean
exp.plot(show=False,saveFig = '../lala2.png');plt.close()
'''


# uncut lensed temperature power spectrum
p2d = fftTools.powerFromLiteMap(Tlens)
l, l, ll, ClT_l_u, l, l = p2d.binInAnnuli('BIN_200_LOG')


# cutting maps edge of Tlens is f'ed up
x0 = 181.65#181.65 #181.25
y0 = -1.74
x1 = 178.35#178.35 #178.75
y1 = 1.74 
Tulens = Tulens.selectSubMap(x0,x1,y0,y1)
Tlens = Tlens.selectSubMap(x0,x1,y0,y1)
phi = phi.selectSubMap(x0,x1,y0,y1)
gradTux = gradTux.selectSubMap(x0,x1,y0,y1)
gradTuy = gradTuy.selectSubMap(x0,x1,y0,y1)
mean = np.mean(Tulens.data)
print  "Tlens:"
print Tlens.info()
Tulens.data[:,:] = Tulens.data[:,:] - mean
mean = np.mean(Tlens.data)
Tlens.data[:,:] = Tlens.data[:,:] - mean
mean = np.mean(phi.data)
phi.data[:,:] = phi.data[:,:] - mean

win = liteMapPol.initializeCosineWindow(phi,100,0)
phi_win = phi.copy()
phi_win.data[:,:] = win.data*phi.data
print "phi_win:"
print phi_win.info()
mean = np.mean(phi_win.data)
phi_win.data[:,:] = phi_win.data[:,:]-mean
phi_win.plot(show=False,saveFig = '../lala5.png');plt.close()

ps2d_win = fftTools.powerFromLiteMap(phi_win)#,applySlepianTaper=True,nresForSlepian=1.)
l, l, ll, Clphi_win, l, l = ps2d_win.binInAnnuli(bin)

'''
ps2d_win = fftTools.powerFromLiteMap(exp)#,applySlepianTaper=True,nresForSlepian=1.)
l, l, ll, Clphi_exp, l, l = ps2d_win.binInAnnuli(bin)

Clphi_uncut *= ll**4/(2*np.pi)
Clphi_exp *= ll**4/(2*np.pi)
Clphi_win *= ll**4/(2*np.pi)
clphi *= lphi**4/(2*np.pi)
plt.plot(lphi,clphi,label='theory')
plt.plot(ll,Clphi_uncut,label='real uncut')
plt.plot(ll,Clphi_win,label='real win')
plt.plot(ll,Clphi_exp,label='real uncut cos')
plt.yscale("log")
plt.xlim([1,20000])
plt.legend()
plt.savefig(savefn)
plt.close()
'''

# map with lensed - unlensed
diff = phi.copy()
diff.data[:,:] = Tlens.data - Tulens.data

# lensed temperature power spectrum
p2d = fftTools.powerFromLiteMap(Tlens)
l, l, ll, ClT_l_c, l, l = p2d.binInAnnuli('BIN_200_LOG')

Fphi_cut = liteMap.fftFromLiteMap(phi)
Fphi_win = liteMap.fftFromLiteMap(phi_win)#,applySlepianTaper=True,nresForSlepian=1.)
'''
Fphi_win = liteMap.fftFromLiteMap(phi,applySlepianTaper=True,nresForSlepian=1.)
phi_win = phi.copy()
phi_win.data[:,:] = Fphi_win.mapFromFFT()
phi_win.plot(show=False,saveFig = '../lala0.png')
phi_win.info()
plt.close()

ps2d_cut = fftTools.powerFromFFT(Fphi_win)#,applySlepianTaper=True,nresForSlepian=1.)
#ps2d_cut = fftTools.powerFromLiteMap(phi,applySlepianTaper=True,nresForSlepian=1.)
l, l, ll, Clphi_cut, l, l = ps2d_cut.binInAnnuli(bin)
'''
lx = Fphi_win.lx
ly = Fphi_win.ly
Nx = Fphi_win.Nx
Ny = Fphi_win.Ny
print phi.info()

# ROOT MEAN SQUARE
rms = np.sqrt(np.mean(gradTux.data**2+gradTuy.data**2))
print rms

Fdiff = liteMap.fftFromLiteMap(diff)

lx[0]=1.

Fphi_e = Fphi_win.copy()
Fphi_e.kMap[:,:] = (Fdiff.kMap[:,:])/(1j*(rms*lx))
#Fphi_e.kMap[:,:] = (FTlens.kMap[:,:])/(1j*(rms*lx))
Fphi_e.kMap[:,0] = 0. # because lx[0] is set to 1

# weighting
#W = (rms*lx)**2/np.interp(lx,ll,ClT_u)

phi_e = phi.copy()
phi_e.data[:,:] = Fphi_e.mapFromFFT()
mean = np.mean(phi_e.data)
phi_e.data[:,:] = phi_e.data[:,:]-mean
phi_e_win = phi.copy()
phi_e_win.data[:,:] = win.data*phi_e.data
phi_e_win.plot(show=False,saveFig = '../lala6.png');plt.close()
'''
# SECOND CUTTING
# cutting maps edge of Tlens is f'ed up
x0 = 181.25
y0 = -1.5
x1 = 178.75
y1 = 1.5 
phi_e = phi_e.selectSubMap(x0,x1,y0,y1)
phi = phi.selectSubMap(x0,x1,y0,y1)
'''

ps2d_e = fftTools.powerFromFFT(Fphi_e)
#ps2d_e = fftTools.powerFromLiteMap(phi_e,applySlepianTaper=True,nresForSlepian=1.)
l, l, ll, Clphi_e, l, l = ps2d_e.binInAnnuli(bin)

ps2d_e = fftTools.powerFromLiteMap(phi_e_win)
#ps2d_e = fftTools.powerFromLiteMap(phi_e,applySlepianTaper=True,nresForSlepian=1.)
l, l, ll, Clphi_e_win, l, l = ps2d_e.binInAnnuli(bin)

ps2d_e = fftTools.powerFromFFT(Fphi_cut)
l, l, ll, Clphi_cut, l, l = ps2d_e.binInAnnuli(bin)

ps2d_c = fftTools.powerFromLiteMap(phi_e_win,phi_win)
#ps2d_c = fftTools.powerFromFFT(Fphi_e,Fphi_win)
l, l, ll, Clphi_c, l, l = ps2d_c.binInAnnuli(bin)

ps2d_c = fftTools.powerFromFFT(Fphi_e,Fphi_cut)
l, l, ll, Clphi_c_cut, l, l = ps2d_c.binInAnnuli(bin)

diff = np.diff(ll)
diff = np.append(diff,diff[-1])
yerr = np.sqrt(Clphi_e*Clphi_win)/np.sqrt(ll*diff*fsky)

f1 = open('out.txt', 'w+')
for i in range(len(ll)):
    print >> f1, ('% 8.1f' % ll[i]), ('% 18.8e' % Clphi_uncut[i]), ('% 18.8e' % Clphi_e[i]), ('% 18.8e' % Clphi_c[i]), ('% 18.8e' % yerr[i])


Clphi_e_win *= ll**4/(2*np.pi)
Clphi_e *= ll**4/(2*np.pi)
Clphi_c *= ll**4/(2*np.pi)
Clphi_uncut *= ll**4/(2*np.pi)
Clphi_c_cut *= ll**4/(2*np.pi)
Clphi_cut *= ll**4/(2*np.pi)
Clphi_win *= ll**4/(2*np.pi)
clphi *= lphi**4/(2*np.pi)
plt.plot(ll,Clphi_e,label='auto')
plt.plot(ll,Clphi_e_win,label='auto win')
plt.plot(lphi,clphi,label='theory')
plt.plot(ll,Clphi_uncut,label='real uncut')
#plt.plot(ll,Clphi_cut,label='real cut')
plt.plot(ll,Clphi_win,label='real win')
plt.plot(ll[np.where(Clphi_c>0.)],Clphi_c[np.where(Clphi_c>0.)],'o',label='cross pos')
#plt.plot(ll,Clphi_cut,label='real cut raw')
#plt.plot(ll,Clphi_c_cut,label='cross cut raw')
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
plt.savefig(savef)
plt.close()
