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
from mpi4py import MPI

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

# data saved in:
outf = 'out_n1_0_'+str(myrank)+'.txt'

# binning
bin = 'BIN_200_LOG'

# load random phi map                                          
dirphi = '/users/boryanah/Blake/bor_sim/'
namep = 'phi.fits'
phi = liteMap.liteMapFromFits(dirphi+namep)

# noise params
thetaFWHMarcmin = .3
noiseUkArcmin = 1.
thetaFWHM = thetaFWHMarcmin*numpy.pi/(180.*60.)
deltaT = noiseUkArcmin/thetaFWHMarcmin
noiseFactor = 1.

scal = 'Aug6_highAcc_CDM_scalCls_new.dat'
theoryPower = np.loadtxt(scal)

tcmb = 2.7255e6
lphi = theoryPower[:,0]
clphi = theoryPower[:,4]/(lphi**4)/tcmb**2

# create phi
phi.fillWithGaussianRandomField(lphi,clphi)#,bufferFactor=1)
mean = np.mean(phi.data)
phi.data[:,:] = phi.data[:,:] - mean  

# setting temperature map
Tulens = phi.copy()
Tulens.data[:,:] = np.array([np.linspace(-1350,1350,2048),]*2048)

# lens T
Tlens, Qlens, Ulens  =  lensMaps(phi,Tulens, Tulens, Tulens)

# add noise
NlT = (deltaT*thetaFWHM)**2*np.exp(lphi*(lphi+1.)*thetaFWHM**2/(8.*np.log(2.)))*noiseFactor**2.
noise = Tlens.copy()
noise.fillWithGaussianRandomField(lphi,NlT)#,bufferFactor=1)  
Tlens.data[:,:]  +=  noise.data[:,:]

area = Tulens.area
totalArea = 41252.96
fsky = area/totalArea

# to take lx and ly
Fphi = liteMap.fftFromLiteMap(phi)

# taking the gradient
gradTu = Tulens.takeGradient()
gradTux = gradTu.gradX
gradTuy = gradTu.gradY

# lensing potential power spectrum
ps2d = fftTools.powerFromFFT(Fphi)
l, l, ll, Clphi_uncut, l, l = ps2d.binInAnnuli('BIN_200_LOG')

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

# rms
rms = np.sqrt(np.mean(gradTux.data**2+gradTuy.data**2))

# cos win
win = liteMapPol.initializeCosineWindow(phi,200,0)
phi_win = phi.copy()
phi_win.data[:,:] = win.data*phi.data

# map with lensed - unlensed
diff = phi.copy()
diff.data[:,:] = Tlens.data - Tulens.data


Fphi_cut = liteMap.fftFromLiteMap(phi)
Fphi_win = liteMap.fftFromLiteMap(phi_win)#,applySlepianTaper=True,nresForSlepian=1.)

lx = Fphi_win.lx
ly = Fphi_win.ly
Nx = Fphi_win.Nx
Ny = Fphi_win.Ny

Fdiff = liteMap.fftFromLiteMap(diff)

lx[0]=1.

Fphi_e = Fphi_win.copy()
Fphi_e.kMap[:,:] = (Fdiff.kMap[:,:])/(1j*(rms*lx))
Fphi_e.kMap[:,0] = 0. # because lx[0] is set to 1


phi_e = phi.copy()
phi_e.data[:,:] = Fphi_e.mapFromFFT()
phi_e_win = phi.copy()
phi_e_win.data[:,:] = win.data*phi_e.data

ps2d_e = fftTools.powerFromLiteMap(phi_e_win)
l, l, ll, Clphi_e_win, l, l = ps2d_e.binInAnnuli(bin)

ps2d_e = fftTools.powerFromLiteMap(phi_win)
l, l, ll, Clphi_win, l, l = ps2d_e.binInAnnuli(bin)

ps2d_c = fftTools.powerFromLiteMap(phi_e_win,phi_win)
l, l, ll, Clphi_c, l, l = ps2d_c.binInAnnuli(bin)


diff = np.diff(ll)
diff = np.append(diff,diff[-1])
yerr = np.sqrt(Clphi_e_win*Clphi_uncut)/np.sqrt(ll*diff*fsky)

f1 = open(outf, 'w+')
for i in range(len(ll)):
    print >> f1, ('% 8.1f' % ll[i]), ('% 18.8e' % Clphi_win[i]), ('% 18.8e' % Clphi_e_win[i]), ('% 18.8e' % Clphi_c[i]), ('% 18.8e' % yerr[i])

