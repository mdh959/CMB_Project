# need to compare the win fun because that decreases the power
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

'''
from mpi4py import MPI

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

# data saved in:
outf = 'out_n0_1_1d_'+str(myrank)+'.txt'
'''

# binning
bin = 'BIN_200_LOG'

# load maps                                                                                                                                                                                   
dirphi = '/users/boryanah/Blake/bor_sim/'
namep = 'phi.fits'
phi = liteMap.liteMapFromFits(dirphi+namep)

'''
# cutting maps
x0 = 180.5
y0 = -1.
x1 = 178.75 
y1 = 0.75
phi = phi.selectSubMap(x0,x1,y0,y1)
'''

# load phi maps
dirT = '/users/boryanah/Blake/bor_sim/constGrad/'
nameT = 'lensedT_constGrad.fits'

# noise params
thetaFWHMarcmin = 0.3
noiseUkArcmin = 1.
thetaFWHM = thetaFWHMarcmin*numpy.pi/(180.*60.)
deltaT = noiseUkArcmin/thetaFWHMarcmin
noiseFactor = .1

scal = 'Aug6_highAcc_CDM_scalCls_new.dat'
theoryPower = np.loadtxt(scal)

tcmb = 2.7255e6
lphi = theoryPower[:,0]
clphi = theoryPower[:,4]/(lphi**4)/tcmb**2

# OPTION 1: DEFINTELY WANT MEAN HERE
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
Tulens.data[:,:] = np.array([np.linspace(-1350,1350,2048),]*2048)

# OPTION 1:
#Tlens, Qlens, Ulens  =  lensMaps(phi,Tulens, Tulens, Tulens)
#Tlens.writeFits(dirT+nameT, overWrite=True)  

# OPTION 2:
Tlens = liteMap.liteMapFromFits(dirT+nameT)

area = Tulens.area
totalArea = 41252.96
fsky = area/totalArea

# to take lx and ly
Fphi = liteMap.fftFromLiteMap(phi)

# *** TESTS
ps2d = fftTools.powerFromFFT(Fphi)
ps2d.powerMap *= 1.e20
print "Fphi"
ps2d.powerMap[0,0] = ps2d.powerMap[1,1]
print np.mean(ps2d.powerMap)
print np.min(ps2d.powerMap)
print np.max(ps2d.powerMap)
ps2d.powerMap = np.log(ps2d.powerMap)

ps2d.plot(show=False,pngFile='../lala1.png',zoomUptoL=20000,valueRange=[-20,7])
plt.close()

'''
print "phiMap"
print Fphi.kMap[500:550,500:550]
Fphi.kMap = np.real(np.conj(Fphi.kMap)*Fphi.kMap)*1.e5
plt.imshow(Fphi.kMap[:,:])
plt.savefig('../lala0.png')
plt.close()
print "myMap"
print Fphi.kMap[:5,:5]
Fphi.plot(show=False)
plt.savefig('../lala2.png')
print "powerMap"
print ps2d.powerMap[1400:1405,1400:1405]
'''
# TESTS ***

# taking the gradient
gradTu = Tulens.takeGradient()
gradTux = gradTu.gradX
gradTuy = gradTu.gradY

# lensing potential power spectrum
#ps2d = fftTools.powerFromFFT(Fphi)
#l, l, ll, Clphi_uncut, l, l = ps2d.binInAnnuli('BIN_200_LOG')


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

# add noise
NlT = (deltaT*thetaFWHM)**2*np.exp(lphi*(lphi+1.)*thetaFWHM**2/(8.*np.log(2.)))*noiseFactor**2.
noise = Tlens.copy()
noise.fillWithGaussianRandomField(lphi,NlT)#,bufferFactor=1)  
Tlens.data[:,:]  +=  noise.data[:,:]

# rms
rms = np.sqrt(np.mean(gradTux.data**2+gradTuy.data**2))

# taking the gradient of phi
gradPhi = phi.takeGradient()
dx = gradPhi.gradX
dy = gradPhi.gradY

# transforming to d
Fphi_cut = liteMap.fftFromLiteMap(phi)
# turn into dd
Fphi_cut.kMap *= Fphi_cut.modLMap
phi.data[:,:] = Fphi_cut.mapFromFFT()#setMeanToZero=True)
# TUK


# WINDOW
win = liteMapPol.initializeCosineWindow(phi,200,0)


# WINDOW ALL
phi_win = phi.copy()
phi_win.data[:,:] = win.data*phi.data
Fphi_win = liteMap.fftFromLiteMap(phi_win)#,applySlepianTaper=True,nresForSlepian=1.)

dx_win = dx.copy()
dx_win.data[:,:] = win.data*dx.data
Fdx_win = liteMap.fftFromLiteMap(dx_win)#,applySlepianTaper=True,nresForSlepian=1.)

noise_win = noise.copy()
noise_win.data[:,:] = win.data*noise.data

Tlens_win = Tlens.copy()
Tlens_win.data[:,:] = win.data*Tlens_win.data

Tulens_win = Tulens.copy()
Tulens_win.data[:,:] = win.data*Tulens_win.data

# POWER SPECTRA
ps2d = fftTools.powerFromLiteMap(phi_win)
ps2d.powerMap *= 1.e20
print "phi_win"
ps2d.powerMap[0,0] = ps2d.powerMap[1,1]
print np.mean(ps2d.powerMap)
print np.min(ps2d.powerMap)
print np.max(ps2d.powerMap)
ps2d.powerMap = np.log(ps2d.powerMap)

ps2d.plot(show=False,pngFile='../lala0.png',zoomUptoL=20000,valueRange=[-20,7])
plt.close()


# map with lensed - unlensed
diff = phi.copy()
#diff.data[:,:] = Tlens.data - Tulens.data
diff.data[:,:] = Tlens_win.data - Tulens_win.data

lx = Fphi_win.lx
ly = Fphi_win.ly
Nx = Fphi_win.Nx
Ny = Fphi_win.Ny

Fdiff = liteMap.fftFromLiteMap(diff)

lx[0]=1.

Fphi_e = Fphi_win.copy()
# TUK actually dx est
Fphi_e.kMap[:,:] = (Fdiff.kMap[:,:])/((rms))
#Fphi_e.kMap[:,0] = 0. # because lx[0] is set to 1

ps2d = fftTools.powerFromFFT(Fphi_e)
#ps2d.powerMap *= 1.e20
print "phi_e"
ps2d.powerMap[0,0] = ps2d.powerMap[1,1]
print np.mean(ps2d.powerMap)
print np.min(ps2d.powerMap)
print np.max(ps2d.powerMap)
ps2d.powerMap = np.log(ps2d.powerMap)

ps2d.plot(show=False,pngFile='../lala3.png',zoomUptoL=20000,valueRange=[-20,7])
plt.close()

phi_e = phi.copy()
phi_e.data[:,:] = Fphi_e.mapFromFFT()
phi_e_win = phi_e.copy()
phi_e_win.data[:,:] = win.data*phi_e.data
Fphi_e_win = liteMap.fftFromLiteMap(phi_e_win)

ps2d = fftTools.powerFromLiteMap(phi_e_win)
#ps2d.powerMap *= 1.e20
print "phi_e_win"
ps2d.powerMap[0,0] = ps2d.powerMap[1,1]
print np.mean(ps2d.powerMap)
print np.min(ps2d.powerMap)
print np.max(ps2d.powerMap)
ps2d.powerMap = np.log(ps2d.powerMap)

ps2d.plot(show=False,pngFile='../lala4.png',zoomUptoL=20000,valueRange=[-20,7])
plt.close()

phi_e_win.plot(show=False,saveFig = '../lala7.png');plt.close()
phi_win.plot(show=False,saveFig = '../lala8.png');plt.close()

Fdiff.kMap[:,:] = (Fdx_win.kMap - Fphi_e_win.kMap)
#Fdiff.kMap[:,:] = (Fphi_win.kMap - Fphi_e_win.kMap)
Fdiff.kMap[:,:] *= rms

ps2d = fftTools.powerFromFFT(Fdiff)
#ps2d.powerMap *= 1.e20
print "Fdiff"
ps2d.powerMap[0,0] = ps2d.powerMap[1,1]
print np.mean(ps2d.powerMap)
print np.min(ps2d.powerMap)
print np.max(ps2d.powerMap)

ps2d.powerMap = np.log(ps2d.powerMap)
ps2d = fftTools.powerFromFFT(Fdiff)
#ps2d.plot(show=False,pngFile='../lala5.png',zoomUptoL=20000,valueRange=[-5,43])
#plt.close()
l, l, ll, Clphi_diff, l, l = ps2d.binInAnnuli(bin)

ps2d = fftTools.powerFromLiteMap(noise)
ps2d.powerMap *= 1.e20
print "noise"
ps2d.powerMap[0,0] = ps2d.powerMap[1,1]
print np.mean(ps2d.powerMap)
print np.min(ps2d.powerMap)
print np.max(ps2d.powerMap)
ps2d.powerMap = np.log(ps2d.powerMap)
ps2d = fftTools.powerFromLiteMap(noise)
#ps2d.plot(show=False,pngFile='../lala6.png',zoomUptoL=20000,valueRange=[-5,43])
#plt.close()
l, l, ll, Clphi_n, l, l = ps2d.binInAnnuli(bin)

ps2d = fftTools.powerFromLiteMap(noise_win)
ps2d.powerMap *= 1.e20
print "noise"
ps2d.powerMap[0,0] = ps2d.powerMap[1,1]
print np.mean(ps2d.powerMap)
print np.min(ps2d.powerMap)
print np.max(ps2d.powerMap)
ps2d.powerMap = np.log(ps2d.powerMap)
ps2d = fftTools.powerFromLiteMap(noise_win)
#ps2d.plot(show=False,pngFile='../lala6.png',zoomUptoL=20000,valueRange=[-5,43])
#plt.close()
l, l, ll, Clphi_n_win, l, l = ps2d.binInAnnuli(bin)

Fnoise = liteMap.fftFromLiteMap(noise)
ps2d = fftTools.powerFromFFT(Fdiff,Fnoise)
print "Fdiff,noise"
ps2d.powerMap[0,0] = ps2d.powerMap[1,1]
print np.mean(ps2d.powerMap)
print np.min(ps2d.powerMap)
print np.max(ps2d.powerMap)

ps2d.powerMap = np.log(ps2d.powerMap)
Fnoise = liteMap.fftFromLiteMap(noise)
ps2d = fftTools.powerFromFFT(Fdiff,Fnoise)
#ps2d.plot(show=False,pngFile='../lala5.png',zoomUptoL=20000,valueRange=[-5,43])
#plt.close()
#l, l, ll, Clphi_ncr, l, l = ps2d.binInAnnuli(bin)

Clphi_n *= ll**4/2*np.pi
Clphi_diff *= ll**2/2*np.pi
Clphi_n_win *= ll**4/2*np.pi
NlT *= lphi**4/2*np.pi

plt.plot(ll,Clphi_n,label='noise')
plt.plot(ll,Clphi_n_win,label='noise win')
plt.plot(ll,Clphi_diff,label='diff')
plt.plot(lphi,NlT,label='input n')
plt.yscale('log')
plt.xlim(1, 20000)
#plt.ylim(2.e-16,2.e-7)
plt.legend(loc='best')
#plt.xlabel(r'$L$')
#plt.ylabel(r'$l^2 N_l^{dd}/2 \pi$')
plt.savefig('CHECK100000000.png')
plt.close()

quit()
ps2d_e = fftTools.powerFromLiteMap(phi_e_win)
l, l, ll, Clphi_e_win, l, l = ps2d_e.binInAnnuli(bin)

ps2d_e = fftTools.powerFromLiteMap(phi_win)
l, l, ll, Clphi_win, l, l = ps2d_e.binInAnnuli(bin)

ps2d_c = fftTools.powerFromLiteMap(phi_e_win,phi_win)
l, l, ll, Clphi_c, l, l = ps2d_c.binInAnnuli(bin)

diff = np.diff(ll)
diff = np.append(diff,diff[-1])
yerr = np.sqrt(Clphi_e_win*Clphi_uncut)/np.sqrt(ll*diff*fsky)

clphi *= lphi**4/(2*np.pi)
Clphi_uncut *= ll**4/(2*np.pi)
Clphi_e *= ll**2/(2*np.pi)
plt.plot(lphi,clphi,label='theory')
plt.plot(ll,Clphi_c,label='cross win')
plt.plot(ll,Clphi_win,label='real win')
plt.plot(ll,Clphi_e_win,label='recon win')
plt.yscale('log')
plt.xlim(1, 20000)
#plt.ylim(2.e-16,2.e-7)
plt.legend(loc='best')
#plt.xlabel(r'$L$')
#plt.ylabel(r'$l^2 N_l^{dd}/2 \pi$')
plt.savefig('CHECK100000001.png')
plt.close()
quit()
f1 = open(outf, 'w+')
for i in range(len(ll)):
    print >> f1, ('% 8.1f' % ll[i]), ('% 18.8e' % Clphi_win[i]), ('% 18.8e' % Clphi_e_win[i]), ('% 18.8e' % Clphi_c[i]), ('% 18.8e' % yerr[i])

