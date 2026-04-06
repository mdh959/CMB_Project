# THE MOMENT YOU HAVE DIVISION BY LX, IT FUCKS THINGS UP
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

beam1d_off = {'apply':False}

fullBeamMatrix = {'apply':False}


# binning
bin = 'BIN_200_LOG'

# load maps                                                                                                                                                                                   
dirphi = '/users/boryanah/Blake/bor_sim/'
namep = 'phi_chopped.fits'
phi = liteMap.liteMapFromFits(dirphi+namep)
#phi = liteMap.upgradePixelPitch(phi,2)

x0 = 180. #181.25                                                                                                                                                                  
y0 = -0.5
x1 = 179. #178.75                                                                                                                                                        
y1 = 0.5
phi = phi.selectSubMap(x0,x1,y0,y1)
print phi.Nx,phi.Ny
print phi.info()
phi.writeFits(dirphi+namep, overWrite=True)  

# load phi maps
dirT = '/users/boryanah/Blake/bor_sim/constGrad/'
nameT = 'lensedT_chopped.fits'
nameTu = 'unlensedT_chopped.fits'

# noise params
thetaFWHMarcmin = 0.3
noiseUkArcmin = 1.
thetaFWHM = thetaFWHMarcmin*numpy.pi/(180.*60.)
deltaT = noiseUkArcmin/thetaFWHMarcmin
noiseFactor = .1

scal = 'Aug6_highAcc_CDM_scalCls_new.dat'
theoryPower = np.loadtxt(scal)

theoryPower = np.loadtxt(scal)
l=theoryPower[:,0]
cl_TT=theoryPower[:,1]
cl_EE=theoryPower[:,2]
cl_TE=theoryPower[:,3]
cl_BB=None

lMax = 40000 if 40000 < max(l) else max(l)


l=l[:lMax]
cl_TT=cl_TT[:lMax]*2*numpy.pi/(l*(l+1))
cl_EE=cl_EE[:lMax]*2*numpy.pi/(l*(l+1))
cl_TE=cl_TE[:lMax]*2*numpy.pi/(l*(l+1))


tcmb = 2.7255e6
lphi = theoryPower[:,0]
clphi = theoryPower[:,4]/(lphi**4)/tcmb**2

# OPTION 1: DEFINTELY WANT MEAN HERE
phi.fillWithGaussianRandomField(lphi,clphi)#,bufferFactor=1)
mean = np.mean(phi.data)
phi.data[:,:] = phi.data[:,:] - mean  
phi.writeFits(dirT+namep, overWrite=True)  

# OPTION 2:
phi = liteMap.liteMapFromFits(dirT+namep)
#print phi.info()


# setting temperature map
# OPTION 1:
Tulens = phi.copy()
Tulens,Q_map,U_map =liteMapPol.simPolMapsFromEandB(phi,l,cl_TT,cl_EE,cl_TE,cl_BB,fullBeamMatrix=fullBeamMatrix,beam1d=beam1d_off)
Tulens.writeFits(dirT+nameTu, overWrite=True)  

# OPTION 2:
Tulens = liteMap.liteMapFromFits(dirT+nameTu)

# OPTION 1:
Tlens, Qlens, Ulens  =  lensMaps(phi,Tulens, Tulens, Tulens)
Tlens.writeFits(dirT+nameT, overWrite=True)  

# OPTION 2:
Tlens = liteMap.liteMapFromFits(dirT+nameT)

area = Tulens.area
totalArea = 41252.96
fsky = area/totalArea

# taking the gradient
gradTu = Tulens.takeGradient()
gradTux = gradTu.gradX
gradTuy = gradTu.gradY

# lensing potential power spectrum
#ps2d = fftTools.powerFromFFT(Fphi)
#l, l, ll, Clphi_uncut, l, l = ps2d.binInAnnuli('BIN_200_LOG')

gradTux.plot(show=False,saveFig = 'gradX.png');plt.close()
gradTuy.plot(show=False,saveFig = 'gradY.png');plt.close()

quit()
# cutting maps into small pieces of constant gradient
# Map Bounds: [(x0,y0), (x1,y1)]: [(181.749999,-1.750271),(178.251710,1.748561)] (degrees)
x0 = 181.74999#181.65 #181.25
y0 = -1.74856
x1 = 181.725#178.35 #178.75
y1 = -1.725
Tulens = Tulens.selectSubMap(x0,x1,y0,y1)
Tlens = Tlens.selectSubMap(x0,x1,y0,y1)
phi = phi.selectSubMap(x0,x1,y0,y1)
gradTux = gradTux.selectSubMap(x0,x1,y0,y1)
gradTuy = gradTuy.selectSubMap(x0,x1,y0,y1)

print gradTux.info()

gradTux.plot(show=False,saveFig = 'gradxs.png');plt.close()
gradTuy.plot(show=False,saveFig = 'gradys.png');plt.close()

# to take lx and ly
Fphi = liteMap.fftFromLiteMap(phi)

print Fphi.lx
print Fphi.ly
print Fphi.lx - Fphi.ly

# lensing potential power spectrum
#ps2d = fftTools.powerFromFFT(Fphi)
#l, l, ll, Clphi_uncut, l, l = ps2d.binInAnnuli('BIN_200_LOG')

# add noise (adding after because of boundary periodicity)
NlT = (deltaT*thetaFWHM)**2*np.exp(lphi*(lphi+1.)*thetaFWHM**2/(8.*np.log(2.)))*noiseFactor**2.
noise = Tlens.copy()
noise.fillWithGaussianRandomField(lphi,NlT)#,bufferFactor=1)  
Tlens.data[:,:]  +=  noise.data[:,:]

# rms
#rms = np.sqrt(np.mean(gradTux.data**2+gradTuy.data**2))
rms_x = np.sqrt(np.mean(gradTux.data**2))
rms_y = np.sqrt(np.mean(gradTuy.data**2))

quit()

# WINDOW
win = liteMapPol.initializeCosineWindow(phi,2,0)

phi_win = phi.copy()
phi_win.data[:,:] = win.data*phi.data

noise_win = noise.copy()
noise_win.data[:,:] = win.data*noise.data

Tlens_win = Tlens.copy()
Tlens_win.data[:,:] = win.data*Tlens_win.data

Tulens_win = Tulens.copy()
Tulens_win.data[:,:] = win.data*Tulens_win.data

phi_win.plot(show=False,saveFig = 'phi_win.png');plt.close()

# FOURIER TRANSFORMS
Fphi = liteMap.fftFromLiteMap(phi)
# temp diff
diff_win = phi.copy()
# MUCH MUCH WORSE
#diff_win.data[:,:] = Tlens_win.data - Tulens_win.data
# THAN
diff_win.data[:,:] = Tlens.data - Tulens.data

#  SLIGHTLY BETTER without win!!!
#diff_win.data *= win.data
# fourier temp
Fdiff_win = liteMap.fftFromLiteMap(diff_win)

lx = Fphi.lx
lx[0]=1.

ly = Fphi.ly
ly[0]=1.


Fphi_e = Fphi.copy()
Fphi_e.kMap[:,:] = (Fdiff_win.kMap[:,:])/(1j*lx*rms_x+1j*ly*rms_y)
Fphi_e.kMap[:,0] = 0. # because lx[0] is set to 1

phi_e = phi.copy()
phi_e.data[:,:] = Fphi_e.mapFromFFT(setMeanToZero=True) # TUKA
phi_e.plot(show=False,saveFig = 'phi_e.png',valueRange=[-0.000007,0.000007]);plt.close()
phi.plot(show=False,saveFig = 'phi_r.png');plt.close()

# THIS INSTEAD BECAUSE THIS IS WHAT WE REALLY WANT
Fdiff = Fphi.copy()
Fdiff.kMap[:,:] = (Fphi_e.kMap - Fphi.kMap)#*1j*lx*rms

quit()

Fnoise = Fphi.copy()
Fnoise = liteMap.fftFromLiteMap(noise_win)
Fnoise.kMap[:,:] = Fnoise.kMap/(1j*lx*rms)
Fnoise.kMap[:,0] = 0. # VERY VERY IMPORTANT

diff_win = phi.copy()
diff_win.data[:,:] = Fdiff.mapFromFFT(setMeanToZero=True) # TUKA
# IF NO WIN GET HIGHER POWER BY CONSTANT
diff_win.data[:,:] *= win.data

diff_win.plot(show=False,saveFig = 'diff_.png');plt.close()
noise_win.plot(show=False,saveFig = 'noise_win.png');plt.close()

ps2d = fftTools.powerFromLiteMap(diff_win)
l, l, ll, Clphi_diff, l, l = ps2d.binInAnnuli(bin)

# TRY
#ps2d = fftTools.powerFromLiteMap(noise_win)
ps2d = fftTools.powerFromFFT(Fnoise)
l, l, ll, Clphi_n_win, l, l = ps2d.binInAnnuli(bin)

#Clphi_n *= ll**4/2*np.pi
Clphi_diff *= ll**4/2*np.pi
Clphi_n_win *= ll**4/2*np.pi
NlT *= lphi**4/2*np.pi

#plt.plot(ll,Clphi_n,label='noise')
plt.plot(ll,Clphi_n_win,label='noise win')
plt.plot(ll,Clphi_diff,label='diff')
#plt.plot(lphi,NlT,label='input n')
plt.yscale('log')
plt.xlim(1, 20000)
#plt.ylim(2.e-16,2.e-7)
plt.legend(loc='best')
#plt.xlabel(r'$L$')
#plt.ylabel(r'$l^2 N_l^{dd}/2 \pi$')
plt.savefig('CHOP.png')
plt.close()


quit()


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

