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
nameT = 'lensedT_sin.fits'

# load random phi map                                          
dirphi = '/users/boryanah/Blake/bor_sim/'
namep = 'phi_sin.fits'
phi = liteMap.liteMapFromFits(dirphi+namep)

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
pixScale = 0.102489
print pixScale

# CONSTANT GRADIENT
# setting temperature map
Tulens = phi.copy()
gradT_th = phi.copy()
Tulens.data[:,:] = np.array([600*np.sin(np.arange(2048)*2*np.pi/2048*1.),]*2048)#np.array([np.linspace(-1350,1350,2048),]*2048)
gradT_th.data[:,:] = np.array([600*2*np.pi/(2048*np.pi/180*pixScale/60)*np.cos(np.arange(2048)*2*np.pi/2048*1.),]*2048)#np.array([np.linspace(-1350,1350,2048),]*2048)
gradT_thX = np.array([600*2*np.pi/(2048*np.pi/180*pixScale/60)*np.cos(np.arange(2048)*2*np.pi/2048*1)])

# OPTION 1:
#Tlens, Qlens, Ulens  =  lensMaps(phi,Tulens, Tulens, Tulens)
#Tlens.writeFits(dirT+nameT, overWrite=True)  

# OPTION 2:
Tlens = liteMap.liteMapFromFits(dirT+nameT)

print "gradT"
print gradT_th.data[:5,:5]
print gradT_th.data[-5:,-5:]

# taking the gradient
gradTu = Tulens.takeGradient()
gradTux = gradTu.gradX
gradTuy = gradTu.gradY

print "gradTux"
print gradTux.data[:5,:5]
print gradTux.data[-5:,-5:]

# ROOT MEAN SQUARE 1/sqrt max value
rms = np.sqrt(np.mean(gradTux.data**2+gradTuy.data**2))
print "rms = ", rms

# add noise
# noise ps
NlT = (deltaT*thetaFWHM)**2*np.exp(lphi*(lphi+1.)*thetaFWHM**2/(8.*np.log(2.)))*noiseFactor**2.
noise = Tlens.copy()
noise.fillWithGaussianRandomField(lphi,NlT)#,bufferFactor=1)  

Tlens.data[:,:]  +=  noise.data[:,:]

savef = 'sin_0_1.png'
savefn = 'power_sin_noise_n0_1.png'

area = Tulens.area
totalArea = 41252.96
fsky = area/totalArea

# to take lx and ly
Fphi = liteMap.fftFromLiteMap(phi)

kappa = phiToKappa(phi)

# gradient of unlensed temperature with killed small modes
#Larr = np.arange(40000.)
#filt = np.ones(len(Larr))
#filt[Larr>2000.] = 0.
#filt[0] = 0.

phi.plot(show=False,saveFig = '../lala5.png');plt.close()
Tlens.plot(show=False,saveFig = '../lala2.png');plt.close()
gradTu.gradX.plot(show=False,saveFig = '../lala4.png');plt.close()
kappa.plot(show=False,saveFig = '../lala3.png');plt.close()


# lensing potential power spectrum
ps2d = fftTools.powerFromFFT(Fphi)
l, l, ll, Clphi_uncut, l, l = ps2d.binInAnnuli('BIN_200_LOG')

lx = Fphi.lx
ly = Fphi.ly
Nx = Fphi.Nx
Ny = Fphi.Ny

# map with lensed - unlensed
diff = phi.copy()
diff.data[:,:] = Tlens.data - Tulens.data

Fdiff = liteMap.fftFromLiteMap(diff)

lx[0]=1.

# TUK
Fphi_e = Fphi.copy()
Fphi_e.kMap[:,:] = (Fdiff.kMap[:,:])/(1j*(gradT_thX))
#Fphi_e.kMap[:,:] = (FTlens.kMap[:,:])/(1j*(rms*lx))
#Fphi_e.kMap[:,0] = 0. # because lx[0] is set to 1
Fphi_e.kMap[:,:] *= lx

# uncut lensed temperature power spectrum
p2d = fftTools.powerFromLiteMap(Tulens)
l, l, ll, ClT_u, l, l = p2d.binInAnnuli('BIN_200_LOG')

# weighting
#W = (gradT_thX*lx)**2/np.interp(lx,ll,ClT_u)
#Fphi_e.kMap *= W
phi_e = phi.copy()
phi_e.data[:,:] = Fphi_e.mapFromFFT()
phi_e.plot(show=False,saveFig = '../lala6.png')#,valueRange=[-0.000005,0.000005]);plt.close()

ps2d_e = fftTools.powerFromFFT(Fphi_e)
#ps2d_e = fftTools.powerFromLiteMap(phi_e,applySlepianTaper=True,nresForSlepian=1.)
l, l, ll, Clphi_e, l, l = ps2d_e.binInAnnuli(bin)

clphi *= lphi**4/(2*np.pi)
Clphi_uncut *= ll**4/(2*np.pi)
Clphi_e /= (2*np.pi)
plt.plot(lphi,clphi,label='theory')
plt.plot(ll,Clphi_uncut,label='real uncut')
plt.plot(ll,Clphi_e,label='recon')
plt.yscale("log")
plt.xlim([1,20000])
plt.savefig(savefn)
quit()
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

'''
# uncut lensed temperature power spectrum
p2d = fftTools.powerFromLiteMap(Tlens)
l, l, ll, ClT_l_u, l, l = p2d.binInAnnuli('BIN_200_LOG')
'''

'''
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
'''

print  "Tlens:"
print Tlens.info()


win = liteMapPol.initializeCosineWindow(phi,0,0)#200,0)
phi_win = phi.copy()
phi_win.data[:,:] = win.data*phi.data
print "phi_win:"
print phi_win.info()
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


'''
# lensed temperature power spectrum
p2d = fftTools.powerFromLiteMap(Tlens)
l, l, ll, ClT_l_c, l, l = p2d.binInAnnuli('BIN_200_LOG')

p2d = fftTools.powerFromLiteMap(Tlens, Tulens)
l, l, ll, ClT_T, l, l = p2d.binInAnnuli('BIN_200_LOG')

for i in range(len(ll)):
    print ('% 8.1f' % ll[i]), ('% 18.8e' % ClT_l_c[i]), ('% 18.8e' % ClT_T[i])
'''

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


Fdiff = liteMap.fftFromLiteMap(diff)

lx[0]=1.

Fphi_e = Fphi_win.copy()
Fphi_e.kMap[:,:] = (Fdiff.kMap[:,:])/(1j*(gradT_th*lx))
#Fphi_e.kMap[:,:] = (FTlens.kMap[:,:])/(1j*(rms*lx))
Fphi_e.kMap[:,0] = 0. # because lx[0] is set to 1

# weighting
#W = (rms*lx)**2/np.interp(lx,ll,ClT_u)


phi_e = phi.copy()
phi_e.data[:,:] = Fphi_e.mapFromFFT()
phi_e_win = phi.copy()
phi_e_win.data[:,:] = win.data*phi_e.data
phi_e_win.plot(show=False,saveFig = '../lala6.png');plt.close()

quit()
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
