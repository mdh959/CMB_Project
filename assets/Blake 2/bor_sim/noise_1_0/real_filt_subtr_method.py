import matplotlib
matplotlib.use('Agg')
import astLib.astWCS
import astLib.astCoords
import astLib.astCalc
import numpy as np
import matplotlib.pyplot as plt
from flipper import *
from lensingTools import *

savef = 'power_filt_subtr_n1_0_realCl.pdf'
savefn = 'power_noise_n1_0.pdf'
saverat = 'power_rat_filt_subtr_n1_0_realCl.pdf'
savekappaest = 'kappa_est.fits'
savekappa = 'kappa.fits'
# ASK IF NEED CL_TU
print savef

# hardwiring constants
#area = 12.239927
area = 3.056636
totalArea = 41252.96
fsky = area/totalArea

nametu = 'T_map1.fits'
nametl = 'Tlens1.fits'
nametln = 'Tlens1_noise_1_0.fits'
namepn = 'phi_new1.fits'

# load maps                                                                                                             
dirphi = '/users/boryanah/Blake/bor_sim/'

#Tlens = liteMap.liteMapFromFits(dirphi+nametl)
Tlens = liteMap.liteMapFromFits(dirphi+nametln)
Tulens = liteMap.liteMapFromFits(dirphi+nametu)
phi = liteMap.liteMapFromFits(dirphi+namepn)

# kappa
kappa = phiToKappa(phi)
print kappa.info()

# to take lx and ly
Fphi = liteMap.fftFromLiteMap(phi)

#Fphi.ix 0 to 2047
lx = Fphi.lx
ly = Fphi.ly
#print ly[100],ly[200],ly[-200]
Nx = Fphi.Nx # lx i think goes from 0 to Ny
Ny = Fphi.Ny

# gradient of unlensed temperature with killed small modes
Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr>2000.] = 0.
filt[0] = 0.
# here we get more reali-STICK
Tlens_filt = Tlens.filterFromList([Larr,filt])
gradTu = Tlens_filt.takeGradient()
gradTux = gradTu.gradX
gradTuy = gradTu.gradY

'''
gradTu.gradX.plot()
gradTu.gradY.plot()
Tulens.plot()
Tulens_filt.plot()
Tlens.plot()
phi.plot()
'''

# unlensed temperature power spectrum ok because only used in Nphiphi
p2d = fftTools.powerFromLiteMap(Tulens)
l, l, ll, ClT_u, l, l = p2d.binInAnnuli('BIN_200_LOG')
'''
# filtered lensed temperature power spectrum
p2d = fftTools.powerFromLiteMap(Tulens)
l, l, ll, ClT_u, l, l = p2d.binInAnnuli('BIN_200_LOG')
'''
# lensing potential power spectrum
ps2d = fftTools.powerFromFFT(Fphi)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_200_LOG')

# parameters for noise estimation
thetaFWHMarcmin = 0.3
noiseUkArcmin = 1.
thetaFWHM = thetaFWHMarcmin*numpy.pi/(180.*60.)
deltaT = noiseUkArcmin/thetaFWHMarcmin
noiseFactor = 1.
NlT = (deltaT*thetaFWHM)**2*np.exp(ll*(ll+1.)*thetaFWHM**2/(8.*np.log(2.)))*noiseFactor**2.

# Check if in correct units - YES
#plt.loglog(ll,NlT)
#plt.loglog(ll,ClT_u)
#plt.show()
#plt.savefig('a.png')

# estimator in Fourier space
Fphi_est = Fphi.copy()
Fphi_est_unnorm = Fphi.copy()
Fphi_est.kMap[:,:] = 0
Fphi_est_unnorm.kMap[:,:] = 0


# estimator in real space
phi_est = phi.copy()
phi_est.data[:,:] = 0

# follows prescription in overleaf
# roughly lx[200] = 20000 which is roughly ll[-1]
# have to have complete l[0:whatever] otherwise lose modes
'''
i_end = 5
l_beg = 0.00001
l_last = 400.
'''

#ell = np.sqrt(lx**2+ly**2)
#print ell[:10]
ell = Fphi.modLMap

C_ellT_u = np.interp(ell,ll,ClT_u)
N_ellT = np.interp(ell,ll,NlT)
powersum = C_ellT_u + N_ellT
Clphi_ar = np.interp(ell,ll,Clphi)

i_end = 102
l_last = 20000#10000.
l_beg = 0.0001#2000.#5000

'''
for i in (range(Nx)[:i_end+1]+range(Nx)[-i_end:]):
    for j in (range(Ny)[:i_end+1]+range(Ny)[-i_end:]):
        if ell == 0. or (ell > ll[-1] and ell < ll):
'''
for i in (range(Nx)[:i_end+1]+range(Nx)[-i_end:]):
    for j in range(Ny)[:i_end+1]:
        #ell = np.sqrt(lx[i]**2+ly[j]**2)
        if (ell[i,j] > l_last or ell[i,j] < l_beg):
            pass
        else:
            # gradT dot ell
            print i, j, ell[i,j]
            gprod = (gradTux.data*lx[i]+ gradTuy.data*ly[j])
           
            # weighting for this ell=lx,ly
            N_phiphi = powersum[i,j]/gprod**2
            # two ways to do weighting
            W_ell = Clphi_ar[i,j]/(N_phiphi+Clphi_ar[i,j])
            #W_ell = 1./N_phiphi
            
            norm = np.sqrt(np.mean(W_ell**2))
            norm_c = np.mean(W_ell)
            norm_c *= 1j
            norm *= 1j
            
            # estimator in real space
            # crucial step
            # subtracting unlensed
            phi_est.data = W_ell*(Tlens.data-Tlens_filt.data)/(gprod)
            # not subtracting unlensed
            #phi_est.data = W_ell*Tlens.data/(gprod)
            
            Fphi_est_ell = liteMap.fftFromLiteMap(phi_est)
            Fphi_est_unnorm.kMap[j,i] = Fphi_est_ell.kMap[j,i]/norm_c
            Fphi_est.kMap[j,i] = Fphi_est_ell.kMap[j,i]/norm
            
            # using fact that phi is imaginary, minus sign from -l(-l)
            Fphi_est.kMap[-j,-i] = np.conj(Fphi_est.kMap[j,i])
            Fphi_est_unnorm.kMap[-j,-i] = np.conj(Fphi_est_unnorm.kMap[j,i])

# needs testing            
#Fphi_est.kMap[:-i_end,::-1] = np.conj(Fphi_est.kMap[:i_end+1,:])
#Fphi_est_unnorm.kMap[:-i_end,::-1] = np.conj(Fphi_est_unnorm.kMap[:i_end+1,:])

ps2d_est = fftTools.powerFromFFT(Fphi_est)
l, l, ll, Clphi_est, l, l = ps2d_est.binInAnnuli('BIN_200_LOG')

ps2d_cross = fftTools.powerFromFFT(Fphi,Fphi_est_unnorm)
l, l, ll, Clphi_cross, l, l = ps2d_cross.binInAnnuli('BIN_200_LOG')

diff = np.diff(ll)
diff = np.append(diff,diff[-1])
yerr = np.sqrt(Clphi_est*Clphi*2)/np.sqrt((2.*ll+1.)*diff*fsky)

print "Printing ll, Cl_phi, Clphi_est, Clphi_cross, yerr..."
for i in range(len(ll)):
    print ('% 8.1f' % ll[i]), ('% 18.8e' % Clphi[i]), ('% 18.8e' % Clphi_est[i]), ('% 18.8e' % Clphi_cross[i]), ('% 18.8e' % yerr[i])

plt.figure()
plt.title("Power Spectrum Reconstruction on Small Scales")
plt.plot(ll,Clphi_est,label='estimated')
plt.plot(ll,Clphi,label='real')
plt.errorbar(ll[np.where(Clphi_cross>0.)],Clphi_cross[np.where(Clphi_cross>0.)],yerr=yerr[np.where(Clphi_cross>0.)], fmt='o',label='cross positive')
plt.errorbar(ll[np.where(Clphi_cross<0.)],-Clphi_cross[np.where(Clphi_cross<0.)],yerr=yerr[np.where(Clphi_cross<0.)], fmt='o',label='cross negative')
#plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig(savef)        
#plt.show()

plt.figure()
plt.title("Power Spectrum Ratio")
plt.errorbar(ll,Clphi_cross/Clphi,yerr=yerr/Clphi, fmt='o',label='cross/real')
#plt.errorbar(ll[np.where(Clphi_c>0.)],Clphi_c[np.where(Clphi_c>0.)]/Clphi[np.where(Clphi_c>0.)],yerr=yerr[np.where(Clphi_c>0.)]/Clphi[np.where(Clphi_c>0.)], fmt='o',label='c+')
plt.plot(ll,np.ones(len(ll)),label='ones')
#plt.errorbar(ll[np.where(Clphi_c<0.)],-Clphi_c[np.where(Clphi_c<0.)]/Clphi[np.where(Clphi_c>0.)],yerr=yerr[np.where(Clphi_c<0.)]/Clphi[np.where(Clphi_c>0.)], fmt='o',label='c-')
#plt.xscale("log")
plt.ylim([-5,5])
plt.legend()
plt.savefig(saverat)

phi_est.data[:,:] = Fphi_est.mapFromFFT()
kappa_est = phiToKappa(phi_est)
kappa_est.writeFits(savekappaest,overWrite = True)
kappa.writeFits(savekappa,overWrite = True)

plt.figure()
plt.title("Noise Power Spectrum")
plt.ylabel('$(C_l^{\hat \phi \hat \phi}-C_l^{ \phi \phi}) l^4 / 2 \pi$')
plt.xlabel('$l$')
plt.plot(ll,(Clphi_est-Clphi)*ll**4/(2*np.pi))
#plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig(savefn)        
