import matplotlib
matplotlib.use('Agg')                                                                                                
import numpy as np
import matplotlib.pyplot as plt
import astLib.astWCS
import astLib.astCoords
import astLib.astCalc
from flipper import *
from flipperPol import *
from lensingTools import *
from numpy.fft import fftshift,fftfreq,fft2,ifft2

'''
from mpi4py import MPI

comm = MPI.COMM_WORLD
myrank = 10+comm.Get_rank()
nproc = comm.Get_size()
'''

myrank = 'Pol'

noiseFactor = .01#999

pwd = '/users/boryanah/Blake/bor_sim/test_noise/polarization/'

if noiseFactor == .999: # CMB-S4
    fname = pwd+'Clphi_S4_N_'+str(myrank)+'.txt'
    thetaFWHMarcmin = 0.7

if noiseFactor == .9999: # CMB-S4 conservative
    fname = pwd+'Clphi_S4_cons_N_'+str(myrank)+'.txt'
    
    thetaFWHMarcmin = 1.

if noiseFactor == 1.:
    fname = pwd+'Clphi_1_0_N_'+str(myrank)+'.txt'
    
    thetaFWHMarcmin = 0.3

if noiseFactor == .1:
    fname = pwd+'Clphi_0_1_N_'+str(myrank)+'.txt'
    
    thetaFWHMarcmin = 0.3

if noiseFactor == .01:
    fname = pwd+'Clphi_0_01_N_'+str(myrank)+'.txt'
    
    thetaFWHMarcmin = 0.3

# parameters for noise estimation
noiseUkArcmin = 1.
thetaFWHM = thetaFWHMarcmin*numpy.pi/(180.*60.)
deltaT = noiseUkArcmin/thetaFWHMarcmin
beam1d_off = {'apply':False}
fullBeamMatrix = {'apply':False}

# CREATION
phi = liteMap.makeEmptyCEATemplate(1, 1,meanRa = 180., meanDec = 0.,\
                      pixScaleXarcmin = 15./64, pixScaleYarcmin = 15./64)

scal = 'Aug6_highAcc_CDM_scalCls_new.dat'
lens = 'Aug6_highAcc_CDM_lensedCls_new.dat'


theoryPower = np.loadtxt(scal)
l=theoryPower[:,0]
cl_TT=theoryPower[:,1]
cl_EE=theoryPower[:,2]
cl_TE=theoryPower[:,3]
cl_BB=None

#for unlensed maps with lensed power                                                                                    
theoryPower_lensed = np.loadtxt(lens)
l_len=theoryPower_lensed[:,0]
cl_TT_len=theoryPower_lensed[:,1]
cl_EE_len=theoryPower_lensed[:,2]
cl_TE_len=theoryPower_lensed[:,3]
cl_BB=None

lMax = 7000#40000 if 40000 < max(l_len) else int(max(l_len))-1

l=l[:lMax]
cl_TT=cl_TT[:lMax]*2*numpy.pi/(l*(l+1))
cl_EE=cl_EE[:lMax]*2*numpy.pi/(l*(l+1))
cl_TE=cl_TE[:lMax]*2*numpy.pi/(l*(l+1))


l_len=l_len[:lMax]
cl_TT_len=cl_TT_len[:lMax]*2*np.pi/(l_len*(l_len+1))
cl_EE_len=cl_EE_len[:lMax]*2*np.pi/(l_len*(l_len+1))
cl_TE_len=cl_TE_len[:lMax]*2*np.pi/(l_len*(l_len+1))


tcmb = 2.7255e6
lphi = theoryPower[:,0]
clphi = theoryPower[:,4]/(lphi**4)/tcmb**2

NlT = (deltaT*thetaFWHM)**2*np.exp(l_len*(l_len+1.)*thetaFWHM**2/(8.*np.log(2.)))*noiseFactor**2.

# OPTION 1
phi_new = phi.copy()
phi_new.fillWithGaussianRandomField(lphi,clphi)#,bufferFactor=1)  
phi_new.data[:,:] -= np.mean(phi_new.data)




# make T, E and B maps
Tulens = phi.copy()
Tulens.fillWithGaussianRandomField(l,cl_TT)
Eulens = phi.copy()
Eulens.fillWithGaussianRandomField(l,cl_EE)
Bulens = phi.copy()
Bulens.data[:,:] = 0.


Tlens, Elens, Blens  =  lensMaps(phi_new,Tulens,Eulens,Bulens)

# alternatively
#T_map,Q_map,U_map =liteMapPol.simPolMapsFromEandB(phi_new,l,cl_TT,cl_EE,cl_TE,cl_BB,fullBeamMatrix=fullBeamMatrix,beam1d=beam1d_off)
#Tlens, Qlens, Ulens  =  lensMaps(phi_new,T_map,Q_map,U_map)

# noise ps
noiseE = Tulens.copy()
noiseB = Tulens.copy()
noiseE.fillWithGaussianRandomField(l_len,NlT*2)#,bufferFactor=1)  
noiseB.fillWithGaussianRandomField(l_len,NlT*2)#,bufferFactor=1)  

# TRY as BLAKE SAID
Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr>30000.] = 0.
filt[0] = 0.
# here we get more reali-STICK
noiseE = noiseE.filterFromList([Larr,filt])
noiseB = noiseB.filterFromList([Larr,filt])



# NEW STUFF
Elens.data[:,:]  +=  noiseE.data[:,:]
Blens.data[:,:]  +=  noiseB.data[:,:]

area = 1.
totalArea = 41252.96
fsky = area/totalArea

# stupid renaming
phi = phi_new.copy()

# to use below
Fphi = liteMap.fftFromLiteMap(phi)
# stepsizelx = 2*pi/(Nx*pi/180*pixsizexarcmin/60)
lx = Fphi.lx
ly = Fphi.ly
Nx = Fphi.Nx 
Ny = Fphi.Ny
ell = Fphi.modLMap
#angLMap = Fphi.angLMap


# gradient of unlensed temperature with killed small modes
Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr>2000.] = 0.
filt[0] = 0.
# here we get more reali-STICK
Elens_filt = Elens.filterFromList([Larr,filt])
Blens_filt = Blens.filterFromList([Larr,filt])
# not different from
#Tlens_filt = Tlens_noN.filterFromList([Larr,filt])
gradEu = Elens_filt.takeGradient()
gradEux = gradEu.gradX
gradEuy = gradEu.gradY
gradBu = Blens_filt.takeGradient()
gradBux = gradBu.gradX
gradBuy = gradBu.gradY

# rms
rms_E = np.sqrt(np.mean(gradEux.data**2+gradEuy.data**2))
rms_B = np.sqrt(np.mean(gradBux.data**2+gradBuy.data**2))
print "rms_E = ",rms_E
print "rms_B = ",rms_B


'''
# unlensed temperature power spectrum ok because only used in Nphiphi
p2d = fftTools.powerFromLiteMap(Eulens)
l, l, ll, ClE_u, l, l = p2d.binInAnnuli('BIN_200_LOG')
'''

# noise power spectrum
p2d = fftTools.powerFromLiteMap(noiseE)
l, l, ll, ClnE, l, l = p2d.binInAnnuli('BIN_200_LOG')

p2d = fftTools.powerFromLiteMap(noiseB)
l, l, ll, ClnB, l, l = p2d.binInAnnuli('BIN_200_LOG')

'''
# lensed power spectra
p2d = fftTools.powerFromLiteMap(Elens)
l, l, ll, ClE_l, l, l = p2d.binInAnnuli('BIN_200_LOG')
p2d = fftTools.powerFromLiteMap(Blens)
l, l, ll, ClB_l, l, l = p2d.binInAnnuli('BIN_200_LOG')
'''

# lensing potential power spectrum
ps2d = fftTools.powerFromFFT(Fphi)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_200_LOG')

# estimator in Fourier space
Fphi_est = Fphi.copy()
Fphi_est_unnorm = Fphi.copy()
Fphi_est.kMap[:,:] = 0
Fphi_est_unnorm.kMap[:,:] = 0

# estimator in real space
phi_est = phi.copy()
phi_est.data[:,:] = 0

C_ellE_u = np.interp(ell,l_len,cl_EE)
# if using alternative noise with lensing
#C_ellT_l = np.interp(ell,ll,ClT_l)

#N_ellT = np.interp(ell,l_len,NlT)
#thetaFWHMarcmin = 1. no difference
N_ellE = 2*(deltaT*thetaFWHM)**2*np.exp(ell*(ell+1.)*thetaFWHM**2/(8.*np.log(2.)))*noiseFactor**2.
N_ellB = 2*(deltaT*thetaFWHM)**2*np.exp(ell*(ell+1.)*thetaFWHM**2/(8.*np.log(2.)))*noiseFactor**2.
powersum_E = C_ellE_u + N_ellE
powersum_B =  N_ellB
# alternative
#powersum = C_ellT_l 
Clphi_ar = np.interp(ell,ll,Clphi)

i_end = 60#102
l_last = lMax#20000.
l_beg = 0.0001#2000.#5000

for i in (range(Nx)[:i_end+1]+range(Nx)[-i_end:]):
    for j in range(Ny)[:i_end+1]:
        if (ell[i,j] > l_last or ell[i,j] < l_beg):
            pass
        else:
            print i, j
            # gradT dot ell
            gprod_E = (gradEux.data*lx[i]+ gradEuy.data*ly[j])
           
            # weighting for this ell=lx,ly
            N_phiphi_E = powersum_E[i,j]/gprod_E**2
            # two ways to do weighting
            #W_ell = Clphi_ar[i,j]/(N_phiphi+Clphi_ar[i,j])
            W_ell_E = 1./N_phiphi_E
            
            norm = np.sqrt(np.mean(W_ell_E**2))
            norm_c = np.mean(W_ell_E)
            norm_c *= 1j
            norm *= 1j
            
            # estimator in real space
            # crucial step
            # subtracting unlensed
            #phi_est.data[:,:] = W_ell*(Tlens.data-Tlens_filt.data)/(gprod)
            # not subtracting unlensed
            phi_est.data = W_ell_E*Elens.data/(gprod_E)
            
            Fphi_est_ell = liteMap.fftFromLiteMap(phi_est)
            Fphi_est_unnorm.kMap[j,i] = Fphi_est_ell.kMap[j,i]/norm_c
            Fphi_est.kMap[j,i] = Fphi_est_ell.kMap[j,i]/norm
            
            # using fact that phi is imaginary, minus sign from -l(-l)
            Fphi_est.kMap[-j,-i] = np.conj(Fphi_est.kMap[j,i])
            Fphi_est_unnorm.kMap[-j,-i] = np.conj(Fphi_est_unnorm.kMap[j,i])

ps2d_est = fftTools.powerFromFFT(Fphi_est)
l, l, ll, Clphi_est, l, l = ps2d_est.binInAnnuli('BIN_200_LOG')

ps2d_cross = fftTools.powerFromFFT(Fphi,Fphi_est_unnorm)
l, l, ll, Clphi_cross, l, l = ps2d_cross.binInAnnuli('BIN_200_LOG')

diff = np.diff(ll)
diff = np.append(diff,diff[-1])
yerr = np.sqrt(Clphi_est*Clphi*2)/np.sqrt((2.*ll+1.)*diff*fsky)

Nlphi = Clphi_est - Clphi
ClE_u = np.interp(ll,l_len,cl_EE)
Nl_pred_u = (ClE_u+ClnE)/((rms_E**2+rms_B**2)*ll**2)

f1 = open(fname, 'w+')
for i in range(len(ll)):
    print >> f1, ('% 8.1f' % ll[i]), ('% 18.8e' % Clphi[i]), ('% 18.8e' % Clphi_est[i]), ('% 18.8e' % Clphi_cross[i]), ('% 18.8e' % yerr[i]),('% 18.8e' % Nl_pred_u[i]), ('% 18.8e' % Nlphi[i])


# reconstructed map
phi_est.data[:,:] = Fphi_est.mapFromFFT()

# For plotting purposes
Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr<5000.] = 0.
filt[Larr>20000.] = 0.
phi = phi.filterFromList([Larr,filt])
phi_est = phi_est.filterFromList([Larr,filt])
Elens = Elens.filterFromList([Larr,filt])
#Tlens_noN = Tlens_noN.filterFromList([Larr,filt])

if noiseFactor == .01:
    phi_est.plot(show=False,saveFig = 'phi_est_0_01_'+str(myrank)+'.png',valueRange=[np.min(phi.data),np.max(phi.data)]);plt.close()
    phi.plot(show=False,saveFig = 'phi_real_0_01_'+str(myrank)+'.png');plt.close()

if noiseFactor == .1:
    phi_est.plot(show=False,saveFig = 'phi_est_0_1_'+str(myrank)+'.png',valueRange=[np.min(phi.data),np.max(phi.data)]);plt.close()
    phi.plot(show=False,saveFig = 'phi_real_0_1_'+str(myrank)+'.png');plt.close()

if noiseFactor == 1.:
    phi_est.plot(show=False,saveFig = 'phi_est_1_0_'+str(myrank)+'.png',valueRange=[np.min(phi.data),np.max(phi.data)]);plt.close()
    phi.plot(show=False,saveFig = 'phi_real_1_0_'+str(myrank)+'.png');plt.close()

if noiseFactor == .999:
    phi_est.plot(show=False,saveFig = 'phi_est_S4_'+str(myrank)+'.png',valueRange=[np.min(phi.data),np.max(phi.data)]);plt.close()
    phi.plot(show=False,saveFig = 'phi_real_S4_'+str(myrank)+'.png');plt.close()

if noiseFactor == .9999:
    phi_est.plot(show=False,saveFig = 'phi_est_S4_cons_'+str(myrank)+'.png',valueRange=[np.min(phi.data),np.max(phi.data)]);plt.close()
    phi.plot(show=False,saveFig = 'phi_real_S4_cons_'+str(myrank)+'.png');plt.close()

#Tlens_noN.plot(show=False,saveFig = 'Tlens_noN.png');plt.close()
#Elens.plot(show=False,saveFig = 'Elens.png',valueRange=[np.min(Tlens_noN.data),np.max(Tlens_noN.data)]);plt.close()

quit()
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
