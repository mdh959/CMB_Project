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
myrank = 30+comm.Get_rank()
nproc = comm.Get_size()

nametu = '/users/boryanah/Blake/bor_sim/many/sims/T_map_'+str(myrank)+'.fits'
nametl = '/users/boryanah/Blake/bor_sim/many/sims/Tlens_'+str(myrank)+'.fits'
nametln = '/users/boryanah/Blake/bor_sim/many/sims/Tlens_noise_0_1_'+str(myrank)+'.fits'
namepn = '/users/boryanah/Blake/bor_sim/many/sims/phi_new_'+str(myrank)+'.fits'
'''

noiseFactor = .999

if noiseFactor == .999: # CMB-S4
    nametu = '/users/boryanah/Blake/bor_sim/test_noise/T_map_S4.fits'
    nametl = '/users/boryanah/Blake/bor_sim/test_noise/Tlens_S4.fits'
    nametln = '/users/boryanah/Blake/bor_sim/test_noise/Tlens_noise_S4.fits'
    namen = '/users/boryanah/Blake/bor_sim/test_noise/noise_S4.fits'
    namepn = '/users/boryanah/Blake/bor_sim/test_noise/phi_new_S4.fits'
    fname = 'Clphi_S4_W.txt'
    print noiseFactor
    thetaFWHMarcmin = 0.7

if noiseFactor == .9999: # CMB-S4 conservative
    nametu = '/users/boryanah/Blake/bor_sim/test_noise/T_map_S4_cons.fits'
    nametl = '/users/boryanah/Blake/bor_sim/test_noise/Tlens_S4_cons.fits'
    nametln = '/users/boryanah/Blake/bor_sim/test_noise/Tlens_noise_S4_cons.fits'
    namen = '/users/boryanah/Blake/bor_sim/test_noise/noise_S4_cons.fits'
    namepn = '/users/boryanah/Blake/bor_sim/test_noise/phi_new_S4_cons.fits'
    fname = 'Clphi_S4_cons_W.txt'
    print noiseFactor
    thetaFWHMarcmin = 1.

if noiseFactor == 1.:
    nametu = '/users/boryanah/Blake/bor_sim/test_noise/T_map_1_0.fits'
    nametl = '/users/boryanah/Blake/bor_sim/test_noise/Tlens_1_0.fits'
    nametln = '/users/boryanah/Blake/bor_sim/test_noise/Tlens_noise_1_0.fits'
    namen = '/users/boryanah/Blake/bor_sim/test_noise/noise_1_0.fits'
    namepn = '/users/boryanah/Blake/bor_sim/test_noise/phi_new_1_0.fits'
    fname = 'Clphi_1_0_W.txt'
    print noiseFactor
    thetaFWHMarcmin = 0.3

if noiseFactor == .1:
    nametu = '/users/boryanah/Blake/bor_sim/test_noise/T_map_0_1.fits'
    nametl = '/users/boryanah/Blake/bor_sim/test_noise/Tlens_0_1.fits'
    nametln = '/users/boryanah/Blake/bor_sim/test_noise/Tlens_noise_0_1.fits'
    namen = '/users/boryanah/Blake/bor_sim/test_noise/noise_0_1.fits'
    namepn = '/users/boryanah/Blake/bor_sim/test_noise/phi_new_0_1.fits'
    fname = 'Clphi_0_1_W.txt'
    print noiseFactor
    thetaFWHMarcmin = 0.3

if noiseFactor == .01:
    nametu = '/users/boryanah/Blake/bor_sim/test_noise/T_map_0_01.fits'
    nametl = '/users/boryanah/Blake/bor_sim/test_noise/Tlens_0_01.fits'
    nametln = '/users/boryanah/Blake/bor_sim/test_noise/Tlens_noise_0_01.fits'
    namen = '/users/boryanah/Blake/bor_sim/test_noise/noise_0_01.fits'
    namepn = '/users/boryanah/Blake/bor_sim/test_noise/phi_new_0_01.fits'
    fname = 'Clphi_0_01_W.txt'
    print noiseFactor
    thetaFWHMarcmin = 0.3

# parameters for noise estimation

noiseUkArcmin = 1.
thetaFWHM = thetaFWHMarcmin*numpy.pi/(180.*60.)
deltaT = noiseUkArcmin/thetaFWHMarcmin


beam1d_off = {'apply':False}

fullBeamMatrix = {'apply':False}
# load maps                                          
dirphi = '/users/boryanah/Blake/bor_sim/test_noise/'
namep = 'phi.fits'
#phi = liteMap.liteMapFromFits(dirphi+namep)

#phi = liteMap.makeEmptyCEATemplate(3.5, 3.5,meanRa = 180., meanDec = 0.,\
#                      pixScaleXarcmin = 0.41015625, pixScaleYarcmin = 0.41015625)

# IF WE WANNA CREATE MAP
# pixsizexarcmin = sizexindeg/(180)*pi/Nx *180*60/pi
phi = liteMap.makeEmptyCEATemplate(1, 1,meanRa = 180., meanDec = 0.,\
                      pixScaleXarcmin = 15./64, pixScaleYarcmin = 15./64)
'''
# cutting maps  
x0 = 180.5 #181.25                                      
y0 = -0.5#-1.
x1 = 179.5#178.75 #178.75                                   
y1 = 0.5#0.75
phi = phi.selectSubMap(x0,x1,y0,y1)
'''

scal = 'Aug6_highAcc_CDM_scalCls_new.dat'
lens = 'Aug6_highAcc_CDM_lensedCls_new.dat'


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

#for unlensed maps with lensed power                                                                                    
theoryPower_lensed = np.loadtxt(lens)
l_len=theoryPower_lensed[:,0]
cl_TT_len=theoryPower_lensed[:,1]
cl_EE_len=theoryPower_lensed[:,2]
cl_TE_len=theoryPower_lensed[:,3]
cl_BB=None

#lMax = 9000 if 9000 < max(l_len) else max(l_len)

l_len=l_len[:lMax]
cl_TT_len=cl_TT_len[:lMax]*2*np.pi/(l_len*(l_len+1))
cl_EE_len=cl_EE_len[:lMax]*2*np.pi/(l_len*(l_len+1))
cl_TE_len=cl_TE_len[:lMax]*2*np.pi/(l_len*(l_len+1))

tcmb = 2.7255e6
lphi = theoryPower[:,0]
clphi = theoryPower[:,4]/(lphi**4)/tcmb**2

NlT = (deltaT*thetaFWHM)**2*np.exp(l_len*(l_len+1.)*thetaFWHM**2/(8.*np.log(2.)))*noiseFactor**2.

'''
# OPTION 1
phi_new = phi.copy()
phi_new.fillWithGaussianRandomField(lphi,clphi)#,bufferFactor=1)  
phi_new.data[:,:] -= np.mean(phi_new.data)

T_map,Q_map,U_map =liteMapPol.simPolMapsFromEandB(phi_new,l,cl_TT,cl_EE,cl_TE,cl_BB,fullBeamMatrix=fullBeamMatrix,beam1d=beam1d_off)

T_map.writeFits(nametu, overWrite=True)

phi_new.writeFits(namepn, overWrite=True)

Tlens, Qlens, Ulens  =  lensMaps(phi_new,T_map,Q_map,U_map)
Tlens.writeFits(nametl, overWrite=True)  

# noise ps
noise = T_map.copy()
noise.fillWithGaussianRandomField(l_len,NlT)#,bufferFactor=1)  

Tlens.data[:,:]  +=  noise.data[:,:]
Tlens.writeFits(nametln, overWrite=True)  
noise.writeFits(namen, overWrite=True)  
'''

# OPTION 2
Tlens = liteMap.liteMapFromFits(nametln)
T_map = liteMap.liteMapFromFits(nametu)
phi_new = liteMap.liteMapFromFits(namepn)
noise = liteMap.liteMapFromFits(namen)

#area = 12.239927
area = 3.056636
totalArea = 41252.96
fsky = area/totalArea

phi = phi_new.copy()
Tulens = T_map.copy()

# to take lx and ly
Fphi = liteMap.fftFromLiteMap(phi)

# stepsizelx = 2*pi/(Nx*pi/180*pixsizexarcmin/60)
lx = Fphi.lx
ly = Fphi.ly

#print lx[:60]
#print ly[:60]

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

# rms
rms = np.sqrt(np.mean(gradTux.data**2+gradTuy.data**2))
print "rms = ",rms

# unlensed temperature power spectrum ok because only used in Nphiphi
p2d = fftTools.powerFromLiteMap(Tulens)
l, l, ll, ClT_u, l, l = p2d.binInAnnuli('BIN_200_LOG')

# noise power spectrum
p2d = fftTools.powerFromLiteMap(noise)
l, l, ll, Cln, l, l = p2d.binInAnnuli('BIN_200_LOG')

# lensed temperature power spectrum
p2d = fftTools.powerFromLiteMap(Tlens)
l, l, ll, ClT_l, l, l = p2d.binInAnnuli('BIN_200_LOG')

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

ell = Fphi.modLMap

C_ellT_u = np.interp(ell,ll,ClT_u)
# if using alternative
#C_ellT_l = np.interp(ell,ll,ClT_l)
N_ellT = np.interp(ell,l_len,NlT)
powersum = C_ellT_u + N_ellT
# alternative
#powersum = C_ellT_l 
Clphi_ar = np.interp(ell,ll,Clphi)

i_end = 60#102
l_last = 21000#10000.
l_beg = 0.0001#2000.#5000

for i in (range(Nx)[:i_end+1]+range(Nx)[-i_end:]):
    for j in range(Ny)[:i_end+1]:
        if (ell[i,j] > l_last or ell[i,j] < l_beg):
            pass
        else:
            # gradT dot ell
            print i, j

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
            phi_est.data[:,:] = W_ell*(Tlens.data-Tlens_filt.data)/(gprod)
            # not subtracting unlensed
            #phi_est.data = W_ell*Tlens.data/(gprod)
            
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

# reconstructed map
phi_est.data[:,:] = Fphi_est.mapFromFFT()
phi_est.writeFits('phi_est.fits',overWrite=True)

# error of plots but like who cares
diff = np.diff(ll)
diff = np.append(diff,diff[-1])
yerr = np.sqrt(Clphi_est*Clphi*2)/np.sqrt((2.*ll+1.)*diff*fsky)

diff = Tlens.copy()
diff.data[:,:] = phi_est.data - phi.data
p2d = fftTools.powerFromLiteMap(diff)
l, l, ll, Nlphi, l, l = p2d.binInAnnuli('BIN_200_LOG')
#Nlphi = Clphi_est - Clphi
Nl_pred = ClT_l/(rms**2*ll**2)
Nl_pred_u = (ClT_u+Cln)/(rms**2*ll**2)

f1 = open(fname, 'w+')
for i in range(len(ll)):
    print >> f1, ('% 8.1f' % ll[i]), ('% 18.8e' % Clphi[i]), ('% 18.8e' % Clphi_est[i]), ('% 18.8e' % Clphi_cross[i]), ('% 18.8e' % yerr[i]),('% 18.8e' % Nl_pred_u[i]), ('% 18.8e' % Nl_pred[i]), ('% 18.8e' % Nlphi[i])



# For plotting purposes
Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr<5000.] = 0.
filt[Larr>20000.] = 0.
phi = phi.filterFromList([Larr,filt])
phi_est = phi_est.filterFromList([Larr,filt])

if noiseFactor == .01:
    phi_est.plot(show=False,saveFig = 'phi_est_0_01_W.png',valueRange=[np.min(phi.data),np.max(phi.data)]);plt.close()
    phi.plot(show=False,saveFig = 'phi_real_0_01_W.png');plt.close()

if noiseFactor == .1:
    phi_est.plot(show=False,saveFig = 'phi_est_0_1_W.png',valueRange=[np.min(phi.data),np.max(phi.data)]);plt.close()
    phi.plot(show=False,saveFig = 'phi_real_0_1_W.png');plt.close()

if noiseFactor == 1.:
    phi_est.plot(show=False,saveFig = 'phi_est_1_0_W.png',valueRange=[np.min(phi.data),np.max(phi.data)]);plt.close()
    phi.plot(show=False,saveFig = 'phi_real_1_0_W.png');plt.close()

if noiseFactor == .999:
    phi_est.plot(show=False,saveFig = 'phi_est_S4_W.png',valueRange=[np.min(phi.data),np.max(phi.data)]);plt.close()
    phi.plot(show=False,saveFig = 'phi_real_S4_W.png');plt.close()

if noiseFactor == .9999:
    phi_est.plot(show=False,saveFig = 'phi_est_S4_cons_W.png',valueRange=[np.min(phi.data),np.max(phi.data)]);plt.close()
    phi.plot(show=False,saveFig = 'phi_real_S4_cons_W.png');plt.close()

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
kappa_est.writeFits(savekappaest,overWrite = True)
kappa.writeFits(savekappa,overWrite = True)

