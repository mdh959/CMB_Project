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

from mpi4py import MPI

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

myrank += 50

noiseFactor = .9999
n_gal = 40 # per arcmin^2 arcmin = 1/60 pi/180
n_gal *= (180*60/np.pi)**2

pwd = '/users/boryanah/Blake/bor_sim/test_noise/galaxy/sims/'

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
clkappa = theoryPower[:,4]/tcmb**2/4.
clphi = theoryPower[:,4]/(lphi**4)/tcmb**2


NlT = (deltaT*thetaFWHM)**2*np.exp(l_len*(l_len+1.)*thetaFWHM**2/(8.*np.log(2.)))*noiseFactor**2.

# load galaxy spectra
Ls = np.arange(2,20001+1)
cl_kg =  np.loadtxt('kg_cl1_0.dat', unpack=True)
cl_gg = np.loadtxt('gg_noshot_cl5.dat', unpack=True)


# IMPORTANT
cl_gg = np.loadtxt('gg_cl2_0.dat', unpack=True)
cl_kk = np.interp(Ls,lphi,clkappa)

# solve for cl_RR and a(l)
a_l = cl_kg/cl_kk
cl_RR = cl_gg - a_l**2*cl_kk

# OPTION 1
phi_new = phi.copy()
phi_new.fillWithGaussianRandomField(lphi,clphi)#,bufferFactor=1)  
phi_new.data[:,:] -= np.mean(phi_new.data)

T_map,Q_map,U_map =liteMapPol.simPolMapsFromEandB(phi_new,l,cl_TT,cl_EE,cl_TE,cl_BB,fullBeamMatrix=fullBeamMatrix,beam1d=beam1d_off)

Tlens_noN, Qlens, Ulens  =  lensMaps(phi_new,T_map,Q_map,U_map)

# noise ps
noise = T_map.copy()
noise.fillWithGaussianRandomField(l_len,NlT)#,bufferFactor=1)  

# TRY as BLAKE SAID
Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr>30000.] = 0.
filt[0] = 0.
# here we get more reali-STICK
noise = noise.filterFromList([Larr,filt])

Tlens = Tlens_noN.copy()
Tlens.data[:,:]  +=  noise.data[:,:]

#area = 12.239927
area = 1.
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
ell = Fphi.modLMap

# producing kappa
Fkappa = Fphi.copy()
Fkappa.kMap[:,:] *= ell**2/2.
kappa = phi.copy()
kappa.data[:,:] = Fkappa.mapFromFFT(setMeanToZero=True)

# simulate galaxy field
G = phi.copy()
G.fillWithGaussianRandomField(Ls,np.abs(cl_RR))
FG = liteMap.fftFromLiteMap(G)
a_ell = np.interp(ell, Ls, a_l)
FG.kMap[:,:] += a_ell*Fkappa.kMap
G.data[:,:] = FG.mapFromFFT(setMeanToZero=True)

# gradient of unlensed temperature with killed small modes
Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr>2000.] = 0.
filt[0] = 0.
# here we get more reali-STICK
Tlens_filt = Tlens.filterFromList([Larr,filt])
# not different from
#Tlens_filt = Tlens_noN.filterFromList([Larr,filt])
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

'''
p2d = fftTools.powerFromLiteMap(Tlens_noN)
l, l, ll, ClT_l_noN, l, l = p2d.binInAnnuli('BIN_200_LOG')

a = ClT_l
b = ClT_l_noN+Cln
c = a-b
for i in range(45):
    print a[i], b[i],c[i]
quit()
'''

# lensing potential power spectrum
ps2d = fftTools.powerFromFFT(Fphi)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_200_LOG')

ps2d = fftTools.powerFromFFT(Fkappa)
l, l, ll, Clkappa, l, l = ps2d.binInAnnuli('BIN_200_LOG')

# estimator in Fourier space
Fphi_est = Fphi.copy()
Fphi_est_unnorm = Fphi.copy()
Fphi_est.kMap[:,:] = 0
Fphi_est_unnorm.kMap[:,:] = 0

# estimator in real space
phi_est = phi.copy()
phi_est.data[:,:] = 0

C_ellT_u = np.interp(ell,ll,ClT_u)
# if using alternative noise with lensing
Clphi_ar = np.interp(ell,ll,Clphi)
C_ellT_l = np.interp(ell,ll,ClT_l)


#N_ellT = np.interp(ell,l_len,NlT)
#thetaFWHMarcmin = 1. no difference
N_ellT = (deltaT*thetaFWHM)**2*np.exp(ell*(ell+1.)*thetaFWHM**2/(8.*np.log(2.)))*noiseFactor**2.
# try tuka
#powersum = C_ellT_u + N_ellT
# alternative
powersum = C_ellT_l - .5*ell**2*rms**2*Clphi_ar

'''
# TRY as BLAKE SAID
Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr>10000.] = 0.
filt[0] = 0.
# here we get more reali-STICK
Tlens = Tlens.filterFromList([Larr,filt])
Tlens_filt = Tlens.filterFromList([Larr,filt])
'''

# TRY TO GET RID OF RINGING
Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr<4000.] = 0.
Tlens = Tlens.filterFromList([Larr,filt])


i_end = 62#102
l_last = 20000.
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
            #N_phiphi = powersum[i,j]/gprod**2
            # two ways to do weighting
            #W_ell = Clphi_ar[i,j]/(N_phiphi+Clphi_ar[i,j])
            #W_ell = 1./N_phiphi
            x = powersum[i,j]/(gprod**2*Clphi_ar[i,j])
            W_ell = 1./(x+1)

            norm = np.sqrt(np.mean(W_ell**2))
            norm_c = np.mean(W_ell)
            norm_c *= 1j
            norm *= 1j
            
            # estimator in real space
            # crucial step
            # subtracting unlensed
            #phi_est.data[:,:] = W_ell*(Tlens.data-Tlens_filt.data)/(gprod)
            # not subtracting unlensed
            phi_est.data = W_ell*ell[i,j]**2*Tlens.data/(2.*gprod)
            
            Fphi_est_ell = liteMap.fftFromLiteMap(phi_est)
            Fphi_est_unnorm.kMap[j,i] = Fphi_est_ell.kMap[j,i]/norm_c
            Fphi_est.kMap[j,i] = Fphi_est_ell.kMap[j,i]/norm
            
            # using fact that phi is imaginary, minus sign from -l(-l)
            Fphi_est.kMap[-j,-i] = np.conj(Fphi_est.kMap[j,i])
            Fphi_est_unnorm.kMap[-j,-i] = np.conj(Fphi_est_unnorm.kMap[j,i])

ps2d_est = fftTools.powerFromFFT(Fphi_est)
l, l, ll, Clphi_est, l, l = ps2d_est.binInAnnuli('BIN_200_LOG')

ps2d_cross = fftTools.powerFromFFT(Fkappa,Fphi_est_unnorm)
l, l, ll, Clphi_cross, l, l = ps2d_cross.binInAnnuli('BIN_200_LOG')

ps2d_cross = fftTools.powerFromFFT(FG,Fphi_est_unnorm)
l, l, ll, Cl_kg, l, l = ps2d_cross.binInAnnuli('BIN_200_LOG')

diff = np.diff(ll)
diff = np.append(diff,diff[-1])
yerr = np.sqrt(Clphi_est*Clphi*2)/np.sqrt((2.*ll+1.)*diff*fsky)

Nlphi = Clphi_est - Clkappa

Nl_pred = ClT_l/(rms**2*ll**2)
Nl_pred_u = (ClT_u+Cln)/(rms**2*ll**2)
# yes, this is better
Nl_pred_u = (ClT_l/(rms**2*ll**2)-.5*Clphi)*ll**4/4.

f1 = open(fname, 'w+')
for i in range(len(ll)):
    print >> f1, ('% 8.1f' % ll[i]), ('% 18.8e' % Clphi[i]), ('% 18.8e' % Clphi_est[i]), ('% 18.8e' % Clphi_cross[i]), ('% 18.8e' % Cl_kg[i]),('% 18.8e' % Nl_pred_u[i]), ('% 18.8e' % Nl_pred[i]), ('% 18.8e' % Nlphi[i]) 
