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
from enlib import enmap
import enlib

from mpi4py import MPI

# Mat's files
root = '/users/boryanah/repos/alhazen/'
fname_noise = root+'for_boryanka/noise_'
fname_lensed = root+'for_boryanka/lensed_'
fname_phi = root+'for_boryanka/phi_'

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

offset = 0
myrank += offset
print myrank

# as of now thethaFWHM is 0.3 and noise is 0.1 uK-arcmin
noiseFactor = .1#.9999
pwd = '/users/boryanah/Blake/bor_sim/test_noise/Mat/sims/test_'

if noiseFactor == .999: # CMB-S4
    fname = pwd+'Clphi_S4_k_'+str(myrank)+'_Mat.txt'
    thetaFWHMarcmin = 0.7

if noiseFactor == .9999: # CMB-S4 conservative
    fname = pwd+'Clphi_S4_cons_k_'+str(myrank)+'_Mat.txt'
    S4 = '_S4'
    thetaFWHMarcmin = 1.

if noiseFactor == 1.:
    fname = pwd+'Clphi_1_0_k_'+str(myrank)+'_Mat.txt'
    
    thetaFWHMarcmin = 0.3

if noiseFactor == .1:
    fname = pwd+'Clphi_0_1_k_'+str(myrank)+'_Mat.txt'
    S4 = ''
    thetaFWHMarcmin = .3

if noiseFactor == .5:
    fname = pwd+'Clphi_0_5_k_'+str(myrank)+'_Mat.txt'
    
    thetaFWHMarcmin = .3


fname_unlensed = root+'for_boryanka/unlensed_'
fname_grad_T_unlensed = root+'for_boryanka/grad_T_unlensed_'

deg_x = .5*1.7066666667
deg_y = .5*1.7066666667
pixXam = .1
pixYam = .1
template = liteMap.makeEmptyCEATemplate(deg_x, deg_y,meanRa = 180., meanDec = 0.,\
                      pixScaleXarcmin = pixXam, pixScaleYarcmin = pixYam)

# downsampled 0.1 arcmin pix and 1x1 deg^2 nope
nois = enmap.read_fits(fname_noise+str(myrank)+S4+'.fits')
lensed = enmap.read_fits(fname_lensed+str(myrank)+S4+'.fits')
grad_T_unlensed = enmap.read_fits(fname_grad_T_unlensed+str(myrank)+S4+'.fits')
# these are the not downsampled ones
unlensed = enmap.read_fits(fname_unlensed+str(myrank)+S4+'.fits')
ph = enmap.read_fits(fname_phi+str(myrank)+S4+'.fits')

noise = template.copy()
phi = template.copy()
Tulens = template.copy()
Tlens = template.copy()

noise.data = enmap.to_flipper(nois).data
phi.data = enmap.to_flipper(ph).data
Tulens.data = enmap.to_flipper(unlensed).data
Tlens.data = enmap.to_flipper(lensed).data

#area = 12.239927
area = deg_x*deg_y
totalArea = 41252.96
fsky = area/totalArea

# to take lx and ly
FT = liteMap.fftFromLiteMap(template)
Fphi = FT.copy()
Fphi = liteMap.fftFromLiteMap(phi)

# stepsizelx = 2*pi/(Nx*pi/180*pixsizexarcmin/60)
lx = FT.lx
ly = FT.ly

Nx = Fphi.Nx # lx i think goes from 0 to Ny
Ny = Fphi.Ny
stepsize_lx = 2*np.pi/(Nx*np.pi/180.*pixXam/60.)

# gradient of unlensed temperature with killed small modes
Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr>2000.] = 0.
filt[0] = 0.
# here we get more reali-STICK
Tlens_filt = Tlens.filterFromList([Larr,filt])
# not different from
#Tlens_filt = Tlens_noN.filterFromList([Larr,filt])

#FTlens_filt = liteMap.fftFromLiteMap(Tlens_filt)

gradTu = template.copy()
gradTux = template.copy()
gradTuy = template.copy()

gradTux.data = enmap.to_flipper(grad_T_unlensed[0,:,:]).data
gradTuy.data = enmap.to_flipper(grad_T_unlensed[1,:,:]).data

# rms
rms = np.sqrt(np.mean(gradTux.data**2+gradTuy.data**2))
print "rms = ",rms

# CHANGE THIS much better god knows why
gradTu = Tulens.takeGradient()
print Tulens.data.shape, Tulens.Nx
gradTux_o = gradTu.gradX
gradTuy_o = gradTu.gradY

# rms
rms = np.sqrt(np.mean(gradTux.data**2+gradTuy.data**2))
print "rms = ",rms

print "gradTux_o.shape, gradTux.shape"
print gradTux_o.data.shape, gradTux.data.shape
print "np.max((gradTux.data-gradTux_o.data)/gradTux.data)"
print np.max((gradTux.data-gradTux_o.data)/gradTux.data)
gradTux = gradTux_o
gradTuy = gradTuy_o

# unlensed temperature power spectrum ok because only used in Nphiphi
#p2d = fftTools.powerFromLiteMap(Tulens)
#l, l, ll, ClT_u, l, l = p2d.binInAnnuli('BIN_300')

# noise power spectrum
p2d = fftTools.powerFromLiteMap(noise)
l, l, ll, Cln, l, l = p2d.binInAnnuli('BIN_300')


# lensed temperature power spectrum
p2d = fftTools.powerFromLiteMap(Tlens)
l, l, ll, ClT_l, l, l = p2d.binInAnnuli('BIN_300')


# lensing potential power spectrum
ps2d = fftTools.powerFromLiteMap(phi)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_300')


# estimator in Fourier space
Fphi_est = Fphi.copy()
Fphi_est_unnorm = Fphi.copy()
Fphi_est.kMap[:,:] = 0
Fphi_est_unnorm.kMap[:,:] = 0


# estimator in real space
phi_est = phi.copy()
phi_est.data[:,:] = 0

ell = FT.modLMap

Fphi.kMap[:,:] *= ell**2/2.
ps2d = fftTools.powerFromFFT(Fphi)
l, l, ll, Clkappa, l, l = ps2d.binInAnnuli('BIN_300')


#C_ellT_u = np.interp(ell,ll,ClT_u)
# if using alternative noise with lensing
Clphi_ar = np.interp(ell,ll,Clphi)
C_ellT_l = np.interp(ell,ll,ClT_l)


#N_ellT = np.interp(ell,l_len,NlT)
N_ellT = np.interp(ell,ll,Cln)
# used to be
#N_ellT = (deltaT*thetaFWHM)**2*np.exp(ell*(ell+1.)*thetaFWHM**2/(8.*np.log(2.)))*noiseFactor**2.

powersum = C_ellT_l - ell**2*rms**2*Clphi_ar*.5


i_end = 100##102

i_end = 201##102
if stepsize_lx*i_end < 40000: 
    print "YOU ARE AN IDIOT"
    quit()
l_last = 40000#15000#39500.
l_beg = 0.0001

print 'lx[i_end-1]:'
print lx[i_end-1]

print 'lx[i_end-1]:'
print lx[i_end-1]

#W_ell = phi.copy()

# TRY TO GET RID OF RINGING
Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr<3000.] = 0.
Tlens = Tlens.filterFromList([Larr,filt])

for i in (range(Nx)[:i_end+1]+range(Nx)[-i_end:]):
    for j in range(Ny)[:i_end+1]:
        if (ell[i,j] > l_last or ell[i,j] < l_beg):
            pass
        else:
            # gradT dot ell
            gprod = (gradTux.data*lx[i]+ gradTuy.data*ly[j])
            x = powersum[i,j]/(gprod**2*Clphi_ar[i,j])
            W_ell = 1./(x+1)
            #x_inv = 1./x

            '''
            if ell[i,j] < 5500 and ell[i,j] > 5000:
                print  1./np.mean(W_ell), Clphi_ar[i,j]
                
                gradTux.data = W_ell
                gradTux.plot(show=False,saveFig = 'Wiener.png');plt.close()
                gradTuy.data = x_inv
                gradTuy.plot(show=False,saveFig = 'Inverse.png');plt.close()
                exit()
            '''

            # everything below this point is kappa
            norm = np.sqrt(np.mean(W_ell**2))
            norm_c = np.mean(W_ell)
            norm_c = norm_c.astype(complex)
            norm = norm.astype(complex)
            norm_c *= 1j
            norm *= 1j
            phi_est.data = W_ell*ell[i,j]**2*Tlens.data/(2.*gprod)
             
            Fphi_est_ell = liteMap.fftFromLiteMap(phi_est)
            Fphi_est_unnorm.kMap[j,i] = Fphi_est_ell.kMap[j,i]/norm_c
            Fphi_est.kMap[j,i] = Fphi_est_ell.kMap[j,i]/norm
            
            # using fact that phi is imaginary, minus sign from -l(-l)
            Fphi_est.kMap[-j,-i] = np.conj(Fphi_est.kMap[j,i])
            Fphi_est_unnorm.kMap[-j,-i] = np.conj(Fphi_est_unnorm.kMap[j,i])

ps2d_est = fftTools.powerFromFFT(Fphi_est)
l, l, ll, Clphi_est, l, l = ps2d_est.binInAnnuli('BIN_300')

ps2d_cross = fftTools.powerFromFFT(Fphi,Fphi_est_unnorm)
l, l, ll, Clphi_cross, l, l = ps2d_cross.binInAnnuli('BIN_300')

diff = np.diff(ll)
diff = np.append(diff,diff[-1])
yerr = np.sqrt(Clphi_est*Clphi*2)/np.sqrt((2.*ll+1.)*diff*fsky)

Nlphi = Clphi_est - Clkappa

Nl_pred = ClT_l/(rms**2*ll**2)
Nl_pred_u = ClT_l/(rms**2*ll**2)-.5*Clphi
Nl_pred_u *= ll**4/4.
Nl_pred *= ll**4/4.


# reconstructed map
phi_est.data[:,:] = Fphi_est.mapFromFFT()

Clphi *= ll**4/4.

f1 = open(fname, 'w+')
for i in range(len(ll)):
    print >> f1, ('% 8.1f' % ll[i]), ('% 18.8e' % Clphi[i]), ('% 18.8e' % Clphi_est[i]), ('% 18.8e' % Clphi_cross[i]), ('% 18.8e' % yerr[i]),('% 18.8e' % Nl_pred_u[i]), ('% 18.8e' % Nl_pred[i]), ('% 18.8e' % Nlphi[i])

'''
# parameters for noise estimation

noiseUkArcmin = 1.
thetaFWHM = thetaFWHMarcmin*numpy.pi/(180.*60.)
deltaT = noiseUkArcmin/thetaFWHMarcmin


beam1d_off = {'apply':False}

fullBeamMatrix = {'apply':False}

# CHANGE THIS
phi = liteMap.makeEmptyCEATemplate(1, 1,meanRa = 180., meanDec = 0.,\
                      pixScaleXarcmin = .1, pixScaleYarcmin = .1)

#scal = '/users/boryanah/Blake/bor_sim/test_noise/Aug6_highAcc_CDM_scalCls_new.dat'
#lens = '/users/boryanah/Blake/bor_sim/test_noise/Aug6_highAcc_CDM_lensedCls_new.dat'
scal = '/users/boryanah/Blake/bor_sim/test_noise/Mat/data/Aug6_highAcc_CDM_scalCls.dat'
lens = '/users/boryanah/Blake/bor_sim/test_noise/Mat/data/Aug6_highAcc_CDM_lensedCls.dat'

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
filt[Larr>40000.] = 0.
filt[0] = 0.
# here we get more reali-STICK
noise = noise.filterFromList([Larr,filt])

# CHANGE THIS
Tlens = Tlens_noN.copy()
Tlens.data[:,:]  +=  noise.data[:,:]

# CHANGE THIS
phi = phi_new.copy()
Tulens = T_map.copy()
# DO TUK
'''
