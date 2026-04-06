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

#from mpi4py import MPI

# Mat's files
root = '/users/boryanah/repos/alhazen_old/'
fname_lensed = root+'for_boryanka/lensed_'
fname_phi = root+'for_boryanka/phi_'
fname_kappa_rec = root+'for_boryanka/kappa_recon_'

#comm = MPI.COMM_WORLD
#myrank = comm.Get_rank()
#nproc = comm.Get_size()
myrank = 0

offset = 100#96#64# 32
dl = ''#2000#5000 #''
myrank += offset
print myrank

# as of now thethaFWHM is 0.3 and noise is 0.1 uK-arcmin
noiseFactor = .1#6.#.1#.9999
pwd = '/users/boryanah/Blake/bor_sim/test_noise/Mat/sims/test_'

if noiseFactor == .999: # CMB-S4
    fname = pwd+'Clphi_S4_k_'+str(myrank)+'_Mat_big.txt'
    thetaFWHMarcmin = 0.7

if noiseFactor == .9999: # CMB-S4 conservative
    fname = pwd+'Clphi_S4_cons_k_'+str(myrank)+'_Mat'+str(dl)+'_big.txt'
    S4 = '_S4'
    SO = '_S4'
    thetaFWHMarcmin = 1.
    fkappa = 'kappa_S4_cons_big.fits'
    fkappa_est = 'kappa_est_S4_cons_big.fits'
    l_last = 12000
    l_ring = 3000
    l_grad = 2000
    if dl == '': dl = 1000
    if myrank > 50: myrank += 128

if noiseFactor == 1.:
    fname = pwd+'Clphi_1_0_k_'+str(myrank)+'_Mat_big.txt'
    
    thetaFWHMarcmin = 0.3

if noiseFactor == .1:
    fname = 'Clphi_0_1_k_'+str(myrank)+'_Mat'+str(dl)+'_big.txt'
    S4 = ''
    SO = ''
    thetaFWHMarcmin = .3
    #fkappa = 'kappa_0_1_big.fits'
    fkappa = 'maps/kappa_0_1_big.fits'
    #fkappa_est = 'kappa_est_0_1_big.fits'
    fkappa_est = 'maps/kappa_est_0_1_big.fits'
    #fkappa_rec = 'kappa_rec_0_1_big.fits'
    fkappa_rec = 'regular_input_kappa_recon_qe_flux_4.fits'
    #fkappa_est_unnorm = 'kappa_est_unnorm_0_1_big.fits'
    fkappa_est_unnorm = 'maps/kappa_est_unnorm_0_1_big.fits'
    l_last = 31000#21000
    l_ring = 3000
    l_grad = 2000
    if dl == '': dl = 1000

if noiseFactor == 6.:
    fname = pwd+'Clphi_SO_k_'+str(myrank)+'_Mat'+str(dl)+'_big.txt'
    S4 = '_SO'
    SO = '_SO'
    thetaFWHMarcmin = 1.4
    fkappa = 'kappa_SO_big.fits'
    fkappa_est = 'kappa_est_SO_big.fits'
    l_last = 10000
    l_ring = 3000
    l_grad = 2000
    if dl == '': dl = 1000

deg_x = 1.7066666667
deg_y = 1.7066666667
area = '2.912603855229585'

pixXam = .1
pixYam = .1
template = liteMap.makeEmptyCEATemplate(deg_x, deg_y,meanRa = 180., meanDec = 0.,\
                      pixScaleXarcmin = pixXam, pixScaleYarcmin = pixYam)

# downsampled 0.1 arcmin pix and 1x1 deg^2 nope
lensed = enmap.read_fits(fname_lensed+str(myrank)+S4+'_'+str(area)+'.fits')
ph = enmap.read_fits(fname_phi+str(myrank)+S4+'_'+str(area)+'.fits')
#k_rec = enmap.read_fits(fname_kappa_rec+str(myrank)+S4+'_'+str(area)+'.fits')
k_rec = enmap.read_fits(fkappa_rec)
k_tru = enmap.read_fits(fkappa)
k_reg = enmap.read_fits('regular_input_kappa.fits')
k_GI_unn = enmap.read_fits(fkappa_est_unnorm)
k_GI = enmap.read_fits(fkappa_est)

phi = template.copy()
Tlens = template.copy()
kappa_rec = template.copy()
kappa_tru = template.copy()
kappa_reg = template.copy()
kappa_GI = template.copy()
kappa_GI_unnorm = template.copy()

phi.data = enmap.to_flipper(ph).data
kappa_rec.data = enmap.to_flipper(k_rec).data
kappa_tru.data = enmap.to_flipper(k_tru).data
kappa_reg.data = enmap.to_flipper(k_reg).data
kappa_GI.data = enmap.to_flipper(k_GI).data
kappa_GI_unnorm.data = enmap.to_flipper(k_GI_unn).data
Tlens.data = enmap.to_flipper(lensed).data

area = deg_x*deg_y
totalArea = 41252.96
fsky = area/totalArea

# to take lx and ly
FT = liteMap.fftFromLiteMap(template)
Fphi = FT.copy()
Fphi = liteMap.fftFromLiteMap(phi)


lx = FT.lx
ly = FT.ly

Nx = Fphi.Nx # lx i think goes from 0 to Ny
Ny = Fphi.Ny
stepsize_lx = 2*np.pi/(Nx*np.pi/180.*pixXam/60.)

# gradient of unlensed temperature with killed small modes
Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr>l_grad] = 0.
filt[0] = 0.
# here we get more reali-STICK
Tlens_filt = Tlens.filterFromList([Larr,filt])

gradTu = Tlens_filt.takeGradient()
gradTux_o = gradTu.gradX
gradTuy_o = gradTu.gradY

# rms
rms = np.sqrt(np.mean(gradTux_o.data**2+gradTuy_o.data**2))
print "rms = ",rms

gradTux = gradTux_o
gradTuy = gradTuy_o

# lensed temperature power spectrum
p2d = fftTools.powerFromLiteMap(Tlens)
l, l, ll, ClT_l, l, l = p2d.binInAnnuli('BIN_'+str(dl)+SO)

# lensing potential power spectrum
ps2d = fftTools.powerFromLiteMap(kappa_tru)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_'+str(dl)+SO)
#plt.plot(ll,Clphi,label='my input kappa')
Clphi /= ll**4/4.
Clphi[0] = 0.

ps2d = fftTools.powerFromLiteMap(phi)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_'+str(dl)+SO)
#plt.plot(ll,Clphi*ll**4/4.,label='my phi input')

kappa = phiToKappa(phi)
# deconvolve the beam
Larr = np.arange(40000)
thetaFWHM =thetaFWHMarcmin*numpy.pi/(180.*60.)
beam = np.sqrt(np.exp(Larr*(Larr+1.)*thetaFWHM**2/(8.*np.log(2.))))
kappa = kappa.filterFromList([Larr,beam])

ps2d = fftTools.powerFromLiteMap(kappa)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_'+str(dl)+SO)
#plt.plot(ll,Clphi,label='my phi input as kappa')

ps2d = fftTools.powerFromLiteMap(kappa_rec,kappa_reg)#(kappa_rec,kappa_tru)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_'+str(dl)+SO)
plt.plot(ll,Clphi,label='rec_QE_cross_input_kappa')

ps2d = fftTools.powerFromLiteMap(kappa_rec)#(kappa_rec,kappa_tru)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_'+str(dl)+SO)
plt.plot(ll,Clphi,label='rec_QE_kappa.fits')

ps2d = fftTools.powerFromLiteMap(kappa_GI_unnorm,kappa_reg)#(kappa_GI,kappa_tru)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_'+str(dl)+SO)
plt.plot(ll,Clphi,label='rec_GI_cross_input_kappa')

ps2d = fftTools.powerFromLiteMap(kappa_GI)#(kappa_GI,kappa_tru)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_'+str(dl)+SO)
plt.plot(ll,Clphi,label='rec_GI_kappa.fits')

ps2d = fftTools.powerFromLiteMap(kappa_reg)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_'+str(dl)+SO)
plt.plot(ll,Clphi,label='regular_input_kappa.fits')


scal = '/users/boryanah/Blake/bor_sim/test_noise/Mat/data/Aug6_highAcc_CDM_scalCls.dat'
scal2 = root+'data/Aug6_highAcc_CDM_scalCls.dat'
theoryPower = np.loadtxt(scal2)
tcmb = 2.7255e6
lphi = theoryPower[:,0]
clphi = theoryPower[:,4]/(lphi**4)/tcmb**2
#plt.plot(lphi,clphi*lphi**4/4.,label='theory')

theoryPower_lensed = np.loadtxt(scal)
tcmb = 2.7255e6
lphi = theoryPower[:,0]
clphi = theoryPower[:,4]/(lphi**4)/tcmb**2
ll_the, Clkk_the = np.loadtxt('ells_clkk.txt',unpack=True)
plt.plot(ll_the,Clkk_the,label='ells_clkk.txt')
plt.yscale('log')
plt.xlim([5000,20000])
plt.ylim([6.e-11,2.2e-9])
plt.legend()
plt.savefig('ps.png')
quit()


# estimator in Fourier space
Fphi_est = Fphi.copy()
Fphi_est_unnorm = Fphi.copy()
Fphi_est.kMap[:,:] = 0
Fphi_est_unnorm.kMap[:,:] = 0

# estimator in real space
phi_est = phi.copy()
phi_est.data[:,:] = 0
phi_est_unnorm = phi.copy()
phi_est_unnorm.data[:,:] = 0

ell = FT.modLMap

Fphi.kMap[:,:] *= ell**2/2.
ps2d = fftTools.powerFromFFT(Fphi)
l, l, ll, Clkappa, l, l = ps2d.binInAnnuli('BIN_'+str(dl)+SO)


Clphi_ar = np.interp(ell,ll,Clphi)
C_ellT_l = np.interp(ell,ll,ClT_l)

powersum = C_ellT_l - ell**2*rms**2*Clphi_ar*.5


i_end = 201##102
if stepsize_lx*i_end < l_last:
    print "YOU ARE AN IDIOT"
    quit()
#l_last = 40000#15000#39500.
l_beg = 3000.#0.0001

print 'lx[i_end-1]:'
print lx[i_end-1]

#W_ell = phi.copy()

# TRY TO GET RID OF RINGING
Larr = np.arange(40000.)
filt = np.ones(len(Larr))
filt[Larr<l_ring] = 0.
filt[Larr>l_last] = 0.
Tlens = Tlens.filterFromList([Larr,filt])

for i in (range(Nx)[:i_end+1]+range(Nx)[-i_end:]):
    for j in range(Ny)[:i_end+1]:
        if (ell[i,j] > l_last or ell[i,j] < l_beg):
            pass
        else:
            # gradT dot ell
            print i,j
            gprod = (gradTux.data*lx[i]+ gradTuy.data*ly[j])
            x = powersum[i,j]/(gprod**2*Clphi_ar[i,j])
            W_ell = 1./(x)
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
l, l, ll, Clphi_est, l, l = ps2d_est.binInAnnuli('BIN_'+str(dl)+SO)

#ps2d_cross = fftTools.powerFromFFT(Fphi,Fphi_est_unnorm)
#ps2d_cross = fftTools.powerFromLiteMap(phi,phi_est_unnorm)
#l, l, ll, Clphi_cross, l, l = ps2d_cross.binInAnnuli('BIN_'+str(dl)+SO)

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
phi_est_unnorm.data[:,:] = Fphi_est_unnorm.mapFromFFT()

Clphi *= ll**4/4.


ps2d_cross = fftTools.powerFromLiteMap(kappa,phi_est_unnorm)
l, l, ll, Clphi_cross, l, l = ps2d_cross.binInAnnuli('BIN_'+str(dl)+SO)

f1 = open(fname, 'w+')
for i in range(len(ll)):
    print >> f1, ('% 8.1f' % ll[i]), ('% 18.8e' % Clphi[i]), ('% 18.8e' % Clphi_est[i]), ('% 18.8e' % Clphi_cross[i]), ('% 18.8e' % yerr[i]),('% 18.8e' % Nl_pred_u[i]), ('% 18.8e' % Nl_pred[i]), ('% 18.8e' % Nlphi[i])

if myrank == offset+0:
    phi_est.data[:,:] = Fphi_est.mapFromFFT()
    phi_est.writeFits(fkappa_est,overWrite=True)
    phi_est_unnorm.writeFits(fkappa_est_unnorm,overWrite=True)
    kappa.writeFits(fkappa,overWrite = True)
    #kappa_tru.writeFits(fkappa,overWrite = True)
    kappa_rec.writeFits(fkappa_rec,overWrite = True)
