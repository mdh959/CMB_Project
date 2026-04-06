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
#root = '/users/boryanah/repos/alhazen_old/'
root = '/users/boryanah/repos/alhazen/'
fname_lensed = root+'for_boryanka/lensed_'
#fname_phi = root+'for_boryanka/phi_'
# always better to use kappa ps and map
fname_phi = root+'for_boryanka/kappa_true_'
fname_kappa_rec = root+'for_boryanka/kappa_recon_'
fname_kappa_true = root+'for_boryanka/kappa_true_'

#comm = MPI.COMM_WORLD
#myrank = comm.Get_rank()
#nproc = comm.Get_size()
myrank = 0

offset = 172#103#100#96#64# 32
dl = 2000#2000#5000 #''
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
    #fname = pwd+'Clphi_0_1_k_'+str(myrank)+'_Mat'+str(dl)+'_big.txt'
    fname = 'Clphi_0_1_k_'+str(myrank)+'_Mat'+str(dl)+'_big.txt'
    S4 = ''
    SO = ''
    thetaFWHMarcmin = .3
    fkappa = 'kappa_0_1_big.fits'
    fkappa_est = 'kappa_est_0_1_big.fits'
    fkappa_rec = 'kappa_rec_0_1_big.fits'
    fkappa_est_unnorm = 'kappa_est_unnorm_0_1_big.fits'
    l_last = 21000#21000
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

#fname_unlensed = root+'for_boryanka/unlensed_'
fname_unlensed = root+'for_boryanka/lensed_'

deg_x = 1.7066666667
deg_y = 1.7066666667
area = '2.91260385523'#'2.912603855229585'

pixXam = .1
pixYam = .1
template = liteMap.makeEmptyCEATemplate(deg_x, deg_y,meanRa = 180., meanDec = 0.,\
                      pixScaleXarcmin = pixXam, pixScaleYarcmin = pixYam)

# downsampled 0.1 arcmin pix and 1x1 deg^2 nope
#lensed = enmap.read_fits(fname_lensed+str(myrank)+S4+'_'+str(area)+'.fits')
# !!!!! important
#lensed = enmap.read_fits('regular_lensed_cmb.fits')
lensed = enmap.read_fits('custom_lensed_cmb_flux_4.fits')
print(fname_lensed+str(myrank)+S4+'_'+str(area)+'.fits')
print(fname_phi+str(myrank)+S4+'_'+str(area)+'.fits')

# !!!!!!!!!!!!!!! important
ll_the, Clkk_the = np.loadtxt('ells_clkk.txt',unpack=True)
#Clphi_the = Clkk_the*4./ll_the**4
#Clphi_the[0] = 0.

# these are the not downsampled ones
unlensed = enmap.read_fits(fname_lensed+str(myrank)+S4+'_'+str(area)+'.fits')


ph = enmap.read_fits(fname_phi+str(myrank)+S4+'_'+str(area)+'.fits')

#k_tru = enmap.read_fits('regular_input_kappa.fits')
#k_tru = enmap.read_fits(fname_kappa_true+str(myrank)+S4+'_'+str(area)+'.fits')
k_tru = enmap.read_fits('custom_input_kappa_flux_4.fits')
k_rec = enmap.read_fits(fname_kappa_rec+str(myrank)+S4+'_'+str(area)+'.fits')


# HERE
k_unn = enmap.read_fits(fkappa_est_unnorm)
k_rec = enmap.read_fits(fkappa_rec)
#k_tru = enmap.read_fits(fkappa)
k_est = enmap.read_fits(fkappa_est)
#HERE


noise = template.copy()
phi = template.copy()
kappa_unn = template.copy()
Tulens = template.copy()
Tlens = template.copy()
kappa_rec = template.copy()
kappa_tru = template.copy()
kappa_est = template.copy()

phi.data = enmap.to_flipper(ph).data
#enmap.to_flipper(k_tru).data#
kappa_rec.data = enmap.to_flipper(k_rec).data
kappa_unn.data = enmap.to_flipper(k_unn).data
kappa_est.data = enmap.to_flipper(k_est).data
kappa_tru.data = enmap.to_flipper(k_tru).data

'''
# HERE
Larr = np.arange(40000)
filt = np.ones(len(Larr))
filt[Larr<10000.] = 0.
filt[0] = 0.
kappa_tru = kappa_tru.filterFromList([Larr,filt])
kappa_rec = kappa_rec.filterFromList([Larr,filt])
kappa_unn = kappa_unn.filterFromList([Larr,filt])
kappa_est = kappa_est.filterFromList([Larr,filt])

kappa_est.writeFits("filt_"+fkappa_est,overWrite=True)
kappa_unn.writeFits("filt_"+fkappa_est_unnorm,overWrite=True)
#kappa.writeFits(fkappa,overWrite = True)
kappa_tru.writeFits("filt_"+fkappa,overWrite = True)
kappa_rec.writeFits("filt_"+fkappa_rec,overWrite = True)
quit()
# HERE
'''

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! important
# for Mat's custom and for his other input not for my own
Tlens.data = enmap.to_flipper(lensed).data/2.7255e6
Tulens.data = enmap.to_flipper(unlensed).data#/2.7255e6

area = deg_x*deg_y
totalArea = 41252.96
fsky = area/totalArea

# deconvolving with the beam
Larr = np.arange(40000)
thetaFWHM =thetaFWHMarcmin*numpy.pi/(180.*60.)
beam = (np.exp(.5*Larr*(Larr+1.)*thetaFWHM**2/(8.*np.log(2.))))
Tlens = Tlens.filterFromList([Larr,beam])
# BS
#phi = phi.filterFromList([Larr,1./beam])

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
# not different from
#Tlens_filt = Tlens_noN.filterFromList([Larr,filt])
#FTlens_filt = liteMap.fftFromLiteMap(Tlens_filt)

# CHANGE THIS much better god knows why
#gradTu = Tulens.takeGradient()
gradTu = Tlens_filt.takeGradient()

gradTux = gradTu.gradX
gradTuy = gradTu.gradY

'''
# HERE
gradTux.writeFits('gradTux.fits',overWrite=True)
gradTuy.writeFits('gradTuy.fits',overWrite=True)
gradTux.data = (gradTux.data**2+gradTuy.data**2)
gradTux.writeFits('gradTu.fits',overWrite=True)
quit()
# HERE
'''

# rms
rms = np.sqrt(np.mean(gradTux.data**2+gradTuy.data**2))
print "rms = ",rms


# lensed temperature power spectrum
p2d = fftTools.powerFromLiteMap(Tulens)
l, l, ll, ClT_l, l, l = p2d.binInAnnuli('BIN_'+str(dl)+SO)
plt.plot(ll,ll**2*ClT_l,label=fname_unlensed)

p2d = fftTools.powerFromLiteMap(Tlens)
l, l, ll, ClT_l, l, l = p2d.binInAnnuli('BIN_'+str(dl)+SO)
plt.plot(ll,ll**2*ClT_l,label='Mat')

#plt.plot(ll,ll**2*ClT_l)

lens = '/users/boryanah/Blake/bor_sim/test_noise/Mat/data/Aug6_highAcc_CDM_lensedCls.dat'
theoryPower_lensed = np.loadtxt(lens)
l_len=theoryPower_lensed[:,0]
cl_TT_len=theoryPower_lensed[:,1]/2.7255e6**2
cl_TT_len*=2*np.pi/(l_len*(l_len+1))

plt.plot(l_len,l_len**2*cl_TT_len,label='theory')
plt.yscale('log')
plt.xlim([100,20000])
plt.ylim([1.e-14,1.e-8])
plt.legend()
plt.savefig('ps.png')
plt.close()

# lensing potential power spectrum
ps2d = fftTools.powerFromLiteMap(phi)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_'+str(dl)+SO)
#Clphi /= 4./ll**4
plt.plot(ll,Clphi,label=fname_phi)
Clphi *= 4./ll**4
# !!!!!!!!!!!!!!!!! important
#phi=phiToKappa(phi)
ps2d = fftTools.powerFromLiteMap(kappa_rec,phi)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_'+str(dl)+SO)
plt.plot(ll,Clphi,label='rec_QE_input_kappa.fits')
Clphi *= 4./ll**4


ps2d = fftTools.powerFromLiteMap(kappa_unn,phi)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_'+str(dl)+SO)
plt.plot(ll,Clphi,label='rec_GI_input_kappa.fits')
Clphi *= 4./ll**4


ps2d = fftTools.powerFromLiteMap(kappa_tru)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_'+str(dl)+SO)
plt.plot(ll,Clphi,label=fname_kappa_true)
Clphi *= 4./ll**4
#plt.plot(ll,Clphi,label='regular_input_kappa.fits')

plt.plot(ll_the,Clkk_the,label="ells_clkk.txt")
plt.xlim([100,20000])
plt.ylim([1.e-11,1.e-7])
plt.yscale('log')
plt.legend()
plt.savefig('ps2.png')
plt.close()
#quit()
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


#C_ellT_u = np.interp(ell,ll,ClT_u)
# if using alternative noise with lensing
Clphi_ar = np.interp(ell,ll,Clphi)
C_ellT_l = np.interp(ell,ll,ClT_l)


#N_ellT = np.interp(ell,l_len,NlT)
#N_ellT = np.interp(ell,ll,Cln)
# used to be
#N_ellT = (deltaT*thetaFWHM)**2*np.exp(ell*(ell+1.)*thetaFWHM**2/(8.*np.log(2.)))*noiseFactor**2.

powersum = C_ellT_l - ell**2*rms**2*Clphi_ar*.5


i_end = 201##102
if stepsize_lx*i_end < l_last:
    print "YOU ARE AN IDIOT"
    quit()
#l_last = 40000#15000#39500.
l_beg = 4000#0.0001

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
            #W_ell = 1./(1-2*x)
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

# !!!!!!!!!!!!!!!!!!!!! important
Clphi *= ll**4/4.

kappa = phiToKappa(phi)
ps2d_cross = fftTools.powerFromLiteMap(kappa_tru,phi_est_unnorm)
l, l, ll, Clphi_cross, l, l = ps2d_cross.binInAnnuli('BIN_'+str(dl)+SO)

ps2d_cross = fftTools.powerFromLiteMap(kappa_tru,phi_est)
l, l, ll, Clphi_cross2, l, l = ps2d_cross.binInAnnuli('BIN_'+str(dl)+SO)

f1 = open(fname, 'w+')
for i in range(len(ll)):
    print >> f1, ('% 8.1f' % ll[i]), ('% 18.8e' % Clphi[i]), ('% 18.8e' % Clphi_est[i]), ('% 18.8e' % Clphi_cross[i]), ('% 18.8e' % Clphi_cross2[i]),('% 18.8e' % Nl_pred_u[i]), ('% 18.8e' % Nl_pred[i]), ('% 18.8e' % Nlphi[i])

if myrank == offset+0:
    phi_est.data[:,:] = Fphi_est.mapFromFFT()
    phi_est.writeFits(fkappa_est,overWrite=True)
    phi_est_unnorm.writeFits(fkappa_est_unnorm,overWrite=True)
    #kappa.writeFits(fkappa,overWrite = True)
    kappa_tru.writeFits(fkappa,overWrite = True)
    kappa_rec.writeFits(fkappa_rec,overWrite = True)
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
