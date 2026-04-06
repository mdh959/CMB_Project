import matplotlib
matplotlib.use('Agg')

import orphics                      
import orphics.tools.stats

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
# dl N_i noise N_sim

# Mat's files
root = '/users/boryanah/repos/alhazen/'
fname_lensed = root+'for_boryanka/lensed_'
fname_phi = root+'for_boryanka/kappa_true_'
fname_kappa_rec = root+'for_boryanka/kappa_recon_'
fname_kappa_true = root+'for_boryanka/kappa_true_'

noiseFactor = .1#.9999#6.#.1#.9999
pwd = '/users/boryanah/Blake/bor_sim/test_noise/Mat/sims/test_'
area = '2.91260385523'

scale = (2.91260385523/(41253.))/0.4

myrank = 0
N_sim_beg = 0
N_sim_end = 360#180#300#360#150#180
N_sim = N_sim_end-N_sim_beg
offset = 0
dl = 2000#2000#''#2000

if dl == '' and noiseFactor == .9999:
    n = 12
if dl == 2000:
    n = 10
if dl == '' and noiseFactor == 6.:
    n = 10
if dl == '' and noiseFactor == .1:
    n = 40
print n 
myrank += offset
print myrank

if noiseFactor == .9999: # CMB-S4 conservative
    tit = 'S4'
    fname = pwd+'Clphi_S4_cons_k_'+str(myrank)+'_Mat'+str(dl)+'_big.txt'
    S4 = '_S4'
    SO = '_S4'
    if dl == '':
        dll = 1000
    fcrosscov = root+'plots/experiment_boryana_1arc_1uk_2000_'+area+'sqdeg_crosscovmat_dl'+str(dll)+'.npy'
    fcov = root+'plots/experiment_boryana_1arc_1uk_2000_'+area+'sqdeg_covmat_dl'+str(dll)+'.npy'
    fbin = root+'plots/experiment_boryana_1arc_1uk_2000_'+area+'sqdeg_lbin_edges_dl'+str(dll)+'.npy'

if noiseFactor == .1:
    tit = 'UL'
    fname = pwd+'Clphi_0_1_k_'+str(myrank)+'_Mat'+str(dl)+'_big.txt'
    S4 = ''
    SO = ''
    if dl == '': dll = 1000
    if dl == 2000: dll = 2000
    fcrosscov = root+'plots/experiment_boryana_0.3arc_0.1uk_2000_'+area+'sqdeg_crosscovmat_dl'+str(dll)+'.npy'
    fcov = root+'plots/experiment_boryana_0.3arc_0.1uk_2000_'+area+'sqdeg_covmat_dl'+str(dll)+'.npy'
    fbin = root+'plots/experiment_boryana_0.3arc_0.1uk_2000_'+area+'sqdeg_lbin_edges_dl'+str(dll)+'.npy'
    
if noiseFactor == 6.:
    tit = 'SO'
    fname = pwd+'Clphi_SO_k_'+str(myrank)+'_Mat'+str(dl)+'_big.txt'
    S4 = '_SO'
    SO = '_SO'
    if dl == '': dll = 1000
    fcrosscov = root+'plots/experiment_boryana_1.4arc_6uk_2000_'+area+'sqdeg_crosscovmat_dl'+str(dll)+'.npy'
    fcov = root+'plots/experiment_boryana_1.4arc_6uk_2000_'+area+'sqdeg_covmat_dl'+str(dll)+'.npy'
    fbin = root+'plots/experiment_boryana_1.4arc_6uk_2000_'+area+'sqdeg_lbin_edges_dl'+str(dll)+'.npy'
    
deg_x = 1.7066666667
deg_y = 1.7066666667

pixXam = .1
pixYam = .1
template = liteMap.makeEmptyCEATemplate(deg_x, deg_y,meanRa = 180., meanDec = 0.,\
                      pixScaleXarcmin = pixXam, pixScaleYarcmin = pixYam)
    
# tuka
Cl_pp = np.zeros((n,N_sim))
Cl_au = np.zeros((n,N_sim))
Cl_cr = np.zeros((n,N_sim))
Cl_cr_QE = np.zeros((n,N_sim))
Cl_au_QE = np.zeros((n,N_sim))

ll_the, Clkk_the = np.loadtxt('ells_clkk.txt',unpack=True)

for myrank in range(N_sim_beg,N_sim_end):
    print myrank
    if noiseFactor == .1:
        fname = pwd+'Clphi_0_1_k_'+str(myrank)+'_Mat'+str(dl)+'_big.txt'
        st = '_UL'
    if noiseFactor == .9999:
        myr = myrank
        #myr = myrank%150
        fname = pwd+'Clphi_S4_cons_k_'+str(myr)+'_Mat'+str(dl)+'_big.txt'
        st = '_S4'
    if noiseFactor == 6.:
        fname = pwd+'Clphi_SO_k_'+str(myrank)+'_Mat'+str(dl)+'_big.txt'
        st = '_SO'
        
    print(fname)

    k_tru = enmap.read_fits(fname_kappa_true+str(myrank)+S4+'_'+str(area)+'.fits')
    '''
    # testing
    print fname_kappa_true+str(myrank)+S4+'_'+str(area)+'.fits'
    print fname_kappa_rec+str(myrank)+S4+'_'+str(area)+'.fits'
    k_GI = enmap.read_fits('kappa_est_unnorm_'+str(myrank)+'_0_1_big.fits')
    #k_tru = enmap.read_fits('kappa_true_'+str(myrank)+'_2.91260385523.fits')
    kappa_GI = template.copy()
    kappa_GI.data = enmap.to_flipper(k_GI).data
    '''

    k_rec = enmap.read_fits(fname_kappa_rec+str(myrank)+S4+'_'+str(area)+'.fits')
    kappa_rec = template.copy()
    kappa_tru = template.copy()


    kappa_rec.data = enmap.to_flipper(k_rec).data
    kappa_tru.data = enmap.to_flipper(k_tru).data


    # testing
    #myrank = myrank/N_sim_beg

    ell,Cl_pp[:,myrank],Cl_au[:,myrank],Cl_cr[:,myrank],d,g,e,f = np.loadtxt(fname,unpack=True)
    
    ps2d = fftTools.powerFromLiteMap(kappa_rec,kappa_tru)
    l, l, ll, Cl_cr_QE[:,myrank], l, l = ps2d.binInAnnuli('BIN_'+str(dll)+SO)

    '''
    # testing
    ps2d = fftTools.powerFromLiteMap(kappa_GI,kappa_tru)
    l, l, ll, Cl_cr_GI, l, l = ps2d.binInAnnuli('BIN_'+str(dll)+SO)
    ps2d = fftTools.powerFromLiteMap(kappa_tru)
    l, l, ll, Cl_t, l, l = ps2d.binInAnnuli('BIN_'+str(dll)+SO)
    # testing
    print Cl_cr_GI
    print Cl_t
    print Cl_cr_QE[:,myrank]
    print Cl_cr[:,myrank]
    '''
    
    
    ps2d = fftTools.powerFromLiteMap(kappa_rec)
    l, l, ll, Cl_au_QE[:,myrank], l, l = ps2d.binInAnnuli('BIN_'+str(dll)+SO)

    
# computing covariance matrix
Clkk_ell = np.interp(ell,ll_the,Clkk_the)
Clkk_ll = np.interp(ll,ll_the,Clkk_the)

Cov_cr_GI = np.cov(Cl_cr)
Cov_au_GI = np.cov(Cl_au)
Cov_cr_QE = np.cov(Cl_cr_QE)
Cov_au_QE = np.cov(Cl_au_QE)

'''
# start testing
Cov_au_QE = np.load(fcov)
Cov_cr_QE = np.load(fcrosscov)
lbin = np.load(fbin)
lbin = (lbin[1:]+lbin[:-1])*.5
print lbin
print ll
N_i = 4#2#5
N_e = 1
Clkk_ll = np.interp(lbin,ll_the,Clkk_the)
lbin = lbin[N_i:-N_e]
ll = ll[N_i:-N_e]
print lbin
print ll
Clkk_ll = Clkk_ll[N_i:-N_e]
Cov_cr_QE = Cov_cr_QE[N_i:-N_e,N_i:-N_e]*scale
Cov_au_QE = Cov_au_QE[N_i:-N_e,N_i:-N_e]*scale
SNR_full_cr_QE = np.sqrt(np.dot(np.dot(Clkk_ll, np.linalg.inv(Cov_cr_QE)),Clkk_ll))
SNR_full_au_QE = np.sqrt(np.dot(np.dot(Clkk_ll, np.linalg.inv(Cov_au_QE)),Clkk_ll))
print "SNR full cr QE"
print SNR_full_cr_QE
print "SNR full au QE"
print SNR_full_au_QE
quit()
# end testing
'''

Cl_cr_GI_mean = np.mean(Cl_cr,axis=1)
Cl_cr_QE_mean = np.mean(Cl_cr_QE,axis=1)
Cl_au_GI_mean = np.mean(Cl_au,axis=1)
Cl_au_QE_mean = np.mean(Cl_au_QE,axis=1)
plt.plot(ll,Cl_cr_QE_mean,label='rec_QE_input_kappa.fits')
plt.plot(ell,Cl_cr_GI_mean,label='rec_GI_input_kappa.fits')
plt.plot(ll,Cl_au_QE_mean,label='rec_QE.fits')
plt.plot(ell,Cl_au_GI_mean,label='rec_GI.fits')
plt.title("Cross and auto power spectrum for "+tit+", "+str(N_sim)+" sims")
plt.plot(ll_the,Clkk_the,label="ells_clkk.txt")
plt.xlim([100,20000])
plt.ylim([1.e-11,1.e-7])
plt.yscale('log')
plt.legend()
plt.savefig('ps'+st+'.png')
plt.close()

Cl_cr_GI_std = np.std(Cl_cr,axis=1)
Cl_cr_QE_std = np.std(Cl_cr_QE,axis=1)
Cl_au_GI_std = np.std(Cl_au,axis=1)
Cl_au_QE_std = np.std(Cl_au_QE,axis=1)
plt.plot(ll,Cl_cr_QE_std,label='rec_QE_input_kappa.fits')
plt.plot(ell,Cl_cr_GI_std,label='rec_GI_input_kappa.fits')
plt.plot(ll,Cl_au_QE_std,label='rec_QE_input_kappa.fits')
plt.plot(ell,Cl_au_GI_std,label='rec_GI_input_kappa.fits')
plt.title("STD of cross and auto power spectrum for "+tit+", "+str(N_sim)+" sims")
plt.xlim([100,20000]) #tuk
#plt.xlim([100,12000])

plt.ylim([1.e-11,1.e-8])
plt.yscale('log')
plt.legend()
plt.savefig('std'+st+'.png')
plt.close()

N_i = 2#2#5
N_e = 1
ll = ll[N_i:-N_e]
ell = ell[N_i:-N_e]
Cl_cr_QE_std = Cl_cr_QE_std[N_i:-N_e]*np.sqrt(scale)
Cl_au_QE_std = Cl_au_QE_std[N_i:-N_e]*np.sqrt(scale)
Cl_cr_GI_std = Cl_cr_GI_std[N_i:-N_e]*np.sqrt(scale)
Cl_au_GI_std = Cl_au_GI_std[N_i:-N_e]*np.sqrt(scale)
Clkk_ell = Clkk_ell[N_i:-N_e]
Clkk_ll = Clkk_ll[N_i:-N_e]
SNR_GI_cr = np.sqrt(np.sum(Clkk_ell**2/Cl_cr_GI_std**2))
SNR_GI_au = np.sqrt(np.sum(Clkk_ell**2/Cl_au_GI_std**2))
SNR_QE_cr = np.sqrt(np.sum(Clkk_ll**2/Cl_cr_QE_std**2))
SNR_QE_au = np.sqrt(np.sum(Clkk_ll**2/Cl_au_QE_std**2))
rat_SNR_GQ_au = SNR_GI_au/SNR_QE_au
rat_SNR_GQ_cr = SNR_GI_cr/SNR_QE_cr


Cov_cr_GI = Cov_cr_GI[N_i:-N_e,N_i:-N_e]*scale
Cov_au_GI = Cov_au_GI[N_i:-N_e,N_i:-N_e]*scale
Cov_cr_QE = Cov_cr_QE[N_i:-N_e,N_i:-N_e]*scale
Cov_au_QE = Cov_au_QE[N_i:-N_e,N_i:-N_e]*scale
SNR_full_cr_GI = np.sqrt(np.dot(np.dot(Clkk_ell, np.linalg.inv(Cov_cr_GI)),Clkk_ell))
SNR_full_au_GI = np.sqrt(np.dot(np.dot(Clkk_ell, np.linalg.inv(Cov_au_GI)),Clkk_ell))
SNR_full_cr_QE = np.sqrt(np.dot(np.dot(Clkk_ll, np.linalg.inv(Cov_cr_QE)),Clkk_ll))
SNR_full_au_QE = np.sqrt(np.dot(np.dot(Clkk_ll, np.linalg.inv(Cov_au_QE)),Clkk_ll))
rat_full_SNR_GQ_au = SNR_full_au_GI/SNR_full_au_QE
rat_full_SNR_GQ_cr = SNR_full_cr_GI/SNR_full_cr_QE

Corr_cr_GI = orphics.tools.stats.cov2corr(Cov_cr_GI)
Corr_cr_QE = orphics.tools.stats.cov2corr(Cov_cr_QE)
Corr_au_GI = orphics.tools.stats.cov2corr(Cov_au_GI)
Corr_au_QE = orphics.tools.stats.cov2corr(Cov_au_QE)

plt.figure(figsize=(9,7))                                                                   
plt.imshow(Corr_cr_GI)  
plt.colorbar()                                                                              
plt.savefig("Corr_cr_GI_"+tit+".png")
plt.close()

plt.figure(figsize=(9,7))                                                                   
plt.imshow(Corr_cr_QE)    
plt.colorbar()                                                                              
plt.savefig("Corr_cr_QE_"+tit+".png")
plt.close()

plt.figure(figsize=(9,7))                                                                   
plt.imshow(Corr_au_GI)
plt.colorbar()                                                                              
plt.savefig("Corr_au_GI_"+tit+".png")
plt.close()

plt.figure(figsize=(9,7))                                                                   
plt.imshow(Corr_au_QE)    
plt.colorbar()                                                                              
plt.savefig("Corr_au_QE_"+tit+".png")
plt.close() 

print "rat GI/QE cr"
print rat_SNR_GQ_cr

print "rat GI/QE au"
print rat_SNR_GQ_au

print "rat full GI/QE au"
print rat_full_SNR_GQ_au

print "rat full GI/QE cr"
print rat_full_SNR_GQ_cr

print "SNR full cr GI"
print SNR_full_cr_GI

print "SNR full au GI"
print SNR_full_au_GI

print "SNR full cr QE"
print SNR_full_cr_QE

print "SNR full au QE"
print SNR_full_au_QE

print "SNR GI cr"
print SNR_GI_cr

print "SNR QE cr"
print SNR_QE_cr

print "SNR GI au"
print SNR_GI_au

print "SNR QE au"
print SNR_QE_au

print "ll"
print ll
print "cr QE"
print Cl_cr_QE_std
print "au QE"
print Cl_au_QE_std

print "ell"
print ell
print "cr GI"
print Cl_cr_GI_std
print "au GI"
print Cl_au_GI_std
quit()
np.savetxt('std_cr_GI'+st+'.txt',np.vstack((ell,Cl_cr_GI_std)).T)
np.savetxt('std_cr_QE'+st+'.txt',np.vstack((ll,Cl_cr_QE_std)).T)


totalArea = 41252.96
fsky = deg_x*deg_y/totalArea
