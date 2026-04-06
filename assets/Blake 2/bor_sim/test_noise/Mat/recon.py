import matplotlib
#matplotlib.use('Agg')                                                                                                
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

#noiseFactor = .9999
noiseFactor = .1

pwd = '/users/boryanah/Blake/bor_sim/test_noise/Mat/plots/'
pwd2 = '/users/boryanah/Blake/bor_sim/test_noise/Mat/sims/'

if noiseFactor == .9999:
    name_kappa = 'kappa_S4_cons_big.fits'
    name_kappa_est = 'kappa_est_S4_cons_big.fits'
    name_kappa_QE = 'kappa_recon_50_S4_2.912603855229585.fits'
    fname = pwd2+'test_Clphi_S4_cons_k_'
    
    N = 100
    a_avg = np.zeros(12)
    b_avg = np.zeros(12)
    for i in range(N):
        l,a,b,c,d,g,e,f = np.loadtxt(fname+str(i)+'_Mat'+'_big.txt',unpack=True)
        print a
        a_avg += a
        b_avg += b
    a_avg /= N
    b_avg /= N

    kappa = liteMap.liteMapFromFits(name_kappa)
    kappa_est = liteMap.liteMapFromFits(name_kappa_est)
    kappa_QE = enmap.read_fits(name_kappa_QE)
    kappa_qe = kappa.copy()
    kappa_qe.data = enmap.to_flipper(kappa_QE).data

    Larr = np.arange(40000.)
    Clkk_sig = np.interp(Larr,l,a_avg)
    Clkk_GI = np.interp(Larr,l,b_avg)
    Nlkk_GI = Clkk_GI-Clkk_sig
    W = Clkk_sig/(Clkk_sig+Nlkk_GI)
    filt = np.ones(len(Larr))
    filt *= W/Larr**2
    filt[Larr<5000.] = 0.
    filt[0] = 0.
    kappa_est = kappa_est.filterFromList([Larr,filt])
    kappa = kappa.filterFromList([Larr,filt])
    kappa_qe = kappa_qe.filterFromList([Larr,filt])

    #kappa_est.plot(show=False,saveFig = pwd+'kappa_est_S4_cons.pdf',valueRange=[np.min(kappa.data),np.max(kappa.data)])
    plt.figure(1)
    kappa_est.plot(show=False,saveFig = pwd+'kappa_est_S4_cons.pdf',valueRange=[np.min(kappa_est.data),np.max(kappa_est.data)])
    plt.close()
    
    plt.figure(2)
    #kappa_qe.plot(show=False,saveFig = pwd+'kappa_qe_S4_cons.pdf',valueRange=[np.min(kappa.data),np.max(kappa.data)])
    kappa_qe.plot(show=False,saveFig = pwd+'kappa_qe_S4_cons.pdf',valueRange=[np.min(kappa_est.data),np.max(kappa_est.data)])#valueRange=[np.min(kappa_est.data),np.max(kappa_est.data)])
    plt.close()

    plt.figure(3)
    #kappa.plot(show=False,saveFig = pwd+'kappa_real_S4_cons.pdf',valueRange=[np.min(kappa.data),np.max(kappa.data)])
    kappa.plot(show=False,saveFig = pwd+'kappa_real_S4_cons.pdf',valueRange=[np.min(kappa_est.data),np.max(kappa_est.data)])#,valueRange=[np.min(kappa_est.data),np.max(kappa_est.data)])
    plt.close()
    plt.show()

if noiseFactor == .1:
    name_kappa = 'kappa_0_1_big.fits'
    name_kappa_est = 'kappa_est_0_1_big.fits'
    name_kappa_QE = '/users/boryanah/repos/alhazen/for_boryanka/kappa_recon_172_2.91260385523.fits'
    kappa = liteMap.liteMapFromFits(name_kappa)
    kappa_est = liteMap.liteMapFromFits(name_kappa_est)
    kappa_QE = enmap.read_fits(name_kappa_QE)
    kappa_qe = kappa.copy()
    kappa_qe.data = enmap.to_flipper(kappa_QE).data

    fname = pwd2+'test_Clphi_0_1_k_' 
    N = 100
    a_avg = np.zeros(10)
    b_avg = np.zeros(10)
    for i in range(N):
        l,a,b,c,d,g,e,f = np.loadtxt(fname+str(i)+'_Mat2000'+'_big.txt',unpack=True)
        print a
        a_avg += a
        b_avg += b
    a_avg /= N
    b_avg /= N

    Larr = np.arange(40000.)
    Clkk_sig = np.interp(Larr,l,a_avg)
    Clkk_GI = np.interp(Larr,l,b_avg)
    Nlkk_GI = Clkk_GI-Clkk_sig
    W = Clkk_sig/(Clkk_sig+Nlkk_GI)
    filt = np.ones(len(Larr))
    filt *= W/Larr**2
    filt[Larr<5000.] = 0.
    filt[0] = 0.
    phi_est = kappa_est.filterFromList([Larr,filt])
    phi = kappa.filterFromList([Larr,filt])
    phi_qe = kappa_qe.filterFromList([Larr,filt])

    phi_est.plot(show=False,saveFig = pwd+'phi_est_0_1.pdf',valueRange=[np.min(phi.data),np.max(phi.data)])
    plt.figure(1)
    #phi_est.plot(show=False,saveFig = pwd+'phi_est_0_1.pdf',valueRange=[np.min(phi_est.data),np.max(phi_est.data)])
    plt.close()
    
    plt.figure(2)
    phi_qe.plot(show=False,saveFig = pwd+'phi_qe_0_1.pdf',valueRange=[np.min(phi.data),np.max(phi.data)])
    #phi_qe.plot(show=False,saveFig = pwd+'phi_qe_0_1.pdf',valueRange=[np.min(phi_est.data),np.max(phi_est.data)])#valueRange=[np.min(phi_est.data),np.max(phi_est.data)])
    plt.close()

    plt.figure(3)
    phi.plot(show=False,saveFig = pwd+'phi_real_0_1.pdf',valueRange=[np.min(phi.data),np.max(phi.data)])
    #phi.plot(show=False,saveFig = pwd+'phi_real_0_1.pdf',valueRange=[np.min(phi_est.data),np.max(phi_est.data)])#,valueRange=[np.min(phi_est.data),np.max(phi_est.data)])
    plt.close()
    #plt.show()
