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

#noiseFactor = .9999
noiseFactor = .1


pwd = '/users/boryanah/Blake/bor_sim/test_noise/Mat/plots/'
pwd2 = '/users/boryanah/Blake/bor_sim/test_noise/Mat/sims/'

if noiseFactor == .9999:
    name_kappa = 'kappa_S4_cons_big.fits'
    name_kappa_est = 'kappa_est_S4_cons_big.fits'
    name_kappa_QE = 'kappa_recon_50_S4_2.912603855229585.fits'
    kappa = liteMap.liteMapFromFits(name_kappa)
    kappa_est = liteMap.liteMapFromFits(name_kappa_est)
    kappa_QE = enmap.read_fits(name_kappa_QE)
    kappa_qe = kappa.copy()
    kappa_qe.data = enmap.to_flipper(kappa_QE).data

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

    GI = kappa.copy()
    GI.data[:,:] = kappa_est.data-kappa.data
    QE = kappa.copy()
    QE.data[:,:] = kappa_qe.data-kappa.data

    plt.figure(1)
    kappa_est.plot(show=False,saveFig = pwd+'GI_diff_S4_cons.pdf',valueRange=[np.min(GI.data),np.max(GI.data)])
    #kappa_est.plot(show=False,saveFig = pwd+'kappa_est_S4_cons.pdf',valueRange=[np.min(kappa_est.data),np.max(kappa_est.data)])
    plt.close()
    
    plt.figure(2)
    #kappa_qe.plot(show=False,saveFig = pwd+'QE_diff_S4_cons.pdf',valueRange=[np.min(GI.data),np.max(GI.data)])
    kappa_qe.plot(show=False,saveFig = pwd+'QE_diff_S4_cons.pdf',valueRange=[np.min(QE.data)/10,np.max(QE.data)/10])

    #kappa_qe.plot(show=False,saveFig = pwd+'kappa_qe_S4_cons.pdf',valueRange=[np.min(kappa_est.data),np.max(kappa_est.data)])#valueRange=[np.min(kappa_est.data),np.max(kappa_est.data)])
    plt.close()

if noiseFactor == .1:
    name_kappa = 'kappa_0_1_big.fits'
    name_kappa_est = 'kappa_est_0_1_big.fits'
    name_kappa_QE = 'kappa_recon_50_2.912603855229585.fits'
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
    kappa_est = kappa_est.filterFromList([Larr,filt])
    kappa = kappa.filterFromList([Larr,filt])
    kappa_qe = kappa_qe.filterFromList([Larr,filt])


    GI = kappa.copy()
    GI.data[:,:] = kappa_est.data-kappa.data
    QE = kappa.copy()
    QE.data[:,:] = kappa_qe.data-kappa.data

    plt.figure(1)
    kappa_est.plot(show=False,saveFig = pwd+'GI_diff_0_1.pdf',valueRange=[np.min(GI.data),np.max(GI.data)])
    #kappa_est.plot(show=False,saveFig = pwd+'kappa_est_0_1.pdf',valueRange=[np.min(kappa_est.data),np.max(kappa_est.data)])
    plt.close()
    
    plt.figure(2)
    kappa_qe.plot(show=False,saveFig = pwd+'QE_diff_0_1.pdf',valueRange=[np.min(GI.data),np.max(GI.data)])
    #kappa_qe.plot(show=False,saveFig = pwd+'kappa_qe_0_1.pdf',valueRange=[np.min(kappa_est.data),np.max(kappa_est.data)])#valueRange=[np.min(kappa_est.data),np.max(kappa_est.data)])
    plt.close()

    S4 = ''
    myrank = 0
    offset = 50
    myrank += offset
    root = '/users/boryanah/repos/alhazen/'
    fname_lensed = root+'for_boryanka/lensed_'
    area = '2.912603855229585'
    lensed = enmap.read_fits(fname_lensed+str(myrank)+S4+'_'+str(area)+'.fits')

    Tlens = kappa.copy()
    Tlens.data = enmap.to_flipper(lensed).data

    Larr = np.arange(40000.)                                                                                                                                                            
    filt = np.ones(len(Larr))                                                                                                                                                           
    filt[Larr>2000.] = 0.                                                                                                                                                               
    filt[0] = 0.                                                                                                                                                                        
    Tlens_filt = Tlens.filterFromList([Larr,filt])

    gradTu = Tlens_filt.takeGradient()                                                                                                                                                 
    gradTux_o = gradTu.gradX                                                                                                                                                            
    gradTuy_o = gradTu.gradY

    grad_max = kappa.copy()

    grad_max.data[:,:] = np.maximum(np.abs(gradTux_o.data),np.abs(gradTuy_o.data))

    gradTux_o.plot(show=False,saveFig = pwd+'grad_Tx.pdf')#,valueRange=[np.min(GI.data),np.max(GI.data)])
    plt.close()

    gradTuy_o.plot(show=False,saveFig = pwd+'grad_Ty.pdf')#,valueRange=[np.min(GI.data),np.max(GI.data)])
    plt.close()

    grad_max.plot(show=False,saveFig = pwd+'grad_max.pdf')#,valueRange=[np.min(GI.data),np.max(GI.data)])
    plt.close()
