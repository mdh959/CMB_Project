import numpy as np
import matplotlib.pyplot as plt
SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE) 

noiseFactor = .01#9999

if noiseFactor == .9999:
    fname = 'Clphi_S4_cons_N.txt'
    Cname = 'Cl_S4_cons.png'
    Nname = 'Nl_S4_cons.pdf'
    Ntitle = "Noise curves for noise level of 1 uK-arcmin and beam of 1 arcmin (CMB-S4)"#r"Noise curves for noiseFactor=1$\mu$K-arcmin$ $\theta_{\rm FWHM}=1$arcmin"
    Ctitle = 'Power spectra for noiseFactor=1.0 thetaFWHM=1.0'
    QEname = 'QE_S4_cons.txt'

if noiseFactor == .999:
    fname = 'Clphi_S4_N.txt'
    Cname = 'Cl_S4.png'
    Nname = 'Nl_S4.png'
    Ntitle = 'Noise curves for noiseFactor=1. thetaFWHM=0.7'
    Ctitle = 'Power spectra for noiseFactor=1.0 thetaFWHM=0.7'
    QEname = 'QE_S4.txt'

if noiseFactor == .1:
    fname = 'Clphi_0_1_N.txt'
    Cname = 'Cl_0_1.pdf'
    Nname = 'Nl_0_1.pdf'
    Ntitle = "Noise curves for noise level of 0.1 uK-arcmin and beam of 0.3 arcmin"#'Noise curves for noiseFactor=0.1 thetaFWHM=0.3'
    Ctitle = 'Power spectra for noiseFactor=0.1 thetaFWHM=0.3'
    QEname = 'QE_0_1.txt'

if noiseFactor == .01:
    fname = 'Clphi_0_01_N.txt'
    Cname = 'Cl_0_01.png'
    Nname = 'Nl_0_01.png'
    Ntitle = 'Noise curves for noiseFactor=0.01 thetaFWHM=0.3'
    Ctitle = 'Power spectra for noiseFactor=0.01 thetaFWHM=0.3'
    QEname = 'QE_0_01.txt'

if noiseFactor == .0001:
    fname = 'Clphi_0_0001_N.txt'
    Cname = 'Cl_0_0001.png'
    Nname = 'Nl_0_0001.png'
    Ntitle = 'Noise curves for noiseFactor=0.0001 thetaFWHM=0.3'
    Ctitle = 'Power spectra for noiseFactor=0.0001 thetaFWHM=0.3'
    QEname = 'QE_0001.txt'

if noiseFactor == 1.:
    fname = 'Clphi_1_0_N.txt'
    Cname = 'Cl_1_0.png'
    Nname = 'Nl_1_0.png'
    Ntitle = 'Noise curves for noiseFactor=1.0 thetaFWHM=0.3'
    Ctitle = 'Power spectra for noiseFactor=1.0 thetaFWHM=0.3'
    QEname = 'QE_1_0.txt'

l,a,b,c,d,g,e,f = np.loadtxt(fname,unpack=True)
Ls, N_QE = np.loadtxt(QEname,unpack=True)
#ls, n_QE = np.load("nl_TT_1_1.npy")
ls, n_QE = np.load("nl_TT_03_01.npy")
N_QE *= 4/Ls**4

f[-1] = g[-1]

for i in range(45):
    print l[i],f[i]/g[i]

plt.figure()
#plt.semilogy(l,e,label='lens');
plt.plot(l,a,'k--',label='Signal')
plt.semilogy(l,g,label='Theory GI');
#plt.semilogy(l,e/np.sqrt(2),label='lens/sqrt2');
#plt.semilogy(l,g,label='unlens');
plt.semilogy(l,f,label='Simulation GI')
plt.semilogy(Ls,N_QE,label='Theory QE');plt.legend(loc='best')#;plt.title(Ntitle)
plt.semilogy(ls,n_QE,label='theory QE');plt.legend(loc='best')#;plt.title(Ntitle)
plt.xlim([1,20000])
plt.xlabel(r"$l$");plt.ylabel(r"$N_l^{\psi \psi}$")
plt.savefig(Nname)#;plt.show()#;plt.close()

plt.figure()
N_QE_l = np.interp(l,Ls,N_QE)
#plt.semilogy(l,e,label='lens');
#plt.plot(l,N_QE_l/g,label='theory - QE/QE');
plt.plot(l,N_QE_l/g,label='QE/theory');
#plt.semilogy(l,e/np.sqrt(2),label='lens/sqrt2');
#plt.semilogy(l,g,label='unlens');
#plt.plot(l,a,'k--',label='signal')
#plt.plot(l,(f-N_QE_l)/N_QE_l,label='sim - QE/QE')
plt.plot(l,N_QE_l/f,label='QE/sim')
plt.plot(Ls,N_QE/N_QE,'k--')#,label='QE and QE')
plt.legend(loc='best');plt.title("Ratio of " + Ntitle)
plt.xlim([5000,20000])
plt.ylim([0,10])
plt.xlabel(r"$l$");plt.ylabel("Ratio")#r"$\Delta N_l^{\psi \psi} / N_l^{\psi \psi}$")
plt.savefig("rat_" + Nname)#;plt.show()#;plt.close()

plt.figure()
#plt.title(Ctitle);
plt.semilogy(l,a,label='Signal');plt.semilogy(l,b,label=r'$\hat \psi \times \hat \psi$')
plt.plot(l,c,label=r'$\hat \psi \times \psi$')
plt.xlabel(r"$l$");plt.ylabel(r"$C_l^{\psi \psi}$")
plt.xlim([1,20000])
plt.legend(loc='best');plt.savefig(Cname)#;plt.show();plt.close()    
