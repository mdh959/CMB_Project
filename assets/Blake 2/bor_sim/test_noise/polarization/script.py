import numpy as np
import matplotlib.pyplot as plt

noiseFactor = .01

if noiseFactor == .9999:
    fname = 'Clphi_S4_cons_N_Pol.txt'
    Cname = 'Cl_S4_cons_Pol.png'
    Nname = 'Nl_S4_cons_Pol.png'
    Ntitle = 'Polar. noise curves for noiseFactor=1. thetaFWHM=1.0'
    Ctitle = 'Polar. phi power spectra for noiseFactor=1.0 thetaFWHM=1.0'
    QEname = 'QE_S4_cons_Pol.txt'

if noiseFactor == .999:
    fname = 'Clphi_S4_N_Pol.txt'
    Cname = 'Cl_S4_Pol.png'
    Nname = 'Nl_S4_Pol.png'
    Ntitle = 'Polar. noise curves for noiseFactor=1. thetaFWHM=0.7'
    Ctitle = 'Polar. phi power spectra for noiseFactor=1.0 thetaFWHM=0.7'
    QEname = 'QE_S4_Pol.txt'

if noiseFactor == .1:
    fname = 'Clphi_0_1_N_Pol.txt'
    Cname = 'Cl_0_1_Pol.png'
    Nname = 'Nl_0_1_Pol.png'
    Ntitle = 'Polar. noise curves for noiseFactor=0.1 thetaFWHM=0.3'
    Ctitle = 'Polar. phi power spectra for noiseFactor=0.1 thetaFWHM=0.3'
    QEname = 'QE_0_1_Pol.txt'

if noiseFactor == .01:
    fname = 'Clphi_0_01_N_Pol.txt'
    Cname = 'Cl_0_01_Pol.png'
    Nname = 'Nl_0_01_Pol.png'
    Ntitle = 'Polar. noise curves for noiseFactor=0.01 thetaFWHM=0.3'
    Ctitle = 'Polar. phi power spectra for noiseFactor=0.01 thetaFWHM=0.3'
    QEname = 'QE_0_01_Pol.txt'

if noiseFactor == 1.:
    fname = 'Clphi_1_0_N_Pol.txt'
    Cname = 'Cl_1_0_Pol.png'
    Nname = 'Nl_1_0_Pol.png'
    Ntitle = 'Polar. noise curves for noiseFactor=1.0 thetaFWHM=0.3'
    Ctitle = 'Polar. phi power spectra for noiseFactor=1.0 thetaFWHM=0.3'
    QEname = 'QE_1_0_Pol.txt'

l,a,b,c,d,e,f = np.loadtxt(fname,unpack=True)
#Ls, N_QE = np.loadtxt(QEname,unpack=True)
#N_QE *= 4/Ls**4

plt.figure()
#plt.semilogy(l,e,label='lens');
plt.semilogy(l,e,label='theory');
#plt.semilogy(l,e/np.sqrt(2),label='lens/sqrt2');
#plt.semilogy(l,g,label='unlens');
plt.semilogy(l,f,label='simulation')
#plt.semilogy(Ls,N_QE,label='QE');
plt.legend(loc='best');plt.title(Ntitle)
plt.xlim([1,7000])
#plt.xlim([1,20000])
#plt.ylim([1.e-28,1.e-14])
plt.xlabel(r"$\ell$");plt.ylabel(r"$N_\ell^{\phi \phi}$")
plt.savefig(Nname)#;plt.show()#;plt.close()

plt.figure()
plt.title(Ctitle);
plt.semilogy(l,a,label='auto real');plt.semilogy(l,b,label='auto reconstr')
plt.plot(l,c,label='cross')#;plt.xlim([1,7000])
#plt.xlim([1,20000])
plt.xlim([1,7000])
plt.xlabel(r"$\ell$");plt.ylabel(r"$C_\ell^{\phi \phi}$")
plt.legend(loc='best');plt.savefig(Cname);plt.show();plt.close()    
