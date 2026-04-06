import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams['lines.linewidth'] = 4.
mpl.rcParams.update({'font.family':'serif'})   
mpl.rcParams.update({'font.size': 18})

SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIGGER_SIZE = 12

#plt.rc('font', size=MEDIUM_SIZE) 

i = 20#31 s4#11 0.1#19
big = '_big'
#big = ''


noiseFactor = .1#6.#.1#.9999


pwd  = '/users/boryanah/Blake/bor_sim/test_noise/Mat'
rootf = pwd+'/sims/'
roots = pwd+'/plots/'
QEdir = '/users/boryanah/repos/alhazen/plots/'

if noiseFactor == .9999:
    fname = rootf+'test_Clphi_S4_cons_k_'+str(i)+'_Mat'+big+'.txt'
    Cname = roots+'Cl_S4_cons.pdf'
    Nname = roots+'Nl_S4_cons.pdf'
    Rname = roots+'rat_S4_cons.pdf'
    Ntitle = "Noise curves for noise level of 1 uK-arcmin and beam of 1 arcmin (CMB-S4)"
    Ntitle = r"Noise curves for noiseFactor=1$\mu$K-arcmin$ $\theta_{\rm FWHM}=1$arcmin"
    Ctitle = 'Power spectra for noiseFactor=1.0 thetaFWHM=1.0'
    QEname = QEdir+'experiment_boryana_1arc_1uk_2000_mean_superdumb_n0_clkk.npy'
    #QEname = '../QE_data/QE_S4_cons.txt'

if noiseFactor == .999:
    fname = 'Clphi_S4_k.txt'
    Cname = 'Cl_S4.pdf'
    Nname = 'Nl_S4.pdf'
    Ntitle = 'Noise curves for noiseFactor=1. thetaFWHM=0.7'
    Ctitle = 'Power spectra for noiseFactor=1.0 thetaFWHM=0.7'
    QEname = 'QE_S4.txt'

if noiseFactor == .1:
    #fname = rootf+'test_Clphi_0_1_k_'+str(i)+'_Mat'+big+'.txt'
    fname = rootf+'test_Clphi_0_1_k_'+str(i)+'_Mat2000'+big+'.txt'
    # !!!!!!!!!!!!!!!!!!
    ll_the, Clkk_the = np.loadtxt('ells_clkk.txt',unpack=True)
    fname = 'Clphi_0_1_k_100_Mat'+big+'.txt'
    Cname = roots+'Cl_0_1.pdf'
    Nname = roots+'Nl_0_1.pdf'
    Rname = roots+'rat_0_1.pdf'
    Ntitle = "Noise curves for noise level of 0.1 uK-arcmin and beam of 0.3 arcmin"
    Ctitle = 'Power spectra for noiseFactor=0.1 thetaFWHM=0.3'
    QEname = QEdir+'experiment_boryana_0.3arc_0.1uk_2000_mean_superdumb_n0_clkk.npy'
    #QEname = '../QE_data/QE_0_1.txt'

if noiseFactor == 6.:
    fname = rootf+'test_Clphi_SO_k_'+str(i)+'_Mat'+big+'.txt'
    Cname = roots+'Cl_SO.png'
    Nname = roots+'Nl_SO.png'
    Rname = roots+'rat_SO.png'
    Ntitle = "Noise curves for noise level of 6. uK-arcmin and beam of 1.4 arcmin"
    Ctitle = 'Power spectra for noiseFactor=6. thetaFWHM=1.4'
    QEname = QEdir+'experiment_boryana_1.4arc_6uk_2000_mean_superdumb_n0_clkk_dl1000.npy'
    #QEname = '../QE_data/QE_0_1.txt'

if noiseFactor == .01:
    fname = 'Clphi_0_01_k.txt'
    Cname = 'Cl_0_01.pdf'
    Nname = 'Nl_0_01.pdf'
    Ntitle = 'Noise curves for noiseFactor=0.01 thetaFWHM=0.3'
    Ctitle = 'Power spectra for noiseFactor=0.01 thetaFWHM=0.3'
    QEname = 'QE_0_01.txt'

if noiseFactor == .0001:
    fname = 'Clphi_0_0001_k.txt'
    Cname = 'Cl_0_0001.pdf'
    Nname = 'Nl_0_0001.pdf'
    Ntitle = 'Noise curves for noiseFactor=0.0001 thetaFWHM=0.3'
    Ctitle = 'Power spectra for noiseFactor=0.0001 thetaFWHM=0.3'
    QEname = 'QE_0001.txt'

if noiseFactor == 1.:
    fname = 'Clphi_1_0_k.txt'
    Cname = 'Cl_1_0.pdf'
    Nname = 'Nl_1_0.pdf'
    Ntitle = 'Noise curves for noiseFactor=1.0 thetaFWHM=0.3'
    Ctitle = 'Power spectra for noiseFactor=1.0 thetaFWHM=0.3'
    QEname = 'QE_1_0.txt'

l,a,b,c,d,g,e,f = np.loadtxt(fname,unpack=True)
# NEW stuff uses Mat's code this is kappa
QEarr = np.load(QEname)
#QEarr = np.loadtxt(QEname)
Ls = QEarr[:,0]
N_QE = QEarr[:,1]

#ls, n_QE = np.load("nl_TT_1_1.npy")
#ls, n_QE = np.load("nl_TT_03_01.npy")
#N_QE *= 4/Ls**4

#f[-1] = g[-1]
l = l[:-1]
f = f[:-1]
g = g[:-1]
a = a[:-1]
b = b[:-1]
c = c[:-1]
e = e[:-1]

for i in range(len(f)):
    print l[i],f[i]/g[i]


fig = plt.figure(figsize=(9,7))

plt.plot(l,a,'k--',label='Signal')
plt.semilogy(l,g,label='GI theory');
plt.semilogy(l,f,label='GI sim')
plt.semilogy(Ls[:-1],N_QE[:-1],label='QE theory');plt.legend(loc='best')
if noiseFactor == 0.9999  or noiseFactor == 6.:
    plt.xlim([2000,10000])
    #plt.xlim([500,40000])
    #plt.ylim([1.e-12,1.e-6])
else:
    plt.xlim([3000,40000])
    #plt.xlim([500,40000])
    plt.ylim([1.e-13,1.e-8])
plt.xlabel(r"$L$");plt.ylabel(r"$N_L^{\kappa \kappa}$")
plt.savefig(Nname)

fig = plt.figure(figsize=(9,7))
N_QE_l = np.interp(l,Ls,N_QE)
plt.plot(l,N_QE_l/f)
plt.plot(Ls,N_QE/N_QE,'k--')
if noiseFactor == 0.9999   or noiseFactor == 6.:
    plt.xlim([2000,10000])
    #plt.xlim([500,15000])
    plt.ylim([0.1,1000])
else:
    plt.xlim([4000,40000])
    #plt.xlim([500,40000])
    plt.ylim([0.1,100])
plt.yscale('log')
plt.xlabel(r"$L$");
plt.ylabel(r'$N^{\kappa \kappa}_{L , {\rm QE}}/N_{L , {\rm GI}}^{\kappa \kappa}$')
plt.savefig(Rname)

fig = plt.figure(figsize=(9,7))
#plt.title(Ctitle);
plt.plot(ll_the,Clkk_the,'k',label='Theory')
plt.semilogy(l,a,'k--',label='Signal')
plt.semilogy(l,b,color="limegreen",ls='-',label=r'$\hat \kappa \times \hat \kappa$')
plt.plot(l,c,color="limegreen",ls='-.',label=r'$\hat \kappa \times \kappa$')
plt.xlabel(r"$L$");plt.ylabel(r"$C_L^{\kappa \kappa}$")
if noiseFactor == 0.9999   or noiseFactor == 6.:
    plt.xlim([3000,10000])
    #plt.xlim([500,40000])
    plt.ylim([1.e-12,1.e-8])
else:
    plt.xlim([4000,40000])
    #plt.xlim([500,40000])
    plt.ylim([1.e-12,1.e-8])
plt.legend(loc='best');plt.savefig(Cname)

#plt.show()
