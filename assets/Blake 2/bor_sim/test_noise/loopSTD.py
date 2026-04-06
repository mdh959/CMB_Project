import numpy as np
import matplotlib.pyplot as plt

noiseFactor = .999
N1 = 0
N2 = 25
N = N2-N1

pwd = '/users/boryanah/Blake/bor_sim/test_noise/sims/'

noiseFactors = [.9999, .999, 1., .1, .01]
#noiseFactors = [ 1., .1, .01]

for noiseFactor in noiseFactors:

    if noiseFactor == .9999:
        fname = pwd+'Clphi_S4_cons_N_'
        Cname = 'CSTD_S4_cons.png'
        Aname = 'ASTD_S4_cons.png'
        Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=1.0'
        Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=1.0'
        QEname = 'QE_S4_cons.txt'

    if noiseFactor == .999:
        fname = pwd+'Clphi_S4_N_'
        Cname = 'CSTD_S4.png'
        Aname = 'ASTD_S4.png'
        Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=0.7'
        Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=0.7'
        QEname = 'QE_S4.txt'

    if noiseFactor == .1:
        fname = pwd+'Clphi_0_1_N_'
        Cname = 'CSTD_0_1.png'
        Aname = 'CSTD_0_1.png'
        Atitle = 'Auto stand. dev. for noiseFactor=0.1 thetaFWHM=0.3'
        Ctitle = 'Cross stand. dev. for noiseFactor=0.1 thetaFWHM=0.3'
        QEname = 'QE_0_1.txt'

    if noiseFactor == .01:
        fname = pwd+'Clphi_0_01_N_'
        Cname = 'CSTD_0_01.png'
        Aname = 'ASTD_0_01.png'
        Atitle = 'Auto stand. dev. for noiseFactor=0.01 thetaFWHM=0.3'
        Ctitle = 'Cross stand. dev. for noiseFactor=0.01 thetaFWHM=0.3'
        QEname = 'QE_0_01.txt'

    if noiseFactor == 1.:
        fname = pwd+'Clphi_1_0_N_'
        Cname = 'CSTD_1_0.png'
        Aname = 'ASTD_1_0.png'
        Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=0.3'
        Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=0.3'
        QEname = 'QE_1_0.txt'

    Ls, N_QE = np.loadtxt(QEname,unpack=True)
    N_QE *= 4/Ls**4

    PMEAN = np.zeros(45)
    AMEAN = np.zeros(45)
    CMEAN = np.zeros(45)
    CSTD = np.zeros(45)
    ASTD = np.zeros(45)
    for i in range(N1,N2):
        l,a,b,c,d,g,e,f = np.loadtxt(fname+str(i)+'.txt',unpack=True)
    #Clphi_est_N[0] = 1.
        PMEAN += a
        AMEAN += b
        CMEAN += c
    PMEAN /= N
    AMEAN /= N
    CMEAN /= N

#    print (CMEAN-PMEAN)/PMEAN

    for i in range(N):
        l,a,b,c,d,g,e,f = np.loadtxt(fname+str(i)+'.txt',unpack=True)
    #Clphi_est_N[0] = 1.
        CSTD += (c-CMEAN)**2
        ASTD += (b-CMEAN)**2

    ASTD = np.sqrt(ASTD/(N-1))
    CSTD = np.sqrt(CSTD/(N-1))

    diff = np.diff(Ls)
    diff = np.append(diff,diff[-1])
    C_t_t = np.interp(Ls,l,PMEAN)
    C_r_r = C_t_t + N_QE
    fsky = 1./41252.96
    D_r_t = np.sqrt((C_t_t**2+C_t_t*C_r_r)/((2.*Ls)*diff*fsky))
    D_r_r = np.sqrt(1./(Ls*diff*fsky))*C_r_r

    plt.figure()
#plt.semilogy(l,PMEAN,label='real');
#plt.errorbar(l,CMEAN,yerr=CSTD,label='cross');
#plt.errorbar(l,AMEAN,yerr=ASTD,label='auto');
    plt.semilogy(Ls,D_r_r,label='QE theory');
    plt.semilogy(l,ASTD,label='GI simulation, N = '+str(N))
    plt.legend(loc='best');plt.title(Atitle)
    plt.xlim([1,20000])
    plt.xlabel(r"$\ell$");plt.ylabel(r"$\sigma[C_\ell^{\hat \phi \hat \phi}]$")
    plt.savefig(Aname)

    plt.figure()
    plt.title(Ctitle);
    plt.semilogy(Ls,D_r_r,label='QE theory');
    plt.semilogy(l,CSTD,label='GI simulation, N = '+str(N))
    plt.xlim([1,20000])
    plt.legend(loc='best');plt.savefig(Cname)
    plt.xlabel(r"$\ell$");plt.ylabel(r"$\sigma[C_\ell^{\hat \phi \phi}]$")
    plt.close()
