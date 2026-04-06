import numpy as np
import matplotlib.pyplot as plt
SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)

noiseFactor = .9999
N1 = 0
N2 = 10
N = N2-N1

pwd = '/users/boryanah/Blake/bor_sim/test_noise/sims/'

if noiseFactor == .9999:
    fname = pwd+'Clphi_S4_cons_W_'
    Cname = 'CSTD_S4_cons_W.png'
    Aname = 'ASTD_S4_cons_W.png'
    Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=1.0'
    Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=1.0'
    QEname = 'QE_S4_cons.txt'

if noiseFactor == .999:
    fname = pwd+'Clphi_S4_W_'
    Cname = 'CSTD_S4_W.png'
    Aname = 'ASTD_S4_W.png'
    Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=0.7'
    Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=0.7'
    QEname = 'QE_S4.txt'

if noiseFactor == .01:
    fname = pwd+'Clphi_0_01_W_'
    Cname = 'CSTD_0_01_W.png'
    Aname = 'ASTD_0_01_W.png'
    Atitle = 'Auto stand. dev. for noiseFactor=0.01 thetaFWHM=0.3'
    Ctitle = 'Cross stand. dev. for noiseFactor=0.01 thetaFWHM=0.3'
    QEname = 'QE_0_01.txt'

if noiseFactor == 1.:
    fname = pwd+'Clphi_1_0_W_'
    Cname = 'CSTD_1_0_W.png'
    Aname = 'ASTD_1_0_W.png'
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

print (CMEAN-PMEAN)/PMEAN


for i in range(N):
    l,a,b,c,d,g,e,f = np.loadtxt(fname+str(i)+'.txt',unpack=True)
    #Clphi_est_N[0] = 1.
    CSTD += (c-CMEAN)**2
    ASTD += (b-CMEAN)**2

#ASTD = np.sqrt(ASTD/(N-1))
ASTD = np.sqrt(ASTD/(N))
#CSTD = np.sqrt(CSTD/(N-1))
CSTD = np.sqrt(CSTD/(N))

N_QE = np.interp(l,Ls,N_QE)
Ls = l

#diff = np.diff(Ls)
# I think this is now correct
diff = np.diff(Ls)
diff = np.append(diff,diff[-1])
C_t_t = np.interp(Ls,l,PMEAN)

#N_QE = AMEAN - PMEAN

C_r_r = C_t_t + N_QE

fsky = 1./41252.96
factor = .5/fsky
fsky = .5#/fsky

CSTD /= np.sqrt(factor)
ASTD /= np.sqrt(factor)

D_r_t = np.sqrt((C_t_t**2+C_t_t*C_r_r)/((2.*Ls)*diff*fsky))
D_r_r = np.sqrt(1./(Ls*diff*fsky))*C_r_r

plt.figure()
plt.semilogy(l,PMEAN,'k--',label='Signal');
#plt.semilogy(l,PMEAN,label='Signal');
plt.errorbar(l,CMEAN,yerr=CSTD,label=r'$\hat \psi \times  \psi$');
plt.plot(l,AMEAN,label=r'$\hat \psi \times \hat \psi$');
plt.xlabel(r"$l$");plt.ylabel(r"$C_l^{\psi \psi}$")
plt.legend(loc='best')#;plt.title('Power with ' + Ctitle)
plt.xlim([1,20000])
#plt.xlim([2500,19000])
plt.savefig('PS_'+Cname)

plt.figure()
plt.semilogy(Ls,D_r_r,label='Theory QE');
plt.semilogy(l,PMEAN,'k--',label='Signal');
plt.semilogy(l,ASTD,label=r'Simulation GI, $N_{\rm sim} =$ '+str(N))
plt.legend(loc='best');plt.title(Atitle)
plt.xlabel(r"$\ell$");plt.ylabel(r"$\sigma[C_\ell^{\hat \psi \hat \psi}]$")
plt.xlim([1,20000])
plt.savefig(Aname)

plt.figure()
plt.title(Ctitle);
plt.semilogy(l,PMEAN,'k--',label='Signal');
plt.semilogy(Ls,D_r_r,label='Theory QE');
plt.semilogy(l,CSTD,label=r'Simulation GI, $N_{\rm sim} =$ '+str(N))
plt.xlim([1,20000])
plt.xlabel(r"$\ell$");plt.ylabel(r"$\sigma[C_\ell^{\hat \psi \psi}]$")
plt.legend(loc='best');plt.savefig(Cname);#plt.show();plt.close()    
