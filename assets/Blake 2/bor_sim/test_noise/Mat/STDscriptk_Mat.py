import numpy as np
import matplotlib.pyplot as plt
SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)

noiseFactor = .1#.9999
N1 = 0
N2 = 2#25
N = N2-N1

pwd = '/users/boryanah/Blake/bor_sim/test_noise/Mat/sims/'
# again migrating to mat's code NEW
#QEdir = '/users/boryanah/Blake/bor_sim/test_noise/QE_data/'
QEdir = '/users/boryanah/repos/alhazen/plots/'

if noiseFactor == .9999:
    fname = pwd+'test_Clphi_S4_cons_k_'
    Cname = 'CSTD_S4_cons.pdf'
    Aname = 'ASTD_S4_cons.png'
    Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=1.0'
    Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=1.0'
    #QEname = QEdir+'QE_S4_cons.txt' NEW mat
    #QEname = QEdir+'experiment_boryana_1arc_1uk_2000_mean_superdumb_n0_clkk_1_1.npy'
    QEname = QEdir+'experiment_boryana_1arc_1uk_2000_mean_superdumb_n0_clkk.npy'
    output = 'STD_S4_cons_Mat.txt'

if noiseFactor == .999:
    fname = pwd+'Clphi_S4_k_'
    Cname = 'CSTD_S4.png'
    Aname = 'ASTD_S4.png'
    Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=0.7'
    Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=0.7'
    QEname = QEdir+'QE_S4.txt'

if noiseFactor == .1:
    fname = pwd+'test_Clphi_0_1_k_'
    Cname = 'CSTD_0_1.png'
    Aname = 'ASTD_0_1.png'
    Atitle = 'Auto stand. dev. for noiseFactor=0.1 thetaFWHM=18 arcsec'
    Ctitle = 'Cross stand. dev. for noiseFactor=0.1 thetaFWHM=18 arcsec'
    #QEname = QEdir+'QE_0_1_Mat.txt' NEW mat
    #QEname = QEdir+'experiment_boryana_0.3arc_0.1uk_2000_mean_superdumb_n0_clkk_0.1_0.3.npy'
    QEname = QEdir+'experiment_boryana_0.3arc_0.1uk_2000_mean_superdumb_n0_clkk.npy'
    output = 'STD_0_1_Mat.txt'

if noiseFactor == .5:
    fname = pwd+'Clphi_0_5_k_'
    Cname = 'CSTD_0_5.png'
    Aname = 'ASTD_0_5.png'
    Atitle = 'Auto stand. dev. for noiseFactor=0.5 thetaFWHM=18 arcsec'
    Ctitle = 'Cross stand. dev. for noiseFactor=0.5 thetaFWHM=18 arcsec'
    QEname = QEdir+'QE_0_5_Mat.txt'
    output = 'STD_0_5_Mat.txt'

if noiseFactor == .01:
    fname = pwd+'Clphi_0_01_k_'
    Cname = 'CSTD_0_01.png'
    Aname = 'ASTD_0_01.png'
    Atitle = 'Auto stand. dev. for noiseFactor=0.01 thetaFWHM=0.3'
    Ctitle = 'Cross stand. dev. for noiseFactor=0.01 thetaFWHM=0.3'
    QEname = QEdir+'QE_0_01.txt'

if noiseFactor == 1.:
    fname = pwd+'Clphi_1_0_k_'
    Cname = 'CSTD_1_0.png'
    Aname = 'ASTD_1_0.png'
    Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=0.3'
    Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=0.3'
    QEname = QEdir+'QE_1_0.txt'

# NEW mat we use real n0 derived
#Ls, N_QE = np.loadtxt(QEname,unpack=True)
#N_QE *= 4/Ls**4

PMEAN = np.zeros(130)
AMEAN = np.zeros(130)
CMEAN = np.zeros(130)
CSTD = np.zeros(130)
ASTD = np.zeros(130)
for i in range(N1,N2):
    l,a,b,c,d,g,e,f = np.loadtxt(fname+str(i)+'_Mat.txt',unpack=True)
    
    # Theoretical
    PMEAN += a
    # Auto
    AMEAN += b
    # Cross
    CMEAN += c
PMEAN /= N
AMEAN /= N
CMEAN /= N

print (CMEAN-PMEAN)/PMEAN


for i in range(N):
    l,a,b,c,d,g,e,f = np.loadtxt(fname+str(i)+'_Mat.txt',unpack=True)
   
    # Auto Stand Dev
    CSTD += (c-CMEAN)**2
    # Cross Stand Dev
    #ASTD += (b-CMEAN)**2


    # SUUUUUUPER IMPORTANT CHANGE
    ASTD += (b-AMEAN)**2

#ASTD = np.sqrt(ASTD/(N-1))
ASTD = (ASTD/(N-1))
#CSTD = np.sqrt(CSTD/(N-1))
CSTD = (CSTD/(N-1))



# NEW stuff uses Mat's code this is kappa
QEarr = np.load(QEname)
Ls = QEarr[:,0]
C_r_r = QEarr[:,1]
C_r_r = np.interp(l,Ls,C_r_r)
Ls = l # these are 300 600 etc

#diff = np.diff(Ls)
#diff = np.append(diff,diff[-1])
# directly use bin file to get bin difference
diff1,diff2, diff = np.loadtxt('BIN_300.txt',unpack=True)
diff = diff2-diff1

C_t_t = np.interp(Ls,l,PMEAN)

# migrating
N_QE = C_r_r - C_t_t
C_r_r = C_t_t + N_QE
C_r_r_GI = C_t_t + g

#fsky = (1024.*.1/60)**2/41252.96
fsky = .1
diff = 300.

CSTD *= (2.91260385523/(4.*41253.)) / 0.1
ASTD *= (2.91260385523/(4.*41253.)) / 0.1

CSTD = np.sqrt(CSTD)
ASTD = np.sqrt(ASTD)

D_r_t = np.sqrt((C_t_t**2+C_t_t*C_r_r)/((2.*Ls)*diff*fsky))
D_r_r = np.sqrt(1./(Ls*diff*fsky))*C_r_r

D_r_t_GI = np.sqrt((C_t_t**2+C_t_t*C_r_r_GI)/((2.*Ls)*diff*fsky))
D_r_r_GI = np.sqrt(1./(Ls*diff*fsky))*C_r_r_GI

D_cosm = np.sqrt(1./(Ls*diff*fsky))*C_t_t

v = np.vstack((l,PMEAN,ASTD,CSTD,D_cosm,D_r_r,D_r_t,D_r_r_GI,D_r_t_GI))
np.savetxt(output,v.T)

plt.figure()
#plt.semilogy(Ls,N_QE,label='QE');
plt.semilogy(l,PMEAN,'k--',label='Signal');
plt.errorbar(l,CMEAN,yerr=CSTD,label=r'$\hat \kappa \times  \kappa$');
#plt.plot(l,AMEAN,label=r'$\hat \kappa \times \hat \kappa$');
plt.xlabel(r"$l$");plt.ylabel(r"$C_l^{\kappa \kappa}$")
plt.legend(loc='best')#;plt.title('Power with ' + Ctitle)
plt.xlim([3000,40000])
#plt.xlim([2500,19000])
plt.savefig('PS_'+Cname)

plt.figure()
plt.semilogy(Ls,D_r_r,label='Theory QE');
plt.semilogy(Ls,D_cosm,label='Cosmic Var');
plt.semilogy(l,PMEAN,'k--',label='Signal');
#plt.semilogy(l,PMEAN/(2*l+1),'k--',label='Signal');
plt.semilogy(l,ASTD,label=r'Simulation GI, $N_{\rm sim} =$ '+str(N))
plt.legend(loc='best');plt.title(Atitle)
plt.xlabel(r"$\ell$");plt.ylabel(r"$\sigma[C_\ell^{\hat \kappa \hat \kappa}]$")
plt.xlim([1,40000])
plt.savefig(Aname)

plt.figure()
plt.title(Ctitle);
plt.semilogy(l,PMEAN,'k--',label='Signal');
plt.semilogy(Ls,D_r_t,label='Theory QE');
plt.semilogy(Ls,D_cosm,label='Cosmic Var');
plt.semilogy(l,CSTD,label=r'Simulation GI, $N_{\rm sim} =$ '+str(N))
plt.xlim([1,40000])
plt.xlabel(r"$\ell$");plt.ylabel(r"$\sigma[C_\ell^{\hat \kappa \kappa}]$")
plt.legend(loc='best');plt.savefig(Cname);#plt.show();plt.close()    
