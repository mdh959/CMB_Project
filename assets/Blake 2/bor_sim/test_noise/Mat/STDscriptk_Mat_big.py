import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)

# noise dl
noiseFactor = .1#6.#.1#.9999
N1 = 0
N2 = 360#180#150
N = N2-N1
dl = 2000#2000 #5000 #''#2000
if dl == '' and noiseFactor == .9999:
    N_bin = 12
else:
    N_bin = 40
if dl == '':
    dll = 1000
if dl == 2000:
    N_bin = 10
    dll = 2000
if dl == 5000:
    N_bin = 8
    dll = 5000
if dl == '' and noiseFactor == 6.:
    N_bin = 10

pwd = '/users/boryanah/Blake/bor_sim/test_noise/Mat/sims/'
# again migrating to mat's code NEW
#QEdir = '/users/boryanah/Blake/bor_sim/test_noise/QE_data/'
QEdir = '/users/boryanah/repos/alhazen/plots/'

if noiseFactor == .9999:
    fname = pwd+'test_Clphi_S4_cons_k_'
    Cname = 'CSTD_S4_cons_big.png'
    Aname = 'ASTD_S4_cons_big.png'
    Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=1.0'
    Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=1.0'
    #QEname = QEdir+'QE_S4_cons.txt' NEW mat
    QEname = QEdir+'experiment_boryana_1arc_1uk_2000_mean_superdumb_n0_clkk_dl1000.npy'
    output = 'STD_S4_cons_Mat'+str(dl)+'_big.txt'

if noiseFactor == .999:
    fname = pwd+'Clphi_S4_k_'
    Cname = 'CSTD_S4.png'
    Aname = 'ASTD_S4.png'
    Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=0.7'
    Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=0.7'
    QEname = QEdir+'QE_S4.txt'

if noiseFactor == .1:
    fname = pwd+'test_Clphi_0_1_k_'
    Cname = 'CSTD_0_1_big.png'
    Aname = 'ASTD_0_1_big.png'
    Atitle = 'Auto stand. dev. for noiseFactor=0.1 thetaFWHM=18 arcsec'
    Ctitle = 'Cross stand. dev. for noiseFactor=0.1 thetaFWHM=18 arcsec'
    #QEname = QEdir+'QE_0_1_Mat.txt' NEW mat
    QEname = QEdir+'experiment_boryana_0.3arc_0.1uk_2000_mean_superdumb_n0_clkk_dl2000.npy'
    #QEname = QEdir+'experiment_boryana_0.3arc_0.1uk_2000_mean_superdumb_n0_clkk_0.1_0.3.npy'
    output = 'STD_0_1_Mat'+str(dl)+'_big.txt'

if noiseFactor == 6.:
    fname = pwd+'test_Clphi_SO_k_'
    Cname = 'CSTD_SO_big.png'
    Aname = 'ASTD_SO_big.png'
    Atitle = 'Auto stand. dev. for noiseFactor=6.0 thetaFWHM=1.4'
    Ctitle = 'Cross stand. dev. for noiseFactor=6.0 thetaFWHM=1.4'
    QEname = QEdir+'experiment_boryana_1.4arc_6uk_2000_mean_superdumb_n0_clkk_dl'+str(dll)+'.npy'
    output = 'STD_SO_Mat'+str(dl)+'_big.txt'

if noiseFactor == .5:
    fname = pwd+'Clphi_0_5_k_'
    Cname = 'CSTD_0_5.png'
    Aname = 'ASTD_0_5.png'
    Atitle = 'Auto stand. dev. for noiseFactor=0.5 thetaFWHM=18 arcsec'
    Ctitle = 'Cross stand. dev. for noiseFactor=0.5 thetaFWHM=18 arcsec'
    QEname = QEdir+'QE_0_5_Mat.txt'
    output = 'STD_0_5_Mat_big.txt'

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

PMEAN = np.zeros(N_bin)
AMEAN = np.zeros(N_bin)
CMEAN = np.zeros(N_bin)
CSTD = np.zeros(N_bin)
ASTD = np.zeros(N_bin)
N_GI = np.zeros(N_bin)
for i in range(N1,N2):
    l,a,b,c,d,g,e,f = np.loadtxt(fname+str(i)+'_Mat'+str(dl)+'_big.txt',unpack=True)
    
    # Theoretical
    PMEAN += a
    # Auto
    AMEAN += b
    # Cross
    CMEAN += c
    # GI noise
    N_GI += g
PMEAN /= N
AMEAN /= N
CMEAN /= N
N_GI /= N

print (CMEAN-PMEAN)/PMEAN


for i in range(N):
    l,a,b,c,d,g,e,f = np.loadtxt(fname+str(i)+'_Mat'+str(dl)+'_big.txt',unpack=True)
   
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
Ls = l # these are 1000 2000 etc

#diff = np.diff(Ls)
#diff = np.append(diff,diff[-1])
# directly use bin file to get bin difference
#diff1,diff2, diff = np.loadtxt('BIN_'+str(dll)+'.txt',unpack=True)
diff = dll*1.

C_t_t = np.interp(Ls,l,PMEAN)

# migrating
N_QE = C_r_r - C_t_t
C_r_r = C_t_t + N_QE
# IMPORTANT
C_r_r_GI = C_t_t + 2*N_GI
#C_r_r_GI = C_t_t + N_GI

#fsky = (1024.*.1/60)**2/41252.96
fsky = .4
#diff = 1000.

CSTD *= (2.91260385523/(41253.)) / 0.4
ASTD *= (2.91260385523/(41253.)) / 0.4

CSTD = np.sqrt(CSTD)
ASTD = np.sqrt(ASTD)

# IMPORTANT
D_r_t = np.sqrt((2*C_t_t**2+C_t_t*C_r_r)/((2.*Ls)*diff*fsky))
D_r_r = np.sqrt(1./(Ls*diff*fsky))*C_r_r

D_r_t_GI = np.sqrt((2*C_t_t**2+C_t_t*C_r_r_GI)/((2.*Ls)*diff*fsky))
D_r_r_GI = np.sqrt(1./(Ls*diff*fsky))*C_r_r_GI

D_cosm = np.sqrt(1./(Ls*diff*fsky))*C_t_t

v = np.vstack((l,PMEAN,ASTD,CSTD,D_cosm,D_r_r,D_r_t,D_r_r_GI,D_r_t_GI))
np.savetxt(output,v.T)
#quit()

#plt.semilogy(Ls,N_QE,label='QE');
plt.semilogy(l,PMEAN,'k--',label='Signal');
plt.errorbar(l,CMEAN,yerr=CSTD,label=r'$\hat \kappa \times  \kappa$');
plt.plot(l,AMEAN,label=r'$\hat \kappa \times \hat \kappa$');
plt.xlabel(r"$l$");plt.ylabel(r"$C_l^{\kappa \kappa}$")
plt.legend(loc='best')#;plt.title('Power with ' + Ctitle)
plt.xlim([2000,N_bin*dll])


plt.savefig('PS_'+Cname)
plt.close()

plt.semilogy(Ls,D_r_r,label='Theory QE');
plt.semilogy(Ls,D_cosm,label='Cosmic Var');
plt.semilogy(l,PMEAN,'k--',label='Signal');
#plt.semilogy(l,PMEAN/(2*l+1),'k--',label='Signal');
plt.semilogy(l,ASTD,label=r'Simulation GI, $N_{\rm sim} =$ '+str(N))
plt.legend(loc='best');plt.title(Atitle)
plt.xlabel(r"$\ell$");plt.ylabel(r"$\sigma[C_\ell^{\hat \kappa \hat \kappa}]$")
plt.xlim([2000,N_bin*dll])
plt.savefig(Aname)
plt.close()

plt.title(Ctitle);
plt.semilogy(l,PMEAN,'k--',label='Signal');
plt.semilogy(Ls,D_r_t,label='Theory QE');
plt.semilogy(Ls,D_cosm,label='Cosmic Var');
plt.semilogy(l,CSTD,label=r'Simulation GI, $N_{\rm sim} =$ '+str(N))
plt.xlim([2000,N_bin*dll])
plt.xlabel(r"$\ell$");plt.ylabel(r"$\sigma[C_\ell^{\hat \kappa \kappa}]$")
plt.legend(loc='best');plt.savefig(Cname);#plt.show();plt.close()    
plt.close()
