import numpy as np
import matplotlib.pyplot as plt
SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)

noiseFactor = 6.#.9999
N1 = 0
N2 = 250#128#64#50#25
N = N2-N1
dl = ''#5000 #''
if dl == 5000:
    N_bin = 8
if dl == '' and noiseFactor == 6.:
    N_bin = 10
else:
    N_bin = 40

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
    QEname = QEdir+'experiment_boryana_1arc_1uk_2000_mean_superdumb_n0_clkk.npy'
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
    #QEname = QEdir+'experiment_boryana_0.3arc_0.1uk_2000_mean_superdumb_n0_clkk.npy'
    QEname = QEdir+'experiment_boryana_0.3arc_0.1uk_2000_mean_superdumb_n0_clkk_0.1_0.3.npy'
    output = 'STD_0_1_Mat'+str(dl)+'_big.txt'

if noiseFactor == 6.:
    fname = pwd+'test_Clphi_SO_k_'
    Cname = 'CSTD_SO_big.png'
    Aname = 'ASTD_SO_big.png'
    Atitle = 'Auto stand. dev. for noiseFactor=6.0 thetaFWHM=1.4'
    Ctitle = 'Cross stand. dev. for noiseFactor=6.0 thetaFWHM=1.4'
    QEname = QEdir+'experiment_boryana_1.4arc_6uk_2000_mean_superdumb_n0_clkk_dl1000.npy'
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

    np.savetxt("for_mat/clkk_SO_"+str(i)+".txt",np.vstack((l,a,b,c)).T)
