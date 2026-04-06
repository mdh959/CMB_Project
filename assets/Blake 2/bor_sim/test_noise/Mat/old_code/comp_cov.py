import numpy as np
import matplotlib.pyplot as plt
SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)

noiseFactor = 0.1
N1 = 0
N2 = 30#25
N = N2-N1

n = 130

pwd = '/users/boryanah/Blake/bor_sim/test_noise/Mat/sims/'
QEdir = '/users/boryanah/Blake/bor_sim/test_noise/QE_data/'

if noiseFactor == .9999:
    fname = pwd+'Clphi_S4_cons_k_'
    Cname = 'CSTD_S4_cons.pdf'
    Aname = 'ASTD_S4_cons.png'
    Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=1.0'
    Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=1.0'
    QEname = QEdir+'QE_S4_cons.txt'

if noiseFactor == .999:
    fname = pwd+'Clphi_S4_k_'
    Cname = 'CSTD_S4.png'
    Aname = 'ASTD_S4.png'
    Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=0.7'
    Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=0.7'
    QEname = QEdir+'QE_S4.txt'

if noiseFactor == .1:
    fname = pwd+'Clphi_0_1_k_'
    Cname = 'CSTD_0_1.png'
    Aname = 'ASTD_0_1.png'
    Atitle = 'Auto stand. dev. for noiseFactor=0.1 thetaFWHM=18 arcsec'
    Ctitle = 'Cross stand. dev. for noiseFactor=0.1 thetaFWHM=18 arcsec'
    QEname = QEdir+'QE_0_1_Mat.txt'
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


Cl_pp = np.zeros((n,N))
Cl_au = np.zeros((n,N))
Cl_cr = np.zeros((n,N))

for i in range(N):
    l,Cl_pp[:,i],Cl_au[:,i],Cl_cr[:,i],d,g,e,f = np.loadtxt(fname+str(N1+i)+'_Mat.txt',unpack=True)
    #l,a,b,c,d,g,e,f = np.loadtxt(fname+str(10+i)+'_Mat.txt',unpack=True)

cov = np.cov(Cl_au[1:,:])
print cov.shape

np.save('cov_mat_40000_'+str(noiseFactor)+'.npy',cov)
