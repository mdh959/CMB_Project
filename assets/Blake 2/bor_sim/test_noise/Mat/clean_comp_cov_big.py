import numpy as np
import matplotlib.pyplot as plt
SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)

noiseFactor = .1#6.#.1#9999
N1 = 0
N2 = 300#150#180
N = N2-N1
dl = 2000#2000#5000 #''
if dl == '' and noiseFactor == .9999:
    n = 12
else:
    n = 40
if dl == 5000:
    n = 8
if dl == 2000:
    n = 10
if dl == '' and noiseFactor == 6.:
    n = 10


pwd = '/users/boryanah/Blake/bor_sim/test_noise/Mat/sims/test_'

if noiseFactor == .9999:
    S4 = '_S4_'
    fname = pwd+'Clphi_S4_cons_k_'
    Cname = 'CSTD_S4_cons.pdf'
    Aname = 'ASTD_S4_cons.png'
    Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=1.0'
    Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=1.0'
    
    noise = '1_1'

if noiseFactor == 6.:
    S4 = '_SO'
    fname = pwd+'Clphi_SO_k_'
    Cname = 'CSTD_SO.pdf'
    Aname = 'ASTD_SO.png'
    Atitle = 'Auto stand. dev. for noiseFactor=6.0 thetaFWHM=1.4'
    Ctitle = 'Cross stand. dev. for noiseFactor=6.0 thetaFWHM=1.4'
    
    noise = ''


if noiseFactor == .999:
    fname = pwd+'Clphi_S4_k_'
    Cname = 'CSTD_S4.png'
    Aname = 'ASTD_S4.png'
    Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=0.7'
    Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=0.7'
    
    S4 = 'S4_opt'
    noise = '1_.7'

if noiseFactor == .1:
    fname = pwd+'Clphi_0_1_k_'
    Cname = 'CSTD_0_1.png'
    Aname = 'ASTD_0_1.png'
    Atitle = 'Auto stand. dev. for noiseFactor=0.1 thetaFWHM=18 arcsec'
    Ctitle = 'Cross stand. dev. for noiseFactor=0.1 thetaFWHM=18 arcsec'
    
    output = 'STD_0_1_Mat.txt'
    S4 = '_'
    noise = str(noiseFactor)

if noiseFactor == .5:
    fname = pwd+'Clphi_0_5_k_'
    Cname = 'CSTD_0_5.png'
    Aname = 'ASTD_0_5.png'
    Atitle = 'Auto stand. dev. for noiseFactor=0.5 thetaFWHM=18 arcsec'
    Ctitle = 'Cross stand. dev. for noiseFactor=0.5 thetaFWHM=18 arcsec'
    
    output = 'STD_0_5_Mat.txt'
    S4 = '_'
    noise = str(noiseFactor)

if noiseFactor == .01:
    fname = pwd+'Clphi_0_01_k_'
    Cname = 'CSTD_0_01.png'
    Aname = 'ASTD_0_01.png'
    Atitle = 'Auto stand. dev. for noiseFactor=0.01 thetaFWHM=0.3'
    Ctitle = 'Cross stand. dev. for noiseFactor=0.01 thetaFWHM=0.3'
    
    S4 = '_'
    noise = str(noiseFactor)

if noiseFactor == 1.:
    fname = pwd+'Clphi_1_0_k_'
    Cname = 'CSTD_1_0.png'
    Aname = 'ASTD_1_0.png'
    Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=0.3'
    Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=0.3'
    
    S4 = '_'
    noise = str(noiseFactor)


Cl_pp = np.zeros((n,N))
Cl_au = np.zeros((n,N))
Cl_cr = np.zeros((n,N))

for i in range(N):
    print(fname+str(N1+i)+'_Mat'+str(dl)+'_big.txt')
    l,Cl_pp[:,i],Cl_au[:,i],Cl_cr[:,i],d,g,e,f = np.loadtxt(fname+str(N1+i)+'_Mat'+str(dl)+'_big.txt',unpack=True)
    #l,a,b,c,d,g,e,f = np.loadtxt(fname+str(10+i)+'_Mat.txt',unpack=True)

cov = np.cov(Cl_au[1:,:])
cov_cr = np.cov(Cl_cr[1:,:])
print cov.shape

np.save('test_cov_mat'+str(dl)+'_40000'+S4+noise+'_big.npy',cov)
np.save('test_cov_cross_mat'+str(dl)+'_40000'+S4+noise+'_big.npy',cov_cr)
