import numpy as np
import matplotlib.pyplot as plt
SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIGGER_SIZE = 12
 
plt.rc('font', size=MEDIUM_SIZE)

noiseFactor = .9999
N1 = 60
N2 = 70
N = N2-N1

pwd = '/users/boryanah/Blake/bor_sim/test_noise/galaxy/sims/'

# loading theoretical model
cl_gg_noshot = np.loadtxt("gg_noshot_cl5.dat")
cl_kg = np.loadtxt("kg_cl1_0.dat")
cl_gg = np.loadtxt("gg_cl2_0.dat")

ls = np.arange(2,20002)

if noiseFactor == .9999:
    fname = pwd+'Clphi_S4_cons_N_'
    Cname = 'CSTD_S4_cons_kg.pdf'
    Aname = 'ASTD_S4_cons.png'
    Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=1.0'
    Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=1.0'
    QEname = 'QE_S4_cons.txt'


if noiseFactor == .999:
    fname = pwd+'Clphi_S4_N_'
    Cname = 'CSTD_S4_kg.pdf'
    Aname = 'ASTD_S4.png'
    Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=0.7'
    Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=0.7'
    QEname = 'QE_S4.txt'

if noiseFactor == .1:
    fname = pwd+'Clphi_0_1_N_'
    Cname = 'CSTD_0_1_kg.png'
    Aname = 'CSTD_0_1.png'
    Atitle = 'Auto stand. dev. for noiseFactor=0.1 thetaFWHM=0.3'
    Ctitle = 'Cross stand. dev. for noiseFactor=0.1 thetaFWHM=0.3'
    QEname = 'QE_0_1.txt'

if noiseFactor == .01:
    fname = pwd+'Clphi_0_01_N_'
    Cname = 'CSTD_0_01_kg.png'
    Aname = 'ASTD_0_01.png'
    Atitle = 'Auto stand. dev. for noiseFactor=0.01 thetaFWHM=0.3'
    Ctitle = 'Cross stand. dev. for noiseFactor=0.01 thetaFWHM=0.3'
    QEname = 'QE_0_01.txt'

if noiseFactor == 1.:
    fname = pwd+'Clphi_1_0_N_'
    Cname = 'CSTD_1_0_kg.png'
    Aname = 'ASTD_1_0.png'
    Atitle = 'Auto stand. dev. for noiseFactor=1.0 thetaFWHM=0.3'
    Ctitle = 'Cross stand. dev. for noiseFactor=1.0 thetaFWHM=0.3'
    QEname = 'QE_1_0.txt'

Ls, N_QE = np.loadtxt(QEname,unpack=True)
#N_QE *= 4/Ls**4

PMEAN = np.zeros(45)
AMEAN = np.zeros(45)
CMEAN = np.zeros(45)
CSTD = np.zeros(45)
ASTD = np.zeros(45)

for i in range(N1,N2):
    l,a,b,c,d,e,f,g = np.loadtxt(fname+str(i)+'.txt',unpack=True)
    PMEAN += a*l**4/4.
    #CMEAN += d*l**4./4.
    CMEAN += d
    AMEAN += c*l**4./4.
PMEAN /= N
CMEAN /= N 
AMEAN /= N 
Cl_kg = np.interp(l,ls,cl_kg)

print "(CMEAN-PMEAN)/PMEAN"
print (CMEAN -PMEAN)/PMEAN
print "(AMEAN -PMEAN)/PMEAN"
print (AMEAN-PMEAN)/PMEAN
print (Cl_kg-CMEAN)/Cl_kg

for i in range(N1,N2):
    l,a,b,c,d,e,f,g = np.loadtxt(fname+str(i)+'.txt',unpack=True)
    CSTD += (d-Cl_kg)**2
    ASTD += (c*l**4./4.-AMEAN)**2

CSTD = np.sqrt(CSTD/(N-1))
ASTD = np.sqrt(ASTD/(N-1))

N_QE = np.interp(l,Ls,N_QE)
Ls = l

#diff = np.diff(Ls)
#diff = np.append(diff,diff[-1])
diff1,diff2, diff = np.loadtxt('BIN_200.txt',unpack=True)
diff = diff2-diff1
C_t_t = np.interp(Ls,l,PMEAN)
C_k_g = np.interp(Ls,ls,cl_kg)
C_g_g = np.interp(Ls,ls,cl_gg)
C_r_r = C_t_t + N_QE
fsky = 1./41252.96

# rescaling
fsky = .5
CSTD /= np.sqrt(41252.96/2)

D_r_t = np.sqrt((C_k_g**2+C_r_r*C_g_g)/((2.*Ls+1)*diff*fsky))
D_r_t_cross = np.sqrt((C_t_t**2+C_t_t*C_r_r)/((2.*Ls+1)*diff*fsky))

m = 23
print Ls[m]

SN_GI = np.sqrt(np.sum((Cl_kg[m:]/CSTD[m:])**2))
SN_QE = np.sqrt(np.sum((C_k_g[m:]/D_r_t[m:])**2))
print "SN_GI, SN_QE"
print SN_GI, SN_QE

plt.figure()
plt.semilogy(l,PMEAN,label=r'$ \kappa \times \kappa$');
plt.plot(l,AMEAN,label=r'$\hat \kappa \times \kappa$');
plt.errorbar(l,CMEAN,fmt='s',yerr=CSTD,label=r'$\hat \kappa \times g$');
plt.plot(ls,cl_kg,label=r'$\kappa \times g$');
plt.xlabel(r"$l$");plt.ylabel(r"$C_l^{\psi \psi}$")
plt.legend(loc='best')#;plt.title('Power with ' + Ctitle)
plt.xlim([1,20000])
#plt.xlim([2500,19000])
plt.savefig('PS_'+Cname)

'''
plt.figure()
plt.semilogy(Ls,D_r_r,label='Theory QE');
plt.semilogy(l,ASTD,label=r'Simulation GI, $N_{\rm sim} =$ '+str(N))
plt.legend(loc='best')#;plt.title(Atitle)
plt.xlim([1,20000])
plt.savefig(Aname)
'''

plt.figure()
plt.semilogy(Ls,D_r_t_cross*4./Ls**4,label='Theory QE');
plt.semilogy(l,ASTD*4./l**4,label=r'Simulation GI, $N_{\rm sim} =$ '+str(N))
plt.xlim([1,20000])
plt.legend(loc='best');plt.savefig(Aname);#plt.show();plt.close()    


plt.figure()
plt.semilogy(Ls,D_r_t,label='Theory QE');
plt.semilogy(l,CSTD,label=r'Simulation GI, $N_{\rm sim} =$ '+str(N))
plt.xlim([1,20000])
plt.legend(loc='best');plt.savefig(Cname);#plt.show();plt.close()    
