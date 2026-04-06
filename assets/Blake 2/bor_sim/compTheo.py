import matplotlib
matplotlib.use('Agg')                                                                                                
import numpy as np
import matplotlib.pyplot as plt
import astLib.astWCS
import astLib.astCoords
import astLib.astCalc
from flipper import *
from flipperPol import *
from lensingTools import *
from numpy.fft import fftshift,fftfreq,fft2,ifft2

# load maps                                          
dirphi = '/users/boryanah/Blake/bor_sim/'
namep = 'phi_new1.fits'
nametu = 'T_map1.fits'
nametl = 'Tlens1.fits'
#Tlens = liteMap.liteMapFromFits(direc+namel)
#Tulens = liteMap.liteMapFromFits(direc+nameu)
phi_new = liteMap.liteMapFromFits(dirphi+namep)
T_map = liteMap.liteMapFromFits(dirphi+nametu)
Tlens = liteMap.liteMapFromFits(dirphi+nametl)

print phi_new.info(), 'phi info'   

Fphi = liteMap.fftFromLiteMap(phi_new)
ps2d = fftTools.powerFromFFT(Fphi)
l, l, ll, Clphi, l, l = ps2d.binInAnnuli('BIN_200_LOG')
p2d = fftTools.powerFromLiteMap(Tlens)
l, l, ll, ClT_l, l, l = p2d.binInAnnuli('BIN_200_LOG')
p2d = fftTools.powerFromLiteMap(T_map)
l, l, ll, ClT_u, l, l = p2d.binInAnnuli('BIN_200_LOG')


scal = 'Aug6_highAcc_CDM_scalCls_new.dat'
lens = 'Aug6_highAcc_CDM_lensedCls_new.dat'


theoryPower = np.loadtxt(scal)
l=theoryPower[:,0]
cl_TT=theoryPower[:,1]
cl_EE=theoryPower[:,2]
cl_TE=theoryPower[:,3]
cl_BB=None

lMax = 20000 if 20000 < max(l) else max(l)
print lMax
#quit()

l=l[:lMax]
cl_TT=cl_TT[:lMax]*2*numpy.pi/(l*(l+1))
cl_EE=cl_EE[:lMax]*2*numpy.pi/(l*(l+1))
cl_TE=cl_TE[:lMax]*2*numpy.pi/(l*(l+1))

#for unlensed maps with lensed power                                                                                    
theoryPower_lensed = np.loadtxt(lens)
l_len=theoryPower_lensed[:,0]
cl_TT_len=theoryPower_lensed[:,1]
cl_EE_len=theoryPower_lensed[:,2]
cl_TE_len=theoryPower_lensed[:,3]
cl_BB=None

#lMax = 9000 if 9000 < max(l_len) else max(l_len)

l_len=l_len[:lMax]
cl_TT_len=cl_TT_len[:lMax]*2*np.pi/(l_len*(l_len+1))
cl_EE_len=cl_EE_len[:lMax]*2*np.pi/(l_len*(l_len+1))
cl_TE_len=cl_TE_len[:lMax]*2*np.pi/(l_len*(l_len+1))

tcmb = 2.7255e6
lphi = theoryPower[:,0]
clphi = theoryPower[:,4]/(lphi**4)/tcmb**2

plt.loglog(ll,Clphi,label='sim')
plt.loglog(lphi,clphi,label='theo')
plt.legend()                                                                                                           
plt.savefig('lala4.png')
plt.close()

plt.loglog(ll, ClT_l,label='sim')
plt.loglog(l,cl_TT_len,label='theo')
plt.legend()                                                                                                           
plt.savefig('lala5.png')
plt.close()

plt.loglog(ll, ClT_u,label='sim')
plt.loglog(l_len,cl_TT,label='theo')
plt.legend()                                                                                                           
plt.savefig('lala6.png')
plt.close()

print np.max(clphi*lphi**4/4)
quit()


# do we divide by anything tt ps and what is p
phi_new = phi.copy()
phi_new.fillWithGaussianRandomField(lphi,clphi)#,bufferFactor=1)  

T_map,Q_map,U_map =liteMapPol.simPolMapsFromEandB(phi_new,l,cl_TT,cl_EE,cl_TE,cl_BB,fullBeamMatrix=fullBeamMatrix,beam1d=beam1d_off)

T_map.writeFits('T_map.fits', overWrite=True)
phi_new.writeFits('phi_new.fits', overWrite=True)

Tlens, Qlens, Ulens  =  lensMaps(phi_new,T_map,Q_map,U_map)
Tlens.writeFits('Tlens.fits', overWrite=True)  
quit()


# to take lx and ly                                                                                                   
Fphi = liteMap.fftFromLiteMap(phi)
lx = Fphi.lx
ly = Fphi.ly

Nx = Fphi.Nx # lx i think goes from 0 to Ny                                                                           
Ny = Fphi.Ny


# gradient of unlensed temperature with killed small modes                                                             
Larr = np.arange(30000.)
filt = np.ones(len(Larr))
filt[Larr>2000.] = 0.
filt[0] = 0.
Tulens_filt = Tulens.filterFromList([Larr,filt])

#Tulens.data[:,:] = np.array([np.linspace(-250,250,2048),]*2048)                                                       
#Tlens, Qlens, Ulens  =  lensMaps(phi,Tulens,Tulens,Tulens)  


