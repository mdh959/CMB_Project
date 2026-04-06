import matplotlib
#matplotlib.use('Agg')                                                                                                
import numpy as np
import matplotlib.pyplot as plt
import astLib.astWCS
import astLib.astCoords
import astLib.astCalc
from flipper import *
from flipperPol import *
from lensingTools import *
from numpy.fft import fftshift,fftfreq,fft2,ifft2
from mpi4py import MPI

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

nametu = '/users/boryanah/Blake/bor_sim/many/sims/T_map_'+str(myrank)+'.fits'
nametl = '/users/boryanah/Blake/bor_sim/many/sims/Tlens_'+str(myrank)+'.fits'
nametln = '/users/boryanah/Blake/bor_sim/many/sims/Tlens_noise_0_01_'+str(myrank)+'.fits'
namepn = '/users/boryanah/Blake/bor_sim/many/sims/phi_new_'+str(myrank)+'.fits'

# parameters for noise estimation
thetaFWHMarcmin = 0.3
noiseUkArcmin = 1.
thetaFWHM = thetaFWHMarcmin*numpy.pi/(180.*60.)
deltaT = noiseUkArcmin/thetaFWHMarcmin
noiseFactor = .01

beam1d_off = {'apply':False}

fullBeamMatrix = {'apply':False}
# load maps                                          
dirphi = '/users/boryanah/Blake/bor_sim/'
namep = 'phi.fits'
#Tlens = liteMap.liteMapFromFits(direc+namel)
#Tulens = liteMap.liteMapFromFits(direc+nameu)
phi = liteMap.liteMapFromFits(dirphi+namep)
print phi.info(), 'phi info'   

# cutting maps  
x0 = 180.5 #181.25                                      
y0 = -1.
x1 = 178.75 #178.75                                   
y1 = 0.75
phi = phi.selectSubMap(x0,x1,y0,y1)
print phi.Nx,phi.Ny
print phi.info(), 'phi info'   

scal = 'Aug6_highAcc_CDM_scalCls_new.dat'
lens = 'Aug6_highAcc_CDM_lensedCls_new.dat'


theoryPower = np.loadtxt(scal)
l=theoryPower[:,0]
cl_TT=theoryPower[:,1]
cl_EE=theoryPower[:,2]
cl_TE=theoryPower[:,3]
cl_BB=None

lMax = 40000 if 40000 < max(l) else max(l)
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


# do we divide by anything tt ps and what is p
phi_new = phi.copy()
phi_new.fillWithGaussianRandomField(lphi,clphi)#,bufferFactor=1)  

T_map,Q_map,U_map =liteMapPol.simPolMapsFromEandB(phi_new,l,cl_TT,cl_EE,cl_TE,cl_BB,fullBeamMatrix=fullBeamMatrix,beam1d=beam1d_off)

T_map.writeFits(nametu, overWrite=True)
phi_new.writeFits(namepn, overWrite=True)

Tlens, Qlens, Ulens  =  lensMaps(phi_new,T_map,Q_map,U_map)
Tlens.writeFits(nametl, overWrite=True)  

# noise ps
NlT = (deltaT*thetaFWHM)**2*np.exp(l_len*(l_len+1.)*thetaFWHM**2/(8.*np.log(2.)))*noiseFactor**2.
noise = T_map.copy()
print len(l_len), len(l)
noise.fillWithGaussianRandomField(l_len,NlT)#,bufferFactor=1)  

Tlens.data[:,:]  +=  noise.data[:,:]
Tlens.writeFits(nametln, overWrite=True)  
quit()
