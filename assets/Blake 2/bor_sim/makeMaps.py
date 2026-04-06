import matplotlib
#matplotlib.use('Agg')                                                                                                
import numpy as np
import matplotlib.pyplot as plt
import astLib.astWCS
import astLib.astCoords
import astLib.astCalc
print "bau"
from flipper import *
from flipperPol import *
print "bau"
from lensingTools import *
print "bau"
from numpy.fft import fftshift,fftfreq,fft2,ifft2

nametu = 'T_map1.fits'
nametl = 'Tlens1.fits'
namepn = 'phi_new1.fits'

print "bau"
beam1d_off = {'apply':False}

fullBeamMatrix = {'apply':False}
print fullBeamMatrix
# load maps                                          
dirphi = '/users/boryanah/Blake/bor_sim/'
namep = 'phi.fits'
#Tlens = liteMap.liteMapFromFits(direc+namel)
#Tulens = liteMap.liteMapFromFits(direc+nameu)
phi = liteMap.liteMapFromFits(dirphi+namep)
print phi.info(), 'phi info'   

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

print np.max(clphi)

# do we divide by anything tt ps and what is p
phi_new = phi.copy()
phi_new.fillWithGaussianRandomField(lphi,clphi)#,bufferFactor=1)  

T_map,Q_map,U_map =liteMapPol.simPolMapsFromEandB(phi_new,l,cl_TT,cl_EE,cl_TE,cl_BB,fullBeamMatrix=fullBeamMatrix,beam1d=beam1d_off)

T_map.writeFits(nametu, overWrite=True)
phi_new.writeFits(namepn, overWrite=True)

Tlens, Qlens, Ulens  =  lensMaps(phi_new,T_map,Q_map,U_map)
Tlens.writeFits(nametl, overWrite=True)  
quit()
