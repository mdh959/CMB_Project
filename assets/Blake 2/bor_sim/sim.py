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

T_map.writeFits('T_map.fits', overWrite=True)
phi_new.writeFits('phi_new.fits', overWrite=True)

Tlens, Qlens, Ulens  =  lensMaps(phi_new,T_map,Q_map,U_map)
Tlens.writeFits('Tlens.fits', overWrite=True)  
quit()

kappaA = liteMap.liteMapFromFits('/home/r/rbond/bsherwin/dis2/JiaSims/Om0.394_Ol0.606_w-1.000_si0.776/WLconv_z1\
100.00_%04dr.fits'%(iii+1))
kappaA.pixScaleX = 3.5/180.*numpy.pi/2048
kappaA.pixScaleY = 3.5/180.*numpy.pi/2048

kappa = liteMapPol.makeEmptyCEATemplate(3.5, 3.5,meanRa = 180., meanDec = 0.,\
                                            pixScaleXarcmin = 0.102539, pixScaleYarcmin=0.102539)
kappa.data = kappaA.data


l_arr, cl_tt, cl_tt_len = np.loadtxt('out.txt', unpack=True)

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


