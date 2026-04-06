from flipper import *
import liteMapPol
import numpy
from scipy.interpolate import splrep,splev


def TQUtoPureTEB(T_map,Q_map,U_map,window,modLMap,angLMap,method='standard'):

    window=liteMapPol.initializeDerivativesWindowfuntions(window)
    
    win =window['Win']
    dWin_dx=window['dWin_dx']
    dWin_dy=window['dWin_dy']
    d2Win_dx2=window['d2Win_dx2'] 
    d2Win_dy2=window['d2Win_dy2']
    d2Win_dxdy=window['d2Win_dxdy']

    T_temp=T_map.copy()
    Q_temp=Q_map.copy()
    U_temp=U_map.copy()

    T_temp.data=T_map.data*win
    fT=fftTools.fftFromLiteMap(T_temp)
    
    Q_temp.data=Q_map.data*win
    fQ=fftTools.fftFromLiteMap(Q_temp)
    
    U_temp.data=U_map.data*win
    fU=fftTools.fftFromLiteMap(U_temp)
    
    fE=fT.copy()
    fB=fT.copy()
    
    fE.kMap[:]=fQ.kMap[:]*numpy.cos(2.*angLMap)+fU.kMap[:]*numpy.sin(2.*angLMap)
    fB.kMap[:]=-fQ.kMap[:]*numpy.sin(2.*angLMap)+fU.kMap[:]*numpy.cos(2.*angLMap)
    
    if method=='standard':
        return fT, fE, fB
    
    Q_temp.data=Q_map.data*dWin_dx
    QWx=fftTools.fftFromLiteMap(Q_temp)
    
    Q_temp.data=Q_map.data*dWin_dy
    QWy=fftTools.fftFromLiteMap(Q_temp)
    
    U_temp.data=U_map.data*dWin_dx
    UWx=fftTools.fftFromLiteMap(U_temp)
    
    U_temp.data=U_map.data*dWin_dy
    UWy=fftTools.fftFromLiteMap(U_temp)
    
    U_temp.data=2.*Q_map.data*d2Win_dxdy-U_map.data*(d2Win_dx2-d2Win_dy2)
    QU_B=fftTools.fftFromLiteMap(U_temp)
 
    U_temp.data=-Q_map.data*(d2Win_dx2-d2Win_dy2)-2.*U_map.data*d2Win_dxdy
    QU_E=fftTools.fftFromLiteMap(U_temp)
    
    modLMap=modLMap+2


    fB.kMap[:] += QU_B.kMap[:]*(1./modLMap)**2
    fB.kMap[:]-= (2.*1j)/modLMap*(numpy.sin(angLMap)*(QWx.kMap[:]+UWy.kMap[:])+numpy.cos(angLMap)*(QWy.kMap[:]-UWx.kMap[:]))
    
    if method=='hybrid':
        return fT, fE, fB
    
    fE.kMap[:]+= QU_E.kMap[:]*(1./modLMap)**2
    fE.kMap[:]-= (2.*1j)/modLMap*(numpy.sin(angLMap)*(QWy.kMap[:]-UWx.kMap[:])-numpy.cos(angLMap)*(QWx.kMap[:]+UWy.kMap[:]))
    
    if method=='pure':
        return fT, fE, fB


def fourierTQU(T_map,Q_map,U_map):
    
    fT=fftTools.fftFromLiteMap(T_map)
    fQ=fftTools.fftFromLiteMap(Q_map)
    fU=fftTools.fftFromLiteMap(U_map)
    
    return(fT, fQ, fU)
    
    

def TQUtoFourierTEB(T_map,Q_map,U_map,window,modLMap,angLMap):

    T_map.data*=window.data
    Q_map.data*=window.data
    U_map.data*=window.data

    fT=fftTools.fftFromLiteMap(T_map)
    
    fQ=fftTools.fftFromLiteMap(Q_map)
        
    fU=fftTools.fftFromLiteMap(U_map)
    
    fE=fT.copy()
    fB=fT.copy()
    fE.kMap[:]=fQ.kMap[:]*numpy.cos(2.*angLMap)+fU.kMap[:]*numpy.sin(2.*angLMap)
    fB.kMap[:]=-fQ.kMap[:]*numpy.sin(2.*angLMap)+fU.kMap[:]*numpy.cos(2.*angLMap)
    
    return(fT, fE, fB)
    
    
def generateKspacePower(liteMap,bufferFactor,l,Cl,ll):
    
    Ny = liteMap.Ny*bufferFactor
    Nx = liteMap.Nx*bufferFactor
    s=splrep(l,Cl,k=3)
    kk = splev(ll,s)
    id = numpy.where(ll>l.max())
    kk[id] = 0.
    area = Nx*Ny*liteMap.pixScaleX*liteMap.pixScaleY
    p = numpy.reshape(kk,[Ny,Nx])/area * (Nx*Ny)**2
    
    return(p)





def generate_EE_Power(Cl_TT,Cl_TE,Cl_EE):
    
    clCorr_EE = Cl_TE/(Cl_TT**0.5)
    tmp= Cl_EE - Cl_TE**2/Cl_TT
    loc = numpy.where(tmp<0.)
    tmp[loc] = 0.
    clUncorr_EE = (tmp)**0.5
    
    return(clCorr_EE,clUncorr_EE)

	
def fourierTEBtoFourierTQU(fT,fE,fB,modLMap,angLMap):
    fQ=fT.copy()
    fU=fT.copy()
    fQ.kMap[:]=fE.kMap[:]*numpy.cos(2.*angLMap)-fB.kMap[:]*numpy.sin(2.*angLMap)
    fU.kMap[:]=fE.kMap[:]*numpy.sin(2.*angLMap)+fB.kMap[:]*numpy.cos(2.*angLMap)
    return fT,fQ,fU
    

def fourierTQUtoPowerTEB(fT,fQ,fU,modLMap,angLMap):
    fE=fT.copy()
    fB=fT.copy()
    fE.kMap[:]=fQ.kMap[:]*numpy.cos(2.*angLMap)+fU.kMap[:]*numpy.sin(2.*angLMap)
    fB.kMap[:]=-fQ.kMap[:]*numpy.sin(2.*angLMap)+fU.kMap[:]*numpy.cos(2.*angLMap)
    
    power_TT=fftTools.powerFromFFT(fT)
    power_TE=fftTools.powerFromFFT(fT,fE)
    power_TB=fftTools.powerFromFFT(fT,fB)
    power_EE=fftTools.powerFromFFT(fE)
    power_BE=fftTools.powerFromFFT(fB,fE)
    power_BB=fftTools.powerFromFFT(fB)
    
    return(power_TT,power_TE,power_TB,power_EE,power_BE,power_BB)
    
    
    
def fourierTEBtoPowerTEB(fT0,fE0,fB0,fT1,fE1,fB1):
    
    TT_power=fftTools.powerFromFFT(fT0,fT1)
    TE_power=fftTools.powerFromFFT(fT0,fE1)
    ET_power=fftTools.powerFromFFT(fE0,fT1)

    TB_power=fftTools.powerFromFFT(fT0,fB1)
    BT_power=fftTools.powerFromFFT(fB0,fT1)

    EE_power=fftTools.powerFromFFT(fE0,fE1)
    BE_power=fftTools.powerFromFFT(fB0,fE1)
    EB_power=fftTools.powerFromFFT(fE0,fB1)

    BB_power=fftTools.powerFromFFT(fB0,fB1)
    
    return(TT_power,TE_power,ET_power,TB_power,BT_power,EE_power,EB_power,BE_power,BB_power)


def fourierTEtoPowerTE(fT0,fE0,fT1,fE1):
    
    TT_power=fftTools.powerFromFFT(fT0,fT1)
    TE_power=fftTools.powerFromFFT(fT0,fE1)
    ET_power=fftTools.powerFromFFT(fE0,fT1)
    EE_power=fftTools.powerFromFFT(fE0,fE1)
    
    return(TT_power,TE_power,ET_power,EE_power)

    
def makeTemplate(m, wl, ell, maxEll, outputFile = None):
    """
    Yanked from Toby's csFilter
    For a given map (m) return a 2D k-space template from a 1D specification wl
    ell = 2pi * i / deltaX
    (m is not overwritten)
    """

    ell = numpy.array(ell)
    wl  = numpy.array(wl)
    
    p2d = fftTools.powerFromLiteMap(m)
    p2d.powerMap[:] = 0.

    l_f = numpy.floor(p2d.modLMap)
    l_c = numpy.ceil(p2d.modLMap)
    
    for i in xrange(numpy.shape(p2d.powerMap)[0]):
        for j in xrange(numpy.shape(p2d.powerMap)[1]):
            if l_f[i,j] > maxEll or l_c[i,j] > maxEll:
                continue
            w_lo = wl[l_f[i,j]]
            w_hi = wl[l_c[i,j]]
            trueL = p2d.modLMap[i,j]
            w = (w_hi-w_lo)*(trueL - l_f[i,j]) + w_lo
            p2d.powerMap[i,j] = w



            
    if outputFile != None:
        p2d.writeFits(outputFile, overWrite = True)
    return p2d
    
    
def makeEllandAngCoordinate(liteMap,bufferFactor=1):
	
    Ny = liteMap.Ny*bufferFactor
    Nx = liteMap.Nx*bufferFactor
    ly = numpy.fft.fftfreq(Ny,d = liteMap.pixScaleY)*(2*numpy.pi)
    lx = numpy.fft.fftfreq(Nx,d = liteMap.pixScaleX)*(2*numpy.pi)
    modLMap = numpy.zeros([Ny,Nx])
    angLMap = numpy.zeros([Ny,Nx])
    iy, ix = numpy.mgrid[0:Ny,0:Nx]
    modLMap[iy,ix] = numpy.sqrt(ly[iy]**2+lx[ix]**2)
    #Trigonometric orientation
    angLMap[iy,ix]= numpy.arctan2(ly[iy],lx[ix])
    
    return(modLMap,angLMap)
	
	


