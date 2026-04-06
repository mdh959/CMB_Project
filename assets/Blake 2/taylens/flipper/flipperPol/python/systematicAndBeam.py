from flipper import *
import liteMapPol
import numpy


def generateBeam(T_map,fullBeamMatrix):

    beamXfwhp = fullBeamMatrix['beam_xfwhp']
    beamYfwhp = fullBeamMatrix['beam_yfwhp']
	
    beamXfwhp=beamXfwhp/60./180.*numpy.pi/T_map.pixScaleX
    beamYfwhp=beamYfwhp/60./180.*numpy.pi/T_map.pixScaleY    

    xSigma=beamXfwhp/(8*numpy.log(2))**0.5
    ySigma=beamYfwhp/(8*numpy.log(2))**0.5

    sizeX = T_map.Nx
    sizeY = T_map.Ny
    
    y,x=numpy.mgrid[-T_map.Ny/2:T_map.Ny/2,-T_map.Nx/2:T_map.Nx/2] 
    
    ang=numpy.arctan2(y/float(T_map.Ny),x/float(T_map.Nx))
    
    beam=numpy.exp(-0.5*((x/(xSigma))**2+(y/(ySigma))**2))
    
    beam=beam/numpy.sum(beam)


    r=(x**2+y**2)**(0.5)
    
    dip1=r*numpy.sin(ang)*beam
    dip2=r*numpy.cos(ang)*beam
    quad1=r**2*numpy.sin(2*ang)*beam
    quad2=r**2*numpy.cos(2*ang)*beam
    
  
    monoMap=T_map.copy()
    dip1Map=T_map.copy()
    dip2Map=T_map.copy()
    quad1Map=T_map.copy()
    quad2Map=T_map.copy()

    monoMap.data[:]=beam/(numpy.sqrt(numpy.sum(beam*beam)))
    
    dip1Map.data[:]=dip1/(numpy.sqrt(numpy.sum(dip1*dip1)))
    dip2Map.data[:]=dip2/(numpy.sqrt(numpy.sum(dip2*dip2)))
    quad1Map.data[:]=quad1/(numpy.sqrt(numpy.sum(quad1*quad1)))
    quad2Map.data[:]=quad2/(numpy.sqrt(numpy.sum(quad2*quad2)))
    
    
    return(monoMap,dip1Map,dip2Map,quad1Map,quad2Map)
 
 

def makeBeam(m0,fullBeamMatrix):

    beamMono=numpy.loadtxt(os.environ['FLIPPERPOL_DIR']+'/params/'+'Monopole.txt')
    beamDip1=numpy.loadtxt(os.environ['FLIPPERPOL_DIR']+'/params/'+'Dipole1.txt') 
    beamDip2=numpy.loadtxt(os.environ['FLIPPERPOL_DIR']+'/params/'+'Dipole2.txt') 
    beamQuad1=numpy.loadtxt(os.environ['FLIPPERPOL_DIR']+'/params/'+'Quadrupole1.txt')
    beamQuad2=numpy.loadtxt(os.environ['FLIPPERPOL_DIR']+'/params/'+'Quadrupole2.txt')

    monoMap,dip1Map,dip2Map,quad1Map,quad2Map=generateBeam(m0,fullBeamMatrix)

    TtoT=m0.copy()
    TtoQ=m0.copy()
    TtoU=m0.copy()
    QtoQ=m0.copy()
    QtoU=m0.copy()
    QtoT=m0.copy()
    UtoU=m0.copy()
    UtoT=m0.copy()
    UtoQ=m0.copy()


    TtoT.data[:]=beamMono[0,0]*monoMap.data[:]+beamDip1[0,0]*dip1Map.data[:]+beamDip2[0,0]*dip2Map.data[:]+beamQuad1[0,0]*quad1Map.data[:]+beamQuad2[0,0]*quad2Map.data[:]
    TtoU.data[:]=beamMono[1,0]*monoMap.data[:]+beamDip1[1,0]*dip1Map.data[:]+beamDip2[1,0]*dip2Map.data[:]+beamQuad1[1,0]*quad1Map.data[:]+beamQuad2[1,0]*quad2Map.data[:]
    TtoQ.data[:]=beamMono[2,0]*monoMap.data[:]+beamDip1[2,0]*dip1Map.data[:]+beamDip2[2,0]*dip2Map.data[:]+beamQuad1[2,0]*quad1Map.data[:]+beamQuad2[2,0]*quad2Map.data[:]
    QtoQ.data[:]=beamMono[2,2]*monoMap.data[:]+beamDip1[2,2]*dip1Map.data[:]+beamDip2[2,2]*dip2Map.data[:]+beamQuad1[2,2]*quad1Map.data[:]+beamQuad2[2,2]*quad2Map.data[:]
    QtoU.data[:]=beamMono[1,2]*monoMap.data[:]+beamDip1[1,2]*dip1Map.data[:]+beamDip2[1,2]*dip2Map.data[:]+beamQuad1[1,2]*quad1Map.data[:]+beamQuad2[1,2]*quad2Map.data[:]
    QtoT.data[:]=beamMono[2,0]*monoMap.data[:]+beamDip1[2,0]*dip1Map.data[:]+beamDip2[2,0]*dip2Map.data[:]+beamQuad1[2,0]*quad1Map.data[:]+beamQuad2[2,0]*quad2Map.data[:]
    UtoU.data[:]=beamMono[1,1]*monoMap.data[:]+beamDip1[1,1]*dip1Map.data[:]+beamDip2[1,1]*dip2Map.data[:]+beamQuad1[1,1]*quad1Map.data[:]+beamQuad2[1,1]*quad2Map.data[:]
    UtoT.data[:]=beamMono[0,1]*monoMap.data[:]+beamDip1[0,1]*dip1Map.data[:]+beamDip2[0,1]*dip2Map.data[:]+beamQuad1[0,1]*quad1Map.data[:]+beamQuad2[0,1]*quad2Map.data[:]
    UtoQ.data[:]=beamMono[2,1]*monoMap.data[:]+beamDip1[2,1]*dip1Map.data[:]+beamDip2[2,1]*dip2Map.data[:]+beamQuad1[2,1]*quad1Map.data[:]+beamQuad2[2,1]*quad2Map.data[:]
    
    
    del(monoMap)
    del(dip1Map)
    del(dip2Map)
    del(quad1Map)
    del(quad2Map)

        
    TtoT.data[:]/=numpy.sum(TtoT.data)
    QtoQ.data[:]/=numpy.sum(QtoQ.data)
    UtoU.data[:]/=numpy.sum(UtoU.data)
    TtoQ.data[:]/=numpy.sqrt(numpy.sum(TtoT.data)*numpy.sum(QtoQ.data))
    TtoU.data[:]/=numpy.sqrt(numpy.sum(TtoT.data)*numpy.sum(UtoU.data))
    QtoT.data[:]/=numpy.sqrt(numpy.sum(TtoT.data)*numpy.sum(QtoQ.data))
    QtoU.data[:]/=numpy.sqrt(numpy.sum(UtoU.data)*numpy.sum(QtoQ.data))
    UtoT.data[:]/=numpy.sqrt(numpy.sum(TtoT.data)*numpy.sum(UtoU.data))
    UtoQ.data[:]/=numpy.sqrt(numpy.sum(UtoU.data)*numpy.sum(QtoQ.data))
    
    TtoT.data=numpy.fft.fftshift(TtoT.data)
    QtoQ.data=numpy.fft.fftshift(QtoQ.data)
    UtoU.data=numpy.fft.fftshift(UtoU.data)
    TtoQ.data=numpy.fft.fftshift(TtoQ.data)
    TtoU.data=numpy.fft.fftshift(TtoU.data)
    QtoT.data=numpy.fft.fftshift(QtoT.data)
    QtoU.data=numpy.fft.fftshift(QtoU.data)
    UtoT.data=numpy.fft.fftshift(UtoT.data)
    UtoQ.data=numpy.fft.fftshift(UtoQ.data)
  
    ft_TtoT=fftTools.fftFromLiteMap(TtoT)
    ft_TtoU=fftTools.fftFromLiteMap(TtoU)
    ft_QtoT=fftTools.fftFromLiteMap(QtoT)
    ft_QtoQ=fftTools.fftFromLiteMap(QtoQ)
    ft_QtoU=fftTools.fftFromLiteMap(QtoU)
    ft_TtoQ=fftTools.fftFromLiteMap(TtoQ)
    ft_UtoT=fftTools.fftFromLiteMap(UtoT)
    ft_UtoQ=fftTools.fftFromLiteMap(UtoQ)
    ft_UtoU=fftTools.fftFromLiteMap(UtoU)
    
    beamArray={}
    
    beamArray[11]=ft_TtoT.kMap[:]
    beamArray[12]=ft_QtoT.kMap[:]
    beamArray[13]=ft_UtoT.kMap[:]
    beamArray[21]=ft_TtoQ.kMap[:]
    beamArray[22]=ft_QtoQ.kMap[:]
    beamArray[23]=ft_UtoQ.kMap[:]
    beamArray[31]=ft_TtoU.kMap[:]
    beamArray[32]=ft_QtoU.kMap[:]
    beamArray[33]=ft_UtoU.kMap[:]
       
    return beamArray
	


def beamConvolutionLiteMap(T_map,Q_map,U_map,BeamArray):


    ft_T=fftTools.fftFromLiteMap(T_map)
    ft_Q=fftTools.fftFromLiteMap(Q_map)
    ft_U=fftTools.fftFromLiteMap(U_map)

	
    ft_T1=ft_T.kMap[:]*BeamArray[11] +  ft_Q.kMap[:]*BeamArray[12]  + ft_U.kMap[:]*BeamArray[13] 
    ft_Q1=ft_T.kMap[:]*BeamArray[21] +  ft_Q.kMap[:]*BeamArray[22]  + ft_U.kMap[:]*BeamArray[23] 
    ft_U1=ft_T.kMap[:]*BeamArray[31]  + ft_Q.kMap[:]*BeamArray[32]  + ft_U.kMap[:]*BeamArray[33] 
    

    
    T_map.data =  numpy.real(numpy.fft.ifft2(ft_T1))
    Q_map.data =  numpy.real(numpy.fft.ifft2(ft_Q1))
    U_map.data =  numpy.real(numpy.fft.ifft2(ft_U1))
    

	
    return T_map,Q_map,U_map
    


def beamDeconvolutionFourier(ft_T,ft_Q,ft_U,BA):

    #BA stands for Beam Array
    
    D=(BA[11]*(BA[22]*BA[33]-BA[23]*BA[32])-BA[12]*(BA[21]*BA[33]-BA[23]*BA[31])+BA[13]*(BA[21]*BA[32]-BA[22]*BA[31]))
    
    D_real=numpy.real(D)
    D_im=numpy.real(-1j*D)
	
    id=numpy.where(numpy.abs(D_real)< 10**-20)
    D_real[id]=10**-20
	
    id=numpy.where(numpy.abs(D_im)< 10**-20)
    D_im[id]=10**-20
	
    D=D_real+1j*D_im
    

    
    ft_T2=(ft_T.kMap[:]*(BA[22]*BA[33]-BA[23]*BA[32])+ft_Q.kMap[:]*(BA[13]*BA[32]-BA[12]*BA[33])+ft_U.kMap[:]*(BA[12]*BA[23]-BA[13]*BA[22]))/D
    ft_Q2=(ft_T.kMap[:]*(BA[23]*BA[31]-BA[21]*BA[33])+ft_Q.kMap[:]*(BA[11]*BA[33]-BA[13]*BA[31])+ft_U.kMap[:]*(BA[13]*BA[21]-BA[11]*BA[23]))/D
    ft_U2=(ft_T.kMap[:]*(BA[21]*BA[32]-BA[22]*BA[31])+ft_Q.kMap[:]*(BA[12]*BA[31]-BA[11]*BA[32])+ft_U.kMap[:]*(BA[11]*BA[22]-BA[12]*BA[21]))/D
    


    ft_T.kMap[:]=ft_T2
    ft_Q.kMap[:]=ft_Q2
    ft_U.kMap[:]=ft_U2
    
    return ft_T,ft_Q,ft_U


