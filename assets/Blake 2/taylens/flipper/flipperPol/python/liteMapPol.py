from flipper import *
import fftPol
import numpy
import pyfits
from scipy.interpolate import splrep,splev
import systematicAndBeam




    
    

def initializeCosineWindow(liteMap,lenApod,pad):
	
    Nx=liteMap.Nx
    Ny=liteMap.Ny
    win=liteMap.copy()
    win.data[:]=1
    
    winX=win.copy()
    winY=win.copy()
        
    for j in range(pad,Ny-pad):
	for i in range(pad,Nx-pad):
	    if i<=(lenApod+pad):
		r=float(i)-pad
		winX.data[j,i]=1./2*(1-numpy.cos(-numpy.pi*r/lenApod))
	    if i>=(Nx-1)-lenApod-pad:
		r=float((Nx-1)-i-pad)
		winX.data[j,i]=1./2*(1-numpy.cos(-numpy.pi*r/lenApod))

    for i in range(pad,Nx-pad):
	for j in range(pad,Ny-pad):
	    if j<=(lenApod+pad):
		r=float(j)-pad
		winY.data[j,i]=1./2*(1-numpy.cos(-numpy.pi*r/lenApod))
	    if j>=(Ny-1)-lenApod-pad:
		r=float((Ny-1)-j-pad)
		winY.data[j,i]=1./2*(1-numpy.cos(-numpy.pi*r/lenApod))
	
    win.data[:]*=winX.data[:,:]*winY.data[:,:]
    win.data[0:pad,:]=0
    win.data[:,0:pad]=0
    win.data[Nx-pad:Nx,:]=0
    win.data[:,Nx-pad:Nx]=0
	
    return(win)
	

def makeMask(liteMap,nHoles,holeSize,lenApodMask,show=False):
        
    pixScaleArcmin=liteMap.pixScaleX*60*360/numpy.pi
    holeSizePix=numpy.int(holeSize/pixScaleArcmin)

    mask=liteMap.copy()
    mask.data[:]=1
    holeMask=mask.copy()
    
    Nx=mask.Nx
    Ny=mask.Ny
    xList=numpy.random.rand(nHoles)*Nx
    yList=numpy.random.rand(nHoles)*Ny    
    
    for k in range(nHoles):
    	print "number of Holes",k
        holeMask.data[:]=1
        for i in range(Nx):
            for j in range(Ny):
            	rad=(i-numpy.int(xList[k]))**2+(j-numpy.int(yList[k]))**2
            	
            	if rad < holeSizePix**2:
                    holeMask.data[j,i]=0
                for pix in range(lenApodMask):
                	
                    if rad <= (holeSizePix+pix)**2 and rad > (holeSizePix+pix-1)**2:
                        	holeMask.data[j,i]=1./2*(1-numpy.cos(-numpy.pi*float(pix)/lenApodMask))
        mask.data[:]*=holeMask.data[:]
    data=mask.data[:]
    
    if show==True:
    	pylab.matshow(data)
    	pylab.show()
    
        
    return mask




def initializeDerivativesWindowfuntions(liteMap):
	
    def matrixShift(l,row_shift,column_shift):	
        m1=numpy.hstack((l[:,row_shift:],l[:,:row_shift]))
        m2=numpy.vstack((m1[column_shift:],m1[:column_shift]))
        return m2
    delta=liteMap.pixScaleX
    Win=liteMap.data[:]
    
    dWin_dx=(-matrixShift(Win,-2,0)+8*matrixShift(Win,-1,0)-8*matrixShift(Win,1,0)+matrixShift(Win,2,0))/(12*delta)
    dWin_dy=(-matrixShift(Win,0,-2)+8*matrixShift(Win,0,-1)-8*matrixShift(Win,0,1)+matrixShift(Win,0,2))/(12*delta)
    d2Win_dx2=(-matrixShift(dWin_dx,-2,0)+8*matrixShift(dWin_dx,-1,0)-8*matrixShift(dWin_dx,1,0)+matrixShift(dWin_dx,2,0))/(12*delta)
    d2Win_dy2=(-matrixShift(dWin_dy,0,-2)+8*matrixShift(dWin_dy,0,-1)-8*matrixShift(dWin_dy,0,1)+matrixShift(dWin_dy,0,2))/(12*delta)
    d2Win_dxdy=(-matrixShift(dWin_dy,-2,0)+8*matrixShift(dWin_dy,-1,0)-8*matrixShift(dWin_dy,1,0)+matrixShift(dWin_dy,2,0))/(12*delta)
    
    #In return we change the sign of the simple gradient in order to agree with numpy convention
    return {'Win':Win, 'dWin_dx':-dWin_dx,'dWin_dy':-dWin_dy, 'd2Win_dx2':d2Win_dx2, 'd2Win_dy2':d2Win_dy2,'d2Win_dxdy':d2Win_dxdy}
	
	

def  simPolMapsFromEandB(Temp,l,cl_TT,cl_EE,cl_TE,cl_BB=None,fullBeamMatrix=None,beam1d=None):
    
    
    
    buffer=1
    Ny = Temp.Ny
    Nx = Temp.Nx
    
    modLMap,angLMap=fftPol.makeEllandAngCoordinate(Temp,buffer)
	
    ll = numpy.ravel(modLMap)
    
    p_TT= fftPol.generateKspacePower(Temp,buffer,l,cl_TT,ll)
    p_EE= fftPol.generateKspacePower(Temp,buffer,l,cl_EE,ll)
    
    
    randomSharedReal=numpy.random.randn(Ny,Nx)
    randomSharedIm=numpy.random.randn(Ny,Nx)
    
    realPart_T = numpy.sqrt(p_TT)*randomSharedReal
    imgPart_T = numpy.sqrt(p_TT)*randomSharedIm
    kMap_T = realPart_T+1j*imgPart_T
    
    realPart_E = numpy.sqrt(p_EE)*numpy.random.randn(Ny,Nx)
    imgPart_E = numpy.sqrt(p_EE)*numpy.random.randn(Ny,Nx)
    kMap_Eold= realPart_E+1j*imgPart_E
    
    ### add correlations - uses generate_2d_power
    
    clCorr_EE,clUncorr_EE= fftPol.generate_EE_Power(cl_TT,cl_TE,cl_EE)
    
    pCorr_EE = fftPol.generateKspacePower(Temp,buffer,l,clCorr_EE,ll)
    pUnCorr_EE = fftPol.generateKspacePower(Temp,buffer,l,clUncorr_EE,ll)
    
    loc2 = numpy.where(p_TT == 0.)
    loc3 = numpy.where(p_EE == 0.)
    p_TT[loc2]=1
    p_EE[loc3]=1
    
    area = Nx*Ny*Temp.pixScaleX*Temp.pixScaleY
    
    
    kMap_E = (pCorr_EE*kMap_T/numpy.sqrt(p_TT)+kMap_Eold/numpy.sqrt(p_EE)*pUnCorr_EE)*numpy.sqrt(area)/(Nx*Ny)
    
    
    kMap_E[loc2] = 0.+0.j
    kMap_E[loc3] = 0.+0.j
    
    
    if cl_BB!=None:
    	p_BB= fftPol.generateKspacePower(Temp,buffer,l,cl_BB,ll)
    	realPart_B = numpy.sqrt(p_BB)*numpy.random.randn(Ny,Nx)
    	imgPart_B = numpy.sqrt(p_BB)*numpy.random.randn(Ny,Nx)
    	kMap_B= realPart_B+1j*imgPart_B
    	kMap_Q=	kMap_E * numpy.cos(2*angLMap) - kMap_B*numpy.sin(2*angLMap)
    	kMap_U=	kMap_E * numpy.sin(2*angLMap) + kMap_B*numpy.cos(2*angLMap)
    
    
    else:
    	kMap_Q=	kMap_E * numpy.cos(2*angLMap)
    	kMap_U=	kMap_E * numpy.sin(2*angLMap)
    
    
    
    data_T = numpy.real(numpy.fft.ifft2(kMap_T))
    data_Q=  numpy.real(numpy.fft.ifft2(kMap_Q))
    data_U=  numpy.real(numpy.fft.ifft2(kMap_U))
    
    
    
    T_map=Temp.copy()
    Q_map=Temp.copy()
    U_map=Temp.copy()
    
    
    T_map.data=data_T-numpy.mean(data_T)
    Q_map.data=data_Q-numpy.mean(data_Q)
    U_map.data=data_U-numpy.mean(data_U)
    
    
    if fullBeamMatrix['apply']==True:
    	print "Apply fullBeamMatrix"
    	
    	beamArray=systematicAndBeam.makeBeam(T_map,fullBeamMatrix)
    	
    	T_map,Q_map,U_map=systematicAndBeam.beamConvolutionLiteMap(T_map,Q_map,U_map,beamArray)
    
    
    
    elif beam1d['apply']==True:
        ell, f_ell = numpy.transpose(numpy.loadtxt(os.environ['SPECK_DIR']+'/data/'+beam1d['file']))
        t = makeTemplate( T_map, f_ell, ell, ell.max())
        f_T = numpy.fft.fft2(T_map.data)
        f_Q = numpy.fft.fft2(Q_map.data)
        f_U = numpy.fft.fft2(U_map.data)
        f_T*= t
        f_Q*= t
        f_U*= t
        T_map.data[:]=numpy.real(numpy.fft.ifft2(f_T))
        Q_map.data[:]=numpy.real(numpy.fft.ifft2(f_Q))
        U_map.data[:]=numpy.real(numpy.fft.ifft2(f_U))
    
    
    return(T_map,Q_map,U_map)



        
def padLiteMap(Small,Big):
    DeltaX=numpy.int(Big.Nx-Small.Nx)
    DeltaY=numpy.int(Big.Ny-Small.Ny)
    lowX=DeltaX/2
    lowY=DeltaY/2
    if DeltaY/2.!=DeltaY/2:
        lowY=lowY+1
    if DeltaX/2.!=DeltaX/2:
        lowX=lowX+1
    
    padMap=Big.copy()
    padMap.data[:]=0
    padMap.data[lowY:Big.Ny-DeltaY/2,lowX:Big.Nx-DeltaX/2]=Small.data[:]
    
    return(padMap)

