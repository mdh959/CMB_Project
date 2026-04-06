from flipper import *
import numpy
from flipperPol import *







def  LensedSimPolMaps(m0,ell,Cell_TT,Cell_EE,Cell_TE,Cell_BB=None,nUp=3,bufferFactor=1):
    
    
    m = liteMap.getEmptyMapWithDifferentDims(m0,m0.Ny*nUp,m0.Nx*nUp)
    
    T_map,Q_map,U_map =liteMapPol.SimPolMapsFromEandB(m,ell,Cell_TT,Cell_EE,Cell_TE,Cell_BB=Cell_BB,bufferFactor=1)
    	
    phi  = liteMap.liteMapFromFits("phikappa/phiMapHiRes.fits")
    alpha = phi.takeGradient()
    iy,ix = numpy.mgrid[0:phi.Ny,0:phi.Nx]
    iyf = iy.flatten()
    ixf = ix.flatten()

    a = numpy.array(alpha.gradX.data/ alpha.gradX.pixScaleX,dtype='int64')
    b = numpy.array(alpha.gradY.data/ alpha.gradY.pixScaleY,dtype='int64')

    iyLensed = iyf.copy()
    ixLensed = ixf.copy()

    iyLensed[:] = iyf[:] + b.flatten()
    ixLensed[:] = ixf[:] + a.flatten()

    id = numpy.where((ixLensed > ixf.max()) | (ixLensed < ixf.min()))
    id2 = numpy.where((iyLensed > iyf.max()) | (iyLensed < iyf.min()))

    ixLensed[id]  = ixf[id]
    iyLensed[id2] = iyf[id2]
	
    lensed_T_map=T_map.copy()
    lensed_Q_map=Q_map.copy()
    lensed_U_map=U_map.copy()
	
    lensed_T_map.data[iyf,ixf] = T_map.data[iyLensed,ixLensed]
    lensed_Q_map.data[iyf,ixf] = Q_map.data[iyLensed,ixLensed]
    lensed_U_map.data[iyf,ixf] = U_map.data[iyLensed,ixLensed]
		
    lensed_T_mapLo = m0.copy()
    lensed_T_mapLo = liteMap.resampleFromHiResMap(lensed_T_map,m0)
    
    lensed_Q_mapLo = m0.copy()
    lensed_Q_mapLo = liteMap.resampleFromHiResMap(lensed_Q_map,m0)
    
    lensed_U_mapLo = m0.copy()
    lensed_U_mapLo = liteMap.resampleFromHiResMap(lensed_U_map,m0)
    
    
    return(lensed_T_mapLo,lensed_Q_mapLo,lensed_U_mapLo)
    

def  GenerateAndSaveLensedSimPolMaps(m0,phi,ell,Cell_TT,Cell_EE,Cell_TE,nUp,iterNum,Cell_BB=None,bufferFactor=1):

    try:
        os.mkdir('unlensedCMBMaps')
        os.mkdir('lensedCMBMaps')
    except:
        pass
    
    m = liteMap.getEmptyMapWithDifferentDims(m0,m0.Ny*nUp,m0.Nx*nUp)
    
    T_map,Q_map,U_map =liteMapPol.SimPolMapsFromEandB(m,ell,Cell_TT,Cell_EE,Cell_TE,Cell_BB=Cell_BB,bufferFactor=1)
    T_map.data[:]=T_map.data[:]-numpy.mean(T_map.data[:])
    Q_map.data[:]=Q_map.data[:]-numpy.mean(Q_map.data[:])
    U_map.data[:]=U_map.data[:]-numpy.mean(U_map.data[:])

    T_mapLo = m0.copy()
    T_mapLo = liteMap.resampleFromHiResMap(T_map,m0)
    T_mapLo.writeFits('unlensedCMBMaps/T_map_%03d.fits'%(iterNum),overWrite=True)
    print "unlensed T_map low resolution done"
    del T_mapLo
    
    Q_mapLo = m0.copy()
    Q_mapLo = liteMap.resampleFromHiResMap(Q_map,m0)
    Q_mapLo.writeFits('unlensedCMBMaps/Q_map_%03d.fits'%(iterNum),overWrite=True)
    print "unlensed Q_map low resolution done"
    del Q_mapLo

    U_mapLo = m0.copy()
    U_mapLo = liteMap.resampleFromHiResMap(U_map,m0)
    U_mapLo.writeFits('unlensedCMBMaps/U_map_%03d.fits'%(iterNum),overWrite=True)
    print "unlensed U_map low resolution done"
    del U_mapLo
    
    alpha = phi.takeGradient()
    iy,ix = numpy.mgrid[0:phi.Ny,0:phi.Nx]
    iyf = iy.flatten()
    ixf = ix.flatten()

    a = numpy.array(alpha.gradX.data/ alpha.gradX.pixScaleX,dtype='int64')
    b = numpy.array(alpha.gradY.data/ alpha.gradY.pixScaleY,dtype='int64')

    iyLensed = iyf.copy()
    ixLensed = ixf.copy()
    
    
    iyLensed[:] = iyf[:] + b.flatten()                                                                                   
    ixLensed[:] = ixf[:] + a.flatten()                                                                                   
                                                                                                                         
    id = numpy.where(ixLensed > ixf.max())                                                                               
    id2 = numpy.where(iyLensed > iyf.max())                                                                              
                                                                                                                         
                                                                                                                         
    ixLensed[id]  = ixLensed[id] - phi.Nx                                                                                
    iyLensed[id2] = iyLensed[id2] - phi.Ny                                                                               
                                                                                                                         
    id = numpy.where(ixLensed < ixf.min())                                                                               
    id2 = numpy.where(iyLensed < iyf.min())                                                                              
                                                                                                                         
    ixLensed[id]  = ixLensed[id] + phi.Nx                                                                                
    iyLensed[id2] = iyLensed[id2] + phi.Ny                                                                               
                                            
	
    lensed_T_map=T_map.copy()
    lensed_Q_map=Q_map.copy()
    lensed_U_map=U_map.copy()
	
    lensed_T_map.data[iyf,ixf] = T_map.data[iyLensed,ixLensed]
    lensed_Q_map.data[iyf,ixf] = Q_map.data[iyLensed,ixLensed]
    lensed_U_map.data[iyf,ixf] = U_map.data[iyLensed,ixLensed]
		
    lensed_T_mapLo = m0.copy()
    lensed_T_mapLo = liteMap.resampleFromHiResMap(lensed_T_map,m0)
    lensed_T_mapLo.writeFits('lensedCMBMaps/lensed_T_map_%03d.fits'%(iterNum),overWrite=True)
    print "lensed T_map low resolution done"
    del lensed_T_mapLo
    
    lensed_Q_mapLo = m0.copy()
    lensed_Q_mapLo = liteMap.resampleFromHiResMap(lensed_Q_map,m0)
    lensed_Q_mapLo.writeFits('lensedCMBMaps/lensed_Q_map_%03d.fits'%(iterNum),overWrite=True)
    print "lensed Q_map low resolution done"
    del lensed_Q_mapLo

    
    lensed_U_mapLo = m0.copy()
    lensed_U_mapLo = liteMap.resampleFromHiResMap(lensed_U_map,m0)
    lensed_U_mapLo.writeFits('lensedCMBMaps/lensed_U_map_%03d.fits'%(iterNum),overWrite=True)
    print "lensed U_map low resolution done"
    del lensed_U_mapLo
    
    return
    

        
