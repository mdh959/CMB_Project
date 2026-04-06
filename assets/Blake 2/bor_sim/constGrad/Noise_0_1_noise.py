# differs from old by proper normalization
# IMPORTANT IS RMS OF SIMULATION
## calcLensingNoise.py
## Based on 'LensingNoise.py' in OxFish_New
## Calculate the minimum variance noise spectrum 
import numpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.integrate as integ
from scipy.interpolate import UnivariateSpline as spline
import sys

## Format labels to Latex font
#plt.rc('text',usetex=True)
#plt.rc('font',**{'family':'serif','serif':['Computer Modern'],'size':26})

#                             ASK
TCMB = 2.7255e6

N1 = 0
N2 = 10
CMEAN = np.zeros(45)
CSTD = np.zeros(45)
NMEAN = np.zeros(45)
NSTD = np.zeros(45)
for i in range(N1,N2):
	namef = 'out_n0_1_'+str(i)+'.txt'
	lls, Clphi, Clphi_est_N, Clphi_cross, Nlphi = np.loadtxt(namef,unpack=True)
	CEST = np.abs((Clphi_est_N-Clphi)*lls**4/(2*np.pi))
	#CEST = ((Clphi_est_N-Clphi)*lls**4/(2*np.pi))
	CMEAN += CEST
	NMEAN += Nlphi*lls**4/(2*np.pi)
CMEAN /= (N2-N1)
NMEAN /= (N2-N1)
for i in range(N1,N2):
	namef = 'out_n0_1_'+str(i)+'.txt'
	lls, Clphi, Clphi_est_N, Clphi_cross, Nlphi = np.loadtxt(namef,unpack=True)
	CEST = (Clphi_est_N-Clphi)*lls**4/(2*np.pi)
	CSTD += (CEST-CMEAN)**2
	NEST = (Nlphi)*lls**4/(2*np.pi)
	NSTD += (NEST-NMEAN)**2

CSTD = np.sqrt(CSTD/(N2-N1-1))
NSTD = np.sqrt(NSTD/(N2-N1-1))
print "CMEAN"
print CMEAN[:15]
print CSTD[:15]

print "NMEAN"
print NMEAN[:15]
print NSTD[:15]

def get_cl2(cl_unlen, cl_tot, l2, lmin, lmax):
	
	## If l2 lies in range of known Cl's, linearly interpolate to get cl2	
	cl2_unlen = np.zeros(np.shape(l2))
	cl2_tot = np.zeros(np.shape(l2))
	deltal = np.zeros(np.shape(l2))

	idxs1 = np.where( np.logical_and( lmin < l2, l2 < lmax) )
	idxs2 = np.where( l2 <= lmin )
	idxs3 = np.where( l2 >= lmax )

	lowl = np.floor(l2).astype(int)
	highl = np.ceil(l2).astype(int)
	deltal[idxs1] = l2[idxs1] - lowl[idxs1]
	deltal[idxs2] = lmin - l2[idxs2]
	deltal[idxs3] = l2[idxs3] - lmax
	
	lowl -= lmin
	highl -= lmin

	cl2_tot[idxs1] = cl_tot[lowl[idxs1]] + deltal[idxs1] * (cl_tot[highl[idxs1]] - cl_tot[lowl[idxs1]])
	cl2_unlen[idxs1] = cl_unlen[lowl[idxs1]] + deltal[idxs1] * (cl_unlen[highl[idxs1]] - cl_unlen[lowl[idxs1]]) 
	
	cl2_tot[idxs2] = cl_tot[0] + deltal[idxs2] * (cl_tot[0] - cl_tot[1]) 
	cl2_unlen[idxs2] = cl_unlen[0] + deltal[idxs2] * (cl_unlen[0] - cl_unlen[1])
	
	cl2_tot[idxs3] = cl_tot[lmax-lmin] + deltal[idxs3]*(cl_tot[lmax-lmin] - cl_tot[lmax-lmin-1]) * np.exp(-deltal[idxs3]**2) 
	cl2_unlen[idxs3] = cl_unlen[lmax-lmin] + deltal[idxs3]*(cl_unlen[lmax-lmin] - cl_unlen[lmax-lmin-1]) * np.exp(-deltal[idxs3]**2)
	
	return cl2_unlen, cl2_tot
	
def noise_kernel(theta, l1, L, field, cl_unlen, cl_tot, lmin, lmax):
	
	## theta is the angle between L and l1 (input to function)
	## Going to integrate over (l1, theta) for fixed L
	## The cl array passed contains all field spectra
	## Evaluate dot products and angles for the filters. 
	Ldotl1 = L * l1 * np.cos(theta)
	Ldotl2 = L**2 - Ldotl1
	l1dotl2 = Ldotl1 - l1**2
	l2 = np.sqrt( L**2 + l1**2 - 2*Ldotl1 )
	l2[ np.where(l2 < 0.000001) ] = 0.000001 ## Avoid nasty things
	cos_phi = l1dotl2 / ( l1 * l2 )
	phi = np.arccos( cos_phi )
	cos_2phi = np.cos( 2 * phi )
	sin_2phi = np.sin( 2 * phi )
	
	l1 -= lmin
	
	if field == 'TT': 
		#print len(cl_unlen['TT'])#; quit()
		cl1_unlen = cl_unlen['TT'][l1]
		
		cl1_tot = cl_tot['TT'][l1]
		cl2_unlen, cl2_tot = get_cl2(cl_unlen['TT'], cl_tot['TT'], l2, lmin, lmax)
		kernel = ( cl1_unlen * Ldotl1 + cl2_unlen * Ldotl2 ) ** 2 / (2 * cl1_tot * cl2_tot)
		
	elif field == 'TE': 
		cl2TT = get_cl2(cl_unlen['TT'], cl_tot['TT'], l2, lmin, lmax)[1]
		cl2_unlen, cl2_tot = get_cl2(cl_unlen['TE'], cl_tot['TE'], l2, lmin, lmax)
		cl2EE = get_cl2(cl_unlen['EE'], cl_tot['EE'], l2, lmin, lmax)[1]
		
		cl1_unlen = cl_unlen['TE'][l1]
		f_l1l2 = cl1_unlen * cos_2phi * Ldotl1 + cl2_unlen * Ldotl2
		f_l2l1 = cl2_unlen * cos_2phi * Ldotl2 + cl1_unlen * Ldotl1

		F_l1l2 = (cl_tot['EE'][l1] * cl2TT * f_l1l2 - cl_tot['TE'][l1] * cl2_tot * f_l2l1) / \
				(cl_tot['TT'][l1]*cl2EE*cl_tot['EE'][l1]*cl2TT - (cl_tot['TE'][l1]*cl2_tot)**2)
		kernel = f_l1l2 * F_l1l2
		
	elif field == 'TB': 
		cl1TT = cl_tot['TT'][l1]
		cl2BB = get_cl2(cl_unlen['BB'], cl_tot['BB'], l2, lmin, lmax)[1]
		cl1_unlen = cl_unlen['TE'][l1]
		kernel = (cl1_unlen * Ldotl1 * sin_2phi ) **2 / (cl1TT * cl2BB)
	
	elif field == 'EE':
		cl1_tot = cl_tot['EE'][l1]
		cl1_unlen = cl_unlen['EE'][l1]
		cl2_unlen, cl2_tot = get_cl2(cl_unlen['EE'], cl_tot['EE'], l2, lmin, lmax)
		kernel = ( (cl1_unlen * Ldotl1 + cl2_unlen*Ldotl2) * cos_2phi ) **2 / (2 * cl1_tot * cl2_tot)
	
	elif field == 'EB':
		cl1_unlen = cl_unlen['EE'][l1]
		cl1EE = cl_tot['EE'][l1]
		cl2_unlen, cl2BB = get_cl2(cl_unlen['BB'], cl_tot['BB'], l2, lmin, lmax)
		f_l1l2 = (cl1_unlen * Ldotl1 - cl2_unlen * Ldotl2) * sin_2phi
		kernel = (f_l1l2 ** 2) / (cl1EE * cl2BB)
	
	elif field == 'BB':
		cl1_tot = cl_tot['BB'][l1]
		cl1_unlen = cl_unlen['BB'][l1]
		cl2_unlen, cl2_tot = get_cl2(cl_unlen['BB'], cl_tot['BB'], l2, lmin, lmax)
		kernel = ( (cl1_unlen * Ldotl1 + cl2_unlen*Ldotl2) * cos_2phi ) **2 / (2 * cl1_tot * cl2_tot)
	
	kernel *= l1 * (2 * np.pi)**(-2.)
	
	return kernel

def get_lensing_noise(l_array, cl_fid, nl, fields, q):
	
	lmin = int(l_array[0])
	lmax = int(l_array[-1])
	
	cl_tot = {}
	for key in fields:
	#for key in ['TT','EE','BB','TE','Td','dd']:
		if key != 'dd' and key !='Td': cl_tot[key] = cl_fid[key] + nl[key]

	cl_unlen = {}
	for key in fields:
	#for key in ['TT','EE','BB','TE','Td','dd']:
		if key != 'dd' and key !='Td': cl_unlen[key] = cl_fid[key+'_unlen']
	# B.H.
	#print len(cl_unlen['TT']); 
	#print len(cl_unlen['EE']); quit()
	## n_Ls specifies approx number of L's at which to 
	## evaluate the lensing noise spectrum.
	## Need unique L's for the splining used later.
	
	n_Ls = 50
	LogLs = np.linspace(np.log(2),np.log(lmax+0.1), n_Ls)
	Ls = np.unique(np.floor(np.exp(LogLs)).astype(int))
		
	lensingSpec = []
	if 'TT' in fields : 
		lensingSpec += ['TT']
	if 'TE' in fields:
		lensingSpec += ['TE','TB']
	if 'EE' in fields: # B.H. if only EE rm EB and fields BB
		lensingSpec += ['EE', 'EB']
	#lensingSpec = ['TT','TE','EE','EB','TB']
	NL = {}
	## Added RA 16/01/14: Only use T,P modes with L_lens_min < l < L_lens_max 
	## for the lensing reconstruction.
	L_lens_min = q['L_lens_min']
	L_lens_max_temp = q['L_lens_max_temp']
	L_lens_max_pol = q['L_lens_max_pol']

	for field in lensingSpec:
		y = []
		for L in Ls:
			N = 50 ## Number of grid points in theta (per l1)
			deltaTheta = 2 * np.pi / N
			thetas = np.arange(0., 2*np.pi, deltaTheta)
			ells = range(lmax-lmin+1)
			if field == 'TT':
				Theta, Ells = np.meshgrid(thetas, l_array.astype(int)[L_lens_min - lmin:L_lens_max_temp - lmin])
			else:
				Theta, Ells = np.meshgrid(thetas, l_array.astype(int)[L_lens_min - lmin:L_lens_max_pol - lmin])
			kernel_grid = noise_kernel(Theta, Ells, L, field, cl_unlen, cl_tot, lmin, lmax)
			integral = deltaTheta * np.sum(np.sum(kernel_grid[np.where(kernel_grid > 0)], axis = 0), axis = 0)
			y.append(L*L*(integral)**(-1))
		NL[field] = np.array(y)
	
	NL_mv = np.zeros(np.shape(Ls))
	
	## Assumes negligible correlation between noise terms
	for field in lensingSpec:
		NL_mv += 1./NL[field]
	NL_mv = 1./NL_mv

	## Interpolate onto integer values in [lmin, lmax]
	## Interpolate the logarithms for stability
	logLs = np.log(np.arange(lmin, lmax+1, 1))
	s = spline(np.log(Ls), np.log(NL_mv), s=0)
	newNL_mv = np.exp(s(logLs))
	
	## Convert to the convergence noise spectrum (from
	## the deflection noise spectrum)
	Ls = np.exp(logLs)
	NL_KK = (Ls + 1) **2 * newNL_mv / 4.

	return Ls, NL_KK

## Read in ells, unlensed, lensed and noise spectra (TT, EE, ...)
## and define which spectra we have measured (fields)
# original
# B.H. for TT
fields = ['TT']#['TT','EE','BB','TE']
# B.H. modification for comparing only polarization noise
#fields = ['EE','BB']
cl, nl = {}, {} 
#file = '/Users/blakesherwin/planckAlone'
file = '/users/boryanah/Blake/noisecalc/Aug6_highAcc_CDM'
ells,cl['TT'],cl['EE'],cl['BB'],cl['TE'] = np.loadtxt(file+'_lensedCls_new.dat',unpack=True)
#print len(cl['EE']); quit()
ellsx,cl['TT_unlen'],cl['EE_unlen'],cl['TE_unlen'], cl_phiphi, cl_Tphi = np.loadtxt(file+'_scalCls_new.dat',unpack=True)

cl['TT'] *= 2. * np.pi / ells / (ells + 1.)
cl['TT_unlen'] *= 2. * np.pi / ellsx / (ellsx + 1.)
cl['EE'] *= 2. * np.pi / ells / (ells + 1.)
cl['EE_unlen'] *= 2. * np.pi / ellsx / (ellsx + 1.)
cl['BB'] *= 2. * np.pi / ells / (ells + 1.)
cl['BB_unlen'] = cl['TT_unlen']*0.
cl['TE'] *= 2. * np.pi / ells / (ells + 1.)
cl['TE_unlen'] *= 2. * np.pi / ellsx / (ellsx + 1.)
# B.H.
cl['TT_unlen'] = cl['TT_unlen'][:len(cl['TT'])]
cl['EE_unlen'] = cl['EE_unlen'][:len(cl['EE'])]
cl['BB_unlen'] = cl['BB_unlen'][:len(cl['BB'])]
cl['TE_unlen'] = cl['TE_unlen'][:len(cl['TE'])]
cl_phiphi = cl_phiphi[:len(cl['TE'])]

ells = ells[:len(cl['TE_unlen'])]
ellsx = ellsx[:len(cl['TE_unlen'])]
print len(cl['TT'])
print len(cl['TT_unlen'])
import pickle
## Read in the noise spectra here; set to zero for CV limited exp.
for noiseFactor in [.1]:
	print noiseFactor
	thetaFWHMarcmin = 1
	noiseUkArcmin = 1.
	noiseUk = noiseUkArcmin*(numpy.pi/(180.*60.))
	thetaFWHM = thetaFWHMarcmin*numpy.pi/(180.*60.)
	deltaT = noiseUk/thetaFWHM
	ll = ells
	nlI = (deltaT*thetaFWHM)**2*numpy.exp(ll*(ll+1.)*thetaFWHM**2/(8.*numpy.log(2.)))*noiseFactor**2.#/TCMB**2
	npI = nlI*2.

	nl['TT'] = nlI
	nl['EE'] = npI
	nl['BB'] = npI
	nl['TE'] = npI*0.
	

	# B.H.
	L_max = 20000#30000
	## Define which scales to use in reconstruction.
	q = { 	'L_lens_min' : 2,
			'L_lens_max_temp' : L_max,
			'L_lens_max_pol' : L_max,
		}
	## Calculate and plot quadratic estimator noise
	Ls, NL_KK = get_lensing_noise(ells, cl, nl, fields, q)

	# tuk i think it needs to be divided by tcmb**2
	plt.semilogy(Ls, NL_KK*4./2./numpy.pi,label=str(noiseFactor)+'QE', color = str((noiseFactor/2.)**0.25))
	
	### BDS Note: most of the code until this point was just to get the quadratic estimator noise. The following is the gradient inversion noise written up in my notes. I am assuming a gradient = RMS = 13 / arcmin
	# B.H. 
	# for temperature used to be 13 added factor of 2
	plt.semilogy(ellsx,ellsx**2.*((cl['TT_unlen']+numpy.interp(ellsx,ll,nlI))/(13.*10800./np.pi)**2./2./numpy.pi),label=str(noiseFactor)+'GI')
	# for polarization grad P = 2 uK-arcmin
	n_p_l = numpy.interp(ellsx,ll,npI)
	fac = (numpy.abs(cl['EE_unlen'])+n_p_l)*n_p_l
	den = 2.*n_p_l+numpy.abs(cl['EE_unlen'])#numpy.sqrt((numpy.abs(cl['EE_unlen'])+n_p_l)**2+n_p_l**2)
	fac /= den
	#plt.semilogy(ellsx,ellsx**2.*fac*4./(2.*10800./3.14)**2.,label=str(noiseFactor)+'GI')

# temperature used to be 13
plt.semilogy(ellsx,ellsx**2.*((cl['TT_unlen']+numpy.interp(ellsx,ll,nlI)*0.)/(2*np.pi)),label='TT')
# polarization
#plt.semilogy(ellsx,ellsx**2.*((2.*numpy.abs(cl['EE_unlen'])+2.*numpy.interp(ellsx,ll,nlI)*0.)/(1.95*10800./3.14)**2.),label='EE')
plt.semilogy(ellsx,cl_phiphi/2./numpy.pi/TCMB**2,ls='dashed',color='k',label='Lensing power')
plt.semilogy(ellsx,ellsx**2.*numpy.interp(ellsx,ll,nlI)/(2*np.pi),ls='dashed',color='y',label='InstNoise')
# i think this is in correct units without division by tcmb**2
plt.errorbar(lls,CMEAN,yerr=CSTD,label='SimConstGrad')
plt.yscale('log')
plt.xlim(1, 10000)
#plt.ylim(2.e-16,2.e-7)
plt.legend(loc='best')
plt.xlabel(r'$L$')
plt.ylabel(r'$l^2 N_l^{dd}/2 \pi$')
plt.savefig('lensingNoise_0_1_noise.pdf')
