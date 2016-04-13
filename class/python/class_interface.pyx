import numpy as np
cimport numpy as np 
cimport cython

ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE_i

from classy import Class

#ombh2          = 0.02230
#omch2          = 0.1188
#omnuh2         = 0.00064
#temp_cmb           = 2.7255


class class_items:
	
	def __init__(self, params=None):
	
		if params is None:
			# default cosmology 
			self.params={
 			'output' : 'tCl lCl,mPk, mTk', 
 			'l_max_scalars' : 4000, 
 			'z_pk' : 0, 
 			'h' : 0.6774,
 			'omega_b' : 0.02230,
 			'omega_cdm': 0.1188, 
 			'k_pivot' : 0.05,
			'A_s' : 2.142e-9,
			'n_s' : 0.9667,
			'P_k_max_1/Mpc' : 1000.0,
			'N_eff' :3.04,
			'Omega_fld' : 0,
			'YHe' : 0.2453,
			'z_reio' : 8.8,
			'T_cmb': 2.7255 }

		
		else:
			self.params=params
	
		self.cosmo=Class()
		self.cosmo.set(self.params)
		self.cosmo.compute()

	def linear_power(self,np.ndarray[DTYPE_t,ndim=1] k):
		
		cdef int i
		cdef np.ndarray[DTYPE_t,ndim=1] P=np.zeros(k.size, 'float64')
		for i in range(k.size):
			if (k[i] > 500): 
				P[i]=self.cosmo.pk(500,0)*(500.**3)/k[i]**3
								
			elif (k[i] < .0001):
				P[i]=self.cosmo.pk(.0001,0)/(.0001)*k[i]
				
			else:
				
				P[i]=self.cosmo.pk(k[i],0)
			
		return P 
		
	def linear_power_grid(self,np.ndarray[DTYPE_t,ndim=1] k, np.ndarray[DTYPE_t,ndim=1] q, \
			np.ndarray[DTYPE_t,ndim=1] mu ):
		
		cdef int a, b 
		cdef np.ndarray[DTYPE_t,ndim=1] P_q=np.zeros(q.size, 'float64')
		cdef np.ndarray[DTYPE_t,ndim=3] P_kq=np.zeros((k.size,q.size,mu.size), 'float64')
		cdef np.ndarray[DTYPE_t,ndim=1] psi=np.zeros(mu.size, 'float64')
		
		for a in range(k.size):
			for b in range(q.size):
				psi=np.sqrt(k[a]**2+q[b]**2 - 2*k[a]*q[b]*mu)
				
				P_kq[a,b,:]=self.linear_power(psi) 
		
		P_q=self.linear_power(q)
			
		return P_q, P_kq 
		
		