import numpy as np
cimport numpy as np 

ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE_i

def F_2(r,mu):
	return (7*mu+3*r-10*r*mu**2)/(1+r**2-2*r*mu)/(14.*r)
		

def SPT(np.ndarray[DTYPE_t, ndim=1] k): 

	cdef int a,b,c
	cdef float mu_min, mu_max
	cdef float r_min, r_max 
		
	cdef np.ndarray[DTYPE_t,ndim=1] P_kr
	cdef np.ndarray[DTYPE_t,ndim=1] P_mukr
	cdef np.ndarray[DTYPE_t,ndim=1] P_sample
	cdef np.ndarray[DTYPE_t,ndim=1] P_22=np.zeros(k.size, 'float64')
	
	cdef np.ndarray[DTYPE_t,ndim=1] hold
	cdef np.ndarray[DTYPE_t,ndim=1] q=k
	
	

	import class_interface as iclass 
	power=iclass.class_items()
	P_sample=power.linear_power(k)
	
	from scipy.integrate import trapz
	
	for a in range(q.size):
		
		hold=(k**2+q[a]**2 -np.max(k)**2)/(2*k*q[a])
		hold=np.hstack((hold,1))
		mu_min=np.max(hold)
		
		hold=(k**2+q[a]**2 -np.min(k)**2)/(2*k*q[a])
		hold=np.hstack((hold,1))
		mu_max=np.max(hold)
		
		print mu_min , mu_max 
		
		
				
	return P_22
	
