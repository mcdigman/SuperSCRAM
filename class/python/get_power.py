from classy import Class
import numpy as np 
import sys

class class_interface:
    def __init__(self,cosmology=None):
		
		if (cosmology==None):
		    # default to planck cosmology 
		    params={
 			    'output' : 'tCl lCl,mPk, mTk', 
 			    'l_max_scalars' : 2000, 
 			    'z_pk' : 0, 
 			    'A_s': 2.3e-9, 
 			    'n_s' : 0.9624, 
 			    'h' : 0.6774,
 			    'omega_b' : 0.02230,
 			    'omega_cdm': 0.1188, 
 			    'k_pivot' : 0.05,
			    'A_s' : 2.142e-9,
			    'n_s' : 0.9667,
			    'P_k_max_1/Mpc' : 500.0,
			    'N_eff' :3.04,
			    'Omega_fld' : 0,
			    'YHe' : 0.2453,
			    'z_reio' : 8.8}
        
		else:
		    params=cosmology 
		
		self.k_max=params['P_k_max_1/Mpc']
		
		self.cosmo=Class()
		self.cosmo.set(params)
		self.cosmo.compute()
		
    def linear_power(self,k):
        if k[-1] > self.k_max:
            print 'power spectrum requested is out of bounds'
            print 'adjust parameters sent to class'
            sys.exit()
        P=np.zeros(k.size)
        for i in range(k.size):
            P[i]=self.cosmo.pk(k[i],0)
        return P 