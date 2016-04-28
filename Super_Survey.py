import numpy as np
from talk_to_class import class_objects 
from FASTPTcode import FASTPT 


class super_survey:
	''' This class holds and returns information for all surveys
	''' 
	def __init__(self, surveys,cosmology=None):
	
	
		if (cosmology is None): 
			cosmology={
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
				
		COSMO=class_objects(cosmology) 
		
		self.N_surveys=len(surveys) 
		self.N_tot_obs=0
		for i in range(self.N_surveys):
			self.N_tot_obs=self.N_tot_obs + surveys[i].N_obs
			
		
		

if __name__=="__main__":

	k=np.logspace(-2,2,500)

	C=class_objects()
	P=C.linear_power(k)  	
	# 	# set the parameters for the power spectrum window and
	# 	# Fourier coefficient window 
	P_window=np.array([.2,.2])  
	C_window=.75	
	# 	
	# 	# bias parameter and padding length 
	nu=-2; n_pad=800
	# 
	from time import time
	# 		
	# 	# initialize the FASTPT class		
	fastpt=FASTPT.FASTPT(k,nu,n_pad=n_pad) 
	# 	
	# 	
	t1=time()	
	# 	
	# 	# with out P_windowing (better if you are using zero padding) 
	P_spt=fastpt.one_loop(P,C_window=C_window) 
	t2=time()
	print('time'), t2-t1 
 