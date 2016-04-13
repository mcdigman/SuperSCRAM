import numpy as np
from FASTPTcode import FASTPT 
from talk_to_class import class_objects 



class survey:

	''' this is the class that will make a particular survey an object. 
	
		The inputs are: 
		 
			* observables, which is a python list of obervables in the survey. For instance, 
				galaxy power spectrum, galaxy-galaxy lensing, shear, etc. The list contains the 
				name of the obervable and the parameters for the observable 
		
			* geometry, which is the geometry of the survey. 
		
			* optionals, which is an optional argument (taken in as a list). 
				optionals can contain things like; redshift, number density of galaxies, 
				sky coverage, mask 
	''' 
		
	
	def __init__(self, observables,geometry, optionals=[], cosmology=None):
	
	
		if (cosmology is None): 
			# if no cosmology is inputed, then use 
			# Planck 2015 cosmology 
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
		 
	
		# the number of observables in the survey
		self.N_obs=len(observables)
		self.geo_type=geometry['type']
		self.geo_params=geometry['params']
		self.O_I=np.zeros(self.N_obs) 
		
		
		I=0
		for obs in observables: 
			
			print 'hi joe', obs
			if obs=='PS_galaxy':
				k=np.logspace(-1,1,100)
				P_lin=COSMO.linear_power(k)
				self.O_I[I]=P_lin 
			
			I=I+1

		print self.O_I 
			  
		# optional parameters of the survey. 
		for opt in optionals:
			if opt=='z_bins':
				self.z_bins=optionals[opt]
				
			elif opt=='n_gal':
				self.n_gal=optionals[opt]
				
			elif opt=='f_sky':
				self.f_sky=optionals[opt]
				
			elif opt=='gal_bias':
				self.gal_bias=optionals[opt] 
				self.gal_bias_type=self.gal_bias['type'] 
				self.gal_bias_params=self.gal_bias['params']
			
			elif opt=='selection_func':
				self.selection_func=optionals[opt]
				
			elif opt=='mask':
				self.mask=optionals[opt]
			
			elif opt=='galaxy_sample_bias':
				self.galaxy_sample_bias=optionals[opt]
				
			
							  
if __name__=="__main__": 
	
	
	l=np.arange(50,3001); n=np.arange(0,10)
	z=[.1,.2,.3]
	O_1={'PS_galaxy':[.1,1],'xi_galaxy':[1,n]} 
	geo_1={'type':'spherical','params': [l,z]}
	
	gal_bias_1={'type':'HOD', 'params': [.5,12,12,12,.95]} 
	opt_1={'n_gal':1e-4, 'z_bins':z, 'gal_bias':gal_bias_1} 
	
	
	# make survey_1 class 
	survey_1=survey(O_1,geo_1,optionals=opt_1)
	
	# check that the class was built properly 
	print 'number of observables in the survey', survey_1.N_obs
	print 'redshift bins ', survey_1.z_bins
	print 'geometry type is', survey_1.geo_type
	print 'the galaxy bias type is ', survey_1.gal_bias_type 
	
	
	from Super_Survey import super_survey
	
	surveys=[survey_1, survey_1]
	S=super_survey(surveys)
	print 'number of surveys', S.N_surveys
	print 'total number of surveys', S.N_tot_obs
	
	