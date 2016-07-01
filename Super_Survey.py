import numpy as np
from talk_to_class import class_objects 
from FASTPTcode import FASTPT 

from cosmopie import CosmoPie 
import sph_basis as basis
import sph as basis
from time import time 


class super_survey:
	''' This class holds and returns information for all surveys
	''' 
	def __init__(self, surveys, r_max, l, k,cosmology=None,P_lin=None):
	    '''
	    1) surveys is an array containing the informaiton for all the surveys.
	    2) r_max is the maximum radial super mode 
	    3) l angular super modes
	    4) cosmology is the comological parameters etc. 
	    ''' 
	    
	    
	    t1=time()
	    self.N_surveys=surveys.size
	    print'this is the number of surveys', self.N_surveys
	    
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
				
		self.N_O_I=0
		self.N_O_a=0   
		for i in range(self.N_surveys):
		
		    survey=surveys[i]
		    self.N_O_I=self.N_O_I + len(survey['O_I'])
		    self.N_O_a=self.N_O_a + len(survey['O_a']) 
		    
		print('these are number of observables', self.N_O_I, self.N_O_a)
		
		
if __name__=="__main__":

	z_max=4; l_max=20 
	cp=CosmoPie()
	r_max=cp.D_comov(z_max)
	print 'this is r max and l_max', r_max , l_max
	
	zbins=np.array([.1,.2,.3])
	l=np.logspace(np.log10(2),np.log10(3000),1000)
	
	shear_data1={'z_bins':zbins,'l':l}
	shear_data2={'z_bins':zbins,'l':l}
	
	O_I1={'shear':shear_data1}
	O_I2={'shear':shear_data2}
	
	n_dat1=np.array([1e3,2.5*1e3])
	n_dat2=np.array([5*1e3,7*1e3])
	
	O_a1={'number density':n_dat1}
	O_a2={'number density':n_dat2}
	
	d_1={'name': 'survey 1', 'area': 18000}
	d_2={'name': 'survey 2', 'area': 18000}
	
	
	survey_1={'details':d_1,'O_I':O_I1, 'O_a':O_a1}
	survey_2={'details':d_2,'O_I':O_I2, 'O_a':O_a2}
	
	surveys=np.array([survey_1, survey_2])
	
	d=np.loadtxt('Pk_Planck15.dat')
	k=d[:,0]; P=d[:,1]
	l=np.arange(1,10)
	
	SS=super_survey(surveys,r_max,l,k,P_lin=P)
	
	