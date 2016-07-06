import numpy as np
#from talk_to_class import class_objects 
from FASTPTcode import FASTPT 

from cosmopie import CosmoPie 
import sph_basis as basis
from sph import sph_basis
from time import time 
from Dn import DO_n
import shear_power as sh_pow

import sys


class super_survey:
	''' This class holds and returns information for all surveys
	''' 
	def __init__(self, surveys_sw, surveys_lw, r_max, l, n_zeros,k,cosmology=None,P_lin=None):
	    
	    '''
	    1) surveys is an array containing the informaiton for all the surveys.
	    2) r_max is the maximum radial super mode 
	    3) l angular super modes
	    4) cosmology is the comological parameters etc. 
	    ''' 
	    	    
	    self.N_surveys_sw=surveys_sw.size
	    self.N_surveys_lw=surveys_lw.size
	    self.geo_lw=surveys_lw[0]['geo']	
	    self.zbins_lw=surveys_lw[0]['zbins']
	    print'this is the number of surveys', self.N_surveys_sw, self.N_surveys_lw
	      
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
				
				
		
		self.CosmoPie=CosmoPie(k=k,P_lin=P_lin)
	    
				
		self.basis=sph_basis(r_max,l,n_zeros,self.CosmoPie)
				
		self.N_O_I=0
		self.N_O_a=0   
		self.O_I=np.array([], dtype=object)
		self.O_a=np.array([], dtype=object)
		for i in range(self.N_surveys_sw):
		
		    survey=surveys_sw[i]
		    self.N_O_I=self.N_O_I + len(survey['O_I'])		    
		    self.O_I=np.append(self.O_I,survey['O_I'])
		    
		for i in range(self.N_surveys_lw):
		
		    survey=surveys_lw[i]
		    self.N_O_a=self.N_O_a + len(survey['O_a']) 
		    self.O_a=np.append(self.O_a,survey['O_a'])
		    
		print('these are number of observables', self.N_O_I, self.N_O_a)
		self.O_a_data=self.get_O_a()
		self.O_I_data=self.get_O_I(k,P_lin)
		
	
	
	def get_O_a(self):
		D_O_a=np.array([],dtype=object)
                result = np.array([],dtype=object)
		for i in range(self.N_O_a):
		    O_a=self.O_a[i]
		    result = np.append(result,{}) 
		    for key in O_a:
		        
		        if key=='number density':
		            print 'hello there'
		            data=O_a['number density']
		            n_obs=np.array([data[0],data[1]])
		            mass=data[2]
		            print n_obs, mass
		            result[i][key] = DO_n(n_obs,self.zbins_lw,mass,self.CosmoPie,self.basis,self.geo_lw)
		    
		
		return result  
		    
	def get_O_I(self,k,P_lin):
		D_O_I=np.array([],dtype=object)
                result = np.array([],dtype=object)
		for i in range(self.N_O_I):
		    O_I=self.O_I[i] 
		    result = np.append(result,{}) 
		    for key in O_I:
		        result[i][key]={}
		        if key=='shear_shear':
                            print key
		            data=O_I[key]
		            z_bins=data['z_bins']
		            ls=data['l']
                            zs = np.arange(0.1,2.0,0.1)

                            sp1 = sh_pow.shear_power(k,self.CosmoPie,zs,ls,P_in=P_lin,pmodel='dc_halofit')
                            sp2 = sh_pow.shear_power(k,self.CosmoPie,zs,ls,P_in=P_lin,pmodel='halofit_nonlinear')

                            dcs = np.zeros((z_bins.size-1,ls.size))
                            #covs = np.array([],dtype=object)

                            #sh_pows = np.array([],dtype=object)
                            for j in range(0,z_bins.size-1):
                                chi_min = self.CosmoPie.D_comov(z_bins[j])
                                chi_max = self.CosmoPie.D_comov(z_bins[j+1])
                                #sh_pow2 = sh_pow.Cll_sh_sh(sp2,chi_max,chi_max,chi_min,chi_min).Cll()
                                dcs[j] = sh_pow.Cll_sh_sh(sp1,chi_max,chi_max,chi_min,chi_min).Cll()
                             #   covs = np.append(covs,np.diagflat(sp2.cov_g_diag(sh_pow2,sh_pow2,sh_pow2,sh_pow2))) #TODO support cross z_bin covariance correctly
                            covs = sp2.cov_mats(z_bins,cname1='shear',cname2='shear')
                            result[i][key]['dc_ddelta'] = dcs
                            result[i][key]['covariance'] = covs
		            #print n_obs, mass
		            #result=DO_n(n_obs,self.zbins_lw,mass,self.CosmoPie,self.basis,self.geo_lw)
		    
		
		return result
		    
		
		
		
if __name__=="__main__":

	z_max=4; l_max=20 
	cp=CosmoPie()
	r_max=cp.D_comov(z_max)
	print 'this is r max and l_max', r_max , l_max
	
	Theta=[np.pi/4,np.pi/2.]
	Phi=[0,np.pi/3.]
	geo=np.array([Theta,Phi])
	zbins=np.array([.1,.2,.3])
	l=np.logspace(np.log10(2),np.log10(3000),1000)
	
	shear_data1={'z_bins':zbins,'l':l}
	shear_data2={'z_bins':zbins,'l':l}
	
	O_I1={'shear_shear':shear_data1}
	O_I2={'shear_shear':shear_data2}
	
	n_dat1=np.array([1e3,2.5*1e3])
	n_dat2=np.array([5*1e3,7*1e3])
	M_cut=1e15
	
	O_a={'number density':np.array([n_dat1,n_dat2,M_cut])}
	
	d_1={'name': 'survey 1', 'area': 18000}
	d_2={'name': 'survey 2', 'area': 18000}
	d_3={'name': 'suvery lw', 'area' :18000}
	
	
	survey_1={'details':d_1,'O_I':O_I1, 'geo':geo}
	survey_2={'details':d_2,'O_I':O_I2, 'geo':geo}
	
	surveys_sw=np.array([survey_1, survey_2])
	
	survey_3={'details':d_3, 'O_a':O_a, 'zbins':zbins,'geo':geo}
	surveys_lw=np.array([survey_3])
	
	d=np.loadtxt('Pk_Planck15.dat')
	k=d[:,0]; P=d[:,1]
	l=np.arange(0,5)
	n_zeros=5
	
	print 'this is r_max', r_max 
	
	SS=super_survey(surveys_sw, surveys_lw,r_max,l,n_zeros,k,P_lin=P)
	
	
