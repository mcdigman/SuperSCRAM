import numpy as np
#from talk_to_class import class_objects 
from FASTPTcode import FASTPT 

from cosmopie import CosmoPie 
import sph_basis as basis
from sph import sph_basis
from time import time 
from Dn import DO_n
import Dn
import shear_power as sh_pow
from hmf import ST_hmf

import sys
from time import time


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
	    
	    t1=time()
	    	    
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
		self.F_0=self.basis.get_F_alpha_beta()
			
		self.cov_mit,self.a_mit=self.get_SSC_covar(mitigation=True)	
		self.cov_no_mit,self.a_no_mit=self.get_SSC_covar(mitigation=False)	
		
		t2=time()
		print 'all done'
		print 'run time', t2-t1
	
	
	def get_SSC_covar(self,mitigation=True):
	    
	    result=np.zeros_like(2,dtype=object)
            if mitigation:
                print "mit"
	        x=self.O_a_data[0]
            else:
                print "no mit"
                x=np.zeros(2)
	    #print x[0]
	    #print self.F_0
	    #print 'here'
	    #sys.exit()
	    F_1=x[0] + self.F_0
	    F_2=x[1] + self.F_0
            print "x[0]: ",x[0]
            print "F_1: ",F_1
	    C_1=np.linalg.pinv(F_1)
	    C_2=np.linalg.pinv(F_2)
           	    
	    #print C_1 
	    #print C_1.shape, C_2.shape
	    C=np.array([C_1,C_2],dtype=object)
	    Cov_SSC = np.zeros((2,2),dtype=object) 
	    a_SSC = np.zeros((2,2)) 
	    for i in range(2):
	        x=self.O_I_data[i]
	        T=x['shear_shear']['ddelta_dalpha']
	        dCdbar=x['shear_shear']['dc_ddelta']
	        for j in range(2):
	            a_SSC[i,j]=np.dot(T[j],np.dot(C[j],T[j]))
	            print "max t",max(T[j])
                    print "max C",np.max(C[j])

                    print "a is: ",a_SSC[i,j]
	            Cov_SSC[i,j]=np.outer(dCdbar[j],dCdbar[j])*a_SSC[i,j]
	            #print Cov_SSC[i,j]
	            #print Cov_SSC[i,j].shape
	          #  np.savetxt('covar_SCC_0.dat',C_SSC)
	        #    sys.exit()
	    #sys.exit()
	    return Cov_SSC,a_SSC
	def get_O_a(self):
		D_O_a=np.zeros(self.N_O_a,dtype=object)
		print 'I have this many long wavelength observables', self.N_O_a 
		for i in range(self.N_O_a):
		    O_a=self.O_a[i]
		    for key in O_a:
		        
		        if key=='number density':
		            print 'hello there'
		            data=O_a['number density']
		            n_obs=np.array([data[0],data[1]])
		            mass=data[2]
		            print n_obs, mass
		            X=DO_n(n_obs,self.zbins_lw,mass,self.CosmoPie,self.basis,self.geo_lw)
		            D_O_a[i] = X.Fisher_alpha_beta()
                            print D_O_a[i]
		# x=D_O_a[0]
# 		print x[0]
# 		print x[1]
# 		print 'hi joe'
# 		print x.shape
# 		sys.exit()
		return D_O_a  
		
		    
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
		            ddelta_dalpha=np.zeros(z_bins.size-1,dtype=object)
		            geo=data['geo']
		            Theta=geo[0]; Phi=geo[1]
		            
		         
		            ls=data['l']
                            zs = np.arange(0.1,2.0,0.1)

                            sp1 = sh_pow.shear_power(k,self.CosmoPie,zs,ls,P_in=P_lin,pmodel='dc_halofit')
                            sp2 = sh_pow.shear_power(k,self.CosmoPie,zs,ls,P_in=P_lin,pmodel='halofit_nonlinear')

                            dcs = np.zeros((z_bins.size-1,ls.size))
                            cs = np.zeros((z_bins.size-1,ls.size))
                            #covs = np.array([],dtype=object)

                            #sh_pows = np.array([],dtype=object)
                            for j in range(0,z_bins.size-1):
                                chi_min = self.CosmoPie.D_comov(z_bins[j])
                                chi_max = self.CosmoPie.D_comov(z_bins[j+1])
                                #sh_pow2 = sh_pow.Cll_sh_sh(sp2,chi_max,chi_max,chi_min,chi_min).Cll()
                                cs[j] = sh_pow.Cll_sh_sh(sp2,chi_max,chi_max,chi_min,chi_min).Cll()
                                dcs[j] = sh_pow.Cll_sh_sh(sp1,chi_max,chi_max,chi_min,chi_min).Cll()
                                ddelta_dalpha[j]=self.basis.D_delta_bar_D_delta_alpha(chi_min,chi_max,Theta,Phi)
                                #print ddelta_dalpha[j]
                                #print self.basis.C_id
                             
		            
                             #   covs = np.append(covs,np.diagflat(sp2.cov_g_diag(sh_pow2,sh_pow2,sh_pow2,sh_pow2))) #TODO support cross z_bin covariance correctly
                            covs = sp2.cov_mats(z_bins,cname1='shear',cname2='shear')
                            result[i][key]['power'] = cs
                            result[i][key]['dc_ddelta'] = dcs
                            result[i][key]['covariance'] = covs
                            result[i][key]['ddelta_dalpha']=ddelta_dalpha
		            #print n_obs, mass
		            #result=DO_n(n_obs,self.zbins_lw,mass,self.CosmoPie,self.basis,self.geo_lw)
		    
		
		return result
		    
		
		
		
if __name__=="__main__":

	z_max=0.5; l_max=20 
	#d=np.loadtxt('Pk_Planck15.dat')
        d=np.loadtxt('camb_m_pow_l.dat')
	k=d[:,0]; P=d[:,1]
	cp=CosmoPie(k=k,P_lin=P)
	r_max=cp.D_comov(z_max)
	print 'this is r max and l_max', r_max , l_max
	
	Theta=[np.pi/4.,np.pi/2.]
	Phi=[0.,np.pi/3.]
	geo=np.array([Theta,Phi])
	zbins=np.array([.1,.2,.3])
	l=np.logspace(np.log10(2),np.log10(3000),1000)
	
	shear_data1={'z_bins':zbins,'l':l,'geo':geo}
	shear_data2={'z_bins':zbins,'l':l,'geo':geo}
	
	O_I1={'shear_shear':shear_data1}
	O_I2={'shear_shear':shear_data2}
	

	#n_dat2=np.array([1.01*5.*1e5/1.7458,1.01*5.0*1e5*1.4166*0.98546])
	M_cut=10**(12.5)
	
        n_avg = np.zeros(2)
        n_avg[0] = ST_hmf(cp).n_avg(M_cut,0.15)
        n_avg[1] = ST_hmf(cp).n_avg(M_cut,0.25)

        V1 = Dn.volume(cp.D_comov(0.1),cp.D_comov(0.2),Theta,Phi)
        V2 = Dn.volume(cp.D_comov(0.2),cp.D_comov(0.3),Theta,Phi)
        #amplitude of fluctuations within a given radius should be given by something like np.sqrt(np.trapz((3.*(sin(k*R)-k*R*cos(k*R))/(k*R)**3)**2*P*k**2,k)*4.*np.pi/(2.*np.pi)**3), where R is the radius of the bin, R=cp.D_comov(z[i])-cp.D_comov(z[i-1])
        #cf https://ned.ipac.caltech.edu/level5/March01/Strauss/Strauss2.html
        #~0.007 for R~400 (check units).
	n_dat1=np.array([n_avg[0]*V1,n_avg[1]*V2])
	n_dat2=np.array([1.1*n_avg[0]*V1,1.1*n_avg[1]*V2])

	O_a={'number density':np.array([n_dat1,n_dat2,M_cut])}
	
	d_1={'name': 'survey 1', 'area': 18000}
	d_2={'name': 'survey 2', 'area': 18000}
	d_3={'name': 'suvery lw', 'area' :18000}
	
	
	survey_1={'details':d_1,'O_I':O_I1, 'geo':geo}
	survey_2={'details':d_2,'O_I':O_I2, 'geo':geo}
	
	surveys_sw=np.array([survey_1, survey_2])
	
	survey_3={'details':d_3, 'O_a':O_a, 'zbins':zbins,'geo':geo}
	surveys_lw=np.array([survey_3])
	
        l=np.arange(0,5)
	n_zeros=8
	
	print 'this is r_max', r_max 
	
	SS=super_survey(surveys_sw, surveys_lw,r_max,l,n_zeros,k,P_lin=P)
	
        print "fractional mitigation: ", SS.a_no_mit/SS.a_mit	

        import matplotlib.pyplot as plt
        ax = plt.subplot(111)
	l=np.logspace(np.log10(2),np.log10(3000),1000)
        cov_ss = np.diag(SS.O_I_data[0]['shear_shear']['covariance'][0,0])
        c_ss = SS.O_I_data[0]['shear_shear']['power'][0]
        ax.loglog(l,cov_ss)
        #ax.loglog(l,np.diag(SS.cov_mit[0,0])/c_ss**2*l/2)
        ax.loglog(l,(np.diag(SS.cov_mit[0,0])))
        ax.loglog(l,(np.diag(SS.cov_no_mit[0,0])))
        ax.legend(['cov','mit','no mit'])
        plt.show()
