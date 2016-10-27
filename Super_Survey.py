import numpy as np
#from talk_to_class import class_objects 
from FASTPTcode import FASTPT 

from cosmopie import CosmoPie 
#import sph_basis as basis
from sph_klim import sph_basis_k
from time import time 
from Dn import DNumberDensityObservable
import Dn
import shear_power as sh_pow
from hmf import ST_hmf

import sys
from geo import rect_geo
from algebra_utils import ch_inv ,get_inv_cholesky
import defaults
import fisher_matrix as fm
import lensing_observables as lo
import copy
from sw_survey import SWSurvey
from lw_survey import LWSurvey
class super_survey:
    ''' This class holds and returns information for all surveys
    ''' 
    def __init__(self, surveys_sw, surveys_lw, r_max, l, n_zeros,k,basis,C,P_lin=None,get_a=False,do_mitigated=True,do_unmitigated=True):
    
      
        '''
        1) surveys is an array containing the informaiton for all the surveys.
        2) r_max is the maximum radial super mode 
        3) l angular super modes
        4) cosmology is the comological parameters etc. 
        ''' 
                         
        t1=time()

        self.get_a = get_a
        self.do_mitigated = do_mitigated
        self.do_unmitigated = do_unmitigated

        self.surveys_lw = surveys_lw
        self.surveys_sw = surveys_sw
        self.N_surveys_sw=surveys_sw.size
        self.N_surveys_lw=surveys_lw.size
        #self.geo1_lw=surveys_lw[0]['geo1']     
        #self.geo2_lw=surveys_lw[0]['geo2']     
        #self.zs_lw=surveys_lw[0]['zs']
        print'Super_Survey: this is the number of surveys', self.N_surveys_sw, self.N_surveys_lw
           
        self.C=C
         
          #k_cut = 0.25 #converge to 0.0004
        self.basis = basis 
        self.N_O_I=0
        self.N_O_a=0   
        self.O_I=np.array([], dtype=object)
    
        #self.O_a=np.array([], dtype=object)
        for i in range(self.N_surveys_sw):
          
            survey=surveys_sw[i]
            self.N_O_I=self.N_O_I + survey.get_N_O_I()          
        #TODO temp     
        for i in range(self.N_surveys_lw):
          
            survey=self.surveys_lw[i]
            self.N_O_a=self.N_O_a + survey.get_N_O_a()
            #self.O_a=np.append(self.O_a,survey['O_a'])
              
        print('Super_Survey: there are '+str(self.N_O_I)+' short wavelength and '+str(self.N_O_a)+' long wavelength observables')
        #self.O_a_data=self.get_O_a()
        #self.O_I_data=self.get_O_I(k,P_lin)
        self.O_I_data = self.get_O_I_all()
        self.F_0 = self.basis.get_fisher()
        if self.get_a:
            if self.do_unmitigated:
                self.cov_no_mit,self.a_no_mit=self.get_SSC_covar(mitigation=False)     
                print "Super_Survey: unmitigated run gave a="+str(self.a_no_mit)
            if self.do_mitigated:
                #self.F_0.clear_cache()
                self.cov_mit,self.a_mit=self.get_SSC_covar(mitigation=True)     
                print "Super_Survey: mitigated run gave a="+str(self.a_mit)
        else:
            if self.do_unmitigated:
                self.cov_no_mit=self.get_SSC_covar(mitigation=False)     
            if self.do_mitigated:
                self.F_0.clear_cache()
                self.cov_mit=self.get_SSC_covar(mitigation=True)     
        t2=time()
        print 'Super_Survey: all done'
        print 'Super_Survey: run time', t2-t1
     
     
    def get_SSC_covar(self,mitigation=True):
         
        result=np.zeros_like(2,dtype=object)
        #TODO handle F_loc differently
        if mitigation:
            print "Super_Survey: getting SSC covar with mitigation"
            F_loc = copy.deepcopy(self.F_0)
            #F_loc = self.basis.get_fisher(allow_caching=True)
            for i in range(0,self.N_surveys_lw):
                self.surveys_lw[i].fisher_accumulate(F_loc)
            F_loc.switch_rep(fm.REP_CHOL)
        else:
            F_loc = self.F_0
            print "Super_Survey: getting SSC covar without mitigation"

        Cov_SSC = np.zeros(2,dtype=object) 
        if self.get_a:
            a_SSC = np.zeros(self.N_surveys_sw,dtype=object) 

        for i in range(0,self.N_surveys_sw):
            survey=self.surveys_sw[i]
            Cov_SSC[i] = survey.get_covars(F_loc,self.basis)
            
            if self.get_a:
                T=self.basis.D_delta_bar_D_delta_alpha(survey.geo,tomography=True)[0]
                a_SSC[i]=F_loc.contract_covar(T.T,T,identical_inputs=True)
                print "Super_Survey: a is "+str(a_SSC[i])+" for survey #"+str(i)

        if self.get_a:
            return Cov_SSC,a_SSC
        else:
            return Cov_SSC

    def get_O_a(self):
        D_O_a=np.zeros(self.N_O_a,dtype=object)
        print 'Super_Survey: I have this many long wavelength observables', self.N_O_a 
        for i in range(self.N_O_a):
            O_a=self.O_a[i]
            for key in O_a:
                  
                if key=='number density':
                    data=O_a['number density']
                    n_obs=np.array([data[0],data[1]])
                    mass=data[2]
                    print n_obs, mass
                    X=DNumberDensityObservable(self.surveys_lw[0],defaults.dn_params,'lw1',self.C,self.basis)
                    D_O_a[i] = X.Fisher_alpha_beta()
                    print "Super_Survey: min D_O_a element for lw observable #"+str(i)+": "+str(np.min(D_O_a[i][0]))
                    print "Super_Survey: D_O_a for lw observable #"+str(i)+": ",D_O_a[i][0]
        return D_O_a  

    def get_O_I_all(self):
        O_I_list = np.zeros(self.N_O_I,dtype=object)
        N_O_I = 0
        for i in range(0,self.N_surveys_sw):
            survey = self.surveys_sw[i]
            N_O_I_survey = survey.get_N_O_I()
            O_I_list[N_O_I:N_O_I+N_O_I_survey] = survey.get_O_I_list()
        return O_I_list

          
if __name__=="__main__":
    t1 = time()
    z_max=1.05; l_max=20 

    #d=np.loadtxt('Pk_Planck15.dat')
    d=np.loadtxt('camb_m_pow_l.dat')
    k=d[:,0]; P=d[:,1]
    C=CosmoPie(k=k,P_lin=P,cosmology=defaults.cosmology)
    r_max=C.D_comov(z_max)
    print 'this is r max and l_max', r_max , l_max
    
    Theta1=[0.,np.pi/4.]
    Phi1=[0.,np.pi/3.]
    Theta2=[0.,np.pi/4.]
    Phi2=[2.*np.pi/3.,np.pi]

    #zs=np.array([.4,0.8,1.2])
    zs=np.array([.01,0.5,1.01])
    z_fine = np.arange(defaults.lensing_params['z_min_integral'],np.max(zs),defaults.lensing_params['z_resolution'])
    #zbins=np.array([.2,.6,1.0])
    #l=np.logspace(np.log10(2),np.log10(3000),1000)
    l_sw = np.logspace(np.log(30),np.log(5000),base=np.exp(1.),num=40)
    #l_sw = np.arange(0,50)
    
    geo1=rect_geo(zs,Theta1,Phi1,C,z_fine)
    geo2=rect_geo(zs,Theta2,Phi2,C,z_fine)
    
    loc_lens_params = defaults.lensing_params.copy()
    loc_lens_params['z_min_dist'] = np.min(zs)
    loc_lens_params['z_max_dist'] = np.max(zs)
    
    lenless_defaults = defaults.sw_survey_params.copy()
    lenless_defaults['needs_lensing'] = False

    survey_1 = SWSurvey(geo1,'survey1',C=C,ls=l_sw,params=defaults.sw_survey_params,observable_list = defaults.sw_observable_list,len_params=loc_lens_params) 
    #survey_2 = SWSurvey(geo1,'survey2',C=C,ls=l_sw,params=defaults.sw_survey_params,observable_list = defaults.sw_observable_list,len_params=loc_lens_params) 
 
    #survey_1 = SWSurvey(geo1,'survey1',C=C,ls=l_sw,params=defaults.sw_survey_params,observable_list = np.array([]),len_params=loc_lens_params) 
    survey_2 = SWSurvey(geo1,'survey2',C=C,ls=l_sw,params=lenless_defaults,observable_list = np.array([]),len_params=loc_lens_params) 
     
     

    M_cut=10**(12.5)
     
     
    surveys_sw=np.array([survey_1])
    
     
    geos = np.array([geo1,geo2])
    #geos = np.array([geo1])
    l_lw=np.arange(0,20)
    n_zeros=49
    k_cut = 0.015
            
    basis=sph_basis_k(r_max,C,k_cut,l_ceil=100)

    survey_3 = LWSurvey(geos,'lw_survey1',basis,C=C,ls = l_lw,params=defaults.lw_survey_params,observable_list=defaults.lw_observable_list,dn_params=defaults.dn_params)
    surveys_lw=np.array([survey_3])
     
     
    print 'main: this is r_max: '+str(r_max)
     
    SS=super_survey(surveys_sw, surveys_lw,r_max,l_sw,n_zeros,k,basis,P_lin=P,C=C,get_a=True,do_unmitigated=True,do_mitigated=True)

    t2 = time()
    print "main: total run time "+str(t2-t1)+" s"
     
    #print "fractional mitigation: ", SS.a_no_mit/SS.a_mit     
    rel_weights = SS.basis.D_delta_bar_D_delta_alpha(SS.surveys_sw[0].geo,tomography=True)[0]*np.dot(SS.F_0.get_cov_cholesky(),SS.basis.D_delta_bar_D_delta_alpha(SS.surveys_sw[0].geo,tomography=True)[0])
    #np.savetxt('rel_weights_k_026.txt',rel_weights)

 #   ax.plot(rel_weights)
 #   plt.show()
    
#     cov_ss = SS.cov_no_mit[0].get_gaussian_covar()#np.diag(SS.O_I_data[0]['shear_shear']['covariance'][0,0])
#    c_ss = SS.O_I_data[0]
   #chol_cov = get_inv_cholesky(cov_ss)
    #mat_retrieved = (np.identity(chol_cov.shape[0])+np.dot(np.dot(chol_cov,SS.cov_no_mit[0,0]),chol_cov.T))
    #eig_ret = np.linalg.eigvals(mat_retrieved)
    SS_eig =  SS.cov_no_mit[0].get_SS_eig()
    SS_eig_mit =  SS.cov_mit[0].get_SS_eig()
    chol_gauss = np.linalg.cholesky(SS.cov_no_mit[0].get_gaussian_covar())
    print "main: unmitigated lambda1,2: "+str(SS_eig[0][-1])+","+str(SS_eig[0][-2])
    print "main: mitigated lambda1,2: "+str(SS_eig_mit[0][-1])+","+str(SS_eig_mit[0][-2])
    print "n lambda>1.00000001: "+str(np.sum(np.abs(SS_eig[0])>1.00000001))
#    print "SS eigenvals:",SS_eig[0]
#    print "(S/N)^2 gaussian: ",np.dot(np.dot(c_ss,np.linalg.inv(cov_ss)),c_ss)
    #print "(S/N)^2 gaussian+no mitigation: ",np.dot(np.dot(c_ss,np.linalg.inv(np.diagflat(cov_ss)+SS.cov_no_mit[0,0])),c_ss)
    #print "(S/N)^2 gaussian+mitigation: ",np.dot(np.dot(c_ss,np.linalg.inv(np.diagflat(cov_ss)+SS.cov_mit[0,0])),c_ss)
    #ax.loglog(l_sw,cov_ss)
    #ax.loglog(l,np.diag(SS.cov_mit[0,0])/c_ss**2*l/2)
   # ax.loglog(l,(np.diag(SS.cov_mit[0,0])))
    #ax.loglog(l_sw,(np.diag(SS.cov_no_mit[0,0])/cov_ss))
    #ax.legend(['cov','mit','no mit'])
    #plt.show()
    print 'r diffs',np.diff(geo1.rs)
    print 'theta width',(geo1.rs[1]+geo1.rs[0])/2.*(Theta1[1]-Theta1[0])
    print 'phi width',(geo1.rs[1]+geo1.rs[0])/2.*(Phi1[1]-Phi1[0])*np.sin((Theta1[1]+Theta1[0])/2)
    ax_ls = np.hstack((l_sw,l_sw))
    import matplotlib.pyplot as plt
    ax = plt.subplot(111)
    for itr in range(1,5):
        #ax.plot(ax_ls,ax_ls*(ax_ls+1.)*np.dot(chol_gauss,SS_eig[1][:,-itr]))
        ax.plot(np.dot(chol_gauss,SS_eig[1][:,-itr]))
    
    ax.legend(['1','2','3','4','5'])
  #  plt.show()
