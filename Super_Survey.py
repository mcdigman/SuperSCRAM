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
from algebra_utils import cholesky_inv, inverse_cholesky 
import defaults
import fisher_matrix as fm
import lensing_observables as lo
from sw_survey import SWSurvey
from lw_survey import LWSurvey
class super_survey:
    ''' This class holds and returns information for all surveys
    ''' 
    def __init__(self, surveys_sw, surveys_lw, r_max, l, n_zeros,k,basis,C,P_lin=None):
    
      
        '''
        1) surveys is an array containing the informaiton for all the surveys.
        2) r_max is the maximum radial super mode 
        3) l angular super modes
        4) cosmology is the comological parameters etc. 
        ''' 
                         
        t1=time()

        self.surveys_lw = surveys_lw
        self.surveys_sw = surveys_sw
        self.N_surveys_sw=surveys_sw.size
        self.N_surveys_lw=surveys_lw.size
        #self.geo1_lw=surveys_lw[0]['geo1']     
        #self.geo2_lw=surveys_lw[0]['geo2']     
        #self.zs_lw=surveys_lw[0]['zs']
        print'this is the number of surveys', self.N_surveys_sw, self.N_surveys_lw
           
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
              
        print('these are number of observables', self.N_O_I, self.N_O_a)
        #self.O_a_data=self.get_O_a()
        #self.O_I_data=self.get_O_I(k,P_lin)
        self.O_I_data = self.get_O_I_all()
        #self.F_0=self.basis.get_F_alpha_beta()
        self.F_0 = self.basis.get_fisher()
        self.cov_no_mit,self.a_no_mit=self.get_SSC_covar(mitigation=False)     
        self.cov_mit,self.a_mit=self.get_SSC_covar(mitigation=True)     
          
        t2=time()
        print 'all done'
        print 'run time', t2-t1
     
     
    def get_SSC_covar(self,mitigation=True):
         
        result=np.zeros_like(2,dtype=object)
        #TODO handle F_loc differently
        if mitigation:
            print "mit"
            #x = self.O_a_data[0]
            F_loc = self.basis.get_fisher()
            for i in range(0,self.N_surveys_lw):
                self.surveys_lw[i].fisher_accumulate(F_loc)
            #F_loc.add_fisher(x[0])
            #F_loc.add_fisher(x[1])
            #print "x[0]: ",x[0]
        else:
            F_loc = self.F_0
            print "no mit"

        Cov_SSC = np.zeros((2,2),dtype=object) 
        a_SSC = np.zeros(self.N_surveys_sw) 

        for i in range(0,self.N_surveys_sw):
            survey=self.surveys_sw[i]
            T=self.basis.D_delta_bar_D_delta_alpha(survey.geo)[0]
            print "max t",max(T)
            print "min t",min(T)
            print T[0]
            a_SSC[i] = F_loc.contract_covar(T,T)
            print "a is: ",a_SSC[i]
            dO_I_ddelta_bar_list = survey.get_dO_I_ddelta_bar_list()
            for j in range(0,dO_I_ddelta_bar_list.size):
                Cov_SSC[i,j]=np.outer(dO_I_ddelta_bar_list[j],dO_I_ddelta_bar_list[j])*a_SSC[i]
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
                    X=DNumberDensityObservable(self.surveys_lw[0],defaults.dn_params,'lw1',self.C,self.basis)
                    D_O_a[i] = X.Fisher_alpha_beta()
                    print "min",np.min(D_O_a[i][0])
                    print D_O_a[i][0]
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

    z_max=1.05; l_max=20 

    #d=np.loadtxt('Pk_Planck15.dat')
    d=np.loadtxt('camb_m_pow_l.dat')
    k=d[:,0]; P=d[:,1]
    C=CosmoPie(k=k,P_lin=P,cosmology=defaults.cosmology)
    r_max=C.D_comov(z_max)
    print 'this is r max and l_max', r_max , l_max
    
    Theta1=[np.pi/4.,5.*np.pi/16.]
    Phi1=[0.,np.pi/12.]
    Theta2=[np.pi/4.,np.pi/2.]
    Phi2=[np.pi/3.,2.*np.pi/3.]
    #geo=np.array([Theta,Phi])

    zs=np.array([.9,1.0])
    #zbins=np.array([.2,.6,1.0])
    #l=np.logspace(np.log10(2),np.log10(3000),1000)
    l_sw = np.arange(2,3000)
    
    geo1=rect_geo(zs,Theta1,Phi1,C)
    geo2=rect_geo(zs,Theta2,Phi2,C)

    survey_1 = SWSurvey(geo1,'survey1',C=C,ls=l_sw,params=defaults.sw_survey_params,observable_list = defaults.sw_observable_list,len_params=defaults.lensing_params) 
    survey_2 = SWSurvey(geo1,'survey2',C=C,ls=l_sw,params=defaults.sw_survey_params,observable_list = defaults.sw_observable_list,len_params=defaults.lensing_params) 

    #shear_data1={'z_bins':zbins,'l':l_sw,'geo':geo1}
    #shear_data2={'z_bins':zbins,'l':l_sw,'geo':geo2}
     
    #O_I1={'shear_shear':shear_data1}
    #O_I2={'shear_shear':shear_data2}
     

    #n_dat2=np.array([1.01*5.*1e5/1.7458,1.01*5.0*1e5*1.4166*0.98546])
    M_cut=10**(12.5)
     
    n_avg = np.zeros(2)
    n_avg[0] = ST_hmf(C).n_avg(M_cut,0.15)
    n_avg[1] = ST_hmf(C).n_avg(M_cut,0.25)
        
    V1 = geo1.volumes[0]
    #V2 = geo1.volumes[1]
    V2 = 0
    #V1 = Dn.volume(C.D_comov(0.1),C.D_comov(0.2),Theta,Phi)
    #V2 = Dn.volume(C.D_comov(0.2),C.D_comov(0.3),Theta,Phi)
    #amplitude of fluctuations within a given radius should be given by something like np.sqrt(np.trapz((3.*(sin(k*R)-k*R*cos(k*R))/(k*R)**3)**2*P*k**2,k)*4.*np.pi/(2.*np.pi)**3), where R is the radius of the bin, R=C.D_comov(z[i])-C.D_comov(z[i-1])
    #cf https://ned.ipac.caltech.edu/level5/March01/Strauss/Strauss2.html
    #~0.007 for R~400 (check units).
    n_dat1=np.array([n_avg[0]*V1,n_avg[1]*V2])
    n_dat2=np.array([1.01*n_avg[0]*V1,1.01*n_avg[1]*V2])

    O_a={'number density':np.array([n_dat1,n_dat2,M_cut])}
     
    #d_1={'name': 'survey 1', 'area': 18000}
    #d_2={'name': 'survey 2', 'area': 18000}
    d_3={'name': 'suvery lw', 'area' :18000}
     
     
    #survey_1={'details':d_1,'O_I':O_I1, 'geo':geo1}
          
          
    #survey_2={'details':d_2,'O_I':O_I2, 'geo':geo2}
     
    surveys_sw=np.array([survey_1, survey_2])
    
     
    #survey_3={'details':d_3, 'O_a':O_a, 'zs':zs,'geo1':geo1,'geo2':geo2}
    geos = np.array([geo1,geo2])
    l_lw=np.arange(0,20)
    n_zeros=49
    k_cut = 0.005
            
    #self.basis=sph_basis(r_max,l,n_zeros,self.CosmoPie)
    basis=sph_basis_k(r_max,C,k_cut,l_ceil=100)
    survey_3 = LWSurvey(geos,'lw_survey1',basis,C=C,ls = l_lw,params=defaults.lw_survey_params,observable_list=defaults.lw_observable_list,dn_params=defaults.dn_params)
    surveys_lw=np.array([survey_3])
     
     
    print 'this is r_max', r_max 
     
    SS=super_survey(surveys_sw, surveys_lw,r_max,l_sw,n_zeros,k,basis,P_lin=P,C=C)
     
    #print "fractional mitigation: ", SS.a_no_mit/SS.a_mit     

    import matplotlib.pyplot as plt
    ax = plt.subplot(111)
    cov_ss = surveys_sw[0].get_covars()[0,0]#np.diag(SS.O_I_data[0]['shear_shear']['covariance'][0,0])
    c_ss = SS.O_I_data[0]

    #try:
    #    np.linalg.cholesky(np.diagflat(cov_ss))
    #except Exception:
    #    print "gaussian covariance is not positive definite"
    #try:
    #    np.linalg.cholesky(np.linalg.inv(SS.cov_no_mit[0,0]))
    #except Exception:
    #    print "unmitigated covariance is not positive definite"
    #try:
    #    np.linalg.cholesky(np.linalg.inv(SS.cov_mit[0,0]))
    #except Exception:
    #    print "mitigated covariance is not positive definite"

    print "(S/N)^2 gaussian: ",np.dot(np.dot(c_ss,np.linalg.inv(np.diagflat(cov_ss))),c_ss)
    print "(S/N)^2 gaussian+no mitigation: ",np.dot(np.dot(c_ss,np.linalg.inv(np.diagflat(cov_ss)+SS.cov_no_mit[0,0])),c_ss)
    print "(S/N)^2 gaussian+mitigation: ",np.dot(np.dot(c_ss,np.linalg.inv(np.diagflat(cov_ss)+SS.cov_mit[0,0])),c_ss)
    #ax.loglog(l_sw,cov_ss)
    #ax.loglog(l,np.diag(SS.cov_mit[0,0])/c_ss**2*l/2)
   # ax.loglog(l,(np.diag(SS.cov_mit[0,0])))
    ax.loglog(l_sw,(np.diag(SS.cov_no_mit[0,0])/cov_ss))
    #ax.legend(['cov','mit','no mit'])
    plt.show()
    print 'r diffs',np.diff(geo1.rs)
    print 'theta width',(geo1.rs[1]+geo1.rs[0])/2.*(Theta1[1]-Theta1[0])
    print 'phi width',(geo1.rs[1]+geo1.rs[0])/2.*(Phi1[1]-Phi1[0])

