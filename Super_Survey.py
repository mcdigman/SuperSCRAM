import numpy as np
#from talk_to_class import class_objects 
from FASTPTcode import FASTPT 

from cosmopie import CosmoPie 
#import sph_basis as basis
from sph_klim import sph_basis_k
from time import time 
from Dn import DO_n
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
class super_survey:
    ''' This class holds and returns information for all surveys
    ''' 
    def __init__(self, surveys_sw, surveys_lw, r_max, l, n_zeros,k,cosmology=defaults.cosmology,P_lin=None):
    
      
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
        self.geo1_lw=surveys_lw[0]['geo1']     
        self.geo2_lw=surveys_lw[0]['geo2']     
        self.zbins_lw=surveys_lw[0]['zbins']
        print'this is the number of surveys', self.N_surveys_sw, self.N_surveys_lw
           
        self.CosmoPie=CosmoPie(k=k,P_lin=P_lin,cosmology=defaults.cosmology)
         
          #k_cut = 0.25 #converge to 0.0004
        k_cut = 0.01
            
          #self.basis=sph_basis(r_max,l,n_zeros,self.CosmoPie)
        self.basis=sph_basis_k(r_max,self.CosmoPie,k_cut,l_ceil=100)
                    
        self.N_O_I=0
        self.N_O_a=0   
        self.O_I=np.array([], dtype=object)
        self.O_a=np.array([], dtype=object)
        for i in range(self.N_surveys_sw):
          
            survey=surveys_sw[i]
            self.N_O_I=self.N_O_I + survey.get_N_O_I()          
        #TODO temp     
        #for i in range(self.N_surveys_lw):
          
        #    survey=surveys_lw[i]
        #    self.N_O_a=self.N_O_a + len(survey['O_a']) 
        #    self.O_a=np.append(self.O_a,survey['O_a'])
              
        print('these are number of observables', self.N_O_I, self.N_O_a)
        #self.O_a_data=self.get_O_a()
        #self.O_I_data=self.get_O_I(k,P_lin)
        self.O_I_data = self.get_O_I_all()
        #self.F_0=self.basis.get_F_alpha_beta()
        self.F_0 = self.basis.get_fisher()
        #self.cov_mit,self.a_mit=self.get_SSC_covar(mitigation=True)     
        self.cov_no_mit,self.a_no_mit=self.get_SSC_covar(mitigation=False)     
          
        t2=time()
        print 'all done'
        print 'run time', t2-t1
     
     
    def get_SSC_covar(self,mitigation=True):
         
        result=np.zeros_like(2,dtype=object)
        #TODO handle F_loc differently
        if mitigation:
            print "mit"
            x = self.O_a_data[0]
            F_loc = self.basis.get_fisher()
            F_loc.add_fisher(x[0])
            F_loc.add_fisher(x[1])
            print "x[0]: ",x[0]
            #print "F_1: ",F_1
            #try:
            #    np.linalg.cholesky(F_1)
            #except Exception:
            #    warn("F_1 fails cholesky")
            #chol_1 = inverse_cholesky(F_1)
            #C_1=cholesky_inv(F_1)
            #print "invert 1"
            #C_2=cholesky_inv(F_2)
            #chol_2=inverse_cholesky(F_2)
            #print "invert 2"
        else:
            F_loc = self.F_0
            print "no mit"
            #chol_1 = inverse_cholesky(self.F_0)
            #chol_2 = chol_1
                
            #C_1 = self.basis.C_alpha_beta
            #C_2 = self.basis.C_alpha_beta
                    
        #chols=np.array([chol_1,chol_2],dtype=object)

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

                #print "max C",np.max(C[j])
                #print "min C",np.min(C[j])

                Cov_SSC[i,j]=np.outer(dO_I_ddelta_bar_list[j],dO_I_ddelta_bar_list[j])*a_SSC[i]
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
                    X=DO_n(self.surveys_lw[0],mass,self.CosmoPie,self.basis)
                    D_O_a[i] = X.Fisher_alpha_beta()
                    print "min",np.min(D_O_a[i][0])
#`                  print "eig",np.linalg.eigvals(D_O_a[i][0])
                    print D_O_a[i][0]
          # x=D_O_a[0]
#           print x[0]
#           print x[1]
#           print 'hi joe'
#           print x.shape
#           sys.exit()
        return D_O_a  

    def get_O_I_all(self):
        O_I_list = np.zeros(self.N_O_I,dtype=object)
        N_O_I = 0
        for i in range(0,self.N_surveys_sw):
            survey = self.surveys_sw[i]
            N_O_I_survey = survey.get_N_O_I()
            O_I_list[N_O_I:N_O_I+N_O_I_survey] = survey.get_O_I_list()
        return O_I_list

#    def get_O_I(self,k,P_lin):
#         
#        D_O_I=np.array([],dtype=object)
#        result = np.array([],dtype=object)
#        for i in range(self.N_O_I):
#            O_I=self.O_I[i] 
#            result = np.append(result,{}) 
#            for key in O_I:
#                result[i][key]={}
#                if key=='shear_shear':
#                    print key
#                    data=O_I[key]
#                    z_bins=data['z_bins']
#                    ddelta_dalpha=np.zeros(z_bins.size-1,dtype=object)
#                    geo=data['geo']
#                    #Theta=geo[0]; Phi=geo[1]
#                      
#                   
#                    ls=data['l']
#                    len_pow = lo.LensingPowerBase(geo,ls,params=defaults.lensing_params,C=self.CosmoPie)
#                    #sp1 = sh_pow.shear_power(k,self.CosmoPie,zs,ls,P_in=P_lin,pmodel='dc_halofit')
#                    #sp2 = sh_pow.shear_power(k,self.CosmoPie,zs,ls,P_in=P_lin,pmodel='halofit_nonlinear')
#
#                    dcs = np.zeros((z_bins.size-1,ls.size))
#                    cs = np.zeros((z_bins.size-1,ls.size))
#                    #covs = np.array([],dtype=object)
#
#                    ddelta_dalpha=self.basis.D_delta_bar_D_delta_alpha(geo)
#                    #sh_pows = np.array([],dtype=object)
#                    for j in range(0,z_bins.size-1):
#                        chi_min = self.CosmoPie.D_comov(z_bins[j])
#                        chi_max = self.CosmoPie.D_comov(z_bins[j+1])
#                        r1 = np.array([chi_min,chi_max])
#                        r2 = np.array([chi_min,chi_max])
#                        #sh_pow2 = sh_pow.Cll_sh_sh(sp2,chi_max,chi_max,chi_min,chi_min).Cll()
#                        obs = lo.ShearShearLensingObservable(len_pow,r1,r2,params=defaults.lensing_params)
#                        cs[j] = obs.get_O_I()
#                        dcs[j] = obs.get_dO_I_ddelta_bar()
#                        #cs[j] = sh_pow.Cll_sh_sh(sp2,chi_max,chi_max,chi_min,chi_min).Cll()
#                        #dcs[j] = sh_pow.Cll_sh_sh(sp1,chi_max,chi_max,chi_min,chi_min).Cll()
#                        #print ddelta_dalpha[j]
#                        #print self.basis.C_id
#                             
#                      
#                 #       covs = np.append(covs,np.diagflat(sp2.cov_g_diag(sh_pow2,sh_pow2,sh_pow2,sh_pow2))) #TODO support cross z_bin covariance correctly
#                    covs = len_pow.C_pow.cov_mats(z_bins,cname1='shear',cname2='shear')
#                    result[i][key]['power'] = cs
#                    result[i][key]['dc_ddelta'] = dcs
#                    result[i][key]['covariance'] = covs
#                    result[i][key]['ddelta_dalpha']=ddelta_dalpha
#                        #print n_obs, mass
#                        #result=DO_n(n_obs,self.zbins_lw,mass,self.CosmoPie,self.basis,self.geo_lw)
#              
#          
#        return result
              
          
          
          
if __name__=="__main__":

    z_max=1.05; l_max=20 

    #d=np.loadtxt('Pk_Planck15.dat')
    d=np.loadtxt('camb_m_pow_l.dat')
    k=d[:,0]; P=d[:,1]
    cp=CosmoPie(k=k,P_lin=P)
    r_max=cp.D_comov(z_max)
    print 'this is r max and l_max', r_max , l_max
    
    Theta1=[np.pi/4.,5.*np.pi/16.]
    Phi1=[0.,np.pi/12.]
    Theta2=[np.pi/4.,np.pi/2.]
    Phi2=[np.pi/3.,2.*np.pi/3.]
    #geo=np.array([Theta,Phi])

    zbins=np.array([.9,1.0])
    #zbins=np.array([.2,.6,1.0])
    #l=np.logspace(np.log10(2),np.log10(3000),1000)
    l_sw = np.arange(2,3000)
    
    geo1=rect_geo(zbins,Theta1,Phi1,cp)
    geo2=rect_geo(zbins,Theta2,Phi2,cp)

    survey_1 = SWSurvey(geo1,'survey1',l_sw,cp,params=defaults.sw_survey_params,observable_list = defaults.sw_observable_list,len_params=defaults.lensing_params) 
    survey_2 = SWSurvey(geo1,'survey2',l_sw,cp,params=defaults.sw_survey_params,observable_list = defaults.sw_observable_list,len_params=defaults.lensing_params) 

    #shear_data1={'z_bins':zbins,'l':l_sw,'geo':geo1}
    #shear_data2={'z_bins':zbins,'l':l_sw,'geo':geo2}
     
    #O_I1={'shear_shear':shear_data1}
    #O_I2={'shear_shear':shear_data2}
     

    #n_dat2=np.array([1.01*5.*1e5/1.7458,1.01*5.0*1e5*1.4166*0.98546])
    M_cut=10**(12.5)
     
    n_avg = np.zeros(2)
    n_avg[0] = ST_hmf(cp).n_avg(M_cut,0.15)
    n_avg[1] = ST_hmf(cp).n_avg(M_cut,0.25)
        
    V1 = geo1.volumes[0]
    #V2 = geo1.volumes[1]
    V2 = 0
    #V1 = Dn.volume(cp.D_comov(0.1),cp.D_comov(0.2),Theta,Phi)
    #V2 = Dn.volume(cp.D_comov(0.2),cp.D_comov(0.3),Theta,Phi)
    #amplitude of fluctuations within a given radius should be given by something like np.sqrt(np.trapz((3.*(sin(k*R)-k*R*cos(k*R))/(k*R)**3)**2*P*k**2,k)*4.*np.pi/(2.*np.pi)**3), where R is the radius of the bin, R=cp.D_comov(z[i])-cp.D_comov(z[i-1])
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
    
     
    survey_3={'details':d_3, 'O_a':O_a, 'zbins':zbins,'geo1':geo1,'geo2':geo2}
    surveys_lw=np.array([survey_3])
     
    l_lw=np.arange(0,20)
    n_zeros=49
     
    print 'this is r_max', r_max 
     
    SS=super_survey(surveys_sw, surveys_lw,r_max,l_sw,n_zeros,k,P_lin=P)
     
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
    #print "(S/N)^2 gaussian+mitigation: ",np.dot(np.dot(c_ss,np.linalg.inv(np.diagflat(cov_ss)+SS.cov_mit[0,0])),c_ss)
    ax.loglog(l_sw,cov_ss)
    #ax.loglog(l,np.diag(SS.cov_mit[0,0])/c_ss**2*l/2)
   # ax.loglog(l,(np.diag(SS.cov_mit[0,0])))
    ax.loglog(l_sw,(np.diag(SS.cov_no_mit[0,0])))
    #ax.legend(['cov','mit','no mit'])
    plt.show()
    print 'r diffs',np.diff(geo1.rs)
    print 'theta width',(geo1.rs[1]+geo1.rs[0])/2.*(Theta1[1]-Theta1[0])
    print 'phi width',(geo1.rs[1]+geo1.rs[0])/2.*(Phi1[1]-Phi1[0])

