import numpy as np
import cosmopie as cp
import defaults
import re
from warnings import warn
import lensing_observables as lo
from sw_cov_mat import CovMat,SWCovMat
class SWSurvey:
    def __init__(self,geo,survey_id,C,ls = np.array([]),params=defaults.sw_survey_params,observable_list=defaults.sw_observable_list,len_params=defaults.lensing_params):
        print "sw_survey: began initializing survey: "+str(survey_id)
        self.geo = geo
        self.params = params
        self.needs_lensing = params['needs_lensing']
        self.C = C # cosmopie 
        self.ls = ls
        self.survey_id = survey_id
        if self.needs_lensing:
            self.len_pow = lo.LensingPowerBase(self.geo,self.ls,survey_id,C=C,params=len_params)
            self.len_params = len_params
        else:
            self.len_pow = None
            self.len_params = None

        self.observable_names = generate_observable_names(self.geo,observable_list,params['cross_bins'])
        self.observables = self.names_to_observables(self.observable_names)
        print "sw_survey: finsihed initializing survey: "+str(survey_id)

    def get_survey_id(self):
        return self.survey_id

    def get_N_O_I(self):
        return self.observables.size

    def get_total_dimension(self):
        dim = 0
        for i in range(self.observables.size):
            dim+=self.observables[i].get_dimension()
        return dim
    
    def get_dimension_list(self):
        dim_list = np.zeros(self.get_N_O_I(),dtype=np.int_)
        for i in range(dim_list.size):
            dim_list[i] = self.observables[i].get_dimension()
        return dim_list 

    def get_O_I_list(self):
        O_I_list = np.zeros(self.observables.size,dtype=object)
        for i in range(0,self.observables.size):
            O_I_list[i] = self.observables[i].get_O_I()
        return O_I_list

    def get_dO_I_ddelta_bar_list(self):
        dO_I_ddelta_bar_list = np.zeros(self.observables.size,dtype=object)
        for i in range(self.observables.size):
            dO_I_ddelta_bar_list[i] = self.observables[i].get_dO_I_ddelta_bar()
        return dO_I_ddelta_bar_list

    def get_gaussian_cov(self):
        cov_mats = np.zeros((self.get_total_dimension(),self.get_total_dimension()))
        ds = self.get_dimension_list()
        #n1 and n2 are to track indices so cov_mats can be a float array instead of an array of objects
        n1 = 0
        for i in range(0,self.get_N_O_I()):
            n2 = 0
            for j in range(0,self.get_N_O_I()):
                cov = SWCovMat(self.observables[i],self.observables[j])
                cov_mats[n1:n1+ds[i],n2:n2+ds[j]] = cov.get_gaussian_covar()
                cov_mats[n2:n2+ds[j],n1:n1+ds[i]] = cov.get_gaussian_covar()
                n2+=ds[j]
            n1+=ds[i]

        return cov_mats

    def get_SSC_cov(self,fisher,basis):
        print "sw_survey: begin computing sw covariance matrices"
        cov_mats = np.zeros((self.get_total_dimension(),self.get_total_dimension()))

        #short circuit get_dO_I_ddelta_bar_list() if the result won't be used
        if self.get_total_dimension()==0:
            print "sw_survey: no sw covariance matrices to compute"
            return cov_mats
        ds = self.get_dimension_list()
        #n1 and n2 are to track indices so cov_mats can be a float array instead of an array of objects
        n1 = 0
        #TODO consider not getting this whole list initially
        dO_I_ddelta_bar_list = self.get_dO_I_ddelta_bar_list()
        #TODO consider optimizing by preevaluating contractions with T (may take more memory)
        for i in range(0,self.get_N_O_I()):
            n2 = 0
            T_i = basis.D_O_I_D_delta_alpha(self.geo,dO_I_ddelta_bar_list[i])
            print "sw_survey: T_i shape: "+str(T_i.shape)
            for j in range(0,self.get_N_O_I()):
                print "sw_survey "+str(self.get_survey_id())+": Calc d delta alpha for observable 1,2 #:"+str(i)+","+str(j)
                T_j = basis.D_O_I_D_delta_alpha(self.geo,dO_I_ddelta_bar_list[j])
                cov_mats[n1:n1+ds[i],n2:n2+ds[j]] = fisher.contract_covar(T_i.T,T_j)
                n2+=ds[j]
            n1+=ds[i]
        print "sw_survey: finished computing sw covariance matrices"
        return cov_mats
    
    def get_nongaussian_cov(self):
        return 0.

    def get_covars(self,fisher,basis):
        return CovMat(self.get_gaussian_cov(),self.get_nongaussian_cov(),self.get_SSC_cov(fisher,basis),self.get_total_dimension())

    def names_to_observables(self,names):
        observables = np.zeros(len(names.keys()),dtype=object)
        itr = 0 
        for key in sorted(names.keys()):
            if self.params['needs_lensing'] and re.match('^len',key):
                r1 = names[key]['r1']
                r2 = names[key]['r2']
                if re.match('^len_shear_shear',key):
                    observables[itr] = lo.ShearShearLensingObservable(self.len_pow,r1,r2,params=self.len_params)
                elif re.match('^len_galaxy_galaxy',key):
                    observables[itr] = lo.GalaxyGalaxyLensingObservable(self.len_pow,r1,r2,params=self.len_params)
                else:
                    warn('unrecognized or unprocessable observable: \'',key,'\', skipping')
                    observables[itr] = None
            else:
                warn('unrecognized or unprocessable observable: \'',key,'\', skipping')
                observables[itr] = None
            itr+=1
        return observables
            
def generate_observable_names(geo,observable_list,cross_bins=defaults.sw_survey_params['cross_bins']):
    rbins = geo.rbins
    names = {}
    for name in observable_list:
        if re.match('^len',name):
            for i in range(0,rbins.shape[0]):
                r1 = rbins[i]
                if cross_bins:
                    for j in range(0,rbins.shape[0]):
                        #Only take r1<=r2
                        if i>j: 
                            pass 
                        else:
                            r2 = rbins[j]
                            name_str = name+'_'+str(i)+'_'+str(j)
                            names[name_str] = {'r1':r1,'r2':r2}

                else:
                    name_str = name+'_'+str(i)+'_'+str(i)
                    names[name_str] = {'r1':r1,'r2':r1}
        else:
            warn('observable name \'',name,'\' unrecognized, ignoring')
    return names

if __name__=='__main__':
    from geo import rect_geo
    Theta = [np.pi/4.,np.pi/2.]
    Phi = [0,np.pi/3.]
    d=np.loadtxt('camb_m_pow_l.dat')
    k=d[:,0]; P=d[:,1]
    C=cp.CosmoPie(k=k,P_lin=P)
    zs = np.array([0.1,0.8])
    z_fine = np.arange(0.01,np.max(zs),0.01)
    ls = np.arange(2,500)
    geo = rect_geo(zs,Theta,Phi,C,z_fine)
    sw_survey = SWSurvey(geo,'survey1',C,ls)
    O_I_list = sw_survey.get_O_I_list()
    dO_I_ddelta_bar_list = sw_survey.get_dO_I_ddelta_bar_list()
    import matplotlib.pyplot as plt
    ax  = plt.subplot(111)
    ax.loglog(ls,O_I_list[0])
    ax.loglog(ls,dO_I_ddelta_bar_list[0])
    plt.xlabel('ls')
    plt.legend(['O_I','dO_I_ddelta_bar'])
    plt.show()


#        cov_mats = np.zeros((self.get_N_O_I(),self.get_N_O_I()),dtype=object)
#        for i in range(0,cov_mats.shape[0]):
#            for j in range(0,cov_mats.shape[1]):
#                cov = SWCovMat(self.observables[i],self.observables[j])
#                cov_mats[i,j] = cov.get_total_covar()
#        return cov_mats
            
if __name__=='__main__':
    from geo import rect_geo
    Theta = [np.pi/4.,np.pi/2.]
    Phi = [0,np.pi/3.]
    d=np.loadtxt('camb_m_pow_l.dat')
    k=d[:,0]; P=d[:,1]
    C=cp.CosmoPie(k=k,P_lin=P)
    zs = np.array([0.1,0.8])
    z_fine = np.arange(0.01,np.max(zs),0.01)
    ls = np.arange(2,500)
    geo = rect_geo(zs,Theta,Phi,C,z_fine)
    sw_survey = SWSurvey(geo,'survey1',C,ls)
    O_I_list = sw_survey.get_O_I_list()
    dO_I_ddelta_bar_list = sw_survey.get_dO_I_ddelta_bar_list()
    import matplotlib.pyplot as plt
    ax  = plt.subplot(111)
    ax.loglog(ls,O_I_list[0])
    ax.loglog(ls,dO_I_ddelta_bar_list[0])
    plt.xlabel('ls')
    plt.legend(['O_I','dO_I_ddelta_bar'])
    plt.show()


