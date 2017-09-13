import numpy as np
import cosmopie as cp
import defaults
import re
from warnings import warn
import lensing_observables as lo
from sw_cov_mat import SWCovMat
import fisher_matrix as fm

class SWSurvey:
    def __init__(self,geo,survey_id,C,ls = np.array([]),cosmo_par_list=np.array([],dtype=object),cosmo_par_epsilons=np.array([]),params=defaults.sw_survey_params,observable_list=defaults.sw_observable_list,len_params=defaults.lensing_params,ps=np.array([])):
        print "sw_survey: began initializing survey: "+str(survey_id)
        self.geo = geo
        self.params = params
        self.needs_lensing = params['needs_lensing']
        self.C = C # cosmopie 
        self.ls = ls
        self.survey_id = survey_id
        self.cosmo_par_list = cosmo_par_list
        if self.needs_lensing:
            self.len_pow = lo.LensingPowerBase(self.geo,self.ls,survey_id,C=C,params=len_params,cosmo_par_list=cosmo_par_list,cosmo_par_epsilons=cosmo_par_epsilons,ps=ps)
            self.len_params = len_params
        else:
            self.len_pow = None
            self.len_params = None

        self.n_param = cosmo_par_list.size
        
        self.observable_names = generate_observable_names(self.geo,observable_list,params['cross_bins'])
        self.observables = self.names_to_observables(self.observable_names)
        print "sw_survey: finished initializing survey: "+str(survey_id)

    def get_survey_id(self):
        return self.survey_id

    def get_N_O_I(self):
        return self.observables.size

    def get_total_dimension(self):
        dim = 0
        for i in xrange(self.observables.size):
            dim+=self.observables[i].get_dimension()
        return dim
    
    def get_dimension_list(self):
        dim_list = np.zeros(self.get_N_O_I(),dtype=np.int_)
        for i in xrange(dim_list.size):
            dim_list[i] = self.observables[i].get_dimension()
        return dim_list 

#    def get_O_I_list(self):
#        O_I_list = np.zeros(self.observables.size,dtype=object)
#        for i in xrange(0,self.observables.size):
#            O_I_list[i] = self.observables[i].get_O_I()
#        return O_I_list
    def get_O_I_array(self):
        O_I_array = np.zeros(self.get_N_O_I(),self.get_total_dimension())
        itr = 0
        ds = self.get_dimension_list()
        for i in xrange(self.observables.size):
            O_I_array[:,itr:itr+ds[i]] = self.observables[i].get_O_I()
            itr+=ds[i]
        return O_I_array
#
#    def get_dO_I_ddelta_bar_list(self):
#        dO_I_ddelta_bar_list = np.zeros(self.observables.size,dtype=object)
#        for i in xrange(self.observables.size):
#            dO_I_ddelta_bar_list[i] = self.observables[i].get_dO_I_ddelta_bar()
#        return dO_I_ddelta_bar_list

    def get_dO_I_ddelta_bar_array(self):
        dO_I_ddelta_bar_array = np.zeros((self.geo.z_fine.size,self.get_total_dimension()))
        itr = 0
        ds = self.get_dimension_list()
        for i in xrange(self.observables.size):
            dO_I_ddelta_bar_array[:,itr:itr+ds[i]] = self.observables[i].get_dO_I_ddelta_bar()
            itr+=ds[i]
        return dO_I_ddelta_bar_array


#    def get_dO_I_dpar_list(self):
#        dO_I_dpar_list = np.zeros((self.n_param,self.observables.size),dtype=object)
#        for i in xrange(self.observables.size):
#            dO_I_dpar_list[:,i] = self.observables[i].get_dO_I_dpars()
#        return dO_I_dpar_list

    def get_dO_I_dpar_array(self):
        #dO_I_dpar_list = self.get_dO_I_dpar_list()
        #print "sw_survey: observable response list shape: "+str(dO_I_dpar_list.shape)
        #TODO make n_param an actual variable
        dO_I_dpar_array = np.zeros((self.get_total_dimension(),self.n_param))
        #print "sw_survey: observable response array shape: "+str(dO_I_dpar_array.shape)
        ds = self.get_dimension_list()
        #for i in xrange(0,self.n_param):
        itr = 0
        for i in xrange(0,self.get_N_O_I()):
            #n_k = dO_I_dpar_list[i,j].size
            dO_I_dpar_array[itr:itr+ds[i],:] = self.observables[i].get_dO_I_dpars()
            itr+=ds[i]
        return dO_I_dpar_array
    #get an array of gaussian, nongaussian sw covariance matrices for the observables
    def get_non_SSC_sw_covar_arrays(self):
        cov_mats = np.zeros((2,self.get_total_dimension(),self.get_total_dimension()))
        ds = self.get_dimension_list()
        #n1 and n2 are to track indices so cov_mats can be a float array instead of an array of objects
        n1 = 0
        for i in xrange(0,self.get_N_O_I()):
            n2 = 0
            for j in xrange(0,self.get_N_O_I()):
                cov = SWCovMat(self.observables[i],self.observables[j])
                cov_mats[0,n1:n1+ds[i],n2:n2+ds[j]] = cov.get_gaussian_covar_array()
                cov_mats[0,n2:n2+ds[j],n1:n1+ds[i]] = cov_mats[0,n1:n1+ds[i],n2:n2+ds[j]]
                cov_mats[1,n1:n1+ds[i],n2:n2+ds[j]] = cov.get_nongaussian_covar_array()
                cov_mats[1,n2:n2+ds[j],n1:n1+ds[i]] = cov_mats[1,n1:n1+ds[i],n2:n2+ds[j]]
                n2+=ds[j]
            n1+=ds[i]

        return cov_mats
#    def get_SSC_cov(self,fisher_set,basis):
#        print "sw_survey: begin computing sw covariance matrices"
#        n_m = self.get_total_dimension()
#        n_c = fisher_set.shape[0]
#        cov_mats = np.zeros((n_c,n_m,n_m))
#
#        #short circuit get_dO_I_ddelta_bar_list() if the result won't be used
#        if n_m==0:
#            print "sw_survey: no sw covariance matrices to compute"
#            return cov_mats
#        ds = self.get_dimension_list()
#        #n1 and n2 are to track indices so cov_mats can be a float array instead of an array of objects
#        n1 = 0
#        #TODO consider not getting this whole list initially
#        dO_I_ddelta_bar_list = self.get_dO_I_ddelta_bar_list()
#
#        n_o = self.get_N_O_I()
#        Ts = np.zeros((n_c,n_o),dtype=object)
#        for i in xrange(0,n_o):
#            dO_ddelta_alpha_i=basis.D_O_I_D_delta_alpha(self.geo,dO_I_ddelta_bar_list[i])
#            for j in xrange(0,n_c):
#                Ts[j,i] = fisher_set[j].contract_chol_right(dO_ddelta_alpha_i)
#            #Ts[i] = basis.D_O_I_D_delta_alpha(self.geo,dO_I_ddelta_bar_list[i])
#        for i in xrange(0,n_o):
#            n2 = 0
#          #  T_i = basis.D_O_I_D_delta_alpha(self.geo,dO_I_ddelta_bar_list[i])
#         #   print "sw_survey: T_i shape: "+str(T_i.shape)
#            for j in xrange(0,i+1):
#                print "sw_survey "+str(self.get_survey_id())+": Calc d delta alpha for observable 1,2 #:"+str(i)+","+str(j)
#                #print Ts[i]
#                #print Ts[j]
#                #print np.dot(Ts[i].T,Ts[j])
#              #  T_j = basis.D_O_I_D_delta_alpha(self.geo,dO_I_ddelta_bar_list[j])
#               # cov_mats[n1:n1+ds[i],n2:n2+ds[j]] = np.dot(np.dot(Ts[i].T,fisher.get_covar()),Ts[j])
#                #cov_mats[n1:n1+ds[i],n2:n2+ds[j]] = fisher.contract_covar(Ts[i].T,Ts[j])
#                for k in xrange(0,n_c):
#                    cov_mats[k,n1:n1+ds[i],n2:n2+ds[j]] = np.dot(Ts[k,i].T,Ts[k,j])
#                    if not i==j:
#                        cov_mats[k,n2:n2+ds[i],n1:n1+ds[j]] = cov_mats[k,n1:n1+ds[i],n2:n2+ds[j]].T
#
#
#                n2+=ds[j]
#            n1+=ds[i]
#        print "sw_survey: finished computing sw covariance matrices"
#        return cov_mats
    
#    def get_nongaussian_covar(self):
#        return np.zeros((self.get_total_dimension(),self.get_total_dimension()))
    
#    #get the covariance matrices, and a fisher matrix for the total covariance
#    def get_covars(self,fisher_set,basis):
#        cov_mats =  CovMat(self.get_gaussian_cov(),self.get_nongaussian_cov(),self.get_SSC_cov(fisher_set,basis),self.get_total_dimension(),self.param_priors)
#        #fisher_c = fm.FisherMatrix(cov_mats.get_total_covar(),input_type=fm.REP_COVAR,fix_input=False)
#        return cov_mats
   
#    def get_cov_tot_pars(self,cov_mats,gaussian_only=False):
#        v = self.get_dO_I_dpar_array()
#        return cov_mats.get_cov_sum_param_basis(v,gaussian_only=gaussian_only)
        
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
            for i in xrange(0,rbins.shape[0]):
                r1 = rbins[i]
                if cross_bins:
                    for j in xrange(0,rbins.shape[0]):
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
    import matter_power_spectrum as mps
    Theta = [np.pi/4.,np.pi/2.]
    Phi = [0,np.pi/3.]
    #d=np.loadtxt('camb_m_pow_l.dat')
    #k=d[:,0]; P=d[:,1]
    C=cp.CosmoPie(cosmology=defaults.cosmology)
    P = mps.MatterPower(C)
    C.P_lin = P
    C.k = P.k
    zs = np.array([0.1,0.8])
    z_fine = np.arange(0.01,np.max(zs),0.01)
    ls = np.arange(2,500)
    geo = rect_geo(zs,Theta,Phi,C,z_fine)
    sw_survey = SWSurvey(geo,'survey1',C,ls)
    O_I_array = sw_survey.get_O_I_array()
    dO_I_ddelta_bar_list = sw_survey.get_dO_I_ddelta_bar_list()
    import matplotlib.pyplot as plt
    ax  = plt.subplot(111)
    ax.loglog(ls,O_I_array[0:ls])
    ax.loglog(ls,dO_I_ddelta_bar_list[0])
    plt.xlabel('ls')
    plt.legend(['O_I','dO_I_ddelta_bar'])
    plt.show()

