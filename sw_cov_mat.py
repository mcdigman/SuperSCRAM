import numpy as np
import cosmopie as cp
import sw_observable as swo
import lensing_observables as lo
import shear_power as sp
from warnings import warn
from algebra_utils import get_inv_cholesky
import fisher_matrix as fm
import sys
#class CovMat:
#    def __init__(self,gaussian_covar,nongaussian_covar,ssc_covar_set,dimension,param_prior=None):
#        self.f_gaussian = fm.fisher_matrix(gaussian_covar,input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR,fix_input=False,silent=True)
#        self.f_nongaussian = fm.fisher_matrix(nongaussian_covar,input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR,fix_input=False,silent=True)
#        self.n_ssc = ssc_covar_set.shape[0]
#        self.f_ssc_set = np.zeros(self.n_ssc,dtype=object)
#        self.f_tot_set = np.zeros(self.n_ssc,dtype=object)
#        for i in xrange(0,self.n_ssc):
#            self.f_ssc_set[i] = fm.fisher_matrix(ssc_covar_set[i],input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR,fix_input=False,silent=True)
#            self.f_tot_set[i] = fm.fisher_matrix(gaussian_covar+nongaussian_covar+ssc_covar_set[i],input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR,fix_input=False,silent=True)
#        self.param_prior = param_prior
#        self.dimension = dimension
#
#    def get_gaussian_covar(self):
#        return self.f_gaussian.get_covar()
#    def get_nongaussian_covar(self):
#        return self.f_nongaussian.get_covar()
#    def get_ssc_covar(self):
#        ssc_cov_set = np.zeros((self.n_ssc,self.dimension,self.dimension))
#        for i in xrange(0,self.n_ssc):
#            ssc_cov_set[i] = self.f_ssc_set[i].get_covar()
#        return ssc_cov_set
#    def get_total_covar(self):
#        tot_cov_set = np.zeros((self.n_ssc,self.dimension,self.dimension))
#        for i in xrange(0,self.n_ssc):
#            tot_cov_set[i] = self.f_tot_set[i].get_covar()
#        return tot_cov_set
#        #return self.get_gaussian_covar()+self.get_nongaussian_covar()+self.get_ssc_covar()
#    def get_dimension(self):
#        return self.dimension
#    def get_cov_sum_param_basis(self,basis,gaussian_only=False):
#        #handle empty basis to avoid errors
#        if basis.size==0:
#            print "CovMat: no elements in basis"
#            return np.array([])
#        else:
#            f_tot_param_set = np.zeros(self.n_ssc,dtype=object)
#            if gaussian_only:
#                for i in xrange(0,self.n_ssc):
#                    f_tot_param_set[i] = self.f_gaussian.project_fisher(basis)
#                self.f_tot_save_g = f_tot_param_set
#                print "g",f_tot_param_set[0].get_covar()
#            else:
#                #f_ng_param =self.f_nongaussian.project_fisher(basis)
#                #f_ssc_param = self.f_ssc.project_fisher(basis)
#                for i in xrange(0,self.n_ssc):
#                    f_tot_param_set[i] = self.f_tot_set[i].project_fisher(basis)
#                self.f_tot_save_full = f_tot_param_set
#                #print "ssc",f_ssc_param.get_covar()
#                #print "g",f_g_param.get_covar()
#            if self.param_prior is not None:
#                #TODO watch number of inverses
#                for i in xrange(0,self.n_ssc):
#                    f_tot_param_set[i].add_fisher(self.param_prior)
#            #TODO save n_params a better way
#            n_params = self.param_prior.shape[0]
#            cov_set = np.zeros((self.n_ssc,n_params,n_params))
#            for i in xrange(0,self.n_ssc):
#                cov_set[i] = f_tot_param_set[i].get_covar()
#            return cov_set#f_ng_param.get_covar()+f_ssc_param.get_covar() 

#    def get_SS_eig(self):
#        chol_cov = get_inv_cholesky(self.get_gaussian_covar())
#        #chol_cov = scipy.linalg.cholesky(self.get_gaussian_covar(),lower=True)
#        mat_retrieved_set = np.zeros((self.n_ssc,self.dimension,self.dimension))
#        eig_set = np.zeros(self.n_ssc,dtype=object)
#        ssc_cov_set = self.get_ssc_covar()
#        for i in xrange(0,mat_retrieved_set.shape[0]):
#            #TODO do not really need to save mat_retrieved_set
#            mat_retrieved_set[i] = (np.identity(self.dimension)+np.dot(np.dot(chol_cov,ssc_cov_set[i]),chol_cov.T))
#            eig_set[i] = np.linalg.eigh(mat_retrieved_set[i])
#        return eig_set
#
#    #TODO refactor
#    def get_SS_eig_param(self,basis,cross=False):
#        f_g_param =  self.f_gaussian.project_fisher(basis)
#        f_tot_param_set = np.zeros(self.n_ssc,dtype=object)
#        for i in xrange(0,self.n_ssc):
#            f_tot_param_set[i] = self.f_tot_set[i].project_fisher(basis)
#            if self.param_prior is not None:
#                f_tot_param_set[i].add_fisher(self.param_prior)
#        if self.param_prior is not None:
#            f_g_param.add_fisher(self.param_prior)
#        if not cross:
#            chol_cov = get_inv_cholesky(f_g_param.get_covar())
#            n_param = self.param_prior.shape[0]
#            mat_retrieved_set = np.zeros((self.n_ssc,n_param,n_param))
#            eig_set = np.zeros(self.n_ssc,dtype=object)
#            for i in xrange(0,self.n_ssc):
#                mat_retrieved_set[i] = (np.identity(chol_cov.shape[0])+np.dot(np.dot(chol_cov,f_tot_param_set[i].get_covar()),chol_cov.T))
#                eig_set[i] = np.linalg.eigh(mat_retrieved_set[i])
#        else:
#            n_param = self.param_prior.shape[0]
#            eig_set = np.zeros((self.n_ssc,self.n_ssc),dtype=object)
#            mat_retrieved_set = np.zeros((self.n_ssc,self.n_ssc,n_param,n_param))
#            for i in xrange(0,self.n_ssc):
#                chol_cov = get_inv_cholesky(f_tot_param_set[i].get_covar())
#                for j in xrange(0,self.n_ssc):
#                    mat_retrieved_set[i,j] = (np.identity(chol_cov.shape[0])+np.dot(np.dot(chol_cov,f_tot_param_set[j].get_covar()),chol_cov.T))
#                    eig_set[i,j] = np.linalg.eigh(mat_retrieved_set[i,j])
#
#        return eig_set



    
    
class SWCovMat:
    def __init__(self,O_I_1,O_I_2):
        self.gaussian_covar = 0.
        self.dimension = 0
        if isinstance(O_I_1,lo.LensingObservable) and isinstance(O_I_2,lo.LensingObservable):
            if O_I_1.get_survey_id() == O_I_2.get_survey_id():
                print "sw_cov_mat: retrieving covariance"
                class_a = O_I_1.q1_pow.__class__
                class_b = O_I_1.q2_pow.__class__
                class_c = O_I_2.q1_pow.__class__
                class_d = O_I_2.q2_pow.__class__
                
                #under current assumptions only need sh_pow1
                sh_pow1 = O_I_1.len_pow.C_pow
                sh_pow2 = O_I_2.len_pow.C_pow
                r1_1 = O_I_1.r1
                r1_2 = O_I_1.r2
                r2_1 = O_I_2.r1
                r2_2 = O_I_2.r2
                n_ac = 0.
                n_ad = 0.
                n_bd = 0.
                n_bc = 0.
                if np.all(r1_1 == r2_1):
                    n_ac = sh_pow1.get_n_shape(class_a,class_c)
                if np.all(r1_1 == r2_2):
                    n_ad = sh_pow1.get_n_shape(class_a,class_d)
                if np.all(r1_2 == r2_2):
                    n_bd = sh_pow1.get_n_shape(class_b,class_d)
                if np.all(r1_2 == r2_1):
                    n_bc = sh_pow1.get_n_shape(class_b,class_c)
                ns = np.array([n_ac,n_ad,n_bd,n_bc])
                #TODO fix ns
                self.gaussian_covar = np.diagflat(sh_pow1.cov_g_diag(np.array([O_I_1.q1_pow,O_I_1.q2_pow,O_I_2.q1_pow,O_I_2.q2_pow]),ns))
                self.dimension = self.gaussian_covar.shape[0]
                print "sw_cov_mat: covariance retrieved"
            else:
                warn("sw_cov_mat: unhandled observable pair in constructor")
        else:
            warn("sw_cov_mat: unhandled observable pair in constructor")
        self.nongaussian_covar = np.zeros(self.gaussian_covar.shape)
#        self.ssc_covar = np.array([0.])
#        CovMat.__init__(self,self.gaussian_covar,self.nongaussian_covar,self.ssc_covar,self.dimension)
    def get_gaussian_covar_array(self):
        return self.gaussian_covar
    def get_nongaussian_covar_array(self):
        return self.nongaussian_covar
