import numpy as np
import cosmopie as cp
import sw_observable as swo
import lensing_observables as lo
import shear_power as sp
from warnings import warn
from algebra_utils import get_inv_cholesky

import sys
class CovMat:
    def __init__(self,gaussian_covar,nongaussian_covar,ssc_covar,dimension):
        self.gaussian_covar = gaussian_covar
        self.nongaussian_covar = nongaussian_covar
        self.ssc_covar = ssc_covar 
        self.dimension = dimension

    def get_gaussian_covar(self):
        return self.gaussian_covar
    def get_nongaussian_covar(self):
        return self.nongaussian_covar
    def get_ssc_covar(self):
        return self.ssc_covar
    def get_total_covar(self):
        return self.gaussian_covar+self.nongaussian_covar+self.ssc_covar
    def get_dimension(self):
        return self.dimension
    #TODO some of this behavior may be replicated in fisher_matrix
    #TODO maybe use eigh instead
    def get_SS_eig(self):
        chol_cov = get_inv_cholesky(self.get_gaussian_covar())
        #chol_cov = scipy.linalg.cholesky(self.get_gaussian_covar(),lower=True)
        mat_retrieved = (np.identity(self.get_dimension())+np.dot(np.dot(chol_cov,self.ssc_covar),chol_cov.T))
        return np.linalg.eigh(mat_retrieved)
    
    

class SWCovMat(CovMat):
    def __init__(self,O_I_1,O_I_2):
        self.gaussian_covar = 0.
        self.dimension = 0
        if isinstance(O_I_1,lo.LensingObservable) and isinstance(O_I_2,lo.LensingObservable):
            if O_I_1.get_survey_id() == O_I_2.get_survey_id():
                print "sw_cov_mat: retrieving covariance"
                class_a = O_I_1.q1_pow.__class__
                class_b = O_I_1.q2_pow.__class__
                class_c = O_I_1.q1_pow.__class__
                class_d = O_I_1.q2_pow.__class__
                
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
                self.gaussian_covar = np.diagflat(sh_pow1.cov_g_diag2(np.array([O_I_1.q1_pow,O_I_1.q2_pow,O_I_2.q1_pow,O_I_2.q2_pow]),ns))
                self.dimension = self.gaussian_covar.shape[0]
                print "sw_cov_mat: covariance retrieved"
            else:
                warn("sw_cov_mat: unhandled observable pair in constructor")
        else:
            warn("sw_cov_mat: unhandled observable pair in constructor")
        #TODO handle nongaussian covariance,return matrix
        self.nongaussian_covar = 0.
        self.ssc_covar = 0.

