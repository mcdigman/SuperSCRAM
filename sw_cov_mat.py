"""
SWCovMat calculates gaussian and nongaussian contributions to the covariance
"""
import numpy as np
import lensing_observables as lo
from warnings import warn
    
class SWCovMat(object):
    def __init__(self,O_I_1,O_I_2,debugging=False):
        """Object to handle retrieving the non SSC short wavelength covariance matrices"""
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

                ns = np.zeros(4)
                #TODO check ns, maybe should use isclose?
                if np.all(r1_1 == r2_1):
                    ns[0] = sh_pow1.get_n_shape(class_a,class_c)
                if np.all(r1_1 == r2_2):
                    ns[1] = sh_pow1.get_n_shape(class_a,class_d)
                if np.all(r1_2 == r2_2):
                    ns[2] = sh_pow1.get_n_shape(class_b,class_d)
                if np.all(r1_2 == r2_1):
                    ns[3] = sh_pow1.get_n_shape(class_b,class_c)
                #ns = np.zeros(4)
                #ns[0:2] =
                self.gaussian_covar = np.diagflat(sh_pow1.cov_g_diag(np.array([O_I_1.q1_pow,O_I_1.q2_pow,O_I_2.q1_pow,O_I_2.q2_pow]),ns))
                self.dimension = self.gaussian_covar.shape[0]
                print "sw_cov_mat: covariance retrieved"

                if debugging:
                    #check that covariance matrix possesses all expected symmetries
                    assert(np.all(self.gaussian_covar.T==self.gaussian_covar))
                    assert(np.all(np.linalg.eigh(self.gaussian_covar)[0]>0.))
                    ns2 = np.array([ns[3],ns[2],ns[1],ns[0]])
                    gaussian_covar2 = np.diagflat(sh_pow1.cov_g_diag(np.array([O_I_1.q2_pow,O_I_1.q1_pow,O_I_2.q1_pow,O_I_2.q2_pow]),ns2))
                    assert(np.all(gaussian_covar2==self.gaussian_covar))
                    ns3 = np.array([ns[2],ns[3],ns[0],ns[1]])
                    gaussian_covar3 = np.diagflat(sh_pow1.cov_g_diag(np.array([O_I_1.q2_pow,O_I_1.q1_pow,O_I_2.q2_pow,O_I_2.q1_pow]),ns3))
                    assert(np.all(gaussian_covar3==self.gaussian_covar))
                    ns4 = np.array([ns[1],ns[0],ns[3],ns[2]])
                    gaussian_covar4 = np.diagflat(sh_pow1.cov_g_diag(np.array([O_I_1.q1_pow,O_I_1.q2_pow,O_I_2.q2_pow,O_I_2.q1_pow]),ns4))
                    assert(np.all(gaussian_covar4==self.gaussian_covar))
            else:
                warn("sw_cov_mat: unhandled observable pair in constructor")
        else:
            warn("sw_cov_mat: unhandled observable pair in constructor")
        self.nongaussian_covar = np.zeros(self.gaussian_covar.shape)

    def get_gaussian_covar_array(self):
        """get gaussian covariance"""
        return self.gaussian_covar

    def get_nongaussian_covar_array(self):
        """get nongaussian covariance"""
        return self.nongaussian_covar
