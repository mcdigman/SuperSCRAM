"""
SWCovMat calculates gaussian and nongaussian contributions to the covariance
"""
from __future__ import division,print_function,absolute_import
from builtins import range
from warnings import warn
import numpy as np
import lensing_observables as lo

DEBUG = True

class SWCovMat(object):
    """Object to handle retrieving the non SSC short wavelength covariance matrices"""
    def __init__(self,O_I_1,O_I_2,silent=False):
        """ O_I_1,O_I_2: SWObservable objects
            debug: do debugging checks
            silent: whether to print(messages)
        """
        self.gaussian_covar = 0.
        if isinstance(O_I_1,lo.LensingObservable) and isinstance(O_I_2,lo.LensingObservable):
            if O_I_1.get_survey_id()==O_I_2.get_survey_id():
                if not silent:
                    print("SWCovMat: retrieving covariance")
                class_a = O_I_1.q1_pow.__class__
                class_b = O_I_1.q2_pow.__class__
                class_c = O_I_2.q1_pow.__class__
                class_d = O_I_2.q2_pow.__class__

                #under current assumptions only need sh_pow1
                sh_pow1 = O_I_1.len_pow.C_pow
                sh_pow2 = O_I_2.len_pow.C_pow
                z1_1 = O_I_1.z1
                z1_2 = O_I_1.z2
                z2_1 = O_I_2.z1
                z2_2 = O_I_2.z2

                ns = np.zeros(4)
                if np.allclose(z1_1, z2_1):
                    ns[0] = sh_pow1.get_n_shape(class_a,class_c)
                if np.allclose(z1_1, z2_2):
                    ns[1] = sh_pow1.get_n_shape(class_a,class_d)
                if np.allclose(z1_2 , z2_2):
                    ns[2] = sh_pow1.get_n_shape(class_b,class_d)
                if np.allclose(z1_2 , z2_1):
                    ns[3] = sh_pow1.get_n_shape(class_b,class_c)
                self.gaussian_covar = np.diagflat(sh_pow1.cov_g_diag(np.array([O_I_1.q1_pow,O_I_1.q2_pow,O_I_2.q1_pow,O_I_2.q2_pow]),ns))
                if not silent:
                    print("SWCovMat: covariance retrieved")

                if DEBUG:
                    #check that covariance matrix possesses all expected symmetries
                    assert np.all(self.gaussian_covar.T==self.gaussian_covar)
                    assert np.all(np.linalg.eigh(self.gaussian_covar)[0]>0.)
                    ns2 = np.array([ns[3],ns[2],ns[1],ns[0]])
                    gaussian_covar2 = np.diagflat(sh_pow1.cov_g_diag(np.array([O_I_1.q2_pow,O_I_1.q1_pow,O_I_2.q1_pow,O_I_2.q2_pow]),ns2))
                    assert np.all(gaussian_covar2==self.gaussian_covar)
                    ns3 = np.array([ns[2],ns[3],ns[0],ns[1]])
                    gaussian_covar3 = np.diagflat(sh_pow1.cov_g_diag(np.array([O_I_1.q2_pow,O_I_1.q1_pow,O_I_2.q2_pow,O_I_2.q1_pow]),ns3))
                    assert np.all(gaussian_covar3==self.gaussian_covar)
                    ns4 = np.array([ns[1],ns[0],ns[3],ns[2]])
                    gaussian_covar4 = np.diagflat(sh_pow1.cov_g_diag(np.array([O_I_1.q1_pow,O_I_1.q2_pow,O_I_2.q2_pow,O_I_2.q1_pow]),ns4))
                    assert np.all(gaussian_covar4==self.gaussian_covar)
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
