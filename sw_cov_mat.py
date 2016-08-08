import numpy as np
import cosmopie as cp
import sw_observable as swo
import lensing_observables as lo
import shear_power as sp


class SWCovMat:
    def __init__(self,O_I_1,O_I_2):
        self.gaussian_covar = 0.
        if isinstance(O_I_1,lo.LensingObservable) and isinstance(O_I_2,lo.LensingObservable):
            if O_I_1.get_survey_id() == O_I_2.get_survey_id():
                #TODO fix ns
                class_a = O_I_1.q1_pow.__class__
                class_b = O_I_1.q2_pow.__class__
                class_c = O_I_1.q1_pow.__class__
                class_d = O_I_1.q2_pow.__class__
                
                #under current assumptions only need sh_pow1
                sh_pow1 = O_I_1.len_pow.C_pow
                sh_pow2 = O_I_2.len_pow.C_pow
                n_ac = sh_pow1.get_n_chip(class_a,class_c)
                n_ad = sh_pow1.get_n_chip(class_a,class_d)
                n_bd = sh_pow1.get_n_chip(class_b,class_d)
                n_bc = sh_pow1.get_n_chip(class_b,class_c)
                ns = np.array([n_ac,n_ad,n_bd,n_bc])
                print ns

                self.gaussian_covar = sh_pow1.cov_g_diag2(np.array([O_I_1.q1_pow,O_I_1.q2_pow,O_I_2.q1_pow,O_I_2.q2_pow]),ns) 
        self.nongaussian_covar=0.
    def get_gaussian_covar(self):
        return self.gaussian_covar
    def get_nongaussian_covar(self):
        return self.nongaussian_covar
    def get_total_covar(self):
        return self.gaussian_covar+self.nongaussian_covar
