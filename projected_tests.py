import cosmopie as cp
import numpy as np
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline
import shear_power as sp
import defaults
from warnings import warn
import unittest

#class TestCosmosisAgreement():

class TestCosmosisAgreement1(unittest.TestCase):
    #test agreement with modified cosmosis demo 15 results 
    #assuming gaussian matter distribution with sigma=0.4 and average z=1
    #will use power spectrum grid directly from cosmosis
    def test_cosmosis_match(self): 
            TOLERANCE_MAX = 0.1
            TOLERANCE_MEAN = 0.05

            C=cp.CosmoPie(cosmology=defaults.cosmology_cosmosis)
            k_in = np.loadtxt('test_inputs/proj_1/k_h.txt')
            zs = np.loadtxt('test_inputs/proj_1/z.txt')
            zs[0] = 10**-3

            ls = np.loadtxt('test_inputs/proj_1/ell.txt')

            sp1 = sp.shear_power(k_in,C,zs,ls,pmodel='cosmosis_nonlinear',cosmology_in=defaults.cosmology_cosmosis)

            sh_pow1 = sp.Cll_sh_sh(sp1).Cll()
            sh_pow1_gg = sp.Cll_g_g(sp1).Cll()
            sh_pow1_sg = sp.Cll_sh_g(sp1).Cll()
            sh_pow1_mm = sp.Cll_mag_mag(sp1).Cll()
    
            sh_pow_cosm = np.loadtxt('test_inputs/proj_1/ss_pow.txt')
            gal_pow_cosm = np.loadtxt('test_inputs/proj_1/gg_pow.txt')
            sg_pow_cosm = np.loadtxt('test_inputs/proj_1/sg_pow.txt')
            mm_pow_cosm = np.loadtxt('test_inputs/proj_1/mm_pow.txt')
    
#            import matplotlib.pyplot as plt
    
#            ax = plt.subplot(111)
#            ax.set_xlabel('l',size=20)
#            ax.set_ylabel('l(l+1)$C^{AB}(2\pi)^{-1}$')
#            ax.loglog(ls*C.h,ls*(ls*C.h+1.)*sh_pow1/(2.*np.pi)/C.h**2)
#            ax.loglog(ls*C.h,ls*(ls*C.h+1.)*sh_pow1_gg/(2.*np.pi)/C.h**2)
#            ax.loglog(ls*C.h,ls*(ls*C.h+1.)*sh_pow1_sg/(2.*np.pi)/C.h**2)
#            ax.loglog(ls*C.h,ls*(ls*C.h+1.)*sh_pow1_mm/(2.*np.pi)/C.h)
#            ax.loglog(ls,ls*(ls+1.)*sh_pow_cosm/(2.*np.pi))
#            ax.loglog(ls,ls*(ls+1.)*gal_pow_cosm/(2.*np.pi))
#            ax.loglog(ls,ls*(ls+1.)*sg_pow_cosm/(2.*np.pi))
#            ax.loglog(ls,ls*(ls+1.)*mm_pow_cosm/(2.*np.pi))
#            ax.legend(["ssc_ss","ssc_gg","ssc_sg","ssc_mm","cosm_ss","cosm_gg","cosm_sg","cosm_mm"],loc=2)
    
            #get ratio of calculated value to expected value from cosmosis
            #use -np.inf as filler for interpolation when l value is not in ls*C.h,filter it later
            ss_rat = (sh_pow_cosm-(interp1d(ls*C.h,sh_pow1/C.h**3,bounds_error=False,fill_value=-np.inf)(ls)))/sh_pow_cosm
    
            gg_rat = (gal_pow_cosm-(interp1d(ls*C.h,sh_pow1_gg/C.h**3,bounds_error=False,fill_value=-np.inf)(ls)))/gal_pow_cosm
            sg_rat = (sg_pow_cosm-(interp1d(ls*C.h,sh_pow1_sg/C.h**3,bounds_error=False,fill_value=-np.inf)(ls)))/sg_pow_cosm
            mm_rat = (mm_pow_cosm-(interp1d(ls*C.h,sh_pow1_mm/C.h**2,bounds_error=False,fill_value=-np.inf)(ls)))/mm_pow_cosm 
            ###TODO### examine discrepancy in powers of h (comes from magnification_prefactor)

            mean_ss_err = np.mean(abs(ss_rat)[abs(ss_rat)<np.inf])
            mean_gg_err = np.mean(abs(gg_rat)[abs(gg_rat)<np.inf])
            mean_sg_err = np.mean(abs(sg_rat)[abs(sg_rat)<np.inf])
            mean_mm_err = np.mean(abs(mm_rat)[abs(mm_rat)<np.inf])

            max_ss_err = max((abs(ss_rat))[abs(ss_rat)<np.inf])
            max_gg_err = max((abs(gg_rat))[abs(gg_rat)<np.inf])
            max_sg_err = max((abs(sg_rat))[abs(sg_rat)<np.inf])
            max_mm_err = max((abs(mm_rat))[abs(mm_rat)<np.inf])
            
            print "ss agreement within: "+str(max_ss_err*100.)+"%"+" mean agreement: "+str(mean_ss_err*100.)+"%"
            print "gg agreement within: "+str(max_gg_err*100.)+"%"+" mean agreement: "+str(mean_gg_err*100.)+"%"
            print "sg agreement within: "+str(max_sg_err*100.)+"%"+" mean agreement: "+str(mean_sg_err*100.)+"%"
            print "mm agreement within: "+str(max_mm_err*100.)+"%"+" mean agreement: "+str(mean_mm_err*100.)+"%"
            
            self.assertTrue(max_ss_err<TOLERANCE_MAX)
            self.assertTrue(max_gg_err<TOLERANCE_MAX)
            self.assertTrue(max_sg_err<TOLERANCE_MAX)
            self.assertTrue(max_mm_err<TOLERANCE_MAX)
            self.assertTrue(mean_ss_err<TOLERANCE_MEAN)
            self.assertTrue(mean_gg_err<TOLERANCE_MEAN)
            self.assertTrue(mean_sg_err<TOLERANCE_MEAN)
            self.assertTrue(mean_mm_err<TOLERANCE_MEAN)
#            plt.grid()
#            plt.show() 

class TestCosmosisHalofitAgreement1(unittest.TestCase):
    #test agreement with modified cosmosis demo 15 results 
    #assuming gaussian matter distribution with sigma=0.4 and average z=1
    #use halofit power spectrum grid
    def test_cosmosis_match(self): 
            TOLERANCE_MAX = 0.15
            TOLERANCE_MEAN = 0.15
            C=cp.CosmoPie(cosmology=defaults.cosmology_cosmosis)
            d = np.loadtxt('test_inputs/proj_1/camb_m_pow_l.dat')
            #d = np.loadtxt('test_inputs/proj_1/p_k_lin.dat')
            k_in = d[:,0]
            P_in = d[:,1]
            zs = np.loadtxt('test_inputs/proj_1/z.txt')
            zs[0] = 10**-3

            ls = np.loadtxt('test_inputs/proj_1/ell.txt')

            sp1 = sp.shear_power(k_in,C,zs,ls,pmodel='cosmosis_nonlinear',cosmology_in=defaults.cosmology_cosmosis)
            sh_pow1 = sp.Cll_sh_sh(sp1).Cll()
            sh_pow1_gg = sp.Cll_g_g(sp1).Cll()
            sh_pow1_sg = sp.Cll_sh_g(sp1).Cll()
            sh_pow1_mm = sp.Cll_mag_mag(sp1).Cll()
   
            sp2 = sp.shear_power(k_in,C,zs,ls,P_in=P_in,pmodel='halofit_nonlinear',cosmology_in=defaults.cosmology_cosmosis)
            sh_pow2 = sp.Cll_sh_sh(sp2).Cll()
            sh_pow2_gg = sp.Cll_g_g(sp2).Cll()
            sh_pow2_sg = sp.Cll_sh_g(sp2).Cll()
            sh_pow2_mm = sp.Cll_mag_mag(sp2).Cll()

            #get ratio of calculated value to expected value from cosmosis
            #use -np.inf as filler for interpolation when l value is not in ls*C.h,filter it later
            ss_rat = (sh_pow2-sh_pow1)/sh_pow2
    
            gg_rat = (sh_pow2_gg-sh_pow1_gg)/sh_pow2_gg
            sg_rat = (sh_pow2_sg-sh_pow1_sg)/sh_pow2_sg
            mm_rat = (sh_pow2_mm-sh_pow1_mm)/sh_pow2_mm
            ###TODO### examine discrepancy in powers of h (comes from magnification_prefactor)

            mean_ss_err = np.mean(abs(ss_rat)[abs(ss_rat)<np.inf])
            mean_gg_err = np.mean(abs(gg_rat)[abs(gg_rat)<np.inf])
            mean_sg_err = np.mean(abs(sg_rat)[abs(sg_rat)<np.inf])
            mean_mm_err = np.mean(abs(mm_rat)[abs(mm_rat)<np.inf])

            max_ss_err = max((abs(ss_rat))[abs(ss_rat)<np.inf])
            max_gg_err = max((abs(gg_rat))[abs(gg_rat)<np.inf])
            max_sg_err = max((abs(sg_rat))[abs(sg_rat)<np.inf])
            max_mm_err = max((abs(mm_rat))[abs(mm_rat)<np.inf])
            
            print "ss agreement within: "+str(max_ss_err*100.)+"%"+" mean agreement: "+str(mean_ss_err*100.)+"%"
            print "gg agreement within: "+str(max_gg_err*100.)+"%"+" mean agreement: "+str(mean_gg_err*100.)+"%"
            print "sg agreement within: "+str(max_sg_err*100.)+"%"+" mean agreement: "+str(mean_sg_err*100.)+"%"
            print "mm agreement within: "+str(max_mm_err*100.)+"%"+" mean agreement: "+str(mean_mm_err*100.)+"%"
            
            self.assertTrue(max_ss_err<TOLERANCE_MAX)
            self.assertTrue(max_gg_err<TOLERANCE_MAX)
            self.assertTrue(max_sg_err<TOLERANCE_MAX)
            self.assertTrue(max_mm_err<TOLERANCE_MAX)
            self.assertTrue(mean_ss_err<TOLERANCE_MEAN)
            self.assertTrue(mean_gg_err<TOLERANCE_MEAN)
            self.assertTrue(mean_sg_err<TOLERANCE_MEAN)
            self.assertTrue(mean_mm_err<TOLERANCE_MEAN)

    
if __name__=='__main__':
    unittest.main()
