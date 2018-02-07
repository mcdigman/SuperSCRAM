"""Some tests of projected observables"""
import unittest
import numpy as np
from scipy.interpolate import interp1d
import cosmopie as cp
import shear_power as sp
import defaults
import matter_power_spectrum as mps
#base params: Omegam,h,OmegaL,sigma8,Omegab
COSMOLOGY_COSMOSIS = {  'Omegab'   :0.049,
                        'Omegabh2' :0.0219961,
                        'Omegac'   :0.261,
                        'Omegach2' :0.117163,
                        'Omegamh2' :0.139159,
                        'OmegaL'   :.69,
                        'OmegaLh2' :0.309741,
                        'Omegam'   :.31,
                        'H0'       :67,
                        'sigma8'   :.81,
                        'h'        :.67,
                        'Omegak'   :0.0,
                        'Omegakh2' :0.0,
                        'Omegar'   :0.0,
                        'Omegarh2' :0.0,
                        'tau'      :None, # eventually fix to real cosmosis values
                        'Yp'       :None,
                        'As'       :2.143*10**-9, #not correct As
                        'ns'       :0.9681,
                        'LogAs'    :np.log(2.143*10**-9),
                        'mnu'      :0. #guess
                     }
#class TestCosmosisAgreement():

class TestCosmosisAgreement1(unittest.TestCase):
    """test agreement with modified cosmosis demo 15 results
    assuming gaussian matter distribution with sigma=0.4 and average z=1
    will use power spectrum grid directly from cosmosis"""
    def test_cosmosis_match(self):
        """test function"""
        TOLERANCE_MAX = 0.1
        TOLERANCE_MEAN = 0.05
        cosmo_fid = COSMOLOGY_COSMOSIS.copy()
        C = cp.CosmoPie(cosmo_fid,p_space='jdem')
        k_in = np.loadtxt('test_inputs/proj_2/k_h.txt')*C.h
        C.k = k_in
        zs = np.loadtxt('test_inputs/proj_2/z.txt')
        zs[0] = 10**-3

        ls = np.loadtxt('test_inputs/proj_2/ell.txt')

        omega_s = np.pi/(3.*np.sqrt(2.))
        params = defaults.lensing_params.copy()
        params['zbar'] = 1.0
        params['sigma'] = 0.4
        params['smodel'] = 'gaussian'
        params['l_min'] = np.min(ls)
        params['l_max'] = np.max(ls)
        params['n_l'] = ls.size
        params['n_gal'] = 118000000*6.
        sp1 = sp.ShearPower(C,zs,omega_s,params, pmodel='cosmosis',mode='power')

        sh_pow1 = sp.Cll_sh_sh(sp1).Cll()
        sh_pow1_gg = sp.Cll_g_g(sp1).Cll()
        sh_pow1_sg = sp.Cll_sh_g(sp1).Cll()
        sh_pow1_mm = sp.Cll_mag_mag(sp1).Cll()

        sh_pow_cosm = np.loadtxt('test_inputs/proj_2/ss_pow.txt')
        gal_pow_cosm = np.loadtxt('test_inputs/proj_2/gg_pow.txt')
        sg_pow_cosm = np.loadtxt('test_inputs/proj_2/sg_pow.txt')
        mm_pow_cosm = np.loadtxt('test_inputs/proj_2/mm_pow.txt')


#            import matplotlib.pyplot as plt

#            ax = plt.subplot(111)
#            ax.set_xlabel('l',size=20)
#            ax.set_ylabel('l(l+1)$C^{AB}(2\pi)^{-1}$')
#            ax.loglog(ls,sh_pow1)
#            ax.loglog(ls*C.h,ls*(ls*C.h+1.)*sh_pow1_gg/(2.*np.pi)/C.h**2)
#            ax.loglog(ls*C.h,ls*(ls*C.h+1.)*sh_pow1_sg/(2.*np.pi)/C.h**2)
#            ax.loglog(ls*C.h,ls*(ls*C.h+1.)*sh_pow1_mm/(2.*np.pi)/C.h)
#            ax.loglog(ls,sh_pow_cosm)
#            ax.loglog(ls,ls*(ls+1.)*gal_pow_cosm/(2.*np.pi))
#            ax.loglog(ls,ls*(ls+1.)*sg_pow_cosm/(2.*np.pi))
#            ax.loglog(ls,ls*(ls+1.)*mm_pow_cosm/(2.*np.pi))
#            ax.legend(["ssc_ss","ssc_gg","ssc_sg","ssc_mm","cosm_ss","cosm_gg","cosm_sg","cosm_mm"],loc=2)

        #get ratio of calculated value to expected value from cosmosis
        #use -np.inf as filler for interpolation when l value is not in ls*C.h,filter it later
        ss_rat = (sh_pow_cosm-(interp1d(ls,sh_pow1,bounds_error=False,fill_value=-np.inf)(ls)))/sh_pow_cosm
        gg_rat = (gal_pow_cosm-(interp1d(ls,sh_pow1_gg,bounds_error=False,fill_value=-np.inf)(ls)))/gal_pow_cosm
        sg_rat = (sg_pow_cosm-(interp1d(ls,sh_pow1_sg,bounds_error=False,fill_value=-np.inf)(ls)))/sg_pow_cosm
        mm_rat = (mm_pow_cosm-(interp1d(ls,sh_pow1_mm*C.h,bounds_error=False,fill_value=-np.inf)(ls)))/mm_pow_cosm
        ###TODO### examine discrepancy in powers of h (comes from magnification_prefactor)
#           ax.plot(ls,ss_rat)
#           plt.show()
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
    """test agreement with modified cosmosis demo 15 results
    assuming gaussian matter distribution with sigma=0.4 and average z=1
    use halofit power spectrum grid"""
    def test_cosmosis_match(self):
        """test function"""
        TOLERANCE_MAX = 0.2
        TOLERANCE_MEAN = 0.2
        power_params = defaults.power_params.copy()
        power_params.camb['force_sigma8'] = True
        power_params.camb['maxkh'] = 1e5
        power_params.camb['kmax'] = 100.
        power_params.camb['npoints'] = 1000
        C = cp.CosmoPie(cosmology=COSMOLOGY_COSMOSIS,p_space='jdem')
        #d = np.loadtxt('test_inputs/proj_1/camb_m_pow_l.dat')
        #d = np.loadtxt('test_inputs/proj_1/p_k_lin.dat')
        #k_in = d[:,0]
        #P_in = d[:,1]
        #k_in,P_in = camb_pow(COSMOLOGY_COSMOSIS)
        P_in = mps.MatterPower(C,power_params)
        k_in = P_in.k
        C.k = k_in
        C.P_lin = P_in
        zs = np.loadtxt('test_inputs/proj_2/z.txt')
        zs[0] = 10**-3

        ls = np.loadtxt('test_inputs/proj_2/ell.txt')
        omega_s = np.pi/(3.*np.sqrt(2.))
        params = defaults.lensing_params.copy()
        params['zbar'] = 1.0
        params['sigma'] = 0.40
        params['smodel'] = 'gaussian'
        params['l_min'] = np.min(ls)
        params['l_max'] = np.max(ls)
        params['n_l'] = ls.size
        params['n_gal'] = 118000000*6.
        sp1 = sp.ShearPower(C,zs,omega_s,params,pmodel='cosmosis',mode='power')
        sh_pow1 = sp.Cll_sh_sh(sp1).Cll()
        sh_pow1_gg = sp.Cll_g_g(sp1).Cll()
        sh_pow1_sg = sp.Cll_sh_g(sp1).Cll()
        sh_pow1_mm = sp.Cll_mag_mag(sp1).Cll()

        sp2 = sp.ShearPower(C,zs,omega_s,params,pmodel='halofit',mode='power')
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
