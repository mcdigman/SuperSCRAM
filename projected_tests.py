"""Some tests of projected observables"""
from __future__ import absolute_import,division,print_function
from builtins import range
import pytest
import numpy as np
#from scipy.interpolate import interp1d
import cosmopie as cp
import shear_power as sp
import defaults
import matter_power_spectrum as mps
import lensing_weight as lw
COSMOLOGY_COSMOSIS2 = { 'Omegab'   :0.04830,#fixed
                        'Omegabh2' :0.0222682803,#computed
                        'Omegac'   :0.2582,#computed
                        'Omegach2' :0.1190407862,#computed
                        'Omegamh2' :0.1413090665,#computed
                        'OmegaL'   :0.6935,#computed
                        'OmegaLh2' :0.3197319335,#computed
                        'Omegam'   :.3065,#fixed
                        'H0'       :67.90,#fixed
                        'sigma8'   :.8154,#fixed
                        'h'        :.6790,#fixed
                        'Omegak'   :0.0,#fixed
                        'Omegakh2' :0.0,#computed
                        'Omegar'   :0.0,#assumed
                        'Omegarh2' :0.0,#assumed
                        'tau'      :0.067, #fixed
                        'Yp'       :None,
                        'As'       :2.143*10**-9, #not correct As
                        'ns'       :0.9681,#fixed
                        'LogAs'    :np.log(2.143*10**-9),#fixed
                        'mnu'      :0.0,#guess
                        'w'        :-1.,#fixed
                        'w0'       :-1.,#fixed
                        'wa'       :0., #fixed
                        'de_model' :'constant_w'
                      }
#Note agreement is better when directly uses cosmosis input power spectrum (0.04%)
#class TestCosmosisAgreement1(unittest.TestCase):
#    """test agreement with modified cosmosis demo 15 results
#    assuming gaussian matter distribution with sigma=0.4 and average z=1
#    will use power spectrum grid directly from cosmosis"""
#    def test_cosmosis_match(self):
#        """test function"""
#        TOLERANCE_MAX = 0.1
#        TOLERANCE_MEAN = 0.05
#        cosmo_fid = COSMOLOGY_COSMOSIS2.copy()
#        C = cp.CosmoPie(cosmo_fid,p_space='jdem')
#        k_in = np.loadtxt('test_inputs/proj_2/k_h.txt')*C.h
#        C.k = k_in
#        zs = np.loadtxt('test_inputs/proj_2/z.txt')
#        zs[0] = 10**-3
#
#        ls = np.loadtxt('test_inputs/proj_2/ell.txt')
#
#        f_sky = np.pi/(3.*np.sqrt(2.))
#        params = defaults.lensing_params.copy()
#        params['zbar'] = 1.0
#        params['sigma'] = 0.4
#        params['smodel'] = 'gaussian'
#        params['l_min'] = np.min(ls)
#        params['l_max'] = np.max(ls)
#        params['n_l'] = ls.size
#        params['n_gal'] = 118000000*6.
#        params['pmodel'] = 'cosmosis'
#        sp1 = sp.ShearPower(C,zs,f_sky,params,mode='power')
#
#        sh_pow1 = sp.Cll_sh_sh(sp1).Cll()
#        sh_pow1_gg = sp.Cll_g_g(sp1).Cll()
#        sh_pow1_sg = sp.Cll_sh_g(sp1).Cll()
#        sh_pow1_mm = sp.Cll_mag_mag(sp1).Cll()
#
#        sh_pow_cosm = np.loadtxt('test_inputs/proj_2/ss_pow.txt')
#        gal_pow_cosm = np.loadtxt('test_inputs/proj_2/gg_pow.txt')
#        sg_pow_cosm = np.loadtxt('test_inputs/proj_2/sg_pow.txt')
#        mm_pow_cosm = np.loadtxt('test_inputs/proj_2/mm_pow.txt')
#
#
##            import matplotlib.pyplot as plt
#
##            ax = plt.subplot(111)
##            ax.set_xlabel('l',size=20)
##            ax.set_ylabel('l(l+1)$C^{AB}(2\pi)^{-1}$')
##            ax.loglog(ls,sh_pow1)
##            ax.loglog(ls*C.h,ls*(ls*C.h+1.)*sh_pow1_gg/(2.*np.pi)/C.h**2)
##            ax.loglog(ls*C.h,ls*(ls*C.h+1.)*sh_pow1_sg/(2.*np.pi)/C.h**2)
##            ax.loglog(ls*C.h,ls*(ls*C.h+1.)*sh_pow1_mm/(2.*np.pi)/C.h)
##            ax.loglog(ls,sh_pow_cosm)
##            ax.loglog(ls,ls*(ls+1.)*gal_pow_cosm/(2.*np.pi))
##            ax.loglog(ls,ls*(ls+1.)*sg_pow_cosm/(2.*np.pi))
##            ax.loglog(ls,ls*(ls+1.)*mm_pow_cosm/(2.*np.pi))
##            ax.legend(["ssc_ss","ssc_gg","ssc_sg","ssc_mm","cosm_ss","cosm_gg","cosm_sg","cosm_mm"],loc=2)
#
#        #get ratio of calculated value to expected value from cosmosis
#        #use -np.inf as filler for interpolation when l value is not in ls*C.h,filter it later
#        ss_rat = (sh_pow_cosm-(interp1d(ls,sh_pow1,bounds_error=False,fill_value=-np.inf)(ls)))/sh_pow_cosm
#        gg_rat = (gal_pow_cosm-(interp1d(ls,sh_pow1_gg,bounds_error=False,fill_value=-np.inf)(ls)))/gal_pow_cosm
#        sg_rat = (sg_pow_cosm-(interp1d(ls,sh_pow1_sg,bounds_error=False,fill_value=-np.inf)(ls)))/sg_pow_cosm
#        mm_rat = (mm_pow_cosm-(interp1d(ls,sh_pow1_mm*C.h,bounds_error=False,fill_value=-np.inf)(ls)))/mm_pow_cosm
#        mean_ss_err = np.mean(abs(ss_rat)[abs(ss_rat)<np.inf])
#        mean_gg_err = np.mean(abs(gg_rat)[abs(gg_rat)<np.inf])
#        mean_sg_err = np.mean(abs(sg_rat)[abs(sg_rat)<np.inf])
#        mean_mm_err = np.mean(abs(mm_rat)[abs(mm_rat)<np.inf])
#
#        max_ss_err = max((abs(ss_rat))[abs(ss_rat)<np.inf])
#        max_gg_err = max((abs(gg_rat))[abs(gg_rat)<np.inf])
#        max_sg_err = max((abs(sg_rat))[abs(sg_rat)<np.inf])
#        max_mm_err = max((abs(mm_rat))[abs(mm_rat)<np.inf])
#
#        print("ss agreement within: "+str(max_ss_err*100.)+"%"+" mean agreement: "+str(mean_ss_err*100.)+"%")
#        print("gg agreement within: "+str(max_gg_err*100.)+"%"+" mean agreement: "+str(mean_gg_err*100.)+"%")
#        print("sg agreement within: "+str(max_sg_err*100.)+"%"+" mean agreement: "+str(mean_sg_err*100.)+"%")
#        print("mm agreement within: "+str(max_mm_err*100.)+"%"+" mean agreement: "+str(mean_mm_err*100.)+"%")
#
#        assert max_ss_err<TOLERANCE_MAX
#        assert max_gg_err<TOLERANCE_MAX
#        assert max_sg_err<TOLERANCE_MAX
#        assert max_mm_err<TOLERANCE_MAX
#        assert mean_ss_err<TOLERANCE_MEAN
#        assert mean_gg_err<TOLERANCE_MEAN
#        assert mean_sg_err<TOLERANCE_MEAN
#        assert mean_mm_err<TOLERANCE_MEAN
##            plt.grid()
##            plt.show()

def test_cosmosis_match():
    """test agreement with modified cosmosis demo 15 results
    assuming gaussian matter distribution with sigma=0.4 and average z=1
    use halofit power spectrum grid"""
    TOLERANCE_MAX = 0.2
    TOLERANCE_MEAN = 0.2
    power_params = defaults.power_params.copy()
    power_params.camb['force_sigma8'] = True
    power_params.camb['maxkh'] = 25000
    power_params.camb['kmax'] = 100.
    power_params.camb['npoints'] = 3200
    C = cp.CosmoPie(cosmology=COSMOLOGY_COSMOSIS2.copy(),p_space='jdem')
    P_in = mps.MatterPower(C,power_params)
    k_in = P_in.k
    C.set_power(P_in)
    zs = np.loadtxt('test_inputs/proj_2/z.txt')
    zs[0] = 10**-3

    ls = np.loadtxt('test_inputs/proj_2/ell.txt')
    f_sky = np.pi/(3.*np.sqrt(2.))
    params = defaults.lensing_params.copy()
    params['zbar'] = 1.0
    params['sigma'] = 0.40
    params['smodel'] = 'gaussian'
    params['l_min'] = np.min(ls)
    params['l_max'] = np.max(ls)
    params['n_l'] = ls.size
    params['n_gal'] = 118000000*6.
    params['pmodel'] = 'halofit'

    sh_pow1 = np.loadtxt('test_inputs/proj_2/ss_pow.txt')
    sh_pow1_gg = np.loadtxt('test_inputs/proj_2/gg_pow.txt')
    sh_pow1_sg = np.loadtxt('test_inputs/proj_2/sg_pow.txt')
    sh_pow1_mm = np.loadtxt('test_inputs/proj_2/mm_pow.txt')/C.h

    sp2 = sp.ShearPower(C,zs,f_sky,params,mode='power')



    q_sh = lw.QShear(sp2)
    q_num = lw.QNum(sp2)
    q_mag = lw.QMag(sp2)
    sh_pow2 = sp.Cll_q_q(sp2,q_sh,q_sh).Cll()
    sh_pow2_gg = sp.Cll_q_q(sp2,q_num,q_num).Cll()
    sh_pow2_sg = sp.Cll_q_q(sp2,q_sh,q_num).Cll()
    sh_pow2_mm = sp.Cll_q_q(sp2,q_mag,q_mag).Cll()

    #get ratio of calculated value to expected value from cosmosis
    #use -np.inf as filler for interpolation when l value is not in ls*C.h,filter it later
    ss_rat = (sh_pow2-sh_pow1)/sh_pow2

    gg_rat = (sh_pow2_gg-sh_pow1_gg)/sh_pow2_gg
    sg_rat = (sh_pow2_sg-sh_pow1_sg)/sh_pow2_sg
    mm_rat = (sh_pow2_mm-sh_pow1_mm)/sh_pow2_mm
    print(sh_pow2)
    mean_ss_err = np.mean(abs(ss_rat)[abs(ss_rat)<np.inf])
    mean_gg_err = np.mean(abs(gg_rat)[abs(gg_rat)<np.inf])
    mean_sg_err = np.mean(abs(sg_rat)[abs(sg_rat)<np.inf])
    mean_mm_err = np.mean(abs(mm_rat)[abs(mm_rat)<np.inf])

    max_ss_err = max((abs(ss_rat))[abs(ss_rat)<np.inf])
    max_gg_err = max((abs(gg_rat))[abs(gg_rat)<np.inf])
    max_sg_err = max((abs(sg_rat))[abs(sg_rat)<np.inf])
    max_mm_err = max((abs(mm_rat))[abs(mm_rat)<np.inf])

    print("ss agreement within: "+str(max_ss_err*100.)+"%"+" mean agreement: "+str(mean_ss_err*100.)+"%")
    print("gg agreement within: "+str(max_gg_err*100.)+"%"+" mean agreement: "+str(mean_gg_err*100.)+"%")
    print("sg agreement within: "+str(max_sg_err*100.)+"%"+" mean agreement: "+str(mean_sg_err*100.)+"%")
    print("mm agreement within: "+str(max_mm_err*100.)+"%"+" mean agreement: "+str(mean_mm_err*100.)+"%")

    assert max_ss_err<TOLERANCE_MAX
    assert max_gg_err<TOLERANCE_MAX
    assert max_sg_err<TOLERANCE_MAX
    assert max_mm_err<TOLERANCE_MAX
    assert mean_ss_err<TOLERANCE_MEAN
    assert mean_gg_err<TOLERANCE_MEAN
    assert mean_sg_err<TOLERANCE_MEAN
    assert mean_mm_err<TOLERANCE_MEAN


if __name__=='__main__':
    pytest.cmdline.main(['projected_tests.py'])
