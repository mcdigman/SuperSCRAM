"""test some exact results for sph_klim, default x_grid_size is sufficiently converged to l tested if succeeds"""

#Test numeric integral compared to mpmath arbitrary precision
#Both the numeric and mpmath results should agree; they are both trustworthy, although the numerical method is still potentially more robust
import numpy as np
import pytest
import mpmath

from geo import RectGeo
import sph_klim as sph
from sph_functions import jn_zeros_cut
from cosmopie import CosmoPie
import defaults
import matter_power_spectrum as mps

def test_alm_match1():
    """Test that alm matches an exact answer retrieved from mathematica"""
    AVG_TOLERANCE = 10e-8
    MAX_TOLERANCE = 10e-7
    ZERO_TOLERANCE = 10e-8


    test_base = './test_inputs/sph1/'
    alm_math = np.loadtxt(test_base+'alm_mathematica.dat')
    lm_table = np.loadtxt(test_base+'lm_table.dat')

    zs = np.array([.1,.2,.3])
    z_fine = np.arange(0.01,0.3,0.001)
    Theta = [np.pi/4,np.pi/2.]
    Phi = [0,np.pi/3.+np.sqrt(2.)/100.]


    C = CosmoPie(cosmology=defaults.cosmology,p_space='jdem')
    geo1 = RectGeo(zs,Theta,Phi,C,z_fine)

    alm_py = np.zeros_like(alm_math)
    for i in xrange(0,lm_table.shape[0]):
        alm_py[i] = geo1.a_lm(lm_table[i,0],lm_table[i,1])# sph.a_lm(geo1,lm_table[i,0],lm_table[i,1])

    alm_math1 = alm_math[alm_math>0]
    alm_py1 = alm_py[alm_math>0]
    avg_diff = np.average(abs(alm_math1-alm_py1)/alm_math1)
    max_diff = np.max(abs(alm_math1-alm_py1)/alm_math1)

    if any(alm_math==0):
        zero_max_diff = np.max(alm_py1[alm_math==0.])
    else:
        zero_max_diff = 0.
    print "n nonzero",alm_math1.size
    print "avg,max diff:",avg_diff,max_diff
    assert avg_diff<AVG_TOLERANCE
    assert max_diff<MAX_TOLERANCE
    assert zero_max_diff<ZERO_TOLERANCE

def test_rint_match_mpmath():
    """
    Test numeric integral compared to mpmath arbitrary precision
    Both the numeric and mpmath results should agree
    scipy's hypergeometric functions would fail this test because of a catastrophic loss of precision,
    mpmath works but to avoid mpmath dependency we just use numerical method
    """
    print "test_rint_match_mpmath: begin testing numeric r integral agreement with exact (arbitrary precision) mpmath solution"
    ZERO_TOLERANCE = 10e-13
    MAX_TOLERANCE = 10e-11
    AVG_TOLERANCE = 10e-13
    ls = np.arange(0,70)
    k_cut = 0.015
    r_max = 4000

    r1 = 100
    r2 = 3000

    r1_m = mpmath.mpf(r1)
    r2_m = mpmath.mpf(r2)

    mpmath.dps = 200
    diff_tot = 0.
    diff_max = 0.
    zero_diff = 0.
    diff_tot2 = 0.
    diff_max2 = 0.
    zero_diff2 = 0.
    n_count = 0
    z_count = 0
    bad_count = 0
    C = CosmoPie(defaults.cosmology,p_space='jdem')
    P_lin = mps.MatterPower(C,defaults.power_params.copy())
    C.set_power(P_lin)
    basis = sph.SphBasisK(r_max,C,k_cut,defaults.basis_params)
    for ll in ls:
        ks = jn_zeros_cut(ll,k_cut*r_max)/r_max
        ll_m = mpmath.mpf(ll)
        for i in xrange(0,ks.size):
            kk_m = mpmath.mpf(ks[i])

            r_int_exact = float(1./((3.+ll_m)*mpmath.gamma(1.5+ll_m))/2.**(ll_m+1.)*kk_m**ll_m*mpmath.sqrt(mpmath.pi)*(mpmath.hyp1f2(1.5+ll_m/2.,2.5+ll_m/2.,1.5+ll_m,-1./4*kk_m**2*r2_m**2)*r2_m**(3+ll_m)-r1_m**(3+ll_m)*mpmath.hyp1f2(1.5+ll_m/2.,2.5+ll_m/2.,1.5+ll_m,-1./4.*kk_m**2*r1_m**2)))
            r_int_compute = sph.R_int([r1,r2],ks[i],ll)
            r_int_compute2 = (basis.rints[ll](r2*ks[i])-basis.rints[ll](r1*ks[i]))/ks[i]**3
            if np.abs(r_int_exact)>0.:
                diff = np.abs(r_int_compute-r_int_exact)/r_int_exact
                diff2 = np.abs(r_int_compute2-r_int_exact)/r_int_exact
                n_count+=1
                diff_tot+=diff
                diff_tot2+=diff2
                if diff>MAX_TOLERANCE:
                    #print "test_rint_match_mpmath: error outside tolerance at l,k,numeric,mpmath: ",ll,ks[i],r_int_compute,r_int_exact
                    bad_count+=1
                diff_max = max(diff_max,diff)
                diff_max2 = max(diff_max2,diff2)
            else:
                zero_diff+=r_int_compute
                zero_diff2+=r_int_compute2
                z_count+=1
    print "test_rint_match_mpmath: zero diff, n zero",zero_diff,z_count
    print "test_rint_match_mpmath: max diff, avg diff,n",diff_max,diff_tot/n_count,n_count
    print "test_rint_match_mpmath: max diff tolerance, avg diff tolerance",MAX_TOLERANCE,AVG_TOLERANCE
    print "test_rint_match_mpmath: n values outside tolerance",bad_count
    if z_count>0:
        assert ZERO_TOLERANCE>zero_diff/z_count
        assert ZERO_TOLERANCE>zero_diff2/z_count
    assert MAX_TOLERANCE>diff_max
    assert MAX_TOLERANCE>diff_max2
    assert AVG_TOLERANCE>diff_tot/n_count
    assert AVG_TOLERANCE>diff_tot2/n_count
    print "test_rint_match_mpmath: finished testing numeric r integral agreement with exact (arbitrary precision) mpmath solution"

#Test scipy exact solution vs numeric integration
#This test fails because of scipy loss of precision: the numeric integration is more accurate
#    def test_rint_match_scipy(self):
#        print "test_rint_match_scipy: begin testing numeric r integral agreement with exact (floating point precision) scipy solution"
#        ZERO_TOLERANCE = 10e-13
#        MAX_TOLERANCE = 10e-2
#        AVG_TOLERANCE = 10e-15
#        ls = np.arange(0,70)
#        k_cut = 0.02
#        r_max = 4000
#
#        r2 = 3000.
#        r1 = 100.
#        diff_tot = 0.
#        diff_max = 0.
#        zero_diff = 0.
#        n_count = 0
#        z_count = 0
#        bad_count = 0
#        for ll in ls:
#            ks = jn_zeros_cut(ll,k_cut*r_max)/r_max
#            for i in xrange(0,ks.size):
#                r_int_exact = 1./((3.+ll)*sp.gamma(1.5+ll))/2.**(ll+1)*ks[i]**ll*np.sqrt(np.pi)*(sp.hyp1f2(1.5+ll/2.,2.5+ll/2.,1.5+ll,-1./4*ks[i]**2*r2**2)[0]*r2**(3+ll)-r1**(3+ll)*sp.hyp1f2(1.5+ll/2.,2.5+ll/2.,1.5+ll,-1./4.*ks[i]**2*r1**2)[0])
#                r_int_compute = sph.R_int([r1,r2],ks[i],ll)
#                if np.abs(r_int_exact)>0.:
#                    diff = np.abs(r_int_compute-r_int_exact)/r_int_exact
#                    n_count+=1
#                    diff_tot+=diff
#                    if diff>MAX_TOLERANCE:
#                        f1 = sp.hyp1f2(1.5+ll/2.,2.5+ll/2.,1.5+ll,-1./4*ks[i]**2*r2**2)
#                        f2 = sp.hyp1f2(1.5+ll/2.,2.5+ll/2.,1.5+ll,-1./4.*ks[i]**2*r1**2)
#                        print "test_rint_match_scipy: err>TOL: l,k,numeric,scipy,hyp1,hyp2: ",ll,ks[i],r_int_compute,r_int_exact,f1,f2
#                        bad_count+=1
#                    diff_max = max(diff_max,diff)
#                else:
#                    zero_diff+=r_int_compute
#                    z_count+=1
#        print "test_rint_match_scipy: zero diff, n zero",zero_diff,z_count
#        print "test_rint_match_scipy: max diff, avg diff,n",diff_max,diff_tot/n_count,n_count
#        print "test_rint_match_scipy: max diff tolerance, avg diff tolerance",MAX_TOLERANCE,AVG_TOLERANCE
#        print "test_rint_match_scipy: n values outside tolerance",bad_count
#        if z_count>0:
#            assert ZERO_TOLERANCE>zero_diff/z_count
#        assert MAX_TOLERANCE>diff_max
#        assert AVG_TOLERANCE>diff_tot/n_count
#        print "test_rint_match_scipy: finished testing numeric r integral agreement with exact (floating point precision) scipy solution"

if __name__=='__main__':
#    unittest.main()
    pytest.cmdline.main(['sph_tests.py'])
