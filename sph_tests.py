import numpy as np 
from geo import rect_geo
import unittest
import sph_klim as sph
import mpmath
from sph_functions import jn_zeros_cut

class TestAlmMathematicaAgree1(unittest.TestCase):
    def test_alm_match1(self):
        AVG_TOLERANCE = 10e-8
        MAX_TOLERANCE = 10e-7
        ZERO_TOLERANCE = 10e-8

        
        test_base = './test_inputs/sph1/'
        alm_math = np.loadtxt(test_base+'alm_mathematica.dat')
        lm_table = np.loadtxt(test_base+'lm_table.dat')
    
        zs=np.array([.1,.2,.3])
        z_fine = np.arange(0.01,0.3,0.001)
        Theta=[np.pi/4,np.pi/2.]
        Phi=[0,np.pi/3.+np.sqrt(2.)/100.]
    
    
        from cosmopie import CosmoPie
        C=CosmoPie()
        geo1 = rect_geo(zs,Theta,Phi,C,z_fine)
    
        alm_py = np.zeros_like(alm_math)
        for i in range(0,lm_table.shape[0]):
            alm_py[i] = sph.a_lm(geo1,lm_table[i,0],lm_table[i,1])
       
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
        self.assertTrue(avg_diff<AVG_TOLERANCE)
        self.assertTrue(max_diff<MAX_TOLERANCE)
        self.assertTrue(zero_max_diff<ZERO_TOLERANCE)

#    def test_alm_match2(self):
#        AVG_TOLERANCE = 10e-8
#        MAX_TOLERANCE = 10e-7
#        ZERO_TOLERANCE = 10e-8
#
#        
#        test_base = './test_inputs/sph1/'
#        alm_math = np.loadtxt(test_base+'alm_mathematica2.dat')
#        lm_table = np.loadtxt(test_base+'lm_table2.dat')
#    
#        zs=np.array([.1,.2,.3])
#        Theta=[np.pi/4,np.pi/2.]
#        Phi=[0,np.pi/3.+np.sqrt(2.)/100.]
#    
#    
#        from cosmopie import CosmoPie
#        cp=CosmoPie()
#        geo1 = rect_geo(zs,Theta,Phi,cp)
#    
#        alm_py = np.zeros_like(alm_math)
#        for i in range(0,lm_table.shape[0]):
#            alm_py[i] = sph.a_lm(geo1,lm_table[i,0],lm_table[i,1])
#       
#        alm_math1 = alm_math[alm_math>0]
#        alm_py1 = alm_py[alm_math>0]
#        avg_diff = np.average(abs(alm_math1-alm_py1)/alm_math1)
#        max_diff = np.max(abs(alm_math1-alm_py1)/alm_math1)
#            
#        if any(alm_math==0):
#            zero_max_diff = np.max(alm_py1[alm_math==0.])
#        else:
#            zero_max_diff = 0.
#        print "n nonzero",alm_math1.size
#        print "avg,max diff:",avg_diff,max_diff
#        self.assertTrue(avg_diff<AVG_TOLERANCE)
#        self.assertTrue(max_diff<MAX_TOLERANCE)
#        self.assertTrue(zero_max_diff<ZERO_TOLERANCE)
#class TestRIntMatchesExact1(unittest.TestCase):
    def test_rint_match(self):
        ZERO_TOLERANCE = 10e-14
        MAX_TOLERANCE = 10e-14
        AVG_TOLERANCE = 10e-15
        ls = np.arange(0,2)
        k_cut = 0.01
        r_max = 1945.65
    
        r2 = 1241.
        r1 = 842.
        diff_tot = 0.
        diff_max = 0.
        zero_diff = 0.
        n_count = 0
        z_count = 0
        for ll in ls:       
            ks = jn_zeros_cut(ll,k_cut*r_max)/r_max
            for i in range(0,ks.size):
                r_int_exact = float(1./((3.+ll)*mpmath.gamma(1.5+ll))/2.**(ll+1)*ks[i]**ll*np.sqrt(np.pi)*(mpmath.hyp1f2(1.5+ll/2.,2.5+ll/2.,1.5+ll,-1./4*ks[i]**2*r2**2)*r2**(3+ll)-r1**(3+ll)*mpmath.hyp1f2(1.5+ll/2.,2.5+ll/2.,1.5+ll,-1./4.*ks[i]**2*r1**2)))
                r_int_compute = sph.R_int([r1,r2],ks[i],ll)
                if np.abs(r_int_exact)>0.:
                    diff = np.abs(r_int_compute-r_int_exact)/r_int_exact
                    n_count+=1
                    diff_tot+=diff
                    diff_max = max(diff_max,diff)
                else:
                    zero_diff+=r_int_compute
                    z_count+=1
        print "zero diff, n zero",zero_diff,z_count
        print "max diff, avg diff n",diff_max,diff_tot/n_count,n_count
        if z_count>0:
            self.assertTrue(ZERO_TOLERANCE>zero_diff/z_count)
        self.assertTrue(MAX_TOLERANCE>diff_max)
        self.assertTrue(AVG_TOLERANCE>diff_tot/n_count)

if __name__=='__main__':
    unittest.main()
