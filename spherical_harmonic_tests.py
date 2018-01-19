"""tests of spherical harmonic utilities"""
#pylint: disable=W0621
#TODO write test for reconstruction when ground truth known
import numpy as np
import pytest
import sph_functions as sph
import ylm_utils as ylmu
import ylm_utils_mpmath as ylmu_mp
from polygon_pixel_geo import get_healpix_pixelation

ATOL_USE = 1e-10
RTOL_USE = 1e-13


class ThetaPhiList(object):
    """get lists of thetas and phis"""
    def __init__(self,key):
        """key: and index"""
        self.key = key
        if key==1:
            n_side = 8
            theta_need=np.linspace(0.,np.pi,n_side)
            phi_need=np.linspace(0.,2.*np.pi,n_side)
            self.thetas = np.zeros(n_side**2)
            self.phis = np.zeros(n_side**2)
            self.l_max = 20
            for i in xrange(0,n_side):
                for j in xrange(0,n_side):
                    self.thetas[n_side*i+j] = theta_need[j]
                    self.phis[n_side*i+j] = phi_need[i]
        elif key==2:
            self.thetas = np.array([np.pi/2.])
            self.phis = np.array([0.])
            self.l_max = 85
        elif key==3:
            pixels = get_healpix_pixelation(res_choose=4)
            self.thetas = pixels[:,0]
            self.phis = pixels[:,1]
            self.l_max = 20
        else:
            raise ValueError("unrecognized key: "+str(key))

@pytest.fixture(params=[1,2,3])
def theta_phis(request):
    """iterate over ThetaPhiList objects"""
    return ThetaPhiList(request.param)

class ALMList(object):
    """test spherical decompositions"""
    def __init__(self,key):
        self.key=key
        if key==1:
            self.l_max = 10
            self.lm_dict,self.ls,self.ms = ylmu.get_lm_dict(self.l_max)
            self.alm_dict = {}
            for d_key in self.lm_dict:
                self.alm_dict[d_key] = 0.5
        elif key==2:
            self.l_max = 10
            self.lm_dict,self.ls,self.ms = ylmu.get_lm_dict(self.l_max)
            self.alm_dict = {}
            for d_key in self.lm_dict:
                self.alm_dict[d_key] = np.random.uniform(-1.,1.)
        else:
            raise ValueError("unrecognized key: "+str(key))

@pytest.fixture(params=[1,2])
def alms(request):
    """iterate over ALMList possibilities"""
    return ALMList(request.param)

def test_alm_recontruct(alms,theta_phis):
    """test that reconstruct_from_alm matches with and without mpmath"""
    if theta_phis.key!=2:
        alm_dict = alms.alm_dict.copy()
        thetas = theta_phis.thetas.copy()
        phis = theta_phis.phis.copy()
        l_max = alms.l_max
        alm1 = ylmu.reconstruct_from_alm(l_max,thetas,phis,alm_dict)
        alm2 = ylmu_mp.reconstruct_from_alm(l_max,thetas,phis,alm_dict)
        assert np.allclose(alm1,alm2,atol=ATOL_USE,rtol=RTOL_USE)

def test_Y_R_agreement_table_direct(theta_phis):
    """test that Y_r matches ylm_utils table and from complex harmonics"""
    thetas = theta_phis.thetas.copy()
    phis = theta_phis.phis.copy()
    l_max = theta_phis.l_max
    Y_r_1s,ls,ms = ylmu.get_Y_r_table(l_max,thetas,phis)
    Y_r_2s = np.zeros_like(Y_r_1s)
    for itr in xrange(0,ls.size):
        Y_r_2s[itr] = sph.Y_r(ls[itr],ms[itr],thetas,phis)
    assert np.allclose(Y_r_1s,Y_r_2s,atol=ATOL_USE,rtol=RTOL_USE)

def test_Y_R_agreement_mp_table_direct(theta_phis):
    """test that Y_r matches ylm_utils_mpmath table and from complex harmonics"""
    thetas = theta_phis.thetas.copy()
    phis = theta_phis.phis.copy()
    l_max = theta_phis.l_max
    Y_r_1s,ls,ms = ylmu_mp.get_Y_r_table(l_max,thetas,phis)
    Y_r_2s = np.zeros_like(Y_r_1s)
    for itr in xrange(0,ls.size):
        Y_r_2s[itr] = sph.Y_r(ls[itr],ms[itr],thetas,phis)
    assert np.allclose(Y_r_1s,Y_r_2s,atol=ATOL_USE,rtol=RTOL_USE)

def test_Y_R_agreement_mp_table_table(theta_phis):
    """test that Y_r matches ylm_utils table and ylm_utils_mpmath table"""
    thetas = theta_phis.thetas.copy()
    phis = theta_phis.phis.copy()
    l_max = theta_phis.l_max
    Y_r_1s,ls,ms = ylmu.get_Y_r_table(l_max,thetas,phis)
    Y_r_2s,l2s,m2s = ylmu_mp.get_Y_r_table(l_max,thetas,phis)
    assert np.allclose(Y_r_1s,Y_r_2s,atol=ATOL_USE,rtol=RTOL_USE)
    assert np.all(ls==l2s)
    assert np.all(ms==m2s)

def test_Y_R_agreement_dict_central():
    """check agreement between exact and approx solution at center standard precision"""
    l_max = 85
    Y_r_1_dict = ylmu.get_Y_r_dict(l_max,np.zeros(1)+np.pi/2.,np.zeros(1))
    Y_r_2_dict = ylmu.get_Y_r_dict_central(l_max)
    keys1 = sorted(Y_r_1_dict.keys())
    keys2 = sorted(Y_r_2_dict.keys())
    dict_test,_,_ = ylmu.get_lm_dict(l_max)
    keys_test = sorted(dict_test.keys())
    assert keys1==keys2
    assert keys1==keys_test
    n_val = len(keys1)
    assert n_val==(l_max+1)**2
    Y_r_1s = np.zeros(n_val)
    Y_r_2s = np.zeros(n_val)

    for itr in xrange(0,n_val):
        Y_r_1s[itr] = Y_r_1_dict[keys1[itr]]
        Y_r_2s[itr] = Y_r_2_dict[keys2[itr]]

    assert np.allclose(Y_r_1s,Y_r_2s,atol=ATOL_USE,rtol=RTOL_USE)

def test_Y_R_agreement_mp_central_central():
    """check agreement between center arbitrary precision and center standard precision"""
    l_max = 85
    Y_r_1_dict = ylmu.get_Y_r_dict_central(l_max)
    Y_r_2_dict = ylmu_mp.get_Y_r_dict_central(l_max)
    keys1 = sorted(Y_r_1_dict.keys())
    keys2 = sorted(Y_r_2_dict.keys())
    dict_test,_,_ = ylmu_mp.get_lm_dict(l_max)
    keys_test = sorted(dict_test.keys())
    assert keys1==keys2
    assert keys1==keys_test
    n_val = len(keys1)
    assert n_val==(l_max+1)**2
    Y_r_1s = np.zeros(n_val)
    Y_r_2s = np.zeros(n_val)

    for itr in xrange(0,n_val):
        Y_r_1s[itr] = Y_r_1_dict[keys1[itr]]
        Y_r_2s[itr] = Y_r_2_dict[keys2[itr]]

    assert np.allclose(Y_r_1s,Y_r_2s,atol=ATOL_USE,rtol=RTOL_USE)

def test_Y_R_agreement_mp_dict_central():
    """check agreement between exact and approx solution at center arbitrary precision"""
    l_max = 100
    Y_r_1_dict = ylmu_mp.get_Y_r_dict(l_max,np.zeros(1)+np.pi/2.,np.zeros(1))
    Y_r_2_dict = ylmu_mp.get_Y_r_dict_central(l_max)
    keys1 = sorted(Y_r_1_dict.keys())
    keys2 = sorted(Y_r_2_dict.keys())
    dict_test,_,_ = ylmu_mp.get_lm_dict(l_max)
    keys_test = sorted(dict_test.keys())
    assert keys1==keys2
    assert keys1==keys_test
    n_val = len(keys1)
    assert n_val==(l_max+1)**2
    Y_r_1s = np.zeros(n_val)
    Y_r_2s = np.zeros(n_val)

    for itr in xrange(0,n_val):
        Y_r_1s[itr] = Y_r_1_dict[keys1[itr]]
        Y_r_2s[itr] = Y_r_2_dict[keys2[itr]]

    assert np.allclose(Y_r_1s,Y_r_2s,atol=ATOL_USE,rtol=RTOL_USE)

if __name__=='__main__':
    pytest.cmdline.main(['spherical_harmonic_tests.py'])
