"""tests for the algebra_utils module"""
#pylint: disable=W0621,duplicate-code
import warnings
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import pytest
import algebra_utils as au
from algebra_utils import cholesky_inplace,get_inv_cholesky,ch_inv,invert_triangular,get_mat_from_inv_cholesky,cholesky_inv_contract,get_cholesky_inv,cholesky_contract,check_is_cholesky,check_is_cholesky_inv,trapz2
au.DEBUG = True
#TODO add testing for inplace and clean parameters
def get_test_mat(key):
    """get a test mat for the numerical key key"""
    if key==1:
        return np.array([[10.,14.],[14.,20.]],order='F')
    elif key==2:
        return np.array([[5.,-1.],[-1.,5.]],order='F')
    elif key==3:
        return np.array([[5.,0],[0,5.]],order='F')
    elif key==4:
        return np.array([[5.,0],[0,4.]],order='F')
    elif key==5:
        return np.array([[4.,0,0],[0,5.,1.],[0,1.,5.]],order='F')
    elif key==6:
        return np.array([[4,0,0],[0,5,1],[0,1,5]],order='F',dtype=np.int_)
    elif key==7:
        return np.array([[4.]],order='F',dtype=np.float_)
    elif key==8:
        nside = 10
        result = np.random.rand(nside,nside)
        result = np.dot(result.T,result).copy('F')+np.diag(np.random.rand(nside,1))
        return result
    elif key==9:
        nside = 1000
        result = np.random.rand(nside,nside)
        result = np.dot(result.T,result).copy('F')
        return result
    elif key==10:
        nside = 1000
        result = np.random.rand(nside,nside)
        result = np.dot(result.T,result).copy('F')
        return result
    else:
        raise ValueError('unrecognized input key:'+str(key))

#choose whether to include some random tests
test_list = [1,2,3,4,5,7,8,8,8,8,8]
#test_list = [1,2,3,4,5,7]


class BasicSetupMatrices(object):
    """holds basic matrices and needed variations"""
    def __init__(self,A):
        """A: the original matrix"""
        self.A = A
        self.chol_i = npl.pinv(npl.cholesky(A))
        self.chol_i_u = npl.pinv(spl.cholesky(A,lower=False))
        self.chol = npl.cholesky(A)
        self.chol_u = spl.cholesky(A,lower=False)
        self.A_i = npl.pinv(A)


@pytest.fixture(params=test_list)
def test_mat(request):
    """iterate through requested tests"""
    A = get_test_mat(request.param)
    return BasicSetupMatrices(A)

class TrapzTester(object):
    """holds arrays to apply trapz to"""
    def __init__(self,key):
        """key: select which test set to use"""
        self.key = key
        if key == 1:
            self.xs = np.arange(0,4)
            self.integrand = np.random.rand(4)
        elif key ==2:
            self.xs = np.arange(0,6)
            self.integrand = np.random.rand(4,6).T
        elif key ==3:
            self.xs = np.arange(0,10)**2
            self.integrand = np.random.rand(15,10).T
        elif key ==4:
            self.xs = np.arange(0,1000)**2
            self.integrand = np.random.rand(200,1000).T
        elif key == 5:
            self.xs = np.arange(0,800)
            self.integrand = np.arange(0,800)
        elif key == 6:
            self.xs = np.arange(0,1300)
            self.integrand = np.outer(np.arange(1,17),np.arange(0,1300)).T
        else:
            raise ValueError('unrecognized key ',key)

trapz_test_list = [1,2,3,4,5,6]

@pytest.fixture(params=trapz_test_list)
def trapz_test(request):
    """iterate through trapz tests"""
    return TrapzTester(request.param)

relax_rtol = 1e-01
relax_atol = 1e-03
tighten_atol = 1e-9
atol_rel_use = 1e-7
rtol_use = 1e-6

    #inv loses precision so have to relax tolerance dramatically for tests involving direct inverse
    #this shows the superiority of using the cholesky inverse
    #note this loss of precision could cause some random matrices to fail while others pass
    #TODO: better precision testing of code could improve battery
def test_basic_setup_succeeded(test_mat):
    """test self consistency of setup"""
    chol_i1 = test_mat.chol_i.copy()
    chol1 = test_mat.chol.copy()
    chol_i1_u = test_mat.chol_i_u.copy()
    chol1_u = test_mat.chol_u.copy()
    A = test_mat.A.copy()
    A_i = test_mat.A_i.copy()

    assert check_is_cholesky_inv(chol_i1,A,atol_rel=atol_rel_use,rtol=rtol_use,lower=True)
    assert check_is_cholesky(chol1,A,lower=True,atol_rel=atol_rel_use,rtol=rtol_use)
    assert check_is_cholesky_inv(chol_i1_u,A,atol_rel=atol_rel_use,rtol=rtol_use,lower=False)
    assert check_is_cholesky(chol1_u,A,lower=False,atol_rel=atol_rel_use,rtol=rtol_use)
    assert np.allclose(npl.pinv(A_i),A,atol=relax_atol,rtol=relax_rtol)
    assert np.allclose(npl.pinv(chol_i1),chol1,atol=relax_atol,rtol=relax_rtol)

def test_get_inv_cholesky_F_order_direct_lower(test_mat):
    """check get_inv_cholesky works with Fortran ordering, lower given"""
    A = test_mat.A.copy()
    chol_i1 = test_mat.chol_i.copy()
    #Test directly with fortran ordering
    test_A = A.copy('F')
    test_chol_i1 = get_inv_cholesky(test_A,lower=True)


    atol_loc1 = np.max(np.abs(A))*atol_rel_use
    atol_loc4 = np.max(np.abs(chol_i1))*atol_rel_use

    assert np.allclose(test_A,A,atol=atol_loc1,rtol=rtol_use)
    assert np.allclose(test_chol_i1,chol_i1,atol=atol_loc4,rtol=rtol_use)
    assert check_is_cholesky_inv(test_chol_i1,A,atol_rel=atol_rel_use,rtol=rtol_use,lower=True)

def test_get_inv_cholesky_F_order_inverse_lower(test_mat):
    """check get_inv_cholesky works with Fortran ordering, lower given"""
    A_i = test_mat.A_i.copy()
    #check L*L.T=A^-1 => (L.T^-1)*L^-1=A=B*B.T note significant loss of precision here Fortran ordering
    test_A_i = A_i.copy('F')
    test_chol2 = get_inv_cholesky(test_A_i,lower=True)

    atol_loc2 = np.max(np.abs(A_i))*atol_rel_use

    assert np.allclose(test_A_i,A_i,atol=atol_loc2,rtol=rtol_use)
    assert check_is_cholesky_inv(test_chol2,A_i,atol_rel=atol_rel_use,rtol=rtol_use,lower=True)
    #print test_chol2.T
    #print chol1
    #assert np.allclose(test_chol2.T,chol1)

def test_get_inv_cholesky_C_order_direct_lower(test_mat):
    """check get_inv_cholesky works with C ordering, lower given"""
    A = test_mat.A.copy()
    chol_i1 = test_mat.chol_i.copy()
    #Test directly with C ordering
    test_A = A.copy('C')
    test_chol_i3 = get_inv_cholesky(test_A,lower=True)

    atol_loc1 = np.max(np.abs(A))*atol_rel_use
    atol_loc4 = np.max(np.abs(chol_i1))*atol_rel_use

    assert np.allclose(test_A,A,atol=atol_loc1,rtol=rtol_use)
    assert np.allclose(test_chol_i3,chol_i1,atol=atol_loc4,rtol=rtol_use)
    assert check_is_cholesky_inv(test_chol_i3,A,atol_rel=tighten_atol,rtol=rtol_use,lower=True)

def test_get_inv_cholesky_C_order_inverse_lower(test_mat):
    """check get_inv_cholesky works with C ordering, lower given"""
    A_i = test_mat.A_i.copy()
    #check L*L.T=A^-1 => (L.T^-1)*L^-1=A=B*B.T note significant loss of precision here C ordering
    test_A_i = A_i.copy('C')
    test_chol4 = get_inv_cholesky(test_A_i,lower=True)

    atol_loc2 = np.max(np.abs(A_i))*atol_rel_use

    assert np.allclose(test_A_i,A_i,atol=atol_loc2,rtol=rtol_use)
    assert check_is_cholesky_inv(test_chol4,A_i,atol_rel=atol_rel_use,rtol=rtol_use,lower=True)


def test_get_inv_cholesky_F_order_direct_upper(test_mat):
    """check get_inv_cholesky works with F ordering, upper given"""
    A = test_mat.A.copy()
    chol_i2 = test_mat.chol_i_u.copy()
    #Test directly with fortran ordering
    test_A = A.copy('F')
    test_chol_i2 = get_inv_cholesky(test_A,lower=False)

    atol_loc1 = np.max(np.abs(A))*atol_rel_use
    atol_loc4 = np.max(np.abs(chol_i2))*atol_rel_use

    assert np.allclose(test_A,A,atol=atol_loc1,rtol=rtol_use)
    assert np.allclose(test_chol_i2,chol_i2,atol=atol_loc4,rtol=rtol_use)
    assert check_is_cholesky_inv(test_chol_i2,A,atol_rel=atol_rel_use,rtol=rtol_use,lower=False)


def test_get_inv_cholesky_F_order_inverse_upper(test_mat):
    """check get_inv_cholesky works with F ordering, upper given"""
    A_i = test_mat.A_i.copy()
    #check L*L.T=A^-1 => (L.T^-1)*L^-1=A=B*B.T note significant loss of precision here Fortran ordering
    test_A_i = A_i.copy('F')
    test_chol2 = get_inv_cholesky(test_A_i,lower=False)

    atol_loc2 = np.max(np.abs(A_i))*atol_rel_use

    assert np.allclose(test_A_i,A_i,atol=atol_loc2,rtol=rtol_use)
    assert check_is_cholesky_inv(test_chol2,A_i,atol_rel=atol_rel_use,rtol=rtol_use,lower=False)
    #print test_chol2.T


def test_get_inv_cholesky_C_order_direct_upper(test_mat):
    """check get_inv_cholesky works with C ordering, upper given"""
    A = test_mat.A.copy()
    #Test directly with C ordering
    test_A = A.copy('C')
    test_chol_i1 = get_inv_cholesky(test_A,lower=False)

    atol_loc1 = np.max(np.abs(A))*atol_rel_use

    assert np.allclose(test_A,A,atol=atol_loc1,rtol=rtol_use)
    assert check_is_cholesky_inv(test_chol_i1,A,atol_rel=tighten_atol,rtol=rtol_use,lower=False)


def test_get_inv_cholesky_C_order_inverse_upper(test_mat):
    """check get_inv_cholesky works with C ordering, upper given"""
    A_i = test_mat.A_i.copy()
    #check L*L.T=A^-1 => (L.T^-1)*L^-1=A=B*B.T note significant loss of precision here C ordering
    test_A_i = A_i.copy('C')
    test_chol2 = get_inv_cholesky(test_A_i,lower=False)

    atol_loc2 = np.max(np.abs(A_i))*atol_rel_use

    assert np.allclose(test_A_i,A_i,atol=atol_loc2,rtol=rtol_use)
    assert check_is_cholesky_inv(test_chol2,A_i,atol_rel=atol_rel_use,rtol=rtol_use,lower=False)
    #print test_chol2.T

def test_ch_inv_chol_given_lower(test_mat):
    """test ch_inv works with lower cholesky"""
    A_i = test_mat.A_i.copy()
    chol_i = test_mat.chol_i.copy()

    test_chol_i = chol_i.copy()

    atol_loc2 = np.max(np.abs(A_i))*atol_rel_use

    assert np.allclose(ch_inv(test_chol_i,cholesky_given=True,lower=True),A_i,atol=atol_loc2,rtol=rtol_use)

def test_ch_inv_chol_given_upper(test_mat):
    """test ch_inv works with upper cholesky"""
    A_i = test_mat.A_i.copy()
    chol = test_mat.chol_u.copy()
    chol_i = npl.pinv(chol)

    test_chol_i = chol_i.copy()

    atol_loc2 = np.max(np.abs(A_i))*atol_rel_use

    assert np.allclose(ch_inv(test_chol_i,cholesky_given=True,lower=False),A_i,atol=atol_loc2,rtol=rtol_use)

def test_ch_inv_not_chol_given_lower(test_mat):
    """test ch_inv works with lower cholesky"""
    A = test_mat.A.copy()
    A_i = test_mat.A_i.copy()

    atol_loc1 = np.max(np.abs(A))*atol_rel_use
    atol_loc2 = np.max(np.abs(A_i))*atol_rel_use

    assert np.allclose(ch_inv(A,cholesky_given=False,lower=True),A_i,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(ch_inv(A_i,cholesky_given=False,lower=True),A,atol=atol_loc1,rtol=rtol_use)

def test_ch_inv_not_chol_given_upper(test_mat):
    """test ch_inv works with upper cholesky"""
    A = test_mat.A.copy()
    A_i = test_mat.A_i.copy()

    atol_loc1 = np.max(np.abs(A))*atol_rel_use
    atol_loc2 = np.max(np.abs(A_i))*atol_rel_use

    assert np.allclose(ch_inv(A,cholesky_given=False,lower=False),A_i,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(ch_inv(A_i,cholesky_given=False,lower=False),A,atol=atol_loc1,rtol=rtol_use)

def test_ch_inv_both_F_order(test_mat):
    """test ch_inv works with F ordering"""
    A = test_mat.A.copy()
    A_i = test_mat.A_i.copy()

    test_A1 = A.copy('F')

    atol_loc1 = np.max(np.abs(A))*atol_rel_use
    atol_loc2 = np.max(np.abs(A_i))*atol_rel_use

    assert np.allclose(ch_inv(test_A1,cholesky_given=False),A_i,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(A,test_A1,atol=atol_loc1,rtol=rtol_use)
    assert np.allclose(ch_inv(ch_inv(test_A1,cholesky_given=False),cholesky_given=False),A,atol=atol_loc1,rtol=rtol_use)
    assert np.allclose(ch_inv(A_i,cholesky_given=False),A,atol=atol_loc1,rtol=rtol_use)

def test_ch_inv_both_C_order(test_mat):
    """test ch_inv works with C ordering"""
    A = test_mat.A.copy()
    A_i = test_mat.A_i.copy()

    test_A1 = A.copy('C')

    atol_loc1 = np.max(np.abs(A))*atol_rel_use
    atol_loc2 = np.max(np.abs(A_i))*atol_rel_use

    assert np.allclose(ch_inv(test_A1,cholesky_given=False),A_i,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(A,test_A1,atol=atol_loc1,rtol=rtol_use)
    assert np.allclose(ch_inv(ch_inv(test_A1,cholesky_given=False),cholesky_given=False),A,atol=atol_loc1,rtol=rtol_use)
    assert np.allclose(ch_inv(A_i,cholesky_given=False),A,atol=atol_loc1,rtol=rtol_use)


def test_ch_inv_F_then_C_order(test_mat):
    """test ch_inv works switching orderings"""
    A = test_mat.A.copy()
    A_i = test_mat.A_i.copy()

    test_A1 = A.copy('F')

    atol_loc1 = np.max(np.abs(A))*atol_rel_use
    atol_loc2 = np.max(np.abs(A_i))*atol_rel_use

    assert np.allclose(ch_inv(test_A1,cholesky_given=False),A_i,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(A,test_A1,atol=atol_loc1,rtol=rtol_use)
    assert np.allclose(ch_inv(ch_inv(test_A1,cholesky_given=False),cholesky_given=False),A,atol=atol_loc1,rtol=rtol_use)
    assert np.allclose(ch_inv(A_i,cholesky_given=False),A,atol=atol_loc1,rtol=rtol_use)


def test_ch_inv_C_then_F_order(test_mat):
    """test ch_inv works switching orderings"""
    A = test_mat.A.copy()
    A_i = test_mat.A_i.copy()

    test_A1 = A.copy('C')

    atol_loc1 = np.max(np.abs(A))*atol_rel_use
    atol_loc2 = np.max(np.abs(A_i))*atol_rel_use

    assert np.allclose(ch_inv(test_A1,cholesky_given=False),A_i,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(A,test_A1,atol=atol_loc1,rtol=rtol_use)
    assert np.allclose(ch_inv(ch_inv(test_A1,cholesky_given=False),cholesky_given=False),A,atol=atol_loc1,rtol=rtol_use)
    assert np.allclose(ch_inv(A_i,cholesky_given=False),A,atol=atol_loc1,rtol=rtol_use)


def test_cholesky_inplace_inplace_F_order(test_mat):
    """test cholesky_inplace works F ordering """
    A = test_mat.A.copy()
    chol1 = test_mat.chol.copy()

    #Test doing cholesky in place with cholesky_inplace
    test_chol_inplace1 = A.copy('F')
    cholesky_inplace(test_chol_inplace1,inplace=True,lower=True)

    atol_loc3 = np.max(np.abs(chol1))*atol_rel_use

    assert check_is_cholesky(test_chol_inplace1,A,lower=True,atol_rel=atol_rel_use,rtol=rtol_use)
    assert np.allclose(chol1,test_chol_inplace1,atol=atol_loc3,rtol=rtol_use)
    test_chol_inplace1 = None


def test_cholesky_inplace_not_inplace_F_order(test_mat):
    """test cholesky_inplace works F ordering """
    A = test_mat.A.copy()
    chol1 = test_mat.chol.copy()
    #Test doing cholesky not in place with cholesky_inplace
    test_chol_inplace2 = A.copy('F')
    test_chol_inplace2_res = cholesky_inplace(test_chol_inplace2,inplace=False,lower=True)

    atol_loc1 = np.max(np.abs(A))*atol_rel_use
    atol_loc3 = np.max(np.abs(chol1))*atol_rel_use

    assert check_is_cholesky(test_chol_inplace2_res,A,lower=True,atol_rel=atol_rel_use,rtol=rtol_use)
    assert np.allclose(chol1,test_chol_inplace2_res,atol=atol_loc3,rtol=rtol_use)
    assert np.allclose(test_chol_inplace2,A,atol=atol_loc1,rtol=rtol_use)

def test_cholesky_inplace_not_inplace_C_order(test_mat):
    """test cholesky_inplace works C ordering """
    A = test_mat.A.copy()
    chol1 = test_mat.chol.copy()
    #Test doing cholesky not in place with C ordering
    test_chol_inplace2 = A.copy('C')
    test_chol_inplace2_res = cholesky_inplace(test_chol_inplace2,inplace=False,lower=True)

    atol_loc1 = np.max(np.abs(A))*atol_rel_use
    atol_loc3 = np.max(np.abs(chol1))*atol_rel_use

    assert check_is_cholesky(test_chol_inplace2_res,A,lower=True,atol_rel=atol_rel_use,rtol=rtol_use)
    assert np.allclose(chol1,test_chol_inplace2_res,atol=atol_loc3,rtol=rtol_use)
    assert np.allclose(test_chol_inplace2,A,atol=atol_loc1,rtol=rtol_use)


def test_cholesky_inplace_inplace_C_order_nonfatal(test_mat):
    """test cholesky_inplace works C ordering with error recovery"""
    A = test_mat.A.copy()
    chol1 = test_mat.chol.copy()
    #Test doing cholesky in place with C ordering (should cause warning unless also F ordering)
    test_chol_inplace4 = A.copy('C')

    atol_loc1 = np.max(np.abs(A))*atol_rel_use
    atol_loc3 = np.max(np.abs(chol1))*atol_rel_use

    with warnings.catch_warnings(record=True) as w:
        if not test_chol_inplace4.flags['F']:
            warnings.simplefilter("always")
            test_chol_inplace4_res = cholesky_inplace(test_chol_inplace4,inplace=True,fatal_errors=False)
            #check if it caused the warning
            assert len(w)==1
            assert issubclass(w[-1].category,RuntimeWarning)
            assert check_is_cholesky(test_chol_inplace4_res,A,atol_rel=atol_rel_use,rtol=rtol_use,lower=True)
            assert np.allclose(chol1,test_chol_inplace4_res,atol=atol_loc3,rtol=rtol_use)
            assert np.allclose(test_chol_inplace4,A,atol=atol_loc1,rtol=rtol_use)
    test_chol_inplace4 = None
    test_chol_inplace4_res = None

def test_cholesky_inplace_inplace_C_order_fatal(test_mat):
    """test cholesky_inplace works F ordering no error recovery"""
    A = test_mat.A.copy()
    #Test doing cholesky in place with C ordering (should cause fatal error)
    test_chol_inplace5 = A.copy('C')
    if not test_chol_inplace5.flags['F']:
        np.testing.assert_raises(RuntimeError, cholesky_inplace,test_chol_inplace5,inplace=True,fatal_errors=True)

def test_invert_triangular_lower(test_mat):
    """test invert_triangular on a lower matrix"""
    chol = test_mat.chol.copy()
    chol_i = test_mat.chol_i.copy()

    chol_i_test = chol_i.copy()
    chol_test = chol.copy()

    atol_loc3 = np.max(np.abs(chol))*atol_rel_use
    atol_loc4 = np.max(np.abs(chol_i))*atol_rel_use

    assert np.allclose(invert_triangular(chol_test,lower=True),chol_i,atol=atol_loc3,rtol=rtol_use)
    assert np.allclose(invert_triangular(chol_i_test,lower=True),chol,atol=atol_loc4,rtol=rtol_use)

def test_invert_triangular_upper(test_mat):
    """test invert_triangular on an upper matrix"""
    chol = test_mat.chol_u.copy()
    chol_i = test_mat.chol_i_u.copy()

    chol_i_test = chol_i.copy()
    chol_test = chol.copy()

    atol_loc3 = np.max(np.abs(chol))*atol_rel_use
    atol_loc4 = np.max(np.abs(chol_i))*atol_rel_use

    assert np.allclose(invert_triangular(chol_test,lower=False),chol_i,atol=atol_loc3,rtol=rtol_use)
    assert np.allclose(invert_triangular(chol_i_test,lower=False),chol,atol=atol_loc4,rtol=rtol_use)

def test_get_mat_from_inv_cholesky_direct_lower(test_mat):
    """test get_mat_from_inv_cholesky lower"""
    A = test_mat.A.copy()
    chol_i = test_mat.chol_i.copy()
    atol_loc1 = np.max(np.abs(A))*atol_rel_use
    assert np.allclose(get_mat_from_inv_cholesky(chol_i,lower=True),A,atol=atol_loc1,rtol=rtol_use)

def test_get_mat_from_inv_cholesky_direct_upper(test_mat):
    """test get_mat_from_inv_cholesky upper"""
    A = test_mat.A.copy()
    chol_i_u = test_mat.chol_i_u.copy()
    atol_loc1 = np.max(np.abs(A))*atol_rel_use
    assert np.allclose(get_mat_from_inv_cholesky(chol_i_u,lower=False),A,atol=atol_loc1,rtol=rtol_use)


def test_cholesky_inv_contract_scalar_direct_lower(test_mat):
    """test cholesky_inv_contract works with a scalar lower triangular"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = np.random.rand(A_i.shape[0])
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_inv_contract_scalar_direct_lower_cholesky_given(test_mat):
    """test cholesky_inv_contract works with a scalar lower triangular"""
    A_i = test_mat.A_i.copy()
    chol_i = test_mat.chol_i.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = np.random.rand(A_i.shape[0])
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=False,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_inv_contract_scalar_direct_upper(test_mat):
    """test cholesky_inv_contract works with a scalar upper triangular"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = np.random.rand(A_i.shape[0])
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_inv_contract_scalar_direct_upper_cholesky_given(test_mat):
    """test cholesky_inv_contract works with a scalar upper triangular"""
    A_i = test_mat.A_i.copy()
    chol_i = test_mat.chol_i_u.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = np.random.rand(A_i.shape[0])
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=False,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_inv_contract_scalar_direct_lower_identical(test_mat):
    """test cholesky_inv_contract works with a scalar lower triangular"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_inv_contract_scalar_direct_lower_cholesky_given_identical(test_mat):
    """test cholesky_inv_contract works with a scalar lower triangular"""
    A_i = test_mat.A_i.copy()
    chol_i = test_mat.chol_i.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_inv_contract_scalar_direct_upper_identical(test_mat):
    """test cholesky_inv_contract works with a scalar upper triangular"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_inv_contract_scalar_direct_upper_cholesky_given_identical(test_mat):
    """test cholesky_inv_contract works with a scalar upper triangular"""
    A_i = test_mat.A_i.copy()
    chol_i = test_mat.chol_i_u.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_inv_contract_matrix_direct_lower(test_mat):
    """test cholesky_inv_contract works with a general lower triangular"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = np.random.rand(dim1,dim2)
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_inv_contract_matrix_direct_lower_cholesky_given(test_mat):
    """test cholesky_inv_contract works with a general lower triangular"""
    A_i = test_mat.A_i.copy()
    chol_i = test_mat.chol_i.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = np.random.rand(dim1,dim2)
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=False,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_inv_contract_matrix_direct_upper(test_mat):
    """test cholesky_inv_contract works with a general upper triangular"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = np.random.rand(dim1,dim2)
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_inv_contract_matrix_direct_upper_cholesky_given(test_mat):
    """test cholesky_inv_contract works with a general upper triangular"""
    A_i = test_mat.A_i.copy()
    chol_i = test_mat.chol_i_u.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = np.random.rand(dim1,dim2)
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=False,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_inv_contract_matrix_direct_lower_identical_copy(test_mat):
    """test cholesky_inv_contract works with a general lower triangular copy"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = (vec1).copy()
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_inv_contract_matrix_direct_lower_cholesky_given_identical_copy(test_mat):
    """test cholesky_inv_contract works with a general lower triangular copy"""
    A_i = test_mat.A_i.copy()
    chol_i = test_mat.chol_i.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = (vec1).copy()
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_inv_contract_matrix_direct_upper_identical_copy(test_mat):
    """test cholesky_inv_contract works with a general upper triangular copy"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = (vec1).copy()
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_inv_contract_matrix_direct_upper_cholesky_given_identical_copy(test_mat):
    """test cholesky_inv_contract works with a general upper triangular copy"""
    A_i = test_mat.A_i.copy()
    chol_i = test_mat.chol_i_u.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = (vec1).copy()
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_inv_contract_matrix_direct_lower_identical_view(test_mat):
    """test cholesky_inv_contract works with a general lower triangular same matrix twice"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_inv_contract_matrix_direct_lower_cholesky_given_identical_view(test_mat):
    """test cholesky_inv_contract works with a general lower triangular same matrix twice"""
    A_i = test_mat.A_i.copy()
    chol_i = test_mat.chol_i.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_inv_contract_matrix_direct_upper_identical_view(test_mat):
    """test cholesky_inv_contract works with a general upper triangular same matrix twice"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_inv_contract_matrix_direct_upper_cholesky_given_identical_view(test_mat):
    """test cholesky_inv_contract works with a general upper triangular same matrix twice"""
    A_i = test_mat.A_i.copy()
    chol_i = npl.pinv(spl.cholesky(test_mat.A,lower=False))
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_i),vec2)
    contract_res_test = cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_contract_scalar_direct_lower(test_mat):
    """test cholesky_contract with a scalar"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = np.random.rand(A_i.shape[0])
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_contract_scalar_direct_lower_cholesky_given(test_mat):
    """test cholesky_contract with a scalar"""
    A_i = test_mat.A_i.copy()
    chol = test_mat.chol.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = np.random.rand(A_i.shape[0])
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(chol,vec1,vec2,cholesky_given=True,identical_inputs=False,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_contract_scalar_direct_upper(test_mat):
    """test cholesky_contract with a scalar"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = np.random.rand(A_i.shape[0])
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_contract_scalar_direct_upper_cholesky_given(test_mat):
    """test cholesky_contract with a scalar"""
    A_i = test_mat.A_i.copy()
    chol = test_mat.chol_u.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = np.random.rand(A_i.shape[0])
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(chol,vec1,vec2,cholesky_given=True,identical_inputs=False,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_contract_scalar_direct_lower_identical(test_mat):
    """test cholesky_contract with a scalar"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_contract_scalar_direct_lower_cholesky_given_identical(test_mat):
    """test cholesky_contract with a scalar"""
    A_i = test_mat.A_i.copy()
    chol = test_mat.chol.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(chol,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_contract_scalar_direct_upper_identical(test_mat):
    """test cholesky_contract with a scalar"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_contract_scalar_direct_upper_cholesky_given_identical(test_mat):
    """test cholesky_contract with a scalar"""
    A_i = test_mat.A_i.copy()
    chol = test_mat.chol_u.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(chol,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_contract_matrix_direct_lower(test_mat):
    """test cholesky_contract with a general matrix"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = np.random.rand(dim1,dim2)
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_contract_matrix_direct_lower_cholesky_given(test_mat):
    """test cholesky_contract with a general matrix"""
    A_i = test_mat.A_i.copy()
    chol = test_mat.chol.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = np.random.rand(dim1,dim2)
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(chol,vec1,vec2,cholesky_given=True,identical_inputs=False,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_contract_matrix_direct_upper(test_mat):
    """test cholesky_contract with a general matrix"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = np.random.rand(dim1,dim2)
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_contract_matrix_direct_upper_cholesky_given(test_mat):
    """test cholesky_contract with a general matrix"""
    A_i = test_mat.A_i.copy()
    chol = test_mat.chol_u.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = np.random.rand(dim1,dim2)
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(chol,vec1,vec2,cholesky_given=True,identical_inputs=False,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_contract_matrix_direct_lower_identical_copy(test_mat):
    """test cholesky_contract with a general matrix"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = (vec1).copy()
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_contract_matrix_direct_lower_cholesky_given_identical_copy(test_mat):
    """test cholesky_contract with a general matrix"""
    A_i = test_mat.A_i.copy()
    chol = test_mat.chol.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = (vec1).copy()
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(chol,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_contract_matrix_direct_upper_identical_copy(test_mat):
    """test cholesky_contract with a general matrix"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = (vec1).copy()
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_contract_matrix_direct_upper_cholesky_given_identical_copy(test_mat):
    """test cholesky_contract with a general matrix"""
    A_i = test_mat.A_i.copy()
    chol = test_mat.chol_u.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = (vec1).copy()
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(chol,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_contract_matrix_direct_lower_identical_view(test_mat):
    """test cholesky_contract with a general matrix"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_cholesky_contract_matrix_direct_lower_cholesky_given_identical_view(test_mat):
    """test cholesky_contract with a general matrix"""
    A_i = test_mat.A_i.copy()
    chol = test_mat.chol.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(chol,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=True)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_contract_matrix_direct_upper_identical_view(test_mat):
    """test cholesky_contract with a general matrix"""
    A_i = test_mat.A_i.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)


def test_cholesky_contract_matrix_direct_upper_cholesky_given_identical_view(test_mat):
    """test cholesky_contract with a general matrix"""
    A_i = test_mat.A_i.copy()
    chol = test_mat.chol_u.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2 = 1
    else:
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2)
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1.T,A_test),vec2)
    contract_res_test = cholesky_contract(chol,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=False)

    atol_loc = np.max(np.abs(contract_res))*atol_rel_use

    assert np.allclose(contract_res,contract_res_test,atol=atol_loc,rtol=rtol_use)

def test_get_cholesky_inv_lower(test_mat):
    """test get_cholesky_inv with lower triangular"""
    A = test_mat.A.copy()
    A_i = test_mat.A_i.copy()
    chol_test_inv = get_cholesky_inv(A,lower=True)
    chol_test = get_cholesky_inv(A_i,lower=True)

    assert check_is_cholesky(chol_test_inv,A_i,lower=True,atol_rel=atol_rel_use,rtol=rtol_use)
    assert check_is_cholesky(chol_test,A,lower=True,atol_rel=atol_rel_use,rtol=rtol_use)

def test_get_cholesky_inv_upper(test_mat):
    """test get_cholesky_inv with upper triangular"""
    A = test_mat.A.copy()
    A_i = test_mat.A_i.copy()
    chol_test_inv = get_cholesky_inv(A,lower=False)
    chol_test = get_cholesky_inv(A_i,lower=False)

    assert check_is_cholesky(chol_test_inv,A_i,lower=False,atol_rel=atol_rel_use,rtol=rtol_use)
    assert check_is_cholesky(chol_test,A,lower=False,atol_rel=atol_rel_use,rtol=rtol_use)

def test_trapz2_array_x(trapz_test):
    """test algebra_utils reimplementation of trapz with array of xs"""
    xs = trapz_test.xs
    integrand = trapz_test.integrand
    atol_use = atol_rel_use*np.max(integrand)

    integrated1 = np.trapz(integrand,xs,axis=0)
    integrated2 = trapz2(integrand,xs=xs)

    assert np.allclose(integrated1,integrated2,atol=atol_use,rtol=rtol_use)

def test_trapz2_array_dx(trapz_test):
    """test algebra_utils reimplementation of trapz with array of dxs"""
    xs = trapz_test.xs
    integrand = trapz_test.integrand
    atol_use = atol_rel_use*np.max(integrand)

    integrated1 = np.trapz(integrand,xs,axis=0)
    integrated2 = trapz2(integrand,dx=np.diff(xs,axis=0))
    assert np.allclose(integrated1,integrated2,atol=atol_use,rtol=rtol_use)

def test_trapz2_constant_dx(trapz_test):
    """test algebra_utils reimplementation of trapz with array of dxs"""
    print trapz_test.key
    xs = trapz_test.xs
    integrand = trapz_test.integrand
    atol_use = atol_rel_use*np.max(integrand)

    dx = np.average(np.diff(xs,axis=0))
    integrated1 = np.trapz(integrand,dx=dx,axis=0)
    integrated2 = trapz2(integrand,dx=dx)

    assert np.allclose(integrated1,integrated2,atol=atol_use,rtol=rtol_use)

def test_trapz2_no_dx(trapz_test):
    """test algebra_utils reimplementation of trapz with array of dxs"""
    integrand = trapz_test.integrand
    atol_use = atol_rel_use*np.max(integrand)

    integrated1 = np.trapz(integrand,axis=0)
    integrated2 = trapz2(integrand)

    assert np.allclose(integrated1,integrated2,atol=atol_use,rtol=rtol_use)

if __name__=='__main__':
    pytest.cmdline.main(['algebra_tests.py'])
