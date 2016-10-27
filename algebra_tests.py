import numpy as np
import numpy.linalg as npl
import scipy as sp
import scipy.linalg as spl
import warnings
from time import time
from algebra_utils import cholesky_inplace,get_inv_cholesky,ch_inv,invert_triangular,get_mat_from_inv_cholesky,cholesky_inv_contract
import sys
import pytest

#Check if A is the cholesky decomposition of B
def check_is_cholesky(A,B,atol=1e-08,rtol=1e-05):
   return np.allclose(np.dot(A,A.T),B,atol=atol,rtol=rtol)

def check_is_inv_cholesky(A,B,atol=1e-08,rtol=1e-05):
    chol = npl.pinv(A)
    return np.allclose(np.dot(chol,chol.T),B,atol=atol,rtol=rtol)

def get_test_mat(key):
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
        #np.testing.assert_raises(ValueError,cholesky_inplace,test_mat6,inplace=True,fatal_errors=True)
    elif key==7:
        return np.array([[4.]],order='F',dtype=np.float_) 
    elif key==8:
        nside = 100
        result = np.random.rand(nside,nside) 
        result = np.dot(result.T,result).copy('F')
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

test_list = [1,2,3,4,5,7,8,8,8,8,8]

class BasicSetupMatrices:
    def __init__(self,A):
        self.A = A
        self.chol_i = npl.pinv(npl.cholesky(A))
        self.chol = npl.cholesky(A)
        self.A_i = npl.pinv(A)


@pytest.fixture(params=test_list)
def test_mat(request):
    A = get_test_mat(request.param)
    return BasicSetupMatrices(A)



    
relax_rtol = 1e-01
relax_atol = 1e-03
tighten_atol = 1e-9

    #inv loses precision so have to relax tolerance dramatically for tests involving direct inverse
    #this shows the superiority of using the cholesky inverse
    #note this loss of precision could cause some random matrices to fail while others pass
    #TODO: better precision testing of code could improve battery
def test_basic_setup_succeeded(test_mat):
    chol_i1 = test_mat.chol_i
    chol1 = test_mat.chol
    A = test_mat.A
    A_i = test_mat.A_i
        

    assert(check_is_cholesky(chol_i1.T,A_i,atol=relax_atol,rtol=relax_rtol))
    assert(check_is_cholesky(chol1,A))
    assert(check_is_inv_cholesky(chol1.T,A_i,atol=relax_atol,rtol=relax_rtol))
    assert(check_is_inv_cholesky(chol_i1,A,atol=tighten_atol))
    assert(np.allclose(npl.pinv(A_i),A,atol=relax_atol,rtol=relax_rtol))
    assert(np.allclose(npl.pinv(chol_i1),chol1,atol=relax_atol,rtol=relax_rtol))

def test_get_inv_cholesky_F_order_direct_lower(test_mat):
    A = test_mat.A
    chol_i1 = test_mat.chol_i
    A_i = test_mat.A_i
    #Test directly with fortran ordering
    test_A = A.copy('F')
    test_chol_i1 = get_inv_cholesky(test_A,lower=True)
    assert(np.allclose(test_A,A))
    assert(np.allclose(test_chol_i1,chol_i1))
    assert(check_is_cholesky(test_chol_i1.T,A_i,atol=relax_atol))
    assert(check_is_inv_cholesky(test_chol_i1,A,atol=tighten_atol))
    test_A = None
    test_chol_i1 = None

def test_get_inv_cholesky_F_order_inverse_lower(test_mat):
    A = test_mat.A
    A_i = test_mat.A_i
    #check L*L.T=A^-1 => (L.T^-1)*L^-1=A=B*B.T note significant loss of precision here Fortran ordering
    test_A_i = A_i.copy('F')
    test_chol2 = get_inv_cholesky(test_A_i,lower=True)
    assert(np.allclose(test_A_i,A_i))
    assert(check_is_cholesky(test_chol2.T,A,atol=relax_atol,rtol=relax_rtol))
    test_A_i = None
    test_chol2 = None
    #print test_chol2.T
    #print chol1
    #assert(np.allclose(test_chol2.T,chol1))

def test_get_inv_cholesky_C_order_direct_lower(test_mat):
    A = test_mat.A
    chol_i1 = test_mat.chol_i
    A_i = test_mat.A_i
    #Test directly with C ordering
    test_A = A.copy('C')
    test_chol_i3 = get_inv_cholesky(test_A,lower=True)
    assert(np.allclose(test_A,A))
    assert(np.allclose(test_chol_i3,chol_i1))
    assert(check_is_cholesky(test_chol_i3.T,A_i,atol=relax_atol))
    assert(check_is_inv_cholesky(test_chol_i3,A,atol=tighten_atol))
    test_A = None
    test_chol_i3 = None

def test_get_inv_cholesky_C_order_inverse_lower(test_mat):
    A = test_mat.A
    A_i = test_mat.A_i
    #check L*L.T=A^-1 => (L.T^-1)*L^-1=A=B*B.T note significant loss of precision here C ordering
    test_A_i = A_i.copy('C')
    test_chol4 = get_inv_cholesky(test_A_i,lower=True)
    assert(np.allclose(test_A_i,A_i))
    assert(check_is_cholesky(test_chol4.T,A,atol=relax_atol,rtol=relax_rtol))
    test_A_i = None
    test_chol4 = None


def test_get_inv_cholesky_F_order_direct_upper(test_mat):
    A = test_mat.A
    chol_i1 = test_mat.chol_i
    A_i = test_mat.A_i
    #Test directly with fortran ordering
    test_A = A.copy('F')
    test_chol_i1 = get_inv_cholesky(test_A,lower=False)
    assert(np.allclose(test_A,A))
    assert(check_is_cholesky(test_chol_i1,A_i,atol=relax_atol))
    assert(check_is_inv_cholesky(test_chol_i1.T,A,atol=tighten_atol))
    test_A = None
    test_chol_i1 = None


def test_get_inv_cholesky_F_order_inverse_upper(test_mat):
    A = test_mat.A
    A_i = test_mat.A_i
    #check L*L.T=A^-1 => (L.T^-1)*L^-1=A=B*B.T note significant loss of precision here Fortran ordering
    test_A_i = A_i.copy('F')
    test_chol2 = get_inv_cholesky(test_A_i,lower=False)
    assert(np.allclose(test_A_i,A_i))
    assert(check_is_cholesky(test_chol2,A,atol=relax_atol,rtol=relax_rtol))
    test_A_i = None
    test_chol2 = None
    #print test_chol2.T


def test_get_inv_cholesky_C_order_direct_upper(test_mat):
    A = test_mat.A
    chol_i1 = test_mat.chol_i
    A_i = test_mat.A_i
    #Test directly with C ordering
    test_A = A.copy('C')
    test_chol_i1 = get_inv_cholesky(test_A,lower=False)
    assert(np.allclose(test_A,A))
    assert(check_is_cholesky(test_chol_i1,A_i,atol=relax_atol))
    assert(check_is_inv_cholesky(test_chol_i1.T,A,atol=tighten_atol))
    test_A = None
    test_chol_i1 = None


def test_get_inv_cholesky_C_order_inverse_upper(test_mat):
    A = test_mat.A
    A_i = test_mat.A_i
    #check L*L.T=A^-1 => (L.T^-1)*L^-1=A=B*B.T note significant loss of precision here C ordering
    test_A_i = A_i.copy('C')
    test_chol2 = get_inv_cholesky(test_A_i,lower=False)
    assert(np.allclose(test_A_i,A_i))
    assert(check_is_cholesky(test_chol2,A,atol=relax_atol,rtol=relax_rtol))
    test_A_i = None
    test_chol2 = None
    #print test_chol2.T

def test_ch_inv_chol_given_lower(test_mat):
    A = test_mat.A
    A_i = test_mat.A_i
    chol = test_mat.chol
    chol_i = test_mat.chol_i
    
    test_chol = chol.copy()
    test_chol_i = chol_i.copy()

    assert(np.allclose(ch_inv(test_chol_i,cholesky_given=True,lower=True),A_i))
    assert(np.allclose(ch_inv(test_chol.T,cholesky_given=True,lower=True),A))
    
def test_ch_inv_chol_given_upper(test_mat):
    A = test_mat.A
    A_i = test_mat.A_i
    chol = test_mat.chol.T
    chol_i = test_mat.chol_i.T
    
    test_chol = chol.copy()
    test_chol_i = chol_i.copy()

    assert(np.allclose(ch_inv(test_chol_i,cholesky_given=True,lower=False),A_i))
    assert(np.allclose(ch_inv(test_chol.T,cholesky_given=True,lower=False),A))
    
def test_ch_inv_not_chol_given_lower(test_mat):
    A = test_mat.A
    A_i = test_mat.A_i
    chol = test_mat.chol.T
    chol_i = test_mat.chol_i.T
    
    test_chol = chol.copy()
    test_chol_i = chol_i.copy()

    assert(np.allclose(ch_inv(A,cholesky_given=False,lower=True),A_i))
    assert(np.allclose(ch_inv(A_i,cholesky_given=False,lower=True),A))

def test_ch_inv_not_chol_given_upper(test_mat):
    A = test_mat.A
    A_i = test_mat.A_i
    chol = test_mat.chol.T
    chol_i = test_mat.chol_i.T
    
    test_chol = chol.copy()
    test_chol_i = chol_i.copy()

    assert(np.allclose(ch_inv(A,cholesky_given=False,lower=False),A_i))
    assert(np.allclose(ch_inv(A_i,cholesky_given=False,lower=False),A))

def test_ch_inv_both_F_order(test_mat):
    A = test_mat.A
    A_i = test_mat.A_i
     
    test_A1 = A.copy('F')
    test_A_i1 = A_i.copy('F')
    assert(np.allclose(ch_inv(test_A1),A_i,atol=relax_atol,rtol=relax_rtol))
    assert(np.allclose(A,test_A1))
    assert(np.allclose(ch_inv(ch_inv(test_A1)),A,atol=relax_atol,rtol=relax_rtol))
    assert(np.allclose(ch_inv(A_i),A,atol=relax_atol,rtol=relax_rtol))
    test_A1 = None
    test_A_i1 = None

def test_ch_inv_both_C_order(test_mat):
    A = test_mat.A
    A_i = test_mat.A_i

    test_A1 = A.copy('C')
    test_A_i1 = A_i.copy('C')
    assert(np.allclose(ch_inv(test_A1),A_i,atol=relax_atol,rtol=relax_rtol))
    assert(np.allclose(A,test_A1))
    assert(np.allclose(ch_inv(ch_inv(test_A1)),A,atol=relax_atol,rtol=relax_rtol))
    assert(np.allclose(ch_inv(A_i),A,atol=relax_atol,rtol=relax_rtol))
    test_A1 = None
    test_A_i1 = None


def test_ch_inv_F_then_C_order(test_mat):
    A = test_mat.A
    A_i = test_mat.A_i

    test_A1 = A.copy('F')
    test_A_i1 = A_i.copy('C')
    assert(np.allclose(ch_inv(test_A1),A_i,atol=relax_atol,rtol=relax_rtol))
    assert(np.allclose(A,test_A1))
    assert(np.allclose(ch_inv(ch_inv(test_A1)),A,atol=relax_atol,rtol=relax_rtol))
    assert(np.allclose(ch_inv(A_i),A,atol=relax_atol,rtol=relax_rtol))
    test_A1 = None
    test_A_i1 = None


def test_ch_inv_C_then_F_order(test_mat):
    A = test_mat.A
    A_i = test_mat.A_i

    test_A1 = A.copy('C')
    test_A_i1 = A_i.copy('F')
    assert(np.allclose(ch_inv(test_A1),A_i,atol=relax_atol,rtol=relax_rtol))
    assert(np.allclose(A,test_A1))
    assert(np.allclose(ch_inv(ch_inv(test_A1)),A,atol=relax_atol,rtol=relax_rtol))
    assert(np.allclose(ch_inv(A_i),A,atol=relax_atol,rtol=relax_rtol))
    test_A1 = None
    test_A_i1 = None


def test_cholesky_inplace_inplace_F_order(test_mat):
    A = test_mat.A
    chol1 = test_mat.chol

    #Test doing cholesky in place with cholesky_inplace
    test_chol_inplace1 = A.copy('F')
    cholesky_inplace(test_chol_inplace1,inplace=True)
    assert(check_is_cholesky(test_chol_inplace1,A))
    assert(np.allclose(chol1,test_chol_inplace1))
    test_chol_inplace1=None


def test_cholesky_inplace_not_inplace_F_order(test_mat):
    A = test_mat.A
    chol1 = test_mat.chol
    #Test doing cholesky not in place with cholesky_inplace
    test_chol_inplace2 = A.copy('F')
    test_chol_inplace2_res = cholesky_inplace(test_chol_inplace2,inplace=False)
    assert(check_is_cholesky(test_chol_inplace2_res,A))
    assert(np.allclose(chol1,test_chol_inplace2_res))
    assert(np.all(test_chol_inplace2==A))
    test_chol_inplace2=None
    test_chol_inplace2_res=None
    
def test_cholesky_inplace_not_inplace_C_order(test_mat):
    A = test_mat.A
    chol1 = test_mat.chol
    #Test doing cholesky not in place with C ordering
    test_chol_inplace3 = A.copy('C')
    test_chol_inplace3_res = cholesky_inplace(test_chol_inplace3,inplace=False)
    assert(check_is_cholesky(test_chol_inplace3_res,A))
    assert(np.allclose(chol1,test_chol_inplace3_res))
    assert(np.all(test_chol_inplace3==A))
    test_chol_inplace3=None
    test_chol_inplace3_res=None

def test_cholesky_inplace_inplace_C_order_nonfatal(test_mat):
    A = test_mat.A
    chol1 = test_mat.chol
    #Test doing cholesky in place with C ordering (should cause warning unless also F ordering)
    test_chol_inplace4 = A.copy('C')
    with warnings.catch_warnings(record=True) as w:
        if not test_chol_inplace4.flags['F']:     
            warnings.simplefilter("always")
            test_chol_inplace4_res = cholesky_inplace(test_chol_inplace4,inplace=True,fatal_errors=False)
            #check if it caused the warning
            assert(len(w)==1)
            assert(issubclass(w[-1].category,RuntimeWarning))
            assert(check_is_cholesky(test_chol_inplace4_res,A))
            assert(np.allclose(chol1,test_chol_inplace4_res))
            assert(np.all(test_chol_inplace4==A))
    test_chol_inplace4=None
    test_chol_inplace4_res=None

def test_cholesky_inplace_inplace_C_order_fatal(test_mat):
    A = test_mat.A
    #Test doing cholesky in place with C ordering (should cause fatal error)
    test_chol_inplace5 = A.copy('C')
    if not test_chol_inplace5.flags['F']:
        np.testing.assert_raises(RuntimeError, cholesky_inplace,test_chol_inplace5,inplace=True,fatal_errors=True)
    test_chol_inplace5 = None
    
def test_invert_triangular_lower(test_mat):
    chol = test_mat.chol
    chol_i = test_mat.chol_i

    chol_i_test = chol_i.copy()
    chol_test = chol.copy()
    assert(np.allclose(invert_triangular(chol_test,lower=True),chol_i))
    assert(np.allclose(invert_triangular(chol_i_test,lower=True),chol))

def test_invert_triangular_upper(test_mat):
    chol = test_mat.chol.T
    chol_i = test_mat.chol_i.T

    chol_i_test = chol_i.copy()
    chol_test = chol.copy()
    assert(np.allclose(invert_triangular(chol_test,lower=False),chol_i))
    assert(np.allclose(invert_triangular(chol_i_test,lower=False),chol))
    
def test_get_mat_from_inv_cholesky_direct(test_mat):
    A = test_mat.A
    chol_i = test_mat.chol_i
 
    chol_i_test = chol_i.copy()
    assert(np.allclose(get_mat_from_inv_cholesky(chol_i_test),A))

def test_get_mat_from_inv_cholesky_inverse(test_mat):
    A_i = test_mat.A_i
    chol = test_mat.chol.T
 
    chol_test = chol.copy()
    assert(np.allclose(get_mat_from_inv_cholesky(chol_test,lower=False),A_i,atol=relax_atol,rtol=relax_rtol))
    

#TODO random may not be the best option here
def test_cholesky_inv_contract_scalar_direct_lower(test_mat):
    A_i = test_mat.A_i
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = np.random.rand(A_i.shape[0])   
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=True)
    assert(np.allclose(contract_res,contract_res_test))
     

def test_cholesky_inv_contract_scalar_direct_lower_cholesky_given(test_mat):
    A_i = test_mat.A_i
    chol_i = test_mat.chol_i.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = np.random.rand(A_i.shape[0])   
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=False,lower=True)
    assert(np.allclose(contract_res,contract_res_test))


def test_cholesky_inv_contract_scalar_direct_upper(test_mat):
    A_i = test_mat.A_i
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = np.random.rand(A_i.shape[0])   
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=False)
    assert(np.allclose(contract_res,contract_res_test))
     

def test_cholesky_inv_contract_scalar_direct_upper_cholesky_given(test_mat):
    A_i = test_mat.A_i
    chol_i = test_mat.chol_i.T.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = np.random.rand(A_i.shape[0])   
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=False,lower=False)
    assert(np.allclose(contract_res,contract_res_test))
    
def test_cholesky_inv_contract_scalar_direct_lower_identical(test_mat):
    A_i = test_mat.A_i
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = vec1 
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=True)
    assert(np.allclose(contract_res,contract_res_test))

def test_cholesky_inv_contract_scalar_direct_lower_cholesky_given_identical(test_mat):
    A_i = test_mat.A_i
    chol_i = test_mat.chol_i.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=True)
    assert(np.allclose(contract_res,contract_res_test))


def test_cholesky_inv_contract_scalar_direct_upper_identical(test_mat):
    A_i = test_mat.A_i
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = vec1 
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=False)
    assert(np.allclose(contract_res,contract_res_test))
     

def test_cholesky_inv_contract_scalar_direct_upper_cholesky_given_identical(test_mat):
    A_i = test_mat.A_i
    chol_i = test_mat.chol_i.T.copy()
    A_test = test_mat.A.copy()
    vec1 = np.random.rand(A_i.shape[0])
    vec2 = vec1
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=False)
    assert(np.allclose(contract_res,contract_res_test))


def test_cholesky_inv_contract_matrix_direct_lower(test_mat):
    A_i = test_mat.A_i
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2=1
    else: 
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2).T
    vec2 = np.random.rand(dim1,dim2)   
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=True)
    assert(np.allclose(contract_res,contract_res_test))
     

def test_cholesky_inv_contract_matrix_direct_lower_cholesky_given(test_mat):
    A_i = test_mat.A_i
    chol_i = test_mat.chol_i.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2=1
    else: 
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2).T
    vec2 = np.random.rand(dim1,dim2)   
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=False,lower=True)
    assert(np.allclose(contract_res,contract_res_test))


def test_cholesky_inv_contract_matrix_direct_upper(test_mat):
    A_i = test_mat.A_i
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2=1
    else: 
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2).T
    vec2 = np.random.rand(dim1,dim2)   
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=False)
    assert(np.allclose(contract_res,contract_res_test))
     

def test_cholesky_inv_contract_matrix_direct_upper_cholesky_given(test_mat):
    A_i = test_mat.A_i
    chol_i = test_mat.chol_i.T.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2=1
    else: 
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2).T
    vec2 = np.random.rand(dim1,dim2)   
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=False,lower=False)
    assert(np.allclose(contract_res,contract_res_test))
    
def test_cholesky_inv_contract_matrix_direct_lower_identical_copy(test_mat):
    A_i = test_mat.A_i
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2=1
    else: 
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2).T
    vec2 = (vec1.T).copy()
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=True)
    assert(np.allclose(contract_res,contract_res_test))

def test_cholesky_inv_contract_matrix_direct_lower_cholesky_given_identical_copy(test_mat):
    A_i = test_mat.A_i
    chol_i = test_mat.chol_i.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2=1
    else: 
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2).T
    vec2 = (vec1.T).copy()
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=True)
    assert(np.allclose(contract_res,contract_res_test))


def test_cholesky_inv_contract_matrix_direct_upper_identical_copy(test_mat):
    A_i = test_mat.A_i
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2=1
    else: 
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2).T
    vec2 = (vec1.T).copy()
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=False)
    assert(np.allclose(contract_res,contract_res_test))
     

def test_cholesky_inv_contract_matrix_direct_upper_cholesky_given_identical_copy(test_mat):
    A_i = test_mat.A_i
    chol_i = test_mat.chol_i.T.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2=1
    else: 
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2).T
    vec2 = (vec1.T).copy()
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=False)
    assert(np.allclose(contract_res,contract_res_test))

def test_cholesky_inv_contract_matrix_direct_lower_identical_view(test_mat):
    A_i = test_mat.A_i
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2=1
    else: 
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2).T
    vec2 = vec1.T
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=True)
    assert(np.allclose(contract_res,contract_res_test))

def test_cholesky_inv_contract_matrix_direct_lower_cholesky_given_identical_view(test_mat):
    A_i = test_mat.A_i
    chol_i = test_mat.chol_i.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2=1
    else: 
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2).T
    vec2 = vec1.T
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=True)
    assert(np.allclose(contract_res,contract_res_test))


def test_cholesky_inv_contract_matrix_direct_upper_identical_view(test_mat):
    A_i = test_mat.A_i
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2=1
    else: 
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2).T
    vec2 = vec1.T
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(A_test,vec1,vec2,cholesky_given=False,identical_inputs=True,lower=False)
    assert(np.allclose(contract_res,contract_res_test))
     

def test_cholesky_inv_contract_matrix_direct_upper_cholesky_given_identical_view(test_mat):
    A_i = test_mat.A_i
    chol_i = test_mat.chol_i.T.copy()
    A_test = test_mat.A.copy()
    dim1 = A_i.shape[0]
    if dim1==1:
        dim2=1
    else: 
        dim2 = dim1-1
    vec1 = np.random.rand(dim1,dim2).T
    vec2 = vec1.T
    contract_res = np.dot(np.dot(vec1,A_i),vec2)
    contract_res_test=cholesky_inv_contract(chol_i,vec1,vec2,cholesky_given=True,identical_inputs=True,lower=False)
    assert(np.allclose(contract_res,contract_res_test))

if __name__=='__main__':
    pytest.cmdline.main(['algebra_tests.py'])
