import numpy as np
import numpy.linalg as npl
import scipy as sp
import scipy.linalg as spl
import warnings
from time import time
from algebra_utils import cholesky_inplace,get_inv_cholesky,ch_inv,invert_triangular
import sys
import pytest
import fisher_matrix as fm

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

class ManualFisherFromFab1:
    def __init__(self,fisher_in):
        self.fab = fisher_in
        self.cov = ch_inv(self.fab,cholesky_given=False,lower=True)
        self.chol_cov = cholesky_inplace(self.cov,inplace=False,lower=True)
        self.chol_cov_i = invert_triangular(self.chol_cov,lower=True)

class ManualFisherFromCov1:
    def __init__(self,cov_in):
        self.cov = cov_in
        self.chol_cov = cholesky_inplace(self.cov,inplace=False,lower=True)
        self.chol_cov_i = invert_triangular(self.chol_cov,lower=True)
        self.fab = ch_inv(self.cov,cholesky_given=False)

test_list = [[1,1],[2,1],[3,1],[4,1],[5,1],[7,1],[8,1],[8,1],[1,0],[2,0],[3,0],[4,0],[5,0],[7,0],[8,0],[8,0]]
@pytest.fixture(params=test_list)
def fisher_input(request):
    A = get_test_mat(request.param[0])
    if request.param[1] == 0:
        return ManualFisherFromFab1(A)
    elif request.param[1] == 1:
        return ManualFisherFromCov1(A)
    else:
        raise ValueError("fisher_input: unrecognized input: "+str(request.param))

class FisherWithManual:
    def __init__(self,fish_in,fisher):
        self.fisher=fisher
        self.fisher_input = fish_in

input_initial = [[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]
#input_initial = [[0,0]]
@pytest.fixture(params=input_initial,scope="function")
def fisher_params(request,fisher_input):
    if request.param[0] == fm.REP_FISHER:
        internal_mat = fisher_input.fab.copy() 
    elif request.param[0] == fm.REP_COVAR:
        internal_mat = fisher_input.cov.copy()
    elif request.param[0] == fm.REP_CHOL:
        internal_mat = fisher_input.chol_cov.copy()
    elif request.param[0] == fm.REP_CHOL_INV:
        internal_mat = fisher_input.chol_cov_i.copy()
    fisher =  fm.fisher_matrix(internal_mat,input_type=request.param[0],initial_state=request.param[1])
    return FisherWithManual(fisher_input,fisher)

def test_basic_setup_succeeded(fisher_input):
    fab = fisher_input.fab
    cov = fisher_input.cov
    chol_cov = fisher_input.chol_cov
    chol_cov_i = fisher_input.chol_cov_i
    assert(np.allclose(npl.pinv(cov),fab))
    assert(np.allclose(npl.pinv(fab),cov))
    assert(np.allclose(npl.pinv(chol_cov),chol_cov_i))
    assert(np.allclose(npl.pinv(chol_cov_i),chol_cov))
    assert(np.allclose(np.tril(chol_cov),chol_cov))
    assert(check_is_cholesky(chol_cov,cov))
    assert(check_is_inv_cholesky(chol_cov_i,cov))
    assert(np.allclose(np.tril(chol_cov_i),chol_cov_i))
    assert(check_is_cholesky(chol_cov_i.T,fab))
    assert(check_is_inv_cholesky(chol_cov.T,fab))

def test_fab_any_input_any_inital(fisher_params):
    fab = fisher_params.fisher_input.fab
    cov = fisher_params.fisher_input.cov
    chol_cov = fisher_params.fisher_input.chol_cov
    chol_cov_i = fisher_params.fisher_input.chol_cov_i 

    fisher = fisher_params.fisher

    fab_res1 = fisher.get_F_alpha_beta()
    assert(np.allclose(fab,fab_res1))
    assert(np.allclose(ch_inv(fab_res1,lower=True),cov))
    assert(np.allclose(ch_inv(cov,lower=True),fab_res1))


def test_chol_cov_any_input_any_inital(fisher_params):
    fab = fisher_params.fisher_input.fab
    cov = fisher_params.fisher_input.cov
    chol_cov = fisher_params.fisher_input.chol_cov
    chol_cov_i = fisher_params.fisher_input.chol_cov_i 

    fisher = fisher_params.fisher
    chol_cov_res1 = fisher.get_cov_cholesky()
    assert(np.allclose(np.tril(chol_cov_res1),chol_cov_res1))
    assert(np.allclose(chol_cov,chol_cov_res1))
    assert(check_is_cholesky(chol_cov_res1,cov))

def test_chol_cov_i_any_input_any_inital(fisher_params):
    fab = fisher_params.fisher_input.fab
    cov = fisher_params.fisher_input.cov
    chol_cov = fisher_params.fisher_input.chol_cov
    chol_cov_i = fisher_params.fisher_input.chol_cov_i 

    fisher = fisher_params.fisher
    chol_cov_i_res1 = fisher.get_fab_cholesky()
    assert(np.allclose(np.tril(chol_cov_i_res1),chol_cov_i_res1))
    assert(np.allclose(chol_cov_i,chol_cov_i_res1))
    assert(check_is_inv_cholesky(chol_cov_i_res1,cov))

def test_cov_any_input_any_inital(fisher_params):
    fab = fisher_params.fisher_input.fab
    cov = fisher_params.fisher_input.cov
    chol_cov = fisher_params.fisher_input.chol_cov
    chol_cov_i = fisher_params.fisher_input.chol_cov_i 

    fisher = fisher_params.fisher
    cov_res1 = fisher.get_covar()
    assert(np.allclose(cov,cov_res1))
    assert(np.allclose(ch_inv(cov_res1,lower=True),fab))
    assert(np.allclose(ch_inv(fab,lower=True),cov_res1))


if __name__=='__main__':
        pytest.cmdline.main(['algebra_tests.py'])
