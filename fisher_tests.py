"""tests for the fisher_matrix module"""
#pylint: disable=W0621,duplicate-code
from __future__ import print_function,absolute_import,division
from builtins import range
import copy
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import pytest
import algebra_utils as au
from algebra_utils import cholesky_inplace,ch_inv,invert_triangular
import fisher_matrix as fm
au.DEBUG = True
fm.DEBUG = True
def check_is_cholesky(A,B,atol_rel=1e-08,rtol=1e-05,lower=True):
    """Check if A is the cholesky decomposition of B"""
    atol_loc1 = np.max(np.abs(B))*atol_rel
    atol_loc3 = np.max(np.abs(A))*atol_rel
    if lower:
        test1 = np.allclose(np.tril(A),A,atol=atol_loc3,rtol=rtol)
        return test1 and np.allclose(np.dot(A,A.T),B,atol=atol_loc1,rtol=rtol)
    else:
        test1 = np.allclose(np.triu(A),A,atol=atol_loc3,rtol=rtol)
        return test1 and np.allclose(np.dot(A.T,A),B,atol=atol_loc1,rtol=rtol)

def check_is_cholesky_inv(A,B,atol_rel=1e-08,rtol=1e-05,lower=True):
    """Check is A is the inverse cholesky decomposition of B"""
    chol = npl.pinv(A)
    B_inv = npl.pinv(B)

    atol_loc1 = atol_rel*np.max(np.abs(B))
    atol_loc2 = atol_rel*np.max(np.abs(B_inv))
    atol_loc4 = atol_rel*np.max(np.abs(A))

    if lower:
        test1 = np.allclose(np.dot(A.T,A),B_inv,atol=atol_loc2,rtol=rtol)
        test2 = np.allclose(np.dot(chol,chol.T),B,atol=atol_loc1,rtol=rtol)
        test3 = np.allclose(np.tril(A),A,atol=atol_loc4,rtol=rtol)
    else:
        test1 = np.allclose(np.dot(A,A.T),B_inv,atol=atol_loc2,rtol=rtol)
        test2 = np.allclose(np.dot(chol.T,chol),B,atol=atol_loc1,rtol=rtol)
        test3 = np.allclose(np.triu(A),A,atol=atol_loc4,rtol=rtol)
    return test1 and test2 and test3

def get_test_mat(key):
    """get the test matrix based on a key"""
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
        nside = 100
        result = np.random.rand(nside,nside)
        result = np.dot(result.T,result).copy('F')+np.diagflat(np.random.rand(nside,1))
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

class ManualFisherFromFab1(object):
    """testing fisher matrix object"""
    def __init__(self,fisher_in):
        """fisher_in: the fisher matrix"""
        self.fab = fisher_in
        self.cov = ch_inv(self.fab,cholesky_given=False,lower=True)
        self.chol_cov = cholesky_inplace(self.cov,inplace=False,lower=True)
        self.chol_cov_i = invert_triangular(self.chol_cov,lower=True)

class ManualFisherFromCov1(object):
    """testing covariance matrix object"""
    def __init__(self,cov_in):
        """cov_in: the covariance matrix"""
        self.cov = cov_in
        self.chol_cov = cholesky_inplace(self.cov,inplace=False,lower=True)
        self.chol_cov_i = invert_triangular(self.chol_cov,lower=True)
        self.fab = ch_inv(self.cov,cholesky_given=False)
atol_rel_use = 1e-7
rtol_use = 1e-6

#includes random tests
test_list = [[1,1],[2,1],[3,1],[4,1],[5,1],[7,1],[8,1],[8,1],[1,0],[2,0],[3,0],[4,0],[5,0],[7,0],[8,0],[8,0]]
#test_list = [[1,1],[2,1],[3,1],[4,1],[5,1],[7,1],[1,0],[2,0],[3,0],[4,0],[5,0],[7,0]]
@pytest.fixture(params=test_list)
def fisher_input(request):
    """iterate through test matrixes"""
    A = get_test_mat(request.param[0])
    if request.param[1]==0:
        return ManualFisherFromFab1(A)
    elif request.param[1]==1:
        return ManualFisherFromCov1(A)
    else:
        raise ValueError("fisher_input: unrecognized input: "+str(request.param))

class FisherWithManual(object):
    """testing fisher matrix object"""
    def __init__(self,fish_in,fisher):
        """fisher: FisherMatrix object"""
        self.fisher = fisher
        self.fisher_input = fish_in

input_initial = [[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]
#input_initial = [[0,0]]
@pytest.fixture(params=input_initial,scope="function")
def fisher_params(request,fisher_input):
    """iterate thorugh fisher matrix starting params and fisher matrices"""
    if request.param[0]==fm.REP_FISHER:
        internal_mat = fisher_input.fab.copy('F')
    elif request.param[0]==fm.REP_COVAR:
        internal_mat = fisher_input.cov.copy('F')
    elif request.param[0]==fm.REP_CHOL:
        internal_mat = fisher_input.chol_cov.copy('F')
    elif request.param[0]==fm.REP_CHOL_INV:
        internal_mat = fisher_input.chol_cov_i.copy('F')
    fisher =  fm.FisherMatrix(internal_mat,input_type=request.param[0],initial_state=request.param[1])
    return FisherWithManual(fisher_input,fisher)

def test_basic_setup_succeeded(fisher_input):
    """test self consistency of setup"""
    fab = fisher_input.fab.copy()
    cov = fisher_input.cov.copy()
    chol_cov = fisher_input.chol_cov.copy()
    chol_cov_i = fisher_input.chol_cov_i.copy()


    atol_loc1 = np.max(np.abs(cov))*atol_rel_use
    atol_loc2 = np.max(np.abs(fab))*atol_rel_use
    atol_loc3 = np.max(np.abs(chol_cov))*atol_rel_use
    atol_loc4 = np.max(np.abs(chol_cov_i))*atol_rel_use

    assert np.allclose(npl.pinv(cov),fab,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(npl.pinv(fab),cov,atol=atol_loc1,rtol=rtol_use)
    assert np.allclose(npl.pinv(chol_cov_i),chol_cov,atol=atol_loc3,rtol=rtol_use)
    assert np.allclose(npl.pinv(chol_cov),chol_cov_i,atol=atol_loc4,rtol=rtol_use)
    assert np.allclose(np.tril(chol_cov),chol_cov,atol=atol_loc3,rtol=rtol_use)
    assert check_is_cholesky(chol_cov,cov,atol_rel=atol_rel_use,rtol=rtol_use)
    assert check_is_cholesky_inv(chol_cov_i,cov,atol_rel=atol_rel_use,rtol=rtol_use)
    assert np.allclose(np.tril(chol_cov_i),chol_cov_i,atol=atol_loc4,rtol=rtol_use)
    assert check_is_cholesky_inv(chol_cov_i,cov,atol_rel=atol_rel_use,rtol=rtol_use)

#def test_perturb_fisher_vec(fisher_params):
#    """test perturb_fisher"""
#    fisher = copy.deepcopy(fisher_params.fisher)
#    fab = fisher_params.fisher_input.fab.copy()
#    vec = np.random.rand(1,fab.shape[0])
#    sigma2 = np.random.uniform(1,10)**2
#    fab2 = fab+sigma2*np.dot(vec.T,vec)
#    fisher.perturb_fisher(vec,sigma2)
#    fab3 = fisher.get_fisher()
#
#    atol_loc2 = np.max(np.abs(fab))*atol_rel_use
#    assert np.allclose(fab2,fab3,atol=atol_loc2,rtol=rtol_use)

def test_perturb_fisher_vec_1(fisher_params):
    """test perturb_fisher"""
    fisher_1 = copy.deepcopy(fisher_params.fisher)
    fisher_2 = copy.deepcopy(fisher_params.fisher)
    fab = fisher_params.fisher_input.fab.copy()
    cov = fisher_params.fisher_input.cov.copy()

    vec = np.random.rand(1,fab.shape[0])
    sigma2_1 = np.random.uniform(0.1,0.99,1)**2
    sigma2_2 = np.random.uniform(1.,5.,1)**2

    fab2_1 = fab+np.dot(vec.T*sigma2_1,vec)
    fab2_2 = fab+np.dot(vec.T*sigma2_2,vec)
    cov2_1 = np.linalg.inv(fab2_1)
    cov2_2 = np.linalg.inv(fab2_2)

    fisher_1.perturb_fisher(vec,sigma2_1)
    cov3_1 = fisher_1.get_covar()
    fab3_1 = fisher_1.get_fisher()
    fisher_2.perturb_fisher(vec,sigma2_2)
    cov3_2 = fisher_2.get_covar()
    fab3_2 = fisher_2.get_fisher()

    atol_loc1 = np.max(np.abs(cov))*atol_rel_use
    atol_loc2 = np.max(np.abs(fab))*atol_rel_use

    assert np.allclose(fab2_1,fab3_1,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(cov2_1,cov3_1,atol=atol_loc1,rtol=rtol_use)

    assert np.allclose(fab2_2,fab3_2,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(cov2_2,cov3_2,atol=atol_loc1,rtol=rtol_use)

def test_perturb_fisher_vec_5(fisher_params):
    """test perturb_fisher"""
    fisher_1 = copy.deepcopy(fisher_params.fisher)
    fisher_2 = copy.deepcopy(fisher_params.fisher)
    fab = fisher_params.fisher_input.fab.copy()
    cov = fisher_params.fisher_input.cov.copy()

    vec = np.random.rand(5,fab.shape[0])
    sigma2_1 = np.random.uniform(0.1,0.99,5)**2
    sigma2_2 = np.random.uniform(1.,5.,5)**2

    fab2_1 = fab+np.dot(vec.T*sigma2_1,vec)
    fab2_2 = fab+np.dot(vec.T*sigma2_2,vec)
    cov2_1 = np.linalg.inv(fab2_1)
    cov2_2 = np.linalg.inv(fab2_2)

    fisher_1.perturb_fisher(vec,sigma2_1)
    cov3_1 = fisher_1.get_covar()
    fab3_1 = fisher_1.get_fisher()
    fisher_2.perturb_fisher(vec,sigma2_2)
    cov3_2 = fisher_2.get_covar()
    fab3_2 = fisher_2.get_fisher()

    atol_loc1 = np.max(np.abs(cov))*atol_rel_use
    atol_loc2 = np.max(np.abs(fab))*atol_rel_use

    assert np.allclose(fab2_1,fab3_1,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(cov2_1,cov3_1,atol=atol_loc1,rtol=rtol_use)

    assert np.allclose(fab2_2,fab3_2,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(cov2_2,cov3_2,atol=atol_loc1,rtol=rtol_use)


def test_fab_any_input_any_initial(fisher_params):
    """test get_fisher consistency"""
    fab = fisher_params.fisher_input.fab.copy()
    cov = fisher_params.fisher_input.cov.copy()


    atol_loc1 = np.max(np.abs(cov))*atol_rel_use
    atol_loc2 = np.max(np.abs(fab))*atol_rel_use

    fisher = copy.deepcopy(fisher_params.fisher)

    fab_res1 = fisher.get_fisher()
    assert np.all(fab_res1==fab_res1.T)
    assert np.allclose(fab,fab_res1,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(ch_inv(fab_res1,lower=True),cov,atol=atol_loc1,rtol=rtol_use)
    assert np.allclose(ch_inv(cov,lower=True),fab_res1,atol=atol_loc2,rtol=rtol_use)


def test_chol_cov_any_input_any_initial(fisher_params):
    """test get_cov_cholesky consistency"""
    cov = fisher_params.fisher_input.cov.copy()
    chol_cov = fisher_params.fisher_input.chol_cov.copy()

    atol_loc3 = np.max(np.abs(chol_cov))*atol_rel_use

    fisher = copy.deepcopy(fisher_params.fisher)
    chol_cov_res1 = fisher.get_cov_cholesky()
    assert np.allclose(np.tril(chol_cov_res1),chol_cov_res1,atol=atol_loc3,rtol=rtol_use)
    assert np.allclose(chol_cov,chol_cov_res1,atol=atol_loc3,rtol=rtol_use)
    assert check_is_cholesky(chol_cov_res1,cov,atol_rel=atol_rel_use,rtol=rtol_use)

def test_chol_cov_i_any_input_any_initial(fisher_params):
    """test get_cov_cholesky_inv consistency"""
    cov = fisher_params.fisher_input.cov.copy()
    chol_cov_i = fisher_params.fisher_input.chol_cov_i.copy()

    atol_loc4 = np.max(np.abs(chol_cov_i))*atol_rel_use

    fisher = copy.deepcopy(fisher_params.fisher)
    chol_cov_i_res1 = fisher.get_cov_cholesky_inv()
    assert np.allclose(np.tril(chol_cov_i_res1),chol_cov_i_res1,atol=atol_loc4,rtol=rtol_use)
    assert np.allclose(chol_cov_i,chol_cov_i_res1,atol=atol_loc4,rtol=rtol_use)
    assert check_is_cholesky_inv(chol_cov_i_res1,cov,atol_rel=atol_rel_use,rtol=rtol_use)

def test_cov_any_input_any_initial(fisher_params):
    """test get_covar consistency"""
    fab = fisher_params.fisher_input.fab.copy()
    cov = fisher_params.fisher_input.cov.copy()

    fisher = copy.deepcopy(fisher_params.fisher)
    cov_res1 = fisher.get_covar()

    atol_loc1 = np.max(np.abs(cov))*atol_rel_use
    atol_loc2 = np.max(np.abs(fab))*atol_rel_use
    assert np.all(cov_res1==cov_res1.T)
    assert np.allclose(cov,cov_res1,atol=atol_loc1,rtol=rtol_use)
    assert np.allclose(ch_inv(cov_res1,lower=True),fab,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(ch_inv(fab,lower=True),cov_res1,atol=atol_loc1,rtol=rtol_use)

def test_contract_cov_self_ident_nofisher(fisher_params):
    """test contract_covar consistency"""
    cov = fisher_params.fisher_input.cov.copy()
    fisher = copy.deepcopy(fisher_params.fisher)
    cov_contracted1 = fisher.contract_covar(cov,cov,identical_inputs=True,return_fisher=False)
    cov_contracted2 = np.dot(cov.T,np.dot(cov,cov))

    atol_loc = np.max(np.abs(cov_contracted1))*atol_rel_use
    assert np.allclose(cov_contracted1,cov_contracted2,atol=atol_loc,rtol=rtol_use)
    assert np.allclose(cov_contracted1,cov_contracted1.T,atol=atol_loc,rtol=rtol_use)

def test_contract_cov_fab_ident_nofisher(fisher_params):
    """test constract_covar consistency"""
    fab = fisher_params.fisher_input.fab.copy()
    cov = fisher_params.fisher_input.cov.copy()

    fisher = copy.deepcopy(fisher_params.fisher)
    cov_contracted1 = fisher.contract_covar(fab,fab,identical_inputs=True,return_fisher=False)
    cov_contracted2 = np.dot(fab.T,np.dot(cov,fab))

    atol_loc = np.max(np.abs(cov_contracted1))*atol_rel_use
    assert np.allclose(cov_contracted1,cov_contracted2,atol=atol_loc,rtol=rtol_use)
    assert np.allclose(cov_contracted1,cov_contracted1.T,atol=atol_loc,rtol=rtol_use)


def test_contract_cov_fab_cov_nofisher(fisher_params):
    """test contract_covar consistency"""
    fab = fisher_params.fisher_input.fab.copy()
    cov = fisher_params.fisher_input.cov.copy()

    fisher = copy.deepcopy(fisher_params.fisher)
    cov_contracted1 = fisher.contract_covar(fab,cov,identical_inputs=False,return_fisher=False)
    cov_contracted2 = np.dot(fab.T,np.dot(cov,cov))

    atol_loc = np.max(np.abs(cov_contracted1))*atol_rel_use
    assert np.allclose(cov_contracted1,cov_contracted2,atol=atol_loc,rtol=rtol_use)
    assert np.allclose(cov_contracted1,cov_contracted1.T,atol=atol_loc,rtol=rtol_use)

def test_contract_cov_range1_nofisher(fisher_params):
    """test contract_covar consistency"""
    cov = fisher_params.fisher_input.cov.copy()
    range_mat = np.zeros((cov.shape[0],1))
    range_mat[:,0] = np.arange(cov.shape[0])
    fisher = copy.deepcopy(fisher_params.fisher)
    cov_contracted1 = fisher.contract_covar(range_mat,range_mat,identical_inputs=True,return_fisher=False)
    cov_contracted2 = np.dot(range_mat.T,np.dot(cov,range_mat))

    atol_loc = np.max(np.abs(cov_contracted1))*atol_rel_use
    assert np.allclose(cov_contracted1,cov_contracted2,atol=atol_loc,rtol=rtol_use)
    assert np.allclose(cov_contracted1,cov_contracted1.T,atol=atol_loc,rtol=rtol_use)

def test_contract_cov_range2_nofisher(fisher_params):
    """test contract_covar consistency"""
    cov = fisher_params.fisher_input.cov.copy()

    range_mat1 = np.zeros((cov.shape[0],cov.shape[0]))
    for i in range(0,cov.shape[0]):
        range_mat1[:,i] = np.arange(cov.shape[0])

    range_mat2 = np.zeros((cov.shape[0],1))
    range_mat2[:,0] = np.arange(cov.shape[0])
    fisher = copy.deepcopy(fisher_params.fisher)
    cov_contracted1 = fisher.contract_covar(range_mat1,range_mat2,identical_inputs=False,return_fisher=False)
    cov_contracted2 = np.dot(range_mat1.T,np.dot(cov,range_mat2))
    cov_contracted3 = fisher.contract_covar(range_mat2,range_mat1,identical_inputs=False,return_fisher=False)
    cov_contracted4 = np.dot(range_mat2.T,np.dot(cov,range_mat1))

    atol_loc = np.max(np.abs(cov_contracted1))*atol_rel_use
    assert np.allclose(cov_contracted1,cov_contracted2,atol=atol_loc,rtol=rtol_use)
    assert np.allclose(cov_contracted1,cov_contracted2,atol=atol_loc,rtol=rtol_use)
    assert np.allclose(cov_contracted3,cov_contracted4.T,atol=atol_loc,rtol=rtol_use)
    assert np.allclose(cov_contracted3,cov_contracted4.T,atol=atol_loc,rtol=rtol_use)

def test_project_cov_self(fisher_params):
    """test project_covar consistency"""
    cov = fisher_params.fisher_input.cov.copy()
    fisher = copy.deepcopy(fisher_params.fisher)
    cov_contracted1 = fisher.project_covar(cov).get_covar()
    cov_contracted2 = np.dot(cov.T,np.dot(cov,cov))

    atol_loc = np.max(np.abs(cov_contracted1))*atol_rel_use
    assert np.all(cov_contracted1==cov_contracted1.T)
    assert np.allclose(cov_contracted1,cov_contracted2,atol=atol_loc,rtol=rtol_use)
    assert np.allclose(cov_contracted1,cov_contracted1.T,atol=atol_loc,rtol=rtol_use)

def test_project_cov_fab(fisher_params):
    """test project_covar consistency"""
    fab = fisher_params.fisher_input.fab.copy()
    cov = fisher_params.fisher_input.cov.copy()

    fisher = copy.deepcopy(fisher_params.fisher)
    cov_contracted1 = fisher.project_covar(fab).get_covar()
    cov_contracted2 = np.dot(fab.T,np.dot(cov,fab))

    atol_loc = np.max(np.abs(cov_contracted1))*atol_rel_use
    assert np.all(cov_contracted1==cov_contracted1.T)
    assert np.allclose(cov_contracted1,cov_contracted2,atol=atol_loc,rtol=rtol_use)
    assert np.allclose(cov_contracted1,cov_contracted1.T,atol=atol_loc,rtol=rtol_use)

def test_project_cov_range1(fisher_params):
    """test project_covar consistency"""
    cov = fisher_params.fisher_input.cov.copy()
    range_mat = np.zeros((cov.shape[0],1))
    range_mat[:,0] = np.arange(cov.shape[0])
    fisher = copy.deepcopy(fisher_params.fisher)
    cov_contracted1 = fisher.project_covar(range_mat).get_covar()
    cov_contracted2 = np.dot(range_mat.T,np.dot(cov,range_mat))

    atol_loc = np.max(np.abs(cov_contracted1))*atol_rel_use
    assert np.all(cov_contracted1==cov_contracted1.T)
    assert np.allclose(cov_contracted1,cov_contracted2,atol=atol_loc,rtol=rtol_use)
    assert np.allclose(cov_contracted1,cov_contracted1.T,atol=atol_loc,rtol=rtol_use)

def test_contract_fab_self_ident_nofisher(fisher_params):
    """test contract_fisher consistency"""
    fab = fisher_params.fisher_input.fab.copy()
    fisher = copy.deepcopy(fisher_params.fisher)
    fab_contracted1 = fisher.contract_fisher(fab,fab,identical_inputs=True,return_fisher=False)
    fab_contracted2 = np.dot(fab.T,np.dot(fab,fab))

    atol_loc2 = np.max(np.abs(fab_contracted1))*atol_rel_use
    assert np.allclose(fab_contracted1,fab_contracted2,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(fab_contracted1,fab_contracted1.T,atol=atol_loc2,rtol=rtol_use)

def test_contract_fab_cov_ident_nofisher(fisher_params):
    """test contract_fisher consistency"""
    fab = fisher_params.fisher_input.fab.copy()
    cov = fisher_params.fisher_input.cov.copy()

    fisher = copy.deepcopy(fisher_params.fisher)
    fab_contracted1 = fisher.contract_fisher(cov,cov,identical_inputs=True,return_fisher=False)
    fab_contracted2 = np.dot(cov.T,np.dot(fab,cov))

    atol_loc2 = np.max(np.abs(fab_contracted1))*atol_rel_use
    assert np.allclose(fab_contracted1,fab_contracted2,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(fab_contracted1,fab_contracted1.T,atol=atol_loc2,rtol=rtol_use)


def test_contract_fab_fab_cov_nofisher(fisher_params):
    """test contract_fisher consistency"""
    fab = fisher_params.fisher_input.fab.copy()
    cov = fisher_params.fisher_input.cov.copy()

    fisher = copy.deepcopy(fisher_params.fisher)
    fab_contracted1 = fisher.contract_fisher(cov,fab,identical_inputs=False,return_fisher=False)
    fab_contracted2 = np.dot(cov.T,np.dot(fab,fab))

    atol_loc2 = np.max(np.abs(fab_contracted1))*atol_rel_use
    assert np.allclose(fab_contracted1,fab_contracted2,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(fab_contracted1,fab_contracted1.T,atol=atol_loc2,rtol=rtol_use)

def test_contract_fab_range1_nofisher(fisher_params):
    """test contract_fisher consistency"""
    fab = fisher_params.fisher_input.fab.copy()
    range_mat = np.zeros((fab.shape[0],1))
    range_mat[:,0] = np.arange(fab.shape[0])
    fisher = copy.deepcopy(fisher_params.fisher)
    fab_contracted1 = fisher.contract_fisher(range_mat,range_mat,identical_inputs=True,return_fisher=False)
    fab_contracted2 = np.dot(range_mat.T,np.dot(fab,range_mat))

    atol_loc2 = np.max(np.abs(fab_contracted1))*atol_rel_use
    assert np.allclose(fab_contracted1,fab_contracted2,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(fab_contracted1,fab_contracted1.T,atol=atol_loc2,rtol=rtol_use)

def test_contract_fab_range2_nofisher(fisher_params):
    """test contract_fisher consistency"""
    fab = fisher_params.fisher_input.fab.copy()

    range_mat1 = np.zeros((fab.shape[0],fab.shape[0]))
    for i in range(0,fab.shape[0]):
        range_mat1[:,i] = np.arange(fab.shape[0])

    range_mat2 = np.zeros((fab.shape[0],1))
    range_mat2[:,0] = np.arange(fab.shape[0])
    fisher = copy.deepcopy(fisher_params.fisher)
    fab_contracted1 = fisher.contract_fisher(range_mat1,range_mat2,identical_inputs=False,return_fisher=False)
    fab_contracted2 = np.dot(range_mat1.T,np.dot(fab,range_mat2))
    fab_contracted3 = fisher.contract_fisher(range_mat2,range_mat1,identical_inputs=False,return_fisher=False)
    fab_contracted4 = np.dot(range_mat2.T,np.dot(fab,range_mat1))

    atol_loc2 = np.max(np.abs(fab_contracted1))*atol_rel_use
    assert np.allclose(fab_contracted1,fab_contracted2,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(fab_contracted1,fab_contracted2,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(fab_contracted3,fab_contracted4.T,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(fab_contracted3,fab_contracted4.T,atol=atol_loc2,rtol=rtol_use)

def test_project_fab_self(fisher_params):
    """test project_fisher consistency"""
    fab = fisher_params.fisher_input.fab.copy()
    fisher = copy.deepcopy(fisher_params.fisher)
    fab_contracted1 = fisher.project_fisher(fab).get_fisher()
    fab_contracted2 = np.dot(fab.T,np.dot(fab,fab))

    atol_loc2 = np.max(np.abs(fab_contracted1))*atol_rel_use
    assert np.all(fab_contracted1==fab_contracted1.T)
    assert np.allclose(fab_contracted1,fab_contracted2,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(fab_contracted1,fab_contracted1.T,atol=atol_loc2,rtol=rtol_use)

def test_project_fab_cov(fisher_params):
    """test project_fisher consistency"""
    fab = fisher_params.fisher_input.fab.copy()
    cov = fisher_params.fisher_input.cov.copy()

    fisher = copy.deepcopy(fisher_params.fisher)
    fab_contracted1 = fisher.project_fisher(cov).get_fisher()
    fab_contracted2 = np.dot(cov.T,np.dot(fab,cov))

    atol_loc2 = np.max(np.abs(fab_contracted1))*atol_rel_use
    assert np.all(fab_contracted1==fab_contracted1.T)
    assert np.allclose(fab_contracted1,fab_contracted2,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(fab_contracted1,fab_contracted1.T,atol=atol_loc2,rtol=rtol_use)

def test_project_fab_range1(fisher_params):
    """test project_fisher consistency"""
    fab = fisher_params.fisher_input.fab.copy()
    range_mat = np.zeros((fab.shape[0],1))
    range_mat[:,0] = np.arange(fab.shape[0])
    fisher = copy.deepcopy(fisher_params.fisher)
    fab_contracted1 = fisher.project_fisher(range_mat).get_fisher()
    fab_contracted2 = np.dot(range_mat.T,np.dot(fab,range_mat))

    atol_loc2 = np.max(np.abs(fab_contracted1))*atol_rel_use
    assert np.all(fab_contracted1==fab_contracted1.T)
    assert np.allclose(fab_contracted1,fab_contracted2,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(fab_contracted1,fab_contracted1.T,atol=atol_loc2,rtol=rtol_use)

def test_add_fisher1(fisher_params):
    """test add_fisher"""
    fab = fisher_params.fisher_input.fab.copy()
    cov = fisher_params.fisher_input.cov.copy()
    fisher = copy.deepcopy(copy.deepcopy(fisher_params.fisher))
    fisher.add_fisher(cov)
    fab2 = fisher.get_fisher()
    cov2 = fisher.get_covar()
    fab3 = fab+cov
    cov3 = np.linalg.pinv(fab3)
    atol_loc2 = np.max(np.abs(fab))*atol_rel_use
    atol_loc1 = np.max(np.abs(cov2))*atol_rel_use

    assert np.allclose(fab2,fab3,atol=atol_loc2,rtol=rtol_use)
    assert np.allclose(cov2,cov3,atol=atol_loc1,rtol=rtol_use)


def test_get_eig_metric_diag_range(fisher_params):
    """test get_cov_eig_metric"""
    cov = fisher_params.fisher_input.cov.copy()
    fisher = copy.deepcopy(fisher_params.fisher)

    metric_mat = np.diag(np.arange(1,cov.shape[0]+1))*1.
    metric_mat_inv = np.linalg.pinv(metric_mat)
    metric = fm.FisherMatrix(metric_mat,input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR)

    eigs_1 = fisher.get_cov_eig_metric(metric)
    atol_loc = np.max(np.abs(eigs_1[0]-1.))*atol_rel_use

    #m_mat = np.identity(cov.shape[0])+np.dot(cov,metric_mat_inv)
    m_mat = np.dot(cov,metric_mat_inv)
    eigs_3 = spl.eig(m_mat)
    #check matrix is positive semidefinite
    assert np.all(eigs_3[0]>0.)
    assert np.all(np.abs(np.imag(eigs_3[0]-1.))[::-1]<atol_loc)
    #check eigenvalues match
    assert np.allclose(np.sort(eigs_1[0]-1.),np.sort(np.real(eigs_3[0]-1.)),atol=atol_loc,rtol=rtol_use)

    chol_metric = spl.cholesky(metric_mat,lower=True)
    chol_inv_metric = npl.pinv(chol_metric)
    #alt_mat = np.identity(cov.shape[0])+np.dot(chol_inv_metric,np.dot(cov,chol_inv_metric.T))
    alt_mat = np.dot(chol_inv_metric,np.dot(cov,chol_inv_metric.T))
    #u vectors
    eigvs_4 = np.dot(chol_inv_metric,eigs_3[1])
    #v vectors
    eigvs_5 = np.dot(chol_metric,eigs_1[1])

    #check eigenvalues work
    assert np.allclose(np.dot(m_mat,eigs_3[1]),eigs_3[0]*eigs_3[1],atol=atol_loc,rtol=rtol_use)
    assert np.allclose(np.dot(alt_mat,eigs_1[1]),eigs_1[0]*eigs_1[1],atol=atol_loc,rtol=rtol_use)
    assert np.allclose(np.identity(cov.shape[0]),np.dot(eigs_1[1].T,eigs_1[1]),atol=atol_loc,rtol=rtol_use)
    p_mat0 = np.dot(eigs_3[1].T,np.dot(metric_mat_inv,eigs_3[1]))
    p_mat0_use = p_mat0-np.diag(np.diag(p_mat0))
    assert np.allclose(np.zeros(cov.shape),p_mat0_use,atol=atol_loc,rtol=rtol_use)

    #check cholesky transforms eigenvalues as expected
    p_mat1 = np.dot(eigvs_4.T,eigvs_4)
    p_mat1_use = p_mat1-np.diag(np.diag(p_mat1))
    assert np.allclose(np.zeros(cov.shape),p_mat1_use,atol=atol_loc,rtol=rtol_use)
    assert np.allclose(np.dot(alt_mat,eigvs_4),(eigs_3[0]*eigvs_4),atol=atol_loc,rtol=rtol_use)
    p_mat2 = np.dot(eigvs_5.T,np.dot(metric_mat_inv,eigvs_5))
    p_mat2_use = p_mat2-np.diag(np.diag(p_mat2))
    assert np.allclose(np.zeros(cov.shape),p_mat2_use,atol=atol_loc,rtol=rtol_use)
    assert np.allclose(np.dot(m_mat,eigvs_5),eigs_1[0]*eigvs_5,atol=atol_loc,rtol=rtol_use)

def test_get_eig_metric_rand(fisher_params):
    """test get_cov_eig_metric"""
    cov = fisher_params.fisher_input.cov.copy()
    fisher = copy.deepcopy(fisher_params.fisher)
    metric_mat = np.random.rand(cov.shape[0],cov.shape[1])
    metric_mat = np.dot(metric_mat.T,metric_mat)
    metric_mat_inv = np.linalg.pinv(metric_mat)
    metric = fm.FisherMatrix(metric_mat,input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR)

    eigs_1 = fisher.get_cov_eig_metric(metric)
    atol_loc = np.max(np.abs(eigs_1[0]-1.))*atol_rel_use

    m_mat = np.dot(cov,metric_mat_inv)
    eigs_3 = spl.eig(m_mat)
    #check matrix is positive semidefinite
    assert np.all(eigs_3[0]>0.)
    assert np.all(np.abs(np.imag(eigs_3[0]-1.))[::-1]<atol_loc)
    #check eigenvalues match
    assert np.allclose(np.sort(eigs_1[0]-1.),np.sort(np.real(eigs_3[0]-1.)),atol=atol_loc,rtol=rtol_use)

    chol_metric = spl.cholesky(metric_mat,lower=True)
    chol_inv_metric = npl.pinv(chol_metric)
    alt_mat = np.dot(chol_inv_metric,np.dot(cov,chol_inv_metric.T))
    #u vectors
    eigvs_4 = np.dot(chol_inv_metric,eigs_3[1])
    #v vectors
    eigvs_5 = np.dot(chol_metric,eigs_1[1])

    #check eigenvalues work
    assert np.allclose(np.dot(m_mat,eigs_3[1]),eigs_3[0]*eigs_3[1],atol=atol_loc,rtol=rtol_use)
    assert np.allclose(np.dot(alt_mat,eigs_1[1]),eigs_1[0]*eigs_1[1],atol=atol_loc,rtol=rtol_use)
    assert np.allclose(np.identity(cov.shape[0]),np.dot(eigs_1[1].T,eigs_1[1]),atol=atol_loc,rtol=rtol_use)
    p_mat0 = np.dot(eigs_3[1].T,np.dot(metric_mat_inv,eigs_3[1]))
    p_mat0_use = p_mat0-np.diag(np.diag(p_mat0))
    assert np.allclose(np.zeros(cov.shape),p_mat0_use,atol=atol_loc,rtol=rtol_use)

    #check cholesky transforms eigenvalues as expected
    p_mat1 = np.dot(eigvs_4.T,eigvs_4)
    p_mat1_use = p_mat1-np.diag(np.diag(p_mat1))
    assert np.allclose(np.zeros(cov.shape),p_mat1_use,atol=atol_loc,rtol=rtol_use)
    assert np.allclose(np.dot(alt_mat,eigvs_4),(eigs_3[0]*eigvs_4),atol=atol_loc,rtol=rtol_use)
    p_mat2 = np.dot(eigvs_5.T,np.dot(metric_mat_inv,eigvs_5))
    p_mat2_use = p_mat2-np.diag(np.diag(p_mat2))
    assert np.allclose(np.zeros(cov.shape),p_mat2_use,atol=atol_loc,rtol=rtol_use)
    assert np.allclose(np.dot(m_mat,eigvs_5),eigs_1[0]*eigvs_5,atol=atol_loc,rtol=rtol_use)




if __name__=='__main__':
    pytest.cmdline.main(['fisher_tests.py'])
