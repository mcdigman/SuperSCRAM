"""
General class for Fisher matrix and covariance matrix manipulations
"""
from __future__ import absolute_import,division,print_function
from builtins import range
import numpy as np
import scipy.linalg as spl
from algebra_utils import get_inv_cholesky,invert_triangular,cholesky_inplace,get_cholesky_inv,mirror_symmetrize,clean_triangle

#codes for different possible internal states of fisher matrix
#Fisher matrix
REP_FISHER = 0
#cholesky decomposition of covariance matrix
REP_CHOL = 1
#inverse of cholesky decomposition of covariance matrix
REP_CHOL_INV = 2
#covariance matrix
REP_COVAR = 3

DEBUG = False

#things to watch out for: the output matrix may mutate if it was the internal matrix
#NOTE: _internal_matrix makes no guarantees about its upper right triangle internally
#so using np.dot without ensuring that it is clean (by calling a get function) is very dangerous

class FisherMatrix(object):
    """General Fisher matrix object for fisher matrix and covariance manipulations"""
    def __init__(self,input_matrix,input_type,initial_state=None,fix_input=False,silent=True):
        """create a fisher matrix object
            input_matrix: an input matrix
            input_type: a code for matrix type of input_matrix, options REP_FISHER, REP_CHOL, REP_CHOL_INV, or REP_COVAR
            initial_state: code for starting state, defaults to input_type
            fix_input: do not mutate input_matrix (ie make a copy)
            silent: if True, less print(statements)
        """
        self.silent = silent
        self.fix_input = fix_input
        #default initial state is not to change
        if initial_state is None:
            initial_state = input_type

        #If not allowed to change the input matrix, then copy it
        if self.fix_input:
            self._internal_mat = input_matrix.copy()
        else:
            self._internal_mat = input_matrix

        self.internal_state = input_type

        #Switch to the specified initial state
        if not initial_state==self.internal_state:
            self.switch_rep(initial_state)
        if not self.silent:
            print("FisherMatrix "+str(id(self))+" created fisher matrix")

    def switch_rep(self,new_state):
        """Switches the internal representation of the fisher matrix to new_state
            may mutate _internal_mat to save memory"""
        old_state = self.internal_state
        self._internal_mat = np.asfortranarray(self._internal_mat)
        if not self.silent:
            print("FisherMatrix "+str(id(self))+": changing internal state from "+str(self.internal_state)+" to "+str(new_state))
        if old_state==new_state:
            if not self.silent:
                print("FisherMatrix "+str(id(self))+": internal state "+str(self.internal_state)+" unchanged")
        elif new_state==REP_CHOL:
            self._internal_mat = self.get_cov_cholesky(inplace=True,copy_output=False,internal=True)
        elif new_state==REP_CHOL_INV:
            self._internal_mat = self.get_cov_cholesky_inv(inplace=True,copy_output=False,internal=True)
        elif new_state==REP_FISHER:
            self._internal_mat = self.get_fisher(inplace=True,copy_output=False,internal=True)
        elif new_state==REP_COVAR:
            self._internal_mat = self.get_covar(inplace=True,copy_output=False,internal=True)
        else:
            raise ValueError("FisherMatrix "+str(id(self))+": unrecognized new state "+str(new_state)+" in state "+str(old_state))
        self.internal_state = new_state
        if not self.silent:
            print("FisherMatrix "+str(id(self))+": internal state changed from "+str(self.internal_state)+" to "+str(new_state))

    def perturb_fisher(self,vs,sigma2s,force_sherman=False):
        """add perturbation the fisher matrix in the form sigma2*v^Tv,
        use Sherman-Morrison formula/Woodbury Matrix identity to avoid inverting/for numerical stability"""
        if not self.silent:
            print("FisherMatrix ",id(self)," perturbing fisher matrix")
        #internal state must be fisher to add right now: could possibly be changed back
        if np.any(sigma2s<0.):
            raise ValueError('perturbation must be positive semidefinite '+str(sigma2s))

        if self.internal_state==REP_FISHER or self.internal_state==REP_CHOL_INV:
            self.switch_rep(REP_FISHER)
            self._internal_mat = np.asfortranarray(self._internal_mat)
            self._internal_mat = spl.blas.dsyrk(1.,(vs.T*np.sqrt(sigma2s)).T,1.,c=self._internal_mat,overwrite_c=True,trans=True,lower=True)
        else:
            self.switch_rep(REP_COVAR)
            self._internal_mat = np.asfortranarray(self._internal_mat)
            #self._internal_mat2 = self._internal_mat.copy()
            if force_sherman:
                for itr in range(0,sigma2s.size):
                    v = vs[itr:itr+1]
                    lhs = spl.blas.dsymm(1.,self._internal_mat,v.T,lower=True,overwrite_c=False,side=False)
                    #use more numerically stable form for large sigma2s, although mathematically equivalent
                    if sigma2s[itr]>1.:
                        mult = -1./(1./sigma2s[itr]+np.dot(v,lhs)[0,0])
                    elif sigma2s[itr]>0.:
                        mult = -sigma2s[itr]/(1.+sigma2s[itr]*np.dot(v,lhs)[0,0])
                    else: #don't bother perturbing the matrix if nothing to add
                        continue
                    self._internal_mat2 = spl.blas.dsyrk(mult,lhs,1.,c=self._internal_mat,lower=True,trans=False,overwrite_c=True)
            else:
                lhs1 = np.asfortranarray(spl.blas.dsymm(1.,self._internal_mat,vs,lower=True,overwrite_c=False,side=True))
                mult_mat = np.asfortranarray(np.diag(1./sigma2s))+spl.blas.dgemm(1.,vs,lhs1,trans_b=True)
                mult_mat_chol_inv = get_cholesky_inv(mult_mat,lower=True,inplace=False,clean=False)
                mult_mat=None
                lhs2 = spl.blas.dtrmm(1.,mult_mat_chol_inv,lhs1,side=False,lower=True,trans_a=True)
                mult_mat_chol_inv = None
                lhs1=None
                self._internal_mat = spl.blas.dsyrk(-1.,lhs2,1.,self._internal_mat,trans=True,lower=True,overwrite_c=True)
                #self._internal_mat = mirror_symmetrize(self._internal_mat,lower=True,inplace=True)
                #self._internal_mat2 = mirror_symmetrize(self._internal_mat2,lower=True,inplace=True)




        if not self.silent:
            print("FisherMatrix ",id(self)," finished perturbing fisher matrix")



    def add_fisher(self,F_n):
        """Add a fisher matrix (ie a perturbation) to the internal fisher matrix"""
        if not self.silent:
            print("FisherMatrix ",id(self)," adding fisher matrix")
        #internal state must be fisher to add right now: could possibly be changed back
        self.switch_rep(REP_FISHER)
        if F_n.__class__ is FisherMatrix:
            print("FisherMatrix: accumulating in internal state "+str(F_n.internal_state))
            self._internal_mat += F_n.get_fisher(copy_output=False,internal=True,inplace=False)
        else:
            print("FisherMatrix: accumulating directly")
            self._internal_mat += F_n

    def add_covar(self,C_n):
        """add a covariance matrix, C_n is numpy array  or FisherMatrix object"""
        if not self.silent:
            print("FisherMatrix ",id(self)," added covariance matrix")
        #internal state must be covar to add right now: could possibly be changed
        self.switch_rep(REP_COVAR)
        if C_n.__class__ is FisherMatrix:
            self._internal_mat += C_n.get_covar(copy_output=False,internal=True,inplace=False)
        else:
            self._internal_mat += C_n

    #copy_output=True should guarantee the output will never be mutated by FisherMatrix
    #(ie copy the output matrix if necessary)
    #copy_output=False will not copy the output matrix in general,
    #which will be more memory efficient but might cause problems if the user is not careful
    def get_covar(self,inplace=False,copy_output=False,internal=False):
        """get covariance matrix
                inputs:
                    inplace: if True, will mutate _internal_mat to be the covariance matrix
                    copy_output: whether to copy the output matrix, to be safe from mutating _internal_mat later
                    internal: should be True only if being called from another FisherMatrix routine, do not need to clean upper triangle"""

        #if _internal_mat changes must change internal state
        if inplace and not internal:
            self.switch_rep(REP_COVAR)

        if not self.silent:
            print("FisherMatrix "+str(id(self))+": getting covariance")
        if self.internal_state==REP_COVAR:
            if not self.silent:
                print("FisherMatrix "+str(id(self))+": covar retrieved from cache")
            result = self._internal_mat
        else:
            if not self.silent:
                print("FisherMatrix "+str(id(self))+": covar cache miss")
            chol_cov = self.get_cov_cholesky(inplace=inplace,internal=True,copy_output=False)
            #dlauum calculates L^TL but have LL^T cholesky convention, rot90s flip to correct convention
            result, info = spl.lapack.dlauum(np.asfortranarray(np.rot90(chol_cov,2)),lower=False,overwrite_c=inplace)
            chol_cov = None
            if info!=0:
                raise RuntimeError('dlauum failed with error code '+str(info))
            result = np.rot90(result,2)

        result = np.asfortranarray(result)

        #make sure guaranteed to change _internal_mat if inplace was true
        if inplace:
            self._internal_mat = result
            self.internal_state = REP_COVAR

        #make sure symmetric if not for internal use by FisherMatrix
        if not internal:
            result = mirror_symmetrize(result,lower=True,inplace=True)

        if copy_output:
            return result.copy()
        else:
            return result

    def contract_covar(self,v1,v2,identical_inputs=False,return_fisher=False,destructive=False):
        """calculates (v1.T).covariance.v2 for getting variance/projecting to another basis
            inputs:
                v1,v2: vectors with one dimension aligned with _internal_mat
                identical_inputs: if True, assume v2=v1 and ignore v2 completely
                return_fisher: if True, return a FisherMatrix object
                destructive: if True, destroy the internal representation of self for a performance gain
        """
        if not self.silent:
            print("FisherMatrix "+str(id(self))+": contracting covariance")
        if return_fisher and not identical_inputs:
            raise ValueError('cannot get FisherMatrix object if inputs not identical')

        if  self.internal_state==REP_COVAR:
            if identical_inputs:
                right_res = spl.blas.dsymm(1.,self._internal_mat,v1,lower=True)
            else:
                right_res = spl.blas.dsymm(1.,self._internal_mat,v2,lower=True)

            if destructive:
                self._internal_mat = None
            result = np.dot(v1.T,right_res)

            right_res = None
            if identical_inputs and not return_fisher:
                result = mirror_symmetrize(result,lower=True,inplace=True)
        else:
            if self.internal_state==REP_FISHER:
                chol_fisher = cholesky_inplace(self._internal_mat,lower=True,inplace=destructive,clean=False)
                if destructive:
                    self._internal_mat = None
                res1 = spl.solve_triangular(chol_fisher,v1,lower=True,check_finite=DEBUG)
                if not identical_inputs:
                    res2 = spl.solve_triangular(chol_fisher,v2,lower=True,check_finite=DEBUG)
                chol_fisher = None
            elif self.internal_state==REP_CHOL_INV:
                res1 = spl.solve_triangular(self._internal_mat,v1,lower=True,check_finite=DEBUG,trans=True)
                if not identical_inputs:
                    res2 = spl.solve_triangular(self._internal_mat,v2,lower=True,check_finite=DEBUG,trans=True)
            elif self.internal_state==REP_CHOL:
                res1 = spl.blas.dtrmm(1.,self._internal_mat,v1,lower=True,trans_a=True)
                if not identical_inputs:
                    res2 = spl.blas.dtrmm(1.,self._internal_mat,v2,lower=True,trans_a=True)
            else:
                raise ValueError('unknown internal state '+str(self.internal_state))

            if destructive:
                self._internal_mat = None

            if identical_inputs:
                result = spl.blas.dsyrk(1.,np.asfortranarray(res1),lower=True,trans=True,overwrite_c=True)
                res1 = None
                if not return_fisher:
                    result = mirror_symmetrize(result,lower=True,inplace=True)
            else:
                result = np.dot(res1.T,res2)
                res1 = None
                res2 = None
        if return_fisher:
            return FisherMatrix(result,input_type=REP_COVAR,initial_state=REP_COVAR,fix_input=False,silent=self.silent)
        else:
            return result

    #needed for projecting to another basis
    def contract_fisher(self,v1,v2,identical_inputs=False,return_fisher=False,destructive=False):
        """calculates (v1.T).Fisher matrix.v2 for getting variance/projecting to another basis
            inputs:
                v1,v2: vectors with one dimension aligned with _internal_mat
                identical_inputs: if True, assume v2=v1 and ignore v2 completely
                return_fisher: if True, return a FisherMatrix object
                destructive: if True, destroy the internal representation of self for a performance gain
        """
        if not self.silent:
            print("FisherMatrix "+str(id(self))+": contracting fisher")

        if return_fisher and not identical_inputs:
            raise ValueError('cannot get FisherMatrix object if inputs not identical')

        if self.internal_state==REP_FISHER:
            if identical_inputs:
                right_res = spl.blas.dsymm(1.,self._internal_mat,v1,lower=True)
            else:
                right_res = spl.blas.dsymm(1.,self._internal_mat,v2,lower=True)
            if destructive:
                self._internal_mat = None
            result = np.dot(v1.T,right_res)
            right_res = None
            if identical_inputs and not return_fisher:
                result = mirror_symmetrize(result,lower=True,inplace=True)
        else:
            if self.internal_state==REP_COVAR:
                chol_cov = self.get_cov_cholesky(copy_output=False,inplace=destructive,internal=True)
                if destructive:
                    self._internal_mat = None
                res1 = spl.solve_triangular(chol_cov,v1,lower=True,check_finite=DEBUG)
                if not identical_inputs:
                    res2 = spl.solve_triangular(chol_cov,v2,lower=True,check_finite=DEBUG)
                chol_cov = None
            elif self.internal_state==REP_CHOL_INV:
                res1 = spl.blas.dtrmm(1.,self._internal_mat,v1,lower=True,trans_a=False)
                if not identical_inputs:
                    res2 = spl.blas.dtrmm(1.,self._internal_mat,v2,lower=True,trans_a=False)
            elif self.internal_state==REP_CHOL:
                res1 = spl.solve_triangular(self._internal_mat,v1,lower=True,check_finite=DEBUG)
                if not identical_inputs:
                    res2 = spl.solve_triangular(self._internal_mat,v2,lower=True,check_finite=DEBUG)
            else:
                raise ValueError('unknown internal state '+str(self.internal_state))

            if destructive:
                self._internal_mat = None

            if identical_inputs:
                result = spl.blas.dsyrk(1.,np.asfortranarray(res1),lower=True,trans=True,overwrite_c=True)
                res1 = None
                if not return_fisher:
                    result = mirror_symmetrize(result,lower=True,inplace=True)
            else:
                result = np.dot(res1.T,res2)
                res1 = None
                res2 = None
        if destructive:
            self._internal_mat = None


        if return_fisher:
            return FisherMatrix(result,input_type=REP_FISHER,initial_state=REP_FISHER,fix_input=False,silent=self.silent)
        else:
            return result

    def project_fisher(self,v1,destructive=False):
        """project using (v1.T).Fisher.v1"""
        result = self.contract_fisher(v1,v1,identical_inputs=True,return_fisher=False,destructive=destructive)
        return FisherMatrix(result,input_type=REP_FISHER,initial_state=REP_FISHER,fix_input=False,silent=self.silent)

    #project using covar
    def project_covar(self,v1,destructive=False):
        """project using (v1.T).Covariance.v1"""
        result = self.contract_covar(v1,v1,identical_inputs=True,return_fisher=True,destructive=destructive)
        return result

    #this is not fisher cholesky, but the cholesky decomposition of the inverse of the covariance
    #which is not exactly the same ie LL^T vs L^TL decompositions
    def get_cov_cholesky_inv(self,inplace=False,copy_output=False,internal=False):
        """get inverse of lower triangular cholesky decomposition of covariance
                inputs:
                    inplace: if True, will mutate _internal_mat to be the inverse cholesky decompostion of the covariance matrix
                    copy_output: whether to copy the output matrix, to be safe from mutating _internal_mat later
                    internal: should be True only if being called from another FisherMatrix routine, do not need to clean upper triangle"""

        #if _internal_mat changes must change internal state
        if inplace and not internal:
            self.switch_rep(REP_CHOL_INV)

        if self.internal_state==REP_CHOL_INV:
            if not self.silent:
                print("FisherMatrix ",id(self)," cholesky decomposition inv retrieved from cache")
            result = self._internal_mat
        else:
            if not self.silent:
                print("FisherMatrix ",id(self)," cholesky decomposition inv cache miss")
            if self.internal_state==REP_CHOL:
                result = invert_triangular(self._internal_mat,lower=True,inplace=inplace,clean=False)
            elif self.internal_state==REP_FISHER:
                result = np.asfortranarray(np.rot90(self._internal_mat,2))
                result = np.rot90(cholesky_inplace(result,inplace=inplace,lower=False,clean=False),2)
                result = np.asfortranarray(result)
            elif self.internal_state==REP_COVAR:
                result = get_inv_cholesky(self._internal_mat,lower=True,inplace=inplace,clean=False)
            else:
                raise ValueError("FisherMatrix "+str(id(self))+": unrecognized internal state "+str(self.internal_state))

        if inplace:
            self._internal_mat = result
            self.internal_state = REP_CHOL_INV

        if not internal:
            result = clean_triangle(result,lower=True,inplace=True)

        if  copy_output:
            return result.copy()
        else:
            return result

    def get_cov_cholesky(self,inplace=False,copy_output=False,internal=False):
        """get lower triangular cholesky decomposition of covariance
                inputs:
                    inplace: if True, will mutate _internal_mat to be the cholesky decompostion of the covariance matrix
                    copy_output: whether to copy the output matrix, to be safe from mutating _internal_mat later
                    internal: should be True only if being called from another FisherMatrix routine, do not need to clean upper triangle"""

        #if _internal_mat changes must change internal state
        if inplace and not internal:
            self.switch_rep(REP_CHOL)

        if self.internal_state==REP_CHOL:
            if not self.silent:
                print("FisherMatrix ",str(id(self))+": cholesky stored, size: "+str(self._internal_mat.nbytes/10**6)+" megabytes")
            result = self._internal_mat
        else:
            if self.internal_state==REP_CHOL_INV :
                if not self.silent:
                    print("FisherMatrix "+str(id(self)),": getting cholesky of covariance from its inverse, size: "+str(self._internal_mat.nbytes/10**6)+" mb")
                result = invert_triangular(self._internal_mat,lower=True,inplace=inplace,clean=False)
            elif self.internal_state==REP_COVAR:
                if not self.silent:
                    print("FisherMatrix "+str(id(self)),": getting cholesky from covariance directly: "+str(self._internal_mat.nbytes/10**6)+" mb")
                result = cholesky_inplace(self._internal_mat,inplace=inplace,lower=True,clean=False)
            elif self.internal_state==REP_FISHER:
                if not self.silent:
                    print("FisherMatrix "+str(id(self)),": getting cholesky of covariance from fisher, size: "+str(self._internal_mat.nbytes/10**6)+" mb")
                result = get_cholesky_inv(self._internal_mat,lower=True,inplace=inplace,clean=False)
            else:
                raise ValueError("FisherMatrix "+str(id(self))+": unrecognized internal state "+str(self.internal_state))

            if not self.silent:
                print("FisherMatrix "+str(id(self)),": found cholesky decomposition of covariance matrix, size: "+str(result.nbytes/10**6)+" megabytes")

        if inplace:
            self._internal_mat = result
            self.internal_state = REP_CHOL

        #if not being requested internally to FisherMatrix, make sure the upper triangle is all zeros
        if not internal:
            result = clean_triangle(result,lower=True,inplace=True)

        if copy_output:
            return result.copy()
        else:
            return result

    def get_fisher(self,copy_output=False,inplace=False,internal=False):
        """get fisher matrix
                inputs:
                    inplace: if True, will mutate _internal_mat to be the fisher matrix
                    copy_output: whether to copy the output matrix, to be safe from mutating _internal_mat later
                    internal: should be True only if being called from another FisherMatrix routine, do not need to clean upper triangle"""

        #if _internal_mat changes must change internal state
        if inplace and not internal:
            self.switch_rep(REP_FISHER)

        if self.internal_state==REP_FISHER:
            if not self.silent:
                print("FisherMatrix "+str(id(self))+": retrieved fisher matrix from cache")
            result = self._internal_mat
        elif self.internal_state==REP_COVAR or self.internal_state==REP_CHOL:
            chol_cov = np.asfortranarray(self.get_cov_cholesky(copy_output=False,internal=True,inplace=inplace))
            result,info = spl.lapack.dpotri(chol_cov,lower=True,overwrite_c=inplace)
            chol_cov = None
            if info!=0:
                raise RuntimeError('dpotri failed with error code '+str(info))
        else:
            if not self.silent:
                print("FisherMatrix "+str(id(self))+": fisher matrix cache miss")
            chol_res = np.asfortranarray(self.get_cov_cholesky_inv(copy_output=False,internal=True,inplace=inplace))
            result,info = spl.lapack.dlauum(chol_res,lower=True,overwrite_c=inplace)
            chol_res = None
            if info!=0:
                raise RuntimeError('dlauum failed with error code '+str(info))

        if inplace:
            self._internal_mat = result
            self.internal_state = REP_FISHER

        #make sure symmetric if not for internal use by FisherMatrix
        if not internal:
            result = mirror_symmetrize(result,lower=True,inplace=True)

        if copy_output:
            return result.copy()
        else:
            return result

    def get_correlation_matrix(self):
        """get the correlation matrix corresponding to the fisher matrix"""
        return np.corrcoef(self.get_covar(copy_output=False,internal=False,inplace=False))

    #TODO actually gets eigensystem with u
    def get_cov_eig_metric(self,metric):
        """get the eigensystem solving C^{ij}metric^{-1 ij}v=lambda v
            metric is itself a FisherMatrix object"""
        metric_chol_inv = metric.get_cov_cholesky_inv(copy_output=False,internal=False,inplace=False)
        #covar_use = self.get_covar()
        #use algebra trick with cholesky decompositions to get symmetric matrix with desired eigenvalues
        #mat_retrieved = np.identity(self._internal_mat.shape[0])+self.project_covar(metric_chol_inv.T).get_covar(copy_output=False)
        mat_retrieved = self.project_covar(metric_chol_inv.T).get_covar(copy_output=False,inplace=False,internal=False)
        #mat_retrieved = np.identity(covar_use.shape[0])+np.dot(np.dot(metric_chol_inv,covar_use),metric_chol_inv.T)
        eig_set = np.linalg.eigh(mat_retrieved)
        return eig_set
