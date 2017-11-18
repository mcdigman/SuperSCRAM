"""
General class for Fisher matrix and covariance matrix manipulations
"""
import numpy as np
import scipy.linalg as spl
from algebra_utils import get_inv_cholesky,invert_triangular,cholesky_inplace,get_cholesky_inv

#codes for different possible internal states of fisher matrix
#Fisher matrix
REP_FISHER = 0
#cholesky decomposition of covariance matrix
REP_CHOL = 1
#inverse of cholesky decomposition of covariance matrix
REP_CHOL_INV = 2
#covariance matrix
REP_COVAR = 3

#things to watch out for: the output matrix may mutate if it was the internal matrix

class FisherMatrix(object):
    def __init__(self,input_matrix,input_type,initial_state = None,fix_input=False,silent=True,triangle_clean=False):
        """create a fisher matrix object
            input_matrix: an input matrix
            input_type: a code for matrix type of input_matrix, options REP_FISHER, REP_CHOL, REP_CHOL_INV, or REP_COVAR
            initial_state: code for starting state, defaults to input_type
            fix_input: do not mutate input_matrix (ie make a copy)
            silent: if True, less print statements
            triangle_clean: whether to assume input matrix is already symmetric/lower triangular
        """
        self.silent=silent
        self.fix_input=fix_input
        #default initial state is not to change
        if initial_state is None:
            initial_state = input_type

        #If not allowed to change the input matrix, then copy it
        if self.fix_input:
            self.internal_mat = input_matrix.copy()
        else:
            self.internal_mat = input_matrix

        self.internal_state = input_type
        self.triangle_clean = triangle_clean

        #Switch to the specified initial state
        if not initial_state==self.internal_state:
            self.switch_rep(initial_state)
        if not self.silent:
            print("FisherMatrix "+str(id(self))+" created fisher matrix")

    def switch_rep(self,new_state):
        """Switches the internal representation of the fisher matrix to new_state
            may mutate internal_mat to save memory"""
        old_state = self.internal_state
        self.internal_mat = np.asfortranarray(self.internal_mat)
        if not self.silent:
            print "FisherMatrix "+str(id(self))+": changing internal state from "+str(self.internal_state)+" to "+str(new_state)
        if old_state==new_state:
            if not self.silent:
                print("FisherMatrix "+str(id(self))+": internal state "+str(self.internal_state)+" unchanged")
        elif new_state == REP_CHOL:
            self.internal_mat = self.get_cov_cholesky(inplace=True,copy_output=False,internal=True)
        elif new_state == REP_CHOL_INV:
            self.internal_mat = self.get_cov_cholesky_inv(inplace=True,copy_output=False,internal=True)
        elif new_state == REP_FISHER:
            self.internal_mat = self.get_fisher(inplace=True,copy_output=False,internal=True)
        elif new_state == REP_COVAR:
            self.internal_mat = self.get_covar(inplace=True,copy_output=False,internal=True)
        else:
            raise ValueError("FisherMatrix "+str(id(self))+": unrecognized new state "+str(new_state)+" when asked to switch from state "+str(old_state))
        self.internal_state = new_state
        if not self.silent:
            print "FisherMatrix "+str(id(self))+": internal state changed from "+str(self.internal_state)+" to "+str(new_state)

    def perturb_fisher(self,v):
        """add perturbation the fisher matrix in the form v^Tv"""
        if not self.silent:
            print "FisherMatrix ",id(self)," perturbing fisher matrix"
        #internal state must be fisher to add right now: could possibly be changed back
        self.switch_rep(REP_FISHER)
        self.internal_mat = spl.blas.dsyrk(1.,v,1.,self.internal_mat,overwrite_c=True,trans=True,lower=True)
        self.triangle_clean = False

    def add_fisher(self,F_n):
        """Add a fisher matrix (ie a perturbation) to the internal fisher matrix"""
        if not self.silent:
            print "FisherMatrix ",id(self)," adding fisher matrix"
        #internal state must be fisher to add right now: could possibly be changed back
        self.switch_rep(REP_FISHER)
        if F_n.__class__ is FisherMatrix:
            print "accumulating in internal state "+str(F_n.internal_state)
            self.internal_mat += F_n.get_fisher(copy_output=False,internal=True)
        else:
            print "accumulating directly"
            self.internal_mat += F_n

    def add_covar(self,C_n):
        """add a covariance matrix, C_n is numpy array  or FisherMatrix object"""
        if not self.silent:
            print "FisherMatrix ",id(self)," added covariance matrix"
        #internal state must be covar to add right now: could possibly be changed
        self.switch_rep(REP_COVAR)
        if C_n.__class__ is FisherMatrix:
            self.internal_mat += C_n.get_covar(copy_output=False,internal=True)
        else:
            self.internal_mat += C_n

    #TODO handle F_alpha_beta with 0 eigenvalues separately
    #copy_output=True should guarantee the output will never be mutated by FisherMatrix (ie copy the output matrix if necessary)
    #copy_output=False will not copy the output matrix in general, which will be more memory efficient but might cause problems if the user is not careful
    def get_covar(self,inplace=False,copy_output=False,internal=False):
        """get covariance matrix
                inputs:
                    inplace: if True, will mutate internal_mat to be the covariance matrix
                    copy_output: whether to copy the output matrix, to be safe from mutating internal_mat later"""
        copy_safe = False
        if not self.silent:
            print("FisherMatrix "+str(id(self))+": getting covariance")
        if self.internal_state==REP_COVAR: #or self.good_caches[REP_COVAR]:
            if not self.silent:
                print("FisherMatrix "+str(id(self))+": covar retrieved from cache")
            result = self.internal_mat#self.covar_cache
        else:
            if not self.silent:
                print("FisherMatrix "+str(id(self))+": covar cache miss")
            copy_safe = True
            chol_cov =self.get_cov_cholesky(inplace=inplace,internal=internal,copy_output=False)
            #dlauum calculates L^TL but have LL^T cholesky convention, rot90s flip to correct convention
            result = np.rot90(spl.lapack.dlauum(np.rot90(chol_cov,2),lower=False,overwrite_c=inplace)[0],2)
            #dlauum only gives 1 triangle of the result because symmetric, so mirror it

        #make sure guaranteed to change internal_mat if inplace was true
        if inplace:
            self.internal_mat = result
            self.internal_state = REP_COVAR

        #make sure symmetric if not for internal use by FisherMatrix
        if not internal and not self.triangle_clean:
            result = np.tril(result)
            result = result+result.T-np.diagflat(np.diag(result))

        #TODO protect internal_mat/allow override
        if copy_output and not copy_safe:
            return result.copy()
        else:
            return result

    #TODO trap return_fisher=True and identical_inputs=False, not sensible
    def contract_covar(self,v1,v2,identical_inputs=False,return_fisher=False,destructive=False):
        """calculates (v1.T).covariance.v2 for getting variance/projecting to another basis
            inputs:
                v1,v2: vectors with one dimension aligned with internal_mat
                identical_inputs: if True, assume v2=v1 and ignore v2 completely
                return_fisher: if True, return a FisherMatrix object
                destructive: if True, destroy the internal representation of self for a performance gain
        """

        if not self.silent:
            print "FisherMatrix "+str(id(self))+": contracting covariance"
        #result = cholesky_inv_contract(self.get_cov_cholesky().T,v1,v2,cholesky_given=True,identical_inputs=identical_inputs,lower=True)
        if return_fisher and not identical_inputs:
            raise ValueError('cannot get FisherMatrix object if inputs not identical')

        if  self.internal_state==REP_COVAR:
            #result = np.dot(np.dot(v1.T,self.get_covar()),v2)
            right_res = spl.blas.dsymm(1.,self.internal_mat,v2,lower=True)
            if destructive:
                self.internal_mat = None
            result = np.dot(v1.T,right_res)
        else:
            if self.internal_state == REP_FISHER:
                chol_fisher = cholesky_inplace(self.internal_mat,lower=True,inplace=destructive,clean=False)
                if destructive:
                    self.internal_mat=None
                res1 = spl.solve_triangular(chol_fisher,v1,lower=True,check_finite=False)
                if not identical_inputs:
                    res2 = spl.solve_triangular(chol_fisher,v2,lower=True,check_finite=False)
            elif self.internal_state==REP_CHOL_INV:
                res1 = spl.solve_triangular(self.internal_mat,v1,lower=True,check_finite=False,trans=True)
                if not identical_inputs:
                    res2 = spl.solve_triangular(self.internal_mat,v2,lower=True,check_finite=False,trans=True)
            elif self.internal_state==REP_CHOL:
                res1 = spl.blas.dtrmm(1.,self.internal_mat,v1,lower=True,trans_a=True)
                if not identical_inputs:
                    res2 = spl.blas.dtrmm(1.,self.internal_mat,v2,lower=True,trans_a=True)
            else:
                raise ValueError('unknown internal state '+str(self.internal_state))

            if destructive:
                self.internal_mat=None

            if identical_inputs:
                result = spl.blas.dsyrk(1.,res1,lower=True,trans=True)
                if not return_fisher:
                    result = result+result.T-np.diagflat(np.diag(result))
            else:
                result = np.dot(res1.T,res2)

        if return_fisher:
            return FisherMatrix(result,input_type=REP_COVAR,initial_state=REP_COVAR,fix_input=False,silent=self.silent,triangle_clean=False)
        else:
            return result

    #TODO maybe handle case where nearly singular when inverting in general

    #needed for projecting to another basis
    def contract_fisher(self,v1,v2,identical_inputs=False,return_fisher=False,destructive=False):
        """calculates (v1.T).Fisher matrix.v2 for getting variance/projecting to another basis
            inputs:
                v1,v2: vectors with one dimension aligned with internal_mat
                identical_inputs: if True, assume v2=v1 and ignore v2 completely
                return_fisher: if True, return a FisherMatrix object
                destructive: if True, destroy the internal representation of self for a performance gain
        """
        if not self.silent:
            print "FisherMatrix "+str(id(self))+": contracting fisher"

        if return_fisher and not identical_inputs:
            raise ValueError('cannot get FisherMatrix object if inputs not identical')

        if self.internal_state==REP_FISHER:
            #result = np.dot(v1.T,spl.blas.dsymm(1.,self.internal_mat,v2,lower=True))
            right_res = spl.blas.dsymm(1.,self.internal_mat,v2,lower=True)
            if destructive:
                self.internal_mat = None
            result = np.dot(v1.T,right_res)
        else:
            if self.internal_state==REP_COVAR:
                chol_cov = self.get_cov_cholesky(copy_output=False,inplace=destructive,internal=True)
                if destructive:
                    self.internal_mat=None
                res1 = spl.solve_triangular(chol_cov,v1,lower=True,check_finite=False)
                if not identical_inputs:
                    res2 = spl.solve_triangular(chol_cov,v2,lower=True,check_finite=False)
            elif self.internal_state==REP_CHOL_INV:
                res1 = spl.blas.dtrmm(1.,self.internal_mat,v1,lower=True,trans_a=False)
                if not identical_inputs:
                    res2 = spl.blas.dtrmm(1.,self.internal_mat,v2,lower=True,trans_a=False)
            elif self.internal_state==REP_CHOL:
                res1 = spl.solve_triangular(self.internal_mat,v1,lower=True,check_finite=False)
                if not identical_inputs:
                    res2 = spl.solve_triangular(self.internal_mat,v2,lower=True,check_finite=False)
            else:
                raise ValueError('unknown internal state '+str(self.internal_state))

            if destructive:
                self.internal_mat = None

            if identical_inputs:
                result = spl.blas.dsyrk(1.,res1,lower=True,trans=True)
                if not return_fisher:
                    result = result+result.T-np.diagflat(np.diag(result))
            else:
                result = np.dot(res1.T,res2)
        if destructive:
            self.internal_mat = None


        if return_fisher:
            return FisherMatrix(result,input_type=REP_FISHER,initial_state=REP_FISHER,fix_input=False,silent=self.silent,triangle_clean=False)
        else:
            return result

    def project_fisher(self,v1,destructive=False):
        """project using (v1.T).Fisher.v1"""
        result = self.contract_fisher(v1,v1,identical_inputs=True,return_fisher=False,destructive=destructive)
        #assert(np.all(result==result.T))
        #symmetrize to avoid accumulating numerical discrepancies
        #TODO add option to not symmetrize, or don't do it for known safe internal states
        #if not (self.internal_state==REP_CHOL or self.internal_state==REP_CHOL_INV):
        #    result = (result+result.T)/2.
        return FisherMatrix(result,input_type=REP_FISHER,initial_state=REP_FISHER,fix_input=False,silent=self.silent,triangle_clean=False)

    #project using covar
    def project_covar(self,v1,destructive=False):
        """project using (v1.T).Covariance.v1"""
        result = self.contract_covar(v1,v1,identical_inputs=True,return_fisher=True,destructive=destructive)
        return result
        #symmetrize to avoid accumulating numerical discrepancies
        #if not (self.internal_state==REP_CHOL or self.internal_state==REP_CHOL_INV):
        #    result = (result+result.T)/2.
        #return FisherMatrix(result,input_type=REP_COVAR,initial_state=REP_COVAR,fix_input=False,silent=self.silent,triangle_clean=False)

    #TODO unused, maybe eliminate
    def contract_chol_right(self,v):
        """contract v with the cholesky decomposition of the covariance, L^T.v"""
        if not self.silent:
            print "FisherMatrix "+str(id(self))+": getting right chol contraction"
        return np.dot(self.get_cov_cholesky(copy_output=False).T,v)

    def contract_chol_inv_right(self,v):
        """contract v with the inverse cholesky decomposition of the covariance, L^{-1}.v"""
        if not self.silent:
            print "FisherMatrix "+str(id(self))+": getting right chol inv contraction"
        return np.dot(self.get_cov_cholesky_inv(copy_output=False),v)

    #this is not actually fisher cholesky, but the cholesky decomposition of the inverse of the covariance, which is not exactly the same ie LL^T vs L^TL decompositions
    #TODO inplace should never be used unless internal, copy_output maybe also redundant
    def get_cov_cholesky_inv(self,inplace=False,copy_output=False,internal=False):
        """get inverse of lower triangular cholesky decomposition of covariance
                inputs:
                    inplace: if True, will mutate internal_mat to be the inverse cholesky decompostion of the covariance matrix
                    copy_output: whether to copy the output matrix, to be safe from mutating internal_mat later"""
        copy_safe = False
        if self.internal_state==REP_CHOL_INV:
            if not self.silent:
                print "FisherMatrix ",id(self)," cholesky decomposition inv retrieved from cache"
            result = self.internal_mat
        else:
            if not self.silent:
                print "FisherMatrix ",id(self)," cholesky decomposition inv cache miss"
            copy_safe = True
            if self.internal_state == REP_CHOL:
                result = invert_triangular(self.internal_mat,lower=True,inplace=inplace)
            elif self.internal_state == REP_FISHER:
                result = np.asfortranarray(np.rot90(cholesky_inplace(np.asfortranarray(np.rot90(self.internal_mat,2)),inplace=inplace,lower=False,clean=not internal),2))
                if internal and inplace:
                    self.triangle_clean=False
            elif self.internal_state == REP_COVAR:
                result = get_inv_cholesky(self.internal_mat,lower=True,inplace=inplace,clean=not internal)
            else:
                raise ValueError("FisherMatrix "+str(id(self))+": unrecognized internal state "+str(self.internal_state))

        if inplace:
            self.internal_mat = result
            self.internal_state = REP_CHOL_INV

        if not internal and not self.triangle_clean:
            result = np.tril(result)

        if  copy_output and not copy_safe:
            return result.copy()
        else:
            return result

    def get_cov_cholesky(self,inplace=False,copy_output=False,internal=False):
        """get lower triangular cholesky decomposition of covariance
                inputs:
                    inplace: if True, will mutate internal_mat to be the cholesky decompostion of the covariance matrix
                    copy_output: whether to copy the output matrix, to be safe from mutating internal_mat later"""
        copy_safe = False
        if self.internal_state==REP_CHOL:
            if not self.silent:
                print "FisherMatrix ",str(id(self))+": cholesky decomposition retrieved from cache, size: "+str(self.internal_mat.nbytes/10**6)+" megabytes"
            result = self.internal_mat
        else:
            if not self.silent:
                print "FisherMatrix "+str(id(self)),": cholesky decomposition cache miss"
            copy_safe = True
            #TODO: evaluate prioritization of ways to find the decomposition, esp. for numerical stability versus speed/memory consumption (cholesky may be more numerically stable than inv)
            if self.internal_state == REP_CHOL_INV :
                if not self.silent:
                    print "FisherMatrix "+str(id(self)),": getting cholesky decomposition of covariance matrix from its inverse, size: "+str(self.internal_mat.nbytes/10**6)+" megabytes"
                result = invert_triangular(self.internal_mat,lower=True,inplace=inplace)
            elif self.internal_state == REP_COVAR:
                if not self.silent:
                    print "FisherMatrix "+str(id(self)),": getting cholesky decomposition of covariance matrix directly, size: "+str(self.internal_mat.nbytes/10**6)+" megabytes"
                result = cholesky_inplace(self.internal_mat,inplace=inplace,lower=True,clean=not internal)
                if internal and inplace:
                    self.triangle_clean=False
            elif self.internal_state == REP_FISHER:
                if not self.silent:
                    print "FisherMatrix "+str(id(self)),": getting cholesky decomposition of covariance matrix from F_alpha_beta, size: "+str(self.internal_mat.nbytes/10**6)+" megabytes"
                result = get_cholesky_inv(self.internal_mat,lower=True,inplace=inplace,clean=not internal)
            else:
                raise ValueError("FisherMatrix "+str(id(self))+": unrecognized internal state "+str(self.internal_state))

            if not self.silent:
                print "FisherMatrix "+str(id(self)),": found cholesky decomposition of covariance matrix, size: "+str(result.nbytes/10**6)+" megabytes"

        if inplace:
            self.internal_mat = result
            self.internal_state = REP_CHOL

        #if not being requested internally to FisherMatrix, make sure the upper triangle is all zeros
        if not internal and not self.triangle_clean:
            result = np.tril(result)

        #TODO default correctly on copy_output
        if copy_output and not copy_safe:
            return result.copy()
        else:
            return result

    def get_fisher(self,copy_output=False,inplace=False,internal=False):
        """get fisher matrix
                inputs:
                    inplace: if True, will mutate internal_mat to be the fisher matrix
                    copy_output: whether to copy the output matrix, to be safe from mutating internal_mat later"""
        copy_safe = False
        if self.internal_state == REP_FISHER:
            if not self.silent:
                print "FisherMatrix "+str(id(self))+": retrieved fisher matrix from cache"
            result = self.internal_mat
        elif self.internal_state == REP_COVAR or self.internal_state==REP_CHOL:
            result = spl.lapack.dpotri(self.get_cov_cholesky(copy_output=False,internal=internal,inplace=inplace),lower=True,overwrite_c=inplace)[0]
        else:
            if not self.silent:
                print "FisherMatrix "+str(id(self))+": fisher matrix cache miss"
            #TODO check efficiency/arguments
            chol_res = self.get_cov_cholesky_inv(copy_output=False,internal=internal,inplace=inplace)
            #TODO careful with mirroring the triangle matrix from dlauum/think about if can safely just use lower triangle
            result = spl.lapack.dlauum(chol_res,lower=True,overwrite_c=inplace)[0]

        if inplace:
            self.internal_mat = result
            self.internal_state = REP_FISHER

        #make sure symmetric if not for internal use by FisherMatrix
        if not internal and not self.triangle_clean:
            result = np.tril(result)
            result = result+result.T-np.diagflat(np.diag(result))

        if copy_output and not copy_safe:
            return result.copy()
        else:
            return result

    def get_correlation_matrix(self):
        return np.corrcoef(self.get_covar(copy_output=False,internal=False))

    #TODO actually gets eigensystem with u
    def get_cov_eig_metric(self,metric):
        """get the eigensystem solving C^{ij}metric^{-1 ij}v=lambda v
            metric is itself a FisherMatrix object"""
        metric_chol_inv = metric.get_cov_cholesky_inv(copy_output=False,internal=False)
        #covar_use = self.get_covar()
        #use algebra trick with cholesky decompositions to get symmetric matrix with desired eigenvalues
        #mat_retrieved = np.identity(self.internal_mat.shape[0])+self.project_covar(metric_chol_inv.T).get_covar(copy_output=False)
        mat_retrieved = self.project_covar(metric_chol_inv.T).get_covar(copy_output=False)
        #mat_retrieved = (mat_retrieved+mat_retrieved.T)/2.
        #mat_retrieved = np.identity(covar_use.shape[0])+np.dot(np.dot(metric_chol_inv,covar_use),metric_chol_inv.T)
        eig_set = np.linalg.eigh(mat_retrieved)
        return eig_set
