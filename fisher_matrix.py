import numpy as np
import scipy as sp
import scipy.linalg as spl
from algebra_utils import cholesky_inv_contract,get_inv_cholesky,invert_triangular,cholesky_inplace,get_cholesky_inv,cholesky_contract
from warnings import warn
import traceback
import sys
REP_FISHER = 0
REP_CHOL = 1
REP_CHOL_INV = 2 
REP_COVAR = 3

#things to watch out for: the output matrix may mutate if it was the internal matrix

class FisherMatrix:
    def __init__(self,input_matrix,input_type,initial_state = None,fix_input=False,silent=True):
        
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

        #Switch to the specified initial state
        if not initial_state==self.internal_state:
            self.switch_rep(initial_state)
        if not self.silent:
            print("FisherMatrix "+str(id(self))+" created fisher matrix")
    
    #Switches the internal representation of the fisher matrix
    def switch_rep(self,new_state):
        old_state = self.internal_state
        if not self.silent:
            print "FisherMatrix "+str(id(self))+": changing internal state from "+str(self.internal_state)+" to "+str(new_state)
        if old_state==new_state:
            if not self.silent:
                print("FisherMatrix "+str(id(self))+": internal state "+str(self.internal_state)+" unchanged")
        elif new_state == REP_CHOL:
            self.internal_mat = self.get_cov_cholesky(inplace=True,copy_output=False)
        elif new_state == REP_CHOL_INV:
            self.internal_mat = self.get_cov_cholesky_inv(inplace=True,copy_output=False)
        elif new_state == REP_FISHER:
            self.internal_mat = self.get_fisher(inplace=True,copy_output=False)
        elif new_state == REP_COVAR: 
            self.internal_mat = self.get_covar(inplace=True,copy_output=False)
        else:
            raise ValueError("FisherMatrix "+str(id(self))+": unrecognized new state "+str(new_state)+" when asked to switch from state "+str(old_state))
        self.internal_state = new_state
        if not self.silent:
            print "FisherMatrix "+str(id(self))+": internal state changed from "+str(self.internal_state)+" to "+str(new_state)

    #Add a fisher matrix (ie a perturbation) to the internal fisher matrix
    def add_fisher(self,F_n):
        if not self.silent:
            print "FisherMatrix ",id(self)," added fisher matrix"
        #internal state must be fisher to add right now: could possibly be changed
        self.switch_rep(REP_FISHER)
        if F_n.__class__ is FisherMatrix:
            self.internal_mat += F_n.get_fisher(copy_output=False)
        else:
            self.internal_mat += F_n

    #add a covariance matrix, C_n is numpy array  or FisherMatrix object
    def add_covar(self,C_n):
        if not self.silent:
            print "FisherMatrix ",id(self)," added covariance matrix"
        #internal state must be covar to add right now: could possibly be changed
        self.switch_rep(REP_COVAR)
        if C_n.__class__ is FisherMatrix:
            self.internal_mat += C_n.get_covar(copy_output=False)
        else:
            self.internal_mat += C_n

    #TODO handle F_alpha_beta with 0 eigenvalues separately
    #copy_output=True should guarantee the output will never be mutated by FisherMatrix (ie copy the output matrix if necessary)
    #copy_output=False will not copy the output matrix in general, which will be more memory efficient but might cause problems if the user is not careful
    def get_covar(self,inplace=False,copy_output=False):
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
            chol_cov =self.get_cov_cholesky(inplace=inplace,copy_output=False)
            #dlauum calculates L^TL but have LL^T cholesky convention, rot90s flip to correct convention
            result = np.rot90(spl.lapack.dlauum(np.rot90(chol_cov,2),lower=False,overwrite_c=inplace)[0],2)
            #dlauum only gives 1 triangle of the result because symmetric, so mirror it
            result += result.T-np.diagflat(np.diag(result))
            #result = np.dot(chol_cov,chol_cov.T)
            #result = (result+result.T)/2.

        #TODO protect internal_mat/allow override
        if copy_output and not copy_safe:
            return result.copy()
        else: 
            return result
        
    #for getting variance/projecting to another basis
    def contract_covar(self,v1,v2,identical_inputs=False,return_fisher=False):
        if not self.silent:
            print "FisherMatrix "+str(id(self))+": contracting covariance"
        #result = cholesky_inv_contract(self.get_cov_cholesky().T,v1,v2,cholesky_given=True,identical_inputs=identical_inputs,lower=True)
        if  self.internal_state==REP_COVAR:
            #result = np.dot(np.dot(v1.T,self.get_covar()),v2)
            result = np.dot(spl.blas.dsymm(1.,self.get_covar(copy_output=False),v1.T,side=True,lower=True),v2)
            
            #TODO commented out logic does work and may be faster/better but may have trouble when matrix nearly singular
            #althouth may just be ignoring that problem
        elif self.internal_state==REP_FISHER or self.internal_state==REP_CHOL_INV:
        #    result = np.dot(spl.solve(self.internal_mat,v1,sym_pos=True).T,v2)
        #elif self.internal_state==REP_CHOL_INV:
            chol_inv = self.get_cov_cholesky_inv(copy_output=False)
            res1 = spl.solve_triangular(chol_inv.T,v1,lower=False)
            if identical_inputs:
                result = np.dot(res1.T,res1)
            else:
                res2 = spl.solve_triangular(chol_inv.T,v2,lower=False)
                result = np.dot(res1.T,res2)
        else:
            result = cholesky_contract(self.get_cov_cholesky(copy_output=False),v1,v2,cholesky_given=True,identical_inputs=identical_inputs,lower=True)

        if return_fisher:
            return FisherMatrix(result,input_type=REP_COVAR,initial_state=REP_COVAR,fix_input=False,silent=self.silent)
        else:
            return result

    #TODO maybe handle case where nearly singular when inverting in general

    #needed for projecting to another basis
    def contract_fisher(self,v1,v2,identical_inputs=False,return_fisher=False):
        if not self.silent:
            print "FisherMatrix "+str(id(self))+": contracting fisher"

        if self.internal_state==REP_FISHER:
            result = np.dot(v1.T,spl.blas.dsymm(1.,self.get_fisher(copy_output=False),v2,lower=True))
        elif self.internal_state == REP_COVAR or self.internal_state==REP_CHOL:
            #result = np.dot(spl.solve(self.internal_mat,v1,sym_pos=True).T,v2)
            #result = cholesky_inv_contract(self.get_cov_cholesky_inv(),v1,v2,cholesky_given=True,identical_inputs=identical_inputs,lower=True)
            chol_cov = self.get_cov_cholesky(copy_output=False)
            res1 = spl.solve_triangular(chol_cov,v1,lower=True)
            if identical_inputs:
                result = np.dot(res1.T,res1)
            else:
                res2 = spl.solve_triangular(chol_cov,v2,lower=True)
                result = np.dot(res1.T,res2)
            #result = cholesky_inv_contract(self.get_cov_cholesky_inv(),v1,v2,cholesky_given=True,identical_inputs=identical_inputs,lower=True)
        else:
            result = cholesky_inv_contract(self.get_cov_cholesky_inv(copy_output=False),v1,v2,cholesky_given=True,identical_inputs=identical_inputs,lower=True)

        if return_fisher:
            return FisherMatrix(result,input_type=REP_FISHER,initial_state=REP_FISHER,fix_input=False,silent=self.silent)
        else:
            return result

    #project using fisher
    def project_fisher(self,v1):
        result = self.contract_fisher(v1,v1,identical_inputs=True,return_fisher=False)
        #assert(np.all(result==result.T))
        #symmetrize to avoid accumulating numerical discrepancies
        result = (result+result.T)/2.
        return FisherMatrix(result,input_type=REP_FISHER,initial_state=REP_FISHER,fix_input=False,silent=self.silent)

    #project using covar
    def project_covar(self,v1):
        result = self.contract_covar(v1,v1,identical_inputs=True,return_fisher=False)
        #symmetrize to avoid accumulating numerical discrepancies
        result = (result+result.T)/2.
        return FisherMatrix(result,input_type=REP_COVAR,initial_state=REP_COVAR,fix_input=False,silent=self.silent)
    #for use in sw_survey
    #contract v with the cholesky decomposition of the covariance, L^T.v
    #TODO maybe eliminate
    def contract_chol_right(self,v):
        if not self.silent:
            print "FisherMatrix "+str(id(self))+": getting right chol contraction"
        return np.dot(self.get_cov_cholesky(copy_output=False).T,v)
   
    #contract v with the inverse cholesky decomposition of the covariance, L^{-1}.v
    def contract_chol_inv_right(self,v):
        if not self.silent:
            print "FisherMatrix "+str(id(self))+": getting right chol inv contraction"
        return np.dot(self.get_cov_cholesky_inv(copy_output=False),v)

    #this is not actually fisher cholesky, but the cholesky decomposition of the inverse of the covariance, which is not exactly the same ie LL^T vs L^TL decompositions
    def get_cov_cholesky_inv(self,inplace=False,copy_output=False):
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
                result = np.rot90(cholesky_inplace(np.asfortranarray(np.rot90(self.internal_mat,2)),inplace=inplace,lower=False),2)
            elif self.internal_state == REP_COVAR:
                result = get_inv_cholesky(self.internal_mat,lower=True,inplace=inplace)
            else:
                raise ValueError("FisherMatrix "+str(id(self))+": unrecognized internal state "+str(self.internal_state))


        if  copy_output and not copy_safe:
            return result.copy()
        else:
            return result

    def get_cov_cholesky(self,inplace=False,copy_output=False):
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
                result = cholesky_inplace(self.internal_mat,inplace=inplace,lower=True)
            elif self.internal_state == REP_FISHER:
                if not self.silent:
                    print "FisherMatrix "+str(id(self)),": getting cholesky decomposition of covariance matrix from F_alpha_beta, size: "+str(self.internal_mat.nbytes/10**6)+" megabytes"
                result = get_cholesky_inv(self.internal_mat,lower=True,inplace=inplace)
            else:
                raise ValueError("FisherMatrix "+str(id(self))+": unrecognized internal state "+str(self.internal_state))

            if not self.silent:
                print "FisherMatrix "+str(id(self)),": found cholesky decomposition of covariance matrix, size: "+str(result.nbytes/10**6)+" megabytes"

        #TODO default correctly on copy_output
        if copy_output and not copy_safe:
            return result.copy()
        else:
            return result

    def get_fisher(self,copy_output=False,inplace=False):
        copy_safe = False 
        if self.internal_state == REP_FISHER:
            if not self.silent:
                print "FisherMatrix "+str(id(self))+": retrieved fisher matrix from cache"
            result = self.internal_mat
        elif self.internal_state == REP_COVAR or self.internal_state==REP_CHOL:
            result = spl.lapack.dpotri(self.get_cov_cholesky(copy_output=False),lower=True)[0]
            result += result.T-np.diagflat(np.diag(result))
        else:
            if not self.silent:
                print "FisherMatrix "+str(id(self))+": fisher matrix cache miss"
            #TODO check efficiency/arguments
            chol_res = self.get_cov_cholesky_inv(copy_output=False)
            #TODO careful with mirroring the triangle matrix from dlauum/think about if can safely just use lower triangle
            result = spl.lapack.dlauum(chol_res,lower=True,overwrite_c=inplace)[0]
            result += result.T-np.diagflat(np.diag(result))
            #result = np.dot(chol_res,chol_res.T)
        if copy_output and not copy_safe:
            return result.copy()
        else:
            return result

    #get the eigenvalues solving C^{ij}metric^{-1 ij}v=lambda v
    #metric is itself a FisherMatrix object
    def get_cov_eig_metric(self,metric):
        metric_chol_inv = metric.get_cov_cholesky_inv(copy_output=False)
        #covar_use = self.get_covar()
        #use algebra trick with cholesky decompositions to get symmetric matrix with desired eigenvalues
        mat_retrieved = np.identity(self.internal_mat.shape[0])+self.project_covar(metric_chol_inv.T).get_covar()
        mat_retrieved = (mat_retrieved+mat_retrieved.T)/2.
        #mat_retrieved = np.identity(covar_use.shape[0])+np.dot(np.dot(metric_chol_inv,covar_use),metric_chol_inv.T)
        eig_set = np.linalg.eigh(mat_retrieved)
        return eig_set

