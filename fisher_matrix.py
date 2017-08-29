import numpy as np
import scipy as sp
import scipy.linalg
from algebra_utils import ch_inv,cholesky_inv_contract,get_inv_cholesky,invert_triangular,get_mat_from_inv_cholesky,cholesky_inplace,get_cholesky_inv
from warnings import warn
REP_FISHER = 0
REP_CHOL = 1
REP_CHOL_INV = 2 
REP_COVAR = 3

#things to watch out for: the output matrix may mutate if it was the internal matrix


class fisher_matrix:


    def __init__(self,input_matrix,input_type=REP_FISHER,initial_state = REP_FISHER,fix_input=False,silent=True):

        self.silent=silent
        self.fix_input=fix_input
        #If not allowed to change the input matrix, then copy it 
        if self.fix_input:
            self.internal_mat = input_matrix.copy()
        else:
            self.internal_mat = input_matrix 
    
        self.internal_state = input_type

        #Switch to the specified initial state
        self.switch_rep(initial_state)
        if not self.silent:
            print("fisher_matrix "+str(id(self))+" created fisher matrix")
    
    #Switches the internal representation of the fisher matrix
    def switch_rep(self,new_state):
        old_state = self.internal_state
        if not self.silent:
            print "fisher matrix "+str(id(self))+": changing internal state from "+str(self.internal_state)+" to "+str(new_state)
        if old_state==new_state:
            if not self.silent:
                print("fisher_matrix "+str(id(self))+": internal state "+str(self.internal_state)+" unchanged")
        elif new_state == REP_CHOL:
            self.internal_mat = self.get_cov_cholesky(inplace=True)
        elif new_state == REP_CHOL_INV:
            self.internal_mat = self.get_cov_cholesky_inv(inplace=True)
        elif new_state == REP_FISHER:
            self.internal_mat = self.get_fisher(inplace=True)
        elif new_state == REP_COVAR: 
            self.internal_mat = self.get_covar(inplace=True)
        else:
            raise ValueError("fisher_matrix "+str(id(self))+": unrecognized new state "+str(new_state)+" when asked to switch from state "+str(old_state))
        self.internal_state = new_state
        if not self.silent:
            print "fisher matrix "+str(id(self))+": internal state changed from "+str(self.internal_state)+" to "+str(new_state)

    #Add a fisher matrix (ie a perturbation) to the internal fisher matrix
    def add_fisher(self,F_n):
        if not self.silent:
            print "fisher_matrix ",id(self)," added fisher matrix"
        #internal state must be fisher to add right now: could possibly be changed
        self.switch_rep(REP_FISHER)
        if F_n.__class__ is fisher_matrix:
            self.internal_mat += F_n.get_fisher()
        else:
            self.internal_mat += F_n

    #add a covariance matrix, C_n is numpy array  or fisher_matrix object
    def add_covar(self,C_n):
        if not self.silent:
            print "fisher_matrix ",id(self)," added covariance matrix"
        #internal state must be covar to add right now: could possibly be changed
        self.switch_rep(REP_COVAR)
        if C_n.__class__ is fisher_matrix:
            self.internal_mat += C_n.get_covar()
        else:
            self.internal_mat += C_n

    #TODO handle F_alpha_beta with 0 eigenvalues separately
    #copy_output=True should guarantee the output will never be mutated by fisher_matrix (ie copy the output matrix if necessary)
    #copy_output=False will not copy the output matrix in general, which will be more memory efficient but might cause problems if the user is not careful
    def get_covar(self,inplace=False,copy_output=False):
        copy_safe = False
        if not self.silent:
            print("fisher_matrix "+str(id(self))+": getting covariance")
        if self.internal_state==REP_COVAR: #or self.good_caches[REP_COVAR]:
            if not self.silent:
                print("fisher_matrix "+str(id(self))+": covar retrieved from cache")
            result = self.internal_mat#self.covar_cache
        else:
            if not self.silent:
                print("fisher_matrix "+str(id(self))+": covar cache miss")
            copy_safe = True
            result = ch_inv(self.get_cov_cholesky(),cholesky_given=True,lower=False)

        #TODO protect internal_mat/allow override
        if copy_output and not copy_safe:
            return result.copy()
        else: 
            return result
        
    #for getting variance/projecting to another basis
    #TODO consider making sensitive to internal state
    def contract_covar(self,v1,v2,identical_inputs=False,return_fisher=False):
        if not self.silent:
            print "fisher_matrix "+str(id(self))+": contracting covariance"
        result = cholesky_inv_contract(self.get_cov_cholesky().T,v1,v2,cholesky_given=True,identical_inputs=identical_inputs)
        if return_fisher:
            return fisher_matrix(result,input_type=REP_COVAR,initial_state=REP_COVAR,fix_input=False,silent=self.silent)
        else:
            return result

    #needed for projecting to another basis
    #TODO consider making sensitive to internal state
    def contract_fisher(self,v1,v2,identical_inputs=False,return_fisher=False):
        if not self.silent:
            print "fisher_matrix "+str(id(self))+": contracting fisher"

        result = np.dot(v1.T,np.dot(self.get_fisher(),v2))
        if return_fisher:
            return fisher_matrix(result,input_type=REP_FISHER,initial_state=REP_FISHER,fix_input=False,silent=self.silent)
        else:
            return result

    #project using fisher
    def project_fisher(self,v1):
        return self.contract_fisher(v1,v1,identical_inputs=True,return_fisher=True)

    #project using covar
    def project_covar(self,v1):
        return self.contract_covar(v1,v1,identical_inputs=True,return_fisher=True)
    #for use in sw_survey
    #contract v with the cholesky decomposition of the covariance, L^T.v
    def contract_chol_right(self,v):
        if not self.silent:
            print "fisher_matrix "+str(id(self))+": getting right chol contraction"
        return np.dot(self.get_cov_cholesky().T,v)
   
    #contract v with the inverse cholesky decomposition of the covariance, L^{-1}.v
    def contract_chol_inv_right(self,v):
        if not self.silent:
            print "fisher_matrix "+str(id(self))+": getting right chol inv contraction"
        return np.dot(self.get_cov_cholesky_inv(),v)

    #this is not actually fisher cholesky, but the cholesky decomposition of the inverse of the covariance, which is not exactly the same ie LL^T vs L^TL decompositions
    def get_cov_cholesky_inv(self,inplace=False,copy_output=False):
        copy_safe = False
        if self.internal_state==REP_CHOL_INV: 
            if not self.silent:
                print "fisher_matrix ",id(self)," cholesky decomposition inv retrieved from cache"
            result = self.internal_mat
        else:
            if not self.silent:
                print "fisher_matrix ",id(self)," cholesky decomposition inv cache miss"
            copy_safe = True
            if self.internal_state == REP_CHOL:
                result = invert_triangular(self.internal_mat)
            elif self.internal_state == REP_FISHER:
                result = np.rot90(cholesky_inplace(np.rot90(self.internal_mat,2),inplace=False,lower=False),2)
            elif self.internal_state == REP_COVAR:
                result = get_inv_cholesky(self.internal_mat)
            else:
                raise ValueError("fisher_matrix "+str(id(self))+": unrecognized internal state "+str(self.internal_state))


        if  copy_output and not copy_safe:
            return result.copy()
        else:
            return result

    def get_cov_cholesky(self,inplace=False,copy_output=False):
        copy_safe = False
        if self.internal_state==REP_CHOL:
            if not self.silent:
                print "fisher_matrix ",str(id(self))+": cholesky decomposition retrieved from cache, size: "+str(self.internal_mat.nbytes/10**6)+" megabytes"
            result = self.internal_mat
        else:
            if not self.silent:
                print "fisher_matrix "+str(id(self)),": cholesky decomposition cache miss"
            copy_safe = True
            #TODO: evaluate prioritization of ways to find the decomposition, esp. for numerical stability versus speed/memory consumption (cholesky may be more numerically stable than inv)
            if self.internal_state == REP_CHOL_INV :
                if not self.silent:
                    print "fisher_matrix "+str(id(self)),": getting cholesky decomposition of covariance matrix from its inverse, size: "+str(self.internal_mat.nbytes/10**6)+" megabytes"
                result = invert_triangular(self.internal_mat)
            elif self.internal_state == REP_COVAR:
                if not self.silent:
                    print "fisher_matrix "+str(id(self)),": getting cholesky decomposition of covariance matrix directly, size: "+str(self.internal_mat.nbytes/10**6)+" megabytes"
                result = cholesky_inplace(self.internal_mat,inplace=inplace)
            elif self.internal_state == REP_FISHER:
                if not self.silent:
                    print "fisher_matrix "+str(id(self)),": getting cholesky decomposition of covariance matrix from F_alpha_beta, size: "+str(self.internal_mat.nbytes/10**6)+" megabytes"
                result = get_cholesky_inv(self.internal_mat)
            else:
                raise ValueError("fisher_matrix "+str(id(self))+": unrecognized internal state "+str(self.internal_state))

            if not self.silent:
                print "fisher_matrix "+str(id(self)),": found cholesky decomposition of covariance matrix, size: "+str(result.nbytes/10**6)+" megabytes"

        #TODO default correctly on copy_output
        if copy_output and not copy_safe:
            return result.copy()
        else:
            return result


    def get_fisher(self,copy_output=False,inplace=False):
        copy_safe = False 
        if self.internal_state == REP_FISHER:
            if not self.silent:
                print "fisher_matrix "+str(id(self))+": retrieved fisher matrix from cache"
            result = self.internal_mat
        else:
            if not self.silent:
                print "fisher_matrix "+str(id(self))+": fisher matrix cache miss"
            #TODO check efficiency/arguments
            chol_res = self.get_cov_cholesky_inv(copy_output=False).T
            result = np.dot(chol_res,chol_res.T)
            
        if copy_output and not copy_safe:
            return result.copy()
        else:
            return result

