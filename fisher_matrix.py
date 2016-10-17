import numpy as np
import scipy as sp
import scipy.linalg
from algebra_utils import ch_inv,cholesky_inv_contract,get_inv_cholesky,invert_triangular,get_mat_from_inv_cholesky,cholesky_inplace
from warnings import warn
REP_FISHER = 0
REP_CHOL = 1
REP_CHOL_INV = 2 
REP_COVAR = 3
DEFAULT_ALLOWED_CACHES = {REP_FISHER:False,REP_CHOL:False,REP_CHOL_INV:False,REP_COVAR:False}

#things to watch out for: the output matrix may mutate if it was the internal matrix


class fisher_matrix:


    def __init__(self,input_matrix,input_type=REP_FISHER,initial_state = REP_FISHER,allowed_caches=DEFAULT_ALLOWED_CACHES,fix_input=False):

        self.fix_input=fix_input
        #If not allowed to change the input matrix, then copy it 
        if self.fix_input:
            self.internal_mat = input_matrix.copy()
        else:
            self.internal_mat = input_matrix 
    
        self.internal_state = input_type

        #Switch to the specified initial state
        self.switch_rep(initial_state)
        print("fisher_matrix "+str(id(self))+" created fisher matrix")
    
    #Switches the internal representation of the fisher matrix
    def switch_rep(self,new_state):
        old_state = self.internal_state
        print "fisher matrix "+str(id(self))+": changing internal state from "+str(self.internal_state)+" to "+str(new_state)
        if old_state==new_state:
            print("fisher_matrix "+str(id(self))+": internal state "+str(self.internal_state)+" unchanged")
        elif new_state == REP_CHOL:
            self.internal_mat = self.get_cov_cholesky(inplace=True)
        elif new_state == REP_CHOL_INV:
            self.internal_mat = self.get_fab_cholesky(inplace=True)
        elif new_state == REP_FISHER:
            self.internal_mat = self.get_F_alpha_beta(inplace=True)
        elif new_state == REP_COVAR: 
            self.internal_mat = self.get_covar(inplace=True)
        else:
            raise ValueError("fisher_matrix "+str(id(self))+": unrecognized new state "+str(new_state)+" when asked to switch from state "+str(old_state))
        self.internal_state = new_state
        print "fisher matrix "+str(id(self))+": internal state changed from "+str(self.internal_state)+" to "+str(new_state)

    #Add a fisher matrix (ie a perturbation) to the internal fisher matrix
    def add_fisher(self,F_n):
        print "fisher_matrix ",id(self)," added fisher matrix"
        #internal state must be fisher to add right now: could possibly be changed
        self.switch_rep(REP_FISHER)
        #TODO check this is ok with views
        self.F_alpha_beta += F_n

    #TODO handle F_alpha_beta with 0 eigenvalues separately
    #copy_output=True should guarantee the output will never be mutated by fisher_matrix (ie copy the output matrix if necessary)
    #copy_output=False will not copy the output matrix in general, which will be more memory efficient but might cause problems if the user is not careful
    def get_covar(self,inplace=False,copy_output=False):
        copy_safe = False
        print("fisher_matrix "+str(id(self))+": getting covariance")
        if self.internal_state==REP_COVAR: #or self.good_caches[REP_COVAR]:
            print("fisher_matrix "+str(id(self))+": covar retrieved from cache")
            result = self.internal_mat#self.covar_cache
        else:
            print("fisher_matrix "+str(id(self))+": covar cache miss")
            copy_safe = True
            result = ch_inv(self.get_cov_cholesky(),cholesky_given=True)

        #TODO protect internal_mat/allow override
        if copy_output and not copy_safe:
            return result.copy()
        else: 
            return result
        
   
    #for getting variance
    def contract_covar(self,v1,v2,identical_inputs=False):
        print "fisher_matrix "+str(id(self))+": contracting covariance"
        return cholesky_inv_contract(self.get_cov_cholesky(),v1,v2,cholesky_given=True,identical_inputs=identical_inputs)
   

    def get_fab_cholesky(self,inplace=False,copy_output=False):
        copy_safe = False
        if self.internal_state==REP_CHOL_INV: #or self.good_caches[REP_CHOL_INV]:
            print "fisher_matrix ",id(self)," cholesky decomposition inv retrieved from cache"
            result = self.internal_mat
        else:
            print "fisher_matrix ",id(self)," cholesky decomposition inv cache miss"
            copy_safe = True
            if self.internal_state == REP_CHOL:# or self.good_caches[REP_CHOL]:
                result = invert_triangular(self.internal_mat)
            elif self.internal_state == REP_FISHER: #or self.good_caches[REP_FISHER]:
                #TODO use cholesky_inplace as appropriate
                result = np.linalg.cholesky(self.internal_mat)
            elif self.internal_state == REP_COVAR:# or self.good_caches[REP_COVAR]:
                result = get_inv_cholesky(self.internal_mat).T
            else:
                raise ValueError("fisher_matrix "+str(id(self))+": unrecognized internal state "+str(self.internal_state))


            if  copy_output and not copy_safe:
                return result.copy()
            else:
                return result

    def get_cov_cholesky(self,inplace=False,copy_output=False):
        copy_safe = False
        if self.internal_state==REP_CHOL: #or self.good_caches[REP_CHOL]:
            print "fisher_matrix ",str(id(self))+": cholesky decomposition retrieved from cache, size: "+str(self.internal_mat.nbytes/10**6)+" megabytes"
            result = self.internal_mat
        else:
            print "fisher_matrix "+str(id(self)),": cholesky decomposition cache miss"
            copy_safe = True
            #TODO: evaluate prioritization of ways to find the decomposition, esp. for numerical stability versus speed/memory consumption (cholesky may be more numerically stable than inv)
            if self.internal_state == REP_CHOL_INV :#or self.good_caches[REP_CHOL_INV]:
                print "fisher_matrix "+str(id(self)),": getting cholesky decomposition of covariance matrix from its inverse, size: "+str(self.internal_mat.nbytes/10**6)+" megabytes"
                result = invert_trianglular(self.internal_mat)
            elif self.internal_state == REP_COVAR:# or self.good_caches[REP_COVAR]:
                print "fisher_matrix "+str(id(self)),": getting cholesky decomposition of covariance matrix directly, size: "+str(self.internal_mat.nbytes/10**6)+" megabytes"
                result = cholesky_inplace(self.internal_mat,inplace=inplace).T
                #sp.linalg.cholesky(self.covar_cache,lower=True,check_finite=False,overwrite_a=True).T
            elif self.internal_state == REP_FISHER:#or self.good_caches[REP_FISHER]:
                print "fisher_matrix "+str(id(self)),": getting cholesky decomposition of covariance matrix from F_alpha_beta, size: "+str(self.internal_mat.nbytes/10**6)+" megabytes"
                result = get_inv_cholesky(self.internal_mat,copy_output)
            else:
                raise ValueError("fisher_matrix "+str(id(self))+": unrecognized internal state "+str(self.internal_state))

            print "fisher_matrix "+str(id(self)),": found cholesky decomposition of covariance matrix, size: "+str(result.nbytes/10**6)+" megabytes"

        #TODO default correctly on copy_output
        if copy_output and not copy_safe:
            return result.copy()
        else:
            return result


    def get_F_alpha_beta(self,copy_output=False,inplace=False):
        copy_safe = False 
        if self.internal_state == REP_FISHER :#or self.good_caches[REP_FISHER]:
            print "fisher_matrix "+str(id(self))+": retrieved fisher matrix from cache"
            result = self.internal_mat
        else:
            print "fisher_matrix "+str(id(self))+": fisher matrix cache miss"
            #TODO check efficiency/arguments
            chol_res = self.get_fab_cholesky(copy_output=False)
            result = np.dot(chol_res,chol_res.T)
            
        if copy_output and not copy_safe:
            return result.copy()
        else:
            return result

