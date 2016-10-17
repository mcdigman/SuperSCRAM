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

        #None of the caches are good initially
        self.good_caches = {REP_FISHER:False,REP_CHOL:False,REP_CHOL_INV:False,REP_COVAR:False}

        self.F_alpha_beta = None
        self.cholesky_cache = None
        self.cholesky_inv_cache = None
        self.covar_cache = None

        self.fix_input=fix_input
        #If not allowed to change the input matrix, then copy it 
        if self.fix_input:
            self.internal_mat = input_matrix.copy()
        else:
            self.internal_mat = input_matrix 
    
#        #Put the input matrix in the appropriate variable
#        if input_type == REP_FISHER:
#            self.F_alpha_beta = internal_matrix
#        elif input_type == REP_CHOL:
#            self.cholesky_cache = internal_matrix
#        elif input_type == REP_CHOL_INV:
#            self.cholesky_inv_cache = internal_matrix
#        elif input_type == REP_COVAR:
#            self.covar_cache = internal_matrix
#        else:
#            raise ValueError("fisher_matrix: unrecognized input_type: "+str(input_type))

#        self.good_caches[input_type] = True      
        self.cache_matrix(self.internal_mat,input_type)
        self.internal_state = input_type
        self.allowed_caches = allowed_caches

        #Switch to the specified initial state
        self.switch_rep(initial_state)
        print("fisher_matrix "+str(id(self))+" created fisher matrix")
    
    #Switches the internal representation of the fisher matrix
    def switch_rep(self,new_state):
        old_state = self.internal_state
        print "fisher matrix "+str(id(self))+": changing internal state from "+str(self.internal_state)+" to "+str(new_state)
        found = False
        if old_state==new_state:
            print("fisher_matrix "+str(id(self))+": internal state "+str(self.internal_state)+" unchanged")
            found = True
        elif self.allowed_caches[old_state]: 
            if self.good_caches[new_state]:
                print ("fisher matrix: swapping internal state")
                self.cache_matrix(self.internal_mat,old_state)
                self.internal_mat = self.retrieve_cached(new_state) 
                found = True
            else:
                print("fisher matrix: caching internal state")
                self.cache_matrix(self.internal_mat.copy(),old_state)
        else:
            pass 

        if found:
            pass
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
        #TODO this is kind of hackish
        self.cache_matrix(self.internal_mat,new_state)
        #The internal state is necessary, so it does not count as a cache
        self.good_caches[new_state] = False
        self.internal_state = new_state
        #Clear the old state cache if I am not allowed to keep it.
        self.clear_cache(keep_allowed=True)
        print "fisher matrix "+str(id(self))+": internal state changed from "+str(self.internal_state)+" to "+str(new_state)

    #Do not check if the cache is allowed, let calling methods do that
    def retrieve_cached(self,cache_state):
        if cache_state == REP_CHOL:
            return self.cholesky_cache
        elif cache_state == REP_CHOL_INV:
            return self.cholesky_inv_cache
        elif cache_state == REP_FISHER:
            return self.F_alpha_beta 
        elif cache_state == REP_COVAR: 
            return self.covar_cache
        else:
            raise ValueError("fisher_matrix "+str(id(self))+": unrecognized state "+str(cache_state)+" when asked to retrieve from cache ")

    #Do not check if the cache is allowed, let calling methods do that
    def cache_matrix(self,cache_matrix,cache_state):
        if cache_state == REP_CHOL:
            self.cholesky_cache = cache_matrix
        elif cache_state == REP_CHOL_INV:
            self.cholesky_inv_cache = cache_matrix
        elif cache_state == REP_FISHER:
            self.F_alpha_beta = cache_matrix
        elif cache_state == REP_COVAR: 
            self.covar_cache = cache_matrix
        else:
            raise ValueError("fisher_matrix "+str(id(self))+": unrecognized state "+str(cache_state)+" when asked to store in cache")
        self.good_caches[cache_state] = False

    #Check if any of the caches are filled
    def some_caches_good(self):
        for key in self.good_caches:
            if self.good_caches[key]:
                return True
        return False

    #Add a fisher matrix (ie a perturbation) to the internal fisher matrix
    def add_fisher(self,F_n):
        print "fisher_matrix ",id(self)," added fisher matrix"
        #internal state must be fisher to add right now: could possibly be changed
        self.switch_rep(REP_FISHER)
        #caches won't be good anymore after changing internal matrix
        if self.some_caches_filled(keep_allowed=False):
            self.clear_cache(keep_allowed=False)
        #TODO check this is ok with views
        self.F_alpha_beta += F_n

    #TODO handle F_alpha_beta with 0 eigenvalues separately
    #copy_output=True should guarantee the output will never be mutated by fisher_matrix (ie copy the output matrix if necessary)
    #copy_output=False will not copy the output matrix in general, which will be more memory efficient but might cause problems if the user is not careful
    def get_covar(self,copy_output=False,inplace=False):
        copy_safe = False
        print("fisher_matrix "+str(id(self))+": getting covariance")
        if self.internal_state==REP_COVAR or self.good_caches[REP_COVAR]:
            print("fisher_matrix "+str(id(self))+": covar retrieved from cache")
            result = self.covar_cache
        else:
            print("fisher_matrix "+str(id(self))+": covar cache miss")
            result = ch_inv(self.get_cov_cholesky(),cholesky_given=True)
            if self.allowed_caches[REP_COVAR]:
                self.good_caches[REP_COVAR]
                self.covar_cache = result
            else:
                copy_safe = True


        if copy_output and not copy_safe:
            return result.copy()
        else: 
            return result
        
   
    #for getting variance
    def contract_covar(self,v1,v2,identical_inputs=False):
        print "fisher_matrix "+str(id(self))+": contracting covariance"
        return cholesky_inv_contract(self.get_cov_cholesky(),v1,v2,cholesky_given=True,identical_inputs=identical_inputs)
   

    def get_fab_cholesky(self,copy_output=False,inplace=False):
        copy_safe = False
        if self.internal_state==REP_CHOL_INV or self.good_caches[REP_CHOL_INV]:
            print "fisher_matrix ",id(self)," cholesky decomposition inv retrieved from cache"
            result = self.cholesky_inv_cache
        else:
            print "fisher_matrix ",id(self)," cholesky decomposition inv cache miss"
            if self.internal_state == REP_CHOL or self.good_caches[REP_CHOL]:
                result = invert_triangular(self.cholesky_cache)
            elif self.internal_state == REP_FISHER or self.good_caches[REP_FISHER]:
                #TODO use cholesky_inplace as appropriate
                result = np.linalg.cholesky(self.F_alpha_beta)
            elif self.internal_state == REP_COVAR or self.good_caches[REP_COVAR]:
                result = get_inv_cholesky(self.covar_cache).T
            else:
                raise ValueError("fisher_matrix "+str(id(self))+": unrecognized internal state "+str(self.internal_state))


            if self.allowed_caches[REP_CHOL_INV]:
                self.cholesky_inv_cache = result
                self.good_caches[REP_CHOL_INV] = True
            else:
                copy_safe = True

            if copy_output and not copy_safe:
                return result.copy()
            else:
                return result

    def get_cov_cholesky(self,copy_output=False,inplace=False):
        copy_safe = False
        if self.internal_state==REP_CHOL or self.good_caches[REP_CHOL]:
            print "fisher_matrix ",str(id(self))+": cholesky decomposition retrieved from cache"
            return self.cholesky_cache
        else:
            print "fisher_matrix "+str(id(self)),": cholesky decomposition cache miss"
            #TODO: evaluate prioritization of ways to find the decomposition, esp. for numerical stability versus speed/memory consumption (cholesky may be more numerically stable than inv)
            if self.internal_state == REP_CHOL_INV or self.good_caches[REP_CHOL_INV]:
                print "fisher_matrix "+str(id(self)),": getting cholesky decomposition of covariance matrix from its inverse, size: "+str(self.cholesky_inv_cache.nbytes/10**6)+" megabytes"
                result = invert_trianglular(self.cholesky_inv_cache)
            elif self.internal_state == REP_COVAR or self.good_caches[REP_COVAR]:
                print "fisher_matrix "+str(id(self)),": getting cholesky decomposition of covariance matrix directly, size: "+str(self.covar_cache.nbytes/10**6)+" megabytes"
                result = cholesky_inplace(self.covar_cache,inplace=inplace).T
                #sp.linalg.cholesky(self.covar_cache,lower=True,check_finite=False,overwrite_a=True).T
            elif self.internal_state == REP_FISHER or self.good_caches[REP_FISHER]:
                print "fisher_matrix "+str(id(self)),": getting cholesky decomposition of covariance matrix from F_alpha_beta, size: "+str(self.F_alpha_beta.nbytes/10**6)+" megabytes"
                result = get_inv_cholesky(self.F_alpha_beta,copy_output)
            else:
                raise ValueError("fisher_matrix "+str(id(self))+": unrecognized internal state "+str(self.internal_state))

            print "fisher_matrix "+str(id(self)),": found cholesky decomposition of covariance matrix, size: "+str(result.nbytes/10**6)+" megabytes"

         
            if self.allowed_caches[REP_CHOL]:
                self.cholesky_cache = result
                if not self.internal_state == REP_CHOL:
                    self.good_caches[REP_CHOL] = True
            else:
                copy_safe = True

            if copy_output and not copy_safe:
                return result.copy()
            else:
                return result


    def get_F_alpha_beta(self,copy_output=False,inplace=False):
        copy_safe = False 
        if self.internal_state == REP_FISHER or self.good_caches[REP_FISHER]:
            print "fisher_matrix "+str(id(self))+": retrieved fisher matrix from cache"
            result = self.F_alpha_beta
        else:
            print "fisher_matrix "+str(id(self))+": fisher matrix cache miss"
            chol_res = self.get_fab_cholesky()
            result = np.dot(chol_res,chol_res.T)
            
        if self.allowed_caches[REP_FISHER]:
            self.F_alpha_beta = result
            self.good_caches[REP_FISHER] = True
        else:
            copy_safe = True

        if copy_output and not copy_safe:
            return result.copy()
        else:
            return result


    def clear_cache(self,keep_allowed=False,specific=False,clear_id=None):
        for key in self.allowed_caches:
            if (keep_allowed and self.allowed_caches[key] and not specific) or (specific and not key==clear_id) or key==self.internal_state:
                pass
            else:
                self.good_caches[key] = False
                if key == REP_FISHER:
                    self.F_alpha_beta = None
                elif key == REP_CHOL:
                    self.cholesky_cache = None
                elif key == REP_CHOL_INV:
                    self.cholesky_inv_cache = None
                elif key == REP_COVAR:
                    self.covar_cache = None
                else:
                    raise ValueError("fisher_matrix "+str(id(self))+": unrecognized state "+str(key)+" when asked to switch to clear cache ")
        print "fisher_matrix ",id(self)," caches cleared"

