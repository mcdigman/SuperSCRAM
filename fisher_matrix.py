import numpy as np
from algebra_utils import ch_inv,cholesky_inv_contract,get_inv_cholesky,invert_triangular,get_mat_from_inv_cholesky
from warnings import warn
REP_FISHER = 0
REP_CHOL = 1
REP_CHOL_INV = 2 
REP_COVAR = 3
DEFAULT_ALLOWED_CACHES = {REP_FISHER:False,REP_CHOL:True,REP_CHOL_INV:False,REP_COVAR:False}

class fisher_matrix:


    def __init__(self,input_matrix,input_type=REP_FISHER,initial_state = REP_FISHER,allowed_caches=DEFAULT_ALLOWED_CACHES):

        self.good_caches = {REP_FISHER:False,REP_CHOL:False,REP_CHOL_INV:False,REP_COVAR:False}

        self.F_alpha_beta = None
        self.cholesky_cache = None
        self.cholesky_inv_cache = None
        self.covar_cache = None

        if input_type == REP_FISHER:
            self.F_alpha_beta = input_matrix
        elif input_type == REP_CHOL:
            self.cholesky_cache = input_matrix
        elif input_type == REP_CHOL_INV:
            self.cholesky_inv_cache = input_matrix
        elif input_type == REP_COVAR:
            self.covar_cache = input_matrix
        else:
            raise ValueError("fisher_matrix: unrecognized input_type: "+str(input_type))

        self.good_caches[input_type] = True      
        self.internal_state = input_type
        self.allowed_caches = allowed_caches

        self.switch_rep(initial_state)
        print("fisher_matrix "+str(id(self))+" created fisher matrix")

    #Switches the internal representation of the fisher matrix
    def switch_rep(self,new_state):
        old_state = self.internal_state
        if old_state == new_state:
            print("fisher_matrix "+str(id(self))+": internal state "+str(self.internal_state)+" unchanged")
        elif self.good_caches[new_state] == True:
            print("fisher_matrix "+str(id(self))+": internal state "+str(self.internal_state)+" switched to "+str(new_state)+" from cache")
            self.internal_state=new_state
        elif new_state == REP_CHOL:
            self.cholesky_cache = self.get_cov_cholesky()
        elif new_state == REP_CHOL_INV:
            self.cholesky_inv_cache = self.get_fab_cholesky()
        elif new_state == REP_FISHER:
            self.F_alpha_beta = self.get_F_alpha_beta()
        elif new_state == REP_COV: 
            self.covar_cache = self.get_covar()
        else:
            raise ValueError("fisher_matrix "+str(id(self))+": unrecognized new state "+str(new_state)+" when asked to switch from state "+str(old_state))
        #The internal state is necessary, so it does not count as a cache
        self.good_caches[new_state] = False
        self.good_caches[old_state] = True
        self.internal_state = new_state
        #Clear the old state cache if I am not allowed to keep it.
        self.clear_cache(keep_allowed=True)

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
            self.clear_cache()
        self.F_alpha_beta += F_n

    #TODO handle F_alpha_beta with 0 eigenvalues separately
    def get_covar(self):
        print("fisher_matrix "+str(id(self))+": getting covariance")
        if self.internal_state==REP_COVAR or self.good_caches[REP_COVAR]:
            print("fisher_matrix "+str(id(self))+": covar retrieved from cache")
            return self.covar_cache
        else:
            print("fisher_matrix "+str(id(self))+": covar cache miss")
            result = ch_inv(self.get_cov_cholesky(),cholesky_given=True)
            if self.allowed_caches[REP_COVAR]:
                self.good_caches[REP_COVAR]
                self.covar_cache = result
            return result
   
    #for getting variance
    def contract_covar(self,v1,v2,identical_inputs=False):
        print "fisher_matrix "+str(id(self))+": contracting covariance"
        return cholesky_inv_contract(self.get_cov_cholesky(),v1,v2,cholesky_given=True,identical_inputs=identical_inputs)
   

    def get_fab_cholesky(self):
        if self.internal_state==REP_CHOL_INV or self.good_caches[REP_CHOL_INV]:
            print "fisher_matrix ",id(self)," cholesky decomposition inv retrieved from cache"
            return self.cholesky_inv_cache
        else:
            print "fisher_matrix ",id(self)," cholesky decomposition inv cache miss"
            if self.internal_state == REP_CHOL or self.good_caches[REP_CHOL]:
                result = invert_triangular(self.cholesky_cache)
            elif self.internal_state == REP_FISHER or self.good_caches[REP_FISHER]:
                result = np.linalg.cholesky(self.F_alpha_beta)
            elif self.internal_state == REP_COVAR or self.good_caches[REP_COVAR]:
                result = get_inv_cholesky(self.covar_cache).T
            else:
                raise ValueError("fisher_matrix "+str(id(self))+": unrecognized internal state "+str(self.internal_state))


            if self.allowed_caches[REP_CHOL_INV]:
                self.cholesky_inv_cache = result
                self.good_caches[REP_CHOL_INV] = True
            return result 

    def get_cov_cholesky(self):
        if self.internal_state==REP_CHOL or self.good_caches[REP_CHOL]:
            print "fisher_matrix ",str(id(self))+": cholesky decomposition retrieved from cache"
            return self.cholesky_cache
        else:
            print "fisher_matrix "+str(id(self)),": cholesky decomposition cache miss"
            if self.internal_state == REP_CHOL_INV or self.good_caches[REP_CHOL_INV]:
                result = invert_trianglular(self.cholesky_inv_cache)
            elif self.internal_state == REP_COVAR or self.good_caches[REP_COVAR]:
                result = np.linalg.cholesky(self.covar_cache).T
            elif self.internal_state == REP_FISHER or self.good_caches[REP_FISHER]:
                result = get_inv_cholesky(self.F_alpha_beta)
            else:
                raise ValueError("fisher_matrix "+str(id(self))+": unrecognized internal state "+str(self.internal_state))


            if self.allowed_caches[REP_CHOL]:
                self.cholesky_cache = result
                self.good_caches[REP_CHOL] = True
            return result 


    def get_F_alpha_beta(self):
        if self.internal_state == REP_FISHER or self.good_caches[REP_FISHER]:
            print "fisher_matrix "+str(id(self))+": retrieved fisher matrix from cache"
            return self.F_alpha_beta
        else:
            print "fisher_matrix "+str(id(self))+": fisher matrix cache miss"
            chol_res = self.get_fab_cholesky()
            result = np.dot(chol_res,chol_res.T)
            
            if self.allowed_caches[REP_FISHER]:
                self.F_alpha_beta = result
                self.good_caches[REP_FISHER] = True
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

