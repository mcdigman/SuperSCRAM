import numpy as np
from algebra_utils import cholesky_inv,cholesky_inv_contract
from warnings import warn

class fisher_matrix:

    def __init__(self,F_0,allow_caching=False):
        self.F_alpha_beta = F_0
        self.allow_caching = allow_caching
        if self.allow_caching:
            self.chol_cache_good = False
            self.cholesky_cache = None
        print "fisher_matrix ",id(self)," created fisher matrix"

    def add_fisher(self,F_n):
        print "fisher_matrix ",id(self)," added fisher matrix"
        if self.allow_caching and self.chol_cache_good:
            self.clear_cache()
        self.F_alpha_beta += F_n

    #TODO handle F_alpha_beta with 0 eigenvalues separately
    def get_covar(self):
        if self.allow_caching:
            if self.chol_cache_good:
                print "fisher_matrix ",id(self)," cholesky decomposition retrieved from cache for calculating inverse in get_covar"
                cov = cholesky_inv(self.cholesky_cache,return_cholesky=False,cholesky_given=True)
            else:
                print "fisher_matrix ",id(self)," cholesky decomposition cache miss for calculating inverse in get_covar"
                cov,self.cholesky_cache = cholesky_inv(self.F_alpha_beta,return_cholesky=True,cholesky_given=False)
                self.chol_cache_good = True
            return cov
        else:
            warn("fisher_matrix ",id(self)," did not cache")
            return cholesky_inv(self.F_alpha_beta,return_cholesky=False)    
   
    #for getting variance
    def contract_covar(self,v1,v2,identical_inputs=False):
        if self.allow_caching:
            if self.chol_cache_good:
                print "fisher_matrix ",id(self)," cholesky decomposition retrieved from cache for calculating contract_covar"
                result = cholesky_inv_contract(self.cholesky_cache,v1,v2,cholesky_given=True,return_cholesky=False,identical_inputs=identical_inputs)
            else:
                print "fisher_matrix ",id(self)," cholesky decomposition cache miss for calculating contract_covar"
                result,self.cholesky_cache = cholesky_inv_contract(self.F_alpha_beta,v1,v2,cholesky_given=False,return_cholesky=True,identical_inputs=identical_inputs)
        else:
            warn("fisher_matrix ",id(self)," did not cache")
            result = cholesky_inv_contract(self.F_alpha_beta,v1,v2,identical_inputs=identical_inputs)
        return result

    def get_F_alpha_beta(self):
        print "fisher_matrix ",id(self)," retrieved fisher matrix"
        return self.F_alpha_beta

    def clear_cache(self):
        self.chol_cache_good=False
        self.cholesky_cache = None
        print "fisher_matrix ",id(self)," cache cleared"

