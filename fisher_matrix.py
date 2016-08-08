import numpy as np
from algebra_utils import cholesky_inv,cholesky_inv_contract

class fisher_matrix:
    def __init__(self,F_0):
        self.F_alpha_beta = F_0

    def add_fisher(self,F_n):
        self.F_alpha_beta += F_n
        return 

    #TODO handle F_alpha_beta with 0 eigenvalues separately
    def get_covar(self):
        return cholesky_inv(self.F_alpha_beta)    
   
    #for getting variance
    def contract_covar(self,v1,v2):
        return cholesky_inv_contract(self.F_alpha_beta,v1,v2)

    def get_F_alpha_beta(self):
        return self.F_alpha_beta
