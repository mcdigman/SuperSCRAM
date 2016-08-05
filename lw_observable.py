import numpy as np
import cosmopie as cp

class lw_observable:
    def __init__(self,C=cp.CosmoPie()):
        #self.C = C
        self.C=C
    #def get_dO_a_ddelta_bar():
    #    raise NotImplementedError('subclasses of lw_observable must implement get_dO_a_ddelta_bar')
    def Fisher_alpha_beta():
        raise NotImplementedError('subclasses of lw_observable must implement Fisher_alpha_beta')
