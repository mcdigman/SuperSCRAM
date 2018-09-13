"""implement NZMatcher by constant number density"""
from __future__ import division,print_function,absolute_import
from builtins import range
import numpy as np

from nz_matcher import NZMatcher#,get_gaussian_smoothed_dN_dz

class NZConstant(NZMatcher):
    """get matcher for constant density"""
    def __init__(self,geo,params):
        """get the constant matcher
            inputs:
                params: a dict of params
        """
        self.params = params
        nz = np.full(geo.z_fine.size,params['nz_constant'])
        dN_dz = nz*geo.r_fine**2/geo.dzdr
        NZMatcher.__init__(self,geo.z_fine,dN_dz)
