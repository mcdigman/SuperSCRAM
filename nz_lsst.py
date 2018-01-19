"""NZMatcher using the n(z) from the LSST whitepaper"""
import numpy as np
from nz_matcher import NZMatcher

class NZLSST(NZMatcher):
    """match the number density defined in the LSST whitepaper"""
    def __init__(self,z_grid, params):
        """ inputs:
                z_grid:a numpy array of zs
                params:
                    i_cut: limiting i band magnitude
        """
        self.params = params
        self.i_cut = self.params['i_cut']
        self.z0 = 0.0417*self.i_cut-0.744
        self.p_z = 1./(2.*self.z0)*(z_grid/self.z0)**2.*np.exp(-z_grid/self.z0)
        self.N_expected = 46*10.**(0.31*(self.i_cut-25))*3600.*180.**2/np.pi**2
        dN_dz = self.N_expected*self.p_z
        NZMatcher.__init__(self,z_grid,dN_dz)
