"""implement NZMatcher by matching the number density
in the CANDELS GOODS-S catalogue for a limiting i band magnitude"""
import numpy as np

from nz_matcher import NZMatcher,get_gaussian_smoothed_dN_dz

class NZCandel(NZMatcher):
    """get matcher for CANDELS dataset using limiting i band magnitude"""
    def __init__(self,params):
        """get the CANDELS matcher
            inputs:
                params: a dict of params
        """
        self.params = params

        #load data and select nz
        self.data = np.loadtxt(self.params['data_source'])
        #use separate internal grid for calculations because smoothing means galaxies beyond max z_fine can create edge effects
        z_grid = np.arange(0.,np.max(self.data[:,1])+self.params['smooth_sigma']*self.params['n_right_extend'],self.params['z_resolution'])
        self.chosen = (self.data[:,5]<self.params['i_cut'])
        #cut off faint galaxies
        self.zs_chosen = self.data[self.chosen,1]
        print "nz_candel: "+str(self.zs_chosen.size)+" available galaxies"
        dN_dz = get_gaussian_smoothed_dN_dz(z_grid,self.zs_chosen,params,normalize=True)
        NZMatcher.__init__(self,z_grid,dN_dz)
