"""implement NZMatcher by matching the number density
in the CANDELS GOODS-S catalogue with  a preapplied cut"""
from __future__ import division,print_function,absolute_import
from builtins import range
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from nz_matcher import NZMatcher,get_gaussian_smoothed_dN_dz

#TODO: check get_nz and get_M_cut agree
class NZWFirstEff(NZMatcher):
    """Match n(z) using given dNdz
    """
    def __init__(self,params):
        """get the CANDELS matcher
            inputs:
                params: a dict of params including:
                    smooth_sigma: a smoothing length scale
                    n_right_extend: number of smoothing scales beyond max z
                    z_resolution: resolution of z grid to use
                    mirror_boundary: True to use mirrored boundary conditions at z=0 for smoothing
                    area_sterad: area of survey in steradians
        """
        self.params = params

        #load data and select nz
        self.data = np.loadtxt('wfirst_dNdz.csv',delimiter=',')
        z_grid = np.arange(0.,np.max(self.data[:,0]),self.params['z_resolution'])
        dN_dz = InterpolatedUnivariateSpline(np.hstack([0.,self.data[:,0]]),(180.**2/np.pi**2)*np.hstack([0.,self.data[:,1]]),k=3,ext=2)(z_grid)
        #strip the redshifts which are exactly 0, as they are redshift failures
        #self.data = self.data[self.data[:,1]>self.params['z_cut'],:]
        #use separate internal grid for calculations because smoothing means galaxies beyond max z_fine can create edge effects
        #z_grid = np.arange(0.,np.max(self.data[:,1])+self.params['smooth_sigma']*self.params['n_right_extend'],self.params['z_resolution'])
        #self.chosen = (self.data[:,5]<self.params['i_cut'])
        #self.chosen = np.full(self.data[:,0].size,True,dtype=bool)
        #cut off faint galaxies
        #self.zs_chosen = self.data[self.chosen,1]
        #print("nz_wfirst: "+str(self.zs_chosen.size)+" available galaxies")
        #dN_dz = get_gaussian_smoothed_dN_dz(z_grid,self.zs_chosen,params,normalize=True)
        NZMatcher.__init__(self,z_grid,dN_dz)
