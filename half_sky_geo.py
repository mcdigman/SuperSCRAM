"""analytic geo of entire sky"""
from __future__ import division,print_function,absolute_import
from builtins import range

from mpmath import mp
mp.dps = 15

import numpy as np
from geo import Geo

class HalfSkyGeo(Geo):
    """analytic geo of entire sky"""
    def __init__(self,zs,C,z_fine,top=True):
        """create an analytic geo of the full sky
                inputs:
                    zs: the tomographic z bins
                    C: a CosmoPie object
                    z_fine: the resolution z slices
                    l_max: the maximum l to compute the alm table to
                    res_healpix: 4 to 9, healpix resolution to use
        """
        self.C = C
        self.top = top
        self.z_fine = z_fine

        if self.top:
            self.sign = 1
        else:
            self.sign = -1
        Geo.__init__(self,zs,C,z_fine)

    def get_overlap_fraction(self,geo2):
        """get overlap fraction between this geometry and another geo"""
        #TODO don't use
        return geo2.angular_area()/self.angular_area()

    def angular_area(self):
        return 4.*np.pi

    #use mpmath to avoid overflow in gamma
    def a_lm(self,ll,mm):
        ll_m = mp.mpf(ll)
        if ll==0 and mm==0:
            return np.sqrt(np.pi)
        elif mm==0 and np.mod(ll,2)==1:
            return float(self.sign*mp.sqrt(1+2*ll_m)*mp.pi/(ll_m*mp.gamma(-ll_m/2)*mp.gamma((3+ll_m)/2.)))
        else:
            return 0.
