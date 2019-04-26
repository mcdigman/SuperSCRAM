"""analytic geo of entire sky"""
from __future__ import division,print_function,absolute_import
from builtins import range

from mpmath import mp

import numpy as np
from geo import Geo

mp.dps = 15

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
        #don't use
        return geo2.angular_area()/self.angular_area()

    def angular_area(self):
        return 2.*np.pi

    #use mpmath to avoid overflow in gamma
    def a_lm(self,l,m):
        ll_m = mp.mpf(l)
        if l==0 and m==0:
            return np.sqrt(np.pi)
        elif m==0 and np.mod(l,2)==1:
            return float(self.sign*mp.sqrt(1+2*ll_m)*mp.pi/(ll_m*mp.gamma(-ll_m/2)*mp.gamma((3+ll_m)/2.)))
        else:
            return 0.
