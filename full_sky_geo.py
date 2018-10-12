"""analytic geo of entire sky"""
from __future__ import division,print_function,absolute_import
from builtins import range

import numpy as np
from geo import Geo

class FullSkyGeo(Geo):
    """analytic geo of entire sky"""
    def __init__(self,zs,C,z_fine):
        """create an analytic geo of the full sky
                inputs:
                    zs: the tomographic z bins
                    C: a CosmoPie object
                    z_fine: the resolution z slices
                    l_max: the maximum l to compute the alm table to
                    res_healpix: 4 to 9, healpix resolution to use
        """
        self.C = C
        self.z_fine = z_fine
        Geo.__init__(self,zs,C,z_fine)

    def get_overlap_fraction(self,geo2):
        """get overlap fraction between this geometry and another geo"""
        return geo2.angular_area()/self.angular_area()

    def angular_area(self):
        return 4.*np.pi

    def a_lm(self,ll,mm):
        if ll==0 and mm==0:
            return np.sqrt(4.*np.pi)
        else:
            return 0.
