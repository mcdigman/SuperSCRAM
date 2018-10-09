"""a healpix pixelated geo of entire sky"""
from __future__ import division,print_function,absolute_import
from builtins import range

import numpy as np

from polygon_utils import get_healpix_pixelation
from pixel_geo import PixelGeo

class FullSkyPixelGeo(PixelGeo):
    """healpix pixelated geo of entire sky"""
    def __init__(self,zs,C,z_fine,l_max,res_healpix):
        """create a spherical polygon defined by vertices
                inputs:
                    zs: the tomographic z bins
                    C: a CosmoPie object
                    z_fine: the resolution z slices
                    l_max: the maximum l to compute the alm table to
                    res_healpix: 4 to 9, healpix resolution to use
        """
        self.C = C
        self.z_fine = z_fine
        self.res_healpix = res_healpix
        hard_l_max = 3.*2**self.res_healpix-1.
        self.all_pixels = get_healpix_pixelation(res_choose=self.res_healpix)
        self.contained =  np.full(self.all_pixels.shape[0],True)
        contained_pixels = self.all_pixels.copy()
        print("FullSkyPixelGeo: total contained pixels in polygon: "+str(np.sum(self.contained*1.)))
        PixelGeo.__init__(self,zs,contained_pixels,C,z_fine,l_max,hard_l_max)

    def get_overlap_fraction(self,geo2):
        """get overlap fraction between this geometry and another PolygonPixelGeo"""
        return geo2.angular_area()/self.angular_area()
