"""difference between two geos assuming first geo completely contains second"""
from __future__ import division,print_function,absolute_import
from builtins import range
import numpy as np

from geo import Geo
class AlmDifferenceGeo(Geo):
    """get the difference between two geos by subtracting alms assuming the first completly contains the second"""
    def __init__(self,geo_in,mask_in,C,zs,z_fine):
        """geo_in,mask_in:    Geo objects"""
        self.geo_in = geo_in
        self.mask_in = mask_in
        assert (self.geo_in.angular_area()-self.mask_in.angular_area())>=0.

        Geo.__init__(self,zs,C,z_fine)
        l_max = np.min([geo_in._l_max,mask_in._l_max])
        self.expand_alm_table(l_max)

    def angular_area(self):
        """get angular area"""
        return self.geo_in.angular_area()-self.mask_in.angular_area()

    def expand_alm_table(self,l_max):
        """expand the internal alm table out to l_max"""
        self.geo_in.expand_alm_table(l_max)
        self.mask_in.expand_alm_table(l_max)
        ls = np.arange(np.int(self._l_max)+1,np.int(l_max)+1)
        if ls.size==0:
            return
        for ll in ls:
            for mm in range(-ll,ll+1):
                self.alm_table[(ll,mm)] = self.geo_in.alm_table[(ll,mm)]-self.mask_in.alm_table[(ll,mm)]
        self._l_max = l_max

    def a_lm(self,l,m):
        """get a(l,m) for the geometry"""
        if l>self._l_max:
            self.expand_alm_table(l)
        alm = self.alm_table.get((l,m))
        if alm is None:
            raise RuntimeError('PolygonGeo: a_lm generation failed for unknown reason at l='+str(l)+',m='+str(m))
        return alm
