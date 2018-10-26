"""difference between two geos assuming first geo completely contains second"""
from __future__ import division,print_function,absolute_import
from builtins import range
import numpy as np
from alm_utils import rot_alm_x,rot_alm_z

from geo import Geo
class AlmRotGeo(Geo):
    """get the geo with alms rotated around euler angles [z,x,z]"""
    def __init__(self,geo_in,C,zs,z_fine,angles,n_double):
        """geo_in    Geo object"""
        self.geo_in = geo_in
        self.angles = angles
        self.n_double = n_double

        Geo.__init__(self,zs,C,z_fine)
        l_max = geo_in._l_max
        self.expand_alm_table(l_max)

    def angular_area(self):
        """get angular area"""
        return self.geo_in.angular_area()

    def expand_alm_table(self,l_max):
        """expand the internal alm table out to l_max"""
        self.geo_in.expand_alm_table(l_max)
        ls = np.arange(np.int(self._l_max)+1,np.int(l_max)+1)
        if ls.size==0:
            return
        d_alm_array1 = np.zeros(ls.size,dtype=object)
        itr1 = 0
        for ll in ls:
            d_alm_array1[itr1] = np.zeros((2*ll+1,1))
            itr2 = 0
            for mm in range(-ll,ll+1):
                d_alm_array1[itr1][itr2,0] = self.geo_in.alm_table[(ll,mm)]
                itr2 = itr2+1
            itr1 = itr1+1
        d_alm_array2 = rot_alm_z(d_alm_array1,self.angles[0:1],ls)
        d_alm_array3 = rot_alm_x(d_alm_array2,self.angles[1:2],ls,self.n_double)
        d_alm_array4 = rot_alm_z(d_alm_array3,self.angles[2:3],ls)

        itr1 = 0
        for ll in ls:
            itr2 = 0
            for mm in range(-ll,ll+1):
                self.alm_table[(ll,mm)] = d_alm_array4[itr1][itr2,0]
                itr2 = itr2+1
            itr1 = itr1+1

        self._l_max = l_max

    def a_lm(self,l,m):
        """get a(l,m) for the geometry"""
        if l>self._l_max:
            print("AlmRotGeo: l value "+str(l)+" exceeds maximum precomputed l "+str(self._l_max)+",expanding table")
            self.expand_alm_table(l)
        alm = self.alm_table.get((l,m))
        if alm is None:
            raise RuntimeError('PolygonGeo: a_lm generation failed for unknown reason at l='+str(l)+',m='+str(m))
        return alm
