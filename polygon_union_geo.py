"""provide ability to compute union of PolygonGeos and use PolygonGeos as masks"""
from __future__ import division,print_function,absolute_import
from builtins import range
import numpy as np
import spherical_geometry.vector as sgv

from polygon_geo import PolygonGeo
from geo import Geo

class PolygonUnionGeo(Geo):
    """get the geo represented by the union of geos minus anything covered by masks, all PolygonGeo objects"""
    def __init__(self,geos,masks,C=None,zs=None,z_fine=None,l_max=None,poly_params=None):
        """geo,masks:    an array of PolygonGeo objects"""
        self.geos = geos
        self.masks = masks
        self.n_g = geos.size
        self.n_m = masks.size
        if zs is None:
            zs = geos[0].zs
        if z_fine is None:
            z_fine = geos[0].z_fine
        if C is None:
            C = geos[0].C
        if l_max is None:
            l_max = geos[0].l_max
        if poly_params is None:
            poly_params = geos[0].poly_params

        self.polys_pos = np.zeros(self.n_g,dtype=object)
        self.polys_mask = np.zeros(self.n_m,dtype=object)

        for itr in range(0,self.masks.size):
            if not isinstance(masks[itr],PolygonGeo):
                raise ValueError('unsupported type for mask')
            print(masks[itr].angular_area())
        for itr in range(0,self.geos.size):
            if not isinstance(geos[itr],PolygonGeo):
                raise ValueError('unsupported type for geo')
            print(geos[itr].angular_area())

        for itr in range(0,self.masks.size):
            self.polys_mask[itr] = masks[itr].sp_poly

        self.union_pos = self.geos[0].sp_poly
        for itr in range(1,self.n_g):
            self.union_pos = self.union_pos.union(self.polys_pos[itr])

        self.union_xyz = list(self.union_pos.points)
        self.union_in = list(self.union_pos.inside)
        self.n_union = len(self.union_xyz)
        self.union_geos = np.zeros(self.n_union,dtype=object)

        if self.n_g==1:
            self.union_geos[0] = self.geos[0]
        else:
            for itr in range(0,self.n_union):
                union_ra,union_dec = sgv.vector_to_radec(self.union_xyz[itr][:,0],self.union_xyz[itr][:,1],self.union_xyz[itr][:,2],degrees=False)
                in_ra,in_dec = sgv.vector_to_radec(self.union_in[itr][0],self.union_in[itr][1],self.union_in[itr][2],degrees=False)
                self.union_geos[itr] = PolygonGeo(zs,union_dec+np.pi/2.,union_ra,in_dec+np.pi/2.,in_ra,C,z_fine,l_max,poly_params)
        if self.n_m>0:
            #get union of all the masks with the union of the inside, ie the intersection, which is the mask to use
            self.union_mask = self.polys_mask[0]
            for itr1 in range(1,self.n_m):
                self.union_mask = self.union_mask.union(self.polys_mask[itr1])
            #note union_mask can be several disjoint polygons
            #print("mask poly",self.union_mask)
            #print("union pos",self.union_pos)
            self.union_mask = self.union_pos.intersection(self.union_mask)
            self.mask_xyz = list(self.union_mask.points)
            in_point = list(self.union_mask.inside)
            self.n_mask = len(self.mask_xyz)
            self.mask_geos = np.zeros(self.n_mask,dtype=object)
            for itr in range(0,self.n_mask):
                mask_ra,mask_dec = sgv.vector_to_radec(self.mask_xyz[itr][:,0],self.mask_xyz[itr][:,1],self.mask_xyz[itr][:,2],degrees=False)
                in_ra,in_dec = sgv.vector_to_radec(in_point[itr][0],in_point[itr][1],in_point[itr][2],degrees=False)
                self.mask_geos[itr] = PolygonGeo(zs,mask_dec+np.pi/2.,mask_ra,in_dec+np.pi/2.,in_ra,C,z_fine,l_max,poly_params)
        else:
            #TODO not right
            self.union_mask = None
            self.mask_geos = None


        #self.sp_poly = self.union_geo

        Geo.__init__(self,zs,C,z_fine)
        self.alm_table = {(0,0):self.angular_area()/np.sqrt(4.*np.pi)}
        self._l_max = 0
        self.expand_alm_table(l_max)

    def angular_area(self):
        """get angular area"""
        if self.mask_geos is not None:
            area = 0
            for itr in range(0,self.n_union):
                area+=self.union_geos[itr].angular_area()
            for itr in range(0,self.n_mask):
                area-=self.mask_geos[itr].angular_area()
            return area
        else:
            return self.union_geos.angular_area()

    def expand_alm_table(self,l_max):
        """expand the internal alm table out to l_max"""
        for itr in range(0,self.n_union):
            self.union_geos[itr].expand_alm_table(l_max)
        for itr in range(0,self.n_mask):
            self.mask_geos[itr].expand_alm_table(l_max)
        ls = np.arange(np.int(self._l_max)+1,np.int(l_max)+1)
        if ls.size==0:
            return
        for ll in ls:
            for mm in range(-ll,ll+1):
                alm = 0
                for itr in range(0,self.n_union):
                    alm+=self.union_geos[itr].alm_table[(ll,mm)]
                for itr in range(0,self.n_mask):
                    alm-=self.mask_geos[itr].alm_table[(ll,mm)]
                self.alm_table[(ll,mm)] = alm
        self._l_max = l_max

    def a_lm(self,l,m):
        """get a(l,m) for the geometry"""
        if l>self._l_max:
            print("PolygonUnionGeo: l value "+str(l)+" exceeds maximum precomputed l "+str(self._l_max)+",expanding table")
            self.expand_alm_table(l)
        alm = self.alm_table.get((l,m))
        if alm is None:
            raise RuntimeError('PolygonGeo: a_lm generation failed for unknown reason at l='+str(l)+',m='+str(m))
        return alm
