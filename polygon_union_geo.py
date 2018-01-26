"""provide ability to compute union of PolygonGeos and use PolygonGeos as masks"""
import numpy as np
import spherical_geometry.vector as sgv

from polygon_geo import PolygonGeo
from geo import Geo

class PolygonUnionGeo(Geo):
    """get the geo represented by the union of geos minus anything covered by masks, all PolygonGeo objects"""
    def __init__(self,geos,masks):
        """geo,masks:    an array of PolygonGeo objects"""
        self.geos = geos
        self.n_g = geos.size
        self.n_m = masks.size
        polys_pos = np.zeros(self.n_g,dtype=object)
        polys_mask = np.zeros(self.n_m,dtype=object)
        for itr1 in xrange(0,self.n_g):
            polys_pos[itr1] = geos[itr1].sp_poly
        for itr2 in xrange(0,self.n_m):
            polys_mask[itr2] = masks[itr2].sp_poly

        self.union_pos = polys_pos[0]
        for itr1 in xrange(1,self.n_g):
            self.union_pos = self.union_pos.union(polys_pos[itr1])
        self.union_xyz = list(self.union_pos.points)
        self.union_in = list(self.union_pos.inside)
        self.n_union = len(self.union_xyz)
        #union_pos can consist of several disjoint polygons, build a PolygonGeo for each
        self.union_geos = np.zeros(self.n_union,dtype=object)
        self.union_phi = np.zeros(self.n_union,dtype=object)
        self.union_theta = np.zeros(self.n_union,dtype=object)
        self.union_phi_in = np.zeros(self.n_union,dtype=object)
        self.union_theta_in = np.zeros(self.n_union,dtype=object)
        for itr in xrange(0,self.n_union):
            union_ra,union_dec = sgv.vector_to_radec(self.union_xyz[itr][:,0],self.union_xyz[itr][:,1],self.union_xyz[itr][:,2],degrees=False)
            self.union_phi[itr] = union_ra
            self.union_theta[itr] = union_dec+np.pi/2.
            in_ra,in_dec = sgv.vector_to_radec(self.union_in[itr][0],self.union_in[itr][1],self.union_in[itr][2],degrees=False)
            self.union_phi_in[itr] = in_ra
            self.union_theta_in[itr] = in_dec+np.pi/2.
            self.union_geos[itr] = PolygonGeo(geos[0].zs,self.union_theta[itr],self.union_phi[itr],self.union_theta_in[itr],self.union_phi_in[itr],geos[0].C,geos[0].z_fine,geos[0].l_max,geos[0].poly_params)
        if self.n_m>0:
            #get union of all the masks with the union of the inside, ie the intersection, which is the mask to use
            self.union_mask = polys_mask[0]
            for itr1 in xrange(1,self.n_m):
                self.union_mask = self.union_mask.union(polys_mask[itr1])
            #note union_mask can be several disjoint polygons
            self.union_mask = self.union_mask.intersection(self.union_pos)
            self.mask_xyz = list(self.union_mask.points)
            self.in_point = list(self.union_mask.inside)
            self.n_mask = len(self.mask_xyz)
            self.mask_ra = np.zeros(self.n_mask,dtype=object)
            self.mask_dec = np.zeros(self.n_mask,dtype=object)
            self.mask_phi = np.zeros(len(self.in_point),dtype=object)
            self.mask_theta = np.zeros(len(self.in_point),dtype=object)
            self.in_ra = np.zeros(self.n_mask,dtype=object)
            self.in_dec = np.zeros(self.n_mask,dtype=object)
            self.mask_geos = np.zeros(self.n_mask,dtype=object)
            for itr in xrange(0,self.n_mask):
                self.mask_ra[itr],self.mask_dec[itr] = sgv.vector_to_radec(self.mask_xyz[itr][:,0],self.mask_xyz[itr][:,1],self.mask_xyz[itr][:,2],degrees=False)
                self.mask_phi[itr] = self.mask_ra[itr]
                self.mask_theta[itr] = self.mask_dec[itr]+np.pi/2.
                self.in_ra[itr],self.in_dec[itr] = sgv.vector_to_radec(self.in_point[itr][0],self.in_point[itr][1],self.in_point[itr][2],degrees=False)
                self.mask_geos[itr] = PolygonGeo(geos[0].zs,self.mask_theta[itr],self.mask_phi[itr],self.in_dec[itr]+np.pi/2.,self.in_ra[itr],geos[0].C,geos[0].z_fine,geos[0].l_max,geos[0].poly_params)
        else:
            self.union_mask = None
            self.mask_xyz = None
            self.mask_phi = None
            self.mask_theta = None
            self.mask_geos = None



        Geo.__init__(self,self.union_geos[0].zs,self.union_geos[0].C,self.union_geos[0].z_fine)
        self.alm_table = {(0,0):self.angular_area()/np.sqrt(4.*np.pi)}
        self._l_max = 0
        self.expand_alm_table(geos[0].l_max)
        #intersect_polys = np.zeros((self.n_g,self.n_g),dtype=object)
        #for itr1 in xrange(0,self.n_g):
        #    for itr2 in xrange(0,self.n_g):
        #        intersect_polys[itr1,itr2] = geos[itr1].sp_poly.intersection(geos[itr2].sp_poly.intersection)

    def angular_area(self):
        """get angular area"""
        if self.mask_geos is not None:
            area = 0
            for itr in xrange(0,self.n_union):
                area+=self.union_geos[itr].angular_area()
            for itr in xrange(0,self.n_mask):
                area-=self.mask_geos[itr].angular_area()
            return area
        else:
            return self.union_geos.angular_area()

    def expand_alm_table(self,l_max):
        """expand the internal alm table out to l_max"""
        for itr in xrange(0,self.n_union):
            self.union_geos[itr].expand_alm_table(l_max)
        for itr in xrange(0,self.n_mask):
            self.mask_geos[itr].expand_alm_table(l_max)
        ls = np.arange(np.int(self._l_max)+1,np.int(l_max)+1)
        if ls.size==0:
            return
        for ll in ls:
            for mm in xrange(-ll,ll+1):
                alm = 0#-self.mask_geo.alm_table[(ll,mm)]
                for itr in xrange(0,self.n_union):
                    alm+=self.union_geos[itr].alm_table[(ll,mm)]
                for itr in xrange(0,self.n_mask):
                    alm-=self.mask_geos[itr].alm_table[(ll,mm)]
                self.alm_table[(ll,mm)] = alm
        self._l_max = l_max

    def a_lm(self,l,m):
        """get a(l,m) for the geometry"""
        if l>self._l_max:
            print "PolygonUnionGeo: l value "+str(l)+" exceeds maximum precomputed l "+str(self._l_max)+",expanding table"
            self.expand_alm_table(l)
        alm = self.alm_table.get((l,m))
        if alm is None:
            raise RuntimeError('PolygonGeo: a_lm generation failed for unknown reason at l='+str(l)+',m='+str(m))
        return alm
