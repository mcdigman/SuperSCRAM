"""provide ability to compute union of PolygonPixelGeos and use PolygonPixelGeos as masks"""
import numpy as np
from polygon_pixel_geo import PolygonPixelGeo

class PolygonPixelUnionGeo(PolygonPixelGeo):
    """geo for union between several PolygonPixelGeo or masks"""
    def __init__(self,geos,masks):
        """ geos: an array of PolygonPixelGeos to combine
            masks: an array of PolygonPixelGeos to block out
        """
        self.geos = geos
        self.n_g = geos.size
        self.n_m = masks.size
        #polys_pos = np.zeros(self.n_g,dtype=object)
        #polys_mask = np.zeros(self.n_m,dtype=object)
        contain_pos = np.zeros_like(self.geos[0].contained)
        contain_mask = np.zeros_like(self.geos[0].contained)
        for itr1 in xrange(0,self.n_g):
            contain_pos = contain_pos | geos[itr1].contained
        for itr2 in xrange(0,self.n_m):
            contain_mask = contain_mask | masks[itr2].sp_poly
        union_mask = contain_pos*(not contain_mask)

        #self.union_pos = polys_pos[0]
        #for itr1 in xrange(1,self.n_g):
        #    self.union_pos = self.union_pos.union(polys_pos[itr1])
        #self.union_xyz = list(self.union_pos.points)[0]
        #union_ra,union_dec = sgv.vector_to_radec(self.union_xyz[:,0],self.union_xyz[:,1],self.union_xyz[:,2],degrees=False)
        #self.union_phi = union_ra
        #self.union_theta = union_dec+np.pi/2.
        #self.union_geo = PolygonGeo(geos[0].zs,self.union_theta,self.union_phi,geos[0].theta_in,geos[0].phi_in,geos[0].C,geos[0].z_fine,geos[0].l_max,geos[0].poly_params)
        #if self.n_g>1:
        #    self.union_pos = polys_pos[0].multi_union(polys_pos[1:])
        #    print "multi"
        #elif self.n_g==1:
        #    self.union_pos = polys_pos[0]
        #else:
        #    self.union_pos = None
        #if self.n_m>0:
        #    self.union_mask = polys_mask[0]
        #    for itr1 in xrange(1,self.n_m):
        #        self.union_mask = self.union_mask.union(polys_mask[itr1])
        #    self.union_mask = self.union_mask.intersection(self.union_pos)
        #    self.mask_xyz = list(self.union_mask.points)[0]
        #    mask_ra,mask_dec = sgv.vector_to_radec(self.mask_xyz[:,0],self.mask_xyz[:,1],self.mask_xyz[:,2],degrees=False)
        #    in_point = list(self.union_mask.inside)[0]
        #    in_ra,in_dec = sgv.vector_to_radec(in_point[0],in_point[1],in_point[2],degrees=False)
        #    self.mask_phi = mask_ra
        #    self.mask_theta = mask_dec+np.pi/2.
        #    self.mask_geo = PolygonGeo(geos[0].zs,self.mask_theta,self.mask_phi,in_dec+np.pi/2.,in_ra,geos[0].C,geos[0].z_fine,geos[0].l_max,geos[0].poly_params)
        #else:
        #    self.union_mask = None
        #    self.mask_xyz = None
        #    self.mask_phi = None
        #    self.mask_theta = None
        #    self.mask_geo = None

        #TODO PRIORITY actually use union_mask
        PolygonPixelGeo.__init__(self,geos[0].zs,geos[0].thetas,geos[0].phis,geos[0].theta_in,geos[0].phi_in,geos[0].C,geos[0].z_fine,geos[0]._l_max,geos[0].res_healpix)
        #Geo.__init__(self,self.union_geo.zs,self.union_geo.C,self.union_geo.z_fine)
        #self.alm_table = {(0,0):self.angular_area()/np.sqrt(4.*np.pi)}
        #self._l_max = 0
        #self.expand_alm_table(geos[0].l_max)
        #intersect_polys = np.zeros((self.n_g,self.n_g),dtype=object)
        #for itr1 in xrange(0,self.n_g):
        #    for itr2 in xrange(0,self.n_g):
        #        intersect_polys[itr1,itr2] = geos[itr1].sp_poly.intersection(geos[itr2].sp_poly.intersection)
    #TODO angular_area
#    def angular_area(self):
#        if self.mask_geo is not None:
#            return self.union_geo.angular_area()-self.mask_geo.angular_area()
#        else:
#            return self.union_geo.angular_area()
#    def expand_alm_table(self,l_max):
#        self.union_geo.expand_alm_table(l_max)
#        self.mask_geo.expand_alm_table(l_max)
#        ls = np.arange(np.int(self._l_max)+1,np.int(l_max)+1)
#        if ls.size==0:
#            return
#        for ll in ls:
#            for mm in xrange(-ll,ll+1):
#                self.alm_table[(ll,mm)] = self.union_geo.alm_table[(ll,mm)]-self.mask_geo.alm_table[(ll,mm)]
#        self._l_max = l_max
#
#    def a_lm(self,l,m):
#        if l>self.l_max:
#            print "PolygonUnionGeo: l value "+str(l)+" exceeds maximum precomputed l "+str(self._l_max)+",expanding table"
#            self.expand_alm_table(l)
#        alm = self.alm_table.get((l,m))
#        if alm is None:
#            raise RuntimeError('PolygonGeo: a_lm generation failed for unknown reason at l='+str(l)+',m='+str(m))
#            return alm
