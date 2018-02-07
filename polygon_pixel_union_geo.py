"""provide ability to compute union of PolygonPixelGeos and use PolygonPixelGeos as masks"""
import numpy as np
from pixel_geo import PixelGeo
class PolygonPixelUnionGeo(PixelGeo):
    """geo for union between several PolygonPixelGeo or masks"""
    def __init__(self,geos,masks):
        """ geos: an array of PolygonPixelGeos to combine
            masks: an array of PolygonPixelGeos to block out
        """
        self.geos = geos
        self.n_g = geos.size
        self.n_m = masks.size

        self.contain_pos = np.zeros_like(self.geos[0].contained)
        self.contain_mask = np.zeros_like(self.geos[0].contained)
        self.all_pixels = geos[0].all_pixels.copy()
        l_max = geos[0].get_current_l_max()
        for itr1 in xrange(0,self.n_g):
            self.contain_pos = self.contain_pos | geos[itr1].contained
            l_max = np.min([geos[itr1].get_current_l_max(),l_max])
            assert np.all(self.all_pixels==geos[itr1].all_pixels)
        for itr2 in xrange(0,self.n_m):
            self.contain_mask = self.contain_mask | masks[itr2].contained
            l_max = np.min([geos[itr2].get_current_l_max(),l_max])
            assert np.all(self.all_pixels==masks[itr2].all_pixels)
        self.contain_mask = self.contain_mask*self.contain_pos
        self.union_contains = self.contain_pos*(~self.contain_mask).astype(bool)
        contained_pixels = self.all_pixels[self.union_contains,:]

        PixelGeo.__init__(self,geos[0].zs.copy(),contained_pixels,geos[0].C,geos[0].z_fine.copy(),l_max)
