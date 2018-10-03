"""provide ability to compute union of PolygonPixelGeos and use PolygonPixelGeos as masks"""
from __future__ import division,print_function,absolute_import
from builtins import range
import numpy as np
from pixel_geo import PixelGeo
class PolygonPixelUnionGeo(PixelGeo):
    """geo for union between several PolygonPixelGeo or masks"""
    def __init__(self,geos,masks,zs=None,z_fine=None):
        """ geos: an array of PolygonPixelGeos to combine
            masks: an array of PolygonPixelGeos to block out
        """
        self.geos = geos
        self.masks = masks
        self.n_g = geos.size
        self.n_m = masks.size

        self.contain_pos = np.zeros_like(self.geos[0].contained)
        self.contain_mask = np.zeros_like(self.geos[0].contained)
        self.all_pixels = geos[0].all_pixels.copy()
        l_max = geos[0].get_current_l_max()
        hard_l_max = geos[0].hard_l_max
        
        for itr1 in range(0,self.n_g):
            self.contain_pos = self.contain_pos | geos[itr1].contained
            l_max = np.min([geos[itr1].get_current_l_max(),l_max])
            assert np.all(self.all_pixels==geos[itr1].all_pixels)
        for itr2 in range(0,self.n_m):
            self.contain_mask = self.contain_mask | masks[itr2].contained
            l_max = np.min([masks[itr2].get_current_l_max(),l_max])
            assert np.all(self.all_pixels==masks[itr2].all_pixels)
        self.contain_mask = self.contain_mask*self.contain_pos
        self.contained = self.contain_pos*(~self.contain_mask).astype(bool)
        contained_pixels = self.all_pixels[self.contained,:]
        if zs is None:
            zs = geos[0].zs.copy() #TODO pointless copy
        if z_fine is  None:
            z_fine = geos[0].z_fine.copy()

        PixelGeo.__init__(self,zs,contained_pixels,geos[0].C,z_fine,l_max,hard_l_max)
