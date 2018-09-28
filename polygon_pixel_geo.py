"""a healpix pixelated polygon with great circle sides as in PolygonGeo"""
from __future__ import division,print_function,absolute_import
from builtins import range
from warnings import warn
from math import isnan


import numpy as np

from pixel_geo import PixelGeo
from polygon_utils import get_poly,get_healpix_pixelation,contains_points

class PolygonPixelGeo(PixelGeo):
    """healpix pixelated spherical polygon geo"""
    def __init__(self,zs,thetas,phis,theta_in,phi_in,C,z_fine,l_max,res_healpix):
        """create a spherical polygon defined by vertices
                inputs:
                    zs: the tomographic z bins
                    thetas,phis: an array of theta values for the edges in radians, last value should be first for closure, edges will be clockwise
                    theta_in,phi_in: a theta and phi known to be outside, needed for finding intersect for now
                    C: a CosmoPie object
                    z_fine: the resolution z slices
                    l_max: the maximum l to compute the alm table to
                    res_healpix: 4 to 9, healpix resolution to use
        """
        #self.thetas = thetas
        #self.phis = phis
        self.C = C
        self.z_fine = z_fine
        self.res_healpix = res_healpix
        hard_l_max = 3.*2**self.res_healpix-1.
        self.all_pixels = get_healpix_pixelation(res_choose=self.res_healpix)
        self.sp_poly = get_poly(thetas,phis,theta_in,phi_in)
        self.theta_in = theta_in
        self.phi_in = phi_in
        if isnan(self.sp_poly.area()):
            raise ValueError("PolygonPixelGeo: Calculated area of polygon is nan, polygon likely invalid")
        #self.contained =  is_contained(all_pixels,self.sp_poly)
        self.contained =  contains_points(self.all_pixels,self.sp_poly)
        contained_pixels = self.all_pixels[self.contained,:]
        #self.n_pix = contained_pixels.shape[0]
        print("PolygonPixelGeo: total contained pixels in polygon: "+str(np.sum(self.contained*1.)))
        print("PolygonPixelGeo: total contained area of polygon: "+str(np.sum(contained_pixels[:,2])))
        print("PolygonPixelGeo: area calculated by SphericalPolygon: "+str(self.sp_poly.area()))
        #check that the true area from angular defect formula and calculated area approximately match
        calc_area = np.sum(contained_pixels[:,2])
        true_area = self.sp_poly.area()
        if not np.isclose(calc_area,true_area,atol=contained_pixels[0,2]*100.,rtol=10**-2):
            warn("discrepancy between area "+str(true_area)+" and est "+str(calc_area)+", may be poorly converged")
        PixelGeo.__init__(self,zs,contained_pixels,C,z_fine,l_max,hard_l_max)

    #TODO make robust
    def get_overlap_fraction(self,geo2):
        """get overlap fraction between this geometry and another PolygonPixelGeo"""
        result = np.sum(self.contained*geo2.contained)*1./np.sum(self.contained)
        #result2 = self.sp_poly.overlap(geo2.sp_poly)
        #print("PolygonPixelGeo: my overlap prediction="+str(result)+" spherical_geometry prediction="+str(result2))
        return result
