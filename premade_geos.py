"""some premade tesing geometries"""
from __future__ import division,print_function,absolute_import
from builtins import range
import numpy as np
from astropy.coordinates import SkyCoord
from polygon_geo import PolygonGeo
from polygon_pixel_geo import PolygonPixelGeo
from polygon_union_geo import PolygonUnionGeo
from polygon_pixel_union_geo import PolygonPixelUnionGeo
from geo_display_utils import display_geo

class WFIRSTGeo(PolygonGeo):
    """replicate WFIRST geometry"""
    def __init__(self,zs,C,z_fine,l_max,poly_params):
        """params as PolygonGeo"""
        thetas_wfirst = np.array([-50.,-35.,-35.,-19.,-19.,-19.,-15.8,-15.8,-40.,-40.,-55.,-78.,-78.,-78.,-55.,-55.,-50.,-50.])*np.pi/180.+np.pi/2.
        phis_wfirst = np.array([-19.,-19.,-11.,-11.,7.,25.,25.,43.,43.,50.,50.,50.,24.,5.,5.,7.,7.,-19.])*np.pi/180.
        phi_in_wfirst = 7./180.*np.pi
        theta_in_wfirst = -35.*np.pi/180.+np.pi/2.
        print("main: begin constructing WFIRST PolygonGeo")
        PolygonGeo.__init__(self,zs,thetas_wfirst,phis_wfirst,theta_in_wfirst,phi_in_wfirst,C,z_fine,l_max,poly_params)

class LSSTGeo(PolygonUnionGeo):
    """replicate LSST geometry"""
    def __init__(self,zs,C,z_fine,l_max,poly_params,theta_high=5.,theta_low=-65.,galactic_width=20.,n_fill=20):
        self.poly_params = poly_params
        self.l_max = l_max
        #theta_high = np.pi/2.+5.*np.pi/180.
        #theta_low = np.pi/2.-65.*np.pi/180.
        #theta_high_fill = np.full(n_fill,theta_high)
        #theta_low_fill = np.full(n_fill,theta_low)
        #theta2s = np.hstack([[theta_high],theta_high_fill,[theta_high,theta_low],theta_low_fill,[theta_low,theta_high]])
        theta2r_high_fill = np.full(n_fill,theta_high)
        theta2r_low_fill = np.full(n_fill, theta_low)
        phi2r_high_fill = np.linspace(180.-360.,180.-0.0001,n_fill)
        phi2r_low_fill = phi2r_high_fill[::-1]
        theta2rs = np.hstack([theta2r_high_fill,theta2r_low_fill,theta2r_high_fill[0]])
        phi2rs = np.hstack([phi2r_high_fill,phi2r_low_fill,phi2r_high_fill[0]])

        theta2s = np.zeros_like(theta2rs)
        phi2s = np.zeros_like(theta2rs)
        for itr in range(0,theta2rs.size):
            coord_gal = SkyCoord(phi2rs[itr], theta2rs[itr], frame='icrs', unit='deg')
            theta2s[itr] = coord_gal.geocentrictrueecliptic.lat.rad+np.pi/2.
            phi2s[itr] = coord_gal.geocentrictrueecliptic.lon.rad
        theta_in2 = SkyCoord(0.,0.,frame='icrs',unit='deg').geocentrictrueecliptic.lat.rad+np.pi/2.
        phi_in2 = SkyCoord(0.,0.,frame='icrs',unit='deg').geocentrictrueecliptic.lon.rad

        poly_geo2 = PolygonGeo(zs,theta2s,phi2s,theta_in2,phi_in2,C,z_fine,l_max,poly_params)

        thetar_high_fill = np.full(n_fill,galactic_width)
        thetar_low_fill = np.full(n_fill, -galactic_width)
        phir_high_fill = np.linspace(160.-360.,160.-galactic_width,n_fill)
        phir_low_fill = np.linspace(160.-360.,160.-galactic_width,n_fill)[::-1]
        thetars = np.hstack([thetar_high_fill,thetar_low_fill,thetar_high_fill[0]])
        phirs = np.hstack([phir_high_fill,phir_low_fill,phir_high_fill[0]])

        thetas_mask = np.zeros_like(thetars)
        phis_mask = np.zeros_like(thetars)
        for itr in range(0,thetars.size):
            coord_gal = SkyCoord(phirs[itr], thetars[itr], frame='galactic', unit='deg')
            thetas_mask[itr] = coord_gal.geocentrictrueecliptic.lat.rad+np.pi/2.
            phis_mask[itr] = coord_gal.geocentrictrueecliptic.lon.rad
        theta_in_mask = SkyCoord(0.,0.,frame='galactic',unit='deg').geocentrictrueecliptic.lat.rad+np.pi/2.
        phi_in_mask = SkyCoord(0.,0.,frame='galactic',unit='deg').geocentrictrueecliptic.lon.rad
        mask_geo = PolygonGeo(zs,thetas_mask,phis_mask,theta_in_mask,phi_in_mask,C,z_fine,0,poly_params)

        PolygonUnionGeo.__init__(self,np.array([poly_geo2]),np.array([mask_geo]),l_max=l_max)

class WFIRSTPixelGeo(PolygonPixelGeo):
    """replicate WFIRST geometry"""
    def __init__(self,zs,C,z_fine,l_max,res_healpix):
        """params as PolygonGeo"""
        thetas_wfirst = np.array([-50.,-35.,-35.,-19.,-19.,-19.,-15.8,-15.8,-40.,-40.,-55.,-78.,-78.,-78.,-55.,-55.,-50.,-50.])*np.pi/180.+np.pi/2.
        phis_wfirst = np.array([-19.,-19.,-11.,-11.,7.,25.,25.,43.,43.,50.,50.,50.,24.,5.,5.,7.,7.,-19.])*np.pi/180.
        phi_in_wfirst = 7./180.*np.pi
        theta_in_wfirst = -35.*np.pi/180.+np.pi/2.
        print("main: begin constructing WFIRST PolygonPixelGeo")
        PolygonPixelGeo.__init__(self,zs,thetas_wfirst,phis_wfirst,theta_in_wfirst,phi_in_wfirst,C,z_fine,l_max,res_healpix)

class LSSTPixelGeo(PolygonPixelUnionGeo):
    """replicate pixelated LSST geometry"""
    def __init__(self,zs,C,z_fine,l_max,res_healpix):
        n_fill = 20
        #theta_high = np.pi/2.+5.*np.pi/180.
        #theta_low = np.pi/2.-65.*np.pi/180.
        #theta_high_fill = np.full(n_fill,theta_high)
        #theta_low_fill = np.full(n_fill,theta_low)
        #theta2s = np.hstack([[theta_high],theta_high_fill,[theta_high,theta_low],theta_low_fill,[theta_low,theta_high]])
        theta2r_high_fill = np.full(n_fill,5.)
        theta2r_low_fill = np.full(n_fill, -65.)
        phi2r_high_fill = np.linspace(180.-360.,180.-0.0001,n_fill)
        phi2r_low_fill = phi2r_high_fill[::-1]
        theta2rs = np.hstack([theta2r_high_fill,theta2r_low_fill,theta2r_high_fill[0]])
        phi2rs = np.hstack([phi2r_high_fill,phi2r_low_fill,phi2r_high_fill[0]])

        theta2s = np.zeros_like(theta2rs)
        phi2s = np.zeros_like(theta2rs)
        for itr in range(0,theta2rs.size):
            coord_gal = SkyCoord(phi2rs[itr], theta2rs[itr], frame='icrs', unit='deg')
            theta2s[itr] = coord_gal.geocentrictrueecliptic.lat.rad+np.pi/2.
            phi2s[itr] = coord_gal.geocentrictrueecliptic.lon.rad
        theta_in2 = SkyCoord(0.,0.,frame='icrs',unit='deg').geocentrictrueecliptic.lat.rad+np.pi/2.
        phi_in2 = SkyCoord(0.,0.,frame='icrs',unit='deg').geocentrictrueecliptic.lon.rad

        poly_geo2 = PolygonPixelGeo(zs,theta2s,phi2s,theta_in2,phi_in2,C,z_fine,l_max,res_healpix)

        thetar_high_fill = np.full(n_fill,20.)
        thetar_low_fill = np.full(n_fill, -20.)
        phir_high_fill = np.linspace(160.-360.,160.-20.,n_fill)
        phir_low_fill = np.linspace(160.-360.,160.-20.,n_fill)[::-1]
        thetars = np.hstack([thetar_high_fill,thetar_low_fill,thetar_high_fill[0]])
        phirs = np.hstack([phir_high_fill,phir_low_fill,phir_high_fill[0]])

        thetas_mask = np.zeros_like(thetars)
        phis_mask = np.zeros_like(thetars)
        for itr in range(0,thetars.size):
            coord_gal = SkyCoord(phirs[itr], thetars[itr], frame='galactic', unit='deg')
            thetas_mask[itr] = coord_gal.geocentrictrueecliptic.lat.rad+np.pi/2.
            phis_mask[itr] = coord_gal.geocentrictrueecliptic.lon.rad
        theta_in_mask = SkyCoord(0.,0.,frame='galactic',unit='deg').geocentrictrueecliptic.lat.rad+np.pi/2.
        phi_in_mask = SkyCoord(0.,0.,frame='galactic',unit='deg').geocentrictrueecliptic.lon.rad
        mask_geo = PolygonPixelGeo(zs,thetas_mask,phis_mask,theta_in_mask,phi_in_mask,C,z_fine,l_max,res_healpix)

        PolygonPixelUnionGeo.__init__(self,np.array([poly_geo2]),np.array([mask_geo]))

class LSSTGeoSimpl(PolygonGeo):
    """create a simplified LLST area geo with no masks"""
    def __init__(self,zs,C,z_fine,l_max,poly_params,phi0=0.,phi1=2.9030540874480577,deg0=-65,deg1=5):
        """params as in PolygonGeo"""
        #use the same redshift bin structure as for WFIRST because we only want LSST for galaxy counts, not lensing
        theta0 = deg0*np.pi/180.+np.pi/2.
        theta1 = deg1*np.pi/180.+np.pi/2.
        #phi1 = 3.074096023740458
        #phi1 = 2.9030540874480577
        thetas_lsst = np.array([theta0,theta1,theta1,theta0,theta0])
        phis_lsst = np.array([phi0,phi0,phi1,phi1,phi0])-phi1/2.
        theta_in_lsst = (theta0+theta1)/2.
        phi_in_lsst = (phi0+phi1)/2.-phi1/2.

        print("main: begin constructing LSST PolygonGeo")
        PolygonGeo.__init__(self,zs,thetas_lsst,phis_lsst,theta_in_lsst,phi_in_lsst,C,z_fine,l_max,poly_params)

class StripeGeo(PolygonGeo):
    """replicate LSST geometry"""
    def __init__(self,zs,C,z_fine,l_max,poly_params,theta_high=5.,theta_low=-65.,left_width=20.,right_width=20.,n_fill=20):
        self.poly_params = poly_params
        self.l_max = l_max
        #theta_high = np.pi/2.+5.*np.pi/180.
        #theta_low = np.pi/2.-65.*np.pi/180.
        #theta_high_fill = np.full(n_fill,theta_high)
        #theta_low_fill = np.full(n_fill,theta_low)
        #theta2s = np.hstack([[theta_high],theta_high_fill,[theta_high,theta_low],theta_low_fill,[theta_low,theta_high]])
        theta2r_high_fill = np.full(n_fill,theta_high)
        theta2r_low_fill = np.full(n_fill, theta_low)
        phi2r_high_fill = np.linspace(180.-360.+left_width,180.-0.0001-right_width,n_fill)
        phi2r_low_fill = phi2r_high_fill[::-1]
        theta2rs = np.hstack([theta2r_high_fill,theta2r_low_fill,theta2r_high_fill[0]])
        phi2rs = np.hstack([phi2r_high_fill,phi2r_low_fill,phi2r_high_fill[0]])

        theta2s = np.zeros_like(theta2rs)
        phi2s = np.zeros_like(theta2rs)
        for itr in range(0,theta2rs.size):
            coord_gal = SkyCoord(phi2rs[itr], theta2rs[itr], frame='icrs', unit='deg')
            theta2s[itr] = coord_gal.geocentrictrueecliptic.lat.rad+np.pi/2.
            phi2s[itr] = coord_gal.geocentrictrueecliptic.lon.rad
        theta_in2 = SkyCoord(0.,0.,frame='icrs',unit='deg').geocentrictrueecliptic.lat.rad+np.pi/2.
        phi_in2 = SkyCoord(0.,0.,frame='icrs',unit='deg').geocentrictrueecliptic.lon.rad

        PolygonGeo.__init__(self,zs,theta2s,phi2s,theta_in2,phi_in2,C,z_fine,l_max,poly_params)

#        thetar_high_fill = np.full(n_fill,galactic_width)
#        thetar_low_fill = np.full(n_fill, -galactic_width)
#        phir_high_fill = np.linspace(160.-360.,160.-galactic_width,n_fill)
#        phir_low_fill = np.linspace(160.-360.,160.-galactic_width,n_fill)[::-1]
#        thetars = np.hstack([thetar_high_fill,thetar_low_fill,thetar_high_fill[0]])
#        phirs = np.hstack([phir_high_fill,phir_low_fill,phir_high_fill[0]])
#
#        thetas_mask = np.zeros_like(thetars)
#        phis_mask = np.zeros_like(thetars)
#        for itr in range(0,thetars.size):
#            coord_gal = SkyCoord(phirs[itr], thetars[itr], frame='galactic', unit='deg')
#            thetas_mask[itr] = coord_gal.geocentrictrueecliptic.lat.rad+np.pi/2.
#            phis_mask[itr] = coord_gal.geocentrictrueecliptic.lon.rad
#        theta_in_mask = SkyCoord(0.,0.,frame='galactic',unit='deg').geocentrictrueecliptic.lat.rad+np.pi/2.
#        phi_in_mask = SkyCoord(0.,0.,frame='galactic',unit='deg').geocentrictrueecliptic.lon.rad
#        mask_geo = PolygonGeo(zs,thetas_mask,phis_mask,theta_in_mask,phi_in_mask,C,z_fine,0,poly_params)
#
#        PolygonUnionGeo.__init__(self,np.array([poly_geo2]),np.array([mask_geo]),l_max=l_max)

class WFIRSTPixelGeo(PolygonPixelGeo):
    """replicate WFIRST geometry"""
    def __init__(self,zs,C,z_fine,l_max,res_healpix):
        """params as PolygonGeo"""
        thetas_wfirst = np.array([-50.,-35.,-35.,-19.,-19.,-19.,-15.8,-15.8,-40.,-40.,-55.,-78.,-78.,-78.,-55.,-55.,-50.,-50.])*np.pi/180.+np.pi/2.
        phis_wfirst = np.array([-19.,-19.,-11.,-11.,7.,25.,25.,43.,43.,50.,50.,50.,24.,5.,5.,7.,7.,-19.])*np.pi/180.
        phi_in_wfirst = 7./180.*np.pi
        theta_in_wfirst = -35.*np.pi/180.+np.pi/2.
        print("main: begin constructing WFIRST PolygonPixelGeo")
        PolygonPixelGeo.__init__(self,zs,thetas_wfirst,phis_wfirst,theta_in_wfirst,phi_in_wfirst,C,z_fine,l_max,res_healpix)
