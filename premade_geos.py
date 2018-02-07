"""some premade tesing geometries"""
import numpy as np
from astropy.coordinates import SkyCoord
from polygon_geo import PolygonGeo
from polygon_union_geo import PolygonUnionGeo

class WFIRSTGeo(PolygonGeo):
    """replicate WFIRST geometry"""
    def __init__(self,zs,C,z_fine,l_max,poly_params):
        """params as PolygonGeo"""
        thetas_wfirst = np.array([-50.,-35.,-35.,-19.,-19.,-19.,-15.8,-15.8,-40.,-40.,-55.,-78.,-78.,-78.,-55.,-55.,-50.,-50.])*np.pi/180.+np.pi/2.
        phis_wfirst = np.array([-19.,-19.,-11.,-11.,7.,25.,25.,43.,43.,50.,50.,50.,24.,5.,5.,7.,7.,-19.])*np.pi/180.
        phi_in_wfirst = 7./180.*np.pi
        theta_in_wfirst = -35.*np.pi/180.+np.pi/2.
        print "main: begin constructing WFIRST PolygonGeo"
        PolygonGeo.__init__(self,zs,thetas_wfirst,phis_wfirst,theta_in_wfirst,phi_in_wfirst,C,z_fine,l_max,poly_params)

class LSSTGeo(PolygonUnionGeo):
    """replicate LSST geometry"""
    def __init__(self,zs,C,z_fine,l_max,poly_params):
        n_fill = 20
        #theta_high = np.pi/2.+5.*np.pi/180.
        #theta_low = np.pi/2.-65.*np.pi/180.
        #theta_high_fill = np.full(n_fill,theta_high)
        #theta_low_fill = np.full(n_fill,theta_low)
        #theta2s = np.hstack([[theta_high],theta_high_fill,[theta_high,theta_low],theta_low_fill,[theta_low,theta_high]])
        theta2r_high_fill = np.full(n_fill,5.)
        theta2r_low_fill = np.full(n_fill, -65.)
        phi2r_high_fill = np.linspace(180.-360.,180.-1.,n_fill)
        phi2r_low_fill = phi2r_high_fill[::-1]
        theta2rs = np.hstack([theta2r_high_fill,theta2r_low_fill,theta2r_high_fill[0]])
        phi2rs = np.hstack([phi2r_high_fill,phi2r_low_fill,phi2r_high_fill[0]])

        theta2s = np.zeros_like(theta2rs)
        phi2s = np.zeros_like(theta2rs)
        for itr in xrange(0,theta2rs.size):
            coord_gal = SkyCoord(phi2rs[itr], theta2rs[itr], frame='icrs', unit='deg')
            theta2s[itr] = coord_gal.geocentrictrueecliptic.lat.rad+np.pi/2.
            phi2s[itr] = coord_gal.geocentrictrueecliptic.lon.rad
        theta_in2 = SkyCoord(0.,0.,frame='icrs',unit='deg').geocentrictrueecliptic.lat.rad+np.pi/2.
        phi_in2 = SkyCoord(0.,0.,frame='icrs',unit='deg').geocentrictrueecliptic.lon.rad

        poly_geo2 = PolygonGeo(zs,theta2s,phi2s,theta_in2,phi_in2,C,z_fine,l_max,poly_params)

        thetar_high_fill = np.full(n_fill,20.)
        thetar_low_fill = np.full(n_fill, -20.)
        phir_high_fill = np.linspace(160.-360.,160.-20.,n_fill)
        phir_low_fill = np.linspace(160.-360.,160.-20.,n_fill)[::-1]
        thetars = np.hstack([thetar_high_fill,thetar_low_fill,thetar_high_fill[0]])
        phirs = np.hstack([phir_high_fill,phir_low_fill,phir_high_fill[0]])

        thetas_mask = np.zeros_like(thetars)
        phis_mask = np.zeros_like(thetars)
        for itr in xrange(0,thetars.size):
            coord_gal = SkyCoord(phirs[itr], thetars[itr], frame='galactic', unit='deg')
            thetas_mask[itr] = coord_gal.geocentrictrueecliptic.lat.rad+np.pi/2.
            phis_mask[itr] = coord_gal.geocentrictrueecliptic.lon.rad
        theta_in_mask = SkyCoord(0.,0.,frame='galactic',unit='deg').geocentrictrueecliptic.lat.rad+np.pi/2.
        phi_in_mask = SkyCoord(0.,0.,frame='galactic',unit='deg').geocentrictrueecliptic.lon.rad
        mask_geo = PolygonGeo(zs,thetas_mask,phis_mask,theta_in_mask,phi_in_mask,C,z_fine,l_max,poly_params)

        PolygonUnionGeo.__init__(self,np.array([poly_geo2]),np.array([mask_geo]))
