"""module for holding some common utils between polygon geos"""
import numpy as np
from numpy.core.umath_tests import inner1d
from astropy.io import fits
import spherical_geometry.vector as sgv
from spherical_geometry.polygon import SphericalPolygon
#Note these are spherical polygons so all the sides are great circles (not lines of constant theta!)
#So area will differ from integral if assuming constant theta
#vertices must have same first and last coordinate so polygon is closed
#last point is arbitrary point inside because otherwise 2 polygons possible.
#Behavior may be unpredictable if the inside point is very close to an edge or vertex.
def get_poly(theta_vertices,phi_vertices,theta_in,phi_in):
    """get the SphericalPolygon object for the geometry"""
    bounding_theta = theta_vertices-np.pi/2. #to radec
    bounding_phi = phi_vertices
    bounding_xyz = np.asarray(sgv.radec_to_vector(bounding_phi,bounding_theta,degrees=False)).T
    inside_xyz = np.asarray(sgv.radec_to_vector(phi_in,theta_in-np.pi/2.,degrees=False))

    sp_poly = SphericalPolygon(bounding_xyz,inside=inside_xyz)
    return sp_poly

#can safely resolve up to lmax~2*nside (although can keep going with loss of precision until lmax=3*nside-1), so if lmax=100,need nside~50
#nside = 2^res, so res=6 => nside=64 should safely resolve lmax=100, for extra safety can choose res=7
#res = 10 takes ~164.5  sec
#res = 9 takes ~50.8 sec
#res = 8 takes ~11 sec
#res = 7 takes ~3.37 sec
#res = 6 takes ~0.688 sec
def get_healpix_pixelation(res_choose=6):
    """get healpix pixels for a selected resolution res_choose from 4 to 9"""
    pixel_info = np.loadtxt('data/pixel_info.dat')
    area = pixel_info[res_choose,4]
    #tables from https://lambda.gsfc.nasa.gov/toolbox/tb_pixelcoords.cfm#pixelinfo
    hdulist = fits.open('data/pixel_coords_map_ring_galactic_res'+str(res_choose)+'.fits')
    data = hdulist[1].data
    pixels = np.zeros((data.size,3))

    pixels[:,0] = data['LATITUDE']*np.pi/180.+np.pi/2.
    pixels[:,1] = data['LONGITUDE']*np.pi/180.
    pixels[:,2] = area

    return pixels


def is_contained(pixels,sp_poly):
    """Pixels is a pixelation (ie what get_healpix_pixelation returns) and sp_poly is a spherical polygon, ie from get_poly"""
    #xyz vals for the pixels
    xyz_vals = sgv.radec_to_vector(pixels[:,1],pixels[:,0]-np.pi/2.,degrees=False)
    contained = np.zeros(pixels.shape[0],dtype=bool)
    #check if each point is contained in the polygon. This is fairly slow if the number of points is huge
    for i in xrange(0,pixels.shape[0]):
        contained[i]= sp_poly.contains_point([xyz_vals[0][i],xyz_vals[1][i],xyz_vals[2][i]])
    return contained

def contains_points(pixels,sp_poly):
    """contains procedure adapted from spherical_geometry but pixels can be a vector so faster"""
    xyz_vals = np.array(sgv.radec_to_vector(pixels[:,1],pixels[:,0]-np.pi/2.,degrees=False)).T
    intersects = np.zeros(pixels.shape[0],dtype=int)
    bounding_xyz = sp_poly._polygons[0]._points
    inside_xyz = sp_poly._polygons[0]._inside
    inside_large = np.zeros_like(xyz_vals)
    inside_large+=inside_xyz
    for itr in xrange(0,bounding_xyz.shape[0]-1):
        #intersects+= great_circle_arc.intersects(bounding_xyz[itr], bounding_xyz[itr+1], inside_large, xyz_vals)
        intersects+= contains_intersect(bounding_xyz[itr], bounding_xyz[itr+1], inside_xyz, xyz_vals)
    return np.mod(intersects,2)==0

def contains_intersect(vertex1,vertex2,inside_point,test_points):
    """adapted from spherical_geometry.great_circle_arc.intersects, but much faster for our purposes
    may behave unpredicatably if one of the points is exactly on an edge"""
    cxd = np.cross(inside_point,test_points)
    axb = np.cross(vertex1,vertex2)
    #T doesn't need to be normalized because we only want signs
    T = np.cross(axb,cxd)
    sign1 = np.sign(np.inner(np.cross(axb,vertex1),T))
    sign2 = np.sign(np.inner(np.cross(vertex2,axb),T))
    #row wise dot product is inner1d
    sign3 = np.sign(inner1d(np.cross(cxd,inside_point),T))
    sign4 = np.sign(inner1d(np.cross(test_points,cxd),T))
    return (sign1==sign2) & (sign1==sign3) & (sign1==sign4)
