"""demonstrate bug in spherical_geometry, fixed in new versions"""
from __future__ import print_function,absolute_import,division
from builtins import range
import numpy as np
from spherical_geometry.polygon import SphericalPolygon
import spherical_geometry.vector as sgv
import pytest

def test_geos1():
    """do second test posted to bug report"""
    n_fill = 10

    theta1_high_fill = np.full(n_fill,5.)
    theta1_low_fill =np.full(n_fill, -65.)
    phi1_high_fill = np.linspace(50.-360.,50.-1.,n_fill)
    phi1_low_fill = phi1_high_fill[::-1]
    theta1s = np.hstack([theta1_high_fill,theta1_low_fill,theta1_high_fill[0]])
    phi1s = np.hstack([phi1_high_fill,phi1_low_fill,phi1_high_fill[0]])

    phi_in1 = 0.
    theta_in1 = 70.

    bounding_xyz1 = np.array(sgv.radec_to_vector(phi1s,theta1s,degrees=True)).T
    inside_xyz1 = np.array(sgv.radec_to_vector(phi_in1,theta_in1,degrees=True))
    sp_poly1 = SphericalPolygon(bounding_xyz1,inside=inside_xyz1)

    theta2_high_fill = np.full(n_fill,-30.)
    theta2_low_fill =np.full(n_fill, -85.)
    phi2_high_fill = np.linspace(65-360.,55.-10.,n_fill)
    phi2_low_fill = np.linspace(5-360.,5.-10.,n_fill)[::-1]
    theta2s = np.hstack([theta2_high_fill,theta2_low_fill,theta2_high_fill[0]])
    phi2s = np.hstack([phi2_high_fill,phi2_low_fill,phi2_high_fill[0]])

    phi_in2 = 0.
    theta_in2 = 70.

    bounding_xyz2 = np.array(sgv.radec_to_vector(phi2s,theta2s,degrees=True)).T
    inside_xyz2 = np.array(sgv.radec_to_vector(phi_in2,theta_in2,degrees=True))
    sp_poly2 = SphericalPolygon(bounding_xyz2,inside=inside_xyz2)

    int_poly = sp_poly2.intersection(sp_poly1)

    theta_out = -50.*np.pi/180.
    phi_out = 0.
    outside_xyz = np.array(sgv.radec_to_vector(phi_out,theta_out,degrees=False))

    theta_in3 = 70.*np.pi/180.
    phi_in3 = 0.
    inside_xyz3 = np.array(sgv.radec_to_vector(phi_in3,theta_in3,degrees=False))

    print("area int, area 1,area 2: ",int_poly.area(),sp_poly1.area(),sp_poly2.area())
    print("Should be True , True : ",int_poly.area()<=sp_poly1.area(),int_poly.area()<=sp_poly2.area())
    assert int_poly.area()<=sp_poly1.area()
    assert int_poly.area()<=sp_poly2.area()

    assert not int_poly.contains_point(outside_xyz)
    assert not sp_poly1.contains_point(outside_xyz)
    assert not sp_poly2.contains_point(outside_xyz)
    print("Should be True ,True ,True : ",int_poly.contains_point(inside_xyz1),sp_poly1.contains_point(inside_xyz1),sp_poly2.contains_point(inside_xyz1))
    assert int_poly.contains_point(inside_xyz1)
    assert sp_poly1.contains_point(inside_xyz1)
    assert sp_poly2.contains_point(inside_xyz1)
    print("Should be True ,True ,True : ",int_poly.contains_point(inside_xyz2),sp_poly1.contains_point(inside_xyz2),sp_poly2.contains_point(inside_xyz2))
    assert int_poly.contains_point(inside_xyz2)
    assert sp_poly1.contains_point(inside_xyz2)
    assert sp_poly2.contains_point(inside_xyz2)
    print("Should be True ,True ,True : ",int_poly.contains_point(inside_xyz3),sp_poly1.contains_point(inside_xyz3),sp_poly2.contains_point(inside_xyz3))
    assert int_poly.contains_point(inside_xyz3)
    assert sp_poly1.contains_point(inside_xyz3)
    assert sp_poly2.contains_point(inside_xyz3)

    inside_res = list(int_poly.inside)
    for itr in range(0,len(inside_res)):
        print("Should be True ,True ,True : ",int_poly.contains_point(inside_res[itr]),sp_poly1.contains_point(inside_res[itr]),sp_poly2.contains_point(inside_res[itr]))
        assert int_poly.contains_point(inside_res[itr])
        assert sp_poly1.contains_point(inside_res[itr])
        assert sp_poly2.contains_point(inside_res[itr])

def test_geos2():
    """do initial test posted to spherical_geometry problem report"""
    n_fill = 10

    theta1_high_fill = np.full(n_fill,5.)
    theta1_low_fill =np.full(n_fill, -65.)
    phi1_high_fill = np.linspace(50.-360.,50.-1.,n_fill)
    phi1_low_fill = phi1_high_fill[::-1]
    theta1s = np.hstack([theta1_high_fill,theta1_low_fill,theta1_high_fill[0]])
    phi1s = np.hstack([phi1_high_fill,phi1_low_fill,phi1_high_fill[0]])

    phi_in1 = 0.
    theta_in1 = 0.

    bounding_xyz1 = np.array(sgv.radec_to_vector(phi1s,theta1s,degrees=True)).T
    inside_xyz1 = np.array(sgv.radec_to_vector(phi_in1,theta_in1,degrees=True))
    sp_poly1 = SphericalPolygon(bounding_xyz1,inside=inside_xyz1)

    theta2_high_fill = np.full(n_fill,-30.)
    theta2_low_fill =np.full(n_fill, -85.)
    phi2_high_fill = np.linspace(65-360.,55.-10.,n_fill)
    phi2_low_fill = np.linspace(5-360.,5.-10.,n_fill)[::-1]
    theta2s = np.hstack([theta2_high_fill,theta2_low_fill,theta2_high_fill[0]])
    phi2s = np.hstack([phi2_high_fill,phi2_low_fill,phi2_high_fill[0]])

    phi_in2 = 0.
    theta_in2 = -70.

    bounding_xyz2 = np.array(sgv.radec_to_vector(phi2s,theta2s,degrees=True)).T
    inside_xyz2 = np.array(sgv.radec_to_vector(phi_in2,theta_in2,degrees=True))
    sp_poly2 = SphericalPolygon(bounding_xyz2,inside=inside_xyz2)

    int_poly = sp_poly2.intersection(sp_poly1)

    theta_out = 30.*np.pi/180.
    phi_out = 0.
    outside_xyz = np.array(sgv.radec_to_vector(phi_out,theta_out,degrees=False))

    theta_in3 = -40.*np.pi/180.
    phi_in3 = 0.
    inside_xyz3 = np.array(sgv.radec_to_vector(phi_in3,theta_in3,degrees=False))

    assert not int_poly.contains_point(outside_xyz)

    assert not int_poly.contains_point(outside_xyz)
    assert not sp_poly1.contains_point(outside_xyz)
    assert not sp_poly2.contains_point(outside_xyz)
    assert not int_poly.contains_point(inside_xyz1)
    assert sp_poly1.contains_point(inside_xyz1)
    assert not sp_poly2.contains_point(inside_xyz1)
    assert not int_poly.contains_point(inside_xyz2)
    assert not sp_poly1.contains_point(inside_xyz2)
    assert sp_poly2.contains_point(inside_xyz2)
    assert int_poly.contains_point(inside_xyz3)
    assert sp_poly1.contains_point(inside_xyz3)
    assert sp_poly2.contains_point(inside_xyz3)

if __name__=='__main__':
    pytest.cmdline.main(['sph_geo_version_test.py'])
