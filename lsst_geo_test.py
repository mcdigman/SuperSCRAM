"""test approximate a circular geometry with PolygonGeo"""
import numpy as np
from cosmopie import CosmoPie
import defaults
from premade_geos import WFIRSTGeo,LSSTGeoSimpl

if __name__=='__main__':
    do_plot = True

    cosmo_fid =  defaults.cosmology.copy()
    C = CosmoPie(cosmo_fid,'jdem')
    l_max = 2
    zs = np.array([0.01,1.])
    z_fine = np.arange(0.01,1.0001,0.01)

#    radius = 0.3126603700269391
#    n_x = 100
#    error_old = 10000000.
#    error_new = 1000000.
#    area_goal = 1000.
#    while error_new<error_old:
#        geo1 = CircleGeo(zs,C,radius,n_x,z_fine,l_max,{'n_double':30})
#        area1 = 2.*np.pi*(1.-np.cos(radius))
#        area2 = geo1.angular_area()
#        radius = radius*np.sqrt(area_goal/(geo1.angular_area()*180**2/np.pi**2))
#        error_old = error_new
#        error_new = np.abs(area2*180**2/np.pi**2-area_goal)
#    print "radius for n_x="+str(n_x)+" 1000 deg^2="+str(radius)
    geo1 = LSSTGeoSimpl(zs,C,z_fine,l_max,{'n_double':30},phi1=2.9030540874480577)
    geo2 = LSSTGeoSimpl(zs,C,z_fine,l_max,{'n_double':30},phi1=2.0*2.9030540874480577)
    geo3 = WFIRSTGeo(zs,C,z_fine,l_max,{'n_double':30})
    if do_plot:
        from mpl_toolkits.basemap import Basemap
        import matplotlib.pyplot as plt
        m = Basemap(projection='moll',lon_0=0)
        geo1.sp_poly.draw(m,color='red')
        geo2.sp_poly.draw(m,color='blue')
        geo3.sp_poly.draw(m,color='blue')
        plt.show()
