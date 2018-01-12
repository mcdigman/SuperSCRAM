"""a healpix pixelated polygon with great circle sides as in PolygonGeo"""
from warnings import warn
from math import isnan

from geo import PixelGeo
from polygon_geo import get_poly

import numpy as np
from numpy.core.umath_tests import inner1d

from astropy.io import fits
import spherical_geometry.vector as sgv
#from spherical_geometry.polygon import SphericalPolygon

from ylm_utils import get_a_lm_table

import defaults

class PolygonPixelGeo(PixelGeo):
    def __init__(self,zs,thetas,phis,theta_in,phi_in,C,z_fine,l_max,res_healpix=defaults.polygon_params['res_healpix'],overwride_precompute=False):
        """get a healpix pixelated spherical polygon geo"""
        self.thetas = thetas
        self.phis = phis
        self.theta_in = theta_in
        self.phi_in = phi_in
        self.C = C
        self.z_fine = z_fine
        self.res_healpix = res_healpix
        self.overwride_precompute = overwride_precompute
        all_pixels = get_healpix_pixelation(res_choose=res_healpix)
        self.sp_poly = get_poly(thetas,phis,theta_in,phi_in)
        if isnan(self.sp_poly.area()):
            raise ValueError("PolygonPixelGeo: Calculated area of polygon is nan, polygon likely invalid")
        #self.contained =  is_contained(all_pixels,self.sp_poly)
        self.contained =  contains_points(all_pixels,self.sp_poly)
        contained_pixels = all_pixels[self.contained,:]
        self.n_pix = contained_pixels.shape[0]
        self.all_pixels = all_pixels
        print "PolygonPixelGeo: total contained pixels in polygon: "+str(np.sum(self.contained*1.))
        print "PolygonPixelGeo: total contained area of polygon: "+str(np.sum(contained_pixels[:,2]))
        print "PolygonPixelGeo: area calculated by SphericalPolygon: "+str(self.sp_poly.area())
        #check that the true area from angular defect formula and calculated area approximately match
        calc_area = np.sum(contained_pixels[:,2])
        if not np.isclose(calc_area,self.sp_poly.area(),atol=10**-2,rtol=10**-3):
            warn("PolygonPixelGeo: significant discrepancy between true area "+str(self.sp_poly.area())+" and calculated area"+str(np.sum(contained_pixels[:,2]))+" results may be poorly converged")
        PixelGeo.__init__(self,zs,contained_pixels,C,z_fine)

        #set a00 to value from pixels for consistency, not angle defect even though angle defect is more accurate
        self.alm_table[(0,0)] = calc_area/np.sqrt(4.*np.pi)
        #precompute a table of alms
        #allow overwriding precompute for testing, should not really do this otherwise
        if not overwride_precompute:
            self.alm_table,_,_,self.alm_dict = self.get_a_lm_table(l_max)
            self._l_max = l_max
        else:
            self.alm_table = {}
            self._l_max = 0


    def a_lm(self,l,m):
        #if not precomputed, regenerate table up to specified l, otherwise read it out of the table
        if l>self._l_max:
            print "PolygonPixelGeo: l value "+str(l)+" exceeds maximum precomputed l "+str(self._l_max)+",expanding table"
            self.alm_table,_,_,self.alm_dict = self.get_a_lm_table(l)
            self._l_max = l
        alm = self.alm_table.get((l,m))
        if alm is None:
            raise RuntimeError("PolygonPixelGeo: alm evaluated to None at l="+str(l)+",m="+str(m)+". l,m may exceed highest available Ylm")
        return alm


    #first try loop takes 22.266 sec for l_max=5 for 290 pixel region
    #second try vectorizing Y_r 0.475 sec l_max=5 (47 times speed up)
    #second try 193.84 sec l_max=50
    #third try (with recurse) 2.295 sec for l_max=50 (84 times speed up over second)
    #third try (with recurse) 0.0267 sec for l_max=5
    #third try 9.47 sec for l_max=100
    #oddly, suddenly gets faster with more pixels; maybe some weird numpy internals issue
    #third try with 16348 pixels l_max =100 takes 0.9288-1.047ss
    #fourth try (precompute some stuff), 16348 pixels l_max=100 takes 0.271s
    #fourth try l_max=50 takes 0.0691s, total ~2800x speed up over 1st try
    def get_a_lm_below_l_max(self,l_max):
        a_lms = {}
        ls = np.zeros((l_max+1)**2)
        ms = np.zeros((l_max+1)**2)

        itr = 0

        for ll in xrange(0,l_max+1):
            for mm in xrange(-ll,ll+1):
                #first try takes ~0.618 sec/iteration for 290 pixel region=> 2.1*10**-3 sec/(iteration pixel), much too slow
                ls[itr] = ll
                ms[itr] = mm
                a_lms[(ll,mm)] = self.a_lm(ll,mm)
                itr+=1
        return a_lms,ls,ms

    #TODO check numerical stability
    def get_a_lm_table(self,l_max):
        return get_a_lm_table(l_max,self.pixels[:,0],self.pixels[:,1],self.pixels[0,2]) 

#    def get_a_lm_table(self,l_max):
#        if l_max>85:
#            raise ValueError('cannot use scipy precision for getting alm for l>85 because 171! is too large')
#        n_tot = (l_max+1)**2
#        pixel_area = self.pixels[0,2]
#
#        ls = np.zeros(n_tot)
#        ms = np.zeros(n_tot)
#        a_lms = {}
#        ### verbatim in 5 locations
#        #TODO merge this known_legendre logic into ylm_utils
#        ###identical to 39 in ylm_utils
#        #TODO n_t for testing only
#        n_t = self.n_pix 
#
#        lm_dict,ls,ms = get_lm_dict(l_max)
#
#        cos_theta = np.cos(self.pixels[:,0])
#        sin_theta = np.sin(self.pixels[:,0])
#        abs_sin_theta = np.abs(sin_theta)
#
#
#        sin_phi_m = np.zeros((l_max+1,n_t))
#        cos_phi_m = np.zeros((l_max+1,n_t))
#        for mm in xrange(0,l_max+1):
#            sin_phi_m[mm] = np.sin(mm*self.pixels[:,1])
#            cos_phi_m[mm] = np.cos(mm*self.pixels[:,1])
#
#        factorials = sp.misc.factorial(np.arange(0,2*l_max+1))
#
#        known_legendre = {(0,0):(np.zeros(n_t)+1.),(1,0):cos_theta,(1,1):-abs_sin_theta}
#        for ll in xrange(0,l_max+1):
#            if ll>=2:
#                known_legendre[(ll,ll-1)] = (2.*ll-1.)*cos_theta*known_legendre[(ll-1,ll-1)]
#                known_legendre[(ll,ll)] = -(2.*ll-1.)*abs_sin_theta*known_legendre[(ll-1,ll-1)]
#            for mm in xrange(0,ll+1):
#                if mm<=ll-2:
#                    known_legendre[(ll,mm)] = ((2.*ll-1.)/(ll-mm)*cos_theta*known_legendre[(ll-1,mm)]-(ll+mm-1.)/(ll-mm)*known_legendre[(ll-2,mm)])
#                prefactor = np.sqrt((2.*ll+1.)/(4.*np.pi)*factorials[ll-mm]/factorials[ll+mm])
#                base = known_legendre[(ll,mm)]
#                #no sin theta because first order integrator
#                if mm==0:
#                    a_lms[(ll,mm)] = pixel_area*prefactor*np.sum(base)
#                else:
#                    #Note: check condon shortley phase convention
#
#                    a_lms[(ll,mm)] = (-1)**(mm)*np.sqrt(2.)*pixel_area*prefactor*np.sum(base*cos_phi_m[mm])
#                    a_lms[(ll,-mm)] = (-1)**(mm)*np.sqrt(2.)*pixel_area*prefactor*np.sum(base*sin_phi_m[mm])
#                if mm<=ll-2:
#                    known_legendre.pop((ll-2,mm),None)
#
#        return a_lms,ls,ms,lm_dict


    #TODO make robust
    def get_overlap_fraction(self,geo2):
        result = np.sum(self.contained*geo2.contained)*1./np.sum(self.contained)
        #result2 = self.sp_poly.overlap(geo2.sp_poly)
        #print "PolygonPixelGeo: my overlap prediction="+str(result)+" spherical_geometry prediction="+str(result2)
        return result


#can safely resolve up to lmax~2*nside (although can keep going with loss of precision until lmax=3*nside-1), so if lmax=100,need nside~50
#nside = 2^res, so res=6 => nside=64 should safely resolve lmax=100, for extra safety can choose res=7
#res = 10 takes ~164.5  sec
#res = 9 takes ~50.8 sec
#res = 8 takes ~11 sec
#res = 7 takes ~3.37 sec
#res = 6 takes ~0.688 sec
def get_healpix_pixelation(res_choose=6):
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


#Pixels is a pixelation (ie what get_healpix_pixelation returns) and sp_poly is a spherical polygon, ie from get_poly
def is_contained(pixels,sp_poly):
    #xyz vals for the pixels
    xyz_vals = sgv.radec_to_vector(pixels[:,1],pixels[:,0]-np.pi/2.,degrees=False)
    contained = np.zeros(pixels.shape[0],dtype=bool)
    #check if each point is contained in the polygon. This is fairly slow if the number of points is huge
    for i in xrange(0,pixels.shape[0]):
        contained[i]= sp_poly.contains_point([xyz_vals[0][i],xyz_vals[1][i],xyz_vals[2][i]])
    return contained

#contains procedure adapted from spherical_geometry but pixels can be a vector
def contains_points(pixels,sp_poly):
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

#adapted from spherical_geometry.great_circle_arc.intersects, but much faster for our purposes
#may behave unpredicatably if one of the points is exactly on an edge
def contains_intersect(vertex1,vertex2,inside_point,test_points):
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

#if __name__=='__main__':
#    from geo import PixelGeo,RectGeo
#    from time import time
#    from cosmopie import CosmoPie
#    theta0=0.
#    theta1=np.pi/2.
#    phi0=0.
#    phi1=2.*np.pi/6.
#
#    thetas = np.array([theta0,theta1,theta1,theta0,theta0])
#    phis = np.array([phi0,phi0,phi1,phi1,phi0])
#    theta_in = np.pi/4.
#    phi_in = np.pi/6.
#    res_choose = 10
#    #pixels = get_healpix_pixelation(res_choose=res_choose)
#    #sp_poly = get_poly(thetas,phis,theta_in,phi_in)
#    #contained = is_contained(pixels,sp_poly)
#
#
#    #some setup to make an actual geo
#    d=np.loadtxt('camb_m_pow_l.dat')
#    k=d[:,0]; P=d[:,1]
#    C=CosmoPie(k=k,P_lin=P,cosmology=defaults.cosmology)
#    zs=np.array([.01,1.01])
#    z_fine = np.arange(defaults.lensing_params['z_min_integral'],np.max(zs),defaults.lensing_params['z_resolution'])
#
#    l_max = 25
#    n_run = 1
#    do_old = False
#    do_rect = False
#    try_plot = False
#    try_plot2 = False
#    do_reconstruct = False
#
#    t0 = time()
#    pp_geo = PolygonPixelGeo(zs,thetas,phis,theta_in,phi_in,C,z_fine,l_max=l_max,res_healpix=res_choose)
#    t1 = time()
#    print "instantiation finished in time: "+str(t1-t0)+"s"
#    #TODO write explicit test case to compare
#    if do_old:
#        print "PolygonPixelGeo: initialization time: "+str(t1-t0)+"s"
#        alm_pps,ls,ms = pp_geo.get_a_lm_below_l_max(l_max)
#    t2 = time()
#    if do_old:
#        print "PolygonPixelGeo: a_lm up to l="+str(l_max)+" time: "+str(t2-t1)+"s"
#    for i in xrange(0,n_run):
#        alm_recurse,ls,ms,_= pp_geo.get_a_lm_table(l_max)
#    t3 = time()
#    print "PolygonPixelGeo: a_lm_recurse in avg time: "+str((t3-t2)/n_run)+"s"
#    if do_old:
#        print "methods match: "+str(np.allclose(alm_pps,alm_recurse))
#
#    if do_rect:
#        r_geo = RectGeo(zs,np.array([theta0,theta1]),np.array([phi0,phi1]),C,z_fine)
#        if do_reconstruct:
#            alm_rect = {}
#            for itr in xrange(0,ls.size):
#                alm_rect[(ls[itr],ms[itr])] = r_geo.a_lm(ls[itr],ms[itr])
#    t4 =time()
#    if do_rect:
#        print "RectGeo: rect geo alms in time"+str(t4-t3)
#
#    #totals_recurse = np.zeros(pp_geo.all_pixels.shape[0])
#    if do_reconstruct:
#        totals_recurse = reconstruct_from_alm(l_max,pp_geo.all_pixels[:,0],pp_geo.all_pixels[:,1],alm_recurse)
#    #totals_old = reconstruct_from_alm(l_max,pp_geo.all_pixels[:,0],pp_geo.all_pixels[:,1],alm_pps)
#    #total_reconstruct = SmoothBivariateSpline(pp_geo.all_pixels[:,0],pp_geo.all_pixels[:,1],totals_recurse)
#    #orig_spline =  SmoothBivariateSpline(pp_geo.all_pixels[:,0],pp_geo.all_pixels[:,1],pp_geo.contained*1.)
#    #if do_rect:
#    #    totals_rect = np.zeros(pp_geo.all_pixels.shape[0])
#    #Y_r_2s = pp_geo.get_Y_r_table(l_max,pp_geo.all_pixels[:,0],pp_geo.all_pixels[:,1])
#
#    t5 = time()
#    print "Y_r_2 table time: "+str(t5-t4)+"s"
#    if do_rect and do_reconstruct:
#        totals_rect = reconstruct_from_alm(l_max,pp_geo.all_pixels[:,0],pp_geo.all_pixels[:,1],alm_rect)
#    #if do_rect:
#    #    for itr in xrange(0,ls.size):
#    #        totals_rect+=alm_rect[itr]*Y_r(ls[itr],ms[itr],pp_geo.all_pixels[:,0],pp_geo.all_pixels[:,1])
#    #        totals_recurse+=alm_recurse[itr]*Y_r(ls[itr],ms[itr],pp_geo.all_pixels[:,0],pp_geo.all_pixels[:,1])
#        #totals_recurse+=alm_recurse[itr]*Y_r_2(ls[itr],ms[itr],pp_geo.pixels[:,0],pp_geo.pixels[:,1],known_legendre)
#    #    if do_rect:
#    t6=time()
#    print "reconstruct time: "+str(t6-t5)+"s"
#    #plot the polygon if basemap is installed, do nothing if it isn't
#    import matplotlib.pyplot as plt
#    if try_plot2:
#        from astropy import wcs
#        from astropy.io.fits import ImageHDU
#        im = ImageHDU(totals_recurse)
#        w = wcs.WCS(im,naxis=1)
#        w.wcs.ctype = ["HPX"]
#        fig=plt.figure()
#        ax = fig.add_subplot(111, projection=w)
#        ax.imshow(im, origin='lower', cmap='cubehelix')
#        plt.show()
#
#    if try_plot and do_reconstruct:
#        #try:
#            from mpl_toolkits.basemap import Basemap
#            m = Basemap(projection='moll',lon_0=0)
#            #m.drawparallels(np.arange(-90.,120.,30.))
#            #m.drawmeridians(np.arange(0.,420.,60.))
#            #restrict = totals_recurse>-1.
#            lats = (pp_geo.all_pixels[:,0]-np.pi/2.)*180/np.pi
#            lons = pp_geo.all_pixels[:,1]*180/np.pi
#            x,y=m(lons,lats)
#            #have to switch because histogram2d considers y horizontal, x vertical
#            fig = plt.figure(figsize=(10,5))
#
#            ax = fig.add_subplot(121)
#            H1,yedges1,xedges1 = np.histogram2d(y,x,100,weights=totals_recurse)
#            X1, Y1 = np.meshgrid(xedges1, yedges1)
#            ax.pcolormesh(X1,Y1,H1)
#            ax.set_aspect('equal')
#            #m.plot(x,y,'bo',markersize=1)
#            pp_geo.sp_poly.draw(m,color='black')
#            if do_rect:
#                ax = fig.add_subplot(122)
#                H2,yedges2,xedges2 = np.histogram2d(y,x,100,weights=1.*totals_rect)
#                X2, Y2 = np.meshgrid(xedges2, yedges2)
#                ax.pcolormesh(X2,Y2,H2)
#                ax.set_aspect('equal')
#                #m.plot(x,y,'bo',markersize=1)
#                pp_geo.sp_poly.draw(m,color='black')
#            plt.show()
#
#        #except:
#        #    pass
