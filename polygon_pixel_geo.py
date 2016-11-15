import numpy as np
from astropy.io import fits
import spherical_geometry.vector as sgv
from spherical_geometry.polygon import SphericalPolygon
from geo import pixel_geo
from time import time
import defaults
import scipy as sp
from cosmopie import CosmoPie

#get a healpix pixelated spherical polygon geo
class polygon_pixel_geo(pixel_geo):
        def __init__(self,zs,thetas,phis,phi_in,theta_in,C,z_fine,res_healpix=defaults.polygon_params['res_healpix']):
            all_pixels = get_healpix_pixelation(res_choose=res_healpix)
            sp_poly = get_poly(thetas,phis,theta_in,phi_in)
            contained = is_contained(all_pixels,sp_poly)
            contained_pixels = all_pixels[contained,:]
            self.n_pix = contained_pixels.shape[0]
            pixel_geo.__init__(self,zs,contained_pixels,C,z_fine)

        
        #first try loop takes 22.266 sec for l_max=5 for 290 pixel region
        #second try vectorizing Y_r 0.475 sec l_max=5 (47 times speed up)
        #second try 193.84 sec l_max=50
        #third try (with recurse) 2.295 sec for l_max=50 (84 times speed up over second)
        #third try (with recurse) 0.0267 sec for l_max=5
        #third try 9.47 sec for l_max=100
        #oddly, suddenly gets faster with more pixels; maybe some weird numpy internals issue
        #third try with 16348 pixels l_max =100 takes 0.9288-1.047ss 
        def get_a_lm_below_l_max(self,l_max):
            a_lms = np.zeros((l_max+1)**2)
            ls = np.zeros((l_max+1)**2)
            ms = np.zeros((l_max+1)**2)

            itr = 0

            for ll in range(0,l_max+1):
                for mm in range(-ll,ll+1):
                    #first try takes ~0.618 sec/iteration for 290 pixel region=> 2.1*10**-3 sec/(iteration pixel), much too slow
                    ls[itr] = ll
                    ms[itr] = mm
                    a_lms[itr] = self.a_lm(ll,mm)
                    itr+=1
            return a_lms,ls,ms
        #TODO check numerical stability
        def get_a_lm_below_l_max_recurse(self,l_max):
            n_tot = (l_max+1)**2
            pixel_area = self.pixels[0,2]

            ls = np.zeros(n_tot)
            ms = np.zeros(n_tot)
            a_lms = np.zeros(n_tot)
            
            lm_dict = {}
            itr = 0
            for ll in range(0,l_max+1):
                for mm in range(-ll,ll+1):
                    lm_dict[(ll,mm)] = itr
                    itr+=1
    
            cos_theta = np.cos(self.pixels[:,0])
            sin_theta = np.sin(self.pixels[:,0])
            abs_sin_theta = np.abs(sin_theta)


            sin_phi_m = np.zeros((l_max+1,self.n_pix))
            cos_phi_m = np.zeros((l_max+1,self.n_pix))
            for mm in range(0,l_max+1):
                sin_phi_m[mm] = np.sin(mm*self.pixels[:,1])
                cos_phi_m[mm] = np.cos(mm*self.pixels[:,1])

            factorials = sp.misc.factorial(np.arange(0,2*l_max+1))

            known_legendre = {(0,0):(np.zeros(self.n_pix)+1.),(1,0):cos_theta,(1,1):-abs_sin_theta}

            #next_row = {}
            #next_row = {(1,0):cos_theta,(1,1):-sin_theta}
            for ll in range(2,l_max+1):
                #for mm in range(0,l_max):
                #    next_row[(ll,mm)] = cos_theta*(2.*ll+1.)*cur_row[(ll-1,mm)]
                known_legendre[(ll,ll-1)] = (2.*ll-1.)*cos_theta*known_legendre[(ll-1,ll-1)]
                known_legendre[(ll,ll)] = -(2.*ll-1.)*abs_sin_theta*known_legendre[(ll-1,ll-1)] 
                for mm in range(0,ll-1):
                    known_legendre[(ll,mm)] = ((2.*ll-1.)/(ll-mm)*cos_theta*known_legendre[(ll-1,mm)]-(ll+mm-1.)/(ll-mm)*known_legendre[(ll-2,mm)])

            for ll in range(0,l_max+1):
                for mm in range(0,ll+1):
                    prefactor = np.sqrt((2.*ll+1.)/(4.*np.pi)*factorials[ll-mm]/factorials[ll+mm])*pixel_area
                    base = sin_theta*known_legendre[(ll,mm)]
                    if mm==0:
                        a_lms[lm_dict[(ll,mm)]] = prefactor*np.sum(base)
                    else:
                        #TODO check sign convention
                        cos_m = cos_phi_m[mm]
                        sin_m = sin_phi_m[mm]

                        a_lms[lm_dict[(ll,mm)]] = (-1)**(mm)*np.sqrt(2.)*prefactor*np.sum(base*cos_m)
                        #a_lms[lm_dict[(ll,mm)]] = (-1)**mm*np.sqrt(2.)*np.sum(base*np.cos(mm*self.pixels[:,1]))
                        a_lms[lm_dict[(ll,-mm)]] = (-1)**(mm)*np.sqrt(2.)*prefactor*np.sum(base*sin_m)

            return a_lms,known_legendre

#alternate way of computing Y_r from the way in sph_functions
def Y_r_2(ll,mm,theta,phi,known_legendre):
    prefactor = np.sqrt((2.*ll+1.)/(4.*np.pi)*sp.misc.factorial(ll-mm)/sp.misc.factorial(ll+mm))
    base = prefactor*known_legendre[(ll,mm)]*(-1)**mm
    if mm==0:
        return base
    elif mm>0:
        return base*np.sqrt(2.)*np.cos(mm*phi)
    else:
        return base*np.sqrt(2.)*np.sin(np.abs(mm)*phi)


#can safely resolve up to lmax~2*nside (although can keep going with loss of precision until lmax=3*nside-1), so if lmax=100,need nside~50
#nside = 2^res, so res=6 => nside=64 should safely resolve lmax=100, for extra safety can choose res=7 
#res = 10 takes ~164.5  sec
#res = 9 takes ~50.8 sec
#res = 8 takes ~11 sec
#res = 7 takes ~3.37 sec
#res = 6 takes ~0.688 sec
def get_healpix_pixelation(res_choose=6):
    pixel_info = np.loadtxt('pixel_info.dat')
    area = pixel_info[res_choose,4]
    #tables from https://lambda.gsfc.nasa.gov/toolbox/tb_pixelcoords.cfm#pixelinfo
    hdulist = fits.open('pixel_coords_map_ring_galactic_res'+str(res_choose)+'.fits')
    data = hdulist[1].data
    pixels = np.zeros((data.size,3))

    pixels[:,0] = data['LONGITUDE']*np.pi/180.
    pixels[:,1] = data['LATITUDE']*np.pi/180.
    pixels[:,2] = area

    return pixels



#Note these are spherical polygons so all the sides are great circles (not lines of constant theta!)
#So area will differ from integral if assuming constant theta
#vertices must have same first and last coordinate so polygon is closed
#last point is arbitrary point inside because otherwise 2 polygons possible. 
#Behavior may be unpredictable if the inside point is very close to an edge or vertex. 
def get_poly(theta_vertices,phi_vertices,theta_in,phi_in):
    bounding_theta = theta_vertices-np.pi/2. #to radec
    bounding_phi = phi_vertices
    bounding_xyz = np.asarray(sgv.radec_to_vector(bounding_phi,bounding_theta,degrees=False)).T
    inside_xyz = np.asarray(sgv.radec_to_vector(phi_in,theta_in-np.pi/2.,degrees=False))

    sp_poly = SphericalPolygon(bounding_xyz,inside=inside_xyz)
    return sp_poly

#Pixels is a pixelation (ie what get_healpix_pixelation returns) and sp_poly is a spherical polygon, ie from get_poly
def is_contained(pixels,sp_poly):
    #xyz vals for the pixels
    xyz_vals = sgv.radec_to_vector(pixels[:,0],pixels[:,1],degrees=False)
    contained = np.zeros(pixels.shape[0],dtype=bool)
    #check if each point is contained in the polygon. This is fairly slow if the number of points is huge
    for i in range(0,pixels.shape[0]):
        contained[i]= sp_poly.contains_point([xyz_vals[0][i],xyz_vals[1][i],xyz_vals[2][i]])
    return contained

#PLAN: explicitly implement Y_r both ways for testing purposes
if __name__=='__main__':
    thetas = np.array([0.,np.pi/2.,np.pi/2.,0.,0.])
    phis = np.array([0.,0.,np.pi/3.,np.pi/3.,0.])
    theta_in = np.pi/8.
    phi_in = np.pi/24.
    pixels = get_healpix_pixelation(res_choose=7)
    sp_poly = get_poly(thetas,phis,theta_in,phi_in)
    contained = is_contained(pixels,sp_poly)

    print "total contained pixels in polygon: "+str(np.sum(contained*1.))
    print "total contained area of polygon: "+str(np.dot(pixels[:,2],contained*1.))
    print "area calculated by SphericalPolygon: "+str(sp_poly.area())
    
    #some setup to make an actual geo
    d=np.loadtxt('camb_m_pow_l.dat')
    k=d[:,0]; P=d[:,1]
    C=CosmoPie(k=k,P_lin=P,cosmology=defaults.cosmology)
    zs=np.array([.01,1.01])
    z_fine = np.arange(defaults.lensing_params['z_min_integral'],np.max(zs),defaults.lensing_params['z_resolution'])

    l_max = 100
    do_old = False

    t0 = time()
    pp_geo = polygon_pixel_geo(zs,thetas,phis,theta_in,phi_in,C,z_fine,defaults.polygon_params['res_healpix'])
    t1 = time()
    #TODO write explicit test case to compare
    if do_old:
        print "polygon_pixel_geo: initialization time: "+str(t1-t0)+"s"
        alm_pps,ls,ms = pp_geo.get_a_lm_below_l_max(l_max)
    t2 = time()
    if do_old:
        print "polygon_pixel_geo: a_lm up to l="+str(l_max)+" time: "+str(t2-t1)+"s" 
    n_run = 30
    for i in range(0,n_run):
        alm_recurse,known_legendre = pp_geo.get_a_lm_below_l_max_recurse(l_max)
    t3 = time()
    print "polygon_pixel_geo: a_lm_recurse in avg time: "+str((t3-t2)/n_run)+"s"
    if do_old:
        print "methods match: "+str(np.allclose(alm_pps,alm_recurse))
    #plot the polygon if basemap is installed, do nothing if it isn't
    try_plot = False
    if try_plot:
        try:
            from mpl_toolkits.basemap import Basemap
            import matplotlib.pyplot as plt
            m = Basemap(projection='moll',lon_0=0)
            m.drawparallels(np.arange(-90.,120.,30.))
            m.drawmeridians(np.arange(0.,420.,60.))
            sp_poly.draw(m,color='blue')
            plt.show()
        except:
            pass
