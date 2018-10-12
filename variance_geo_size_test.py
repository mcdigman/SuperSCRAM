"""demonstration of ability to compute super sample variance term in a geometry"""
from __future__ import print_function,division,absolute_import
from builtins import range
from time import time
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
#from polygon_geo import PolygonGeo
from cosmopie import CosmoPie
import defaults
from sph_klim import SphBasisK
import matter_power_spectrum as mps
from premade_geos import LSSTGeoSimpl,WFIRSTGeo
from circle_geo import CircleGeo
from ring_pixel_geo import RingPixelGeo
from half_sky_geo import HalfSkyGeo
from polygon_utils import get_healpix_pixelation

if __name__=='__main__':
    time0 = time()
    camb_params = { 'npoints':2000,
                    'minkh':1.1e-4,
                    'maxkh':1.476511342960e+02,
                    'kmax':1.476511342960e+02,
                    'leave_h':False,
                    'force_sigma8':False,
                    'return_sigma8':False,
                    'accuracy':1,
                    'pivot_scalar':0.002
                  }
    print("main: building cosmology")
    power_params = defaults.power_params.copy()
    power_params.camb = camb_params
    C = CosmoPie(defaults.cosmology.copy(),p_space='jdem')
    P_lin = mps.MatterPower(C,power_params)
    C.set_power(P_lin)
    time1 = time()

    #x_cut = 527.
    #l_max = 511
    #x_cut = 360
    #l_max = 346
    #x_cut = 150
    #l_max = 139
    #x_cut = 93
    l_max = 84
    x_cut = 5
    #l_max = 20

    do_plot = True

    l_sw = np.logspace(np.log(30),np.log(5000),base=np.exp(1.),num=20)

    print("main: building geometries")
    polygon_params = defaults.polygon_params
    polygon_params['n_double'] = 80
    z_coarse = np.array([0.,3.])
    #z_max = np.max(z_coarse)
    z_max = 3.01
    #z_fine = np.arange(0.0001,z_max,0.0001)
    z_fine = np.linspace(0.001,z_max,2)

    print("main: building basis")
    basis_params = defaults.basis_params.copy()
    basis_params['n_bessel_oversample'] = 400000
    basis_params['x_grid_size'] = 100000

    r_max = C.D_comov(z_max)
    k_cut = x_cut/r_max
    k_tests = np.linspace(20./r_max,k_cut,25)
    n_basis = np.zeros(k_tests.size)
    variances = np.zeros((k_tests.size,z_coarse.size-1,z_coarse.size-1))

    time4 = time()
    basis = SphBasisK(r_max,C,k_cut,basis_params,l_ceil=l_max)
    time5 = time()

    time2 = time()
    pixelated = True
    if pixelated:
        res_healpix = 9
        max_n_pixels = 12*(2**res_healpix)**2
        #n_pixels = np.unique(np.hstack([np.logspace(3,np.log10(max_n_pixels),40).astype(np.int),np.linspace(1000,max_n_pixels,40).astype(np.int)]))
        n_pixels = np.array([max_n_pixels/2]).astype(np.int)
        #n_pixels = np.array([1000,2000,3000,4000,5000,10000,20000,30000,40000,49152]) 
        n_bins = n_pixels.size
        all_pixels = get_healpix_pixelation(res_choose=res_healpix)
        #n_pixels = np.linspace(1,max_n_pixels,80).astype(np.int)#np.unique(np.hstack([np.logspace(0,np.log10(max_n_pixels),40).astype(np.int),np.linspace(1,max_n_pixels,40).astype(np.int)]))
    else:
        radii = np.array([0.1,0.2,0.3,0.4,0.8,1.6,3.])
        n_bins = radii.size

    variances_res = np.zeros(n_bins)
    areas_res = np.zeros(n_bins)
    geo1s = np.zeros(n_bins,dtype=object)
    for itr in range(0,n_bins):
        if pixelated:
            geo1s[itr] = RingPixelGeo(z_coarse,C,z_fine,l_max,res_healpix,n_pixels[itr],all_pixels)
        else:
            geo1s[itr] = CircleGeo(z_coarse,C,radii[itr],20,z_fine,l_max,polygon_params)
        variances_res[itr] = basis.get_variance(geo1s[itr])
        areas_res[itr] = geo1s[itr].angular_area()
    time3 = time()


    time6 = time()
    print("main: finished geo in "+str(time3-time2)+" s")
    print("main: finished basis in "+str(time5-time4)+" s")
    print("main: finished all in "+str(time6-time0)+" s")
    times = np.array([time0,time1,time2,time3,time4,time5,time6])
    if do_plot:
        import matplotlib.pyplot as plt
        plt.loglog(areas_res,1./areas_res*areas_res[-10]*variances_res[-10])
        plt.loglog(areas_res,variances_res)
        plt.show()

    do_dump = False
    if do_dump:
        import dill
        results = [z_coarse,z_fine,k_cut,l_max,camb_params,power_params,l_sw,z_max,r_max,k_tests,n_basis,variances,variance_res,r_width,theta_width,phi_width,volume,square_equiv,times]
        dump_f = open('dump_var_con.pkl','w')
        dill.dump(results,dump_f)
        dump_f.close()


