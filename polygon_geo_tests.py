import numpy as np
from polygon_pixel_geo import polygon_pixel_geo,get_Y_r_dict
from sph_functions import Y_r
from geo import pixel_geo,rect_geo,geo
from time import time
import defaults
import scipy as sp
from cosmopie import CosmoPie
from scipy.interpolate import SmoothBivariateSpline
from warnings import warn
import polygon_geo as pg
import pytest

def check_mutually_orthonormal(vectors):
    fails = 0
    for itr1 in xrange(0,vectors.shape[0]):
        for itr2  in xrange(0,vectors.shape[0]):
            prod =np.sum(vectors[itr1]*vectors[itr2],axis=1)
            if itr1==itr2:
                if not np.allclose(prod,np.zeros(vectors[itr1].shape[0])+1.):
                    warn("normality failed on vector pair: "+str(itr1)+","+str(itr2))
                    print prod
                    fails+=1
            else:
                if not np.allclose(prod,np.zeros(vectors[itr1].shape[0])):
                    warn("orthogonality failed on vector pair: "+str(itr1)+","+str(itr2))
                    print prod
                    fails+=1
    return fails

class GeoTestSet:
    def __init__(self,params):
        #some setup to make an actual geo
        d=np.loadtxt('camb_m_pow_l.dat')
        k=d[:,0]; P=d[:,1]
        self.C=CosmoPie(k=k,P_lin=P,cosmology=defaults.cosmology)
        self.zs=np.array([.01,1.01])
        self.z_fine = np.arange(defaults.lensing_params['z_min_integral'],np.max(self.zs),defaults.lensing_params['z_resolution'])

        self.poly_params = defaults.polygon_params.copy()
        self.poly_geo = pg.polygon_geo(self.zs,params['thetas'],params['phis'],self.C,self.z_fine,l_max=params['l_max_poly'],poly_params=self.poly_params)
        if params['do_pp_geo1']:
            self.pp_geo1 = polygon_pixel_geo(self.zs,params['thetas'],params['phis'],params['theta_in'],params['phi_in'],self.C,self.z_fine,l_max=params['l_max_poly'],res_healpix=params['res_choose1'])
        if params['do_pp_geo2']:
            self.pp_geo2 = polygon_pixel_geo(self.zs,params['thetas'],params['phis'],params['theta_in'],params['phi_in'],self.C,self.z_fine,l_max=params['l_max_poly'],res_healpix=params['res_choose2'])
        if params['do_rect_geo']:
            self.r_geo = rect_geo(self.zs,params['r_thetas'],params['r_phis'],self.C,self.z_fine)
        self.params = params
        
def get_param_set(indices):
    test_type = indices[0]
    param_index = indices[1]
    if test_type == 0:
        #compare to rect_geo result
        params = {'do_rect_geo':True,'do_pp_geo1':True,'do_pp_geo2':True,'res_choose1':6,'res_choose2':7,'l_max_rect':10,'l_max_poly':50}
        if 0<=param_index and param_index<12:
            poffset = np.pi/3.*np.mod(param_index,6)
            if param_index<6:
                toffset = 0.
            else: 
                toffset = np.pi/2.
            theta0=0.+toffset
            theta1=np.pi/2.+toffset
            phi0 = 0.+poffset
            phi1 = 2.*np.pi/3.+poffset
            params['theta_in'] = np.pi/4.+toffset
            params['phi_in'] = np.pi/6.+poffset

        params['thetas'] = np.array([theta0,theta1,theta1,theta0,theta0])
        params['phis'] = np.array([phi0,phi0,phi1,phi1,phi0])
        params['r_thetas'] = np.array([theta0,theta1])
        params['r_phis'] = np.array([phi0,phi1])
    elif test_type == 1:
        #some rectangular geometries that cannot be directly compared to rect_geo
        params = {'do_rect_geo':False,'do_pp_geo1':True,'do_pp_geo2':True,'res_choose1':6,'res_choose2':7,'l_max_rect':10,'l_max_poly':50}
        if param_index==0 or param_index==1:
            if param_index==0:
                poffset=0.
            else:
                poffset=2.*np.pi/3
            toffset = np.pi/6.
            theta0=0.+toffset
            theta1=np.pi/2.+toffset
            phi0 = 0.+poffset
            phi1 = 2.*np.pi/3.+poffset
            params['theta_in'] = np.pi/4.+toffset
            params['phi_in'] = np.pi/6.+poffset
        elif param_index==2:
            poffset = 0.
            toffset = np.pi/6.
            theta0=0.+toffset
            theta1=2.*np.pi/3.+toffset
            phi0 = 0.+poffset
            phi1 = 2.*np.pi/3.+poffset
            params['theta_in'] = np.pi/4.+toffset
            params['phi_in'] = np.pi/6.+poffset
        elif param_index==3:
            poffset = 0.
            toffset = np.pi/6.
            theta0=0.+toffset
            theta1=2.*np.pi/3.+toffset
            phi0 = 0.+poffset
            phi1 = 4.*np.pi/3.+poffset
            params['theta_in'] = np.pi/4.+toffset
            params['phi_in'] = np.pi/6.+poffset
        elif param_index==4:
            poffset = 0.
            toffset = np.pi/6.
            theta0=0.+toffset
            theta1=2.*np.pi/3.+toffset
            phi0 = 0.+poffset
            phi1 = 2.*np.pi/3.+poffset
            params['theta_in'] = np.pi/4.+toffset
            params['phi_in'] = -np.pi/6.+poffset
        else:
            raise ValueError('unknown param_index '+str(param_index))

        params['thetas'] = np.array([theta0,theta1,theta1,theta0,theta0])
        params['phis'] = np.array([phi0,phi0,phi1,phi1,phi0])
        if param_index==4:
            params['thetas'] = params['thetas'][::-1]
            params['phis'] = params['phis'][::-1]
    elif test_type==2:
        #other geometries
        params = {'do_rect_geo':False,'do_pp_geo1':True,'do_pp_geo2':True,'res_choose1':6,'res_choose2':7,'l_max_rect':10,'l_max_poly':50}
        if param_index==0:
            toffset = np.pi/2.
            theta0=np.pi/6+toffset
            theta1=np.pi/3.+toffset
            theta2=np.pi/3.+0.1+toffset
            theta3=theta2-np.pi/3.
            theta4 = theta3+np.pi/6.
            offset=0.
            phi0=0.+offset
            phi1 = np.pi/3.+offset
            phi2 = phi1+np.pi/2.
            phi3 = phi2-np.pi/6.
            phi4 = phi3-np.pi/3.
            params['thetas'] = np.array([theta0,theta1,theta1,theta2,theta0,theta3,theta3,theta4,theta0])
            params['phis'] = np.array([phi0,phi0,phi1,phi1,phi2,phi3,phi4,phi4,phi0])
            params['theta_in'] = np.pi/4.+toffset
            params['phi_in'] = np.pi/6.+offset
        else:
            raise ValueError('unknown param_index '+str(param_index))
    else:
        raise ValueError('unknown test type '+str(test_type))

    return params

index_sets = [[2,0],[1,0],[1,1],[1,2],[1,3],[1,4]]
for itr in xrange(0,6):
    index_sets.append(np.array([0,itr]))
    index_sets.append(np.array([0,itr+6]))

@pytest.fixture(params=index_sets,scope="module")
def geo_input(request):
    return GeoTestSet(get_param_set(request.param))

def test_alm_rect_agreement(geo_input):
    if geo_input.params['do_rect_geo']:
        #relatively low tolerance for differences because both should be nearly exact
        #main source of error is angle doubling formula, increasing number of doublings would decrease error
        ABS_TOL = 10**-7
        REL_TOL = 10**-8
        alm_array_poly = geo_input.poly_geo.get_alm_array(geo_input.params['l_max_rect'])[0]
        alm_array_rect = geo_input.r_geo.get_alm_array(geo_input.params['l_max_rect'])[0]
        assert(np.allclose(alm_array_poly,alm_array_rect,atol=ABS_TOL,rtol=REL_TOL))

def test_alm_pp_agreement1(geo_input):
    if geo_input.params['do_pp_geo1']:
        #relatively high tolerance for differences because polygon_geo is intended to be more accurate than polygon_pixel_geo
        ABS_TOL = 10**-2
        REL_TOL = 10**-2
        alm_array_poly = geo_input.poly_geo.get_alm_array(geo_input.params['l_max_poly'])[0]
        alm_array_pp2 = geo_input.pp_geo1.get_alm_array(geo_input.params['l_max_poly'])[0]
        assert(np.allclose(alm_array_poly,alm_array_pp2,atol=ABS_TOL,rtol=REL_TOL))

def test_alm_pp_agreement2(geo_input):
    if geo_input.params['do_pp_geo2']:
        #relatively high tolerance for differences because polygon_geo is intended to be more accurate than polygon_pixel_geo
        ABS_TOL = 10**-2
        REL_TOL = 10**-2
        alm_array_poly = geo_input.poly_geo.get_alm_array(geo_input.params['l_max_poly'])[0]
        alm_array_pp1 = geo_input.pp_geo1.get_alm_array(geo_input.params['l_max_poly'])[0]
        assert(np.allclose(alm_array_poly,alm_array_pp1,atol=ABS_TOL,rtol=REL_TOL))

#test coarse reconstruction agreement
#note tolerance should really depend on l max
def test_absolute_reconstruction1(geo_input):
    MSE_TOL = 10**-1
    if geo_input.params['do_pp_geo1']:
        pp_geo1 = geo_input.pp_geo1
        totals_poly = pp_geo1.reconstruct_from_alm(geo_input.params['l_max_poly'],pp_geo1.all_pixels[:,0],pp_geo1.all_pixels[:,1],geo_input.poly_geo.alm_table)
        abs_error = np.abs(totals_poly-pp_geo1.contained*1.)
        mse = np.sqrt(np.average(abs_error**2))
        assert(MSE_TOL>mse)

#test fine reconstruction improves on polygon_pixel_geo  
#also test polygon_pixel_geo error while we're at it
def test_absolute_improvement2(geo_input):
    MSE_TOL = 10**-1
    if geo_input.params['do_pp_geo2']:
        pp_geo2 = geo_input.pp_geo2
        totals_poly = pp_geo2.reconstruct_from_alm(geo_input.params['l_max_poly'],pp_geo2.all_pixels[:,0],pp_geo2.all_pixels[:,1],geo_input.poly_geo.alm_table)
        totals_pp1 = pp_geo2.reconstruct_from_alm(geo_input.params['l_max_poly'],pp_geo2.all_pixels[:,0],pp_geo2.all_pixels[:,1],geo_input.pp_geo1.alm_table)
        abs_error_poly = np.abs(totals_poly-pp_geo2.contained*1.)
        mse_poly = np.sqrt(np.average(abs_error_poly**2))
        assert(MSE_TOL>mse_poly)

        abs_error_pp1 = np.abs(totals_pp1-pp_geo2.contained*1.)
        mse_pp1 = np.sqrt(np.average(abs_error_pp1**2))
        assert(MSE_TOL>mse_pp1)
        assert(mse_pp1>mse_poly)

#do a bunch of tests to make sure rotations are working as expected
#do a bunch of tests to make sure rotations are working as expected
def test_rotational_suite(geo_input):
    poly_geo = geo_input.poly_geo

    nt = poly_geo.n_v
   
    x1_1 = np.zeros((nt,3))
    x1_1[:,0] = 1.
    y1_1 = np.zeros((nt,3))
    y1_1[:,1] = 1.
    z1_1 = np.zeros((nt,3))
    z1_1[:,2] = 1.

    assert(0==check_mutually_orthonormal(np.array([x1_1,y1_1,z1_1])))
    assert(np.allclose(np.cross(x1_1,y1_1),z1_1))

    x1_g=poly_geo.bounding_xyz[0:nt]
    z1_g=poly_geo.z_hats
    y1_g=np.cross(z1_g,x1_g)

    assert(np.allclose(poly_geo.bounding_xyz[1:nt+1],np.expand_dims(np.cos(poly_geo.betas),1)*x1_g-np.expand_dims(np.sin(poly_geo.betas),1)*y1_g))

    assert(0==check_mutually_orthonormal(np.array([x1_g,y1_g,z1_g])))
    assert(np.allclose(np.cross(x1_g,y1_g),z1_g))

    omegas = poly_geo.omega_alphas
    rot12 = np.zeros((nt,3,3))
    x2_1 = np.zeros((nt,3))
    y2_1 = np.zeros((nt,3))
    z2_1 = np.zeros((nt,3))
    for itr in xrange(0,nt):
        rot12[itr] = np.array([[np.cos(omegas[itr]),np.sin(omegas[itr]),0],[-np.sin(omegas[itr]),np.cos(omegas[itr]),0],[0,0,1]]) 
        x2_1[itr] = np.dot(rot12[itr].T,x1_1[itr])
        y2_1[itr] = np.dot(rot12[itr].T,y1_1[itr])
        z2_1[itr] = np.dot(rot12[itr].T,z1_1[itr])
    assert(0==check_mutually_orthonormal(np.array([x2_1,y2_1,z2_1])))
    assert(np.allclose(np.cross(x2_1,y2_1),z2_1))

    x2_g_alt =np.expand_dims(np.sum(x2_1*x1_1,axis=1),1)*x1_g+np.expand_dims(np.sum(y2_1*x1_1,axis=1),1)*y1_g+np.expand_dims(np.sum(z2_1*x1_1,axis=1),1)*z1_g
    y2_g_alt =np.expand_dims(np.sum(x2_1*y1_1,axis=1),1)*x1_g+np.expand_dims(np.sum(y2_1*y1_1,axis=1),1)*y1_g+np.expand_dims(np.sum(z2_1*y1_1,axis=1),1)*z1_g
    z2_g_alt =np.expand_dims(np.sum(x2_1*z1_1,axis=1),1)*x1_g+np.expand_dims(np.sum(y2_1*z1_1,axis=1),1)*y1_g+np.expand_dims(np.sum(z2_1*z1_1,axis=1),1)*z1_g
    assert(0==check_mutually_orthonormal(np.array([x2_g_alt,y2_g_alt,z2_g_alt])))
    assert(np.allclose(np.cross(x2_g_alt,y2_g_alt),z2_g_alt))


    x2_g = np.zeros((nt,3))
    x2_g[:,0] = np.cos(poly_geo.gamma_alphas)
    x2_g[:,1] = np.sin(poly_geo.gamma_alphas)
    z2_g=z1_g
    y2_g=np.cross(z2_g,x2_g)
    assert(0==check_mutually_orthonormal(np.array([x2_g,y2_g,z2_g])))
    assert(np.allclose(np.cross(x2_g,y2_g),z2_g))

    assert(np.allclose(np.zeros(nt),np.linalg.norm(np.cross(x2_g,np.cross(np.array([0.,0.,1.]),z1_g)),axis=1)))
    assert(np.allclose(np.cos(omegas),np.sum(x2_g_alt*x1_g,axis=1)))
    assert(np.allclose(np.cos(omegas),np.sum(x2_g*x1_g,axis=1)))
    assert(np.allclose(np.cos(omegas),np.sum(y2_g*y1_g,axis=1)))
    assert(np.allclose(1.,np.sum(z2_g*z1_g,axis=1)))

    x2_2 = x1_1
    y2_2 = y1_1
    z2_2 = z1_1
    assert(0==check_mutually_orthonormal(np.array([x2_2,y2_2,z2_2])))
    assert(np.allclose(np.cross(x2_2,y2_2),z2_2))



    x3_g =x2_g
    z3_g = np.zeros((nt,3))
    z3_g[:,2] = 1.
    y3_g=np.cross(z3_g,x3_g)
    assert(0==check_mutually_orthonormal(np.array([x3_g,y3_g,z3_g])))
    assert(np.allclose(np.cross(x3_g,y3_g),z3_g))


    thetas_a = poly_geo.theta_alphas
    rot23 = np.zeros((nt,3,3))
    x3_2 = np.zeros((nt,3))
    y3_2 = np.zeros((nt,3))
    z3_2 = np.zeros((nt,3))
    for itr in xrange(0,nt):
        rot23[itr] = np.array([[1.,0.,0.],[0.,np.cos(thetas_a[itr]),np.sin(thetas_a[itr])],[0,-np.sin(thetas_a[itr]),np.cos(thetas_a[itr])]]) 
        x3_2[itr] = np.dot(rot23[itr].T,x2_2[itr])
        y3_2[itr] = np.dot(rot23[itr].T,y2_2[itr])
        z3_2[itr] = np.dot(rot23[itr].T,z2_2[itr])
    assert(0==check_mutually_orthonormal(np.array([x3_2,y3_2,z3_2])))
    assert(np.allclose(np.cross(x3_2,y3_2),z3_2))

    assert(np.allclose(np.cos(thetas_a),np.sum(z3_g*z2_g,axis=1)))
    assert(np.allclose(np.cos(thetas_a),np.sum(y3_g*y2_g,axis=1)))
    assert(np.allclose(1.,np.sum(x3_g*x2_g,axis=1)))


    x3_g_alt =np.expand_dims(np.sum(x3_2*x2_2,axis=1),1)*x2_g_alt+np.expand_dims(np.sum(y3_2*x2_2,axis=1),1)*y2_g_alt+np.expand_dims(np.sum(z3_2*x2_2,axis=1),1)*z2_g_alt
    y3_g_alt =np.expand_dims(np.sum(x3_2*y2_2,axis=1),1)*x2_g_alt+np.expand_dims(np.sum(y3_2*y2_2,axis=1),1)*y2_g_alt+np.expand_dims(np.sum(z3_2*y2_2,axis=1),1)*z2_g_alt
    z3_g_alt =np.expand_dims(np.sum(x3_2*z2_2,axis=1),1)*x2_g_alt+np.expand_dims(np.sum(y3_2*z2_2,axis=1),1)*y2_g_alt+np.expand_dims(np.sum(z3_2*z2_2,axis=1),1)*z2_g_alt
    assert(0==check_mutually_orthonormal(np.array([x3_g_alt,y3_g_alt,z3_g_alt])))
    assert(np.allclose(np.cross(x3_g_alt,y3_g_alt),z3_g_alt))

    assert(np.allclose(np.zeros(nt),np.linalg.norm(np.cross(x2_g_alt,np.cross(np.array([0.,0.,1.]),z1_g)),axis=1)))
    assert(np.allclose(x3_g_alt,x2_g_alt))
    x3_3 = x1_1
    y3_3 = y1_1
    z3_3 = z1_1
    assert(0==check_mutually_orthonormal(np.array([x3_3,y3_3,z3_3])))
    assert(np.allclose(np.cross(x3_3,y3_3),z3_3))


    gammas = poly_geo.gamma_alphas
    rot34 = np.zeros((nt,3,3))
    x4_3 = np.zeros((nt,3))
    y4_3 = np.zeros((nt,3))
    z4_3 = np.zeros((nt,3))
    for itr in xrange(0,nt):
        rot34[itr] = np.array([[np.cos(gammas[itr]),np.sin(gammas[itr]),0],[-np.sin(gammas[itr]),np.cos(gammas[itr]),0],[0,0,1]]) 
        x4_3[itr] = np.dot(rot34[itr].T,x3_3[itr])
        y4_3[itr] = np.dot(rot34[itr].T,y3_3[itr])
        z4_3[itr] = np.dot(rot34[itr].T,z3_3[itr])
    assert(0==check_mutually_orthonormal(np.array([x4_3,y4_3,z4_3])))
    assert(np.allclose(np.cross(x4_3,y4_3),z4_3))


    x4_g_alt =np.expand_dims(np.sum(x4_3*x3_3,axis=1),1)*x3_g_alt+np.expand_dims(np.sum(y4_3*x3_3,axis=1),1)*y3_g_alt+np.expand_dims(np.sum(z4_3*x3_3,axis=1),1)*z3_g_alt
    y4_g_alt =np.expand_dims(np.sum(x4_3*y3_3,axis=1),1)*x3_g_alt+np.expand_dims(np.sum(y4_3*y3_3,axis=1),1)*y3_g_alt+np.expand_dims(np.sum(z4_3*y3_3,axis=1),1)*z3_g_alt
    z4_g_alt =np.expand_dims(np.sum(x4_3*z3_3,axis=1),1)*x3_g_alt+np.expand_dims(np.sum(y4_3*z3_3,axis=1),1)*y3_g_alt+np.expand_dims(np.sum(z4_3*z3_3,axis=1),1)*z3_g_alt
    assert(0==check_mutually_orthonormal(np.array([x4_g_alt,y4_g_alt,z4_g_alt])))
    assert(np.allclose(np.cross(x4_g_alt,y4_g_alt),z4_g_alt))

    assert(np.allclose(np.cos(gammas),np.sum(x4_g_alt*x3_g,axis=1)))
    assert(np.allclose(np.cos(gammas),np.sum(y4_g_alt*y3_g,axis=1)))
    assert(np.allclose(1.,np.sum(z4_g_alt*z3_g,axis=1)))

    assert(np.allclose(x4_g_alt,np.array([1,0,0.]))) 
    assert(np.allclose(y4_g_alt,np.array([0,1,0.]))) 
    assert(np.allclose(z4_g_alt,np.array([0,0,1.]))) 


    assert(np.allclose(x3_g,x3_g_alt))
    assert(np.allclose(y3_g,y3_g_alt))
    assert(np.allclose(z3_g,z3_g_alt))

    assert(np.allclose(x2_g,x2_g_alt))
    assert(np.allclose(y2_g,y2_g_alt))
    assert(np.allclose(z2_g,z2_g_alt))

    #alm_rats0 = np.zeros(l_max+1)


if __name__=='__main__':
    
    params = get_param_set(np.array([0,0]))
    gts = GeoTestSet(params)
    l_max = params['l_max']
    res_choose = params['res_choose1']
    poly_geo = gts.poly_geo
    pp_geo = gts.pp_geo1
    pp_geo2 = gts.pp_geo2

     
    nt = poly_geo.n_v
   
    my_table = poly_geo.alm_table.copy()
    #get rect_geo to cache the values in the table
    for ll in xrange(0,l_max+1):
        for mm in xrange(0,ll+1):
            gts.r_geo.a_lm(ll,mm)
            if mm>0:
                gts.r_geo.a_lm(ll,-mm)
    #r_alm_table = r_geo.alm_table
    #reconstruct at higher resolution to mitigate resolution effects in determining accuracy
    totals_pp= pp_geo2.reconstruct_from_alm(l_max,pp_geo2.all_pixels[:,0],pp_geo2.all_pixels[:,1],gts.r_geo.alm_table)
    totals_poly = pp_geo2.reconstruct_from_alm(l_max,pp_geo2.all_pixels[:,0],pp_geo2.all_pixels[:,1],my_table)
    avg_diff = np.average(np.abs(totals_pp-totals_poly))
    print "mean absolute difference between pixel and exact geo reconstruction: "+str(avg_diff)
    poly_error = np.sqrt(np.average(np.abs(totals_poly-pp_geo2.contained*1.)**2))
    pp_error = np.sqrt(np.average(np.abs(totals_pp-pp_geo2.contained*1.)**2))
    print "rms reconstruction error of exact geo: "+str(poly_error)
    print "rms reconstruction error of pixel geo at res "+str(res_choose)+": "+str(pp_error)
    print "improvement in rms reconstruction accuracy: "+str((pp_error-poly_error)/pp_error*100)+"%"

    #totals_alm = pp_geo.reconstruct_from_alm(l_max,pp_geo.all_pixels[:,0],pp_geo.all_pixels[:,1],r_alm_table)
    try_plot=True
    do_poly=True
    if try_plot:
               #try:
            from mpl_toolkits.basemap import Basemap
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            m = Basemap(projection='moll',lon_0=0)
            #m.drawparallels(np.arange(-90.,120.,30.))
            #m.drawmeridians(np.arange(0.,420.,60.))
            #restrict = totals_recurse>-1.
            lats = (pp_geo2.all_pixels[:,0]-np.pi/2.)*180/np.pi
            lons = pp_geo2.all_pixels[:,1]*180/np.pi
            x,y=m(lons,lats)
            #have to switch because histogram2d considers y horizontal, x vertical
            fig = plt.figure(figsize=(10,5))
            minC = np.min([totals_poly,totals_pp])
            maxC = np.max([totals_poly,totals_pp])
            bounds = np.linspace(minC,maxC,10)
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            ax = fig.add_subplot(121)
            H1,yedges1,xedges1 = np.histogram2d(y,x,100,weights=totals_pp)
            X1, Y1 = np.meshgrid(xedges1, yedges1)
            pc1 = ax.pcolormesh(X1,Y1,-H1,cmap='gray')
            ax.set_aspect('equal')
            ax.set_title("polygon_pixel_geo reconstruction")
            #fig.colorbar(pc1,ax=ax)
            #m.plot(x,y,'bo',markersize=1)
            pp_geo2.sp_poly.draw(m,color='red')
            if do_poly:
                ax = fig.add_subplot(122)
                H2,yedges2,xedges2 = np.histogram2d(y,x,100,weights=1.*totals_poly)
                X2, Y2 = np.meshgrid(xedges2, yedges2)
                ax.pcolormesh(X2,Y2,-H2,cmap='gray')
                ax.set_aspect('equal')
                #m.plot(x,y,'bo',markersize=1)
                ax.set_title("polygon_geo reconstruction")
                pp_geo2.sp_poly.draw(m,color='red')
            plt.show()

