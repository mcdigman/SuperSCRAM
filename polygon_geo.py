import numpy as np
from math import isnan
from astropy.io import fits
import spherical_geometry.vector as sgv
from spherical_geometry.polygon import SphericalPolygon
import spherical_geometry.great_circle_arc as gca
from polygon_pixel_geo import polygon_pixel_geo,get_Y_r_dict
from sph_functions import Y_r
from geo import pixel_geo,rect_geo,geo
from time import time
import defaults
import scipy as sp
from cosmopie import CosmoPie
from scipy.interpolate import SmoothBivariateSpline
from warnings import warn
import scipy.special as spp
import alm_utils as au
import sys
from assoc_legendre import assoc_legendre_p
#get an exact spherical polygon geo
class polygon_geo(geo):
        #TODO check order of thetas,phis
        #vertices should be specified such that the clockwise oriented contour contains the area
        def __init__(self,zs,thetas,phis,C,z_fine,l_max,poly_params=defaults.polygon_params):
            #self.sp_poly = get_poly(thetas,phis,theta_in,phi_in)
            #if isnan(self.sp_poly.area()):
            #    raise ValueError("polygon_geo: Calculated area of polygon is nan, likely invalid polygon")

            self.n_double = poly_params['n_double']

            self.n_v = thetas.size-1

            self.bounding_theta = thetas-np.pi/2. #to radec
            self.bounding_phi = np.pi-phis
            self.bounding_xyz = np.asarray(sgv.radec_to_vector(self.bounding_phi,self.bounding_theta,degrees=False)).T

            #this gets correct internal angles with specified vertex order (copied from spherical_polygon clas) 
            self.internal_angles = 2.*np.pi-np.hstack([gca.angle(self.bounding_xyz[:-2], self.bounding_xyz[1:-1], self.bounding_xyz[2:], degrees=False),gca.angle(self.bounding_xyz[-2], self.bounding_xyz[0], self.bounding_xyz[1], degrees=False)])

            geo.__init__(self,zs,C,z_fine)

            self.alm_table = {(0,0):self.angular_area()/np.sqrt(4.*np.pi)}
            

            self.z_hats = np.zeros((self.n_v,3))
            self.y_hats = np.zeros_like(self.z_hats)
            self.xps = np.zeros_like(self.z_hats)

            self.betas = np.zeros(self.n_v)
            self.betas_alt = np.zeros(self.n_v)
            self.theta_alphas = np.zeros(self.n_v)
            self.omega_alphas = np.zeros(self.n_v)
            self.gamma_alphas = np.zeros(self.n_v)

            self.internal_angles1 = np.zeros(self.n_v)
            self.internal_angles2 = np.zeros(self.n_v)
            self.internal_angles3 = np.zeros(self.n_v)
            self.internal_angles4 = np.zeros(self.n_v)
            self.internal_angles5 = np.zeros(self.n_v)
            self.angle_opps = np.zeros(self.n_v)

            for itr1 in range(0,self.n_v):
                itr2 = itr1+1
                pa1 = self.bounding_xyz[itr1] #vertex 1
                pa2 = self.bounding_xyz[itr2] #vertex 2
                cos_beta12 = np.dot(pa1,pa2) #cos of angle between pa1 and pa2
                #TODO check consistent beta12 and beta12_alt
                self.betas[itr1] = np.arccos(cos_beta12) #angle between pa1 and pa2
                cross_12 = np.cross(pa2,pa1)
                sin_beta12 = np.linalg.norm(cross_12) #magnitude of cross product
                assert(np.isclose(sin_beta12,np.sin(self.betas[itr1])))
                self.betas_alt[itr1] = np.arcsin(sin_beta12)
                #TODO  don't use isclose here
                if np.isclose(self.betas[itr1],0.):
                    print "polygon_geo: side length 0, directions uncontrained, picking directions arbitrarily"
                    #z_hat is not uniquely defined here so arbitrarily pick one orthogonal to pa1, #TODO: probably a better way to do this
                    arbitrary = np.zeros(3)
                    if not np.isclose(np.abs(pa1[0]),1.):
                        arbitrary[0] = 1.
                    elif not isclose(np.abs(pa1[1]),1.):
                        arbitrary[1] = 1.
                    else:
                        arbitrary[2] = 1.
                    cross_12 = np.cross(arbitrary,pa1)
                    sin_beta12 = np.linalg.norm(cross_12)
                #TODO check if isclose is right
                elif np.isclose(self.betas[itr1],np.pi):
                    raise RuntimeError("polygon_geo: Spherical polygons with sides of length pi are not uniquely determined")
                self.z_hats[itr1] = cross_12/sin_beta12 #direction of cross product
                #three euler rotation angles #TODO check, especially signs
                if not (np.isclose(self.z_hats[itr1,1],0.) and np.isclose(self.z_hats[itr1,0],0.)):
                    self.theta_alphas[itr1] = -np.arccos(self.z_hats[itr1,2])
                    y1 = np.cross(self.z_hats[itr1],pa1)
                    self.y_hats[itr1] = y1
                    assert(np.allclose(pa1*np.cos(self.betas[itr1])-y1*np.sin(self.betas[itr1]),pa2))
                    assert(np.allclose(np.cross(pa1,y1),self.z_hats[itr1]))
                    self.xps[itr1] = np.array([self.z_hats[itr1][1]*pa1[0]-self.z_hats[itr1][0]*pa1[1],
                        self.z_hats[itr1][1]*y1[0]-self.z_hats[itr1][0]*y1[1],0.])
                    
                    #self.gamma_alphas[itr1] = np.arctan2(self.z_hats[itr1,1],self.z_hats[itr1,0])-np.pi/2.
                    self.gamma_alphas[itr1] = np.mod(np.arctan2(-self.z_hats[itr1,0],self.z_hats[itr1,1]),2.*np.pi)
                    self.omega_alphas[itr1] = -np.arctan2(self.xps[itr1,1],self.xps[itr1,0])
                    #self.omega_alphas[itr1] = np.mod(np.arccos(np.dot(pa1,np.array([np.cos(self.gamma_alphas[itr1]),np.sin(self.gamma_alphas[itr1]),0]))),2.*np.pi)
                else:
                    self.omega_alphas[itr1] = 0.
                    self.gamma_alphas[itr1] = np.mod(np.arctan2(pa1[1],pa1[0]),2.*np.pi)
                    #need to handle the case where z||z_hat separately (so don't divide by 0)
                    if self.z_hats[itr1,2]<0:
                        print "polygon_geo: setting theta_alpha to pi at "+str(itr1)
                        #TODO check
                        self.theta_alphas[itr1] = np.pi
                    else:
                        print "polygon_geo: setting theta_alpha to 0 at "+str(itr1)
                        self.theta_alphas[itr1] = 0.

            
            self.expand_alm_table(np.arange(1,l_max+1))
            print "polygon_geo: finished initialization" 

        #use angle defect formula (Girard's theorem) to get a00 
        def angular_area(self):
            return (np.sum(self.internal_angles)-(self.n_v-2.)*np.pi)

        def expand_alm_table(self,ls):
            if not isinstance(ls,np.ndarray):
                ls = np.array([ls])
            n_l = ls.size
            d_alm_table1 = np.zeros(n_l,dtype=object)
            Y_r_dict = get_Y_r_dict(np.max(ls),np.zeros(1)+np.pi/2.,np.zeros(1))
            self.factorials = sp.misc.factorial(np.arange(0,2*np.max(ls)+1))
            #coeff_table = {(1,0):1,(1,1):0}
#            for ll in range(1,np.max(ls)):
#                #TODO use other legendre function to be safe
#                coeff_table[(ll+1,0)] = assoc_legendre_p(ll,0,0.)
#                coeff_table[(ll+1,ll+1)] = 0.
#                for mm in range(1,ll+1):
#                    if mm<ll:
#                        if ll==25 and mm==3:
#                            print coeff_table[(ll,mm+1)]
#                            print coeff_table[(ll,mm-1)]
#                        #    sys.exit()
#                        coeff_table[(ll+1,mm)] = -1./(2.*mm)*(np.sqrt((ll-mm+1)*(ll-mm))*coeff_table[(ll,mm+1)]+(ll+mm-1.)*np.sqrt((ll+mm)/(ll+mm+1.))*coeff_table[(ll,mm-1)])
#                    else:
#                        coeff_table[(ll+1,mm)] = -1./(2.*mm)*((ll+mm-1.)*np.sqrt((ll+mm)/(ll+mm+1.))*coeff_table[(ll,mm-1)])
                   # sys.exit()
            for l_itr in range(0,ls.size):
                ll=ls[l_itr]
                d_alm_table1[l_itr] = np.zeros((2*ll+1,self.n_v))
                for mm in range(0,ll+1):
                    #TODO lpmv may be introducing error
                    #if np.abs(coeff_table[(ll,mm)]-np.sqrt(self.factorials[ll-mm]/self.factorials[ll+mm])*spp.lpmv(mm,ll-1,0.))/(np.sqrt(self.factorials[ll-mm]/self.factorials[ll+mm])*spp.lpmv(mm,ll-1,0.))>1.0:
                    #    print ll,mm
                    #    print coeff_table[(ll,mm)]
                    #    sys.exit()
                    #if not np.isclose(coeff_table[ll,mm],np.sqrt(self.factorials[ll-mm]/self.factorials[ll+mm])*spp.lpmv(mm,ll-1,0.)):
                    #    print ll,mm
                    #    print coeff_table[ll,mm]
                    #prefactor = (ll+mm)/(ll*(ll+1.))*np.sqrt((2.*ll+1.)/(4.*np.pi))*coeff_table[(ll,mm)]#self.factorials[ll-mm]/self.factorials[ll+mm])*spp.lpmv(mm,ll-1,0.)
                    #if ll-1>=mm:
                    #    prefactor = (ll+mm)/(ll*(ll+1.))*np.sqrt((2.*ll+1.)/(4.*np.pi)*self.factorials[ll-mm]/self.factorials[ll+mm])*assoc_legendre_p(ll-1,mm,0.)
                    #else:
                    #prefactor = (ll+mm)/(ll*(ll+1.))*np.sqrt((2.*ll+1.)/(4.*np.pi)*self.factorials[ll-mm]/self.factorials[ll+mm])*spp.lpmv(mm,ll-1,0.)
                    if mm>ll-1:
                        prefactor = 0.
                    else:
                        prefactor = (-1)**mm*np.sqrt((4.*ll**2-1.)*(ll-mm)*(ll+mm)/2.)/(ll*(-1+ll+2*ll**2))*Y_r_dict[(ll-1,mm)]
                    #else:
                    #    prefactor=0
                    if np.isnan(prefactor):
                        raise ValueError('prefactor is nan at l='+str(ll)+',m='+str(mm))
                    if mm==0:
                        d_alm_table1[l_itr][ll+mm] = np.sqrt(2.)*prefactor*self.betas
                    else:
                        d_alm_table1[l_itr][ll+mm] = prefactor*np.sqrt(2.)/mm*np.sin(self.betas*mm)
                        d_alm_table1[l_itr][ll-mm] = prefactor*np.sqrt(2.)/mm*(1.-np.cos(self.betas*mm))

            d_alm_table2 = au.rot_alm_z(d_alm_table1,self.omega_alphas,ls)
            d_alm_table3 = au.rot_alm_x(d_alm_table2,self.theta_alphas,ls,n_double=self.n_double)
            d_alm_table4 = au.rot_alm_z(d_alm_table3,self.gamma_alphas,ls)

            for l_itr in range(0,ls.size):
                ll = ls[l_itr] 
                for mm in range(0,ll+1):
                    if mm==0:
                        self.alm_table[(ll,mm)] = np.sum(d_alm_table4[l_itr][ll+mm])
                    else:
                        self.alm_table[(ll,mm)] = np.sum(d_alm_table4[l_itr][ll+mm])
                        self.alm_table[(ll,-mm)] =np.sum(d_alm_table4[l_itr][ll-mm])

        def a_lm(self,l,m):
            alm = self.alm_table.get((l,m))
            #does not currently support fetching values not cached
            if alm is None:
                self.expand_alm_table(l)
                alm = self.alm_table[(l,m)] 
                if alm is None:
                    raise RuntimeError('polygon_geo: a_lm generation failed for unknown reason')
            return alm

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

def check_mutually_orthonormal(vectors):
    fails = 0
    for itr1 in range(0,vectors.shape[0]):
        for itr2  in range(0,vectors.shape[0]):
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

if __name__=='__main__':
    toffset = np.pi/6.
    theta0=np.pi/6+toffset
    theta1=np.pi/3.+toffset
    theta2=np.pi/3.+0.1+toffset
    theta3=theta2-np.pi/3.
    theta4 = theta3+np.pi/6.
    offset=np.pi/6.
    phi0=0.+offset
    phi1 = np.pi/4.+offset
    phi2 = phi1+np.pi/2.
    phi3 = phi2-np.pi/6.
    phi4 = phi3-np.pi/3.
  

    #thetas = np.array([theta0,theta1,theta1,theta0,theta0])
    #phis = np.array([phi0,phi0,phi1,phi1,phi0])
    #thetas = np.array([theta0,theta1,theta1,theta0,theta0])
    #phis = np.array([phi0,phi0,phi1,phi1,phi0])
    thetas = np.array([theta0,theta1,theta1,theta2,theta0,theta3,theta3,theta4,theta0])
    phis = np.array([phi0,phi0,phi1,phi1,phi2,phi3,phi4,phi4,phi0])
    theta_in = np.pi/4.+toffset
    phi_in = np.pi/6.+offset
    res_choose = 6
    res_choose2 = 7
    l_max = 50

    
    #some setup to make an actual geo
    d=np.loadtxt('camb_m_pow_l.dat')
    k=d[:,0]; P=d[:,1]
    C=CosmoPie(k=k,P_lin=P,cosmology=defaults.cosmology)
    zs=np.array([.01,1.01])
    z_fine = np.arange(defaults.lensing_params['z_min_integral'],np.max(zs),defaults.lensing_params['z_resolution'])

    
    t0 = time()
    poly_params = defaults.polygon_params.copy()
    poly_geo = polygon_geo(zs,thetas,phis,C,z_fine,l_max=l_max,poly_params=poly_params)
    t1 = time()
    print "polygon_geo initialized in time: "+str(t1-t0) 
    pp_geo = polygon_pixel_geo(zs,thetas,phis,theta_in,phi_in,C,z_fine,l_max=l_max,res_healpix=res_choose)
    t2 = time()
    print "polygon_pixel_geo initialized at res "+str(res_choose)+" in time: "+str(t2-t1)
    pp_geo2 = polygon_pixel_geo(zs,thetas,phis,theta_in,phi_in,C,z_fine,l_max=l_max,res_healpix=res_choose2)
    t3 = time()
    print "polygon_pixel_geo initialized at res "+str(res_choose2)+" in time: "+str(t3-t2)
    #r_geo = rect_geo(zs,np.array([theta0,theta1]),np.array([phi0,phi1]),C,z_fine)

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
    for itr in range(0,nt):
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
    for itr in range(0,nt):
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
    for itr in range(0,nt):
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
    #alm_ratsfp = np.zeros(l_max+1)
    #alm_ratsfm = np.zeros(l_max+1)
    #for ll in range(0,l_max+1):
    #    alm_rats0[ll] = pp_geo.a_lm(ll,0)/poly_geo.alm_table[(ll,0)]
    #    alm_ratsfp[ll] = pp_geo.a_lm(ll,ll)/poly_geo.alm_table[(ll,ll)]
    #    alm_ratsfm[ll] = pp_geo.a_lm(ll,-ll)/poly_geo.alm_table[(ll,-ll)]

    #import matplotlib.pyplot as plt 
    #plt.plot(1./alm_rats0)
    #plt.plot(1./alm_ratsfp)
    #plt.plot(1./alm_ratsfm)
    #plt.show()
    my_table = poly_geo.alm_table.copy()
    #get rect_geo to cache the values in the table
    #for ll in range(0,l_max+1):
    #    for mm in range(0,ll+1):
    #        r_geo.a_lm(ll,mm)
    #        if mm>0:
    #            r_geo.a_lm(ll,-mm)
    #r_alm_table = r_geo.alm_table
    #reconstruct at higher resolution to mitigate resolution effects in determining accuracy
    totals_pp= pp_geo2.reconstruct_from_alm(l_max,pp_geo2.all_pixels[:,0],pp_geo2.all_pixels[:,1],pp_geo.alm_table)
    totals_poly = pp_geo2.reconstruct_from_alm(l_max,pp_geo2.all_pixels[:,0],pp_geo2.all_pixels[:,1],my_table)
    avg_diff = np.average(np.abs(totals_pp-totals_poly))
    print "mean absolute difference between pixel and exact geo reconstruction: "+str(avg_diff)
    poly_error = np.average(np.abs(totals_poly-pp_geo2.contained*1.))
    pp_error = np.average(np.abs(totals_pp-pp_geo2.contained*1.))
    print "mean absolute reconstruction error of exact geo: "+str(poly_error)
    print "mean absolute reconstruction error of pixel geo at res "+str(res_choose)+": "+str(pp_error)
    print "improvement in absolute reconstruction accuracy: "+str((pp_error-poly_error)/pp_error*100)+"%"

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
