"""Module for a spherical polygon Geo, defined by vertices with great circle edges"""
from __future__ import division,print_function,absolute_import
from builtins import range
from warnings import warn
import numpy as np
import spherical_geometry.vector as sgv
import spherical_geometry.great_circle_arc as gca
#from polygon_pixel_geo import get_Y_r_dict
from ylm_utils_mpmath import get_Y_r_dict_central
#from ylm_utils import get_Y_r_dict
#from ylm_utils_mpmath import get_Y_r_dict as get_Y_r_dict4
import alm_utils as au
from geo import Geo
from polygon_utils import get_poly

#get an exact spherical polygon geo
class PolygonGeo(Geo):
    """create a spherical polygon defined by clockwise vertices"""
    #vertices should be specified such that the clockwise oriented contour contains the area
    #theta_in and phi_in do not actually determine the inside, but are necessary for now if generating intersects with SphericalPolygon
    def __init__(self,zs,thetas,phis,theta_in,phi_in,C,z_fine,l_max,poly_params):
        """     inputs:
                    zs: the tomographic z bins
                    thetas,phis: an array of theta values for the edges in radians, last value should be first for closure, edges will be clockwise
                    theta_in,phi_in: a theta and phi known to be outside, needed for finding intersect for now
                    C: a CosmoPie object
                    z_fine: the resolution z slices
                    l_max: the maximum l to compute the alm table to
                    poly_params: a dict of parameters
        """

        self.poly_params = poly_params
        self.n_double = poly_params['n_double']
        self.l_max = l_max

        #maximum alm already available, only a00 available at start
        self._l_max = 0
        self.n_v = thetas.size-1

        self.bounding_theta = thetas-np.pi/2. #to radec
        self.bounding_phi = np.pi-phis
        self.bounding_xyz = np.asarray(sgv.radec_to_vector(self.bounding_phi,self.bounding_theta,degrees=False)).T
        self.theta_in = theta_in
        self.phi_in = phi_in
        self.thetas_orig = thetas
        self.phis_orig = phis

        #this gets correct internal angles with specified vertex order (copied from spherical_polygon clas)
        angle_body = gca.angle(self.bounding_xyz[:-2], self.bounding_xyz[1:-1], self.bounding_xyz[2:], degrees=False)
        angle_end = gca.angle(self.bounding_xyz[-2], self.bounding_xyz[0], self.bounding_xyz[1], degrees=False)
        self.internal_angles = 2.*np.pi-np.hstack([angle_body,angle_end])

        Geo.__init__(self,zs,C,z_fine)

        self.sp_poly = get_poly(thetas,phis,theta_in,phi_in)
        print("PolygonGeo: area calculated by SphericalPolygon: "+str(self.sp_poly.area()))
        print("PolygonGeo: area calculated by PolygonGeo: "+str(self.angular_area())+" sr or "+str(self.angular_area()*(180./np.pi)**2)+" deg^2")

        self.alm_table = {(0,0):self.angular_area()/np.sqrt(4.*np.pi)}


        self.z_hats = np.zeros((self.n_v,3))
        self.y_hats = np.zeros_like(self.z_hats)
        self.xps = np.zeros_like(self.z_hats)

        self.betas = np.zeros(self.n_v)
        self.theta_alphas = np.zeros(self.n_v)
        self.omega_alphas = np.zeros(self.n_v)
        self.gamma_alphas = np.zeros(self.n_v)

        for itr1 in range(0,self.n_v):
            itr2 = itr1+1
            pa1 = self.bounding_xyz[itr1] #vertex 1
            pa2 = self.bounding_xyz[itr2] #vertex 2
            cos_beta12 = np.dot(pa2,pa1) #cos of angle between pa1 and pa2
            cross_12 = np.cross(pa2,pa1)
            sin_beta12 = np.linalg.norm(cross_12) #magnitude of cross product
            self.betas[itr1] = np.arctan2(sin_beta12,cos_beta12)  #angle between pa1 and pa2
            #angle should be in quadrant expected by arccos because angle should be <pi
            beta_alt = np.arccos(cos_beta12)

            assert np.isclose(sin_beta12,np.sin(self.betas[itr1]))
            assert np.isclose(sin_beta12**2+cos_beta12**2,1.)
            assert np.isclose(beta_alt,self.betas[itr1])

            #define z_hat if possible
            if np.isclose(self.betas[itr1],0.):
                print("PolygonGeo: side length 0, directions unconstrained, picking directions arbitrarily")
                #z_hat is not uniquely defined here so arbitrarily pick one orthogonal to pa1
                arbitrary = np.zeros(3)
                if not np.isclose(np.abs(pa1[0]),1.):
                    arbitrary[0] = 1.
                elif not np.isclose(np.abs(pa1[1]),1.):
                    arbitrary[1] = 1.
                else:
                    arbitrary[2] = 1.
                cross_12 = np.cross(arbitrary,pa1)
                self.z_hats[itr1] = cross_12/np.linalg.norm(cross_12)
            elif np.isclose(self.betas[itr1],np.pi):
                raise RuntimeError("PolygonGeo: Spherical polygons with sides of length pi are not uniquely determined")
            else:
                self.z_hats[itr1] = cross_12/sin_beta12 #direction of cross product

            #three euler rotation angles #TODO check, especially signs
            #TODO maybe be more specific than isclose here, specify tolerance
            if not (np.isclose(self.z_hats[itr1,1],0.) and np.isclose(self.z_hats[itr1,0],0.)):
                #TODO can theta_alphas be defined with an arctan2?
                self.theta_alphas[itr1] = -np.arccos(self.z_hats[itr1,2])
                y1 = np.cross(self.z_hats[itr1],pa1)
                self.y_hats[itr1] = y1
                assert np.allclose(pa1*np.cos(self.betas[itr1])-y1*np.sin(self.betas[itr1]),pa2)
                assert np.allclose(np.cross(pa1,y1),self.z_hats[itr1])
                self.xps[itr1] = np.array([self.z_hats[itr1][1]*pa1[0]-self.z_hats[itr1][0]*pa1[1],self.z_hats[itr1][1]*y1[0]-self.z_hats[itr1][0]*y1[1],0.])

                self.gamma_alphas[itr1] = np.arctan2(-self.z_hats[itr1,0],self.z_hats[itr1,1])
                gamma_alpha2 =  np.arctan2(self.z_hats[itr1,1],self.z_hats[itr1,0])-np.pi/2.
                assert np.isclose(np.mod(self.gamma_alphas[itr1],2.*np.pi),np.mod(gamma_alpha2,2.*np.pi))
                self.omega_alphas[itr1] = -np.arctan2(self.xps[itr1,1],self.xps[itr1,0])
            else:
                self.omega_alphas[itr1] = 0.
                self.gamma_alphas[itr1] = np.arctan2(pa1[1],pa1[0])
                #need to handle the case where z||z_hat separately (so don't divide by 0)
                if self.z_hats[itr1,2]<0:
                    print("PolygonGeo: setting theta_alpha to pi at "+str(itr1))
                    self.theta_alphas[itr1] = np.pi
                else:
                    print("PolygonGeo: setting theta_alpha to 0 at "+str(itr1))
                    self.theta_alphas[itr1] = 0.


        self.expand_alm_table(l_max)
        print("PolygonGeo: finished initialization")

    #use angle defect formula (Girard's theorem) to get a00
    def angular_area(self):
        return np.sum(self.internal_angles)-(self.n_v-2.)*np.pi

    def expand_alm_table(self,l_max):
        """expand the table of alms out to specified l_max"""
        if self._l_max<0:
            self.alm_table[(0,0)] = self.angular_area()/np.sqrt(4.*np.pi)
            self._l_max = 0

        ls = np.arange(np.int(self._l_max)+1,np.int(l_max)+1)
        n_l = ls.size
        if n_l==0:
            return
        d_alm_table1 = np.zeros(n_l,dtype=object)
        #Y_r(l,m,pi/2.,0.) has an analytic solution, use it
        Y_r_dict = get_Y_r_dict_central(np.max(ls))

#for checking consistency of Y_r_dict2 and Y_r_dict
#        Y_r_dict2 = get_Y_r_dict(np.max(ls),np.zeros(1)+np.pi/2.,np.zeros(1))
#        #Y_r_dict3 = get_Y_r_dict3(np.max(ls),np.zeros(1)+np.pi/2.,np.zeros(1))
#        Y_r_dict4 = get_Y_r_dict4(np.max(ls),np.zeros(1)+np.pi/2.,np.zeros(1))
#        keys1 = sorted(Y_r_dict.keys())
#        keys2 = sorted(Y_r_dict2.keys())
#        #keys3 = sorted(Y_r_dict3.keys())
#        keys4 = sorted(Y_r_dict4.keys())
#        assert keys1==keys1
#        #assert keys1==keys3
#        assert keys1==keys4
#        n_k = len(keys1)
#        vals1 = np.zeros(n_k)
#        vals2 = np.zeros(n_k)
#        #vals3 = np.zeros(n_k)
#        vals4 = np.zeros(n_k)
#        for itr in range(0,n_k):
#            vals1[itr] = Y_r_dict[keys1[itr]]
#            vals2[itr] = Y_r_dict2[keys2[itr]]
#        #    vals3[itr] = Y_r_dict3[keys3[itr]]
#            vals4[itr] = Y_r_dict4[keys4[itr]]
#        assert np.allclose(vals1,vals2)
#        #assert np.allclose(vals1,vals3)
#        assert np.allclose(vals1,vals4)
#####
        for l_itr in range(0,ls.size):
            ll = ls[l_itr]
            d_alm_table1[l_itr] = np.zeros((2*ll+1,self.n_v))
            for mm in range(0,ll+1):
                if mm>ll-1:
                    prefactor = 0.
                else:
                    #if ll-mm is odd or mm<0, then Y_r(ll,mm,pi/2,0)=0 anayltically, enforce
                    if mm<0 or  (mm-ll-1) % 2==1:
                        prefactor = 0.
                    else:
                        prefactor = (-1)**mm*np.sqrt((4.*ll**2-1.)*(ll-mm)*(ll+mm)/2.)/(ll*(-1+ll+2*ll**2))*Y_r_dict[(ll-1,mm)]
                #else:
                #    prefactor = 0
                if not np.all(np.isfinite(prefactor)):
                    #print(ll,mm,Y_r_dict[(ll-1,mm)])
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
                    self.alm_table[(ll,-mm)] = np.sum(d_alm_table4[l_itr][ll-mm])
        self._l_max = l_max

    def a_lm(self,l,m):
        """get a(l,m) for the geometry"""

        if l>self._l_max:
            print("PolygonGeo: l value "+str(l)+" exceeds maximum precomputed l "+str(self._l_max)+",expanding table")
            self.expand_alm_table(l)

        alm = self.alm_table.get((l,m))
        if alm is None:
            raise RuntimeError('PolygonGeo: a_lm generation failed for unknown reason at l='+str(l)+',m='+str(m))
        return alm

    #TODO make robust for type of poly2
    def get_overlap_fraction(self,geo2):
        """get the overlap fraction between this geometry and another SphericalPolygon"""
        #there is a bug in spherical_geometry that causes overlap to fail if geometries are nested and 1 side is identical, handle this case unless they fix it
        try:
            result = self.sp_poly.overlap(geo2.sp_poly)
        except Exception:
            warn('spherical_geometry overlap failed, assuming total overlap')
            if self.angular_area()<=geo2.angular_area():
                result = 1.
            else:
                result = geo2.angular_area()/self.angular_area()
        return result
