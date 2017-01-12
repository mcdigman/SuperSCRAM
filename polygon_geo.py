import numpy as np
from astropy.io import fits
import spherical_geometry.vector as sgv
from spherical_geometry.polygon import SphericalPolygon
from polygon_pixel_geo import polygon_pixel_geo
from sph_functions import Y_r
from geo import pixel_geo,rect_geo,geo
from time import time
import defaults
import scipy as sp
from cosmopie import CosmoPie
from scipy.interpolate import SmoothBivariateSpline
from warnings import warn
import scipy.special as spp

#get an exact spherical polygon geo
class polygon_geo(geo):
        def __init__(self,zs,thetas,phis,theta_in,phi_in,C,z_fine,l_max=defaults.polygon_params['l_max']):
            self.sp_poly = get_poly(thetas,phis,theta_in,phi_in)
            geo.__init__(self,zs,C,z_fine)
            #check that the true area from angular defect formula and calculated area approximately match
#            if not np.isclose(np.sum(contained_pixels[:,2]),self.sp_poly.area(),atol=10**-2,rtol=10**-3):
#                warn("polygon_pixel_geo: significant discrepancy between true area "+str(self.sp_poly.area())+" and calculated area"+str(np.sum(contained_pixels[:,2]))+" results may be poorly converged")
            self.bounding_theta = thetas-np.pi/2. #to radec
            self.bounding_phi = np.pi-phis
            #self.bounding_phi = phis
            self.bounding_xyz = np.asarray(sgv.radec_to_vector(self.bounding_phi,self.bounding_theta,degrees=False)).T
            #flip z axis to match my convention
            self.betas = np.zeros(thetas.size-1)
            self.betas_alt = np.zeros(thetas.size-1)
            self.z_hats = np.zeros((thetas.size-1,3))
            self.theta_alphas = np.zeros((thetas.size-1))
            self.omega_alphas = np.zeros((thetas.size-1))
            self.gamma_alphas = np.zeros((thetas.size-1))
            self.angle2 = np.zeros((thetas.size-1))
            self.ys = np.zeros_like(self.z_hats)
            self.xps = np.zeros_like(self.z_hats)
            for itr1 in range(0,thetas.size-1):
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
                #TODO handle 0s properly
                if np.isclose(self.betas[itr1],0.):
                    print "side length 0, directions ambiguous, picking directions arbitrarily"
                    #z_hat is not uniquely defined here so arbitrarily pick one orthogonal to pa1, #TODO: probably a better way to do this
                    arbitrary = np.zeros(3)
                    if not np.isclose(np.abs(pa1[0]),1.):
                        arbitrary[0] = 1.
                    elif not isclose(np.abs(pa1[1]),1.):
                        arbitrary[1] = 1.
                    else:
                        arbitrary[2] = 1.
                    #if not np.isclose(np.abs(pa1[0]),1.):
                    #    arbitrary = np.dot(pa1,np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]]))
                    #elif not np.isclose(np.abs(pa1[0]),1.):
                    #    arbitrary = np.dot(pa1,np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]))
                    #else:
                    #    arbitrary = np.dot(pa1,np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]]))
                    cross_12 = np.cross(arbitrary,pa1)
                    sin_beta12 = np.linalg.norm(cross_12)
                #TODO check if isclose is right
                elif np.isclose(self.betas[itr1],np.pi):
                    raise RuntimeError("Spherical polygons with sides of length pi are not uniquely determined")
                self.z_hats[itr1] = cross_12/sin_beta12 #direction of cross product
                #three euler rotation angles #TODO check, especially signs
                self.theta_alphas[itr1] = -np.arccos(self.z_hats[itr1,2])
                if not (np.isclose(self.z_hats[itr1,1],0.) and np.isclose(self.z_hats[itr1,0],0.)):
                    y1 = np.cross(self.z_hats[itr1],pa1)
                    self.ys[itr1] = y1
                    assert(np.allclose(pa1*np.cos(self.betas[itr1])-y1*np.sin(self.betas[itr1]),pa2))
                    assert(np.allclose(np.cross(pa1,y1),self.z_hats[itr1]))
                    self.xps[itr1] = np.array([self.z_hats[itr1][1]*pa1[0]-self.z_hats[itr1][0]*pa1[1],
                        self.z_hats[itr1][1]*y1[0]-self.z_hats[itr1][0]*y1[1],0.])
                    
                    #self.gamma_alphas[itr1] = np.arctan2(self.z_hats[itr1,1],self.z_hats[itr1,0])-np.pi/2.
                    self.gamma_alphas[itr1] = np.mod(np.arctan2(-self.z_hats[itr1,0],self.z_hats[itr1,1]),2.*np.pi)
                    self.omega_alphas[itr1] = -np.arctan2(self.xps[itr1,1],self.xps[itr1,0])
                    #self.omega_alphas[itr1] = np.mod(np.arccos(np.dot(pa1,np.array([np.cos(self.gamma_alphas[itr1]),np.sin(self.gamma_alphas[itr1]),0]))),2.*np.pi)
                    #self.gamma_alphas[itr1]-= np.pi
                else:
                    #Hack to handle the edge case (so don't divide by 0), TODO needs to be handled differently
                    if self.z_hats[itr1,2]<0:
                        print "setting gamma_alpha to 0 at "+str(itr1)
                        self.omega_alphas[itr1] = -np.arccos(np.dot(pa1,np.array([1.,0.,0.])))
                        #TODO check
                        if np.mod(self.bounding_phi[itr1],2.*np.pi)>np.pi:
                            self.gamma_alphas[itr1] = np.pi
                        else:
                            self.gamma_alphas[itr1] = 0.#-self.omega_alphas[itr1]
                        #self.gamma_alphas[itr1] = 0.
                    else:
                        print "setting gamma_alpha to pi at "+str(itr1)
                        self.omega_alphas[itr1] = np.arccos(np.dot(pa1,np.array([1.,0.,0.])))
                        self.gamma_alphas[itr1] = 0.#-self.omega_alphas[itr1]
                        #self.gamma_alphas[itr1] = np.pi/2.
                #else:
                #    print itr1
                #    self.gamma_alphas[itr1] = 0.
                #else:
                #    self.gamma_alphas[itr1] = np.pi/2.
                #self.omega_alphas[itr1] = np.arccos(np.dot(pa1,np.cross(np.array([0,0,1]),self.z_hats[itr1]))) 
             #   self.gamma_alphas[itr1]+=np.pi/3.
                #TODO check, what is this?
                #self.angle2[itr1] =np.pi -np.arcsin(self.p_hats[itr1,1]/np.sin(self.theta_alphas[itr1]))

            factorials = sp.misc.factorial(np.arange(0,2*l_max+1))
            self.d_alm_table1 = {}#{(0,0):self.sp_poly.area()/np.sqrt(4.*np.pi)}
            self.d_alm_table2 = {}#{(0,0):self.sp_poly.area()/np.sqrt(4.*np.pi)}
            self.d_alm_table3 = {}#{(0,0):self.sp_poly.area()/np.sqrt(4.*np.pi)}
            self.d_alm_table4 = {}#{(0,0):self.sp_poly.area()/np.sqrt(4.*np.pi)}

            for ll in range(1,l_max+1):
                for mm in range(0,ll+1):
                    #TODO lpmv may be introducing error
                    prefactor = (ll+mm)/(ll*(ll+1.))*np.sqrt((2.*ll+1.)/(4.*np.pi)*factorials[ll-mm]/factorials[ll+mm])*spp.lpmv(mm,ll-1,0.)
                    #TODO check sins
                    if mm==0:
                        self.d_alm_table1[(ll,mm)] = prefactor*self.betas
                    else:
                        self.d_alm_table1[(ll,mm)] = prefactor*np.sqrt(2.)/mm*np.sin(self.betas*mm)
                        self.d_alm_table1[(ll,-mm)] = prefactor*np.sqrt(2.)/mm*(1.-np.cos(self.betas*mm))
            #TODO check all this, make efficient 
            for ll in range(1,l_max+1):
                for mm in range(0,ll+1):
                    if mm==0:
                        self.d_alm_table2[(ll,mm)] = self.d_alm_table1[(ll,mm)]
                    else:
                        self.d_alm_table2[(ll,mm)] = np.cos(mm*self.omega_alphas)*self.d_alm_table1[(ll,mm)]+np.sin(mm*self.omega_alphas)*self.d_alm_table1[(ll,-mm)]
                        self.d_alm_table2[(ll,-mm)] = -np.sin(mm*self.omega_alphas)*self.d_alm_table1[(ll,mm)]+np.cos(mm*self.omega_alphas)*self.d_alm_table1[(ll,-mm)]

            #NOTE: flipped signs
            n_double = 30
            self.rot_mat = {}
            self.el_mats = np.zeros(thetas.size-1,dtype=object)
            for itr in range(0,thetas.size-1):
                angle1 = self.theta_alphas[itr]
                epsilon = angle1/2**n_double
                self.el_mats[itr] = np.zeros(l_max+1,dtype=object)
                for ll in range(1,l_max+1): 
                    lplus = np.zeros((2*ll+1,2*ll+1))
                    lminus = np.zeros((2*ll+1,2*ll+1))
                    for mm in range(-ll,ll+1):
                        if mm+ll+1<lplus.shape[1]:
                            lplus[mm+ll,mm+ll+1] =np.sqrt(ll*(ll+1.)-mm*(mm+1.))
                        if mm+ll>0:
                            lminus[mm+ll,mm+ll-1] =np.sqrt(ll*(ll+1.)-mm*(mm-1.))
                    el_mat = -1j*epsilon*(lplus+lminus)/2.
                    m_mat = np.zeros_like(el_mat)
                    for mm in range(0,ll+1):
                        if mm==0:
                            m_mat[mm+ll,mm+ll]=1.
                        else:
                            m_mat[mm+ll,mm+ll]=1./np.sqrt(2.)
                            m_mat[mm+ll,-mm+ll] = -1j/np.sqrt(2.)
                            m_mat[-mm+ll,mm+ll]=(-1)**mm/np.sqrt(2.)
                            m_mat[-mm+ll,-mm+ll] = 1j*(-1)**mm/np.sqrt(2.)

                    m_mat_i = np.zeros_like(el_mat)
                    for mm in range(0,ll+1):
                        if mm==0:
                            m_mat_i[mm+ll,mm+ll]=1.
                        else:
                            m_mat_i[mm+ll,mm+ll]=1./np.sqrt(2.)
                            m_mat_i[mm+ll,-mm+ll] = (-1)**mm/np.sqrt(2.)
                            m_mat_i[-mm+ll,mm+ll]=1j/np.sqrt(2.)
                            m_mat_i[-mm+ll,-mm+ll] = -1j*(-1)**mm/np.sqrt(2.)
                    #check m_mat and its inverse are actually inverse of each other
                    assert(np.allclose(np.identity(m_mat.shape[0]),np.dot(m_mat_i,m_mat)))
                    #infinitesimal form of El(epsilon)
                    el_mat = np.real(np.dot(np.dot(m_mat_i,el_mat),m_mat))
                    self.el_mats[itr][ll] = el_mat
                    #ensure E_l matrices are antihermitian
                    assert(np.all(el_mat==-el_mat.T))
                    #TODO add checks
                    #use angle doubling fomula to get to correct angle
                    for itr2 in range(0,n_double):
                        el_mat = 2*el_mat+np.dot(el_mat,el_mat)
                    d_mat = el_mat+np.identity(el_mat.shape[0])
                    for mm in range(-ll,ll+1):
                        for mmp in range(-ll,ll+1):
                            if not (ll,mmp,mm) in self.rot_mat:
                                self.rot_mat[(ll,mmp,mm)]=np.zeros(thetas.size-1)
                            self.rot_mat[(ll,mmp,mm)][itr] = d_mat[mmp+ll,mm+ll]

            for ll in range(1,l_max+1):
                for mm in range(0,ll+1):
                    if mm==0:
                        self.d_alm_table3[(ll,mm)] = np.zeros(thetas.size-1)
                    else:
                        self.d_alm_table3[(ll,mm)] = np.zeros(thetas.size-1)
                        self.d_alm_table3[(ll,-mm)] = np.zeros(thetas.size-1)
            for ll in range(1,l_max+1):
                for mm in range(0,ll+1):
                    prev1 = self.d_alm_table2[(ll,mm)]
                    prev2 = self.d_alm_table2[(ll,-mm)]

                    for mmp in range(0,ll+1):
                        if mmp==0 and mm==0:
                            self.d_alm_table3[(ll,mmp)]+=prev1*self.rot_mat[(ll,mmp,mm)]
                        elif mmp==0 and mm>0:
                            self.d_alm_table3[(ll,mmp)]+=prev1*self.rot_mat[(ll,mmp,mm)]+prev2*self.rot_mat[(ll,mmp,-mm)]
                        elif mmp>0 and mm==0:
                            self.d_alm_table3[(ll,mmp)]+=prev1*self.rot_mat[(ll,mmp,mm)]
                            self.d_alm_table3[(ll,-mmp)]+=prev1*self.rot_mat[(ll,-mmp,mm)]
                        elif mmp>0 and mm>0:
                            self.d_alm_table3[(ll,mmp)]+=prev1*self.rot_mat[(ll,mmp,mm)]+prev2*self.rot_mat[(ll,mmp,-mm)]
                            self.d_alm_table3[(ll,-mmp)]+=prev1*self.rot_mat[(ll,-mmp,mm)]+prev2*self.rot_mat[(ll,-mmp,-mm)]


            for ll in range(1,l_max+1):
                for mm in range(0,ll+1):
                    if mm==0:
                        self.d_alm_table4[(ll,mm)] = self.d_alm_table3[(ll,mm)]
                    else:
                        self.d_alm_table4[(ll,mm)] = np.cos(mm*self.gamma_alphas)*self.d_alm_table3[(ll,mm)]+np.sin(mm*self.gamma_alphas)*self.d_alm_table3[(ll,-mm)]
                        self.d_alm_table4[(ll,-mm)] = -np.sin(mm*self.gamma_alphas)*self.d_alm_table3[(ll,mm)]+np.cos(mm*self.gamma_alphas)*self.d_alm_table3[(ll,-mm)]

            self.alm_table = {(0,0):self.sp_poly.area()/np.sqrt(4.*np.pi)}
            for ll in range(1,l_max+1):
                for mm in range(0,ll+1):
                    if mm==0:
                        self.alm_table[(ll,mm)] = np.sum(self.d_alm_table4[(ll,mm)])
                    else:
                        self.alm_table[(ll,mm)] = np.sum(self.d_alm_table4[(ll,mm)])
                        self.alm_table[(ll,-mm)] =np.sum(self.d_alm_table4[(ll,-mm)])
        
            #precompute a table of alms
            #self.l_max = l_max
            #self.alm_table,ls,ms,self.alm_dict = self.get_a_lm_table(l_max)

        def angular_area(self):
            return self.sp_poly.area()

        def a_lm(self,l,m):
            #if not precomputed, get alm slow way, otherwise read it out of the table
            alm = self.alm_table[(l,m)] 
            if alm is None:
                alm = pixel_geo.a_lm(self,l,m)
                self.alm_table[(l,m)] = alm
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

            for ll in range(0,l_max+1):
                for mm in range(-ll,ll+1):
                    #first try takes ~0.618 sec/iteration for 290 pixel region=> 2.1*10**-3 sec/(iteration pixel), much too slow
                    ls[itr] = ll
                    ms[itr] = mm
                    a_lms[(ll,mm)] = self.a_lm(ll,mm)
                    itr+=1
            return a_lms,ls,ms
        #TODO check numerical stability
        def get_a_lm_table(self,l_max):
            n_tot = (l_max+1)**2
            pixel_area = self.pixels[0,2]

            ls = np.zeros(n_tot)
            ms = np.zeros(n_tot)
            a_lms = {}
            
            lm_dict = {}
            itr = 0
            for ll in range(0,l_max+1):
                for mm in range(-ll,ll+1):
                    ms[itr] = mm
                    ls[itr] = ll
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

            for ll in range(0,l_max+1):
                if ll>=2:
                    known_legendre[(ll,ll-1)] = (2.*ll-1.)*cos_theta*known_legendre[(ll-1,ll-1)]
                    known_legendre[(ll,ll)] = -(2.*ll-1.)*abs_sin_theta*known_legendre[(ll-1,ll-1)] 
                for mm in range(0,ll+1):
                    if mm<=ll-2:
                        known_legendre[(ll,mm)] = ((2.*ll-1.)/(ll-mm)*cos_theta*known_legendre[(ll-1,mm)]-(ll+mm-1.)/(ll-mm)*known_legendre[(ll-2,mm)])

                    prefactor = np.sqrt((2.*ll+1.)/(4.*np.pi)*factorials[ll-mm]/factorials[ll+mm])*pixel_area
                    #no sin theta because first order integrator
                    base = known_legendre[(ll,mm)]
                    if mm==0:
                        a_lms[(ll,mm)] = prefactor*np.sum(base)
                    else:
                        #Note: check condon shortley phase convention

                        a_lms[(ll,mm)] = (-1)**(mm)*np.sqrt(2.)*prefactor*np.sum(base*cos_phi_m[mm])
                        a_lms[(ll,-mm)] = (-1)**(mm)*np.sqrt(2.)*prefactor*np.sum(base*sin_phi_m[mm])
                    if mm<=ll-2:
                        known_legendre.pop((ll-2,mm),None)

            return a_lms,ls,ms,lm_dict

        #may be some loss of precision; fails to identify possible exact 0s
        def reconstruct_from_alm(self,l_max,thetas,phis,alms):
            n_tot = (l_max+1)**2

            ls = np.zeros(n_tot)
            ms = np.zeros(n_tot)
            reconstructed = np.zeros(thetas.size)
            
            lm_dict = {}
            itr = 0
            for ll in range(0,l_max+1):
                for mm in range(-ll,ll+1):
                    ms[itr] = mm
                    ls[itr] = ll
                    lm_dict[(ll,mm)] = itr
                    itr+=1
    
            cos_theta = np.cos(thetas)
            sin_theta = np.sin(thetas)
            abs_sin_theta = np.abs(sin_theta)


            sin_phi_m = np.zeros((l_max+1,thetas.size))
            cos_phi_m = np.zeros((l_max+1,thetas.size))
            for mm in range(0,l_max+1):
                sin_phi_m[mm] = np.sin(mm*phis)
                cos_phi_m[mm] = np.cos(mm*phis)

            factorials = sp.misc.factorial(np.arange(0,2*l_max+1))

            known_legendre = {(0,0):(np.zeros(thetas.size)+1.),(1,0):cos_theta,(1,1):-abs_sin_theta}

            for ll in range(0,l_max+1):
                if ll>=2:
                    known_legendre[(ll,ll-1)] = (2.*ll-1.)*cos_theta*known_legendre[(ll-1,ll-1)]
                    known_legendre[(ll,ll)] = -(2.*ll-1.)*abs_sin_theta*known_legendre[(ll-1,ll-1)] 
                for mm in range(0,ll+1):
                    if mm<=ll-2:
                        known_legendre[(ll,mm)] = ((2.*ll-1.)/(ll-mm)*cos_theta*known_legendre[(ll-1,mm)]-(ll+mm-1.)/(ll-mm)*known_legendre[(ll-2,mm)])

                    prefactor = np.sqrt((2.*ll+1.)/(4.*np.pi)*factorials[ll-mm]/factorials[ll+mm])
                    base = known_legendre[(ll,mm)]
                    if mm==0:
                        reconstructed += prefactor*alms[(ll,mm)]*base
                    else:
                        #Note: check condon shortley phase convention
                        reconstructed+= (-1)**(mm)*np.sqrt(2.)*alms[(ll,mm)]*prefactor*base*cos_phi_m[mm]
                        reconstructed+= (-1)**(mm)*np.sqrt(2.)*alms[(ll,-mm)]*prefactor*base*sin_phi_m[mm]
                if mm<=ll-2:
                    known_legendre.pop((ll-2,mm),None)

            return reconstructed

#alternate way of computing Y_r from the way in sph_functions
def Y_r_2(ll,mm,theta,phi,known_legendre):
    prefactor = np.sqrt((2.*ll+1.)/(4.*np.pi)*sp.misc.factorial(ll-np.abs(mm))/sp.misc.factorial(ll+np.abs(mm)))
    base = (prefactor*(-1)**mm)*known_legendre[(ll,np.abs(mm))]
    if mm==0:
        return base
    elif mm>0:
        return base*np.sqrt(2.)*np.cos(mm*phi)
    else:
        return base*np.sqrt(2.)*np.sin(np.abs(mm)*phi)
def get_Y_r_table(l_max,thetas,phis):
    n_tot = (l_max+1)**2

    ls = np.zeros(n_tot)
    ms = np.zeros(n_tot)
    Y_lms = np.zeros((n_tot,thetas.size))
            
    lm_dict = {}
    itr = 0
    for ll in range(0,l_max+1):
        for mm in range(-ll,ll+1):
            ms[itr] = mm
            ls[itr] = ll
            lm_dict[(ll,mm)] = itr
            itr+=1
    
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    abs_sin_theta = np.abs(sin_theta)


    sin_phi_m = np.zeros((l_max+1,thetas.size))
    cos_phi_m = np.zeros((l_max+1,thetas.size))
    for mm in range(0,l_max+1):
        sin_phi_m[mm] = np.sin(mm*phis)
        cos_phi_m[mm] = np.cos(mm*phis)

        factorials = sp.misc.factorial(np.arange(0,2*l_max+1))

    known_legendre = {(0,0):(np.zeros(thetas.size)+1.),(1,0):cos_theta,(1,1):-abs_sin_theta}

    for ll in range(0,l_max+1):
        if ll>=2:
            known_legendre[(ll,ll-1)] = (2.*ll-1.)*cos_theta*known_legendre[(ll-1,ll-1)]
            known_legendre[(ll,ll)] = -(2.*ll-1.)*abs_sin_theta*known_legendre[(ll-1,ll-1)] 
        for mm in range(0,ll+1):
            if mm<=ll-2:
                known_legendre[(ll,mm)] = ((2.*ll-1.)/(ll-mm)*cos_theta*known_legendre[(ll-1,mm)]-(ll+mm-1.)/(ll-mm)*known_legendre[(ll-2,mm)])

            prefactor = np.sqrt((2.*ll+1.)/(4.*np.pi)*factorials[ll-mm]/factorials[ll+mm])
            base = known_legendre[(ll,mm)]
            if mm==0:
                Y_lms[lm_dict[(ll,mm)]] = prefactor*base
            else:
                #Note: check condon shortley phase convention
 
                Y_lms[lm_dict[(ll,mm)]] = (-1)**(mm)*np.sqrt(2.)*prefactor*base*cos_phi_m[mm]
                Y_lms[lm_dict[(ll,-mm)]] = (-1)**(mm)*np.sqrt(2.)*prefactor*base*sin_phi_m[mm]
            if mm<=ll-2:
                known_legendre.pop((ll-2,mm),None)
              

    return Y_lms,ls,ms

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
    xyz_vals = sgv.radec_to_vector(pixels[:,1],pixels[:,0]-np.pi/2.,degrees=False)
    contained = np.zeros(pixels.shape[0],dtype=bool)
    #check if each point is contained in the polygon. This is fairly slow if the number of points is huge
    for i in range(0,pixels.shape[0]):
        contained[i]= sp_poly.contains_point([xyz_vals[0][i],xyz_vals[1][i],xyz_vals[2][i]])
    return contained

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

#PLAN: explicitly implement Y_r both ways for testing purposes
#FIX: make a_lm theta, phi pole actually at 0
#TODO fix issue if any thetas are greater than pi/2 (probably a quadrants issue)
if __name__=='__main__':
    toffset = np.pi/3.
    theta0=np.pi/6+toffset
    theta1=np.pi/3.+toffset
    theta2=np.pi/3.+0.1+toffset
    theta3=theta2-np.pi/3.
    theta4 = theta3+np.pi/6.
    offset=np.pi/6.#np.pi/2.
    phi0=0.+offset#np.pi/7.
    #phi1=np.pi+2.*np.pi/6.
    phi1 = np.pi/4.+offset
    phi2 = phi1+np.pi/2.
    phi3 = phi2-np.pi/6.
    phi4 = phi3-np.pi/3.
  

    #thetas = np.array([theta0,theta1,theta1,theta0,theta0])
    #phis = np.array([phi0,phi0,phi1,phi1,phi0])
   # thetas = np.array([theta0,theta1,theta1,theta0,theta0])
    #phis = np.array([phi0,phi0,phi1,phi1,phi0])
    thetas = np.array([theta0,theta1,theta1,theta2,theta0,theta3,theta3,theta4,theta0])
    phis = np.array([phi0,phi0,phi1,phi1,phi2,phi3,phi4,phi4,phi0])
    #thetas = np.array([np.pi/3.,np.pi/3.,np.pi/3,np.pi/3.])
    #phis = np.array([0.,2.*np.pi/3.,4.*np.pi/3.,0.])
    theta_in = np.pi/4.+toffset
    phi_in = np.pi/6.+offset
    res_choose = 6
    l_max = 85
    #pixels = get_healpix_pixelation(res_choose=res_choose)
    #sp_poly = get_poly(thetas,phis,theta_in,phi_in)
    #contained = is_contained(pixels,sp_poly)

    
    #some setup to make an actual geo
    d=np.loadtxt('camb_m_pow_l.dat')
    k=d[:,0]; P=d[:,1]
    C=CosmoPie(k=k,P_lin=P,cosmology=defaults.cosmology)
    zs=np.array([.01,1.01])
    z_fine = np.arange(defaults.lensing_params['z_min_integral'],np.max(zs),defaults.lensing_params['z_resolution'])

    
    t0 = time()
    poly_geo = polygon_geo(zs,thetas,phis,theta_in,phi_in,C,z_fine,l_max=l_max)
    t1 = time()
    
    pp_geo = polygon_pixel_geo(zs,thetas,phis,theta_in,phi_in,C,z_fine,res_healpix=res_choose,l_max=l_max)
    r_geo = rect_geo(zs,np.array([theta0,theta1]),np.array([phi0,phi1]),C,z_fine)
    #assert(np.allclose(poly_geo.betas,poly_geo.betas_alt))
    nt = thetas.size-1
   

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

    #p1_g = poly_geo.bounding_xyz[1:nt+1]
  
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

    alm_rats0 = np.zeros(l_max+1)
    alm_ratsfp = np.zeros(l_max+1)
    alm_ratsfm = np.zeros(l_max+1)
    for ll in range(0,l_max+1):
        alm_rats0[ll] = pp_geo.a_lm(ll,0)/poly_geo.alm_table[(ll,0)]
        alm_ratsfp[ll] = pp_geo.a_lm(ll,ll)/poly_geo.alm_table[(ll,ll)]
        alm_ratsfm[ll] = pp_geo.a_lm(ll,-ll)/poly_geo.alm_table[(ll,-ll)]

    #import matplotlib.pyplot as plt 
    #plt.plot(1./alm_rats0)
    #plt.plot(1./alm_ratsfp)
    #plt.plot(1./alm_ratsfm)
    #factorials = sp.misc.factorial(np.arange(l_max,2*l_max+1))/sp.misc.factorial(0,l_max+1)
    #plt.loglog(1./np.sqrt(factorials/factorials[0]))
    #plt.show()
    my_table = poly_geo.alm_table.copy()
    #get rect_geo to cache the values in the table
    #for ll in range(0,l_max+1):
    #    for mm in range(0,ll+1):
    #        r_geo.a_lm(ll,mm)
    #        if mm>0:
    #            r_geo.a_lm(ll,-mm)
    #r_alm_table = r_geo.alm_table
    totals_pp= pp_geo.reconstruct_from_alm(l_max,pp_geo.all_pixels[:,0],pp_geo.all_pixels[:,1],pp_geo.alm_table)
    totals_poly = pp_geo.reconstruct_from_alm(l_max,pp_geo.all_pixels[:,0],pp_geo.all_pixels[:,1],my_table)
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
            lats = (pp_geo.all_pixels[:,0]-np.pi/2.)*180/np.pi
            lons = pp_geo.all_pixels[:,1]*180/np.pi
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
            pp_geo.sp_poly.draw(m,color='red')
            if do_poly:
                ax = fig.add_subplot(122)
                H2,yedges2,xedges2 = np.histogram2d(y,x,100,weights=1.*totals_poly)
                X2, Y2 = np.meshgrid(xedges2, yedges2)
                ax.pcolormesh(X2,Y2,-H2,cmap='gray')
                ax.set_aspect('equal')
                #m.plot(x,y,'bo',markersize=1)
                ax.set_title("polygon_geo reconstruction")
                pp_geo.sp_poly.draw(m,color='red')
            plt.show()

