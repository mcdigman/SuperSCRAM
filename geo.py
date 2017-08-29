import numpy as np
from scipy.integrate import dblquad
from sph_functions import Y_r
from scipy.interpolate import InterpolatedUnivariateSpline

#Abstract class defining a geometry for a survey. 
#At the moment, a geometry must 
#1) Have a definite coarse z bin structure (for tomography)
#2) Have a definite fine z bin structure (for integrating over z)
#3) Be able to compute some kind of surface integral
#Most of the behavior should be defined in subclasses

class geo:
    def __init__(self,z_coarse,C,z_fine):
        self.zs = z_coarse #index of the starts of the tomography bins
        self.C = C #cosmopie
        self.rs = np.zeros(self.zs.size) #comoving distances associated with the zs 
        for i in xrange(0,self.zs.size):
            self.rs[i] = self.C.D_comov(self.zs[i])
        self.z_fine = z_fine
        self.r_fine = np.zeros(self.z_fine.size)
        for i in xrange(0,self.r_fine.size):
            self.r_fine[i] = self.C.D_comov(self.z_fine[i])

        tot_area = self.angular_area() 
        self.volumes = np.zeros(self.zs.size-1) #volume of each tomography bin
        for i in xrange(0,self.volumes.size):
            self.volumes[i] += (self.rs[i+1]**3-self.rs[i]**3)/3.*tot_area
        self.v_total = np.sum(self.volumes) #total volume of geo

        #list r and  z bins as [rmin,rmax] pairs (min in bin, max in bin) for convenience
        self.rbins = np.zeros((self.rs.size-1,2)) 
        self.zbins = np.zeros((self.zs.size-1,2))
        for i in xrange(0,self.rs.size-1):
            self.rbins[i,:] = self.rs[i:i+2]
            self.zbins[i,:] = self.zs[i:i+2]
        #TODO go to 0 more elegantly maybe
        self.rbins_fine = np.zeros((self.r_fine.size,2)) 
        self.zbins_fine = np.zeros((self.z_fine.size,2))
        self.rbins_fine[0,:] = np.array([0.,self.r_fine[0]])
        self.zbins_fine[0,:] = np.array([0.,self.z_fine[0]])
        for i in xrange(0,self.r_fine.size-1):
            self.rbins_fine[i+1,:] = self.r_fine[i:i+2]
            self.zbins_fine[i+1,:] = self.z_fine[i:i+2]

        #create list of indices of coarse bin starts in fine grid
        #TODO check handling bin edges correctly
        self.fine_indices = np.zeros((self.zs.size-1,2),dtype=np.int)    
        self.fine_indices[0,0] = 0
        for i in xrange(1,self.zs.size-1):
            self.fine_indices[i-1,1] = np.argmax(self.z_fine>=self.zs[i])
            self.fine_indices[i,0] = self.fine_indices[i-1,1]
        self.fine_indices[-1,1] = self.z_fine.size

        self.dzdr = InterpolatedUnivariateSpline(self.r_fine,self.z_fine).derivative()(self.r_fine)
        #smalles possible difference for a sum
        self.eps=np.finfo(float).eps
        
        #for caching a_lm
        self.alm_table = {}
    #integrate a function over the surface of the geo (interpretation of the meaning of the "surface" is up to the subclass)
    #mainly used for calculating area and a_lm at the moment
    #TODO: consider making specific to a tomography bin (or just let subclasses do that?)
    def surface_integral(self,function):
        raise NotImplementedError, "Subclasses of geo should implement surface_integral"

    #get the angular area (in square radians) occupied by the geometry using the surface integral TODO: consider option to return in degrees or fsky
    def angular_area(self):
        return self.surface_integral(lambda theta,phi: 1.0)

    # returns \int d theta d phi \sin(theta) Y_lm(theta, phi) (the spherical harmonic a_lm for the area)
    # l and m are the indices for the spherical harmonics 
    #automatically cache alm. Don't explicitly memoize because other geos will precompute alm_table
    def a_lm(self,l,m):
        alm = self.alm_table.get((l,m))
        if alm is None:
            def integrand(phi,theta):
                return Y_r(l,m,theta,phi)
            alm = self.surface_integral(integrand)
            self.alm_table[(l,m)] = alm

        return alm

    def get_alm_array(self,l_max):
        ls = np.zeros((l_max+1)**2,dtype=int)
        ms = np.zeros((1+l_max)**2,dtype=int)    
        alms = np.zeros((1+l_max)**2)    
        itr = 0
        for ll in xrange(0,l_max+1):
            for mm in xrange(-ll,ll+1):
                ls[itr] = ll
                ms[itr] = mm
                alms[itr] = self.a_lm(ll,mm)
                itr+=1
        return alms,ls,ms
        

    #volume ddeltabar_dalpha

#class implementing a geometry of rectangles (on the surface of a sphere, defined by theta,phi coordinates of vertices)
class rect_geo(geo): 
    def __init__(self,zs,Theta,Phi,C,z_fine):
            self.Theta = Theta
            self.Phi = Phi
            
            phi1,phi2=self.Phi
            theta1,theta2=self.Theta

            
            #volumes = np.zeros(zs.size-1)
            #for i in xrange(0,volumes.size):
            #    volumes[i] = (phi2-phi1)*(np.cos(theta1)- np.cos(theta2))*(rs[i+1]**3-rs[i]**3)/3.

            #v_total=(phi2-phi1)*(np.cos(theta1)- np.cos(theta2))*(rs[-1]**3-rs[0]**3)/3.


            geo.__init__(self,zs,C,z_fine)

    #function(phi,theta)
    def surface_integral(self,function):
        def integrand(phi,theta):
            return function(phi,theta)*np.sin(theta)
        I=dblquad(integrand,self.Theta[0],self.Theta[1], lambda phi: self.Phi[0], lambda phi: self.Phi[1])[0]
        #if (np.absolute(I) <= self.eps):
        #    return 0.0
        #else:
        return I

#same pixels at every redshift.
class pixel_geo(geo):
    def __init__(self,zs,pixels,C,z_fine):
        #pixel format np.array([(theta,phi,area)])
        #area should be in steradians for now
        self.pixels = pixels
    
        geo.__init__(self,zs,C,z_fine)
        
    #TODO consider vectorizing sum
    def surface_integral(self,function):
        total = 0.
        for i in xrange(0,self.pixels.shape[0]):
            total+=function(self.pixels[i,0],self.pixels[i,1])*self.pixels[i,2] #f(theta,phi)*A
        return total

    #vectorized a_lm computation relies on vector Y_r
    def a_lm(self,l,m):
        alm = self.alm_table.get((l,m))
        if alm is None:
            alm = np.sum(Y_r(l,m,self.pixels[:,0],self.pixels[:,1])*self.pixels[:,2])
            self.alm_table[(l,m)] = alm
        return alm
            
        
        #TODO: Implement polygon_geo, allowing arbitrary polygons, using stokes theorem method
