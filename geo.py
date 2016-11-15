import numpy as np
import cosmopie as cp
from scipy.integrate import dblquad
from sph_functions import Y_r

#Abstract class defining a geometry for a survey. 
#At the moment, a geometry must 
#1) Have a definite coarse z bin structure (for tomography)
#2) Have a definite fine z bin structure (for integrating over z)
#3) Be able to compute some kind of surface integral
#Most of the behavior should be defined in subclasses

class geo:
    def __init__(self,z_coarse,volumes,v_total,r_coarse,C,z_fine):
        self.zs = z_coarse #index of the starts of the tomography bins
        self.C = C #cosmopie
        self.volumes = volumes #volume of each tomography bin
        self.v_total = v_total #total volume of geo
        self.rs = r_coarse #comoving distances associated with the zs (TODO: just calculate from zs)
        self.z_fine = z_fine
        self.r_fine = np.zeros(self.z_fine.size)
        for i in range(0,self.r_fine.size):
            self.r_fine[i] = self.C.D_comov(self.z_fine[i])
        #list r and  z bins as [rmin,rmax] pairs (min in bin, max in bin) for convenience
        self.rbins = np.zeros((self.rs.size-1,2)) 
        self.zbins = np.zeros((self.zs.size-1,2))
        for i in range(0,self.rs.size-1):
            self.rbins[i,0] = self.rs[i]
            self.rbins[i,1] = self.rs[i+1]
            self.zbins[i,0] = self.zs[i]
            self.zbins[i,1] = self.zs[i+1]
        #smalles possible difference for a sum
        self.eps=np.finfo(float).eps
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
    def a_lm(self,l,m):
        def integrand(phi,theta):
            return np.sin(theta)*Y_r(l,m,theta,phi)
        I2 = self.surface_integral(integrand)
        return I2


    #volume ddeltabar_dalpha

#class implementing a geometry of rectangles (on the surface of a sphere, defined by theta,phi coordinates of vertices)
class rect_geo(geo): 
    def __init__(self,zs,Theta,Phi,C,z_fine):
            self.Theta = Theta
            self.Phi = Phi
            
            phi1,phi2=self.Phi
            theta1,theta2=self.Theta

            rs = np.zeros(zs.size)
            for i in range(0,zs.size):
                rs[i] = C.D_comov(zs[i])
            
            
            volumes = np.zeros(zs.size-1)
            for i in range(0,volumes.size):
                volumes[i] = (phi2-phi1)*(np.cos(theta1)- np.cos(theta2))*(rs[i+1]**3-rs[i]**3)/3.

            v_total=(phi2-phi1)*(np.cos(theta1)- np.cos(theta2))*(rs[-1]**3-rs[0]**3)/3.


            geo.__init__(self,zs,volumes,v_total,rs,C,z_fine)

    #function(phi,theta)
    def surface_integral(self,function):
        I=dblquad(function,self.Theta[0],self.Theta[1], lambda phi: self.Phi[0], lambda phi: self.Phi[1])[0]
        if (np.absolute(I) <= self.eps):
            return 0.0
        else:
            return I

#same pixels at every redshift.
class pixel_geo(geo):
    def __init__(self,zs,pixels,C,z_fine):
        #pixel format np.array([(theta,phi,area)])
        #area should be in steradians for now
        self.pixels = pixels
    

        rs = np.zeros(zs.size)
        for i in range(0,zs.size):
            rs[i] = C.D_comov(zs[i])
           
        volumes = np.zeros(zs.size-1)
        tot_area = np.sum(pixels[:,2])
        for i in range(0,volumes.size):
            volumes[i] += (rs[i+1]**3-rs[i]**3)/3.*tot_area


        v_total = np.sum(volumes)

        geo.__init__(self,zs,volumes,v_total,rs,C,z_fine)
        
    #TODO consider vectorizing
    def surface_integral(self,function):
        total = 0.
        for i in range(0,self.pixels.shape[0]):
            total+=function(self.pixels[i,0],self.pixels[i,1])*self.pixels[i,2] #f(theta,phi)*A
        return total

    #vectorized a_lm computation relies on vector Y_r
    def a_lm(self,l,m):
        return np.sum(np.sin(self.pixels[:,0])*Y_r(l,m,self.pixels[:,0],self.pixels[:,1])*self.pixels[:,2])
            
        
        #TODO: Implement polygon_geo, allowing arbitrary polygons, using either healpix, boundary conditions, or both ways
