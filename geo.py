"""Specificatio for a survey geometry"""
import numpy as np

from scipy.integrate import dblquad
from scipy.interpolate import InterpolatedUnivariateSpline
from sph_functions import Y_r

#Abstract class defining a geometry for a survey.
#At the moment, a geometry must
#1) Have a definite coarse z bin structure (for tomography)
#2) Have a definite fine z bin structure (for integrating over z)
#3) Be able to compute some kind of surface integral
#Most of the behavior should be defined in subclasses

class Geo(object):
    """Abstract class for a survey geometry"""
    def __init__(self,z_coarse,C,z_fine):
        """    inputs:
                z_coarse: tomographic bins for the survey
                C: a CosmoPie object
                z_fine: fine slices (resolution bins) for the survey
        """
        self.zs = z_coarse #index of the starts of the tomography bins
        self.C = C #cosmopie
        #comoving distances associated with the zs
        self.rs = self.C.D_comov(self.zs)
        self.z_fine = z_fine
        self.r_fine = self.C.D_comov(self.z_fine)

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

        self.dzdr = InterpolatedUnivariateSpline(self.r_fine,self.z_fine,ext=2).derivative()(self.r_fine)
        #smalles possible difference for a sum
        self.eps = np.finfo(float).eps

        #for caching a_lm
        self.alm_table = {}

    #TODO: consider making specific to a tomography bin (or just let subclasses do that?)
    def surface_integral(self,function):
        """integrate function over the surface of the geo (interpretation of the meaning of the "surface" is up to the subclass)
        mainly used for calculating area and a_lm at the moment"""
        raise NotImplementedError, "Subclasses of geo should implement surface_integral"

    #TODO: consider option to return in degrees or fsky
    def angular_area(self):
        """get the angular area (in square radians) occupied by the geometry using the surface integral"""
        return self.surface_integral(lambda theta,phi: 1.0)

    #automatically cache alm. Don't explicitly memoize because other geos will precompute alm_table
    def a_lm(self,l,m):
        r""" returns \int d theta d phi \sin(theta) Y_lm(theta, phi) (the spherical harmonic decomposition a_lm for the window function)
                inputs:
                    l,m: indices for the spherical harmonics
        """
        alm = self.alm_table.get((l,m))
        if alm is None:
            def _integrand(phi,theta):
                return Y_r(l,m,theta,phi)
            alm = self.surface_integral(_integrand)
            self.alm_table[(l,m)] = alm

        return alm

    def get_alm_array(self,l_max):
        """get a(l,m) as an array up to l_max"""
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


class RectGeo(Geo):
    """implements a geometry of rectangles on the surface of a sphere, constant latitude and longitude sides"""
    def __init__(self,zs,Theta,Phi,C,z_fine):
        """ inputs:
                zs: the tomographic zs
                Theta,phi: coordinates of the vertices
                z_fine: the resolution z slices
        """
        self.Theta = Theta
        self.Phi = Phi

        Geo.__init__(self,zs,C,z_fine)

    def surface_integral(self,function):
        """do the integral with quadrature over a function(phi,theta)"""
        def _integrand(phi,theta):
            return function(phi,theta)*np.sin(theta)
        I = dblquad(_integrand,self.Theta[0],self.Theta[1], lambda phi: self.Phi[0], lambda phi: self.Phi[1])[0]
        #if (np.absolute(I) <= self.eps):
        #    return 0.0
        #else:
        return I

#same pixels at every redshift.
class PixelGeo(Geo):
    """generic pixelated geometry"""
    def __init__(self,zs,pixels,C,z_fine):
        """pixelated geomtery
            inputs:
                zs: tomographic z bins
                pixels: pixels in format np.array([(theta,phi,area)]), area in steradians
                C: CosmoPie object
                z_fine: the fine z slices
        """
        self.pixels = pixels

        Geo.__init__(self,zs,C,z_fine)

    #TODO consider vectorizing sum
    def surface_integral(self,function):
        """do the surface integral by summing over values at the discrete pixels"""
        total = 0.
        for i in xrange(0,self.pixels.shape[0]):
            total+=function(self.pixels[i,0],self.pixels[i,1])*self.pixels[i,2] #f(theta,phi)*A
        return total

    #TODO should probably just use PolygonPixelGeo method
    def a_lm(self,l,m):
        """vectorized a_lm computation relies on vector Y_r"""
        alm = self.alm_table.get((l,m))
        if alm is None:
            alm = np.sum(Y_r(l,m,self.pixels[:,0],self.pixels[:,1])*self.pixels[:,2])
            self.alm_table[(l,m)] = alm
        return alm
