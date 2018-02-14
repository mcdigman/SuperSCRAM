"""Specification for a survey geometry"""
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
        if tot_area > 0.:#check area is not negative
            for i in xrange(0,self.volumes.size):
                self.volumes[i] += (self.rs[i+1]**3-self.rs[i]**3)/3.*tot_area
        else:
            if tot_area<-1.e-14: #don't raise exception for rounding error just let volume be 0
                raise ValueError('total area '+str(tot_area)+' cannot be negative')
            else:
                tot_area = 0.
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
        #self.eps = np.finfo(float).eps

        #for caching a_lm
        self.alm_table = {}
        self._l_max = -1


        #lock all internals
        self.zs.flags['WRITEABLE'] = False
        self.z_fine.flags['WRITEABLE'] = False
        self.zbins_fine.flags['WRITEABLE'] = False
        self.zbins.flags['WRITEABLE'] = False
        self.rs.flags['WRITEABLE'] = False
        self.r_fine.flags['WRITEABLE'] = False
        self.rbins_fine.flags['WRITEABLE'] = False
        self.rbins.flags['WRITEABLE'] = False
        self.fine_indices.flags['WRITEABLE'] = False
        self.dzdr.flags['WRITEABLE'] = False
        self.volumes.flags['WRITEABLE'] = False


#    def surface_integral(self,function):
#        """integrate function over the surface of the geo (interpretation of the meaning of the "surface" is up to the subclass)
#        mainly used for calculating area and a_lm at the moment"""
#        raise NotImplementedError, "Subclasses of geo should implement surface_integral"

    def angular_area(self):
        """get the angular area (in square radians) occupied by the geometry"""
        raise NotImplementedError('subclasses must implement angular_area')
#        return self.a_lm(0,0)*np.sqrt(4.*np.pi)
        #return self.surface_integral(lambda theta,phi: 1.0)

    #automatically cache alm. Don't explicitly memoize because other geos will precompute alm_table
    def a_lm(self,l,m):
        r""" returns \int d theta d phi \sin(theta) Y_lm(theta, phi) (the spherical harmonic decomposition a_lm for the window function)
                inputs:
                    l,m: indices for the spherical harmonics
        """
        raise NotImplementedError('subclasses must implement a_lm')
#        alm = self.alm_table.get((l,m))
#        if alm is None:
#            def _integrand(phi,theta):
#                return Y_r(l,m,theta,phi)
#            alm = self.surface_integral(_integrand)
#            self.alm_table[(l,m)] = alm

#        return alm

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

    def get_current_l_max(self):
        """get maximum l currently available in table"""
        return self._l_max

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
    def angular_area(self):
        """return angular area"""
        return self.surface_integral(lambda theta,phi: 1.0)

    def surface_integral(self,function):
        """do the integral with quadrature over a function(phi,theta)"""
        def _integrand(phi,theta):
            return function(phi,theta)*np.sin(theta)
        result = dblquad(_integrand,self.Theta[0],self.Theta[1], lambda phi: self.Phi[0], lambda phi: self.Phi[1])[0]
        return result

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
