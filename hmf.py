'''
    Halo mass function classes
'''
from time import time
from scipy.interpolate import interp1d,RectBivariateSpline,InterpolatedUnivariateSpline
from scipy.integrate import cumtrapz
import numpy as np
from algebra_utils import trapz2
DEBUG = True

#NOTE if useful could add arXiv:astro-ph/9907024 functionality
class ST_hmf(object):
    """Sheth Tormen Halo Mass Function"""
    def __init__(self,C,params,delta_bar=None):
        """ C: a CosmoPie
            delta_bar: mean density fluctuation
            params:
                log10_min_mass,log10_max_mass: log10(minimum/maximum mass to go to)
                n_grid: number of grid points for mass
        """
        self.A = 0.3222
        self.a = 0.707
        self.p = 0.3

        self.params = params
        # log 10 of the minimum and maximum halo mass
        self.min = params['log10_min_mass']
        self.max = params['log10_max_mass']
        # number of grid points in M
        n_grid = self.params['n_grid']
        # I add an additional point because I will take derivatives latter
        # and the finite difference scheme misses that last grid point.
        self.dlog_M = (self.max-self.min)/float(n_grid)
        self.M_grid = 10**np.linspace(self.min,self.max+self.dlog_M, n_grid+1)
        self.mass_grid = self.M_grid[0:-1]


        self.nu_array = np.zeros(n_grid+1)
        self.sigma = np.zeros(n_grid+1)
        self.C = C

        #self.delta_c=C.delta_c(0)
        #TODO consider having option to overwride delta_c
        self.delta_c = 1.686
        if delta_bar is not None:
            self.delta_c = self.delta_c - delta_bar
        self.rho_bar = C.rho_bar(0)

        self.Growth = C.G_norm


        #print 'rhos', 0,self.rho_bar/(1e10), C.rho_crit(0)/(1e10)

        self.R = (3./4.*self.M_grid/self.rho_bar/np.pi)**(1./3.)
        #manually fixed h factor to agree with convention
        self.sigma = C.sigma_r(0.,self.R/C.h)

        # calculated at z=0
        self.nu_array = (self.delta_c/self.sigma)**2
        sigma_inv = self.sigma**(-1)
        dsigma_dM = np.diff(np.log(sigma_inv))/(np.diff(self.M_grid))

        self.sigma_of_M = interp1d(self.M_grid[:-1],self.sigma[:-1])
        self.nu_of_M = interp1d(self.M_grid[:-1], self.nu_array[:-1])
        self.M_of_nu = interp1d(self.nu_array[:-1],self.M_grid[:-1])
        self.dsigma_dM_of_M = interp1d(self.M_grid[:-1],dsigma_dM)

        #grid for precomputing z dependent quantities
        self.z_grid = np.linspace(self.params['z_min'],self.params['z_max'],self.params['n_z'])
        self.G_grid = C.G_norm(self.z_grid)

        #TODO why does this change the code's overall behavior?
        #TODO remove this logic if nothing needs
        self.b_norm_overwride = self.params['b_norm_overwride']
        if self.b_norm_overwride:
            self.b_norm_grid = self.bias_norm(self.G_grid)
            self.b_norm_cache = InterpolatedUnivariateSpline(self.G_grid[::-1],self.b_norm_grid[::-1],k=3,ext=2)

        self.f_norm_overwride = self.params['f_norm_overwride']
        if self.f_norm_overwride:
            self.f_norm_grid = self.f_norm(self.G_grid)
            self.f_norm_cache = InterpolatedUnivariateSpline(self.G_grid[::-1],self.f_norm_grid[::-1],k=3,ext=2)

    def f_norm(self,G):
        """ get normalization factor to ensure all mass is in a halo
            accepts either a numpy array or a scalar input for G"""
        if isinstance(G,np.ndarray):
            nu = np.outer(self.nu_array,1./G**2)
            sigma_inv = np.outer(1./self.sigma,1./G**2)
        else:
            nu = self.nu_array/G**2
            sigma_inv = 1./self.sigma*1./G**2
        f = self.A*np.sqrt(2*self.a/np.pi)*(1. + (1./self.a/nu)**self.p)*np.sqrt(nu)*np.exp(-self.a*nu/2.)
        norm = np.trapz(f,np.log(sigma_inv),axis=0)
        return norm

    def bias_norm(self,G):
        """ gets the normalization for bias b(r) for Sheth-Tormen mass function
            supports numpy array or scalar for G"""
        if isinstance(G,np.ndarray):
            nu = np.outer(self.nu_array,1./G**2)
        else:
            nu = self.nu_array/G**2

        bias = self.bias_nu(nu)
        f = self.f_nu(nu)
        #TODO can fix to use trapz2
        norm = np.trapz(f*bias,nu,axis=0)
        return norm

    def bias_nu(self,nu):
        """return eulerian bias as a function of a numpy matrix nu"""
        return 1. + (self.a*nu-1.)/self.delta_c + 2.*self.p/(self.delta_c*(1.+(self.a*nu)**self.p))

    def f_nu(self,nu):
        """return f as a function of a numpy matrix nu"""
        return self.A*np.sqrt(2.*self.a/np.pi)*(1. + (1./self.a/nu)**self.p)*np.sqrt(nu)*np.exp(-self.a*nu/2.)


    def f_sigma(self,M,G,norm_in=None):
        """ gets Sheth-Tormen mass function normalized so all mass is in a halo
            implements equation 7 in arXiv:astro-ph/0005260
            both M and G can be numpy arrays
            renormalizing for input mass range is optional"""
        if norm_in is None and self.f_norm_overwride:
            norm = self.f_norm_cache(G)
        elif norm_in is not None:
            norm = norm_in 
        else:
            norm = 1.

        #note  nu = delta_c**2/sigma**2
        if isinstance(G,np.ndarray) and isinstance(M,np.ndarray):
            nu = np.outer(self.nu_of_M(M),1./G**2)
        else:
            nu = self.nu_of_M(M)/G**2

        f = self.f_nu(nu)
        if self.f_norm_overwride or norm_in is not None:
            return f/norm
        else:
            return f

    def mass_func(self,M,G):
        """ get Sheth-Tormen mass function
            implements equation 3 in   arXiv:astro-ph/0607150
            supports M and G to be numpy arrays or scalars"""
        f = self.f_sigma(M,G)
        sigma = self.sigma_of_M(M)
        dsigma_dM = self.dsigma_dM_of_M(M)
        if isinstance(G,np.ndarray) and isinstance(M,np.ndarray):
            mf = (self.rho_bar/M*dsigma_dM*f.T).T
        else:
            mf = self.rho_bar/M*dsigma_dM*f
        # return sigma,f,and the mass function dn/dM
        return sigma,f,mf


    def dndM_G(self,M,G):
        """ gets Sheth-Tormen dn/dM as a function of G
            both M and G can be numpy arrays or scalars"""
        _,_,mf = self.mass_func(M,G)
        return mf

    def dndM(self,M,z):
        """ gets Sheth-Tormen dn/dM as a function of z
            both M and z can be numpy arrays or scalars"""
        G = self.Growth(z)
        _,_,mf = self.mass_func(M,G)
        return mf

    def M_star(self):
        """get M(nu=1.0)"""
        return self.M_of_nu(1.0)

    def bias(self,M,z,norm_in=None):
        """get Sheth-Tormen bias as a function of z"""
        G = self.Growth(z)
        return self.bias_G(M,G,norm_in=norm_in)

    def bias_G(self,M,G,norm_in=None):
        """get bias as a function of G
        implements Sheth-Tormen 1999 arXiv:astro-ph/9901122 eq 12"""
        if norm_in is None and self.b_norm_overwride:
            norm = self.b_norm_cache(G)
        elif norm_in is not None: 
            norm = norm_in
        else:
            norm = 1.

        if isinstance(M,np.ndarray) and isinstance(G,np.ndarray):
            nu = np.outer(self.nu_of_M(M),1./G**2)
        else:
            nu = self.nu_of_M(M)/G**2
        bias = self.bias_nu(nu)
        if self.b_norm_overwride or norm_in is not None:
            result = bias/norm
        else:
            result = bias
            
        if DEBUG:
            assert np.all(result>=0)
        return result

    def bias_n_avg(self,min_mass,z):
        """ get average b(z)n(z) for Sheth Tormen mass function
            min_mass can be a function of z or a constant
            if min_mass is a function of z it should be an array of the same size as z
            z can be scalar or an array"""
        if isinstance(z,np.ndarray):
            if isinstance(min_mass,np.ndarray) and not min_mass.size==z.size:
                raise ValueError('min_mass and z must be the same size')

        if np.any(min_mass<self.mass_grid[0]):
            raise ValueError('specified minimum mass too low, increase log10_min_mass')

        G = self.Growth(z)
        if self.b_norm_overwride:
            norm = self.b_norm_cache(G)
        else:
            norm = None

        if isinstance(min_mass,np.ndarray) and isinstance(z,np.ndarray):
            result = np.zeros(min_mass.size)
            for i in xrange(0,min_mass.size):
                mass = _mass_cut(self.mass_grid,min_mass[i])
                mf_i = self.dndM_G(mass,G[i])
                b_i = self.bias_G(mass,G[i],norm)
                result[i] = trapz2(mf_i*b_i,mass)
        else:
            if isinstance(min_mass,np.ndarray):
                result = np.zeros(min_mass.size)
                for i in xrange(0,min_mass.size):
                    mass = _mass_cut(self.mass_grid,min_mass[i])
                    mf = self.dndM_G(mass,G)
                    b_array = self.bias_G(mass,G,norm_in=norm)
                    result[i] = trapz2(b_array*mf,mass)
            else:
                mass = _mass_cut(self.mass_grid,min_mass)
                mf = self.dndM_G(mass,G)
                b_array = self.bias_G(mass,G,norm_in=norm)
                result = trapz2(b_array*mf,mass)

        if DEBUG: 
            assert np.all(result>=0.)

        return result

    #TODO PRIORTIY check for factor h**3
    def n_avg(self,min_mass,z):
        """ M is lower cutoff mass
            min_mass can be a function of z or a constant
            if min_mass is a function of z it should be an array of the same size as z
            z can be scalar or an array"""
        if isinstance(z,np.ndarray):
            if isinstance(min_mass,np.ndarray) and not min_mass.size==z.size:
                raise ValueError('min_mass and z must be the same size')

        if np.any(min_mass<self.mass_grid[0]):
            raise ValueError('specified minimum mass too low, increase log10_min_mass')

        G = self.Growth(z)

        if isinstance(min_mass,np.ndarray) and isinstance(z,np.ndarray):
            result = np.zeros(min_mass.size)
            for i in xrange(0,min_mass.size):
                mass = _mass_cut(self.mass_grid,min_mass[i])
                mf_i = self.dndM_G(mass,G[i])
                result[i] = trapz2(mf_i,mass)
        else:
            if isinstance(min_mass,np.ndarray):
                result = np.zeros(min_mass.size)
                for i in xrange(0,min_mass.size):
                    mass = _mass_cut(self.mass_grid,min_mass[i]) 
                    mf = self.dndM_G(mass,G)
                    result[i] = trapz2(mf,mass)
            else:
                mass = _mass_cut(self.mass_grid,min_mass)
                mf = self.dndM_G(mass,G)
                result = trapz2(mf,mass)

        if DEBUG:
            assert np.all(result>=0.)
        return result

#    def sigma_m(self,mass,z):
#        """ RMS power on a scale of R(mass)
#         rho = mass/volume=mass"""
#        R = 3/4.*mass/self.rho_bar(z)/np.pi
#        R = R**(1/3.)
#        return self.C.sigma_r(z,R)

    def delta_c_z(self,z):
        """ critical threshold for spherical collapse, as given
        in the appendix of NFW 1997"""
        A = 0.15*(12.*np.pi)**(2/3.)

        if (self.C.Omegam==1) and (self.C.OmegaL==0):
            d_crit = A
        elif (self.C.Omegam < 1) and (self.C.OmegaL==0):
            d_crit = A*self.Omegam**(0.0185)
        elif (self.C.Omegam + self.C.OmegaL)==1.0:
            d_crit = A*self.C.Omegam**(0.0055)
        else:
            d_crit = A*self.C.Omegam**(0.0055)
            warn('inexact equality to 1~='+str(self.C.Omegam)+"+"+str(self.C.OmegaL)+"="+str(self.C.OmegaL+self.C.Omegam))
        d_c = d_crit#/self.C.G_norm(z)
        return d_c

def _mass_cut(mass_grid,min_mass):
    """helper function to trim mass ranges to minimum min_mass"""
    itr_cut = np.argmax(mass_grid>=min_mass)
    if itr_cut==0 or mass_grid[itr_cut]==min_mass:
        mass = mass_grid[itr_cut::]
    else:
        mass = np.hstack([min_mass,mass_grid[itr_cut::]])
    return mass
#
#    def delta_v(self,z):
#        """over density for virialized halo"""
#        A = 178.0
#        if (self.C.Omegam_z(z)==1) and (self.C.OmegaL_z(z)==0):
#            d_v = A
#        if (self.C.Omegam_z(z) < 1) and (self.C.OmegaL_z(z)==0):
#            d_v = A/self.C.Omegam_z(z)**(0.7)
#        if (self.C.Omegam_z(z) + self.C.OmegaL_z(z))==1.0:
#            d_v = A/self.C.Omegam_z(z)**(0.55)
#
#        return d_v/self.G_norm(z)
#
#
#    def nu(self,z,mass):
#        """calculates nu=(delta_c/sigma(M))^2
#         delta_c is the overdensity for collapse"""
#        return (self.delta_c(z)/self.sigma_m(mass,z))**2
