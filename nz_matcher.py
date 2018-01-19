"""module for getting n(z) and the bias b(z) for an input source distribution
by abundance matching a cutoff mass M to the halo mass function"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import algebra_utils as au

class NZMatcher(object):
    """master class for matching n(z) distributions"""
    def __init__(self,z_grid,dN_dz):
        """make a n(z) matching object
                inputs:
                    z_grid: a numpy array of z
                    dN_dz: the number density of sources
        """
        self.z_grid = z_grid
        self.dN_dz = dN_dz
        self.dN_dz_interp = interp1d(self.z_grid,self.dN_dz)

    def get_dN_dzdOmega(self,z_fine):
        """get the number density of sources on a z grid"""
        return self.dN_dz_interp(z_fine)

    def get_N_projected(self,z_fine,omega_tot):
        """get total number of galaxies in area omega_tot steradians"""
        return au.trapz2(self.dN_dz_interp(z_fine),z_fine)*omega_tot

    def get_nz(self,geo):
        """get n(z) for the input geometry's fine z grid"""
        dN_dz = self.get_dN_dzdOmega(geo.z_fine)
        return 1./geo.r_fine**2*dN_dz*geo.dzdr

    def get_M_cut(self,mf,geo):
        """get the abundance matched mass cutoff from the halo mass function
            inputs:
                mf: a ST_hmf object
                geo: a Geo object
        """
        mass = mf.mass_grid
        nz = self.get_nz(geo)
        m_cuts = np.zeros(geo.z_fine.size)
        #TODO maybe can be faster
        #TODO check this
        Gs = mf.Growth(geo.z_fine)
        dns = mf.dndM_G(mass,Gs)
        for itr in xrange(0,geo.z_fine.size):
            dn = dns[:,itr]
            n_avgs = np.hstack((-(cumtrapz(dn[::-1],mass[::-1]))[::-1],0.))
            n_avg_index = np.argmin(n_avgs >= nz[itr]) #TODO check edge cases
            #TODO only need 1 interpolating function
            if n_avg_index == 0:
                m_cuts[itr] = mass[n_avg_index]
            else:
                m_interp = interp1d(n_avgs[n_avg_index-1:n_avg_index+1],mass[n_avg_index-1:n_avg_index+1])(nz[itr])
                m_cuts[itr] = m_interp
        return m_cuts

def get_gaussian_smoothed_dN_dz(z_grid,zs_chosen,params,normalize):
    """ apply gaussian smoothing width smooth_sigma to get number density over z_grid
        with galaxies at locations specified by zs_chosen,
        mirror boundary at z=0 if mirror_boundary=True,
        if normalize=True then normalize so density integrates to total number of galaxies
        (in limit as maximum z_grid is much larger than maximum zs_chosen normalizing should have no effect)"""
    dN_dz = np.zeros(z_grid.size)
    sigma=params['smooth_sigma']
    for itr in xrange(0,zs_chosen.size):
        if params['mirror_boundary']:
            dN_dz += np.exp(-(z_grid-zs_chosen[itr])**2/(2.*sigma**2))+np.exp(-(z_grid+zs_chosen[itr])**2/(2.*sigma**2))
        else:
            dN_dz += np.exp(-(z_grid-zs_chosen[itr])**2/(2.*sigma**2))

    dN_dz = dN_dz/(sigma*np.sqrt(2.*np.pi))
    if normalize:
        dN_dz = zs_chosen.size*dN_dz/(au.trapz2(dN_dz,dx=params['z_resolution'])*params['area_sterad'])
    return dN_dz
