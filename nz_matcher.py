"""module for getting n(z) and the bias b(z) for an input source distribution
by abundance matching a cutoff mass M to the halo mass function"""

import numpy as np
import algebra_utils as au

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
class NZMatcher(object):
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
        #mass = mf.mass_grid[mf.mass_grid >= M]
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
            #n_avg_interp = interp1d(n_avgs,mass)
            #m_cuts[itr] = n_avg_interp(nz[itr])
            # n_avgs = np.trapz(dn,mass)-(cumtrapz(dn,mass))
            n_avg_index = np.argmin(n_avgs >= nz[itr]) #TODO check edge cases
            #print nz[itr]
            #print n_avgs
            #TODO only need 1 interpolating function
            if n_avg_index == 0:
                m_cuts[itr] = mass[n_avg_index]
            else:
                m_interp = interp1d(n_avgs[n_avg_index-1:n_avg_index+1],mass[n_avg_index-1:n_avg_index+1])(nz[itr])
                #print mass[n_avg_index-1],mass[n_avg_index],m_interp,m_interp/mass[n_avg_index]
                m_cuts[itr] = m_interp#+(nz[itr]-n_avgs[n_avg_index-1])*(mass[n_avg_index]-mass[n_avg_index-1])/(n_avgs[n_avg_index]-n_avgs[n_avg_index-1])
        #print mass
        return m_cuts
