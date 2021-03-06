"""module for getting n(z) and the bias b(z) for an input source distribution
by abundance matching a cutoff mass M to the halo mass function"""
from __future__ import division,print_function,absolute_import
from builtins import range

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
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
        #don't divide by 0
        if geo.r_fine[0]==0:
            r_fine = geo.r_fine.copy()
            r_fine[0]+=r_fine[1]*0.0001
        else:
            r_fine = geo.r_fine
        result = 1./r_fine**2*(dN_dz*geo.dzdr)
        return result

    def get_M_cut(self,mf,geo):
        """get the abundance matched mass cutoff from the halo mass function
            inputs:
                mf: a ST_hmf object
                geo: a Geo object
        """
        mass = mf.mass_grid
        nz = self.get_nz(geo)*mf.C.h**3
        m_cuts = np.zeros(geo.z_fine.size)
        #note could do bisection search if not accurate enough
        for itr in range(0,geo.z_fine.size):
            #no number density to match
            if nz[itr]==0.:
                m_cuts[itr] = mass[-1]
                continue
            n_avgs = mf.n_avg(mass,geo.z_fine[itr])
            n_i = np.argmin(n_avgs>=nz[itr])

            if n_i==0:
                print("FLOORED",geo.z_fine[itr],nz[itr],n_avgs[0],n_i)
                m_cuts[itr] = mass[0]
            elif m_cuts[itr]>=mass[-1] or n_i==mass.size-1:
                m_cuts[itr] = mass[-1]
            else:
                log_norm_n = np.log(n_avgs[n_i-1:n_i+1]/n_avgs[0])
                log_norm_nz = np.log(nz[itr]/n_avgs[0])
                m_interp = np.exp(interp1d(log_norm_n,np.log(mass[n_i-1:n_i+1]))(log_norm_nz))
                m_cuts[itr] = m_interp
        return m_cuts

def get_gaussian_smoothed_dN_dz(z_grid,zs_chosen,params,normalize):
    """ apply gaussian smoothing width smooth_sigma to get number density over z_grid
        with galaxies at locations specified by zs_chosen,
        mirror boundary at z=0 if mirror_boundary=True,
        if normalize=True then normalize so density integrates to total number of galaxies
        (in limit as maximum z_grid is much larger than maximum zs_chosen normalizing should have no effect)
        if suppress=True cut off z below z_cut, to avoid numerical issues elsewhere"""
    dN_dz = np.zeros(z_grid.size)
    sigma = params['smooth_sigma']
    delta_dist = np.zeros(z_grid.size)
    for itr in range(0,zs_chosen.size):
        delta_dist[np.argmax(z_grid>=zs_chosen[itr])]+=1.
    #assume z_grid uniform, in case first bin is set to something else to avoid going to 0.
    dz = (z_grid[2]-z_grid[1])
    delta_dist = delta_dist/dz

    if params['mirror_boundary']:
        dN_dz = gaussian_filter1d(delta_dist,sigma/dz,truncate=params['n_right_extend'],mode='mirror')
    else:
        dN_dz = gaussian_filter1d(delta_dist,sigma/dz,truncate=params['n_right_extend'],mode='constant')

    if normalize:
        dN_dz = zs_chosen.size*dN_dz/(au.trapz2(dN_dz,z_grid))
    dN_dz = dN_dz/params['area_sterad']
    return dN_dz
