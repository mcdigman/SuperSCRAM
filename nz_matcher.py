import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
#get dN/(dz dOmega)
class NZMatcher(object):
    def __init__(self,z_grid,dN_dz):
        #self.params = params

        #load data and select nz
        #self.data = np.loadtxt(self.params['data_source'])
        #use separate internal grid for calculations because smoothing means galaxies beyond max z_fine can create edge effects
        #self.z_grid = np.arange(0,np.max(self.data[:,1])+self.params['smooth_sigma']*self.params['n_right_extend'],self.params['z_resolution'])
        #self.chosen = (self.data[:,5]<self.params['i_cut'])
        #cut off faint galaxies
        #self.zs_chosen = self.data[self.chosen,1]
        #print "nz_candel: "+str(self.zs_chosen.size)+" available galaxies"
        self.z_grid = z_grid
        self.dN_dz = dN_dz
        #self.dN_dz = np.zeros(self.z_grid.size)
        #apply gaussian smoothing width sigma
        #sigma = self.params['smooth_sigma']
        #for itr in xrange(0,self.zs_chosen.size):
        #    if self.params['mirror_boundary']:
        #        self.dN_dz += np.exp(-(self.z_grid-self.zs_chosen[itr])**2/(2.*sigma**2))+np.exp(-(self.z_grid+self.zs_chosen[itr])**2/(2.*sigma**2))
        #    else:
        #        self.dN_dz += np.exp(-(self.z_grid-self.zs_chosen[itr])**2/(2.*sigma**2))

        #self.dN_dz = self.dN_dz/(sigma*np.sqrt(2.*np.pi))
        #normalize to correct numerical errors/for galaxies beyond endpoints of grid
        #TODO maybe shouldn't do this
        #result is in galaxy density/steradian
        #self.dN_dz = self.zs_chosen.size*self.dN_dz/(trapz2(self.dN_dz,dx=self.z_grid)*self.params['area_sterad'])
        #self.dN_dz = self.zs_chosen.size*self.dN_dz/(trapz2(self.dN_dz,dx=self.params['z_resolution'],given_dx=True)*self.params['area_sterad'])
        self.dN_dz_interp = interp1d(self.z_grid,self.dN_dz)

    def get_dN_dzdOmega(self,z_fine):
        return self.dN_dz_interp(z_fine)

    #get n(z)
    #TODO careful with which z_fine
    def get_nz(self,geo):
        dN_dz = self.get_dN_dzdOmega(geo.z_fine)
        return 1./geo.r_fine**2*dN_dz*geo.dzdr

    def get_M_cut(self,mf,geo):
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
