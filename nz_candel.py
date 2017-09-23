import numpy as np
import scipy

import defaults
import hmf
from cosmopie import CosmoPie
from algebra_utils import trapz2
from polygon_pixel_geo import PolygonPixelGeo
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline
from scipy.integrate import cumtrapz
from nz_matcher import NZMatcher
from nz_lsst import NZLSST
#get dN/(dz dOmega)
class NZCandel(NZMatcher):
    def __init__(self,params):
        self.params = params

        #load data and select nz
        self.data = np.loadtxt(self.params['data_source'])
        #use separate internal grid for calculations because smoothing means galaxies beyond max z_fine can create edge effects
        z_grid = np.arange(0.,np.max(self.data[:,1])+self.params['smooth_sigma']*self.params['n_right_extend'],self.params['z_resolution'])
        self.chosen = (self.data[:,5]<self.params['i_cut'])
        #cut off faint galaxies
        self.zs_chosen = self.data[self.chosen,1]
        print "nz_candel: "+str(self.zs_chosen.size)+" available galaxies"

        dN_dz = np.zeros(z_grid.size)
        #apply gaussian smoothing width sigma
        sigma = self.params['smooth_sigma']
        for itr in xrange(0,self.zs_chosen.size):
            if self.params['mirror_boundary']:
                dN_dz += np.exp(-(z_grid-self.zs_chosen[itr])**2/(2.*sigma**2))+np.exp(-(z_grid+self.zs_chosen[itr])**2/(2.*sigma**2))
            else:
                dN_dz += np.exp(-(z_grid-self.zs_chosen[itr])**2/(2.*sigma**2))

        dN_dz = dN_dz/(sigma*np.sqrt(2.*np.pi))
        #normalize to correct numerical errors/for galaxies beyond endpoints of grid
        #result is in galaxy density/steradian
        #dN_dz = self.zs_chosen.size*dN_dz/(trapz2(dN_dz,dx=z_grid)*self.params['area_sterad'])
        #print trapz2(dN_dz,dx=self.params['z_resolution'],given_dx=True)
        #TODO maybe shouldn't normalize like this
        dN_dz = self.zs_chosen.size*dN_dz/(trapz2(dN_dz,dx=self.params['z_resolution'],given_dx=True)*self.params['area_sterad'])
        #dN_dz = dN_dz/self.params['area_sterad']
        #print trapz2(dN_dz,dx=self.params['z_resolution'],given_dx=True)*self.params['area_sterad']
#        dN_dz_interp = interp1d(z_grid,dN_dz)
        NZMatcher.__init__(self,z_grid,dN_dz)

#    def get_dN_dzdOmega(self,z_fine):
#        return self.dN_dz_interp(z_fine)
#
#    #get n(z)
#    #TODO careful with which z_fine
#    def get_nz(self,geo):
#        dN_dz = self.get_dN_dzdOmega(geo.z_fine)
#        return 1./geo.r_fine**2*dN_dz*geo.dzdr
#
#    def get_M_cut(self,mf,geo):
#        #mass = mf.mass_grid[mf.mass_grid >= M]
#        mass = mf.mass_grid
#        nz = self.get_nz(geo)
#        m_cuts = np.zeros(geo.z_fine.size)
#        #TODO maybe can be faster
#        #TODO check this
#        Gs = mf.Growth(geo.z_fine)
#        dns = mf.dndM_G(mass,Gs)
#        for itr in xrange(0,geo.z_fine.size):
#            dn = dns[:,itr]
#            n_avgs = np.hstack((-(cumtrapz(dn[::-1],mass[::-1]))[::-1],0.))
#            #n_avg_interp = interp1d(n_avgs,mass)
#            #m_cuts[itr] = n_avg_interp(nz[itr])
#            # n_avgs = np.trapz(dn,mass)-(cumtrapz(dn,mass))
#            n_avg_index = np.argmin(n_avgs >= nz[itr]) #TODO check edge cases
#            #print nz[itr]
#            #print n_avgs
#            #TODO only need 1 interpolating function
#            if n_avg_index == 0:
#                m_cuts[itr] = mass[n_avg_index]
#            else:
#                m_interp = interp1d(n_avgs[n_avg_index-1:n_avg_index+1],mass[n_avg_index-1:n_avg_index+1])(nz[itr])
#                #print mass[n_avg_index-1],mass[n_avg_index],m_interp,m_interp/mass[n_avg_index]
#                m_cuts[itr] = m_interp#+(nz[itr]-n_avgs[n_avg_index-1])*(mass[n_avg_index]-mass[n_avg_index-1])/(n_avgs[n_avg_index]-n_avgs[n_avg_index-1])
#        #print mass
#        return m_cuts


#TODO: check get_nz and get_M_cut agree
if __name__ == '__main__':
#def main():   
    from time import time
    import matter_power_spectrum as mps
    #d = np.loadtxt('camb_m_pow_l.dat')
    #k = d[:,0]; P = d[:,1]
    C = CosmoPie(cosmology=defaults.cosmology)
    P = mps.MatterPower(C)
    C.P_lin = P
    C.k = C.P_lin.k
    theta0 = 0.*np.pi/16.
    theta1 = 16.*np.pi/16.
    phi0 = 0.
    phi1 = np.pi/3.

    theta1s = np.array([theta0,theta1,theta1,theta0,theta0])
    phi1s = np.array([phi0,phi0,phi1,phi1,phi0])
    theta_in1 = np.pi/2.
    phi_in1 = np.pi/12.
    res_choose = 6

    zs = np.array([.01,1.01])
    z_fine = np.arange(0.01,4.0,0.01)

    l_max = 25
    geo1 = PolygonPixelGeo(zs,theta1s,phi1s,theta_in1,phi_in1,C,z_fine,l_max,res_healpix = res_choose)
    n_run = 1
    nz_params = defaults.nz_params_wfirst.copy()
    #nz_params['mirror_boundary'] = False
    nz_lsst_params = defaults.nz_params_lsst.copy()
    nz_lsst_params['i_cut'] = 25.3
    ts = np.zeros(n_run+1)
    ts[0] = time()
    for i in xrange(0,n_run):
        nzc = NZCandel(nz_params)
        mf = hmf.ST_hmf(C)
        t1 = time()
        dN_dz = nzc.get_dN_dzdOmega(z_fine)
        t2 = time()
        density_res = trapz2(dN_dz,dx=0.01,given_dx=True)
        print "wfirst total galaxies/steradian: "+str(density_res)+" galaxies/2200 deg^2 = "+str(density_res*np.pi**2/180**2*2200)+" g/arcmin^2="+str(density_res*np.pi**2/180**2/3600.)
        print "found in: "+str(t2-t1)+" s"
        nz = nzc.get_nz(geo1)
        t3 = time()
        print "nz found in: "+str(t3-t2)+" s"
        m_cuts = nzc.get_M_cut(mf,geo1)
        t4 = time()
        print "m cuts found in: "+str(t4-t3)+" s"
        n_halo = np.zeros(m_cuts.size)
        n_halo = mf.n_avg(m_cuts,z_fine)
        print "avg reconstruction error: "+str(np.average((n_halo-nz)/nz))
        ts[i+1] = time()
    #tf = time()
    #print "avg tot time: "+str((ts[-1]-ts[0])/n_run)+" s"
    print "avg tot time: "+str(np.average(np.diff(ts)))+" s"
    print "std dev tot time: "+str(np.std(np.diff(ts)))+" s"
    nz_lsst = NZLSST(nzc.z_grid,nz_lsst_params)
    dN_dz_lsst = nz_lsst.get_dN_dzdOmega(z_fine)
    m_cuts_lsst = nz_lsst.get_M_cut(mf,geo1)
    density_res_lsst = trapz2(dN_dz_lsst,dx=0.01,given_dx=True)
    print "lsst total galaxies/steradian: "+str(density_res_lsst)+" galaxies/20000 deg^2 = "+str(density_res_lsst*np.pi**2/180**2*20000)+" g/arcmin^2="+str(density_res_lsst*np.pi**2/180**2/3600.)
    do_plot =True
    if do_plot:
        import matplotlib.pyplot as plt
        ax = plt.subplot(111)
        #ax.loglog(z_fine,m_cuts)
        #ax.loglog(z_fine,6675415160366.4219*z_fine**2.3941494934544996)
        plt.plot(z_fine,dN_dz)
        #i_cut_use = 25.3
        #z0 = 0.0417*i_cut_use-0.744
        #ps_lsst = 1./(2.*z0)*(z_fine/z0)**2.*np.exp(-z_fine/z0)
        #n_per_rad_lsst = 46*10.**(0.31*(i_cut_use-25))*3600.*180.**2/np.pi**2
        #ns_lsst = n_per_rad_lsst*ps_lsst
        #plt.plot(z_fine,ns_lsst)
        plt.plot(z_fine,dN_dz_lsst)
        plt.xlabel('z')
        plt.ylabel('dN/dz(z)')
        plt.show()
