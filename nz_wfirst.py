"""implement NZMatcher by matching the number density
in the CANDELS GOODS-S catalogue with  a preapplied cut"""
import numpy as np

import defaults
import hmf
from cosmopie import CosmoPie
from algebra_utils import trapz2
from polygon_pixel_geo import PolygonPixelGeo
from nz_matcher import NZMatcher,get_gaussian_smoothed_dN_dz
from nz_lsst import NZLSST

#TODO: check get_nz and get_M_cut agree
class NZWFirst(NZMatcher):
    """Match n(z) using the CANDELS dataset with pre applied cut"""
    def __init__(self,params):
        """get the CANDELS matcher
            inputs:
                params: a dict of params including:
                    smooth_sigma: a smoothing length scale
                    n_right_extend: number of smoothing scales beyond max z
                    z_resolution: resolution of z grid to use
                    mirror_boundary: True to use mirrored boundary conditions at z=0 for smoothing
                    area_sterad: area of survey in steradians
        """
        self.params = params

        #load data and select nz
        self.data = np.loadtxt(self.params['data_source'])
        #use separate internal grid for calculations because smoothing means galaxies beyond max z_fine can create edge effects
        z_grid = np.arange(0.,np.max(self.data[:,1])+self.params['smooth_sigma']*self.params['n_right_extend'],self.params['z_resolution'])
        #self.chosen = (self.data[:,5]<self.params['i_cut'])
        self.chosen = np.full(self.data[:,0].size,True,dtype=bool)
        #cut off faint galaxies
        self.zs_chosen = self.data[self.chosen,1]
        print "nz_wfirst: "+str(self.zs_chosen.size)+" available galaxies"
        dN_dz = get_gaussian_smoothed_dN_dz(z_grid,self.zs_chosen,params,normalize=True)
        NZMatcher.__init__(self,z_grid,dN_dz)


if __name__ == '__main__':
#def main():
    from time import time
    import matter_power_spectrum as mps
    #d = np.loadtxt('camb_m_pow_l.dat')
    #k = d[:,0]; P = d[:,1]
    C = CosmoPie(cosmology=defaults.cosmology)
    P = mps.MatterPower(C,defaults.power_params)
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
    mf_params = defaults.hmf_params.copy()
    nz_params = defaults.nz_params_wfirst_gal.copy()
    nz_params_candel = nz_params.copy()
    nz_params['data_source'] = 'data/H-5x140s.dat'
    nz_params['area_sterad'] =  0.040965*np.pi**2/180**2
    nz_params['smooth_sigma'] = 0.01
    nz_params['n_right_extend'] = 16
    nz_params_candel['smooth_sigma'] = 0.01
    nz_params_candel['n_right_extend'] = 4
    #nz_params['mirror_boundary'] = False
    nz_lsst_params = defaults.nz_params_lsst.copy()
    nz_lsst_params['i_cut'] = 25.3
    ts = np.zeros(n_run+1)
    ts[0] = time()
    for i in xrange(0,n_run):
        nzc = NZWFirst(nz_params)
        mf = hmf.ST_hmf(C,mf_params)
        t1 = time()
        dN_dz_res = nzc.get_dN_dzdOmega(z_fine)
        t2 = time()
        density_res = trapz2(dN_dz_res,dx=0.01)
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
    from nz_candel import NZCandel
    nz2 = NZCandel(nz_params_candel)
    nz_lsst = NZLSST(nzc.z_grid,nz_lsst_params)
    dN_dz_lsst = nz_lsst.get_dN_dzdOmega(z_fine)
    dN_dz_candel = nz2.get_dN_dzdOmega(z_fine)
    m_cuts_lsst = nz_lsst.get_M_cut(mf,geo1)
    density_res_lsst = trapz2(dN_dz_lsst,dx=0.01)
    print "lsst total galaxies/steradian: "+str(density_res_lsst)+" galaxies/20000 deg^2 = "+str(density_res_lsst*np.pi**2/180**2*20000)+" g/arcmin^2="+str(density_res_lsst*np.pi**2/180**2/3600.)
    do_plot = True
    if do_plot:
        import matplotlib.pyplot as plt
        ax = plt.subplot(111)
        #ax.loglog(z_fine,m_cuts)
        #ax.loglog(z_fine,6675415160366.4219*z_fine**2.3941494934544996)
        plt.plot(z_fine,dN_dz_res)
        #i_cut_use = 25.3
        #z0 = 0.0417*i_cut_use-0.744
        #ps_lsst = 1./(2.*z0)*(z_fine/z0)**2.*np.exp(-z_fine/z0)
        #n_per_rad_lsst = 46*10.**(0.31*(i_cut_use-25))*3600.*180.**2/np.pi**2
        #ns_lsst = n_per_rad_lsst*ps_lsst
        #plt.plot(z_fine,ns_lsst)
        plt.plot(z_fine,dN_dz_lsst)
        plt.plot(z_fine,dN_dz_candel)
        plt.xlabel('z')
        plt.ylabel('dN/dz(z)')
        plt.show()
