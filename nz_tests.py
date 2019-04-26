"""test nz modules"""
from __future__ import absolute_import,print_function,division
from builtins import range
import numpy as np
import pytest

import defaults
import hmf
from cosmopie import CosmoPie
from full_sky_geo import FullSkyGeo
from nz_lsst import NZLSST
from nz_wfirst import NZWFirst
import matter_power_spectrum as mps
from nz_candel import NZCandel
from nz_wfirst_eff import NZWFirstEff

def test_hmf_consistency():
    """test consistency of m_cut and n_avg"""
    power_params = defaults.power_params.copy()
    power_params.camb['maxkh'] = 5000.
    power_params.camb['kmax'] = 100.
    power_params.camb['npoints'] = 1000
    C = CosmoPie(defaults.cosmology.copy(),'jdem')
    P = mps.MatterPower(C,power_params)
    C.set_power(P)

    mf_params = defaults.hmf_params.copy()
    mf_params['log10_min_mass'] = 4
    mf_params['n_grid'] = 5000
    mf = hmf.ST_hmf(C,mf_params)

    zs = np.arange(0.2,1.21,1.)
    z_fine = np.linspace(0.0,3.,4000)
    geo1 = FullSkyGeo(zs,C,z_fine)


    nz_matchers = np.zeros(4,dtype=object)
    nz_lsst_params = defaults.nz_params_lsst.copy()
    nz_matchers[0] = NZLSST(z_fine,nz_lsst_params)
    nz_wfirst_params = defaults.nz_params_wfirst_gal.copy()
    nz_wfirst_params['z_cut'] = 0.2
    nz_wfirst_params['smooth_sigma'] = 0.03
    nz_matchers[1] = NZWFirst(nz_wfirst_params)
    nz_matchers[2] = NZWFirstEff(nz_wfirst_params)
    nz_matchers[3] = NZCandel(nz_wfirst_params)

    for itr in range(0,nz_matchers.size):
        m_cuts = nz_matchers[itr].get_M_cut(mf,geo1)
        nz_got = nz_matchers[itr].get_nz(geo1)
        nz_hmf = mf.n_avg(m_cuts,z_fine)
        assert np.allclose(nz_got,nz_hmf/C.h**3)

if __name__=='__main__':
    pytest.cmdline.main(['nz_tests.py'])

#    do_test1 = False
#    if do_test1:
#        C = CosmoPie(defaults.cosmology.copy(),'jdem')
#        power_params = defaults.power_params.copy()
#        power_params.camb['maxkh'] = 100.
#        power_params.camb['kmax'] = 100.
#        power_params.camb['npoints'] = 1000
#        P = mps.MatterPower(C,power_params)
#        C.set_power(P)
#        theta0 = 0.*np.pi/16.
#        theta1 = 15.*np.pi/16.
#        phi0 = 0.
#        phi1 = np.pi/3.
#
#        theta1s = np.array([theta0,theta1,theta1,theta0,theta0])
#        phi1s = np.array([phi0,phi0,phi1,phi1,phi0])
#        theta_in1 = np.pi/2.
#        phi_in1 = np.pi/12.
#        res_choose = 6
#
#        zs = np.array([.001,1.01])
#        z_fine = np.arange(0.0,3.9,0.001)
#
#        l_max = 25
#        geo1 = PolygonPixelGeo(zs,theta1s,phi1s,theta_in1,phi_in1,C,z_fine,l_max,res_choose)
#        n_run = 1
#        mf_params = defaults.hmf_params.copy()
#        mf_params['n_grid'] = 5000
#        mf_params['log10_min_mass'] = 10
#        nz_params = defaults.nz_params_wfirst_gal.copy()
#        nz_params_candel = nz_params.copy()
#        nz_params['data_source'] = 'data/H-5x140s.dat'
#        nz_params['area_sterad'] =  0.040965*np.pi**2/180**2
#        nz_params['smooth_sigma'] = 0.1
#        nz_params['n_right_extend'] = 16
#        nz_params_candel['smooth_sigma'] = 0.02
#        nz_params_candel['n_right_extend'] = 8
#        nz_params_candel['i_cut'] = 25.3
#        #nz_params['mirror_boundary'] = False
#        nz_lsst_params = defaults.nz_params_lsst.copy()
#        nz_lsst_params['i_cut'] = 24.1
#        ts = np.zeros(n_run+1)
#        ts[0] = time()
#
#        for i in range(0,n_run):
#            nzc = NZWFirst(nz_params)
#            mf = hmf.ST_hmf(C,mf_params)
#            t1 = time()
#            dN_dz_res = nzc.get_dN_dzdOmega(z_fine)
#            t2 = time()
#            d_wfirst = trapz2(dN_dz_res,z_fine)
#            d_wfirst_deg = d_wfirst*np.pi**2/180**2
#            print("wfirst tot g/sr: "+str(d_wfirst)+" g/2200 deg^2 = "+str(d_wfirst_deg*2200)+" g/arcmin^2="+str(d_wfirst_deg/3600.))
#            print("found in: "+str(t2-t1)+" s")
#            nz = nzc.get_nz(geo1)
#            t3 = time()
#            print("nz found in: "+str(t3-t2)+" s")
#            m_cuts = nzc.get_M_cut(mf,geo1)
#            t4 = time()
#            print("m cuts found in: "+str(t4-t3)+" s")
#            m_floor = m_cuts!=np.min(mf.mass_grid)
#            m_restrict = m_cuts[m_floor]
#            n_halo = np.zeros(m_restrict.size)
#            n_halo = mf.n_avg(m_restrict,z_fine[m_floor])/C.h**3
#            diff_halo_z = np.abs(n_halo-nz[m_floor])
#
#            print("avg abs reconstruction error: "+str(np.average(diff_halo_z/nz[m_floor])))
#            print("max abs reconstruction error: "+str(np.max(diff_halo_z/nz[m_floor])))
#            ts[i+1] = time()
#        #tf = time()
#        #print("avg tot time: "+str((ts[-1]-ts[0])/n_run)+" s")
#        print("avg tot time: "+str(np.average(np.diff(ts)))+" s")
#        print("std dev tot time: "+str(np.std(np.diff(ts)))+" s")
#        nz2 = NZCandel(nz_params_candel)
#        nz_wfirst_eff = NZWFirstEff(nz_params)
#        nz_lsst = NZLSST(nzc.z_grid,nz_lsst_params)
#        dN_dz_lsst = nz_lsst.get_dN_dzdOmega(z_fine)
#        dN_dz_candel = nz2.get_dN_dzdOmega(z_fine)
#        dN_dz_wfirst_eff = nz_wfirst_eff.get_dN_dzdOmega(z_fine)
#        m_cuts_lsst = nz_lsst.get_M_cut(mf,geo1)
#        d_lsst = trapz2(dN_dz_lsst,z_fine)
#        d_lsst_deg = d_lsst*np.pi**2/180**2
#        d_wfirst_eff = trapz2(dN_dz_wfirst_eff,z_fine)
#        d_wfirst_eff_deg = d_wfirst_eff*np.pi**2/180**2
#        d_cand = trapz2(dN_dz_candel,z_fine)
#        d_cand_deg = d_cand*np.pi**2/180**2
#        print("lsst tot g/sr: "+str(d_lsst)+" g/20000 deg^2 = "+str(d_lsst_deg*20000)+" g/arcmin^2="+str(d_lsst_deg/3600.))
#        print("cand tot g/sr: "+str(d_cand)+" g/20000 deg^2 = "+str(d_cand_deg*20000)+" g/arcmin^2="+str(d_cand_deg/3600.))
#        print("nz wfirst eff tot g/sr: "+str(d_wfirst_eff)+" g/20000 deg^2 = "+str(d_wfirst_eff_deg*20000)+" g/arcmin^2="+str(d_wfirst_eff_deg/3600.))
#        do_plot = True
#        if do_plot:
#            import matplotlib.pyplot as plt
#            ax = plt.subplot(111)
#            #ax.loglog(z_fine,m_cuts)
#            #ax.loglog(z_fine,6675415160366.4219*z_fine**2.3941494934544996)
#            plt.plot(z_fine,dN_dz_res)
#            #i_cut_use = 25.3
#            #z0 = 0.0417*i_cut_use-0.744
#            #ps_lsst = 1./(2.*z0)*(z_fine/z0)**2.*np.exp(-z_fine/z0)
#            #n_per_rad_lsst = 46*10.**(0.31*(i_cut_use-25))*3600.*180.**2/np.pi**2
#            #ns_lsst = n_per_rad_lsst*ps_lsst
#            #plt.plot(z_fine,ns_lsst)
#            plt.plot(z_fine,dN_dz_lsst*d_wfirst/d_lsst)
#            plt.plot(z_fine,dN_dz_candel*d_wfirst/d_cand)
#            plt.plot(z_fine,dN_dz_wfirst_eff*d_wfirst/d_wfirst_eff)
#            plt.xlabel('z')
#            plt.ylabel('dN/dz(z)')
#            plt.legend(['wfirst1','lsst','candel','wfirst_eff'])
#            plt.show()
#
#        #from scipy.ndimage import gaussian_filter1d
#        #dN_smooth = gaussian_filter1d(dN_dz_candel,0.001,truncate=5.)
