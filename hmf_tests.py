"""Halo mass function tests"""
from __future__ import division,print_function,absolute_import
from builtins import range
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import trapz
import numpy as np
import pytest
from hmf import ST_hmf
import defaults
import cosmopie as cp
import matter_power_spectrum as mps

def test_hmf():
    """run various hmf tests as a block"""
    cosmo_fid = defaults.cosmology.copy()

    cosmo_fid['h'] = 0.65
    cosmo_fid['Omegamh2'] = 0.148
    cosmo_fid['Omegabh2'] = 0.02
    cosmo_fid['OmegaLh2'] = 0.65*0.65**2
    cosmo_fid['sigma8'] = 0.92
    cosmo_fid['ns'] = 1.
    cosmo_fid = cp.add_derived_pars(cosmo_fid,p_space='basic')
    power_params = defaults.power_params.copy()
    power_params.camb['force_sigma8'] = True
    power_params.camb['npoints'] = 1000
    power_params.camb['maxkh'] = 20000.
    power_params.camb['kmax'] = 100.#0.899999976158
    power_params.camb['accuracy'] = 2.
    C = cp.CosmoPie(cosmo_fid,'basic')
    P = mps.MatterPower(C,power_params)
    C.set_power(P)
    params = defaults.hmf_params.copy()
    params['z_min'] = 0.0
    params['z_max'] = 5.0
    params['log10_min_mass'] = 6
    params['log10_max_mass'] = 18.63
    params['n_grid'] = 1264
    params['n_z'] = 5./0.5
    hmf = ST_hmf(C,params=params)
    Ms = hmf.mass_grid

    do_sanity_checks = True
    if do_sanity_checks:
        print("sanity")
        #some sanity checks
        zs = np.arange(0.001,5.,0.1)
        Gs = C.G_norm(zs)

        #arbitrary input M(z) cutoff
        m_z_in = np.exp(np.linspace(np.log(np.min(Ms)),np.log(1.e15),zs.size))
        m_z_in[0] = Ms[0]

        #check normalized to unity (all dark matter is in some halo)


#        f_norm_residual = trapz(hmf.f_sigma(Ms,Gs).T,np.log(hmf.sigma[:-1:]**-1),axis=1)
        #assert np.allclose(np.zeros(Gs.size)+1.,norm_residual)
        _,_,dndM = hmf.mass_func(Ms,Gs)
        n_avgs_alt = np.zeros(zs.size)
        bias_n_avgs_alt = np.zeros(zs.size)
        dndM_G_alt = np.zeros((Ms.size,zs.size))
        for i in range(0,zs.size):
            n_avgs_alt[i] = hmf.n_avg(m_z_in[i],zs[i])
            bias_n_avgs_alt[i] = hmf.bias_n_avg(m_z_in[i],zs[i])
            dndM_G_alt[:,i] = hmf.dndM_G(Ms,Gs[i])
        dndM_G = hmf.dndM_G(Ms,Gs)
        #consistency checks for vector method
        assert np.allclose(dndM_G,dndM_G_alt)
        n_avgs = hmf.n_avg(m_z_in,zs)
        bias_n_avgs = hmf.bias_n_avg(m_z_in,zs)
        assert np.allclose(n_avgs,n_avgs_alt)
        assert np.all(n_avgs>=0.)
        assert np.allclose(bias_n_avgs,bias_n_avgs_alt)
        #check integrating dn/dM over all M actually gives n
        assert np.allclose(trapz(dndM.T,Ms,axis=1),hmf.n_avg(np.zeros(zs.size)+Ms[0],zs))
        test_xs = np.outer(hmf.nu_of_M(Ms),1./Gs**2)
        test_integrand = hmf.f_sigma(Ms,Gs)*hmf.bias_G(Ms,Gs,hmf.bias_norm(Gs))
        #not sure why, but this is true (if ignore h)
        #test_term = np.trapz(test_integrand,test_xs,axis=0)*hmf.f_norm(Gs)
        #assert np.allclose(1.,test_term,rtol=1e-3)
        b_norm_residual = np.trapz(test_integrand,test_xs,axis=0)
        assert np.allclose(np.zeros(zs.size)+1.,b_norm_residual)
        #b_norm_residual_alt = np.trapz(hmf.f_sigma(Ms,Gs)*hmf.bias_G(Ms,Gs),test_xs,axis=0)
        b_norm_residual_alt2 = np.trapz(hmf.f_sigma(Ms,Gs)*hmf.bias_G(Ms,Gs,1.),test_xs,axis=0)
        #check including norm_in behaves appropriately does not matter
        #assert np.allclose(b_norm_residual_alt,b_norm_residual_alt2)
        assert np.allclose(b_norm_residual,b_norm_residual_alt2/hmf.bias_norm(Gs))
        #hcekc all the matter in a halo if include normalization factor
        assert np.allclose(np.trapz(hmf.f_sigma(Ms,1.,1.),np.log(1./hmf.sigma[:-1:])),hmf.f_norm(1.))

        #sanity check M_star
        assert np.round(np.log10(hmf.M_star())) == 13.
        assert np.isclose(C.sigma_r(0.,8./C.h),cosmo_fid['sigma8'],rtol=1.e-3)
        assert np.isclose(C.sigma_r(1.,8./C.h),C.G_norm(1.)*cosmo_fid['sigma8'],rtol=1.e-3)

        #M_restrict = 10**np.linspace(np.log10(4.*10**13),16,100)
        #n_restrict = hmf.n_avg(M_restrict,0.)
        nu = hmf.nu_of_M(Ms)
        bias_nu = hmf.bias_nu(nu)
        bias_G = hmf.bias_G(Ms,1.)
        bias_z = hmf.bias(Ms,0.)
        assert np.allclose(bias_nu,bias_G)
        assert np.allclose(bias_nu,bias_z)
        assert np.allclose(bias_G,bias_z)
        assert np.all(bias_nu>=0)
        dndm = hmf.dndM_G(Ms,1.)
#        bias_avg = np.trapz(bias_nu*dndm,Ms)

        n_avg2 = hmf.n_avg(Ms,0.)
        assert np.all(n_avg2>=0.)
#        bias_n_avg1 = bias_nu*n_avg2
        #bias_n_avg2 = hmf.bias_n_avg(Ms)

#        integ_pred = np.trapz(hmf.f_sigma(Ms,1.,1.),np.log(1./hmf.sigma[:-1:]))
#        integ_res = np.trapz(Ms*hmf.dndM_G(Ms,1.),Ms)/C.rho_bar(0.)
        #assert np.isclose(integ_res,integ_pred,rtol=1.e-2)
        #xs n#np.linspace(np.log(np.sqrt(nu[0])),np.log(nu[-1]),10000)
        #xs = np.exp(np.linspace(np.log(np.sqrt(nu[0]),np.log(1.e36),10000))
        #F = -cumtrapz(1./np.sqrt(2.*np.pi)*np.exp(-xs[::-1]**2/2.),xs[::-1])[::-1]
        #cons_res = np.trapz(hmf.dndM_G(Ms,1.)*Ms*hmf.bias(Ms,1.),Ms)/C.rho_bar(0.)
        assert np.isclose(np.trapz(bias_nu*hmf.f_nu(nu),np.sqrt(nu)),1.,rtol=1.e-1)
        #assert np.isclose(np.trapz(hmf.f_nu(nu)/np.sqrt(nu),np.sqrt(nu)),1.,rtol=2.e-1)

        #bias_avg = np.trapz(hmf.dndM_G(Ms,1.)*hmf.bias_G(Ms,1.),Ms)/np.trapz(hmf.dndM_G(Ms,1.),Ms)
        #check for various edge cases of n_avg and bias_n_avg
        z2s = np.array([0.,1.])
        m2s = np.array([1.e8,1.e9])
        n1 = hmf.n_avg(m2s[0],z2s[0])
        n2 = hmf.n_avg(m2s[0],z2s)
        n3 = hmf.n_avg(m2s,z2s[0])
        n4 = hmf.n_avg(m2s,z2s)
        n5 = hmf.n_avg(m2s[1],z2s[1])
        n6 = hmf.n_avg(m2s[1],z2s[0])
        n7 = hmf.n_avg(m2s[0],z2s[1])
        assert np.isclose(n1,n2[0])
        assert np.isclose(n1,n3[0])
        assert np.isclose(n1,n4[0])
        assert np.isclose(n5,n4[1])
        assert np.isclose(n2[1],n7)
        assert np.isclose(n3[1],n6)
        bn1 = hmf.bias_n_avg(m2s[0],z2s[0])
        bn2 = hmf.bias_n_avg(m2s[0],z2s)
        bn3 = hmf.bias_n_avg(m2s,z2s[0])
        bn4 = hmf.bias_n_avg(m2s,z2s)
        bn5 = hmf.bias_n_avg(m2s[1],z2s[1])
        bn6 = hmf.bias_n_avg(m2s[1],z2s[0])
        bn7 = hmf.bias_n_avg(m2s[0],z2s[1])
        assert np.isclose(bn1,bn2[0])
        assert np.isclose(bn1,bn3[0])
        assert np.isclose(bn1,bn4[0])
        assert np.isclose(bn5,bn4[1])
        assert np.isclose(bn2[1],bn7)
        assert np.isclose(bn3[1],bn6)

        n_avgs_0 = hmf.n_avg(hmf.mass_grid,0.)
        n_avgs_1 = np.zeros(hmf.mass_grid.size)
        for itr in range(0,hmf.mass_grid.size):
            n_avgs_1[itr] = hmf.n_avg(hmf.mass_grid[itr],0.)
        assert np.allclose(n_avgs_0,n_avgs_1)

        bn_avgs_0 = hmf.bias_n_avg(hmf.mass_grid,0.)
        bn_avgs_1 = np.zeros(hmf.mass_grid.size)
        for itr in range(0,hmf.mass_grid.size):
            bn_avgs_1[itr] = hmf.bias_n_avg(hmf.mass_grid[itr],0.)
        assert np.allclose(bn_avgs_0,bn_avgs_1)

        print("PASS: sanity passed")

    do_plot_test2 = True
    if do_plot_test2:
        #Ms = 10**(np.linspace(11,14,500))
       # dndM_G=hmf.dndM_G(Ms,Gs)
        do_jenkins_comp = True
        #should look like dotted line in figure 3 of    arXiv:astro-ph/0005260
        if do_jenkins_comp:
            print("jenkins_comp")
            zs = np.array([0.])
            Gs = C.G_norm(zs)
            dndM_G = hmf.f_sigma(Ms,Gs)
            input_dndm = np.loadtxt('test_inputs/hmf/dig_jenkins_fig3.csv',delimiter=',')
            res_j = np.exp(input_dndm[:,1])
            res_i = InterpolatedUnivariateSpline(1./hmf.sigma[:-1:],dndM_G,k=3,ext=2)(np.exp(input_dndm[:,0]))
            assert np.all(np.abs((res_j/res_i-1.)[0:19])<0.03)
            print("PASS: jenkins_comp")

        do_sheth_bias_comp1 = True
        #agrees pretty well, maybe as well as it should
        #compare to rightmost arXiv:astro-ph/9901122 figure 3
        if do_sheth_bias_comp1:
            print("sheth_bias_comp1")
            cosmo_fid2 = cosmo_fid.copy()
            cosmo_fid2['Omegamh2'] = 0.3*0.7**2
            cosmo_fid2['Omegabh2'] = 0.05*0.7**2
            cosmo_fid2['OmegaLh2'] = 0.7*0.7**2
            cosmo_fid2['sigma8'] = 0.9
            cosmo_fid2['h'] = 0.7
            cosmo_fid2['ns'] = 1.0
            cosmo_fid2 = cp.add_derived_pars(cosmo_fid2,p_space='basic')
            C2 = cp.CosmoPie(cosmo_fid2,'basic')
            P2 = mps.MatterPower(C2,power_params)
            C2.set_power(P2)
            params2 = params.copy()
            hmf2 = ST_hmf(C2,params=params2)
            zs = np.array([0.,1.,2.,4.])
            Gs = C2.G_norm(zs)
            input_bias = np.loadtxt('test_inputs/hmf/dig_sheth_fig3.csv',delimiter=',')
            bias = hmf2.bias_G(10**input_bias[:,0],Gs)
            error = np.abs(1.-10**input_bias[:,1:5]/bias**2)
            error_max = np.array([ 0.06,  0.4,  0.2,  0.4])
            assert np.all(np.max(error,axis=0)<error_max)
            print("PASS: sheth_bias_comp1")

        do_sheth_bias_comp2 = True
        #agrees well
        #compare to bottom arXiv:astro-ph/9901122 figure 4
        if do_sheth_bias_comp2:
            print("sheth_bias_comp2")
            cosmo_fid2 = cosmo_fid.copy()
            cosmo_fid2['Omegamh2'] = 0.3*0.7**2
            cosmo_fid2['Omegabh2'] = 0.05*0.7**2
            cosmo_fid2['OmegaLh2'] = 0.7*0.7**2
            cosmo_fid2['sigma8'] = 0.9
            cosmo_fid2['h'] = 0.7
            cosmo_fid2['ns'] = 1.0
            cosmo_fid2 = cp.add_derived_pars(cosmo_fid2,p_space='basic')
            C2 = cp.CosmoPie(cosmo_fid2,'basic')
            P2 = mps.MatterPower(C2,power_params)
            C2.set_power(P2)
            C2.k = P2.k
            params2 = params.copy()
            hmf2 = ST_hmf(C2,params=params2)
            input_bias = np.loadtxt('test_inputs/hmf/dig_sheth2.csv',delimiter=',')
            bias = hmf2.bias_nu(10**input_bias[:,0])
            observable = 1.+(bias-1.)*hmf2.delta_c
            error = np.abs(1.-10**input_bias[:,1]/observable)
            error_max = 0.07
            assert np.all(np.max(error)<error_max)
            print("PASS: sheth_bias_comp2")


        #should agree, does
        #match fig 2 arXiv:astro-ph/0203169
        do_hu_bias_comp2 = True
        if do_hu_bias_comp2:
            print("hu_bias_comp2")
            zs = np.array([0.])
            Gs = C.G_norm(zs)
            bias = hmf.bias_G(Ms,Gs,1.)[:,0] #why 1
            #maybe should have k pivot 0.01
            input_bias = np.loadtxt('test_inputs/hmf/dig_hu_bias2.csv',delimiter=',')
#            masses = 10**np.linspace(np.log10(10**11),np.log10(10**16),100)
            bias_hu_i = 10**input_bias[:,1]#InterpolatedUnivariateSpline(10**input_bias[:,0],10**input_bias[:,1])(masses)
            bias_i = hmf.bias_G(10**input_bias[:,0],Gs,1.)[:,0]
            assert np.max(np.abs(bias_hu_i/bias_i-1.))<0.06
            print("PASS: hu_bias_comp2 passed")

        #should agree, does
        #match fig 1 arXiv:astro-ph/0203169
        do_hu_sigma_comp = True
        if do_hu_sigma_comp:
            print("hu_sigma_comp")
            assert np.isclose(hmf.M_star(),1.2*10**13,rtol=1.e-1)
            input_sigma = np.loadtxt('test_inputs/hmf/dig_hu_sigma.csv',delimiter=',')
            res_sigma = InterpolatedUnivariateSpline(hmf.R,hmf.sigma,k=3,ext=2)(10**input_sigma[:,0])
            assert np.all(np.abs((res_sigma/10**input_sigma[:,1])[4::]-1.)<0.02)
            print("PASS: hu_sigma_comp")

if __name__=="__main__":
    pytest.cmdline.main(['hmf_tests.py'])
#if __name__=="__main__":
#    cosmo_fid = defaults.cosmology.copy()
#
#    cosmo_fid['h'] = 0.65
#    cosmo_fid['Omegamh2'] = 0.148
#    cosmo_fid['Omegabh2'] = 0.02
#    cosmo_fid['OmegaLh2'] = 0.65*0.65**2
#    cosmo_fid['sigma8'] = 0.92
#    cosmo_fid['ns'] = 1.
#    cosmo_fid = cp.add_derived_pars(cosmo_fid,p_space='basic')
#    power_params = defaults.power_params.copy()
#    power_params.camb['force_sigma8'] = True
#    power_params.camb['npoints'] = 1000
#    power_params.camb['maxkh'] = 20000.
#    power_params.camb['kmax'] = 100.#0.899999976158
#    power_params.camb['accuracy'] = 2.
#    C = cp.CosmoPie(cosmo_fid,'basic')
#    P = mps.MatterPower(C,power_params)
#    C.set_power(P)
#    params = defaults.hmf_params.copy()
#    params['z_min'] = 0.0
#    params['z_max'] = 5.0
#    params['log10_min_mass'] = 6
#    params['log10_max_mass'] = 18.63
#    params['n_grid'] = 1264
#    params['n_z'] = 5./0.5
##
#    hmf = ST_hmf(C,params=params)
#    Ms = hmf.mass_grid
##    dlog_M = 48.63/500.
##    mass_grid = 10**np.linspace(-30.,18.63+dlog_M, 500+1)[:-1:]
#
#    #M=np.logspace(8,15,150)
#
#    import matplotlib.pyplot as plt
#
#    do_sanity_checks = True
#    if do_sanity_checks:
#        print("sanity")
#        #some sanity checks
#        #Gs = np.arange(0.01,1.0,0.01)
#        #zs = np.arange(0.,4.,0.5)
#        zs = np.arange(0.001,5.,0.1)
#        Gs = C.G_norm(zs)
#
#        #arbitrary input M(z) cutoff
#        m_z_in = np.exp(np.linspace(np.log(np.min(Ms)),np.log(1.e15),zs.size))
#        m_z_in[0] = Ms[0]
#
#        #check normalized to unity (all dark matter is in some halo)
#
#
#        f_norm_residual = trapz(hmf.f_sigma(Ms,Gs).T,np.log(hmf.sigma[:-1:]**-1),axis=1)
#        #assert np.allclose(np.zeros(Gs.size)+1.,norm_residual)
#        _,_,dndM = hmf.mass_func(Ms,Gs)
#        n_avgs_alt = np.zeros(zs.size)
#        bias_n_avgs_alt = np.zeros(zs.size)
#        dndM_G_alt = np.zeros((Ms.size,zs.size))
#        for i in range(0,zs.size):
#            n_avgs_alt[i] = hmf.n_avg(m_z_in[i],zs[i])
#            bias_n_avgs_alt[i] = hmf.bias_n_avg(m_z_in[i],zs[i])
#            dndM_G_alt[:,i] = hmf.dndM_G(Ms,Gs[i])
#        dndM_G = hmf.dndM_G(Ms,Gs)
#        #consistency checks for vector method
#        assert np.allclose(dndM_G,dndM_G_alt)
#        n_avgs = hmf.n_avg(m_z_in,zs)
#        bias_n_avgs = hmf.bias_n_avg(m_z_in,zs)
#        assert np.allclose(n_avgs,n_avgs_alt)
#        assert np.all(n_avgs>=0.)
#        assert np.allclose(bias_n_avgs,bias_n_avgs_alt)
#        #check integrating dn/dM over all M actually gives n
#        assert np.allclose(trapz(dndM.T,Ms,axis=1),hmf.n_avg(np.zeros(zs.size)+Ms[0],zs))
#        test_xs = np.outer(hmf.nu_of_M(Ms),1./Gs**2)
#        test_integrand = hmf.f_sigma(Ms,Gs)*hmf.bias_G(Ms,Gs,hmf.bias_norm(Gs))
#        #not sure why, but this is true (if ignore h)
#        #test_term = np.trapz(test_integrand,test_xs,axis=0)*hmf.f_norm(Gs)
#        #assert np.allclose(1.,test_term,rtol=1e-3)
#        b_norm_residual = np.trapz(test_integrand,test_xs,axis=0)
#        assert np.allclose(np.zeros(zs.size)+1.,b_norm_residual)
#        #b_norm_residual_alt = np.trapz(hmf.f_sigma(Ms,Gs)*hmf.bias_G(Ms,Gs),test_xs,axis=0)
#        b_norm_residual_alt2 = np.trapz(hmf.f_sigma(Ms,Gs)*hmf.bias_G(Ms,Gs,1.),test_xs,axis=0)
#        #check including norm_in behaves appropriately does not matter
#        #assert np.allclose(b_norm_residual_alt,b_norm_residual_alt2)
#        assert np.allclose(b_norm_residual,b_norm_residual_alt2/hmf.bias_norm(Gs))
#        #hcekc all the matter in a halo if include normalization factor
#        assert np.allclose(np.trapz(hmf.f_sigma(Ms,1.,1.),np.log(1./hmf.sigma[:-1:])),hmf.f_norm(1.))
#
#        #sanity check M_star
#        assert np.round(np.log10(hmf.M_star())) == 13.
#        assert np.isclose(C.sigma_r(0.,8./C.h),cosmo_fid['sigma8'],rtol=1.e-3)
#        assert np.isclose(C.sigma_r(1.,8./C.h),C.G_norm(1.)*cosmo_fid['sigma8'],rtol=1.e-3)
#
#        #M_restrict = 10**np.linspace(np.log10(4.*10**13),16,100)
#        #n_restrict = hmf.n_avg(M_restrict,0.)
#        nu = hmf.nu_of_M(Ms)
#        bias_nu = hmf.bias_nu(nu)
#        bias_G = hmf.bias_G(Ms,1.)
#        bias_z = hmf.bias(Ms,0.)
#        assert np.allclose(bias_nu,bias_G)
#        assert np.allclose(bias_nu,bias_z)
#        assert np.allclose(bias_G,bias_z)
#        assert np.all(bias_nu>=0)
#        dndm = hmf.dndM_G(Ms,1.)
#        bias_avg = np.trapz(bias_nu*dndm,Ms)
#
#        n_avg2 = hmf.n_avg(Ms,0.)
#        assert np.all(n_avg2>=0.)
#        bias_n_avg1 = bias_nu*n_avg2
#        #bias_n_avg2 = hmf.bias_n_avg(Ms)
#
#        integ_pred = np.trapz(hmf.f_sigma(Ms,1.,1.),np.log(1./hmf.sigma[:-1:]))
#        integ_res = np.trapz(Ms*hmf.dndM_G(Ms,1.),Ms)/C.rho_bar(0.)
#        #assert np.isclose(integ_res,integ_pred,rtol=1.e-2)
#        #xs n#np.linspace(np.log(np.sqrt(nu[0])),np.log(nu[-1]),10000)
#        #xs = np.exp(np.linspace(np.log(np.sqrt(nu[0]),np.log(1.e36),10000))
#        #F = -cumtrapz(1./np.sqrt(2.*np.pi)*np.exp(-xs[::-1]**2/2.),xs[::-1])[::-1]
#        #cons_res = np.trapz(hmf.dndM_G(Ms,1.)*Ms*hmf.bias(Ms,1.),Ms)/C.rho_bar(0.)
#        assert np.isclose(np.trapz(bias_nu*hmf.f_nu(nu),np.sqrt(nu)),1.,rtol=1.e-1)
#        #assert np.isclose(np.trapz(hmf.f_nu(nu)/np.sqrt(nu),np.sqrt(nu)),1.,rtol=2.e-1)
#
#        #bias_avg = np.trapz(hmf.dndM_G(Ms,1.)*hmf.bias_G(Ms,1.),Ms)/np.trapz(hmf.dndM_G(Ms,1.),Ms)
#        #check for various edge cases of n_avg and bias_n_avg
#        z2s = np.array([0.,1.])
#        m2s = np.array([1.e8,1.e9])
#        n1 = hmf.n_avg(m2s[0],z2s[0])
#        n2 = hmf.n_avg(m2s[0],z2s)
#        n3 = hmf.n_avg(m2s,z2s[0])
#        n4 = hmf.n_avg(m2s,z2s)
#        n5 = hmf.n_avg(m2s[1],z2s[1])
#        n6 = hmf.n_avg(m2s[1],z2s[0])
#        n7 = hmf.n_avg(m2s[0],z2s[1])
#        assert np.isclose(n1,n2[0])
#        assert np.isclose(n1,n3[0])
#        assert np.isclose(n1,n4[0])
#        assert np.isclose(n5,n4[1])
#        assert np.isclose(n2[1],n7)
#        assert np.isclose(n3[1],n6)
#        bn1 = hmf.bias_n_avg(m2s[0],z2s[0])
#        bn2 = hmf.bias_n_avg(m2s[0],z2s)
#        bn3 = hmf.bias_n_avg(m2s,z2s[0])
#        bn4 = hmf.bias_n_avg(m2s,z2s)
#        bn5 = hmf.bias_n_avg(m2s[1],z2s[1])
#        bn6 = hmf.bias_n_avg(m2s[1],z2s[0])
#        bn7 = hmf.bias_n_avg(m2s[0],z2s[1])
#        assert np.isclose(bn1,bn2[0])
#        assert np.isclose(bn1,bn3[0])
#        assert np.isclose(bn1,bn4[0])
#        assert np.isclose(bn5,bn4[1])
#        assert np.isclose(bn2[1],bn7)
#        assert np.isclose(bn3[1],bn6)
#
#        n_avgs_0 = hmf.n_avg(hmf.mass_grid,0.)
#        n_avgs_1 = np.zeros(hmf.mass_grid.size)
#        for itr in range(0,hmf.mass_grid.size):
#            n_avgs_1[itr] = hmf.n_avg(hmf.mass_grid[itr],0.)
#        assert np.allclose(n_avgs_0,n_avgs_1)
#
#        bn_avgs_0 = hmf.bias_n_avg(hmf.mass_grid,0.)
#        bn_avgs_1 = np.zeros(hmf.mass_grid.size)
#        for itr in range(0,hmf.mass_grid.size):
#            bn_avgs_1[itr] = hmf.bias_n_avg(hmf.mass_grid[itr],0.)
#        assert np.allclose(bn_avgs_0,bn_avgs_1)
#
#        print("PASS: sanity passed")
#
#    do_plot_test2 = True
#    if do_plot_test2:
#        #Ms = 10**(np.linspace(11,14,500))
#       # dndM_G=hmf.dndM_G(Ms,Gs)
#        do_jenkins_comp = True
#        #should look like dotted line in figure 3 of    arXiv:astro-ph/0005260
#        if do_jenkins_comp:
#            print("jenkins_comp")
#            zs = np.array([0.])
#            Gs = C.G_norm(zs)
#            dndM_G = hmf.f_sigma(Ms,Gs)
#            input_dndm = np.loadtxt('test_inputs/hmf/dig_jenkins_fig3.csv',delimiter=',')
#            plt.plot(np.log(1./hmf.sigma[:-1:]),np.log(dndM_G))
#            plt.plot(input_dndm[:,0],input_dndm[:,1])
#            plt.xlim([-1.3,1.25])
#            plt.ylim([-10.5,-0.5])
#            plt.grid()
#            plt.show()
#            res_j = np.exp(input_dndm[:,1])
#            res_i = InterpolatedUnivariateSpline(1./hmf.sigma[:-1:],dndM_G,k=3,ext=2)(np.exp(input_dndm[:,0]))
#            assert np.all(np.abs((res_j/res_i-1.)[0:19])<0.03)
#            print("PASS: jenkins_comp")
#
#        do_sheth_bias_comp1 = True
#        #agrees pretty well, maybe as well as it should
#        #compare to rightmost arXiv:astro-ph/9901122 figure 3
#        if do_sheth_bias_comp1:
#            print("sheth_bias_comp1")
#            cosmo_fid2 = cosmo_fid.copy()
#            cosmo_fid2['Omegamh2'] = 0.3*0.7**2
#            cosmo_fid2['Omegabh2'] = 0.05*0.7**2
#            cosmo_fid2['OmegaLh2'] = 0.7*0.7**2
#            cosmo_fid2['sigma8'] = 0.9
#            cosmo_fid2['h'] = 0.7
#            cosmo_fid2['ns'] = 1.0
#            cosmo_fid2 = cp.add_derived_pars(cosmo_fid2,p_space='basic')
#            C2 = cp.CosmoPie(cosmo_fid2,'basic')
#            P2 = mps.MatterPower(C2,power_params)
#            C2.set_power(P2)
#            params2 = params.copy()
#            hmf2 = ST_hmf(C2,params=params2)
#            zs = np.array([0.,1.,2.,4.])
#            Gs = C2.G_norm(zs)
#            input_bias = np.loadtxt('test_inputs/hmf/dig_sheth_fig3.csv',delimiter=',')
#            bias = hmf2.bias_G(10**input_bias[:,0],Gs)
#            plt.loglog(10**input_bias[:,0],bias**2)
#            plt.loglog(10**input_bias[:,0],10**input_bias[:,1:5])
#            plt.xlim([10**11,10**14])
#            plt.ylim([0.2,80])
#            plt.grid()
#            plt.show()
#            error = np.abs(1.-10**input_bias[:,1:5]/bias**2)
#            error_max = np.array([ 0.06,  0.4,  0.2,  0.4])
#            assert np.all(np.max(error,axis=0)<error_max)
#            print("PASS: sheth_bias_comp1")
#
#        do_sheth_bias_comp2 = True
#        #agrees well
#        #compare to bottom arXiv:astro-ph/9901122 figure 4
#        if do_sheth_bias_comp2:
#            print("sheth_bias_comp2")
#            cosmo_fid2 = cosmo_fid.copy()
#            cosmo_fid2['Omegamh2'] = 0.3*0.7**2
#            cosmo_fid2['Omegabh2'] = 0.05*0.7**2
#            cosmo_fid2['OmegaLh2'] = 0.7*0.7**2
#            cosmo_fid2['sigma8'] = 0.9
#            cosmo_fid2['h'] = 0.7
#            cosmo_fid2['ns'] = 1.0
#            cosmo_fid2 = cp.add_derived_pars(cosmo_fid2,p_space='basic')
#            C2 = cp.CosmoPie(cosmo_fid2,'basic')
#            P2 = mps.MatterPower(C2,power_params)
#            C2.set_power(P2)
#            C2.k = P2.k
#            params2 = params.copy()
#            hmf2 = ST_hmf(C2,params=params2)
#            input_bias = np.loadtxt('test_inputs/hmf/dig_sheth2.csv',delimiter=',')
#            bias = hmf2.bias_nu(10**input_bias[:,0])
#            plt.loglog(10**input_bias[:,0],1.+(bias-1.)*hmf2.delta_c)
#            plt.loglog(10**input_bias[:,0],10**input_bias[:,1])
#            plt.xlim([0.1,19.95])
#            plt.ylim([0.2,16.94])
#            plt.grid()
#            plt.show()
#            observable = 1.+(bias-1.)*hmf2.delta_c
#            error = np.abs(1.-10**input_bias[:,1]/observable)
#            error_max = 0.07
#            assert np.all(np.max(error)<error_max)
#            print("PASS: sheth_bias_comp2")
#
#
#        #might agree
#        do_hu_bias_comp1 = False
#        if do_hu_bias_comp1:
#            print("hu_bias_comp1")
#            zs = np.array([0.])
#            Gs = C.G_norm(zs)
#            bias = hmf.bias_G(Ms,Gs,1.)[:,0]
#            input_bias = np.loadtxt('test_inputs/hmf/dig_hu.csv',delimiter=',')
#            plt.semilogx(Ms,bias)
#            plt.semilogx(10**input_bias[:,0],input_bias[:,1])
#            plt.xlim([10**13.5,10**15.5])
#            plt.ylim([1,8])
#            plt.grid()
#            plt.show()
#            #masses = 10**np.linspace(np.log10(3*10**14),np.log10(2*10**15),100)
#            #bias_hu_i = InterpolatedUnivariateSpline(10**input_bias[:,0],input_bias[:,1])(masses)
#            bias_hu_i = input_bias[:,1]
#            bias_i = hmf.bias_G(10**input_bias[:,0],Gs,1.)[:,0]
#            #bias_i = InterpolatedUnivariateSpline(Ms,bias,k=3,ext=2)(10**input_bias[:,0])
#        #should agree, does
#        #match fig 2 arXiv:astro-ph/0203169
#        do_hu_bias_comp2 = True
#        if do_hu_bias_comp2:
#            print("hu_bias_comp2")
#            zs = np.array([0.])
#            Gs = C.G_norm(zs)
#            bias = hmf.bias_G(Ms,Gs,1.)[:,0] #why 1
#            #maybe should have k pivot 0.01
#            input_bias = np.loadtxt('test_inputs/hmf/dig_hu_bias2.csv',delimiter=',')
#            plt.loglog(Ms,bias)
#            plt.loglog(10**input_bias[:,0],10**input_bias[:,1])
#            plt.xlim([10**11,10**16])
#            plt.ylim([0.01,10])
#            plt.grid()
#            plt.show()
#            masses = 10**np.linspace(np.log10(10**11),np.log10(10**16),100)
#            bias_hu_i = 10**input_bias[:,1]#InterpolatedUnivariateSpline(10**input_bias[:,0],10**input_bias[:,1])(masses)
#            bias_i = hmf.bias_G(10**input_bias[:,0],Gs,1.)[:,0]
#            #bias_i = InterpolatedUnivariateSpline(Ms,bias,k=3,ext=2)(10**input_bias[:,0])
#            assert np.max(np.abs(bias_hu_i/bias_i-1.))<0.06
#            print("PASS: hu_bias_comp2 passed")
#
#        do_hu_bias_comp3 = False
#        if do_hu_bias_comp3:
#            print("hu_bias_comp3")
#            zs = np.array([0.])
#            Gs = C.G_norm(zs)
#            bias = hmf.bias_G(Ms,Gs,1.)[:,0] #why 1
#            #maybe should have k pivot 0.01
#            #input_bias = np.loadtxt('test_inputs/hmf/dig_hu_bias2.csv',delimiter=',')
#            plt.loglog(hmf.nu_of_M(Ms),(bias-1.)*1.686+1.)
#            #plt.loglog(10**input_bias[:,0],10**input_bias[:,1])
#            plt.xlim([0.1,20])
#            plt.ylim([0.1,20])
#            plt.grid()
#            plt.show()
#            #masses = 10**np.linspace(np.log10(10**11),np.log10(10**16),100)
#            #bias_hu_i = 10**input_bias[:,1]#InterpolatedUnivariateSpline(10**input_bias[:,0],10**input_bias[:,1])(masses)
#            #bias_i = hmf.bias_G(10**input_bias[:,0],Gs,1.)[:,0]
#            #bias_i = InterpolatedUnivariateSpline(Ms,bias,k=3,ext=2)(10**input_bias[:,0])
#            #assert np.max(np.abs(bias_hu_i/bias_i-1.))<0.06
#            #print("PASS: hu_bias_comp3 passed")
#        #might agree
#        #match fig B10 arXiv:astro-ph/0203169
#        do_hu_navg_comp1 = False
#        if do_hu_navg_comp1:
#            print("hu_navg_comp")
#            cosmo_fid2 = cosmo_fid.copy()
#            cosmo_fid2['h'] = 0.65
#            cosmo_fid2['Omegamh2'] = 0.15*0.65**2
#            cosmo_fid2['Omegabh2'] = 0.02
#            cosmo_fid2['OmegaLh2'] = (1-0.15)*cosmo_fid2['h']**2
#            cosmo_fid2['sigma8'] = 1.068
#            cosmo_fid2['ns'] = 1.
#            cosmo_fid2 = cp.add_derived_pars(cosmo_fid2,p_space='basic')
#            C2 = cp.CosmoPie(cosmo_fid2,'basic')
#            P2 = mps.MatterPower(C2,power_params)
#            C2.set_power(P2)
#            C2.k = P2.k
#            zs = np.array([0.])
#            Gs = C2.G_norm(zs)
#            hmf2 = ST_hmf(C2,params=params)
#            input_navg = np.loadtxt('test_inputs/hmf/dig_hu_navg.csv',delimiter=',')
#            navg = hmf2.n_avg(10**input_navg[:,0],0.)
#            plt.loglog(10**input_navg[:,0],navg)
#            plt.loglog(10**input_navg[:,0],10**input_navg[:,1])
#            plt.xlim([10**13.6,10**16])
#            plt.ylim([10**-9,10**-4])
#            plt.grid()
#            plt.show()
#            #masses = 10**np.linspace(np.log10(10**13.6),np.log10(10**16),100)
#            navg_hu_i = 10**input_navg[:,1]#InterpolatedUnivariateSpline(10**input_navg[:,0],10**input_navg[:,1])(masses)
#            navg_i = navg#InterpolatedUnivariateSpline(10**input_navg[:,0],navg)(masses)
#        #should agree, does
#        #match fig 1 arXiv:astro-ph/0203169
#        do_hu_sigma_comp = True
#        if do_hu_sigma_comp:
#            print("hu_sigma_comp")
#            assert np.isclose(hmf.M_star(),1.2*10**13,rtol=1.e-1)
#            input_sigma = np.loadtxt('test_inputs/hmf/dig_hu_sigma.csv',delimiter=',')
#            plt.loglog(hmf.R,hmf.sigma)
#            plt.loglog(10**input_sigma[:,0],10**input_sigma[:,1])
#            res_sigma = InterpolatedUnivariateSpline(hmf.R,hmf.sigma,k=3,ext=2)(10**input_sigma[:,0])
#            plt.xlim([0.1,200])
#            plt.ylim([0.001,10])
#            plt.grid()
#            plt.show()
#            assert np.all(np.abs((res_sigma/10**input_sigma[:,1])[4::]-1.)<0.02)
#            print("PASS: hu_sigma_comp")
######only possible if hmf module installed
##    do_hmf_comp = False
##    if do_hmf_comp:
##        print("hmf_comp")
##        power_params = defaults.power_params.copy()
##        power_params.camb['force_sigma8'] = True
##        power_params.camb['npoints'] = 1000
##        power_params.camb['maxkh'] = 20000.
##        power_params.camb['kmax'] = 0.899999976158
##        C = cp.CosmoPie(cosmo_fid,'basic')
##        P = mps.MatterPower(C,power_params)
##        C.set_power(P)
##        params = defaults.hmf_params.copy()
##        params['z_min'] = 0.0
##        params['z_max'] = 5.0
##        params['log10_min_mass'] = 6
##        params['log10_max_mass'] = 18.63
##        params['n_grid'] = 12640
##        params['n_z'] = 5./0.5
##        hmf = ST_hmf(C,params=params)
##        Ms = hmf.mass_grid
##        from hmf import MassFunction
##        from hmf_hold import ST_hmf
##        from hmf import cosmo
##        from hmf import integrate_hmf
##        from hmf import fitting_functions as ff
##        from astropy.cosmology import FlatLambdaCDM
##        cosmo_params = {'Ob0':C.Omegab,'Om0':C.Omegam,'H0':C.H0}
##        transfer_params = {}
##        cosmo_model = FlatLambdaCDM(H0=C.H0,Ob0=C.Omegab,Om0=C.Omegam,Tcmb0=2.7255,Neff=3.046)
##        hmf_alt = MassFunction(z=0,n=1.,sigma_8=0.92,cosmo_params=cosmo_params,cosmo_model=cosmo_model,Mmin=6.,Mmax=18.631,lnk_min=np.log(np.min(C.k)/C.h),lnk_max=np.log(np.max(C.k)/C.h)+0.001,delta_wrt='mean',hmf_model=ff.ST,dlnk=0.019037555300007725,dlog10m=0.001,transfer_params=transfer_params)
##        assert np.allclose(hmf.dndM_G(hmf_alt.m[:-1:],1.),hmf_alt.dndm[:-1:],rtol=1.e-3)
##        n_avg_i = hmf.n_avg(hmf_alt.m[:-1:],0.)
##        n_avg_j = integrate_hmf.hmf_integral_gtm(hmf_alt.m[:-1:],hmf_alt.dndm[:-1:])
##        assert np.allclose(n_avg_i,n_avg_j,rtol=1.e-2)
##        assert np.allclose(hmf.f_sigma(hmf_alt.m[:-1:],1.),hmf_alt.fsigma[:-1:],rtol=1.e-1)
##        assert np.allclose(hmf.nu_of_M(hmf_alt.m[:-1:]),hmf_alt.nu[:-1:],rtol=1.e-2)
##        assert np.allclose(hmf.sigma_of_M(hmf_alt.m[:-1:]),hmf_alt.sigma[:-1:],rtol=1.e-2)
##        assert np.allclose(hmf.mass_func(hmf_alt.m[:-1:],1.)[-1]*hmf_alt.m[:-1:],hmf_alt.dndlnm[:-1:],rtol=1.e-2)
##        print("PASS: hmf_comp")
