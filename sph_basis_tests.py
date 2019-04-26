"""do some tests for sph_klim"""
from __future__ import division,print_function,absolute_import
from builtins import range

import numpy as np
import pytest

from cosmopie import CosmoPie
from sph_klim import SphBasisK
from matter_power_spectrum import MatterPower
from half_sky_geo import HalfSkyGeo
from full_sky_geo import FullSkyGeo

import defaults

def test_full_sky():
    """do some tests with a full sky geo known results"""
    #get dictionaries of parameters that various functions will need
    #cosmo = defaults.cosmology_wmap.copy()
    cosmo = {   'Omegabh2':0.02223,
                    'Omegach2':0.1153,
                    'Omegab'  :0.04283392714316876,
                    'Omegac'  :0.22216607285683124,
                    'Omegamh2':0.13752999999999999,
                    'OmegaL'  :0.735,
                    'OmegaLh2':0.38145113207547166,
                    'Omegam'  :0.265,
                    'H0'      :72.04034509047493,
                    'sigma8'  : 0.8269877678406697, #from the code
                    'h'       :0.7204034509047493,
                    'Omegak'  : 0.0,
                    'Omegakh2': 0.0,
                    'Omegar'  : 0.0,
                    'Omegarh2': 0.0,
                    'ns'      : 0.9608,
                    'tau'     : 0.081,
                    'Yp'      :0.299,
                    'As'      : 2.464*10**-9,
                    'LogAs'   :-19.821479791275138,
                    'w'       :-1.0,
                    'de_model':'constant_w',#dark energy model
                    'mnu'     :0.}
    cosmo['de_model'] = 'constant_w'
    cosmo['wa'] = 0.
    cosmo['w0'] = -1.
    cosmo['w'] = -1.
    p_space = 'jdem'
    camb_params = defaults.camb_params.copy()
    camb_params['force_sigma8'] = False
    camb_params['maxkh'] = 100.
    camb_params['kmax'] = 30.
    camb_params['npoints'] = 10000
    camb_params['pivot_scalar'] = 0.002

    power_params = defaults.power_params.copy()
    power_params.camb = camb_params
    power_params.camb['accuracy'] = 1
    prior_params = defaults.prior_fisher_params.copy()


    zs = np.array([0.0001,3.])
    z_fine = np.linspace(0.0001,zs[-1],10000)

    basis_params = defaults.basis_params.copy()
    basis_params['n_bessel_oversample'] = 400000*10

    C = CosmoPie(cosmo,p_space=p_space)
    l_max = 0
    x_cut = 10*np.pi

    z_max = z_fine[-1]+0.000001
    r_max = C.D_comov(z_max)
    k_cut = x_cut/r_max
    P = MatterPower(C,power_params)
    C.set_power(P)
    geo1 = FullSkyGeo(zs,C,z_fine)

    basis = SphBasisK(r_max,C,k_cut,basis_params,l_ceil=l_max,needs_m=True)

    ddbar = basis.get_ddelta_bar_ddelta_alpha(geo1,tomography=True)
    ns = np.arange(1,ddbar.size+1)
    ddbar_pred = -3./(2.*np.pi**(5./2.))*(-1)**ns/ns**2
    assert np.allclose(ddbar,ddbar_pred)
    cov_pred = np.zeros((ns.size,ns.size))
    xs = P.k*r_max
    P_in = P.get_matter_power(np.array([0.]),pmodel='linear')[:,0]
    for i1 in range(0,ns.size):
        for i2 in range(i1,ns.size):
            integrand1 = 8.*np.pi**3*(-1)**(ns[i1]+ns[i2])*ns[i1]**2*ns[i2]**2/r_max**3*np.sin(xs)**2/((ns[i1]**2*np.pi**2-xs**2)*(ns[i2]**2*np.pi**2-xs**2))
            cov_pred[i1,i2] = np.trapz(integrand1*P_in,xs)
            if i1!=i2:
                cov_pred[i2,i1] = cov_pred[i1,i2]
    cov_got = basis.get_covar_array()
    var_pred = np.dot(ddbar_pred.T,np.dot(cov_pred,ddbar_pred))
    var_got = basis.get_variance(geo1)
    assert np.all(cov_got==cov_got.T)
    assert np.allclose(basis.C_id[:,1],ns*np.pi/r_max)
    assert np.allclose(cov_pred,cov_got,atol=1.e-15,rtol=1.e-3)
    assert np.isclose(var_got[0,0],var_pred,atol=1.e-15,rtol=1.e-3)
    ddbar_got2 = basis.get_dO_I_ddelta_alpha(geo1,geo1.r_fine**2)*3./r_max**3
    assert np.allclose(ddbar,ddbar_got2,rtol=1.e-3)
    #ddbar_got1_fine = basis.get_ddelta_bar_ddelta_alpha(geo1,tomography=False)
    #ddbar_got2_fine = basis.get_dO_I_ddelta_alpha(geo1,tomography=False)


def test_half_sky():
    """do some tests with a half sky geo known results"""
    #get dictionaries of parameters that various functions will need
    cosmo = {   'Omegabh2':0.0222,
                    'Omegach2':0.1153,
                    'Omegab'  :0.04283392714316876,
                    'Omegac'  :0.22216607285683124,
                    'Omegamh2':0.13752999999999999,
                    'OmegaL'  :0.735,
                    'OmegaLh2':0.38145113207547166,
                    'Omegam'  :0.265,
                    'H0'      :72.04034509047493,
                    'sigma8'  : 0.8269877678406697, #from the code
                    'h'       :0.7204034509047493,
                    'Omegak'  : 0.0,
                    'Omegakh2': 0.0,
                    'Omegar'  : 0.0,
                    'Omegarh2': 0.0,
                    'ns'      : 0.9608,
                    'tau'     : 0.081,
                    'Yp'      :0.299,
                    'As'      : 2.464*10**-9,
                    'LogAs'   :-19.821479791275138,
                    'w'       :-1.0,
                    'de_model':'constant_w',#dark energy model
                    'mnu'     :0.}
    cosmo['de_model'] = 'constant_w'
    cosmo['wa'] = 0.
    cosmo['w0'] = -1.
    cosmo['w'] = -1.
    p_space = 'jdem'
    camb_params = defaults.camb_params.copy()
    camb_params['force_sigma8'] = False
    camb_params['maxkh'] = 100.
    camb_params['kmax'] = 30.
    camb_params['npoints'] = 10000
    camb_params['pivot_scalar'] = 0.002

    power_params = defaults.power_params.copy()
    power_params.camb = camb_params
    power_params.camb['accuracy'] = 1
    prior_params = defaults.prior_fisher_params.copy()


    zs = np.array([0.0001,3.])
    z_fine = np.linspace(0.0001,zs[-1],10000)

    basis_params = defaults.basis_params.copy()
    basis_params['n_bessel_oversample'] = 400000*10
    #test with half sky using results calculated in mathematica
    basis_params = defaults.basis_params.copy()
    basis_params['n_bessel_oversample'] = 400000*10

    C = CosmoPie(cosmo,p_space=p_space)
    l_max = 4
    x_cut = 10

    z_max = z_fine[-1]+0.000001
    r_max = C.D_comov(z_max)
    k_cut = x_cut/r_max

    geo1 = HalfSkyGeo(zs,C,z_fine)

    P = MatterPower(C,power_params)
    P.P_lin = 1./(P.k*r_max)
    C.set_power(P)

    basis = SphBasisK(r_max,C,k_cut,basis_params,l_ceil=l_max,needs_m=False)
    basis2 = SphBasisK(r_max,C,k_cut,basis_params,l_ceil=l_max,needs_m=True)

    ddbar = basis.get_ddelta_bar_ddelta_alpha(geo1,tomography=True)
    ddbar2 = basis2.get_ddelta_bar_ddelta_alpha(geo1,tomography=True)
    ddbar_got2 = basis.get_dO_I_ddelta_alpha(geo1,geo1.r_fine**2)*3./r_max**3

    ns = np.arange(1,ddbar.size+1)
    cov_got = basis.get_covar_array()
    cov_got2 = basis2.get_covar_array()
    var_got = basis.get_variance(geo1)
    var_got2 = basis2.get_variance(geo1)
    cov_pred0 = np.array([[3.220393570704632e-11,-9.51309537893619e-12,9.955267964241254e-12],[-9.51309537893619e-12,5.937410951604845e-11,-1.2342996089442917e-11],[9.955267964241254e-12,-1.2342996089442917e-11,8.524795995567036e-11]])
    cov_pred1 = np.array([[4.306831299006748e-11,-7.977015797357624e-12],[-7.977015797357624e-12,6.907871421795387e-11]])
    cov_pred2 = np.array([[5.467066924435653e-11,-7.056047462103054e-12],[-7.056047462103054e-12,7.984696153964485e-11]])
    cov_pred3 = np.array([[6.70079506270691e-11]])
    cov_pred4 = np.array([[7.985214180284307e-11]])
    cov_pred = np.zeros((ddbar.size,ddbar.size))
    cov_pred[0:3,0:3] = cov_pred0
    cov_pred[3:5,3:5] = cov_pred1
    cov_pred[5:7,5:7] = cov_pred2
    cov_pred[7,7] = cov_pred3
    cov_pred[8,8] = cov_pred4
    var_pred = np.dot(ddbar,np.dot(cov_pred,ddbar.T))
    var_pred2 = np.dot(ddbar2,np.dot(cov_got2,ddbar2.T))
    assert np.all(cov_got==cov_got.T)
    assert np.all(cov_got2==cov_got2.T)
    assert np.isclose(var_got[0,0],var_pred,atol=1.e-15,rtol=1.e-3)
    assert np.isclose(var_got2[0,0],var_pred2,atol=1.e-15,rtol=1.e-3)
    assert np.isclose(var_got2[0,0],var_pred,atol=1.e-15,rtol=1.e-3)
    assert np.allclose(cov_pred,cov_got,atol=1.e-15,rtol=1.e-3)
    assert np.allclose(ddbar,ddbar_got2,rtol=1.e-3,atol=1.e-15)
    for itr in range(0,basis.C_id.shape[0]):
        if basis.C_id[itr,0]>0 and np.mod(basis.C_id[itr,0],2)==0.:
            assert ddbar[0,itr] == 0.
    for itr in range(0,basis2.C_id.shape[0]):
        if basis2.C_id[itr,0]>0 and np.mod(basis2.C_id[itr,0],2)==0.:
            assert ddbar2[0,itr] == 0.
        elif basis2.C_id[itr,2]!=0:
            assert ddbar2[0,itr] == 0.

if __name__=='__main__':
    pytest.cmdline.main(['sph_basis_tests.py'])
