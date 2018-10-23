"""demonstration results invariant under global rotations"""
from __future__ import print_function,division,absolute_import
from builtins import range
import numpy as np
import pytest

from cosmopie import CosmoPie
import defaults
from sph_klim import SphBasisK
import matter_power_spectrum as mps
from premade_geos import WFIRSTGeo,LSSTGeo
from alm_rot_geo import AlmRotGeo
from nz_wfirst_eff import NZWFirstEff
from sw_survey import SWSurvey
from lw_survey import LWSurvey
from super_survey import SuperSurvey

def test_rotational_invariance():
    """test invariance of results under global rotations"""
    camb_params = { 'npoints':2000,
                    'minkh':1.1e-4,
                    'maxkh':1.476511342960e+02,
                    'kmax':1.476511342960e+02,
                    'leave_h':False,
                    'force_sigma8':False,
                    'return_sigma8':False,
                    'accuracy':1,
                    'pivot_scalar':0.002
                  }
    print("main: building cosmology")
    power_params = defaults.power_params.copy()
    power_params.camb = camb_params
    power_params.camb['accuracy'] = 1
    power_params.camb['maxkh'] = 100.
    power_params.camb['kmax'] = 30.
    power_params.camb['npoints'] = 1000
    C = CosmoPie(defaults.cosmology.copy(),p_space='jdem')
    P_lin = mps.MatterPower(C,power_params)
    C.set_power(P_lin)

    l_max = 23
    x_cut = 30.

    print("main: building geometries")
    polygon_params = defaults.polygon_params.copy()
    polygon_params['n_double'] = 80
    z_coarse = np.array([0.2,1.,2.,3.])
    zs_lsst = np.linspace(0.,1.2,3)
    #z_max = np.max(z_coarse)
    #z_fine = np.arange(0.0001,z_max,0.0001)
    z_fine = np.linspace(0.001,3.,500)
    z_max = z_fine[-1]+0.001

    print("main: building basis")
    basis_params = defaults.basis_params.copy()
    basis_params['n_bessel_oversample'] = 400000
    basis_params['x_grid_size'] = 100000

    r_max = C.D_comov(z_max)
    k_cut = x_cut/r_max

    basis = SphBasisK(r_max,C,k_cut,basis_params,l_ceil=l_max)

    geo1 = WFIRSTGeo(z_coarse,C,z_fine,l_max,polygon_params)#HalfSkyGeo(z_coarse,C,z_fine)
    geo2 = LSSTGeo(zs_lsst,C,z_fine,l_max,polygon_params)#HalfSkyGeo(z_coarse,C,z_fine)
    geo1_rot1 = AlmRotGeo(geo1,C,z_coarse,z_fine,np.array([0.,np.pi/2.,np.pi]),polygon_params['n_double'])
    geo1_rot2 = AlmRotGeo(geo1_rot1,C,z_coarse,z_fine,np.array([0.,np.pi/2.,np.pi]),polygon_params['n_double'])
    geo1_rot3 = AlmRotGeo(geo1_rot2,C,z_coarse,z_fine,np.array([0.005,1.2496,1.72]),polygon_params['n_double'])
    geo2_rot1 = AlmRotGeo(geo2,C,zs_lsst,z_fine,np.array([0.,np.pi/2.,np.pi]),polygon_params['n_double'])
    geo2_rot2 = AlmRotGeo(geo2_rot1,C,zs_lsst,z_fine,np.array([0.,np.pi/2.,np.pi]),polygon_params['n_double'])
    geo2_rot3 = AlmRotGeo(geo2_rot2,C,zs_lsst,z_fine,np.array([0.005,1.2496,1.72]),polygon_params['n_double'])
    assert np.allclose(geo1.get_alm_array(l_max),geo1_rot2.get_alm_array(l_max))
    var_geo1 = basis.get_variance(geo1,k_cut_in=k_cut)
    var_geo1_rot1 = basis.get_variance(geo1_rot1,k_cut_in=k_cut)
    var_geo1_rot2 = basis.get_variance(geo1_rot2,k_cut_in=k_cut)
    var_geo1_rot3 = basis.get_variance(geo1_rot3,k_cut_in=k_cut)
    assert np.allclose(var_geo1,var_geo1_rot1,atol=1.e-20,rtol=1.e-8)
    assert np.allclose(var_geo1,var_geo1_rot2,atol=1.e-20,rtol=1.e-8)
    assert np.allclose(var_geo1,var_geo1_rot3,atol=1.e-20,rtol=1.e-8)
    var_geo2 = basis.get_variance(geo2,k_cut_in=k_cut)
    var_geo2_rot1 = basis.get_variance(geo2_rot1,k_cut_in=k_cut)
    var_geo2_rot2 = basis.get_variance(geo2_rot2,k_cut_in=k_cut)
    var_geo2_rot3 = basis.get_variance(geo2_rot3,k_cut_in=k_cut)
    assert np.allclose(var_geo2,var_geo2_rot1,atol=1.e-20,rtol=1.e-8)
    assert np.allclose(var_geo2,var_geo2_rot2,atol=1.e-20,rtol=1.e-8)
    assert np.allclose(var_geo2,var_geo2_rot3,atol=1.e-20,rtol=1.e-8)

    cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w'])
    cosmo_par_eps = np.array([0.002,0.00025,0.0001,0.00025,0.1,0.01])

    nz_params_wfirst_lens = defaults.nz_params_wfirst_lens.copy()
    nz_params_wfirst_lens['i_cut'] = 26.3
    nz_params_wfirst_lens['data_source'] = './data/CANDELS-GOODSS2.dat'
    nz_wfirst_lens = NZWFirstEff(nz_params_wfirst_lens)
    sw_params = defaults.sw_survey_params.copy()
    lw_params = defaults.lw_survey_params.copy()
    sw_observable_list = defaults.sw_observable_list.copy()
    lw_observable_list = defaults.lw_observable_list.copy()
    len_params = defaults.lensing_params.copy()
    mf_params = defaults.hmf_params.copy()
    mf_params['n_grid'] = 2000
    mf_params['log10_min_mass'] = 10.
    n_params_lsst = defaults.nz_params_lsst_use.copy()
    n_params_lsst['i_cut'] = 24.1 #gold standard subset of LSST 1 year (10 year 25.3)

    dn_params = defaults.dn_params.copy()
    dn_params['nz_select'] = 'LSST'
    dn_params['sigma0'] = 0.1
    lw_param_list = np.array([{'dn_params':dn_params,'n_params':n_params_lsst,'mf_params':mf_params}])
    prior_params = defaults.prior_fisher_params.copy()

    sw_survey_geo1 = SWSurvey(geo1,'geo1',C,sw_params,cosmo_par_list,cosmo_par_eps,sw_observable_list,len_params,nz_wfirst_lens)
    sw_survey_geo1_rot1 = SWSurvey(geo1_rot1,'geo1_rot1',C,sw_params,cosmo_par_list,cosmo_par_eps,sw_observable_list,len_params,nz_wfirst_lens)
    sw_survey_geo1_rot2 = SWSurvey(geo1_rot2,'geo1_rot2',C,sw_params,cosmo_par_list,cosmo_par_eps,sw_observable_list,len_params,nz_wfirst_lens)
    sw_survey_geo1_rot3 = SWSurvey(geo1_rot3,'geo1_rot3',C,sw_params,cosmo_par_list,cosmo_par_eps,sw_observable_list,len_params,nz_wfirst_lens)

    survey_lw1 = LWSurvey(np.array([geo1,geo2]),'lw1',basis,C,lw_params,observable_list=lw_observable_list,param_list=lw_param_list)
    survey_lw1_rot1 = LWSurvey(np.array([geo1_rot1,geo2_rot1]),'lw1',basis,C,lw_params,observable_list=lw_observable_list,param_list=lw_param_list)
    survey_lw1_rot2 = LWSurvey(np.array([geo1_rot2,geo2_rot2]),'lw1',basis,C,lw_params,observable_list=lw_observable_list,param_list=lw_param_list)
    survey_lw1_rot3 = LWSurvey(np.array([geo1_rot3,geo2_rot3]),'lw1',basis,C,lw_params,observable_list=lw_observable_list,param_list=lw_param_list)
    SS_geo1 = SuperSurvey(np.array([sw_survey_geo1]),np.array([survey_lw1]),basis,C,prior_params,get_a=True,do_unmitigated=True,do_mitigated=True,include_sw=True)
    SS_geo1_rot1 = SuperSurvey(np.array([sw_survey_geo1_rot1]),np.array([survey_lw1_rot1]),basis,C,prior_params,get_a=True,do_unmitigated=True,do_mitigated=True,include_sw=True)
    SS_geo1_rot2 = SuperSurvey(np.array([sw_survey_geo1_rot2]),np.array([survey_lw1_rot2]),basis,C,prior_params,get_a=True,do_unmitigated=True,do_mitigated=True,include_sw=True)
    SS_geo1_rot3 = SuperSurvey(np.array([sw_survey_geo1_rot3]),np.array([survey_lw1_rot3]),basis,C,prior_params,get_a=True,do_unmitigated=True,do_mitigated=True,include_sw=True)

    for i in xrange(0,3):
        for j in xrange(1,3):
            assert np.allclose(SS_geo1.f_set_nopriors[i][j].get_covar(),SS_geo1_rot1.f_set_nopriors[i][j].get_covar(),atol=1.e-30,rtol=1.e-8)
            assert np.allclose(SS_geo1.f_set_nopriors[i][j].get_covar(),SS_geo1_rot2.f_set_nopriors[i][j].get_covar(),atol=1.e-30,rtol=1.e-8)
            assert np.allclose(SS_geo1.f_set_nopriors[i][j].get_covar(),SS_geo1_rot3.f_set_nopriors[i][j].get_covar(),atol=1.e-30,rtol=1.e-8)
    a_geo1 = SS_geo1.multi_f.get_a_lw()
    a_geo1_rot1 = SS_geo1_rot1.multi_f.get_a_lw()
    a_geo1_rot2 = SS_geo1_rot2.multi_f.get_a_lw()
    a_geo1_rot3 = SS_geo1_rot3.multi_f.get_a_lw()
    assert np.allclose(a_geo1[0],a_geo1_rot1[0],atol=1.e-20,rtol=1.e-8)
    assert np.allclose(a_geo1[1],a_geo1_rot1[1],atol=1.e-20,rtol=1.e-8)
    assert np.allclose(a_geo1[0],a_geo1_rot2[0],atol=1.e-20,rtol=1.e-8)
    assert np.allclose(a_geo1[1],a_geo1_rot2[1],atol=1.e-20,rtol=1.e-8)
    assert np.allclose(a_geo1[0],a_geo1_rot3[0],atol=1.e-20,rtol=1.e-8)
    assert np.allclose(a_geo1[1],a_geo1_rot3[1],atol=1.e-20,rtol=1.e-8)
    assert np.allclose(a_geo1[0],var_geo1,atol=1.e-30,rtol=1.e-13)
    assert np.allclose(a_geo1_rot1[0],var_geo1_rot1,atol=1.e-30,rtol=1.e-11)
    assert np.allclose(a_geo1_rot2[0],var_geo1_rot2,atol=1.e-30,rtol=1.e-11)
    assert np.allclose(a_geo1_rot3[0],var_geo1_rot3,atol=1.e-30,rtol=1.e-11)

    for i in xrange(0,2):
        for j in xrange(0,2):
            eig_geo1 = SS_geo1.eig_set[i][j][0]
            eig_geo1_rot1 = SS_geo1_rot1.eig_set[i][j][0]
            eig_geo1_rot2 = SS_geo1_rot2.eig_set[i][j][0]
            eig_geo1_rot3 = SS_geo1_rot3.eig_set[i][j][0]
            assert np.allclose(eig_geo1,eig_geo1_rot1)
            assert np.allclose(eig_geo1,eig_geo1_rot2)
            assert np.allclose(eig_geo1,eig_geo1_rot3)

if __name__=='__main__':
    pytest.cmdline.main(['variance_rot_test.py'])
