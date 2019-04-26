"""test consistency of perturbing cosmopies"""
from __future__ import division,print_function,absolute_import
from builtins import range
from copy import deepcopy
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import pytest
import cosmopie as cp
import defaults
import power_parameter_response as ppr
import matter_power_spectrum as mps
import sw_survey as sws
from full_sky_geo import FullSkyGeo
from nz_wfirst_eff import NZWFirstEff
from super_survey import SuperSurvey
from sph_klim import SphBasisK
from camb_power import camb_sigma8
from change_parameters import rotate_jdem_to_lihu,rotate_lihu_to_jdem

def test_pipeline_consistency():
    """test full pipeline consistency with rotation jdem vs lihu"""
    cosmo_base = defaults.cosmology_wmap.copy()
    cosmo_base = cp.add_derived_pars(cosmo_base,'jdem')
    cosmo_base['de_model'] = 'constant_w'
    cosmo_base['w'] = -1.
    power_params = defaults.power_params.copy()
    power_params.camb['maxkh'] = 1.
    power_params.camb['kmax'] = 1.
    power_params.camb['npoints'] = 1000
    power_params.camb['accuracy'] = 2
    power_params.camb['leave_h'] = False

    cosmo_jdem = cosmo_base.copy()
    cosmo_jdem['p_space'] = 'jdem'
    C_fid_jdem = cp.CosmoPie(cosmo_jdem,'jdem')
    P_jdem = mps.MatterPower(C_fid_jdem,power_params.copy())
    C_fid_jdem.set_power(P_jdem)

    cosmo_lihu = cosmo_base.copy()
    cosmo_lihu['p_space'] = 'lihu'
    C_fid_lihu = cp.CosmoPie(cosmo_lihu,'lihu')
    P_lihu = mps.MatterPower(C_fid_lihu,power_params.copy())
    C_fid_lihu.set_power(P_lihu)

    zs = np.arange(0.2,1.41,0.40)
    z_fine = np.linspace(0.001,1.4,1000)

    geo_jdem = FullSkyGeo(zs,C_fid_jdem,z_fine)
    geo_lihu = FullSkyGeo(zs,C_fid_lihu,z_fine)

    jdem_pars = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w'])
    jdem_eps = np.array([0.002,0.00025,0.0001,0.00025,0.1,0.01])

    lihu_pars = np.array(['ns','Omegach2','Omegabh2','h','LogAs','w'])
    lihu_eps = np.array([0.002,0.00025,0.0001,0.00025,0.1,0.01])

    sw_params = defaults.sw_survey_params.copy()
    len_params = defaults.lensing_params.copy()
    sw_observable_list = defaults.sw_observable_list.copy()
    nz_wfirst_lens = NZWFirstEff(defaults.nz_params_wfirst_lens.copy())
    prior_params = defaults.prior_fisher_params.copy()
    basis_params = defaults.basis_params.copy()

    sw_survey_jdem = sws.SWSurvey(geo_jdem,'wfirst',C_fid_jdem,sw_params,jdem_pars,jdem_eps,sw_observable_list,len_params,nz_wfirst_lens)
    sw_survey_lihu = sws.SWSurvey(geo_lihu,'wfirst',C_fid_lihu,sw_params,lihu_pars,lihu_eps,sw_observable_list,len_params,nz_wfirst_lens)

    #dO_dpar_jdem = sw_survey_jdem.get_dO_I_dpar_array()
    #dO_dpar_lihu = sw_survey_lihu.get_dO_I_dpar_array()

    response_pars = np.array(['ns','Omegach2','Omegabh2','Omegamh2','OmegaLh2','h','LogAs','w'])
    response_derivs_jdem_pred = np.array([[1.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,1.,0.,1./(2.*C_fid_jdem.cosmology['h']),0.,0.],[0.,-1.,1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,1./(2.*C_fid_jdem.cosmology['h']),0.,0.],[0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,1.]]).T
    response_derivs_lihu_pred = np.array([[1.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,1.,-1.,0.,0.,0.],[0.,0.,1.,1.,-1.,0.,0.,0.],[0.,0.,0.,0.,2.*C_fid_lihu.cosmology['h'],1.,0.,0.],[0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,1.]]).T

    l_max = 24

    r_max_jdem = geo_jdem.r_fine[-1]
    k_cut_jdem = 30./r_max_jdem
    basis_jdem = SphBasisK(r_max_jdem,C_fid_jdem,k_cut_jdem,basis_params,l_ceil=l_max,needs_m=True)
    SS_jdem = SuperSurvey(np.array([sw_survey_jdem]),np.array([]),basis_jdem,C_fid_jdem,prior_params,get_a=False,do_unmitigated=True,do_mitigated=False)

    r_max_lihu = geo_lihu.r_fine[-1]
    k_cut_lihu = 30./r_max_lihu
    basis_lihu = SphBasisK(r_max_lihu,C_fid_lihu,k_cut_lihu,basis_params,l_ceil=l_max,needs_m=True)
    SS_lihu = SuperSurvey(np.array([sw_survey_lihu]),np.array([]),basis_lihu,C_fid_lihu,prior_params,get_a=False,do_unmitigated=True,do_mitigated=False)

    #dO_dpar_jdem_to_lihu = np.zeros_like(dO_dpar_jdem)
    #dO_dpar_lihu_to_jdem = np.zeros_like(dO_dpar_lihu)

    project_lihu_to_jdem = np.zeros((jdem_pars.size,lihu_pars.size))

    #f_g_jdem_to_lihu = np.zeros((lihu_pars.size,lihu_pars.size))
    #f_g_lihu_to_jdem = np.zeros((jdem_pars.size,jdem_pars.size))
    response_derivs_jdem = np.zeros((response_pars.size,jdem_pars.size))
    response_derivs_lihu = np.zeros((response_pars.size,lihu_pars.size))
    for i in range(0,response_pars.size):
        for j in range(0,jdem_pars.size):
            response_derivs_jdem[i,j] = (sw_survey_jdem.len_pow.Cs_pert[j,0].cosmology[response_pars[i]]-sw_survey_jdem.len_pow.Cs_pert[j,1].cosmology[response_pars[i]])/(jdem_eps[j]*2.)
            response_derivs_lihu[i,j] = (sw_survey_lihu.len_pow.Cs_pert[j,0].cosmology[response_pars[i]]-sw_survey_lihu.len_pow.Cs_pert[j,1].cosmology[response_pars[i]])/(lihu_eps[j]*2.)
    assert np.allclose(response_derivs_jdem,response_derivs_jdem_pred)
    assert np.allclose(response_derivs_lihu,response_derivs_lihu_pred)

    project_jdem_to_lihu = np.zeros((lihu_pars.size,jdem_pars.size))
    project_lihu_to_jdem = np.zeros((jdem_pars.size,lihu_pars.size))
    for itr1 in range(0,lihu_pars.size):
        for itr2 in range(0,response_pars.size):
            if response_pars[itr2] in jdem_pars:
                name = response_pars[itr2]
                i = np.argwhere(jdem_pars==name)[0,0]
                project_jdem_to_lihu[itr1,i] = response_derivs_lihu[itr2,itr1]
    for itr1 in range(0,jdem_pars.size):
        for itr2 in range(0,response_pars.size):
            if response_pars[itr2] in lihu_pars:
                name = response_pars[itr2]
                i = np.argwhere(lihu_pars==name)[0,0]
                project_lihu_to_jdem[itr1,i] = response_derivs_jdem[itr2,itr1]
    #assert np.allclose(np.dot(dO_dpar_jdem,project_jdem_to_lihu.T),dO_dpar_lihu,rtol=1.e-3,atol=np.max(dO_dpar_lihu)*1.e-4)
    #assert np.allclose(np.dot(dO_dpar_lihu,project_lihu_to_jdem.T),dO_dpar_jdem,rtol=1.e-3,atol=np.max(dO_dpar_jdem)*1.e-4)

    #lihu p_space cannot currently do priors by itself
    f_p_priors_lihu = np.dot(project_jdem_to_lihu,np.dot(SS_jdem.multi_f.fisher_priors.get_fisher(),project_jdem_to_lihu.T))

    f_set_jdem_in = np.zeros(3,dtype=object)
    f_set_lihu_in = np.zeros(3,dtype=object)
    for i in range(0,3):
        f_set_jdem_in[i] = SS_jdem.f_set_nopriors[i][2].get_fisher().copy()
        f_set_lihu_in[i] = SS_lihu.f_set_nopriors[i][2].get_fisher().copy()

    f_np_lihu2 = rotate_jdem_to_lihu(f_set_jdem_in,C_fid_jdem)
    f_np_jdem2 = rotate_lihu_to_jdem(f_set_lihu_in,C_fid_lihu)
    f_np_lihu3 = rotate_jdem_to_lihu(f_np_jdem2,C_fid_jdem)
    f_np_jdem3 = rotate_lihu_to_jdem(f_np_lihu2,C_fid_lihu)

    for i in range(0,3):
        f_np_jdem = SS_jdem.f_set_nopriors[i][2].get_fisher().copy()
        f_np_lihu = SS_lihu.f_set_nopriors[i][2].get_fisher().copy()
        f_np_jdem_to_lihu = np.dot(project_jdem_to_lihu,np.dot(f_np_jdem,project_jdem_to_lihu.T))
        f_np_lihu_to_jdem = np.dot(project_lihu_to_jdem,np.dot(f_np_lihu,project_lihu_to_jdem.T))
        assert np.allclose(f_set_lihu_in[i],f_np_lihu3[i])
        assert np.allclose(f_set_jdem_in[i],f_np_jdem3[i])
        assert np.allclose(f_np_jdem_to_lihu,f_np_lihu2[i])
        assert np.allclose(f_np_lihu_to_jdem,f_np_jdem2[i])
        assert np.allclose(f_np_jdem_to_lihu,f_np_lihu,rtol=1.e-3)
        assert np.allclose(f_np_lihu_to_jdem,f_np_jdem,rtol=1.e-3)
        assert np.allclose(f_set_lihu_in[i],f_np_lihu2[i],rtol=1.e-3)
        assert np.allclose(f_set_jdem_in[i],f_np_jdem2[i],rtol=1.e-3)
        assert np.allclose(f_np_jdem3[i],f_np_jdem2[i],rtol=1.e-3)
        assert np.allclose(f_np_lihu3[i],f_np_lihu2[i],rtol=1.e-3)
        f_p_jdem = SS_jdem.f_set[i][2].get_fisher().copy()
        f_p_lihu = SS_lihu.f_set_nopriors[i][2].get_fisher().copy()+f_p_priors_lihu.copy()
        f_p_jdem_to_lihu = np.dot(project_jdem_to_lihu,np.dot(f_p_jdem,project_jdem_to_lihu.T))
        f_p_lihu_to_jdem = np.dot(project_lihu_to_jdem,np.dot(f_p_lihu,project_lihu_to_jdem.T))
        assert np.allclose(f_p_jdem_to_lihu,f_p_lihu,rtol=1.e-3)
        assert np.allclose(f_p_lihu_to_jdem,f_p_jdem,rtol=1.e-3)

def test_agreement_with_sigma8():
    """test sigma8 works basic to jdem"""
    cosmo_base = defaults.cosmology_wmap.copy()
    cosmo_base = cp.add_derived_pars(cosmo_base,'jdem')
    cosmo_base['de_model'] = 'constant_w'
    cosmo_base['w'] = -1.
    cosmo_base['sigma8'] = 0.7925070693605805
    power_params = defaults.power_params.copy()
    power_params.camb['maxkh'] = 3.
    power_params.camb['kmax'] = 10.
    power_params.camb['npoints'] = 1000
    power_params.camb['accuracy'] = 2
    power_params.camb['leave_h'] = False
    power_params_jdem = deepcopy(power_params)
    power_params_jdem.camb['force_sigma8'] = False
    power_params_basi = deepcopy(power_params)
    power_params_basi.camb['force_sigma8'] = True


    cosmo_jdem = cosmo_base.copy()
    cosmo_jdem['p_space'] = 'jdem'
    C_fid_jdem = cp.CosmoPie(cosmo_jdem,'jdem')
    P_jdem = mps.MatterPower(C_fid_jdem,power_params_jdem)
    C_fid_jdem.set_power(P_jdem)

    cosmo_basi = cosmo_base.copy()
    cosmo_basi['p_space'] = 'basic'
    C_fid_basi = cp.CosmoPie(cosmo_basi,'basic')
    P_basi = mps.MatterPower(C_fid_basi,power_params_basi)
    C_fid_basi.set_power(P_basi)


    zs = np.arange(0.2,1.41,0.40)
    z_fine = np.linspace(0.001,1.4,1000)

    geo_jdem = FullSkyGeo(zs,C_fid_jdem,z_fine)
    geo_basi = FullSkyGeo(zs,C_fid_basi,z_fine)

    jdem_pars = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w'])
    jdem_eps = np.array([0.002,0.00025,0.0001,0.00025,0.01,0.01])

    basi_pars = np.array(['ns','Omegamh2','Omegabh2','h','sigma8','w'])
    basi_eps = np.array([0.002,0.00025,0.0001,0.00025,0.001,0.01])

    sw_params = defaults.sw_survey_params.copy()
    len_params = defaults.lensing_params.copy()
    sw_observable_list = defaults.sw_observable_list.copy()
    nz_wfirst_lens = NZWFirstEff(defaults.nz_params_wfirst_lens.copy())
    prior_params = defaults.prior_fisher_params.copy()
    basis_params = defaults.basis_params.copy()

    sw_survey_jdem = sws.SWSurvey(geo_jdem,'wfirst',C_fid_jdem,sw_params,jdem_pars,jdem_eps,sw_observable_list,len_params,nz_wfirst_lens)

    sw_survey_basi = sws.SWSurvey(geo_basi,'wfirst',C_fid_basi,sw_params,basi_pars,basi_eps,sw_observable_list,len_params,nz_wfirst_lens)
    #need to fix As because the code cannot presently do this
    for itr in range(0,basi_pars.size):
        for i in range(0,2):
            cosmo_alt_basi = sw_survey_basi.len_pow.Cs_pert[itr,i].cosmology.copy()
            n_As = 10
            logAs = np.linspace(cosmo_alt_basi['LogAs']*0.9,cosmo_alt_basi['LogAs']*1.1,n_As)
            As = np.exp(logAs)
            sigma8s = np.zeros(n_As)
            for itr2 in xrange(0,n_As):
                cosmo_alt_basi['As'] = As[itr2]
                cosmo_alt_basi['LogAs'] = logAs[itr2]
                sigma8s[itr2] = camb_sigma8(cosmo_alt_basi,power_params_basi.camb)
            logAs_interp = InterpolatedUnivariateSpline(sigma8s[::-1],logAs[::-1],ext=2,k=3)

            sw_survey_basi.len_pow.Cs_pert[itr,i].cosmology['LogAs'] = logAs_interp(sw_survey_basi.len_pow.Cs_pert[itr,i].cosmology['sigma8'])
            sw_survey_basi.len_pow.Cs_pert[itr,i].cosmology['As'] = np.exp(sw_survey_basi.len_pow.Cs_pert[itr,i].cosmology['LogAs'])

    dO_dpar_jdem = sw_survey_jdem.get_dO_I_dpar_array()
    dO_dpar_basi = sw_survey_basi.get_dO_I_dpar_array()

    response_pars = np.array(['ns','Omegach2','Omegabh2','Omegamh2','OmegaLh2','h','LogAs','w','sigma8'])

    l_max = 24

    r_max_jdem = geo_jdem.r_fine[-1]
    k_cut_jdem = 30./r_max_jdem
    basis_jdem = SphBasisK(r_max_jdem,C_fid_jdem,k_cut_jdem,basis_params,l_ceil=l_max,needs_m=True)
    SS_jdem = SuperSurvey(np.array([sw_survey_jdem]),np.array([]),basis_jdem,C_fid_jdem,prior_params,get_a=False,do_unmitigated=True,do_mitigated=False)

    r_max_basi = geo_basi.r_fine[-1]
    k_cut_basi = 30./r_max_basi
    basis_basi = SphBasisK(r_max_basi,C_fid_basi,k_cut_basi,basis_params,l_ceil=l_max,needs_m=True)
    SS_basi = SuperSurvey(np.array([sw_survey_basi]),np.array([]),basis_basi,C_fid_basi,prior_params,get_a=False,do_unmitigated=True,do_mitigated=False)

    #dO_dpar_jdem_to_basi = np.zeros_like(dO_dpar_jdem)
    #dO_dpar_basi_to_jdem = np.zeros_like(dO_dpar_basi)

    project_basi_to_jdem = np.zeros((jdem_pars.size,basi_pars.size))

    response_derivs_jdem = np.zeros((response_pars.size,jdem_pars.size))
    response_derivs_basi = np.zeros((response_pars.size,basi_pars.size))
    for i in range(0,response_pars.size):
        for j in range(0,jdem_pars.size):
            response_derivs_jdem[i,j] = (sw_survey_jdem.len_pow.Cs_pert[j,0].cosmology[response_pars[i]]-sw_survey_jdem.len_pow.Cs_pert[j,1].cosmology[response_pars[i]])/(jdem_eps[j]*2.)
            response_derivs_basi[i,j] = (sw_survey_basi.len_pow.Cs_pert[j,0].cosmology[response_pars[i]]-sw_survey_basi.len_pow.Cs_pert[j,1].cosmology[response_pars[i]])/(basi_eps[j]*2.)

    project_jdem_to_basi = np.zeros((basi_pars.size,jdem_pars.size))
    project_basi_to_jdem = np.zeros((jdem_pars.size,basi_pars.size))
    for itr1 in range(0,basi_pars.size):
        for itr2 in range(0,response_pars.size):
            if response_pars[itr2] in jdem_pars:
                name = response_pars[itr2]
                i = np.argwhere(jdem_pars==name)[0,0]
                project_jdem_to_basi[itr1,i] = response_derivs_basi[itr2,itr1]
    for itr1 in range(0,jdem_pars.size):
        for itr2 in range(0,response_pars.size):
            if response_pars[itr2] in basi_pars:
                name = response_pars[itr2]
                i = np.argwhere(basi_pars==name)[0,0]
                project_basi_to_jdem[itr1,i] = response_derivs_jdem[itr2,itr1]
    assert np.allclose(np.dot(dO_dpar_jdem,project_jdem_to_basi.T),dO_dpar_basi,rtol=1.e-3,atol=np.max(dO_dpar_basi)*1.e-4)
    assert np.allclose(np.dot(dO_dpar_basi,project_basi_to_jdem.T),dO_dpar_jdem,rtol=1.e-3,atol=np.max(dO_dpar_jdem)*1.e-4)

    #basi p_space cannot currently do priors by itself
    f_p_priors_basi = np.dot(project_jdem_to_basi,np.dot(SS_jdem.multi_f.fisher_priors.get_fisher(),project_jdem_to_basi.T))

    for i in range(0,1):
        f_np_jdem = SS_jdem.f_set_nopriors[i][2].get_fisher().copy()
        f_np_basi = SS_basi.f_set_nopriors[i][2].get_fisher().copy()
        f_np_jdem_to_basi = np.dot(project_jdem_to_basi,np.dot(f_np_jdem,project_jdem_to_basi.T))
        f_np_basi_to_jdem = np.dot(project_basi_to_jdem,np.dot(f_np_basi,project_basi_to_jdem.T))
        assert np.allclose(f_np_jdem_to_basi,f_np_basi,rtol=1.e-2)
        assert np.allclose(f_np_basi_to_jdem,f_np_jdem,rtol=1.e-2)

        f_p_jdem = SS_jdem.f_set[i][2].get_fisher().copy()
        f_p_basi = SS_basi.f_set_nopriors[i][2].get_fisher().copy()+f_p_priors_basi.copy()
        f_p_jdem_to_basi = np.dot(project_jdem_to_basi,np.dot(f_p_jdem,project_jdem_to_basi.T))
        f_p_basi_to_jdem = np.dot(project_basi_to_jdem,np.dot(f_p_basi,project_basi_to_jdem.T))
        assert np.allclose(f_p_jdem_to_basi,f_p_basi,rtol=1.e-2)
        assert np.allclose(f_p_basi_to_jdem,f_p_jdem,rtol=1.e-2)
    print(f_np_jdem/f_np_basi_to_jdem)
    print(f_np_basi/f_np_jdem_to_basi)

def test_power_agreement():
    """test agreement of powers extracted in two different cosmological parametrizations"""
    cosmo_base = defaults.cosmology_wmap.copy()
    cosmo_base = cp.add_derived_pars(cosmo_base,'jdem')
    cosmo_base['de_model'] = 'constant_w'
    cosmo_base['w'] = -1.
    power_params = defaults.power_params.copy()
    power_params.camb['maxkh'] = 3.
    power_params.camb['kmax'] = 10.
    power_params.camb['npoints'] = 1000
    power_params.camb['accuracy'] = 2
    power_params.camb['leave_h'] = False

    cosmo_jdem = cosmo_base.copy()
    cosmo_jdem['p_space'] = 'jdem'
    C_fid_jdem = cp.CosmoPie(cosmo_jdem,'jdem')
    P_jdem = mps.MatterPower(C_fid_jdem,power_params.copy())
    C_fid_jdem.set_power(P_jdem)

    cosmo_lihu = cosmo_base.copy()
    cosmo_lihu['p_space'] = 'lihu'
    C_fid_lihu = cp.CosmoPie(cosmo_lihu,'lihu')
    P_lihu = mps.MatterPower(C_fid_lihu,power_params.copy())
    C_fid_lihu.set_power(P_lihu)

    jdem_pars = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs'])
    jdem_eps = np.array([0.002,0.00025,0.0001,0.00025,0.1])
    C_pert_jdem = ppr.get_perturbed_cosmopies(C_fid_jdem,jdem_pars,jdem_eps)

    lihu_pars = np.array(['ns','Omegach2','Omegabh2','h','LogAs'])
    lihu_eps = np.array([0.002,0.00025,0.0001,0.00025,0.1])
    C_pert_lihu = ppr.get_perturbed_cosmopies(C_fid_lihu,lihu_pars,lihu_eps)

    response_pars = np.array(['Omegach2','Omegabh2','Omegamh2','OmegaLh2','h'])
    response_derivs_jdem = np.zeros((response_pars.size,3))
    response_derivs_jdem_pred = np.array([[1.,0.,1.,0.,1./(2.*C_fid_jdem.cosmology['h'])],[-1.,1.,0.,0.,0.],[0.,0.,0.,1.,1./(2.*C_fid_jdem.cosmology['h'])]]).T
    response_derivs_lihu = np.zeros((response_pars.size,3))
    response_derivs_lihu_pred = np.array([[1.,0.,1.,-1.,0.],[0.,1.,1.,-1.,0.],[0.,0.,0.,2.*C_fid_lihu.cosmology['h'],1.]]).T
    for i in range(0,response_pars.size):
        for j in range(1,4):
            response_derivs_jdem[i,j-1] = (C_pert_jdem[j,0].cosmology[response_pars[i]]-C_pert_jdem[j,1].cosmology[response_pars[i]])/(jdem_eps[j]*2.)
            response_derivs_lihu[i,j-1] = (C_pert_lihu[j,0].cosmology[response_pars[i]]-C_pert_lihu[j,1].cosmology[response_pars[i]])/(lihu_eps[j]*2.)
    assert np.allclose(response_derivs_jdem_pred,response_derivs_jdem)
    assert np.allclose(response_derivs_lihu_pred,response_derivs_lihu)

    power_derivs_jdem = np.zeros((3,C_fid_jdem.k.size))
    power_derivs_lihu = np.zeros((3,C_fid_lihu.k.size))

    for pmodel in ['linear','fastpt','halofit']:
        for j in range(1,4):
            power_derivs_jdem[j-1] = (C_pert_jdem[j,0].P_lin.get_matter_power([0.],pmodel=pmodel)[:,0]-C_pert_jdem[j,1].P_lin.get_matter_power([0.],pmodel=pmodel)[:,0])/(jdem_eps[j]*2.)
            power_derivs_lihu[j-1] = (C_pert_lihu[j,0].P_lin.get_matter_power([0.],pmodel=pmodel)[:,0]-C_pert_lihu[j,1].P_lin.get_matter_power([0.],pmodel=pmodel)[:,0])/(lihu_eps[j]*2.)

        assert np.allclose((power_derivs_jdem[1]+power_derivs_jdem[0]-power_derivs_jdem[2]),power_derivs_lihu[1],rtol=1.e-2,atol=1.e-4*np.max(np.abs(power_derivs_lihu[1])))
        assert np.allclose((power_derivs_jdem[0]-power_derivs_jdem[2]),power_derivs_lihu[0],rtol=1.e-2,atol=1.e-4*np.max(np.abs(power_derivs_lihu[0])))
        assert np.allclose(power_derivs_jdem[2]*2*C_fid_lihu.cosmology['h'],power_derivs_lihu[2],rtol=1.e-2,atol=1.e-4*np.max(np.abs(power_derivs_lihu[2])))



#if __name__=='__main__':
#    """test sigma8 works basic to jdem"""
#    cosmo_base = {   'Omegabh2':0.02223,
#                    'Omegach2':0.1153,
#                    'Omegab'  :0.04283392714316876,
#                    'Omegac'  :0.22216607285683124,
#                    'Omegamh2':0.13752999999999999,
#                    'OmegaL'  :0.735,
#                    'OmegaLh2':0.38145113207547166,
#                    'Omegam'  :0.265,
#                    'H0'      :72.04034509047493,
#                    'sigma8'  : 0.8269877678406697, #from the code
#                    'h'       :0.7204034509047493,
#                    'Omegak'  : 0.0,
#                    'Omegakh2': 0.0,
#                    'Omegar'  : 0.0,
#                    'Omegarh2': 0.0,
#                    'ns'      : 0.9608,
#                    'tau'     : 0.081,
#                    'Yp'      :0.299,
#                    'As'      : 2.464*10**-9,
#                    'LogAs'   :-19.821479791275138,
#                    'w'       :-1.0,
#                    'de_model':'constant_w',#dark energy model
#                    'mnu'     :0.}
#    cosmo_base = cp.add_derived_pars(cosmo_base,'jdem')
#    cosmo_base['de_model'] = 'constant_w'
#    cosmo_base['w'] = -1.
#    cosmo_base['sigma8'] = 0.880798667856577
#    power_params = defaults.power_params.copy()
#    power_params.camb['maxkh'] = 3.
#    power_params.camb['kmax'] = 10.
#    power_params.camb['npoints'] = 1000
#    power_params.camb['accuracy'] = 2
#    power_params.camb['leave_h'] = False
#    power_params_jdem = deepcopy(power_params)
#    power_params_jdem.camb['force_sigma8'] = False
#    power_params_basi = deepcopy(power_params)
#    power_params_basi.camb['force_sigma8'] = True
#
#
#    cosmo_jdem = cosmo_base.copy()
#    cosmo_jdem['p_space'] = 'jdem'
#    C_fid_jdem = cp.CosmoPie(cosmo_jdem,'jdem')
#    P_jdem = mps.MatterPower(C_fid_jdem,power_params_jdem)
#    C_fid_jdem.set_power(P_jdem)
#
#    cosmo_basi = cosmo_base.copy()
#    cosmo_basi['p_space'] = 'basic'
#    C_fid_basi = cp.CosmoPie(cosmo_basi,'basic')
#    P_basi = mps.MatterPower(C_fid_basi,power_params_basi)
#    C_fid_basi.set_power(P_basi)
#
#
#    zs = np.arange(0.2,1.41,0.40)
#    z_fine = np.linspace(0.001,1.4,1000)
#
#    geo_jdem = FullSkyGeo(zs,C_fid_jdem,z_fine)
#    geo_basi = FullSkyGeo(zs,C_fid_basi,z_fine)
#
#    jdem_pars = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w'])
#    jdem_eps = np.array([0.002,0.00025,0.0001,0.00025,0.01,0.01])
#
#    basi_pars = np.array(['ns','Omegamh2','Omegabh2','h','sigma8','w'])
#    basi_eps = np.array([0.002,0.00025,0.0001,0.00025,0.001,0.01])
#
#    sw_params = defaults.sw_survey_params.copy()
#    len_params = defaults.lensing_params.copy()
#    sw_observable_list = defaults.sw_observable_list.copy()
#    nz_wfirst_lens = NZWFirstEff(defaults.nz_params_wfirst_lens.copy())
#    prior_params = defaults.prior_fisher_params.copy()
#    basis_params = defaults.basis_params.copy()
#
#    sw_survey_jdem = sws.SWSurvey(geo_jdem,'wfirst',C_fid_jdem,sw_params,jdem_pars,jdem_eps,sw_observable_list,len_params,nz_wfirst_lens)
#
#    sw_survey_basi = sws.SWSurvey(geo_basi,'wfirst',C_fid_basi,sw_params,basi_pars,basi_eps,sw_observable_list,len_params,nz_wfirst_lens)
#    #need to fix As because the code cannot presently do this
#    for itr in range(0,basi_pars.size):
#        for i in range(0,2):
#            cosmo_alt_basi = sw_survey_basi.len_pow.Cs_pert[itr,i].cosmology.copy()
#            n_As = 10
#            logAs = np.linspace(cosmo_alt_basi['LogAs']*0.9,cosmo_alt_basi['LogAs']*1.1,n_As)
#            As = np.exp(logAs)
#            sigma8s = np.zeros(n_As)
#            for itr2 in xrange(0,n_As):
#                cosmo_alt_basi['As'] = As[itr2]
#                cosmo_alt_basi['LogAs'] = logAs[itr2]
#                sigma8s[itr2] = camb_sigma8(cosmo_alt_basi,power_params_basi.camb)
#            logAs_interp = InterpolatedUnivariateSpline(sigma8s[::-1],logAs[::-1],ext=2,k=3)
#
#            sw_survey_basi.len_pow.Cs_pert[itr,i].cosmology['LogAs'] = logAs_interp(sw_survey_basi.len_pow.Cs_pert[itr,i].cosmology['sigma8'])
#            sw_survey_basi.len_pow.Cs_pert[itr,i].cosmology['As'] = np.exp(sw_survey_basi.len_pow.Cs_pert[itr,i].cosmology['LogAs'])
#
#    dO_dpar_jdem = sw_survey_jdem.get_dO_I_dpar_array()
#    dO_dpar_basi = sw_survey_basi.get_dO_I_dpar_array()
#
#    response_pars = np.array(['ns','Omegach2','Omegabh2','Omegamh2','OmegaLh2','h','LogAs','w','sigma8'])
#
#    l_max = 24
#
#    r_max_jdem = geo_jdem.r_fine[-1]
#    k_cut_jdem = 30./r_max_jdem
#    basis_jdem = SphBasisK(r_max_jdem,C_fid_jdem,k_cut_jdem,basis_params,l_ceil=l_max,needs_m=True)
#    SS_jdem = SuperSurvey(np.array([sw_survey_jdem]),np.array([]),basis_jdem,C_fid_jdem,prior_params,get_a=False,do_unmitigated=True,do_mitigated=False)
#
#    r_max_basi = geo_basi.r_fine[-1]
#    k_cut_basi = 30./r_max_basi
#    basis_basi = SphBasisK(r_max_basi,C_fid_basi,k_cut_basi,basis_params,l_ceil=l_max,needs_m=True)
#    SS_basi = SuperSurvey(np.array([sw_survey_basi]),np.array([]),basis_basi,C_fid_basi,prior_params,get_a=False,do_unmitigated=True,do_mitigated=False)
#
#    dO_dpar_jdem_to_basi = np.zeros_like(dO_dpar_jdem)
#    dO_dpar_basi_to_jdem = np.zeros_like(dO_dpar_basi)
#
#    project_basi_to_jdem = np.zeros((jdem_pars.size,basi_pars.size))
#
#    response_derivs_jdem = np.zeros((response_pars.size,jdem_pars.size))
#    response_derivs_basi = np.zeros((response_pars.size,basi_pars.size))
#    for i in range(0,response_pars.size):
#        for j in range(0,jdem_pars.size):
#            response_derivs_jdem[i,j] = (sw_survey_jdem.len_pow.Cs_pert[j,0].cosmology[response_pars[i]]-sw_survey_jdem.len_pow.Cs_pert[j,1].cosmology[response_pars[i]])/(jdem_eps[j]*2.)
#            response_derivs_basi[i,j] = (sw_survey_basi.len_pow.Cs_pert[j,0].cosmology[response_pars[i]]-sw_survey_basi.len_pow.Cs_pert[j,1].cosmology[response_pars[i]])/(basi_eps[j]*2.)
#
#    project_jdem_to_basi = np.zeros((basi_pars.size,jdem_pars.size))
#    project_basi_to_jdem = np.zeros((jdem_pars.size,basi_pars.size))
#    for itr1 in range(0,basi_pars.size):
#        for itr2 in range(0,response_pars.size):
#            if response_pars[itr2] in jdem_pars:
#                name = response_pars[itr2]
#                i = np.argwhere(jdem_pars==name)[0,0]
#                project_jdem_to_basi[itr1,i] = response_derivs_basi[itr2,itr1]
#    for itr1 in range(0,jdem_pars.size):
#        for itr2 in range(0,response_pars.size):
#            if response_pars[itr2] in basi_pars:
#                name = response_pars[itr2]
#                i = np.argwhere(basi_pars==name)[0,0]
#                project_basi_to_jdem[itr1,i] = response_derivs_jdem[itr2,itr1]
#    assert np.allclose(np.dot(dO_dpar_jdem,project_jdem_to_basi.T),dO_dpar_basi,rtol=1.e-3,atol=np.max(dO_dpar_basi)*1.e-4)
#    assert np.allclose(np.dot(dO_dpar_basi,project_basi_to_jdem.T),dO_dpar_jdem,rtol=1.e-3,atol=np.max(dO_dpar_jdem)*1.e-4)
#
#    #basi p_space cannot currently do priors by itself
#    f_p_priors_basi = np.dot(project_jdem_to_basi,np.dot(SS_jdem.multi_f.fisher_priors.get_fisher(),project_jdem_to_basi.T))
#
#    for i in range(0,1):
#        f_np_jdem = SS_jdem.f_set_nopriors[i][2].get_fisher().copy()
#        f_np_basi = SS_basi.f_set_nopriors[i][2].get_fisher().copy()
#        f_np_jdem_to_basi = np.dot(project_jdem_to_basi,np.dot(f_np_jdem,project_jdem_to_basi.T))
#        f_np_basi_to_jdem = np.dot(project_basi_to_jdem,np.dot(f_np_basi,project_basi_to_jdem.T))
#        #assert np.allclose(f_np_jdem_to_basi,f_np_basi,rtol=1.e-2)
#        #assert np.allclose(f_np_basi_to_jdem,f_np_jdem,rtol=1.e-2)
#
#        f_p_jdem = SS_jdem.f_set[i][2].get_fisher().copy()
#        f_p_basi = SS_basi.f_set_nopriors[i][2].get_fisher().copy()+f_p_priors_basi.copy()
#        f_p_jdem_to_basi = np.dot(project_jdem_to_basi,np.dot(f_p_jdem,project_jdem_to_basi.T))
#        f_p_basi_to_jdem = np.dot(project_basi_to_jdem,np.dot(f_p_basi,project_basi_to_jdem.T))
#        #assert np.allclose(f_p_jdem_to_basi,f_p_basi,rtol=1.e-2)
#        #assert np.allclose(f_p_basi_to_jdem,f_p_jdem,rtol=1.e-2)
#    print(f_np_jdem/f_np_basi_to_jdem)
#    print(f_np_basi/f_np_jdem_to_basi)
if __name__=='__main__':
    pytest.cmdline.main(['cosmo_response_tests.py'])
