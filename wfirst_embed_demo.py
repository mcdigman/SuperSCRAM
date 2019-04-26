"""Mathtew Digman 2019 
Demonstration case for SUPERSCRAM code with wfirst embeded in lsst footprint 
to mitigate covariance as shown in paper"""
from __future__ import division,print_function,absolute_import
from builtins import range
from time import time

import numpy as np

import matplotlib.pyplot as plt

from super_survey import SuperSurvey,make_standard_ellipse_plot
from lw_survey import LWSurvey
from sw_survey import SWSurvey
from cosmopie import CosmoPie
import cosmopie as cp
from sph_klim import SphBasisK
from matter_power_spectrum import MatterPower
from nz_wfirst_eff import NZWFirstEff
from premade_geos import WFIRSTGeo,LSSTGeo,WFIRSTPixelGeo,LSSTPixelGeo

import defaults


if __name__=='__main__':
    print("main: begin WFIRST")
    time0 = time()
    #get dictionaries of parameters that various functions will need
    cosmo = defaults.cosmology_wmap.copy()
    cosmo['de_model'] = 'w0wa'
    cosmo['wa'] = 0.
    cosmo['w0'] = -1.
    cosmo['w'] = -1.
    p_space = 'jdem'

    if cosmo['de_model']=='jdem':
        for i in range(0,36):
            cosmo['ws36_'+str(i).zfill(2)] = cosmo['w']

    camb_params = defaults.camb_params.copy()
    camb_params['force_sigma8'] = False
    camb_params['maxkh'] = 200.
    camb_params['kmax'] = 30.
    camb_params['npoints'] = 4000
    camb_params['pivot_scalar'] = 0.05
    poly_params = {'n_double':80}
    len_params = defaults.lensing_params.copy()
    len_params['l_max'] = 5000
    len_params['l_min'] = 30
    len_params['n_l'] = 20
    nz_params_wfirst_lens = defaults.nz_params_wfirst_lens.copy()
    sw_params = defaults.sw_survey_params.copy()
    lw_params = defaults.lw_survey_params.copy()
    sw_observable_list = defaults.sw_observable_list.copy()
    lw_observable_list = defaults.lw_observable_list.copy()
    mf_params = defaults.hmf_params.copy()
    mf_params['n_grid'] = 2000
    mf_params['log10_min_mass'] = 10.
    n_params_lsst = defaults.nz_params_lsst_use.copy()
    n_params_lsst['i_cut'] = 24.1 #gold standard subset of LSST 1 year (10 year 25.3)

    power_params = defaults.power_params.copy()
    power_params.camb = camb_params
    power_params.camb['accuracy'] = 1
    power_params.matter_power['w_step'] = 0.2
    power_params.matter_power['a_step'] = 0.05
    power_params.wmatcher['a_step'] = 0.0001
    power_params.wmatcher['w_step'] = 0.001
    prior_params = defaults.prior_fisher_params.copy()
    dn_params = defaults.dn_params.copy()
    dn_params['nz_select'] = 'LSST'
    dn_params['sigma0'] = 0.1
    lw_param_list = np.array([{'dn_params':dn_params,'n_params':n_params_lsst,'mf_params':mf_params}])
    basis_params = defaults.basis_params.copy()
    basis_params['n_bessel_oversample'] = 400000#*10

    #creat a CosmoPie object to manage the cosmology details
    print("main: begin constructing CosmoPie")
    C = CosmoPie(cosmo,p_space=p_space)
    print("main: finish constructing CosmoPie")

    #get the matter power spectrum and give it to the CosmoPie
    print("main: begin constructing MatterPower")
    P = MatterPower(C,power_params)
    print("main: finish constructing MatterPower")
    C.set_power(P)

    #create the WFIRST geometry
    #zs are the bounding redshifts of the tomographic bins
    zs = np.arange(0.2,3.01,0.4)
    zs_lsst = np.linspace(0.,1.2,3)
    #z_fine are the resolution redshift slices to be integrated over
    z_fine = np.linspace(0.001,np.max([zs[-1],zs_lsst[-1]]),500)

    #l_max is the highest l that should be precomputed
    l_max = 72
    res_healpix = 6
    use_pixels = False
    print("main: begin constructing WFIRST PolygonGeo")
    if use_pixels:
        geo_wfirst = WFIRSTPixelGeo(zs,C,z_fine,l_max,res_healpix)
    else:
        geo_wfirst = WFIRSTGeo(zs,C,z_fine,l_max,poly_params)

    print("main: finish constructing WFIRST PolygonGeo")

    #create the LSST geometry, for our purposes, a 20000 square degree survey
    #encompassing the wfirst survey with galactic plane masked)
    print("main: begin constructing LSST PolygonGeo")
    if use_pixels:
        geo_lsst = LSSTPixelGeo(zs_lsst,C,z_fine,l_max,res_healpix)
    else:
        geo_lsst = LSSTGeo(zs_lsst,C,z_fine,l_max,poly_params)
    print("main: finish constructing LSST PolygonGeo")

    #create the short wavelength survey (SWSurvey) object
    #list of comsological parameters that will need to be varied
    if cosmo['de_model']=='w0wa':
        cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w0','wa'])
        cosmo_par_eps = np.array([0.002,0.00025,0.0001,0.00025,0.1,0.01,0.035])
    elif cosmo['de_model']=='constant_w':
        cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w'])
        cosmo_par_eps = np.array([0.002,0.00025,0.0001,0.00025,0.1,0.01])
    elif cosmo['de_model']=='jdem':
        cosmo_par_list = ['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs']
        cosmo_par_list.extend(cp.JDEM_LIST)
        cosmo_par_list = np.array(cosmo_par_list,dtype=object)
        cosmo_par_eps = np.full(41,0.5)
        cosmo_par_eps[0:5] = np.array([0.002,0.00025,0.0001,0.00025,0.1])
    else:
        raise ValueError('unrecognized de_model '+str(cosmo['de_model']))

    #create the number density of lensing sources
    print("main: begin constructing lensing source density for WFIRST")
    nz_params_wfirst_lens['i_cut'] = 26.3
    nz_params_wfirst_lens['data_source'] = './data/CANDELS-GOODSS2.dat'
    nz_wfirst_lens = NZWFirstEff(nz_params_wfirst_lens)
    print("main: finish constructing lensing source density for WFIRST")
    #set some parameters for lensing
    len_params['smodel'] = 'nzmatcher'
    len_params['pmodel'] = 'halofit'

    #create the lw basis
    #z_max is the maximum radial extent of the basis
    z_max = z_fine[-1]+0.001
    r_max = C.D_comov(z_max)
    #k_cut is the maximum k value for the bessel function zeros that define the basis
    x_cut = 80.
    k_cut = x_cut/r_max
    #l_max caps maximum l regardless of k
    print("main: begin constructing basis for long wavelength fluctuations")
    basis = SphBasisK(r_max,C,k_cut,basis_params,l_ceil=l_max,needs_m=True)
    print("main: finish constructing basis for long wavelength fluctuations")

    #create the lw survey
    geos = np.array([geo_wfirst,geo_lsst],dtype=object)
    print("main: begin constructing LWSurvey for mitigation")
    survey_lw = LWSurvey(geos,'combined_survey',basis,C,lw_params,observable_list=lw_observable_list,param_list=lw_param_list)
    print("main: finish constructing LWSurvey for mitigation")
    #surveys_lw = np.array([])
    surveys_lw = np.array([survey_lw])

    #create the actual sw survey
    print("main: begin constructing SWSurvey for wfirst")
    sw_survey_wfirst = SWSurvey(geo_wfirst,'wfirst',C,sw_params,cosmo_par_list,cosmo_par_eps,sw_observable_list,len_params,nz_wfirst_lens)
    print("main: finish constructing SWSurvey for wfirst")
    surveys_sw = np.array([sw_survey_wfirst])

    #create the SuperSurvey with mitigation
    print("main: begin constructing SuperSurvey")
    SS = SuperSurvey(surveys_sw,surveys_lw,basis,C,prior_params,get_a=False,do_unmitigated=True,do_mitigated=True)

    time1 = time()
    print("main: finished construction tasks in "+str(time1-time0)+" s")

    SS.print_standard_analysis()

f_set = SS.f_set

fig = make_standard_ellipse_plot(f_set,cosmo_par_list)
plt.show(fig)
