"""demonstration case for wfirst embeded in lsst footprint to mitigate covariance"""
from __future__ import division,print_function,absolute_import
from builtins import range
from time import time

import numpy as np

from super_survey import SuperSurvey,make_ellipse_plot
from lw_survey import LWSurvey
from sw_survey import SWSurvey
from cosmopie import CosmoPie
from sph_klim import SphBasisK
from matter_power_spectrum import MatterPower
from nz_wfirst_eff import NZWFirstEff
from nz_wfirst import NZWFirst
from premade_geos import WFIRSTGeo,LSSTGeo

import defaults


if __name__=='__main__':
    print("main: begin WFIRST")
    time0 = time()
    #get dictionaries of parameters that various functions will need
    cosmo = defaults.cosmology.copy()
    cosmo['de_model'] = 'w0wa'
    cosmo['wa'] = 0.
    cosmo['w0'] = -1.
    cosmo['w'] = -1.
    p_space = 'jdem'
    camb_params = defaults.camb_params.copy()
    camb_params['force_sigma8'] = False
    camb_params['maxkh'] = 100.
    camb_params['kmax'] = 30.
    camb_params['npoints'] = 2000
    #fpt_params = defaults.fpt_params.copy()
    #wmatcher_params = defaults.wmatcher_params.copy()
    #halofit_params = defaults.halofit_params.copy()
    #matter_power_params = defaults.matter_power_params.copy()
    #hf_params = defaults.halofit_params.copy()
    poly_params = {'n_double':80}
    len_params = defaults.lensing_params.copy()
    len_params['l_max'] = 5000
    len_params['l_min'] = 30
    len_params['n_l'] = 20
    nz_params_wfirst_lens = defaults.nz_params_wfirst_lens.copy()
    nz_params_wfirst_lens['smooth_sigma'] = 0.1
    nz_params_wfirst_lens['n_right_extend'] = 5
    sw_params = defaults.sw_survey_params.copy()
    lw_params = defaults.lw_survey_params.copy()
    sw_observable_list = defaults.sw_observable_list
    #TODO don't use defaults for setting up the core demo
    lw_observable_list = defaults.lw_observable_list
    mf_params = defaults.hmf_params.copy()
    mf_params['n_grid'] = 2000
    mf_params['log10_min_mass'] = 10.
    n_params_wfirst = defaults.nz_params_wfirst_gal.copy()
    n_params_wfirst['z_resolution'] = 0.0001
    n_params_wfirst['nz_constant'] = 10**-3*cosmo['h']**3
    n_params_wfirst['n_right_extend'] = 5
    n_params_wfirst['smooth_sigma'] = 0.1
    n_params_wfirst['i_cut'] = 24.1 #gold standard subset of LSST 1 year (10 year 25.3)

    power_params = defaults.power_params.copy()
    power_params.camb = camb_params
    power_params.camb['accuracy'] = 1
    prior_params = defaults.prior_fisher_params.copy()
    lw_param_list = np.array([{'dn_params':{'nz_select':'LSST'},'n_params':n_params_wfirst,'mf_params':mf_params}])
    basis_params = defaults.basis_params.copy()

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
    #zs = np.array([0.2,0.43,.63,0.9, 1.3])
    #TODO check no off by one errors in final bin
    zs = np.arange(0.2,3.01,0.2)
    zs_lsst = np.linspace(0.,1.2,5)
    #zs = np.array([0.2,0.4,0.6])
    #z_fine are the resolution redshift slices to be integrated over
    z_fine = np.linspace(0.001,np.max(zs),2000)

    #z_fine[0] = 0.0001

    #l_max is the highest l that should be precomputed
    l_max = 84

    print("main: begin constructing WFIRST PolygonGeo")
    geo_wfirst = WFIRSTGeo(zs,C,z_fine,l_max,poly_params)
    print("main: finish constructing WFIRST PolygonGeo")

    #create the LSST geometry, for our purposes, a 20000 square degree survey 
    #encompassing the wfirst survey with galactic plane masked)
    print("main: begin constructing LSST PolygonGeo")
    geo_lsst = LSSTGeo(zs_lsst,C,z_fine,l_max,poly_params)#,phi1=2.9030540874480577)
    print("main: finish constructing LSST PolygonGeo")

    #create the short wavelength survey (SWSurvey) object
    #list of comsological parameters that will need to be varied
    if cosmo['de_model']=='w0wa':
        #TODO PRIORITY if using priors enforce cosmo_par_list ordered correctly, handle priors correcly
        cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w0','wa'])
        cosmo_par_eps = np.array([0.002,0.0005,0.0001,0.0005,0.1,0.01,0.07])
    elif cosmo['de_model']=='constant_w':
        cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w'])
        cosmo_par_eps = np.array([0.002,0.0005,0.0001,0.0005,0.1,0.01])
    else:
        raise ValueError('unrecognized de_model '+str(cosmo['de_model']))

    #create the number density of lensing sources
    print("main: begin constructing lensing source density for WFIRST")
    nz_wfirst_lens = NZWFirstEff(nz_params_wfirst_lens)
    #nz_wfirst_lens = NZWFirst(nz_params_wfirst_lens)
    print("main: finish constructing lensing source density for WFIRST")
    #set some parameters for lensing
    len_params['smodel'] = 'nzmatcher'
    #len_params['z_min_dist'] = np.min(zs)
    #len_params['z_max_dist'] = np.max(zs)
    len_params['pmodel'] = 'halofit'

    #create the lw basis
    #z_max is the maximum radial extent of the basis
    z_max = zs[-1]+0.001
    r_max = C.D_comov(z_max)
    #k_cut is the maximum k value for the bessel function zeros that define the basis
    #x_cut = 80.
    #x_cut = 100.
    x_cut = 85
    #x_cut = 30.
    k_cut = x_cut/r_max
    #l_max caps maximum l regardless of k
    print("main: begin constructing basis for long wavelength fluctuations")
    basis = SphBasisK(r_max,C,k_cut,basis_params,l_ceil=l_max)
    print("main: finish constructing basis for long wavelength fluctuations")

    #create the actual sw survey
    print("main: begin constructing SWSurvey for wfirst")
    sw_survey_wfirst = SWSurvey(geo_wfirst,'wfirst',C,sw_params,cosmo_par_list,cosmo_par_eps,sw_observable_list,len_params,None,nz_wfirst_lens)
    print("main: finish constructing SWSurvey for wfirst")
    surveys_sw = np.array([sw_survey_wfirst])


    #create the lw survey
    geos = np.array([geo_wfirst,geo_lsst],dtype=object)
    print("main: begin constructing LWSurvey for mitigation")
    survey_lw = LWSurvey(geos,'combined_survey',basis,C,lw_params,observable_list=lw_observable_list,param_list=lw_param_list)
    print("main: finish constructing LWSurvey for mitigation")
    surveys_lw = np.array([survey_lw])

    #create the SuperSurvey with mitigation
    print("main: begin constructing SuperSurvey")
    SS = SuperSurvey(surveys_sw,surveys_lw,basis,C,prior_params,get_a=False,do_unmitigated=True,do_mitigated=True)

    time1 = time()
    print("main: finished construction tasks in "+str(time1-time0)+"s")

    mit_eigs_par = SS.eig_set[1,1]
    no_mit_eigs_par = SS.eig_set[1,0]
    print("main: unmitigated parameter lambda1,2: "+str(no_mit_eigs_par[0][-1])+","+str(no_mit_eigs_par[0][-2]))
    print("main: mitigated parameter lambda1,2: "+str(mit_eigs_par[0][-1])+","+str(mit_eigs_par[0][-2]))

    #needed to make ellipse plot
    no_mit_color = np.array([1.,0.,0.])
    mit_color = np.array([0.,1.,0.])
    g_color = np.array([0.,0.,1.])
    color_set = np.array([mit_color,no_mit_color,g_color])
    opacity_set = np.array([1.0,1.0,1.0])
    box_widths = np.array([0.015,0.005,0.0005,0.005,0.1,0.05])
    dchi2 = 2.3
    #cov_set = np.array([SS.covs_params[1],SS.covs_params[0],SS.covs_g_pars[0]])
    cov_set = np.array([SS.f_set[2][2].get_covar(),SS.f_set[1][2].get_covar(),SS.f_set[0][2].get_covar()])
    label_set = np.array(["ssc+mit+g","ssc+g","g"])

    SS.print_standard_analysis()
    #make the ellipse plot
    print('\a')
    SS.make_standard_ellipse_plot()

do_c_rots = True
if do_c_rots:
#cov_g = SS.f_set_nopriors[0][2].get_covar()
    cov_g_inv = SS.f_set_nopriors[0][2].get_fisher()
    chol_g = SS.f_set_nopriors[0][2].get_cov_cholesky()
#chol_g_inv = SS.f_set_nopriors[0][2].get_cov_cholesky_inv()
    u_no_mit = SS.eig_set[1][0][1]
    v_no_mit = np.dot(chol_g,u_no_mit)
    of_no_mit = np.dot(cov_g_inv,v_no_mit)
#u_mit = SS.eig_set[1][1][1]
#v_mit = np.dot(chol_g,u_mit)
#of_mit = np.dot(cov_g_inv,v_mit)

    c_rot_eig_no_mit = np.dot(of_no_mit.T,np.dot(SS.f_set_nopriors[1][2].get_covar(),of_no_mit))
#c_rot_eig_no_mit = (c_rot_eig_no_mit+c_rot_eig_no_mit.T)/2.
#f_rot_eig_no_mit = np.linalg.inv(c_rot_eig_no_mit)
#f_rot_eig_no_mit = (f_rot_eig_no_mit+f_rot_eig_no_mit.T)/2.

    c_rot_eig_mit = np.dot(of_no_mit.T,np.dot(SS.f_set_nopriors[2][2].get_covar(),of_no_mit))
#c_rot_eig_mit = (c_rot_eig_mit+c_rot_eig_mit.T)/2.
#f_rot_eig_mit = np.linalg.inv(c_rot_eig_mit)
#f_rot_eig_mit = (f_rot_eig_mit+f_rot_eig_mit.T)/2.

    c_rot_eig_g = np.dot(of_no_mit.T,np.dot(SS.f_set_nopriors[0][2].get_covar(),of_no_mit))
#c_rot_eig_g = (c_rot_eig_g+c_rot_eig_g.T)/2.
#f_rot_eig_g = np.linalg.inv(c_rot_eig_g)
#f_rot_eig_g = (f_rot_eig_g+f_rot_eig_g.T)/2.

#assert np.allclose(np.diagonal(c_rot_eig_no_mit/c_rot_eig_g),SS.eig_set[1][0][0])
#assert np.allclose(np.diagonal(c_rot_eig_mit/c_rot_eig_g),SS.eig_set[1][1][0])

#f_rot_eig_mit2 = np.dot(rot_of_mit,np.dot(SS.f_set[1][2].get_fisher(),rot_of_mit.T))
#f_rot_eig_mit2 = (f_rot_eig_mit2+f_rot_eig_mit2.T)/2.
#c_rot_eig_mit2 = np.linalg.inv(f_rot_eig_mit2)
#c_rot_eig_mit2 = (c_rot_eig_mit2+c_rot_eig_mit2.T)/2.

#m_mat = np.dot(chol_g_inv,np.dot(SS.f_set[1][2].get_fisher(),chol_g_inv.T))f
    opacities = np.array([1.,1.,1.])
    colors = np.array([[0,1,0],[1,0,0],[0,0,1]]) 
    pnames = np.array(['p7','p6','p5','p4','p3','p2','p1'])
    names = np.array(['g','no mit','mit'])
    boxes = np.array([5.,5.,5.,5.,5.,5.,5.,])
    cov_set_1 = np.array([c_rot_eig_g,c_rot_eig_no_mit,c_rot_eig_mit])
    cov_set_2 = np.array([c_rot_eig_g[-2:,-2:],c_rot_eig_no_mit[-2:,-2:],c_rot_eig_mit[-2:,-2:]])
    import matplotlib.pyplot as plt
    Dn = survey_lw.observables[0]
#make_ellipse_plot(cov_set_2,colors,opacities,names,np.array([0.0002,0.0025]),pnames[-2:],dchi2,1.05)
#make_ellipse_plot(cov_set_1,colors,opacities,names,boxes,pnames,dchi2,1.05)
#make_ellipse_plot(cov_set_2,colors,opacities,names,boxes[-2:],pnames[-2:],dchi2,1.05)
#make_ellipse_plot(cov_set_2,colors,opacities,names,boxes[-2:],pnames[-2:],dchi2,1.05,False)
    a_lw = SS.multi_f.get_a_lw(destructive=True)
    v_db = np.diag(a_lw[0])
    n_exp = geo_wfirst.volumes*Dn.n_avg_bin1
    b_exp = Dn.b_ns1/Dn.n_avg_bin1
    v_db_b2 = b_exp**2*v_db
    n_bin = zs.size
    angles = np.zeros((n_bin,n_bin))
    for i in range(0,n_bin):
        for j in range(0,n_bin):
            angles[i,j] = np.dot(Dn.vs[i],Dn.vs[j])/(np.linalg.norm(Dn.vs[i])*np.linalg.norm(Dn.vs[j]))
