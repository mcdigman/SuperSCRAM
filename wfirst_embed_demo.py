"""demonstration case for wfirst embeded in lsst footprint to mitigate covariance"""
from time import time

import numpy as np


from Super_Survey import SuperSurvey,make_ellipse_plot
from lw_survey import LWSurvey
from sw_survey import SWSurvey
from cosmopie import CosmoPie
from polygon_geo import PolygonGeo
from sph_klim import SphBasisK
from matter_power_spectrum import MatterPower
from nz_wfirst import NZWFirst

import defaults


if __name__=='__main__':
    print "main: begin WFIRST demo"
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
    #fpt_params = defaults.fpt_params.copy()
    #wmatcher_params = defaults.wmatcher_params.copy()
    #halofit_params = defaults.halofit_params.copy()
    #matter_power_params = defaults.matter_power_params.copy()
    #hf_params = defaults.halofit_params.copy()
    poly_params = defaults.polygon_params.copy()
    lensing_params = defaults.lensing_params.copy()
    nz_params_wfirst_lens = defaults.nz_params_wfirst_lens.copy()
    sw_survey_params = defaults.sw_survey_params.copy()
    lw_survey_params = defaults.lw_survey_params.copy()
    sw_observable_list = defaults.sw_observable_list
    #TODO don't use defaults for setting up the core demo
    lw_observable_list = defaults.lw_observable_list
    dn_params = defaults.dn_params.copy()
    mf_params = defaults.hmf_params.copy()
    n_params_wfirst = defaults.nz_params_wfirst_gal.copy()
    power_params = defaults.power_params.copy()
    lw_param_list = np.array([{'dn_params':dn_params,'n1_params':n_params_wfirst,'n2_params':n_params_wfirst,'mf_params':mf_params}])
    #wmatcher_params['w_step']=0.05
    basis_params = defaults.basis_params.copy()

    #creat a CosmoPie object to manage the cosmology details
    print "main: begin constructing CosmoPie"
    C = CosmoPie(cosmo,p_space=p_space)

    #get the matter power spectrum and give it to the CosmoPie
    print "main: begin constructing MatterPower"
    P = MatterPower(C,power_params)
    C.set_power(P)

    #create the WFIRST geometry
    #zs are the bounding redshifts of the tomographic bins
    #zs = np.array([0.2,0.43,.63,0.9, 1.3])
    zs = np.arange(0.2,3.01,0.2)
    #z_fine are the resolution redshift slices to be integrated over
    z_fine = np.arange(lensing_params['z_min_integral'],np.max(zs),lensing_params['z_resolution'])
    #l_max is the highest l that should be precomputed
    l_max = 100

    #phi1s and theta1s as the coordinates of the vertices of the bounding polygon
    thetas_wfirst = np.array([-50.,-35.,-35.,-19.,-19.,-19.,-15.8,-15.8,-40.,-40.,-55.,-78.,-78.,-78.,-55.,-55.,-50.,-50.])*np.pi/180.+np.pi/2.
    phis_wfirst = np.array([-19.,-19.,-11.,-11.,7.,25.,25.,43.,43.,50.,50.,50.,24.,5.,5.,7.,7.,-19.])*np.pi/180.
    phi_in_wfirst = 7./180.*np.pi
    theta_in_wfirst = -35.*np.pi/180.+np.pi/2.
    print "main: begin constructing WFIRST PolygonGeo"
    geo_wfirst = PolygonGeo(zs,thetas_wfirst,phis_wfirst,theta_in_wfirst,phi_in_wfirst,C,z_fine,l_max,poly_params)

    #create the LSST geometry (for our purposes, a 20000 square degree survey encompassing the wfirst survey)
    #use the same redshift bin structure as for WFIRST because we only want LSST for galaxy counts, not lensing
    theta0 = np.pi/4.
    theta1 = 3.*np.pi/4.
    phi0 = 0.
    phi1 = 3.074096023740458
    thetas_lsst = np.array([theta0,theta1,theta1,theta0,theta0])
    phis_lsst = np.array([phi0,phi0,phi1,phi1,phi0])-phi1/2.
    theta_in_lsst = np.pi/2.
    phi_in_lsst = 0.

    print "main: begin constructing LSST PolygonGeo"
    geo_lsst = PolygonGeo(zs,thetas_lsst,phis_lsst,theta_in_lsst,phi_in_lsst,C,z_fine,l_max,poly_params)

    #create the short wavelength survey (SWSurvey) object
    #list of comsological parameters that will need to be varied
    if cosmo['de_model']=='w0wa':
        #TODO PRIORITY if using priors enforce cosmo_par_list ordered correctly, handle priors correcly
        cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w0','wa'])
        cosmo_par_string = ["$n_s$",r"$\Omega_m h^2$",r"$\Omega_{de} h^2$","$ln(A_s)$","$w_0$","$w_a$"]
        cosmo_par_epsilons = np.array([0.002,0.0005,0.0001,0.0005,0.1,0.01,0.07])
    elif cosmo['de_model']=='constant_w':
        cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w'])
        cosmo_par_string = ["$n_s$",r"$\Omega_m h^2$",r"$\Omega_{de} h^2$","$ln(A_s)$","$w_0$"]
        cosmo_par_epsilons = np.array([0.002,0.0005,0.0001,0.0005,0.1,0.01])
    else:
        raise ValueError('unrecognized de_model '+str(cosmo['de_model']))

    #create the number density of lensing sources
    print "main: begin constructing lensing source density for WFIRST"
    nz_wfirst_lens = NZWFirst(nz_params_wfirst_lens)
    #set some parameters for lensing
    lensing_params['n_gal'] = nz_wfirst_lens.get_N_projected(z_fine,geo_wfirst.angular_area())
    lensing_params['smodel'] = 'nzmatcher'
    lensing_params['z_min_dist'] = np.min(zs)
    lensing_params['z_max_dist'] = np.max(zs)
    lensing_params['pmodel_O'] = 'halofit'
    lensing_params['pmodel_dO_ddelta'] = 'halofit'
    lensing_params['pmodel_dO_dpar'] = 'halofit'

    #l bins for lensing
    l_sw = np.logspace(np.log(30),np.log(5000),base=np.exp(1.),num=40)
    #create the actual sw survey
    print "main: begin constructing SWSurvey for wfirst"
    sw_survey_wfirst = SWSurvey(geo_wfirst,'wfirst',C,l_sw,sw_survey_params,observable_list = sw_observable_list,cosmo_par_list = cosmo_par_list,cosmo_par_epsilons=cosmo_par_epsilons,len_params=lensing_params,nz_matcher=nz_wfirst_lens)
    surveys_sw = np.array([sw_survey_wfirst])

    #create the lw basis
    #z_max is the maximum radial extent of the basis
    z_max = zs[-1]+0.001
    r_max = C.D_comov(z_max)
    #k_cut is the maximum k balue for the bessel function zeros that define the basis
    #x_cut = 70.
    x_cut = 30.
    #x_cut = 10.
    k_cut = x_cut/r_max
    #l_max caps maximum l regardless of k
    print "main: begin constructing basis for long wavelength fluctuations"
    basis = SphBasisK(r_max,C,k_cut,basis_params,l_ceil=l_max)

    #create the lw survey
    geos = np.array([geo_wfirst,geo_lsst],dtype=object)
    print "main: begin constructing LWSurvey for mitigation"
    #TODO control cut out here
    survey_lw = LWSurvey(geos,'combined_survey',basis,C,lw_survey_params,observable_list=lw_observable_list,param_list=lw_param_list)
    surveys_lw = np.array([survey_lw])

    #create the SuperSurvey with mitigation
    print "main: begin constructing SuperSurvey"
    SS = SuperSurvey(surveys_sw, surveys_lw,basis,C=C,get_a=False,do_unmitigated=True,do_mitigated=True)

    time1 = time()
    print "main: finished construction tasks in "+str(time1-time0)+"s"

    mit_eigs_par = SS.eig_set[1,1]
    no_mit_eigs_par = SS.eig_set[1,0]
    print "main: unmitigated paramter lambda1,2: "+str(no_mit_eigs_par[0][-1])+","+str(no_mit_eigs_par[0][-2])
    print "main: mitigated parameter lambda1,2: "+str(mit_eigs_par[0][-1])+","+str(mit_eigs_par[0][-2])

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
    SS.make_standard_ellipse_plot()

#    valfound_g = {}
#    f_base_g = SS.f_set[0][2].get_fisher()
#    c_base_g = np.linalg.inv(f_base_g)
#    valfound_g[()] = c_base_g[4,4]
#    valfound_mit = {}
#    f_base_mit = SS.f_set[2][2].get_fisher()
#    c_base_mit = np.linalg.inv(f_base_mit)
#    valfound_mit[()] = c_base_mit[4,4]
#    valfound_no_mit = {}
#    f_base_no_mit = SS.f_set[1][2].get_fisher()
#    c_base_no_mit = np.linalg.inv(f_base_no_mit)
#    valfound_no_mit[()] = c_base_no_mit[4,4]
#    import itertools
#    for itr in xrange(1,7):
#        for combo in itertools.combinations(np.hstack([np.arange(0,4),np.array([5,6])]),itr):
#            fix_mat = np.zeros_like(SS.f_set[1][2].get_covar())
#            for index in combo:
#                fix_mat[index,index] = 1e9
#            f_mat_g = f_base_g+fix_mat
#            c_mat_g = np.linalg.inv(f_mat_g)
#            valfound_g[combo] = c_mat_g[4,4]
#            f_mat_mit = f_base_mit+fix_mat
#            c_mat_mit = np.linalg.inv(f_mat_mit)
#            valfound_mit[combo] = c_mat_mit[4,4]
#            f_mat_no_mit = f_base_no_mit+fix_mat
#            c_mat_no_mit = np.linalg.inv(f_mat_no_mit)
#            valfound_no_mit[combo] = c_mat_no_mit[4,4]
#f_fix_h = np.zeros_like(f_base_no_mit)
#sigma_fix_h = 0.00001
#f_fix_h[1,1] = 1./sigma_fix_h**2*(1./2.*1./(C.cosmology['Omegamh2']+C.cosmology['OmegaLh2']))**2
#f_fix_h[1,3] = 1./sigma_fix_h**2*(1./2.*1./(C.cosmology['Omegamh2']+C.cosmology['OmegaLh2']))**2
#f_fix_h[3,1] = 1./sigma_fix_h**2*(1./2.*1./(C.cosmology['Omegamh2']+C.cosmology['OmegaLh2']))**2
#f_fix_h[3,3] = 1./sigma_fix_h**2*(1./2.*1./(C.cosmology['Omegamh2']+C.cosmology['OmegaLh2']))**2
#f_h_fixed = f_base_no_mit+f_fix_h
#c_h_fixed = np.linalg.inv(f_h_fixed)
#
#dsigma8_dLogAs = (SS.surveys_sw[0].len_pow.Cs_pert[4][0].get_sigma8()-SS.surveys_sw[0].len_pow.Cs_pert[4][1].get_sigma8())/(2*cosmo_par_epsilons[4])
#dsigma8_dOmegamh2 = (SS.surveys_sw[0].len_pow.Cs_pert[1][0].get_sigma8()-SS.surveys_sw[0].len_pow.Cs_pert[1][1].get_sigma8())/(2*cosmo_par_epsilons[1])
#dsigma8_dOmegaLh2 = (SS.surveys_sw[0].len_pow.Cs_pert[3][0].get_sigma8()-SS.surveys_sw[0].len_pow.Cs_pert[3][1].get_sigma8())/(2*cosmo_par_epsilons[3])
#alpha_S8 = 0.5
#dS8_dLogAs = dsigma8_dLogAs*(C.cosmology['Omegam']/0.3)**alpha_S8
#dS8_dOmegamh2 = C.get_sigma8()*alpha_S8*C.cosmology['OmegaLh2']/(0.3**alpha_S8*C.h**4)*(C.cosmology['Omegam'])**(alpha_S8-1.)+dsigma8_dOmegamh2*(C.cosmology['Omegam']/0.3)**alpha_S8
#dS8_dOmegaLh2 = -C.get_sigma8()*alpha_S8/(0.3**alpha_S8*C.h**2)*(C.cosmology['Omegam'])**(alpha_S8)+dsigma8_dOmegaLh2*(C.cosmology['Omegam']/0.3)**alpha_S8
#dS8_dLogAs = (SS.surveys_sw[0].len_pow.Cs_pert[4][0].get_sigma8()*(SS.surveys_sw[0].len_pow.Cs_pert[4][0].cosmology['Omegam']/0.3)**alpha_S8-SS.surveys_sw[0].len_pow.Cs_pert[4][1].get_sigma8()*(SS.surveys_sw[0].len_pow.Cs_pert[4][1].cosmology['Omegam']/0.3)**alpha_S8)/(2*cosmo_par_epsilons[4])
#dS8_dns = (SS.surveys_sw[0].len_pow.Cs_pert[0][0].get_sigma8()*(SS.surveys_sw[0].len_pow.Cs_pert[0][0].cosmology['Omegam']/0.3)**alpha_S8-SS.surveys_sw[0].len_pow.Cs_pert[0][1].get_sigma8()*(SS.surveys_sw[0].len_pow.Cs_pert[0][1].cosmology['Omegam']/0.3)**alpha_S8)/(2*cosmo_par_epsilons[0])
#dS8_dOmegabh2 = (SS.surveys_sw[0].len_pow.Cs_pert[2][0].get_sigma8()*(SS.surveys_sw[0].len_pow.Cs_pert[2][0].cosmology['Omegam']/0.3)**alpha_S8-SS.surveys_sw[0].len_pow.Cs_pert[2][1].get_sigma8()*(SS.surveys_sw[0].len_pow.Cs_pert[2][1].cosmology['Omegam']/0.3)**alpha_S8)/(2*cosmo_par_epsilons[0])
#dS8_dw0 = (SS.surveys_sw[0].len_pow.Cs_pert[5][0].get_sigma8()*(SS.surveys_sw[0].len_pow.Cs_pert[5][0].cosmology['Omegam']/0.3)**alpha_S8-SS.surveys_sw[0].len_pow.Cs_pert[5][1].get_sigma8()*(SS.surveys_sw[0].len_pow.Cs_pert[5][1].cosmology['Omegam']/0.3)**alpha_S8)/(2*cosmo_par_epsilons[0])
#dS8_dwa = (SS.surveys_sw[0].len_pow.Cs_pert[6][0].get_sigma8()*(SS.surveys_sw[0].len_pow.Cs_pert[6][0].cosmology['Omegam']/0.3)**alpha_S8-SS.surveys_sw[0].len_pow.Cs_pert[6][1].get_sigma8()*(SS.surveys_sw[0].len_pow.Cs_pert[6][1].cosmology['Omegam']/0.3)**alpha_S8)/(2*cosmo_par_epsilons[0])
#
#f_rot_param_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','S8','w0','wa'])
#cosmo_param_fid = np.array([C.cosmology['ns'],C.cosmology['Omegamh2'],C.cosmology['Omegabh2'],C.cosmology['OmegaLh2'],C.cosmology['LogAs'],C.cosmology['w0'],C.cosmology['wa']])
##f_rot_no_mit = c_base_no_mit.copy()
#proj_S8 = np.identity(f_rot_param_list.size)
#proj_S8[4,4] = 1./dS8_dLogAs
##proj_S8[4,1] = 1./dS8_dOmegamh2
#proj_S8[1,4] = 1./dS8_dOmegamh2
##proj_S8[4,2] = 1./dS8_dOmegabh2
#proj_S8[2,4] = 1./dS8_dOmegabh2
##proj_S8[4,3] = 1./dS8_dOmegaLh2
#proj_S8[3,4] = 1./dS8_dOmegaLh2
##proj_S8[4,5] = 1./dS8_dw0
#proj_S8[5,4] = 1./dS8_dw0
##proj_S8[4,6] = 1./dS8_dwa
#proj_S8[6,4] = 1./dS8_dwa
#f_rot_no_mit = np.dot(proj_S8,np.dot(f_base_no_mit,proj_S8.T))
#f_rot_no_mit = (f_rot_no_mit+f_rot_no_mit.T)/2.
#c_rot_no_mit = np.linalg.inv(f_rot_no_mit)

cov_g = SS.f_set_nopriors[0][2].get_covar()
cov_g_inv = SS.f_set_nopriors[0][2].get_fisher()
chol_g = SS.f_set_nopriors[0][2].get_cov_cholesky()
chol_g_inv = SS.f_set_nopriors[0][2].get_cov_cholesky_inv()
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
#make_ellipse_plot(np.array([c_rot_eig_g[-2:,-2:],c_rot_eig_no_mit[-2:,-2:],c_rot_eig_mit[-2:,-2:]]),np.array([[0,1,0],[1,0,0],[0,0,1]]),np.array([1.,1.,1.]),np.array(['g','no mit','mit']),np.array([0.0002,0.0025]),np.array(['p2','p1']),dchi2,adaptive_mult=1.05)
#make_ellipse_plot(np.array([c_rot_eig_g,c_rot_eig_no_mit,c_rot_eig_mit]),np.array([[0,1,0],[1,0,0],[0,0,1]]),np.array([1.,1.,1.]),np.array(['g','no mit','mit']),np.array([5.,5.,5.,5.,5.,5.,5.,]),np.array(['p7','p6','p5','p4','p3','p2','p1']),dchi2,adaptive_mult=1.05)
#make_ellipse_plot(np.array([c_rot_eig_g[-2:,-2:],c_rot_eig_no_mit[-2:,-2:],c_rot_eig_mit[-2:,-2:]]),np.array([[0,1,0],[1,0,0],[0,0,1]]),np.array([1.,1.,1.]),np.array(['g','no mit','mit']),np.array([5.,5.]),np.array(['p2','p1']),dchi2,adaptive_mult=1.05)
make_ellipse_plot(np.array([c_rot_eig_g[-2:,-2:],c_rot_eig_no_mit[-2:,-2:],c_rot_eig_mit[-2:,-2:]]),np.array([[0,1,0],[1,0,0],[0,0,1]]),np.array([1.,1.,1.]),np.array(['g','no mit','mit']),np.array([5.,5.]),np.array(['p2','p1']),dchi2,adaptive_mult=1.05,include_diag=False)
