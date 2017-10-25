import numpy as np

from time import time 

from Super_Survey import SuperSurvey
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
    cosmo['de_model'] = 'constant_w'
    p_space = 'jdem' 
    camb_params = defaults.camb_params.copy()
    poly_params = defaults.polygon_params.copy()
    lensing_params = defaults.lensing_params.copy()
    nz_params_wfirst_lens = defaults.nz_params_wfirst_lens.copy()
    sw_survey_params=defaults.sw_survey_params.copy()
    lw_survey_params=defaults.lw_survey_params.copy()
    sw_observable_list = defaults.sw_observable_list
    lw_observable_list = defaults.lw_observable_list
    dn_params = defaults.dn_params.copy()

    #creat a CosmoPie object to manage the cosmology details
    print "main: begin constructing CosmoPie"
    C = CosmoPie(cosmo,camb_params=camb_params,p_space=p_space) 

    #get the matter power spectrum and give it to the CosmoPie
    print "main: begin constructing MatterPower"
    P = MatterPower(C,camb_params=camb_params)
    C.set_power(P)
        
    #create the WFIRST geometry
    #zs are the bounding redshifts of the tomographic bins
    zs = np.array([0.2,0.43,.63,0.9, 1.3])
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
    geo_wfirst = PolygonGeo(zs,thetas_wfirst,phis_wfirst,theta_in_wfirst,phi_in_wfirst,C,z_fine,l_max=l_max,poly_params=poly_params)

    #create the LSST geometry (for our purposes, a 20000 square degree survey encompassing the wfirst survey)
    #use the same redshift bin structure as for WFIRST because we only want LSST for galaxy counts, not lensing
    theta0=np.pi/4.
    theta1=3.*np.pi/4.
    phi0=0.
    phi1 = 3.074096023740458
    thetas_lsst = np.array([theta0,theta1,theta1,theta0,theta0])
    phis_lsst = np.array([phi0,phi0,phi1,phi1,phi0])-phi1/2.
    theta_in_lsst = np.pi/2.
    phi_in_lsst = 0.

    print "main: begin constructing LSST PolygonGeo"
    geo_lsst = PolygonGeo(zs,thetas_lsst,phis_lsst,theta_in_lsst,phi_in_lsst,C,z_fine,l_max=l_max,poly_params=poly_params)

    #create the short wavelength survey (SWSurvey) object
    #list of comsological parameters that will need to be varied
    if cosmo['de_model']=='w0wa':
        cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w0','wa'])
        cosmo_par_epsilons = np.array([0.002,0.0005,0.0001,0.0005,0.1,0.01,0.07])
    elif cosmo['de_model']=='constant_w':
        cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w'])
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
    sw_survey_wfirst = SWSurvey(geo_wfirst,'wfirst',C=C,ls=l_sw,params=sw_survey_params,observable_list = sw_observable_list,cosmo_par_list = cosmo_par_list,cosmo_par_epsilons=cosmo_par_epsilons,len_params=lensing_params,nz_matcher=nz_wfirst_lens)
    surveys_sw=np.array([sw_survey_wfirst])

    #create the lw basis
    #z_max is the maximum radial extent of the basis
    z_max = zs[-1]+0.001
    r_max = C.D_comov(z_max)
    #k_cut is the maximum k balue for the bessel function zeros that define the basis
    k_cut = 0.005 
    #l_max caps maximum l regardless of k
    print "main: begin constructing basis for long wavelength fluctuations"
    basis = SphBasisK(r_max,C,k_cut,l_ceil=l_max)

    #create the lw survey
    geos = np.array([geo_wfirst,geo_lsst],dtype=object)
    print "main: begin constructing LWSurvey for mitigation"
    survey_lw = LWSurvey(geos,'combined_survey',basis,C=C,params=lw_survey_params,observable_list=lw_observable_list,dn_params=dn_params)
    surveys_lw=np.array([survey_lw])

    #create the SuperSurvey with mitigation
    print "main: begin constructing SuperSurvey"
    SS=SuperSurvey(surveys_sw, surveys_lw,basis,C=C,get_a=False,do_unmitigated=True,do_mitigated=True)
    
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
