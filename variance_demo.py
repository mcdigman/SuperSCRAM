import numpy as np
from sw_survey import SWSurvey
from lw_survey import LWSurvey
from polygon_geo import PolygonGeo
from geo import RectGeo
from cosmopie import CosmoPie
import defaults
from sph_klim import SphBasisK
from Super_Survey import SuperSurvey
from time import time

if __name__ == '__main__':
    time0 = time()
    camb_params = {'npoints':1000,
        'minkh':1.1e-4,
        'maxkh':1.476511342960e+02,
        'kmax':1.476511342960e+02,
        'leave_h':False,
        'force_sigma8':True,
        'return_sigma8':False
    }
    print "main: building cosmology"
    cosmo_use = defaults.cosmology.copy()
    cosmo_use['h'] = 0.6774
    cosmo_use['Omegabh2'] = 0.02230
    cosmo_use['mnu'] = 0.06
    cosmo_use['Omegamh2'] = 0.1188
    cosmo_use['OmegaL'] = 0.7
    cosmo_use['de_model'] = 'constant_w'
    cosmo_use['w'] = -0.9
    cosmo_use['ns'] = 0.9667
    cosmo_use['As'] = 2.142e-9
    cosmo_use['OmegaLh2'] = 0.7*0.6774**2
    cosmo_use['Omegakh2'] = 0.
    cosmo_use['LogAs'] = np.log(2.142e-9)
    C = CosmoPie(defaults.cosmology.copy(),p_space='jdem',camb_params=camb_params,needs_power=True)

    #x_cut = 527.
    #l_max = 515
    x_cut = 30
    l_max = 515

    param_use = 3
    do_plot = True
    if param_use == 1:
        #matches to  0.918455921357 for l_max=340,x_cut=360, not monotonic
        theta0=np.pi/2.-np.pi/1801.87*10.*30.*1.3
        theta1=np.pi/2.+np.pi/1801.87*10.*30.*1.3
        phi0=0.
        phi1=np.pi/900.93*10.*30.*1.3

        z_coarse = np.array([0.001,0.005])
        z_max = 0.0055
        z_fine = np.arange(0.00001,0.2,0.00001)

        variance_pred1=6.94974e-01
        variance_pred2 = 6.47471e-01
    elif param_use == 2:
        #matches to 0.984883733033 for l_max=340,x_cut=360, monotonic 
        theta0=np.pi/2.-np.pi/1801.87*10.
        theta1=np.pi/2.+np.pi/1801.87*10.
        phi0=0.
        phi1=np.pi/900.93*10.

        z_coarse = np.array([0.9,1.0])
        z_fine = np.arange(0.0005,1.,0.0005)
        z_max = 1.05

        variance_pred1=1.32333e-02
        variance_pred2=1.38770e-02
    elif param_use ==3:
        #matches to  0.256984236659 for l_max=340,x_cut=360, monotonic but convergence appears poor
        theta0=np.pi/2.-np.pi/1801.87
        theta1=np.pi/2.+np.pi/1801.87
        phi0=0.
        phi1=np.pi/900.93

        z_coarse = np.array([0.99525,1.0])
        z_fine = np.arange(0.0005,1.,0.0005)
        z_max = 1.05

        variance_pred1=1.18262e+00
        variance_pred2 = 1.18253e+00
    elif param_use==4:
        #matches to 0.982322621415 for l_max=340,x_cut=360, monotonic, convergence looks good 
        theta0=np.pi/2.-np.pi/1801.87*47
        theta1=np.pi/2.+np.pi/1801.87*47
        phi0=0.
        phi1=np.pi/900.93*47

        z_coarse = np.array([0.8,1.0])
        z_fine = np.arange(0.0005,1.,0.0005)
        z_max = 1.05

        variance_pred1=4.39584e-04
        variance_pred2=4.38021e-04
    elif param_use==5:
        #matches to side 1.17896133281 vol 0.914043964767 for l_max=340,x_cut=360, appears converged
        theta0=np.pi/2.-np.pi/1801.87*100*8.7
        theta1=np.pi/2.+np.pi/1801.87*100*8.7
        phi0=0.
        phi1=np.pi/900.93*100*8.7

        z_coarse = np.array([0.6,1.3])
        z_fine = np.arange(0.0005,1.,0.0005)
        z_max = 1.301

        #by sides 
        variance_pred1=1.45123e-07
        #by volume
        variance_pred2 = 1.87184e-07

    thetas = np.array([theta0,theta1,theta1,theta0,theta0])
    phis = np.array([phi0,phi0,phi1,phi1,phi0])
    theta_in = (theta1+theta0)/2.
    phi_in = (phi1+phi0)/2.

    l_sw = np.logspace(np.log(30),np.log(5000),base=np.exp(1.),num=40)

    print "main: building geometries"
    polygon_params = defaults.polygon_params
    polygon_params['n_double'] = 30
    geo1 = PolygonGeo(z_coarse,thetas,phis,theta_in,phi_in,C,z_fine,l_max=l_max,poly_params=defaults.polygon_params)
    print 'main: r diffs',np.diff(geo1.rs)
    print 'main: theta width',(geo1.rs[1]+geo1.rs[0])/2.*(theta1-theta0)
    #print 'main: phi width',(geo1.rs[1]+geo1.rs[0])/2.*(phi1-phi0)*np.sin((theta1+theta0)/2)
    print 'main: phi width',(geo1.rs[1]+geo1.rs[0])/2.*(phi1-phi0)
#    geo2 = PolygonGeo(z_coarse,thetas,phis+phi1,theta_in,phi_in+phi1,C,z_fine,l_max=l_max,poly_params=defaults.polygon_params)
#    #geo1 = RectGeo(z_coarse,np.array([theta0,theta1]),np.array([phi0,phi1]),C,z_fine)
#    #geo2 = RectGeo(z_coarse,np.array([theta0,theta1]),np.array([phi0,phi1])+phi1,C,z_fine)

    print "main: building basis"
    basis_params = defaults.basis_params.copy()
    #basis_params['n_bessel_oversample']*=4
    basis_params['x_grid_size']*=10

    r_max = C.D_comov(z_max)
    k_cut = x_cut/r_max
    k_tests = np.linspace(20./r_max,k_cut,100)
    n_basis = np.zeros(k_tests.size)
    variances = np.zeros(k_tests.size)

    basis = SphBasisK(r_max,C,k_cut,l_ceil=l_max,params=basis_params)

    for i in xrange(0,k_tests.size):
        print "r_max,k_cut",r_max,k_tests[i]
        n_basis[i] = np.sum(basis.C_id[:,1]<=k_tests[i])
        variances[i] = basis.get_variance(geo1,k_cut_in=k_tests[i])
        print "main: with k_cut="+str(k_tests[i])+" size="+str(n_basis[i])+" got variance="+str(variances[i])

    if do_plot:
        import matplotlib.pyplot as plt
        plt.plot(n_basis,variances)
        plt.show()
#
#    print "main: building sw survey"
#    #cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w'])
#    #cosmo_par_epsilons = np.array([0.002,0.0005,0.0001,0.0005,0.1,0.01])
#    cosmo_par_list = np.array([])
#    cosmo_par_epsilons = np.array([])
#    loc_lens_params = defaults.lensing_params.copy()
#    loc_lens_params['z_min_dist'] = np.min(z_coarse)
#    loc_lens_params['z_max_dist'] = np.max(z_coarse)
#    loc_lens_params['pmodel_O'] = 'halofit'
#    loc_lens_params['pmodel_dO_ddelta'] = 'halofit'
#    loc_lens_params['pmodel_dO_dpar'] = 'halofit'
#
#    sw_survey_1 = SWSurvey(geo1,'survey1',C=C,ls=l_sw,params=defaults.sw_survey_params,observable_list = defaults.sw_observable_list,cosmo_par_list = cosmo_par_list,cosmo_par_epsilons=cosmo_par_epsilons,len_params=loc_lens_params)
#    surveys_sw = np.array([sw_survey_1])

#    do_mit = False
#    if do_mit:
#        print "main: building lw survey"
#        geos = np.array([geo1,geo2])
#        lw_survey_1 = LWSurvey(geos,'lw_survey1',basis,C=C,params=defaults.lw_survey_params,observable_list=defaults.lw_observable_list,dn_params=defaults.dn_params)
#        surveys_lw = np.array([lw_survey_1])
#    else:
#        geos = np.array([geo1,geo2])
#        surveys_lw = np.array([])
#    print "main: building super survey"
#    SS=SuperSurvey(surveys_sw, surveys_lw,basis,C=C,get_a=True,do_unmitigated=True,do_mitigated=do_mit)
    print "main: getting variance"
    variance_res = basis.get_variance(geo1)
    time1 = time()
    print "main: finished building in "+str(time1-time0)+"s"

    r_width = np.diff(geo1.rs)[0]
    theta_width=(geo1.rs[1]+geo1.rs[0])/2.*(theta1-theta0)
    phi_width=(geo1.rs[1]+geo1.rs[0])/2.*(phi1-phi0)
    volume = 4./3.*np.pi*(geo1.rs[1]**3-geo1.rs[0]**3)*geo1.angular_area()/(4.*np.pi)
    square_equiv = volume**(1./3.)
    print 'main: r diffs',r_width*C.h
    print 'main: theta width',theta_width*C.h
    #print 'main: phi width',(geo1.rs[1]+geo1.rs[0])/2.*(phi1-phi0)*np.sin((theta1+theta0)/2)
    print 'main: phi width',phi_width*C.h
    print 'main: square side length',square_equiv*C.h
    print "main: variance is "+str(variance_res)
    print "main: variance/predicted by sides is "+str(variances[-1]/variance_pred1)
    print "main: variance/predicted by volume is "+str(variances[-1]/variance_pred2)
    #print "main: rate of change of variance is "+str((variances[-1]-variances[-2])/(k_tests[-1]-k_tests[-2]))
    approx_deriv = np.average((np.diff(variances)/np.diff(n_basis))[-5::])
    estim_change = approx_deriv*n_basis[-1]*2.
    estim_converge = estim_change/variances[-1]
    print "main: estimate variance converged to within "+str(estim_converge*100.)+"%, first approx of true value ~"+str(estim_change+variances[-1])