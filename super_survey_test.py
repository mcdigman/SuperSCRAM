"""class for manipulating and plotting the Super Sample term"""
from __future__ import print_function,division,absolute_import
from builtins import range
from time import time

import numpy as np
import pytest

import defaults
import multi_fisher as mf
from super_survey import SuperSurvey
from polygon_pixel_geo import PolygonPixelGeo
from polygon_geo import PolygonGeo
from sph_klim import SphBasisK
from geo import RectGeo
from sw_survey import SWSurvey
from lw_survey import LWSurvey
from warnings import warn
import prior_fisher
import matter_power_spectrum as mps
import cosmopie as cp

@pytest.fixture(params=['constant_w','w0wa'])
def de_model(request):
    return request.param

def test_super_survey(de_model):
    """run some eigenvalue tests of SuperSurvey pipeline"""
    t1 = time()
    z_max = 1.35
    l_max = 50

    camb_params = defaults.camb_params.copy()
    camb_params['force_sigma8'] = False
    camb_params['kmax'] = 5.
    camb_params['npoints'] = 1000
    cosmo_fid = defaults.cosmology_jdem.copy()
    cosmo_fid['w'] = -1.
    cosmo_fid['w0'] = cosmo_fid['w']
    cosmo_fid['wa'] = 0.
    cosmo_fid['de_model'] = de_model
    if cosmo_fid['de_model']=='jdem':
        for i in range(0,36):
            cosmo_fid['ws36_'+str(i).zfill(2)] = cosmo_fid['w']

    C = cp.CosmoPie(cosmology=cosmo_fid,p_space='jdem')
    power_params = defaults.power_params.copy()
    power_params.camb = camb_params
    P = mps.MatterPower(C,power_params)
    C.set_power(P)
    r_max = C.D_comov(z_max)
    print('this is r max and l_max', r_max , l_max)

    phi1s = np.array([-19.,-19.,-11.,-11.,7.,25.,25.,43.,43.,50.,50.,50.,24.,5.,5.,7.,7.,-19.])*np.pi/180.
    theta1s = np.array([-50.,-35.,-35.,-19.,-19.,-19.,-15.8,-15.8,-40.,-40.,-55.,-78.,-78.,-78.,-55.,-55.,-50.,-50.])*np.pi/180.+np.pi/2.
    phi_in1 = 7./180.*np.pi
    theta_in1 = -35.*np.pi/180.+np.pi/2.

    theta0 = np.pi/4.
    theta1 = 3.*np.pi/4.
    phi0 = 0.
    phi1 = 3.074096023740458
    phi2 = np.pi/3.
    phi3 = phi2+(phi1-phi0)
    theta2s = np.array([theta0,theta1,theta1,theta0,theta0])
    phi2s = np.array([phi0,phi0,phi1,phi1,phi0])-phi1/2.
    theta_in2 = np.pi/2.
    phi_in2 = 0.
    res_choose = 7



    Theta1 = [theta0,theta1]
    Phi1 = [phi0,phi1]
    Theta2 = [theta0,theta1]
    Phi2 = [phi2,phi3]

    zs = np.array([0.2,0.43,.63,0.9, 1.3])
    z_fine = np.linspace(0.001,np.max(zs),2000) #use linspace
    use_poly = True
    use_poly2 = True
    if use_poly:
        if use_poly2:
            geo1 = PolygonGeo(zs,theta1s,phi1s,theta_in1,phi_in1,C,z_fine,l_max,defaults.polygon_params.copy())
            geo2 = PolygonGeo(zs,theta2s,phi2s,theta_in2,phi_in2,C,z_fine,l_max,defaults.polygon_params.copy())
        else:
            geo1 = PolygonPixelGeo(zs,theta1s,phi1s,theta_in1,phi_in1,C,z_fine,l_max,res_choose)
            geo2 = PolygonPixelGeo(zs,theta2s,phi2s,theta_in2,phi_in2,C,z_fine,l_max,res_choose)
    else:
        geo1 = RectGeo(zs,Theta1,Phi1,C,z_fine)
        geo2 = RectGeo(zs,Theta2,Phi2,C,z_fine)

    len_params = defaults.lensing_params.copy()
    len_params['pmodel'] = 'halofit'

    lenless_defaults = defaults.sw_survey_params.copy()
    lenless_defaults['needs_lensing'] = False
    if cosmo_fid['de_model']=='w0wa':
        cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w0','wa'])
        cosmo_par_eps = np.array([0.002,0.0005,0.0001,0.0005,0.1,0.01,0.07])
    elif cosmo_fid['de_model']=='constant_w':
        cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w'])
        cosmo_par_eps = np.array([0.002,0.0005,0.0001,0.0005,0.1,0.01])
    elif cosmo_fid['de_model']=='jdem':
        cosmo_par_list = ['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs']
        cosmo_par_list.extend(cp.JDEM_LIST)
        cosmo_par_list = np.array(cosmo_par_list,dtype=object)
        cosmo_par_eps = np.full(41,0.5)
        cosmo_par_eps[0:5] = np.array([0.002,0.0005,0.0001,0.0005,0.1])

    else:
        raise ValueError('not prepared to handle '+str(cosmo_fid['de_model']))

    #note that currently (poorly implemented derivative) in jdem,
    #OmegaLh2 and LogAs are both almost completely unconstrained but nondegenerate,
    #while in basic, h and sigma8 are not constrained but are almost completely degenerate
    nz_params = defaults.nz_params_wfirst_lens.copy()

    from nz_wfirst import NZWFirst
    nz_matcher = NZWFirst(nz_params)
    len_params['smodel'] = 'nzmatcher'
    sw_params = defaults.sw_survey_params.copy()
    obs_list = defaults.sw_observable_list.copy()
    survey_1 = SWSurvey(geo1,'s1',C,sw_params,cosmo_par_list,cosmo_par_eps,obs_list,len_params,nz_matcher)

    surveys_sw = np.array([survey_1])


    geos = np.array([geo1,geo2])
    k_cut = 0.005

    basis = SphBasisK(r_max,C,k_cut,defaults.basis_params.copy(),l_ceil=100)
    lw_param_list = defaults.lw_param_list.copy()
    lw_observable_list = defaults.lw_observable_list.copy()
    survey_3 = LWSurvey(geos,'lw_survey1',basis,C,defaults.lw_survey_params.copy(),observable_list=lw_observable_list,param_list=lw_param_list)
    surveys_lw = np.array([survey_3])


    print('main: this is r_max: '+str(r_max))
    SS = SuperSurvey(surveys_sw,surveys_lw,basis,C,defaults.prior_fisher_params.copy(),get_a=False,do_unmitigated=True,do_mitigated=True,include_sw=True)

    t2 = time()
    print("main: total run time "+str(t2-t1)+" s")

    mit_eigs_sw = SS.eig_set[0,1]
    no_mit_eigs_sw = SS.eig_set[0,0]
    mit_eigs_par = SS.eig_set[1,1]
    no_mit_eigs_par = SS.eig_set[1,0]
    #TODO check eigenvalue interlace for this projection
    SS.print_standard_analysis()
#    print("main: unmitigated sw lambda1,2: "+str(no_mit_eigs_sw[0][-1])+","+str(no_mit_eigs_sw[0][-2]))
#    print("main: mitigated sw lambda1,2: "+str(mit_eigs_sw[0][-1])+","+str(mit_eigs_sw[0][-2]))
#    print("main: n sw mit lambda>1.00000001: "+str(np.sum(np.abs(mit_eigs_sw[0])>1.00000001)))
#    print("main: n sw no mit lambda>1.00000001: "+str(np.sum(np.abs(no_mit_eigs_sw[0])>1.00000001)))
#    print("main: unmitigated par lambda1,2: "+str(no_mit_eigs_par[0][-1])+","+str(no_mit_eigs_par[0][-2]))
#    print("main: mitigated par lambda1,2: "+str(mit_eigs_par[0][-1])+","+str(mit_eigs_par[0][-2]))
#    print("main: n par mit lambda>1.00000001: "+str(np.sum(np.abs(mit_eigs_par[0])>1.00000001)))
#    print("main: n par no mit lambda>1.00000001: "+str(np.sum(np.abs(no_mit_eigs_par[0])>1.00000001)))
    v_no_mit_par = np.dot(SS.f_set_nopriors[0][2].get_cov_cholesky(),no_mit_eigs_par[1])
    v_mit_par = np.dot(SS.f_set_nopriors[0][2].get_cov_cholesky(),mit_eigs_par[1])

    v_no_mit_sw = np.dot(SS.f_set_nopriors[0][1].get_cov_cholesky(),no_mit_eigs_sw[1])
    v_mit_sw = np.dot(SS.f_set_nopriors[0][1].get_cov_cholesky(),mit_eigs_sw[1])

    test_v = True
    v_test_fails = 0
    if test_v:
        m_mat_no_mit_par = np.identity(mit_eigs_par[0].size)+np.dot(SS.f_set_nopriors[1][2].get_covar(),SS.f_set_nopriors[0][2].get_fisher())
        m_mat_mit_par = np.identity(mit_eigs_par[0].size)+np.dot(SS.f_set_nopriors[2][2].get_covar(),SS.f_set_nopriors[0][2].get_fisher())

        if not np.allclose(np.dot(m_mat_no_mit_par,v_no_mit_par)/(1.+no_mit_eigs_par[0]),v_no_mit_par):
            v_test_fails+=1
            warn('some no mit eig vectors may be bad')
        if not np.allclose(np.dot(m_mat_mit_par,v_mit_par)/(1.+mit_eigs_par[0]),v_mit_par):
            v_test_fails+=1
            warn('some mit eig vectors may be bad')
    if v_test_fails == 0:
        print("PASS: eigenvector decomposition checks passed")
    else:
        raise RuntimeError('FAIL: '+str(v_test_fails)+' eigenvector decomposition checks failed')


    get_hold_mats = False
    if get_hold_mats:
        no_prior_hold = SS.f_set[0][2].get_fisher()
        if C.de_model=='jdem':
            no_prior_project = prior_fisher.project_w0wa(no_prior_hold,defaults.prior_fisher_params.copy(),prior_fisher.JDEM_LABELS)

    print('main: r diffs',np.diff(geo1.rs))
    print('main: theta width',(geo1.rs[1]+geo1.rs[0])/2.*(Theta1[1]-Theta1[0]))
    print('main: phi width',(geo1.rs[1]+geo1.rs[0])/2.*(Phi1[1]-Phi1[0])*np.sin((Theta1[1]+Theta1[0])/2))

    test_perturbation = True
    pert_test_fails = 0
    if test_perturbation:
        #TOLERANCE below which an eigenvalue less than TOLERANCE*max eigenvalue is considered 0
        REL_TOLERANCE = 10**-8
        f0 = SS.multi_f.get_fisher(mf.f_spec_no_mit,mf.f_return_lw)[0].get_fisher()
        f1 = SS.multi_f.get_fisher(mf.f_spec_mit,mf.f_return_lw)[0].get_fisher()
        if not np.all(f0.T==f0):
            pert_test_fails+=1
            warn("unperturbed fisher matrix not symmetric, unacceptable")
        if not np.all(f1.T==f1):
            pert_test_fails+=1
            warn("perturbed fisher matrix not symmetric, unacceptable")
        #get eigenvalues and set numerically zero values to 0
        eigf0 = np.linalg.eigh(f0)[0]
        eigf0[np.abs(eigf0)<REL_TOLERANCE*np.max(np.abs(eigf0))] = 0.
        eigf1 = np.linalg.eigh(f1)[0]
        eigf1[np.abs(eigf1)<REL_TOLERANCE*np.max(np.abs(eigf1))] = 0.
        #check positive semidefinite
        if np.any(eigf0<0.):
            pert_test_fails+=1
            warn("unperturbed fisher matrix not positive definite within tolerance, unacceptable")
        if np.any(eigf1<0.):
            pert_test_fails+=1
            warn("perturbed fisher matrix not positive definite within tolerance, unacceptable")

        #check nondecresasing
        diff_eig = eigf1-eigf0
        diff_eig[np.abs(diff_eig)<REL_TOLERANCE*np.max(np.abs(diff_eig))] = 0
        if np.any(diff_eig<0):
            pert_test_fails+=1
            warn("some eigenvalues decreased within tolerance, unacceptable")

        #check interlace theorem satisfied (eigenvalues cannot be reordered by more than rank of perturbation)
        n_offset = SS.surveys_lw[0].get_total_rank()
        rolled_eig = (eigf1[::-1][n_offset:eigf0.size]-eigf0[::-1][0:eigf0.size-n_offset])
        rolled_eig[np.abs(rolled_eig)<REL_TOLERANCE*np.max(np.abs(rolled_eig))] = 0.
        if np.any(rolled_eig>0):
            pert_test_fails+=1
            warn("some eigenvalues fail interlace theorem, unacceptable")

        c0 = SS.multi_f.get_fisher(mf.f_spec_no_mit,mf.f_return_lw)[0].get_covar()
        c1 = SS.multi_f.get_fisher(mf.f_spec_mit,mf.f_return_lw)[0].get_covar()
        if not np.all(c0==c0.T):
            pert_test_fails+=1
            warn("unperturbed covariance not symmetric, unacceptable")
        if not np.all(c1==c1.T):
            warn("perturbed covariance not symmetric, unacceptable")
        eigc0 = np.linalg.eigh(c0)[0]
        eigc1 = np.linalg.eigh(c1)[0]
        if np.any(eigc0<0):
            pert_test_fails+=1
            warn("unperturbed covariance not positive semidefinite, unacceptable")
        if np.any(eigc1<0):
            pert_test_fails+=1
            warn("perturbed covariance not positive semidefinite, unacceptable")
        fdiff_eigc = (eigc1-eigc0)/eigc0
        fdiff_eigc[np.abs(fdiff_eigc)<REL_TOLERANCE] = 0.
        if np.any(fdiff_eigc>0):
            pert_test_fails+=1
            warn("some covariance eigenvalues increase, unacceptable")

        if pert_test_fails==0:
            print("PASS: All fisher matrix sanity checks passed")
        else:
            raise RuntimeError("FAIL: "+str(pert_test_fails)+" fisher matrix sanity checks failed")
    test_eigs = True
    eig_test_fails = 0
    if test_eigs:
        REL_TOLERANCE = 10**-8
        c_ssc0 = SS.multi_f.get_fisher(mf.f_spec_SSC_no_mit,mf.f_return_sw)[1].get_covar()#SS.covs_sw[0].get_ssc_covar()
        if not np.allclose(c_ssc0,c_ssc0.T):
            eig_test_fails+=1
            warn("unperturbed result covariance not symmetric, unacceptable")
        c_ssc1 = SS.multi_f.get_fisher(mf.f_spec_SSC_mit,mf.f_return_sw)[1].get_covar()#SS.covs_sw[0].get_ssc_covar()
        if not np.allclose(c_ssc1,c_ssc1.T):
            eig_test_fails+=1
            warn("perturbed result covariance not symmetric, unacceptable")
        eigsys_ssc0 = np.linalg.eigh(c_ssc0)
        eigsys_ssc1 = np.linalg.eigh(c_ssc1)
        eig_ssc0 = eigsys_ssc0[0].copy()
        eig_ssc1 = eigsys_ssc1[0].copy()
        eig_ssc0[np.abs(eig_ssc0)<np.max(np.abs(eig_ssc0))*REL_TOLERANCE] = 0
        eig_ssc1[np.abs(eig_ssc0)<np.max(np.abs(eig_ssc0))*REL_TOLERANCE] = 0
        if np.any(eig_ssc0<0):
            eig_test_fails+=1
            warn("unperturbed result cov not positive semidefinite, unacceptable")
        if np.any(eig_ssc1<0):
            eig_test_fails+=1
            warn("perturbed result cov not positive semidefinite, unacceptable")
        cg = SS.f_set_nopriors[0][1].get_covar()
        eigsys_cg = np.linalg.eigh(cg)
        eig_mitprod = np.real(np.linalg.eig(np.dot(np.linalg.inv(c_ssc0+cg),c_ssc1+cg))[0])
        eig_mitprod[np.abs(eig_mitprod-1.)<REL_TOLERANCE] = 1.
        if np.any(eig_mitprod>1):
            eig_test_fails+=1
            warn("mitigation making covariance worse, unacceptable")
        n_offset = SS.surveys_lw[0].get_total_rank()
        if np.sum(eig_mitprod<1.)>n_offset:
            eig_test_fails+=1
            warn("mitigation changing too many eigenvalues, unacceptable")
        eig_diff = eig_ssc1-eig_ssc0
        eig_diff[np.abs(eig_diff)<np.max(np.abs(eig_diff))*REL_TOLERANCE] = 0.
        if np.any(eig_diff>0):
            eig_test_fails+=1
            warn("mitigation making covariance worse, unacceptable")


        if eig_test_fails==0:
            print("PASS: All sw eigenvalue sanity checks passed")
        else:
            raise RuntimeError("FAIL: "+str(pert_test_fails)+" eigenvalue sanity checks failed")

        do_eig_interlace_check = True
        if do_eig_interlace_check:
            eig_interlace_fails_mit = 0
            eig_interlace_fails_no_mit = 0
            n_sw = mit_eigs_sw[0].size
            n_par = mit_eigs_par[0].size
            d_n = n_sw-n_par
            eig_l_mit_par = mit_eigs_par[0][::-1]
            eig_l_no_mit_par = no_mit_eigs_par[0][::-1]
            eig_l_mit_sw = mit_eigs_sw[0][::-1]
            eig_l_no_mit_sw = no_mit_eigs_sw[0][::-1]
            for i in range(0,n_par):
                if eig_l_mit_par[i]>eig_l_mit_sw[i]:
                    eig_interlace_fails_mit+=1
                if eig_l_no_mit_par[i]>eig_l_no_mit_sw[i]:
                    eig_interlace_fails_no_mit+=1
                if eig_l_mit_par[i]<eig_l_mit_sw[i+d_n]:
                    eig_interlace_fails_mit+=1
                if eig_l_no_mit_par[i]<eig_l_no_mit_sw[i+d_n]:
                    eig_interlace_fails_no_mit+=1
            if eig_interlace_fails_mit==0 and eig_interlace_fails_no_mit==0:
                print("PASS: All parameter eigenvalue interlace tests passed")
            else:
                raise RuntimeError("FAIL: "+str(eig_interlace_fails_mit)+" mitigation and "+str(eig_interlace_fails_no_mit)+" no mitigation failures in parameter eigenvalue interlace tests")

if __name__=="__main__":
    pytest.cmdline.main(["super_survey_test.py"])

#if __name__=="__main__":
#    from polygon_pixel_geo import PolygonPixelGeo
#    from polygon_geo import PolygonGeo
#    from sph_klim import SphBasisK
#    from geo import RectGeo
#    from sw_survey import SWSurvey
#    from lw_survey import LWSurvey
#    from warnings import warn
##    import fisher_matrix as fm
#    import prior_fisher
#    import matter_power_spectrum as mps
#    import cosmopie as cp
#    t1 = time()
#    z_max = 1.35
#    l_max = 50
#
#    #d=np.loadtxt('Pk_Planck15.dat')
#    #d=np.loadtxt('camb_m_pow_l.dat')
#    #k=d[:,0]; P=d[:,1]
#    camb_params = defaults.camb_params.copy()
#    camb_params['force_sigma8'] = False
#    camb_params['kmax'] = 5.
#    #camb_params['maxkh'] = 100.
#    camb_params['npoints'] = 1000
#    cosmo_fid = defaults.cosmology_jdem.copy()
#    cosmo_fid['w'] = -1.
#    cosmo_fid['w0'] = cosmo_fid['w']
#    cosmo_fid['wa'] = 0.
#    cosmo_fid['de_model'] = 'constant_w'
#    if cosmo_fid['de_model']=='jdem':
#        for i in range(0,36):
#            cosmo_fid['ws36_'+str(i).zfill(2)] = cosmo_fid['w']
#
#    C = cp.CosmoPie(cosmology=cosmo_fid,p_space='jdem')
#    #C=cp.CosmoPie(cosmology=defaults.cosmology.copy(),p_space='basic')
#    #k,P=C.get_P_lin()
#    power_params = defaults.power_params.copy()
#    power_params.camb = camb_params
#    P = mps.MatterPower(C,power_params)
#    k = P.k
#    C.set_power(P)
#    r_max = C.D_comov(z_max)
#    print('this is r max and l_max', r_max , l_max)
#
##    #theta0=0.
##    #theta1=np.pi/2.
#    phi1s = np.array([-19.,-19.,-11.,-11.,7.,25.,25.,43.,43.,50.,50.,50.,24.,5.,5.,7.,7.,-19.])*np.pi/180.
#    theta1s = np.array([-50.,-35.,-35.,-19.,-19.,-19.,-15.8,-15.8,-40.,-40.,-55.,-78.,-78.,-78.,-55.,-55.,-50.,-50.])*np.pi/180.+np.pi/2.
#    phi_in1 = 7./180.*np.pi
#    theta_in1 = -35.*np.pi/180.+np.pi/2.
#
#    theta0 = np.pi/4.
#    theta1 = 3.*np.pi/4.
#    phi0 = 0.
##    #phi1=5.*np.pi**2/162. #gives exactly a 1000 square degree field of view
#    phi1 = 3.074096023740458
#    phi2 = np.pi/3.
#    phi3 = phi2+(phi1-phi0)
##    theta1s = np.array([theta0,theta1,theta1,theta0,theta0])
##    phi1s = np.array([phi0,phi0,phi1,phi1,phi0])-phi1/2.
##    theta_in1 = 3.*np.pi/8.
##    #phi_in1 = 4.*np.pi**2/162.
##    phi_in1 = 5*np.pi**2/(2.*np.sqrt(2)*162.)
##
##    #theta2s = np.array([theta0,theta1,theta1,theta0,theta0])
##    #phi2s = np.array([phi2,phi2,phi3,phi3,phi2])
##    #theta_in2 = 3.*np.pi/8.
##    #phi_in2 = phi2+(phi_in1-phi0)
##
#    theta2s = np.array([theta0,theta1,theta1,theta0,theta0])
#    phi2s = np.array([phi0,phi0,phi1,phi1,phi0])-phi1/2.
#    theta_in2 = np.pi/2.
#    phi_in2 = 0.
#    #phi2s*=4.*0.7745282
#    #phi_in2 = 5*np.pi**2/(2.*np.sqrt(2)*162.)
#    #theta0=np.pi/16.
#    #theta1=np.pi/2.
#    #phi0=0.
#    #phi1=np.pi/6.
#    #phi2=2.*np.pi/3.
#    #phi3=5.*np.pi/6.
#
#    #theta1s = np.array([theta0,theta1,theta1,theta0,theta0])
#    #phi1s = np.array([phi0,phi0,phi1,phi1,phi0])
#    #theta_in1 = np.pi/8.
#    #phi_in1 = np.pi/12.
#    #theta2s = np.array([theta0,theta1,theta1,theta0,theta0])
#    #phi2s = np.array([phi2,phi2,phi3,phi3,phi2])
#    #theta_in2 = np.pi/8.
#    #phi_in2 = np.pi/12+2.*np.pi/3.
#    res_choose = 7
#
#
#
#    Theta1 = [theta0,theta1]
#    Phi1 = [phi0,phi1]
#    Theta2 = [theta0,theta1]
#    Phi2 = [phi2,phi3]
#
#    #zs=np.array([.4,0.8,1.2])
#    #zs=np.array([.6,0.8,1.01])
#    #zs=np.array([0.2,0.43,.63,0.9,1.178125])
#    zs = np.array([0.2,0.43,.63,0.9, 1.3])
#    z_fine = np.linspace(0.001,np.max(zs),2000) #use linspace
#    #z_fine = np.linspace(0.0005,np.max(zs),590)
#    #zbins=np.array([.2,.6,1.0])
#    #l=np.logspace(np.log10(2),np.log10(3000),1000)
#    l_sw = np.logspace(np.log(30),np.log(5000),base=np.exp(1.),num=10)
#    #l_sw = np.arange(0,50)
#    use_poly = True
#    use_poly2 = True
#    if use_poly:
#        if use_poly2:
#            geo1 = PolygonGeo(zs,theta1s,phi1s,theta_in1,phi_in1,C,z_fine,l_max,defaults.polygon_params.copy())
#            geo2 = PolygonGeo(zs,theta2s,phi2s,theta_in2,phi_in2,C,z_fine,l_max,defaults.polygon_params.copy())
#        else:
#            geo1 = PolygonPixelGeo(zs,theta1s,phi1s,theta_in1,phi_in1,C,z_fine,l_max,res_choose)
#            geo2 = PolygonPixelGeo(zs,theta2s,phi2s,theta_in2,phi_in2,C,z_fine,l_max,res_choose)
#    else:
#        geo1 = RectGeo(zs,Theta1,Phi1,C,z_fine)
#        geo2 = RectGeo(zs,Theta2,Phi2,C,z_fine)
#
#    len_params = defaults.lensing_params.copy()
##    len_params['z_min_dist'] = np.min(zs)
##    len_params['z_max_dist'] = np.max(zs)
#    len_params['pmodel'] = 'halofit'
#
#    lenless_defaults = defaults.sw_survey_params.copy()
#    lenless_defaults['needs_lensing'] = False
#    if cosmo_fid['de_model']=='w0wa':
#        cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w0','wa'])
#        cosmo_par_eps = np.array([0.002,0.0005,0.0001,0.0005,0.1,0.01,0.07])
#    elif cosmo_fid['de_model']=='constant_w':
#        cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w'])
#        cosmo_par_eps = np.array([0.002,0.0005,0.0001,0.0005,0.1,0.01])
#    elif cosmo_fid['de_model']=='jdem':
#        cosmo_par_list = ['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs']
#        cosmo_par_list.extend(cp.JDEM_LIST)
#        cosmo_par_list = np.array(cosmo_par_list,dtype=object)
#        cosmo_par_eps = np.full(41,0.5)
#        cosmo_par_eps[0:5] = np.array([0.002,0.0005,0.0001,0.0005,0.1])
#
#    else:
#        raise ValueError('not prepared to handle '+str(cosmo_fid['de_model']))
#
#    #cosmo_par_list = np.array(['LogAs','w'])
#    #cosmo_par_eps = np.array([0.001,0.001])
#    #param_priors = prior_fisher.get_w0wa_projected(params=defaults.prior_fisher_params.copy())
#    #param_priors = prior_fisher.get_w0_projected(params=defaults.prior_fisher_params.copy())
#    #cosmo_par_list = np.array(['Omegamh2','Omegabh2'])
#    #cosmo_par_eps = np.array([0.001,0.001])
#    #cosmo_par_list = np.array(['Omegamh2','Omegabh2','ns','h','sigma8'])
#    #note that currently (poorly implemented derivative) in jdem,
#    #OmegaLh2 and LogAs are both almost completely unconstrained but nondegenerate,
#    #while in basic, h and sigma8 are not constrained but are almost completely degenerate
#    nz_params = defaults.nz_params_wfirst_lens.copy()
#    #nz_params['data_source'] = 'data/H-5x140s.dat'
#    #nz_params['area_sterad'] =  0.040965*np.pi**2/180**2
#    #nz_params['smooth_sigma'] = 0.01
#    #nz_params['n_right_extend'] = 16
#
#    from nz_wfirst import NZWFirst
#    nz_matcher = NZWFirst(nz_params)
#    #len_params['n_gal'] = au.trapz2(nz_matcher.get_dN_dzdOmega(geo1.z_fine),geo1.z_fine)*geo1.angular_area()
#    len_params['smodel'] = 'nzmatcher'
#    sw_params = defaults.sw_survey_params.copy()
#    obs_list = defaults.sw_observable_list.copy()
#    survey_1 = SWSurvey(geo1,'s1',C,sw_params,cosmo_par_list,cosmo_par_eps,obs_list,len_params,nz_matcher)
#    #survey_2 = SWSurvey(geo1,'s2',C,sw_params,cosmo_par_list,cosmo_par_eps,obs_list,len_params,nz_matcher)
#
#    #survey_1 = SWSurvey(geo1,'s1',C,defaults.sw_survey_params.copy(),observable_list = np.array([]),len_params=len_params)
#    #survey_2 = SWSurvey(geo1,'s2',C,lenless_defaults, np.array([],dtype=object),np.array([]),np.array([]))
#
#    surveys_sw = np.array([survey_1])
#
#
#    geos = np.array([geo1,geo2])
#    #geos = np.array([geo1])
#    #l_lw=np.arange(0,30)
#    #n_zeros=49
#    #k_cut = 0.005
#    #k_cut = 0.016
#    #k_cut = 0.019
#    k_cut = 0.005
#    #about biggest possible, take 414 sec
#    #k_cut = 0.0214
#    #k_cut = 0.03
#
#    basis = SphBasisK(r_max,C,k_cut,defaults.basis_params.copy(),l_ceil=100)
#    lw_param_list = defaults.lw_param_list.copy()
#    lw_observable_list = defaults.lw_observable_list.copy()
#    survey_3 = LWSurvey(geos,'lw_survey1',basis,C,defaults.lw_survey_params.copy(),observable_list=lw_observable_list,param_list=lw_param_list)
#    surveys_lw = np.array([survey_3])
#n
#
#    print('main: this is r_max: '+str(r_max))
#    SS = SuperSurvey(surveys_sw,surveys_lw,basis,C,defaults.prior_fisher_params.copy(),get_a=False,do_unmitigated=True,do_mitigated=True,include_sw=True)
#
#    t2 = time()
#    print("main: total run time "+str(t2-t1)+" s")
#
#    #print("fractional mitigation: ", SS.a_no_mit/SS.a_mit)
#    #coeff = SS.basis.get_ddelta_bar_ddelta_alpha(SS.surveys_sw[0].geo,tomography=True)[0]
#    #rel_weights1 = coeff*np.dot(SS.multi_f.get_fisher(mf.f_spec_no_mit,mf.f_return_lw)[0].get_cov_cholesky(),coeff)
#    #rel_weights2 = coeff*np.dot(SS.multi_f.get_fisher(mf.f_spec_mit   ,mf.f_return_lw)[0].get_cov_cholesky(),coeff)
#
#
#
#    #mat_retrieved = (np.identity(chol_cov.shape[0])+np.dot(np.dot(chol_cov,SS.cov_no_mit[0,0]),chol_cov.T))
#    #eig_ret = np.linalg.eigvals(mat_retrieved)
#    #SS_eig =  SS.cov_no_mit[0].get_SS_eig()[0]
#    #SS_eig_mit =  SS.cov_mit[0].get_SS_eig()[0]
#    try:
#        #mit_eigs_sw = SS.f_set[2][1].get_cov_eig_metric(SS.f_set[0][1])
#        #no_mit_eigs_sw = SS.f_set[1][1].get_cov_eig_metric(SS.f_set[0][1])
#
#        #mit_eigs_par = SS.f_set[2][2].get_cov_eig_metric(SS.f_set[0][2])
#        #no_mit_eigs_par = SS.f_set[1][2].get_cov_eig_metric(SS.f_set[0][2])
#        mit_eigs_sw = SS.eig_set[0,1]
#        no_mit_eigs_sw = SS.eig_set[0,0]
#        mit_eigs_par = SS.eig_set[1,1]
#        no_mit_eigs_par = SS.eig_set[1,0]
#        print("main: unmitigated sw lambda1,2: "+str(no_mit_eigs_sw[0][-1])+","+str(no_mit_eigs_sw[0][-2]))
#        print("main: mitigated sw lambda1,2: "+str(mit_eigs_sw[0][-1])+","+str(mit_eigs_sw[0][-2]))
#        print("main: n sw mit lambda>1.00000001: "+str(np.sum(np.abs(mit_eigs_sw[0])>1.00000001)))
#        print("main: n sw no mit lambda>1.00000001: "+str(np.sum(np.abs(no_mit_eigs_sw[0])>1.00000001)))
#        print("main: unmitigated par lambda1,2: "+str(no_mit_eigs_par[0][-1])+","+str(no_mit_eigs_par[0][-2]))
#        print("main: mitigated par lambda1,2: "+str(mit_eigs_par[0][-1])+","+str(mit_eigs_par[0][-2]))
#        print("main: n par mit lambda>1.00000001: "+str(np.sum(np.abs(mit_eigs_par[0])>1.00000001)))
#        print("main: n par no mit lambda>1.00000001: "+str(np.sum(np.abs(no_mit_eigs_par[0])>1.00000001)))
#        v_no_mit_par = np.dot(SS.f_set_nopriors[0][2].get_cov_cholesky(),no_mit_eigs_par[1])
#        v_mit_par = np.dot(SS.f_set_nopriors[0][2].get_cov_cholesky(),mit_eigs_par[1])
#
#        v_no_mit_sw = np.dot(SS.f_set_nopriors[0][1].get_cov_cholesky(),no_mit_eigs_sw[1])
#        v_mit_sw = np.dot(SS.f_set_nopriors[0][1].get_cov_cholesky(),mit_eigs_sw[1])
#
#        test_v = True
#        v_test_fails = 0
#        if test_v:
#            m_mat_no_mit_par = np.identity(mit_eigs_par[0].size)+np.dot(SS.f_set_nopriors[1][2].get_covar(),SS.f_set_nopriors[0][2].get_fisher())
#            m_mat_mit_par = np.identity(mit_eigs_par[0].size)+np.dot(SS.f_set_nopriors[2][2].get_covar(),SS.f_set_nopriors[0][2].get_fisher())
#
#            if not np.allclose(np.dot(m_mat_no_mit_par,v_no_mit_par)/(1.+no_mit_eigs_par[0]),v_no_mit_par):
#                v_test_fails+=1
#                warn('some no mit eig vectors may be bad')
#            if not np.allclose(np.dot(m_mat_mit_par,v_mit_par)/(1.+mit_eigs_par[0]),v_mit_par):
#                v_test_fails+=1
#                warn('some mit eig vectors may be bad')
#        if v_test_fails == 0:
#            print("PASS: eigenvector decomposition checks passed")
#        else:
#            warn('FAIL: '+str(v_test_fails)+' eigenvector decomposition checks failed')
#
#
#    except Exception:
#        warn('Eig value failed')
##    print("SS eigenvals:",SS_eig[0])
##    print("(S/N)^2 gaussian: ",np.dot(np.dot(c_ss,np.linalg.inv(cov_ss)),c_ss))
#    #print("(S/N)^2 gaussian+no mitigation: ",np.dot(np.dot(c_ss,np.linalg.inv(np.diagflat(cov_ss)+SS.cov_no_mit[0,0])),c_ss))
#    #print("(S/N)^2 gaussian+mitigation: ",np.dot(np.dot(c_ss,np.linalg.inv(np.diagflat(cov_ss)+SS.cov_mit[0,0])),c_ss))
#    #ax.loglog(l_sw,cov_ss)
#    #ax.loglog(l,np.diag(SS.cov_mit[0,0])/c_ss**2*l/2)
#    #ax.loglog(l_sw,(np.diag(SS.cov_no_mit[0,0])/cov_ss))
#    #ax.legend(['cov','mit','no mit'])
#    #plt.show()
#    get_hold_mats = False
#    if get_hold_mats:
#        no_prior_hold = SS.f_set[0][2].get_fisher()
#        if C.de_model=='jdem':
#            no_prior_project = prior_fisher.project_w0wa(no_prior_hold,defaults.prior_fisher_params.copy(),prior_fisher.JDEM_LABELS)
#
#    print('main: r diffs',np.diff(geo1.rs))
#    print('main: theta width',(geo1.rs[1]+geo1.rs[0])/2.*(Theta1[1]-Theta1[0]))
#    print('main: phi width',(geo1.rs[1]+geo1.rs[0])/2.*(Phi1[1]-Phi1[0])*np.sin((Theta1[1]+Theta1[0])/2))
#    #ax_ls = np.hstack((l_sw,l_sw))
#
#    #v = SS.surveys_sw[0].get_dO_I_dpar_array()
#    #eig_nm = SS.cov_no_mit[0].get_SS_eig_param(v)
#    #eig_m = SS.cov_mit[0].get_SS_eig_param(v)
##    ellipse_plot_setup = True
##    if ellipse_plot_setup:
##        no_mit_color = np.array([1.,0.,0.])
##        mit_color = np.array([0.,1.,0.])
##        g_color = np.array([0.,0.,1.])
##        color_set = np.array([mit_color,no_mit_color,g_color])
##        opacity_set = np.array([1.0,1.0,1.0])
##        box_widths = np.array([0.015,0.005,0.0005,0.005,0.1,0.05])
##        dchi2 = 2.3
##        #cov_set = np.array([SS.covs_params[1],SS.covs_params[0],SS.covs_g_pars[0]])
##        cov_set = np.array([SS.f_set[2][2].get_covar(),SS.f_set[1][2].get_covar(),SS.f_set[0][2].get_covar()])
##        label_set = np.array(["ssc+mit+g","ssc+g","g"])
#
##    ellipse_plot = False
##    if ellipse_plot:
##        make_ellipse_plot(cov_set,color_set,opacity_set,label_set,'adaptive',cosmo_par_list,dchi2=dchi2)
##        SS.make_standard_ellipse_plot()
#
#
#    test_perturbation = True
#    pert_test_fails = 0
#    if test_perturbation:
#        #TOLERANCE below which an eigenvalue less than TOLERANCE*max eigenvalue is considered 0
#        REL_TOLERANCE = 10**-8
#        f0 = SS.multi_f.get_fisher(mf.f_spec_no_mit,mf.f_return_lw)[0].get_fisher()
#        f1 = SS.multi_f.get_fisher(mf.f_spec_mit,mf.f_return_lw)[0].get_fisher()
#        if not np.all(f0.T==f0):
#            pert_test_fails+=1
#            warn("unperturbed fisher matrix not symmetric, unacceptable")
#        if not np.all(f1.T==f1):
#            pert_test_fails+=1
#            warn("perturbed fisher matrix not symmetric, unacceptable")
#        #get eigenvalues and set numerically zero values to 0
#        eigf0 = np.linalg.eigh(f0)[0]
#        eigf0[np.abs(eigf0)<REL_TOLERANCE*np.max(np.abs(eigf0))] = 0.
#        eigf1 = np.linalg.eigh(f1)[0]
#        eigf1[np.abs(eigf1)<REL_TOLERANCE*np.max(np.abs(eigf1))] = 0.
#        #check positive semidefinite
#        if np.any(eigf0<0.):
#            pert_test_fails+=1
#            warn("unperturbed fisher matrix not positive definite within tolerance, unacceptable")
#        if np.any(eigf1<0.):
#            pert_test_fails+=1
#            warn("perturbed fisher matrix not positive definite within tolerance, unacceptable")
#
#        #check nondecresasing
#        diff_eig = eigf1-eigf0
#        diff_eig[np.abs(diff_eig)<REL_TOLERANCE*np.max(np.abs(diff_eig))] = 0
#        if np.any(diff_eig<0):
#            pert_test_fails+=1
#            warn("some eigenvalues decreased within tolerance, unacceptable")
#
#        #check interlace theorem satisfied (eigenvalues cannot be reordered by more than rank of perturbation)
#        n_offset = SS.surveys_lw[0].get_total_rank()
#        rolled_eig = (eigf1[::-1][n_offset:eigf0.size]-eigf0[::-1][0:eigf0.size-n_offset])
#        rolled_eig[np.abs(rolled_eig)<REL_TOLERANCE*np.max(np.abs(rolled_eig))] = 0.
#        if np.any(rolled_eig>0):
#            pert_test_fails+=1
#            warn("some eigenvalues fail interlace theorem, unacceptable")
#
#        c0 = SS.multi_f.get_fisher(mf.f_spec_no_mit,mf.f_return_lw)[0].get_covar()
#        c1 = SS.multi_f.get_fisher(mf.f_spec_mit,mf.f_return_lw)[0].get_covar()
#        if not np.all(c0==c0.T):
#            pert_test_fails+=1
#            warn("unperturbed covariance not symmetric, unacceptable")
#        if not np.all(c1==c1.T):
#            warn("perturbed covariance not symmetric, unacceptable")
#        eigc0 = np.linalg.eigh(c0)[0]
#        eigc1 = np.linalg.eigh(c1)[0]
#        if np.any(eigc0<0):
#            pert_test_fails+=1
#            warn("unperturbed covariance not positive semidefinite, unacceptable")
#        if np.any(eigc1<0):
#            pert_test_fails+=1
#            warn("perturbed covariance not positive semidefinite, unacceptable")
#        fdiff_eigc = (eigc1-eigc0)/eigc0
#        fdiff_eigc[np.abs(fdiff_eigc)<REL_TOLERANCE] = 0.
#        if np.any(fdiff_eigc>0):
#            pert_test_fails+=1
#            warn("some covariance eigenvalues increase, unacceptable")
#
#        if pert_test_fails==0:
#            print("PASS: All fisher matrix sanity checks passed")
#        else:
#            warn("FAIL: "+str(pert_test_fails)+" fisher matrix sanity checks failed")
#    test_eigs = True
#    eig_test_fails = 0
#    if test_eigs:
#        REL_TOLERANCE = 10**-8
#        #c_sscs = SS.multi_f.get_fisher(mf.f_spec_SSC_no_mit,mf.f_return_sw).get_covar()#SS.covs_sw[0].get_ssc_covar()
#        c_ssc0 = SS.multi_f.get_fisher(mf.f_spec_SSC_no_mit,mf.f_return_sw)[1].get_covar()#SS.covs_sw[0].get_ssc_covar()
#        #c_ssc0 = c_sscs[0]
#        if not np.allclose(c_ssc0,c_ssc0.T):
#            eig_test_fails+=1
#            warn("unperturbed result covariance not symmetric, unacceptable")
#        #c_ssc1 = c_sscs[1]
#        c_ssc1 = SS.multi_f.get_fisher(mf.f_spec_SSC_mit,mf.f_return_sw)[1].get_covar()#SS.covs_sw[0].get_ssc_covar()
#        if not np.allclose(c_ssc1,c_ssc1.T):
#            eig_test_fails+=1
#            warn("perturbed result covariance not symmetric, unacceptable")
#        eigsys_ssc0 = np.linalg.eigh(c_ssc0)
#        eigsys_ssc1 = np.linalg.eigh(c_ssc1)
#        eig_ssc0 = eigsys_ssc0[0].copy()
#        eig_ssc1 = eigsys_ssc1[0].copy()
#        eig_ssc0[np.abs(eig_ssc0)<np.max(np.abs(eig_ssc0))*REL_TOLERANCE] = 0
#        eig_ssc1[np.abs(eig_ssc0)<np.max(np.abs(eig_ssc0))*REL_TOLERANCE] = 0
#        if np.any(eig_ssc0<0):
#            eig_test_fails+=1
#            warn("unperturbed result cov not positive semidefinite, unacceptable")
#        if np.any(eig_ssc1<0):
#            eig_test_fails+=1
#            warn("perturbed result cov not positive semidefinite, unacceptable")
#        cg = SS.f_set_nopriors[0][1].get_covar()
#        eigsys_cg = np.linalg.eigh(cg)
#        eig_cg = eigsys_cg[0].copy()
#        eig_mitprod = np.real(np.linalg.eig(np.dot(np.linalg.inv(c_ssc0+cg),c_ssc1+cg))[0])
#        eig_mitprod[np.abs(eig_mitprod-1.)<REL_TOLERANCE] = 1.
#        if np.any(eig_mitprod>1):
#            eig_test_fails+=1
#            warn("mitigation making covariance worse, unacceptable")
#        n_offset = SS.surveys_lw[0].get_total_rank()
#        if np.sum(eig_mitprod<1.)>n_offset:
#            eig_test_fails+=1
#            warn("mitigation changing too many eigenvalues, unacceptable")
#        eig_diff = eig_ssc1-eig_ssc0
#        eig_diff[np.abs(eig_diff)<np.max(np.abs(eig_diff))*REL_TOLERANCE] = 0.
#        if np.any(eig_diff>0):
#            eig_test_fails+=1
#            warn("mitigation making covariance worse, unacceptable")
#
#
#        if eig_test_fails==0:
#            print("PASS: All sw eigenvalue sanity checks passed")
#        else:
#            warn("FAIL: "+str(pert_test_fails)+" eigenvalue sanity checks failed")
#
#        do_eig_interlace_check = True
#        if do_eig_interlace_check:
#            eig_interlace_fails_mit = 0
#            eig_interlace_fails_no_mit = 0
#            n_sw = mit_eigs_sw[0].size
#            n_par = mit_eigs_par[0].size
#            d_n = n_sw-n_par
#            eig_l_mit_par = mit_eigs_par[0][::-1]
#            eig_l_no_mit_par = no_mit_eigs_par[0][::-1]
#            eig_l_mit_sw = mit_eigs_sw[0][::-1]
#            eig_l_no_mit_sw = no_mit_eigs_sw[0][::-1]
#            for i in range(0,n_par):
#                if eig_l_mit_par[i]>eig_l_mit_sw[i]:
#                    eig_interlace_fails_mit+=1
#                if eig_l_no_mit_par[i]>eig_l_no_mit_sw[i]:
#                    eig_interlace_fails_no_mit+=1
#                if eig_l_mit_par[i]<eig_l_mit_sw[i+d_n]:
#                    eig_interlace_fails_mit+=1
#                if eig_l_no_mit_par[i]<eig_l_no_mit_sw[i+d_n]:
#                    eig_interlace_fails_no_mit+=1
#            if eig_interlace_fails_mit==0 and eig_interlace_fails_no_mit==0:
#                print("PASS: All parameter eigenvalue interlace tests passed")
#            else:
#                warn("FAIL: "+str(eig_interlace_fails_mit)+" mitigation and "+str(eig_interlace_fails_no_mit)+" no mitigation failures in parameter eigenvalue interlace tests")
