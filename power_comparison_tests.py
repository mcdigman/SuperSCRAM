"""test consistency of different ways of getting power spectrum"""
#pylint: disable=W0621
import numpy as np
import pytest
from scipy.interpolate import InterpolatedUnivariateSpline
import camb

import defaults
from camb_power import camb_pow
import matter_power_spectrum as mps
import cosmopie as cp

#test h independence of code, self consistent handling of h
#note my code does not recover the low k dependence of the transfer functions on z
test_list = [[False,False,'halofit'],[False,True,'linear'],[False,True,'halofit']]
@pytest.fixture(params=test_list)
def param_set(request):
    """choose params from test_list"""
    return request.param

vary_list = defaults.cosmology.keys()
@pytest.fixture(params=vary_list)
def param_vary(request):
    """iterate over all possible params, including unused ones/params that should not matter for robustness"""
    return request.param

def test_vary_1_parameter(param_set,param_vary):
    """test variation of parameters for halofit and linear match expectations from camb"""
    atol_rel = 1.e-8
    rtol = 3.e-3
    eps = 0.01

    camb_params = defaults.camb_params.copy()

    camb_params['force_sigma8'] = param_set[0]
    camb_params['leave_h'] = param_set[1]
    power_params = defaults.power_params.copy()
    power_params.camb = camb_params

    cosmo_fid = defaults.cosmology.copy()

    if param_set[2]=='halofit':
        nonlinear_model = camb.model.NonLinear_both
    else:
        nonlinear_model = camb.model.NonLinear_none

    if isinstance(cosmo_fid[param_vary],float):
        cosmo_pert = cosmo_fid.copy()
        cosmo_pert[param_vary]*=(1.+eps)

        C_pert = cp.CosmoPie(cosmo_pert,p_space='jdem')
        P_pert = mps.MatterPower(C_pert,power_params)
        k_pert = P_pert.k
        C_pert.k = k_pert

        P_res1 = P_pert.get_matter_power(np.array([0.]),pmodel=param_set[2])[:,0]
        k_res2,P_res2 = camb_pow(C_pert.cosmology,zbar=np.array([0.]),camb_params=camb_params,nonlinear_model=nonlinear_model)

        atol_power = np.max(P_res1)*atol_rel
        atol_k = np.max(k_pert)*atol_rel

        assert np.allclose(k_res2,k_pert,atol=atol_k,rtol=rtol)
        assert np.allclose(P_res1,P_res2,atol=atol_power,rtol=rtol)


if __name__=='__main__':
    do_pytest = False
    if do_pytest:
        pytest.cmdline.main(['power_comparison_tests.py'])
    do_linear_test = True
    do_plots = True
    if do_plots:
        import matplotlib.pyplot as plt
    #param_sets = [[False,False,'halofit'],[False,True,'linear'],[False,True,'halofit']]
    param_sets = [[False,True,'halofit']]
    if do_linear_test:
        for param in param_sets:
            atol_rel = 1.e-8
            #there is an ~0.05% discrepancy with camb halofit, probably due to the change of spectral_parameters
            #this version may or may not be better converged, convergence doesn't really seem to be the problem
            #for some reason the discrepancy is a function of pivot_scalar and is minimizaed around pivot_scalar=0.01-0.0001
            rtol = 3.e-3
            eps = 0.01

            cosmo_fid = defaults.cosmology.copy()
            camb_params = defaults.camb_params.copy()
            camb_params['force_sigma8'] = param[0]
            camb_params['leave_h'] = param[1]
            camb_params['npoints'] = 3000
            camb_params['kmax'] = 2.
            camb_params['maxkh'] = 2.
            power_params = defaults.power_params.copy()
            power_params.camb = camb_params
            #camb_params['minkh'] = 1e-3

            C_fid = cp.CosmoPie(cosmo_fid,p_space='jdem')
            P_fid = mps.MatterPower(C_fid,power_params)
            k_fid = P_fid.k
            C_fid.P_lin = P_fid
            P_lin1 = P_fid.get_matter_power(np.array([0.]),pmodel='linear')[:,0]
            C_fid.k = k_fid

            P_res1 = P_fid.get_matter_power(np.array([0.]),pmodel=param[2])[:,0]
            if param[2]=='halofit':
                nonlinear_model = camb.model.NonLinear_both
            else:
                nonlinear_model = camb.model.NonLinear_none

            k_res2,P_res2 = camb_pow(C_fid.cosmology,zbar=np.array([0.]),camb_params=camb_params,nonlinear_model=nonlinear_model)
            #k_res2 = k_fid
            #P_res2 = P_fid.get_matter_power(np.array([0.]),pmodel='linear')[:,0]


            #atol_power = np.max(P_res1)*atol_rel
            atol_k = np.max(k_fid)*atol_rel

            k_mask = (k_fid<10.)

            if do_plots:
                #plt.loglog(k_fid[k_mask],P_lin1[k_mask])
           #     plt.plot(k_fid[k_mask],(P_res1-P_lin1)[k_mask])
           #     plt.plot(k_res2[k_mask],(P_res2-P_lin1)[k_mask])
                plt.semilogx(k_fid[k_mask],((P_res1-P_res2)/P_res2)[k_mask])

            mean_abs_err_lin = np.average(np.abs((P_res2-P_res1)/P_res2)[k_mask])
            print "mean absolute diff leave_h="+str(param[1])+" force_sigma8="+str(param[0])+" pmodel="+str(param[2])+": "+str(mean_abs_err_lin)
            #assert np.allclose(k_res2,k_fid,atol=atol_k,rtol=rtol)
            #assert np.allclose(P_res1,P_res2,atol=atol_power,rtol=rtol)
        #print "PASS: passed all assertions"
#        do_jvp_comp = True
#        if do_jvp_comp:
#            from halofit2 import PowerSpectrum
#            jvp_halo = PowerSpectrum(0.,C_fid.Omegam,C_fid.OmegaL,C_fid.get_sigma8(),0.21)
#            P_jvp = 2.*np.pi**2*jvp_halo.D2_NL(k_fid)/k_fid**3
#            plt.semilogx(k_fid,(P_res1-P_jvp)/P_jvp)
        k_res3,P_lin3 = camb_pow(C_fid.cosmology,zbar=np.array([0.]),camb_params=camb_params,nonlinear_model=camb.model.NonLinear_none)
        P_interp1 = InterpolatedUnivariateSpline(k_fid,P_res1,k=3)
        P_interp2 = InterpolatedUnivariateSpline(k_fid,P_res2,k=3)
        P_interp3 = InterpolatedUnivariateSpline(k_fid,P_lin3,k=3)
        #plt.semilogx(k_fid[k_mask],P_interp1.derivative(2)(k_fid[k_mask]))
        #plt.semilogx(k_fid[k_mask],P_interp2.derivative(2)(k_fid[k_mask]))
        #plt.semilogx(k_fid[k_mask],P_interp3.derivative(2)(k_fid[k_mask]))
        #plt.xlabel('k')
        #plt.ylabel('d^2 P/dk^2')
        if do_plots:
            plt.show()

        #plt.semilogx(k_fid[k_mask],(P_res2/P_lin3)[k_mask])
