"""tests of WMatcher class"""
#pylint: disable=W0621
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import pytest

import defaults
import cosmopie as cp
from w_matcher import WMatcher

#insert a random test to better test grid effects
w0_list = [-0.8,np.random.uniform(-1.,-0.8),-1.0,np.random.uniform(-1.2,-1.),-1.2]
#w0_list = [np.random.uniform(-1.,-0.8)]
#w0_list = [-0.900193719903521,-0.8]
@pytest.fixture(params=w0_list)
def w0_test(request):
    """get w0 for test"""
    return request.param

wa_list = [-0.5,np.random.uniform(0.,-0.5),0.,np.random.uniform(0.,0.5),0.5]
@pytest.fixture(params=wa_list)
def wa_test(request):
    """get wa for test"""
    return request.param

@pytest.fixture(scope='module')
def cosmo_input():
    """get cosmology for test"""
    cosmo_start = defaults.cosmology.copy()
    cosmo_start['w'] = -1
    cosmo_start['de_model'] = 'constant_w'
    C_start = cp.CosmoPie(cosmology=cosmo_start,p_space='jdem')
    #base set
    #params = {'w_step':0.005,'w_min':-3.50,'w_max':0.1,'a_step':0.001,'a_min':0.000916674,'a_max':1.00}
    #mod se
    params = {'w_step':0.005,'w_min':-3.5,'w_max':0.1,'a_step':0.001,'a_min':0.000916674,'a_max':1.00}

    wm = WMatcher(C_start,params)
    a_grid = np.arange(1.00,0.001,-0.01)
    zs = 1./a_grid-1.

    return [cosmo_start,C_start,params,wm,a_grid,zs]

def test_const_match(w0_test,cosmo_input):
    """test w_matcher gets constant w cosmology correct"""
    cosmo_start = cosmo_input[0]
    wm = cosmo_input[3]
    zs = cosmo_input[5]

    w_use_int = w0_test
    cosmo_match_a = cosmo_start.copy()
    cosmo_match_a['de_model'] = 'jdem'
    cosmo_match_a['w0'] = w_use_int
    cosmo_match_a['wa'] = 0.
    cosmo_match_a['w'] = w_use_int
    for i in xrange(0,36):
        cosmo_match_a['ws36_'+str(i).zfill(2)] = w_use_int

    cosmo_match_b = cosmo_match_a.copy()
    cosmo_match_b['de_model'] = 'w0wa'

    cosmo_match_c = cosmo_match_a.copy()
    cosmo_match_c['de_model'] = 'constant_w'

    C_match_a = cp.CosmoPie(cosmology=cosmo_match_a,p_space='jdem')
    C_match_b = cp.CosmoPie(cosmology=cosmo_match_b,p_space='jdem')
    C_match_c = cp.CosmoPie(cosmology=cosmo_match_c,p_space='jdem')


    w_a = wm.match_w(C_match_a,zs)
    w_b = wm.match_w(C_match_b,zs)
    w_c = wm.match_w(C_match_c,zs)

    mult_a = wm.match_growth(C_match_a,zs,w_a)
    mult_b = wm.match_growth(C_match_b,zs,w_b)
    mult_c = wm.match_growth(C_match_c,zs,w_c)

    error_a_1 = np.linalg.norm(w_a-C_match_a.de_object.w_of_z(zs))/w_a.size
    error_a_2 = np.linalg.norm(w_use_int-w_a)/zs.size
    error_b_1 = np.linalg.norm(w_b-C_match_b.de_object.w_of_z(zs))/w_b.size
    error_b_2 = np.linalg.norm(w_use_int-w_b)/zs.size
    error_c_1 = np.linalg.norm(w_c-C_match_c.de_object.w_of_z(zs))/w_c.size
    error_c_2 = np.linalg.norm(w_use_int-w_c)/zs.size
    #should usually do more like 1.e-12 or better if no grid issues
    atol_loc = 1.e-10
    assert error_a_1<atol_loc
    assert error_a_2<atol_loc
    assert error_b_1<atol_loc
    assert error_b_2<atol_loc
    assert error_c_1<atol_loc
    assert error_c_2<atol_loc
    assert np.allclose(w_a,C_match_a.de_object.w_of_z(zs))
    assert np.allclose(w_b,C_match_b.de_object.w_of_z(zs))
    assert np.allclose(w_c,C_match_c.de_object.w_of_z(zs))
    assert np.allclose(w_a,w_use_int)
    assert np.allclose(w_b,w_use_int)
    assert np.allclose(w_c,w_use_int)
    assert np.allclose(mult_a,mult_b)
    assert np.allclose(mult_a,mult_c)
    assert np.allclose(mult_b,mult_c)

def test_jdem_w0wa_match(w0_test,wa_test,cosmo_input):
    """check w_matcher gets w0wa cosmology compatible with binned w(z) cosmology"""
    cosmo_start = cosmo_input[0]
    wm = cosmo_input[3]
    zs = cosmo_input[5]


    w0_use = w0_test
    wa_use = wa_test
    cosmo_match_w0wa = cosmo_start.copy()
    cosmo_match_jdem = cosmo_start.copy()
    cosmo_match_w0wa['de_model'] = 'w0wa'
    cosmo_match_w0wa['w0'] = w0_use
    cosmo_match_w0wa['wa'] = wa_use
    cosmo_match_w0wa['w'] = w0_use+0.9*wa_use

    cosmo_match_jdem['de_model'] = 'jdem'
    cosmo_match_jdem['w'] = w0_use+0.9*wa_use
    a_jdem = 1.-0.025*np.arange(0,36)
    for i in xrange(0,36):
        cosmo_match_jdem['ws36_'+str(i).zfill(2)] = w0_use+(1.-(a_jdem[i]-0.025/2.))*wa_use

    C_match_w0wa = cp.CosmoPie(cosmology=cosmo_match_w0wa,p_space='jdem')
    C_match_jdem = cp.CosmoPie(cosmology=cosmo_match_jdem,p_space='jdem')

    w_w0wa = wm.match_w(C_match_w0wa,zs)
    w_jdem = wm.match_w(C_match_jdem,zs)
    mult_w0wa = wm.match_growth(C_match_w0wa,zs,w_w0wa)
    mult_jdem = wm.match_growth(C_match_jdem,zs,w_jdem)

    error_w0wa_jdem = np.linalg.norm(w_w0wa-w_jdem)/w_jdem.size
    error_mult_w0wa_jdem = np.linalg.norm(mult_w0wa-mult_jdem)/w_jdem.size
    assert error_w0wa_jdem<4.e-4
    assert error_mult_w0wa_jdem<1.e-3
    #icurrent agreement ~0.0627 is reasonable given imperfect approximation of w0wa
    #depends on setting of default value for jdem at end

def test_casarini_match(cosmo_input):
    """check code matches results extracted from a figure"""
    cosmo_start = cosmo_input[0]
    wm = cosmo_input[3]
    #zs = cosmo_input[5]

    #should match arXiv:1601.07230v3 figure 2
    cosmo_match_a = cosmo_start.copy()
    cosmo_match_a['de_model'] = 'w0wa'
    cosmo_match_a['w0'] = -1.2
    cosmo_match_a['wa'] = 0.5
    cosmo_match_a['w'] = -1.2

    cosmo_match_b = cosmo_match_a.copy()
    cosmo_match_b['w0'] = -0.6
    cosmo_match_b['wa'] = -1.5
    cosmo_match_b['w'] = -0.6


    C_match_a = cp.CosmoPie(cosmology=cosmo_match_a,p_space='jdem')
    C_match_b = cp.CosmoPie(cosmology=cosmo_match_b,p_space='jdem')



    weff_2 = np.loadtxt('test_inputs/wmatch/weff_2.dat')
    ws_a_pred = weff_2[:,1]
    zs_a_pred = weff_2[:,0]


    weff_1 = np.loadtxt('test_inputs/wmatch/weff_1.dat')
    ws_b_pred = weff_1[:,1]
    zs_b_pred = weff_1[:,0]

    z_max_a = np.max(zs_a_pred)
    z_max_b = np.max(zs_b_pred)
    z_min_a = np.min(zs_a_pred)
    z_min_b = np.min(zs_b_pred)
    zs_a = np.arange(z_min_a,z_max_a,0.05)
    zs_b = np.arange(z_min_b,z_max_b,0.05)

    ws_a = wm.match_w(C_match_a,zs_a)
    ws_b = wm.match_w(C_match_b,zs_b)

    ws_a_interp = InterpolatedUnivariateSpline(zs_a_pred,ws_a_pred,k=1,ext=2)(zs_a)
    ws_b_interp = InterpolatedUnivariateSpline(zs_b_pred,ws_b_pred,k=1,ext=2)(zs_b)

    ws_a = wm.match_w(C_match_a,zs_a)
    ws_b = wm.match_w(C_match_b,zs_b)

    mse_w_a = np.linalg.norm((ws_a-ws_a_interp)/ws_a_interp)/ws_a_interp.size
    mse_w_b = np.linalg.norm((ws_b-ws_b_interp)/ws_b_interp)/ws_b_interp.size
    print mse_w_a,mse_w_b
    assert mse_w_a<7.e-4
    assert mse_w_b<7.e-4


    sigma_a_in = np.loadtxt('test_inputs/wmatch/sigma_2.dat')
    sigma_b_in = np.loadtxt('test_inputs/wmatch/sigma_1.dat')

    zs_a_pred_2 = sigma_a_in[:,0]
    zs_b_pred_2 = sigma_a_in[:,0]
    z_max_a_2 = np.max(zs_a_pred_2)
    z_max_b_2 = np.max(zs_b_pred_2)
    z_min_a_2 = np.min(zs_a_pred_2)
    z_min_b_2 = np.min(zs_b_pred_2)
    zs_a_2 = np.arange(z_min_a_2,z_max_a_2,0.05)
    zs_b_2 = np.arange(z_min_b_2,z_max_b_2,0.05)

    pow_mults_a = wm.match_growth(C_match_a,zs_a_2,ws_a)
    pow_mults_b = wm.match_growth(C_match_b,zs_b_2,ws_b)

    sigma_a_interp = InterpolatedUnivariateSpline(sigma_a_in[:,0],sigma_a_in[:,1],k=3,ext=2)(zs_a_2)
    sigma_b_interp = InterpolatedUnivariateSpline(sigma_b_in[:,0],sigma_b_in[:,1],k=3,ext=2)(zs_b_2)

    sigma_a = np.sqrt(pow_mults_a)*0.83
    sigma_b = np.sqrt(pow_mults_b)*0.83

    mse_sigma_a = np.linalg.norm((sigma_a-sigma_a_interp)/sigma_a_interp)/sigma_a_interp.size
    mse_sigma_b = np.linalg.norm((sigma_b-sigma_b_interp)/sigma_b_interp)/sigma_b_interp.size
    print mse_sigma_a,mse_sigma_b
    assert mse_sigma_a<1.e-4
    assert mse_sigma_b<1.e-4


if __name__=='__main__':
    do_test_battery = True
    do_other_tests = False
    if do_test_battery:
        pytest.cmdline.main(['w_matcher_tests.py'])

    if do_other_tests:
        cosmo_start = defaults.cosmology.copy()
        cosmo_start['w'] = -1
        cosmo_start['de_model'] = 'constant_w'
        C_start = cp.CosmoPie(cosmology=cosmo_start,p_space='jdem')
        params = {'w_step':0.01,'w_min':-3.50,'w_max':0.1,'a_step':0.001,'a_min':0.000916674,'a_max':1.00}

        do_convergence_test_w0wa = False
        do_convergence_test_jdem = False
        do_match_casarini = True
        do_plots = True

        fails = 0

        if do_plots:
            import matplotlib.pyplot as plt

        if do_convergence_test_w0wa:
            cosmo_match_w0wa = cosmo_start.copy()
            cosmo_match_w0wa['de_model'] = 'w0wa'
            cosmo_match_w0wa['w0'] = -1.0
            cosmo_match_w0wa['wa'] = 0.5
            cosmo_match_w0wa['w'] = -1.0+0.1*0.5

            params_1 = params.copy()
            params_1['w_step'] = 0.01
            params_2 = params.copy()
            params_2['w_step'] = 0.001
            params_3 = params.copy()
            params_3['a_step'] = params['a_step']/10.

            C_match_w0wa = cp.CosmoPie(cosmology=cosmo_match_w0wa,p_space='jdem')

            a_grid = np.arange(1.00,0.001,-0.01)
            zs = 1./a_grid-1.

            wm_1 = WMatcher(C_start,params_1)
            wm_2 = WMatcher(C_start,params_2)
            wm_3 = WMatcher(C_start,params_3)

            w_w0wa_1 = wm_1.match_w(C_match_w0wa,zs)
            w_w0wa_2 = wm_2.match_w(C_match_w0wa,zs)
            w_w0wa_3 = wm_3.match_w(C_match_w0wa,zs)
            print "rms discrepancy w0wa_1 and w0wa_2="+str(np.linalg.norm(w_w0wa_1-w_w0wa_2)/w_w0wa_1.size)
            print "rms % discrepancy w0wa_1 and w0wa_2="+str(np.linalg.norm((w_w0wa_1-w_w0wa_2)/w_w0wa_2)/w_w0wa_1.size)
            print "rms discrepancy w0wa_1 and w0wa_3="+str(np.linalg.norm(w_w0wa_1-w_w0wa_3)/w_w0wa_1.size)
            print "rms % discrepancy w0wa_1 and w0wa_3="+str(np.linalg.norm((w_w0wa_1-w_w0wa_3)/w_w0wa_3)/w_w0wa_1.size)

        if do_convergence_test_jdem:
            cosmo_match_jdem = cosmo_start.copy()
            cosmo_match_jdem['de_model'] = 'jdem'
            cosmo_match_jdem['w0'] = -1.0
            cosmo_match_jdem['wa'] = 0.5
            cosmo_match_jdem['w'] = -1.0+0.1*0.5


            for i in xrange(0,36):
                cosmo_match_jdem['ws36_'+str(i).zfill(2)] = -1.
            cosmo_match_jdem['ws36_'+str(1).zfill(2)] = -1.5

            params_1 = params.copy()
            params_1['w_step'] = 0.01
            params_2 = params.copy()
            params_2['w_step'] = 0.005
            params_3 = params.copy()
            params_3['a_step'] = params_1['a_step']/10.


            C_match_jdem = cp.CosmoPie(cosmology=cosmo_match_jdem,p_space='jdem')

            a_grid = np.arange(1.00,0.001,-0.01)
            zs = 1./a_grid-1.

            wm_1 = WMatcher(C_start,params_1)
            wm_2 = WMatcher(C_start,params_2)
            wm_3 = WMatcher(C_start,params_3)

            w_jdem_1 = wm_1.match_w(C_match_jdem,zs)
            w_jdem_2 = wm_2.match_w(C_match_jdem,zs)
            w_jdem_3 = wm_3.match_w(C_match_jdem,zs)
            print "rms discrepancy jdem_1 and jdem_2="+str(np.linalg.norm(w_jdem_1-w_jdem_2)/w_jdem_1.size)
            print "mean absolute % discrepancy jdem_1 jdem_2="+str(np.average(np.abs((w_jdem_1-w_jdem_2)/w_jdem_1)/w_jdem_1.size*100.))
            print "rms discrepancy jdem_1 and jdem_3="+str(np.linalg.norm(w_jdem_1-w_jdem_3)/w_jdem_1.size)
            print "mean absolute % discrepancy jdem_1 jdem_3="+str(np.average(np.abs((w_jdem_1-w_jdem_3)/w_jdem_1)/w_jdem_1.size*100.))

        if do_match_casarini:
            #should match arXiv:1601.07230v3 figure 2
            cosmo_match_a = cosmo_start.copy()
            cosmo_match_a['de_model'] = 'w0wa'
            cosmo_match_a['w0'] = -1.2
            cosmo_match_a['wa'] = 0.5
            cosmo_match_a['w'] = -1.2

            cosmo_match_b = cosmo_match_a.copy()
            cosmo_match_b['w0'] = -0.6
            cosmo_match_b['wa'] = -1.5
            cosmo_match_b['w'] = -0.6


            wmatcher_params = defaults.wmatcher_params.copy()
            C_start = cp.CosmoPie(cosmology=cosmo_start,p_space='jdem')
            C_match_a = cp.CosmoPie(cosmology=cosmo_match_a,p_space='jdem')
            C_match_b = cp.CosmoPie(cosmology=cosmo_match_b,p_space='jdem')

            wm = WMatcher(C_start,wmatcher_params)
            zs = np.arange(0.,1.51,0.05)
            a_s = 1./(1.+zs)

            ws_a = wm.match_w(C_match_a,zs)
            ws_b = wm.match_w(C_match_b,zs)

            weff_2 = np.loadtxt('test_inputs/wmatch/weff_2.dat')
            ws_a_pred = weff_2[:,1]
            zs_a_pred = weff_2[:,0]


            weff_1 = np.loadtxt('test_inputs/wmatch/weff_1.dat')
            ws_b_pred = weff_1[:,1]
            zs_b_pred = weff_1[:,0]

            ws_a_interp = InterpolatedUnivariateSpline(zs_a_pred,ws_a_pred,k=3)(zs)
            ws_b_interp = InterpolatedUnivariateSpline(zs_b_pred,ws_b_pred,k=3)(zs)

            mse_w_a = np.linalg.norm((ws_a-ws_a_interp)/ws_a_interp)/ws_a_interp.size
            mse_w_b = np.linalg.norm((ws_b-ws_b_interp)/ws_b_interp)/ws_b_interp.size
            print "mse a/point: "+str(mse_w_a)
            print "mse b/point: "+str(mse_w_b)

            if mse_w_a>7.e-3:
                print "FAIL: matching w_a failed"
                fails+=1
            else:
                print "PASS: matched w_a passed"
            if mse_w_b>7.e-3:
                print "FAIL: matching w_b failed"
                fails+=1
            else:
                print "PASS: matched w_b passed"


            #w1s,w2s = wm.match_w2(C_match,zs)
            pow_mults_a = wm.match_growth(C_match_a,zs,ws_a)
            pow_mults_b = wm.match_growth(C_match_b,zs,ws_b)
            #print wm.match_w(C_start,np.array([1.0]))

            sigma_a_in = np.loadtxt('test_inputs/wmatch/sigma_2.dat')
            sigma_b_in = np.loadtxt('test_inputs/wmatch/sigma_1.dat')

            sigma_a_interp = InterpolatedUnivariateSpline(sigma_a_in[:,0],sigma_a_in[:,1],k=3)(zs)
            sigma_b_interp = InterpolatedUnivariateSpline(sigma_b_in[:,0],sigma_b_in[:,1],k=3)(zs)

            sigma_a = np.sqrt(pow_mults_a)*0.83
            sigma_b = np.sqrt(pow_mults_b)*0.83

            mse_sigma_a = np.linalg.norm((sigma_a-sigma_a_interp)/sigma_a_interp)/sigma_a_interp.size
            mse_sigma_b = np.linalg.norm((sigma_b-sigma_b_interp)/sigma_b_interp)/sigma_b_interp.size
            print "mse sigma a/point: "+str(mse_sigma_a)
            print "mse sigma b/point: "+str(mse_sigma_b)
            if mse_sigma_a>1.e-4:
                print "FAIL: matching sigma8_a failed"
                fails+=1
            else:
                print "PASS: matched sigma8_a passed"
            if mse_sigma_b>1.e-4:
                print "FAIL: matching sigma8_b failed"
                fails+=1
            else:
                print "PASS: matched sigma8_b passed"
            if do_plots:
                #plt.plot(a_s,ws)
                plt.plot(zs,(cosmo_match_a['w0']+(1-a_s)*cosmo_match_a['wa']))
                plt.plot(zs,ws_a)
                plt.plot(zs,ws_a_interp)
                plt.ylim([-1.55,-0.4])
                plt.xlim([1.5,0.])
                plt.show()

                plt.plot(zs,(cosmo_match_b['w0']+(1-a_s)*cosmo_match_b['wa']))
                plt.plot(zs,ws_b)
                plt.plot(zs,ws_b_interp)
                plt.ylim([-1.55,-0.4])
                plt.xlim([1.5,0.])
                plt.show()

                plt.plot(zs,np.sqrt(pow_mults_a)*0.83)
                plt.plot(zs,np.sqrt(pow_mults_b)*0.83)
                plt.plot(zs,sigma_a_interp)
                plt.plot(zs,sigma_b_interp)
                plt.xlim([1.5,0.])
                plt.ylim([0.8,0.9])
                plt.show()
                #plt.plot(wm.w_Es)
                #plt.show()
        if fails==0:
            print "PASS: all checks passed"
        else:
            print "FAIL: "+str(fails)+" checks failed"
