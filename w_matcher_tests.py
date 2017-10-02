from scipy.interpolate import InterpolatedUnivariateSpline,RectBivariateSpline

import defaults

import cosmopie as cp
import numpy as np
from w_matcher import WMatcher

if __name__=='__main__':
    cosmo_start = defaults.cosmology.copy()
    cosmo_start['w'] = -1
    cosmo_start['de_model']='constant_w'
    params = {'w_step':0.01,'w_min':-3.50,'w_max':0.1,'a_step':0.001,'a_min':0.000916674,'a_max':1.00}

    do_w_int_test=False
    do_jdem_w0wa_match_test = False
    do_convergence_test_w0wa = False
    do_convergence_test_jdem = False
    do_match_casarini=True

    fails = 0

    do_plots=False
    if do_plots:
        import matplotlib.pyplot as plt

    if do_w_int_test:
        w_use_int = -1.2
        cosmo_match_a = cosmo_start.copy()
        cosmo_match_a['de_model']='jdem'
        cosmo_match_a['w0'] = w_use_int
        cosmo_match_a['wa'] = 0.
        cosmo_match_a['w'] = w_use_int
        for i in xrange(0,36):
            cosmo_match_a['ws36_'+str(i)] = w_use_int

        cosmo_match_b = cosmo_match_a.copy()
        cosmo_match_b['de_model'] = 'w0wa'

        wmatcher_params = defaults.wmatcher_params.copy()
        C_start = cp.CosmoPie(cosmology=cosmo_start)
        C_match_a = cp.CosmoPie(cosmology=cosmo_match_a)
        C_match_b = cp.CosmoPie(cosmology=cosmo_match_b)

        wm = WMatcher(C_start,wmatcher_params=params)
        a_grid = np.arange(1.00,0.001,-0.01)
        zs = 1./a_grid-1.

      #  zs = np.arange(0.,20.,0.5)

        w_a = wm.match_w(C_match_a,zs)
        w_b = wm.match_w(C_match_b,zs)

        #w2s = wm.match_w2(C_match_a,zs)
        #w3s = wm.match_w3(C_match_a,zs)
        #print "rms discrepancy methods 1 and 3="+str(np.linalg.norm(w3s-w1s)/w1s.size)
        #expect perfect match to numerical precision, ~4.767*10^-13
        print "rms discrepancy jdem and input="+str(np.linalg.norm(w_a-C_match_a.w_interp(a_grid))/w_a.size)
        print "rms discrepancy w0wa and input="+str(np.linalg.norm(w_b-C_match_b.w_interp(a_grid))/w_b.size)
        #import matplotlib.pyplot as plt
        #a_s = 1./(1.+zs)
        #plt.plot(a_s,w1s)
        #plt.plot(a_s,w3s)
        #plt.show()
    
    if do_jdem_w0wa_match_test:
        cosmo_match_w0wa = cosmo_start.copy()
        cosmo_match_jdem = cosmo_start.copy()
        cosmo_match_w0wa['de_model'] = 'w0wa'
        cosmo_match_w0wa['w0'] = -1.0
        cosmo_match_w0wa['wa'] = 0.5
        cosmo_match_w0wa['w'] = -1.0+0.9*0.5

        cosmo_match_jdem['de_model'] = 'jdem'
        cosmo_match_jdem['w'] = -1.0+0.9*0.5
        a_jdem = 1.-0.025*np.arange(0,36)
        for i in xrange(0,36):
            cosmo_match_jdem['ws36_'+str(i)] = -1.+(1.-(a_jdem[i]-0.025/2.))*0.5

        C_match_w0wa = cp.CosmoPie(cosmology=cosmo_match_w0wa)
        C_match_jdem = cp.CosmoPie(cosmology=cosmo_match_jdem)

        a_grid = np.arange(1.00,0.001,-0.01)
        zs = 1./a_grid-1.

        w_w0wa = wm.match_w(C_match_w0wa,zs)
        w_jdem = wm.match_w(C_match_jdem,zs)

        print "rms discrepancy w0wa and jdem="+str(np.linalg.norm(w_w0wa-w_jdem)/w_jdem.size)
        #current agreement ~0.0627 is reasonable given imperfect approximation of w0wa
        #depends on setting of default value for jdem at end

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

        C_match_w0wa = cp.CosmoPie(cosmology=cosmo_match_w0wa)

        a_grid = np.arange(1.00,0.001,-0.01)
        zs = 1./a_grid-1.

        wm_1 = WMatcher(C_start,wmatcher_params=params_1)
        wm_2 = WMatcher(C_start,wmatcher_params=params_2)
        wm_3 = WMatcher(C_start,wmatcher_params=params_3)

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
            cosmo_match_jdem['ws36_'+str(i)] = -1.
        cosmo_match_jdem['ws36_'+str(1)] = -1.5

        params_1 = params.copy() 
        params_1['w_step'] = 0.01
        params_2 = params.copy() 
        params_2['w_step'] = 0.005
        params_3 = params.copy() 
        params_3['a_step'] = params_1['a_step']/10.


        C_match_jdem = cp.CosmoPie(cosmology=cosmo_match_jdem)

        a_grid = np.arange(1.00,0.001,-0.01)
        zs = 1./a_grid-1.

        wm_1 = WMatcher(C_start,wmatcher_params=params_1)
        wm_2 = WMatcher(C_start,wmatcher_params=params_2)
        wm_3 = WMatcher(C_start,wmatcher_params=params_3)

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
        C_start = cp.CosmoPie(cosmology=cosmo_start)
        C_match_a = cp.CosmoPie(cosmology=cosmo_match_a)
        C_match_b = cp.CosmoPie(cosmology=cosmo_match_b)

        wm = WMatcher(C_start)
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
        
        mse_sigma_a=np.linalg.norm((sigma_a-sigma_a_interp)/sigma_a_interp)/sigma_a_interp.size
        mse_sigma_b=np.linalg.norm((sigma_b-sigma_b_interp)/sigma_b_interp)/sigma_b_interp.size
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
