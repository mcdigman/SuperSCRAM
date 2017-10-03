from warnings import warn
from scipy.interpolate import InterpolatedUnivariateSpline,RectBivariateSpline
from scipy.integrate import cumtrapz

import defaults

import cosmopie as cp
import numpy as np
class WMatcher(object):
    def __init__(self,C_fid,wmatcher_params=defaults.wmatcher_params):
        self.C_fid = C_fid
        self.cosmo_fid = C_fid.cosmology.copy()
        self.cosmo_fid['w']=-1.
        self.cosmo_fid['de_model']='constant_w'

        self.w_step = wmatcher_params['w_step']
        self.w_min = wmatcher_params['w_min']
        self.w_max = wmatcher_params['w_max']
        self.ws = np.arange(self.w_min,self.w_max,self.w_step)
        self.n_w = self.ws.size

        self.a_step = wmatcher_params['a_step']
        self.a_min = wmatcher_params['a_min']
        self.a_max = wmatcher_params['a_max']
        #self.a_s = np.arange(self.a_min,self.a_max,self.a_step)
        self.a_s = np.arange(self.a_max,self.a_min-self.a_step/10.,-self.a_step)
        self.n_a = self.a_s.size

        self.zs = 1./self.a_s-1.#np.arange(self.a_min,self.a_max,self.a_step)
        #self.Cs = np.zeros(self.n_w,dtype=object)
        self.cosmos = np.zeros(self.n_w,dtype=object)

        self.integ_Es = np.zeros((self.n_w,self.n_a))
        self.Gs = np.zeros((self.n_w,self.n_a))


        for i in xrange(0,self.n_w):
            self.cosmos[i] = self.cosmo_fid.copy()
            self.cosmos[i]['w'] = self.ws[i]
            C_i = cp.CosmoPie(cosmology=self.cosmos[i],silent=True,needs_power=False)
            E_as = C_i.Ez(self.zs)
            #TODO check initial 0 on this integral is right
            self.integ_Es[i] = cumtrapz(1./(self.a_s**2*E_as)[::-1],self.a_s[::-1],initial=0.)
            self.Gs[i] = C_i.G(self.zs)
        self.G_interp = RectBivariateSpline(self.ws,self.a_s[::-1],self.Gs[:,::-1],kx=2,ky=2)

        self.ind_switches = np.argmax(np.diff(self.integ_Es,axis=0)<0,axis=0)+1
        #there is a purely numerical issue that causes the integral to be non-monotonic, this loop eliminates the spurious behavior
        for i in xrange(1,self.n_a):
            if self.ind_switches[i]>1:
                if self.integ_Es[self.ind_switches[i]-1,i]-self.integ_Es[0,i]>=0:
                    self.integ_Es[0:(self.ind_switches[i]-1),i]=self.integ_Es[self.ind_switches[i]-1,i]
                else:
                    raise RuntimeError( "Nonmonotonic integral, solution is not unique at "+str(self.a_s[i]))

        self.integ_E_interp = RectBivariateSpline(self.ws,self.a_s[::-1],self.integ_Es,kx=2,ky=2)

    #accurate to within numerical precision
    #match effective constant w as in casarini paper
    def match_w(self,C_in,z_match):
        z_match=np.asanyarray(z_match)
        a_match = 1./(1.+z_match)
        E_in = C_in.Ez(self.zs)
        integ_E_in = cumtrapz(1./(self.a_s**2*E_in)[::-1],self.a_s[::-1],initial=0.)
        integ_E_in_interp = InterpolatedUnivariateSpline(self.a_s[::-1],integ_E_in,k=2,ext=2)
        integ_E_targets = integ_E_in_interp(a_match)
        w_grid1 = np.zeros(a_match.size)
        for itr in xrange(0,z_match.size):
            iE_vals = self.integ_E_interp(self.ws,a_match[itr]).T[0]-integ_E_targets[itr]
            iG = np.argmax(iE_vals<=0.)
            #require some padding so can get very accurate interpolation results
            if iG-2>=0 and iG+2<self.ws.size:
                w_grid1[itr] = InterpolatedUnivariateSpline(iE_vals[iG-2:iG+2][::-1],self.ws[iG-2:iG+2:][::-1],k=2)(0.)
            else:
                warn("w is too close to edge of range, using nearest neighbor w, consider expanding w range")
                w_grid1[itr] = self.ws[iG]
        return w_grid1


    #matches the redshift dependence of the growth factor for the input model to the equivalent model with constant w
    def match_growth(self,C_in,z_in,w_in):
        a_in = 1./(1.+z_in)
        G_norm_ins = C_in.G_norm(z_in)
        n_z_in = z_in.size
        pow_mult = np.zeros(n_z_in)
        #G_norm_fid = self.C_fid.G_norm(z_in)
        #TODO vectorize correctly
        for itr in xrange(0,n_z_in):
            pow_mult[itr]=(G_norm_ins[itr]/(self.G_interp(w_in[itr],a_in[itr])/self.G_interp(w_in[itr],1.)))**2
        #return multiplier for linear power spectrum from effective constant w model
        return pow_mult

    #get an interpolated growth factor for a given w, a_in is a vector
    def growth_interp(self,w_in,a_in):
        #TODO why grid=False?
        return self.G_interp(w_in,a_in,grid=False).T

    #match scaling (ie sigma8) for the input model compared to the fiducial model, not used
    def match_scale(self,z_in,w_in):
        n_z_in = z_in.size
        pow_scale = np.zeros(n_z_in)
        G_fid = self.C_fid.G(0)
        #TODO vectorize correctly
        for itr in xrange(0,n_z_in):
            pow_scale[itr]=(self.G_interp(w_in[itr],1.)/G_fid)**2
        #return multiplier for linear power spectrum from effective constant w model
        return pow_scale


#if __name__=='__main__':
#    cosmo_start = defaults.cosmology.copy()
#    cosmo_start['w'] = -1
#    cosmo_start['de_model']='constant_w'
#    params = {'w_step':0.01,'w_min':-3.50,'w_max':0.1,'a_step':0.001,'a_min':0.000916674,'a_max':1.00}
#
#    do_w_int_test=False
#    if do_w_int_test:
#        w_use_int = -1.2
#        cosmo_match_a = cosmo_start.copy()
#        cosmo_match_a['de_model']='jdem'
#        cosmo_match_a['w0'] = w_use_int
#        cosmo_match_a['wa'] = 0.
#        cosmo_match_a['w'] = w_use_int
#        for i in xrange(0,36):
#            cosmo_match_a['ws36_'+str(i)] = w_use_int
#
#        cosmo_match_b = cosmo_match_a.copy()
#        cosmo_match_b['de_model'] = 'w0wa'
#
#        wmatcher_params = defaults.wmatcher_params.copy()
#        C_start = cp.CosmoPie(cosmology=cosmo_start)
#        C_match_a = cp.CosmoPie(cosmology=cosmo_match_a)
#        C_match_b = cp.CosmoPie(cosmology=cosmo_match_b)
#
#        wm = WMatcher(C_start,wmatcher_params=params)
#        a_grid = np.arange(1.00,0.001,-0.01)
#        zs = 1./a_grid-1.
#
#      #  zs = np.arange(0.,20.,0.5)
#
#        w_a = wm.match_w(C_match_a,zs)
#        w_b = wm.match_w(C_match_b,zs)
#
#        #w2s = wm.match_w2(C_match_a,zs)
#        #w3s = wm.match_w3(C_match_a,zs)
#        #print "rms discrepancy methods 1 and 3="+str(np.linalg.norm(w3s-w1s)/w1s.size)
#        #expect perfect match to numerical precision, ~4.767*10^-13
#        print "rms discrepancy jdem and input="+str(np.linalg.norm(w_a-C_match_a.w_interp(a_grid))/w_a.size)
#        print "rms discrepancy w0wa and input="+str(np.linalg.norm(w_b-C_match_b.w_interp(a_grid))/w_b.size)
#        #import matplotlib.pyplot as plt
#        #a_s = 1./(1.+zs)
#        #plt.plot(a_s,w1s)
#        #plt.plot(a_s,w3s)
#        #plt.show()
#
#    do_jdem_w0wa_match_test = False
#    if do_jdem_w0wa_match_test:
#        cosmo_match_w0wa = cosmo_start.copy()
#        cosmo_match_jdem = cosmo_start.copy()
#        cosmo_match_w0wa['de_model'] = 'w0wa'
#        cosmo_match_w0wa['w0'] = -1.0
#        cosmo_match_w0wa['wa'] = 0.5
#        cosmo_match_w0wa['w'] = -1.0+0.9*0.5
#
#        cosmo_match_jdem['de_model'] = 'jdem'
#        cosmo_match_jdem['w'] = -1.0+0.9*0.5
#        a_jdem = 1.-0.025*np.arange(0,36)
#        for i in xrange(0,36):
#            cosmo_match_jdem['ws36_'+str(i)] = -1.+(1.-(a_jdem[i]-0.025/2.))*0.5
#
#        C_match_w0wa = cp.CosmoPie(cosmology=cosmo_match_w0wa)
#        C_match_jdem = cp.CosmoPie(cosmology=cosmo_match_jdem)
#
#        a_grid = np.arange(1.00,0.001,-0.01)
#        zs = 1./a_grid-1.
#
#        w_w0wa = wm.match_w(C_match_w0wa,zs)
#        w_jdem = wm.match_w(C_match_jdem,zs)
#
#        print "rms discrepancy w0wa and jdem="+str(np.linalg.norm(w_w0wa-w_jdem)/w_jdem.size)
#        #current agreement ~0.0627 is reasonable given imperfect approximation of w0wa
#        #depends on setting of default value for jdem at end
#
#    do_convergence_test_w0wa = False
#    if do_convergence_test_w0wa:
#        cosmo_match_w0wa = cosmo_start.copy()
#        cosmo_match_w0wa['de_model'] = 'w0wa'
#        cosmo_match_w0wa['w0'] = -1.0
#        cosmo_match_w0wa['wa'] = 0.5
#        cosmo_match_w0wa['w'] = -1.0+0.1*0.5
#
#        params_1 = params.copy()
#        params_1['w_step'] = 0.01
#        params_2 = params.copy()
#        params_2['w_step'] = 0.001
#        params_3 = params.copy()
#        params_3['a_step'] = params['a_step']/10.
#
#        C_match_w0wa = cp.CosmoPie(cosmology=cosmo_match_w0wa)
#
#        a_grid = np.arange(1.00,0.001,-0.01)
#        zs = 1./a_grid-1.
#
#        wm_1 = WMatcher(C_start,wmatcher_params=params_1)
#        wm_2 = WMatcher(C_start,wmatcher_params=params_2)
#        wm_3 = WMatcher(C_start,wmatcher_params=params_3)
#
#        w_w0wa_1 = wm_1.match_w(C_match_w0wa,zs)
#        w_w0wa_2 = wm_2.match_w(C_match_w0wa,zs)
#        w_w0wa_3 = wm_3.match_w(C_match_w0wa,zs)
#        print "rms discrepancy w0wa_1 and w0wa_2="+str(np.linalg.norm(w_w0wa_1-w_w0wa_2)/w_w0wa_1.size)
#        print "rms % discrepancy w0wa_1 and w0wa_2="+str(np.linalg.norm((w_w0wa_1-w_w0wa_2)/w_w0wa_2)/w_w0wa_1.size)
#        print "rms discrepancy w0wa_1 and w0wa_3="+str(np.linalg.norm(w_w0wa_1-w_w0wa_3)/w_w0wa_1.size)
#        print "rms % discrepancy w0wa_1 and w0wa_3="+str(np.linalg.norm((w_w0wa_1-w_w0wa_3)/w_w0wa_3)/w_w0wa_1.size)
#
#    do_convergence_test_jdem = False
#    if do_convergence_test_jdem:
#        cosmo_match_jdem = cosmo_start.copy()
#        cosmo_match_jdem['de_model'] = 'jdem'
#        cosmo_match_jdem['w0'] = -1.0
#        cosmo_match_jdem['wa'] = 0.5
#        cosmo_match_jdem['w'] = -1.0+0.1*0.5
#
#
#        for i in xrange(0,36):
#            cosmo_match_jdem['ws36_'+str(i)] = -1.
#        cosmo_match_jdem['ws36_'+str(1)] = -1.5
#
#        params_1 = params.copy()
#        params_1['w_step'] = 0.01
#        params_2 = params.copy()
#        params_2['w_step'] = 0.005
#        params_3 = params.copy()
#        params_3['a_step'] = params_1['a_step']/10.
#
#
#        C_match_jdem = cp.CosmoPie(cosmology=cosmo_match_jdem)
#
#        a_grid = np.arange(1.00,0.001,-0.01)
#        zs = 1./a_grid-1.
#
#        wm_1 = WMatcher(C_start,wmatcher_params=params_1)
#        wm_2 = WMatcher(C_start,wmatcher_params=params_2)
#        wm_3 = WMatcher(C_start,wmatcher_params=params_3)
#
#        w_jdem_1 = wm_1.match_w(C_match_jdem,zs)
#        w_jdem_2 = wm_2.match_w(C_match_jdem,zs)
#        w_jdem_3 = wm_3.match_w(C_match_jdem,zs)
#        print "rms discrepancy jdem_1 and jdem_2="+str(np.linalg.norm(w_jdem_1-w_jdem_2)/w_jdem_1.size)
#        print "mean absolute % discrepancy jdem_1 jdem_2="+str(np.average(np.abs((w_jdem_1-w_jdem_2)/w_jdem_1)/w_jdem_1.size*100.))
#        print "rms discrepancy jdem_1 and jdem_3="+str(np.linalg.norm(w_jdem_1-w_jdem_3)/w_jdem_1.size)
#        print "mean absolute % discrepancy jdem_1 jdem_3="+str(np.average(np.abs((w_jdem_1-w_jdem_3)/w_jdem_1)/w_jdem_1.size*100.))
#
#    do_match_casarini=True
#    if do_match_casarini:
#        cosmo_match_a = cosmo_start.copy()
#        cosmo_match_a['de_model'] = 'w0wa'
#        cosmo_match_a['w0'] = -1.2
#        cosmo_match_a['wa'] = 0.5
#        cosmo_match_a['w'] = -1.2
#
#        cosmo_match_b = cosmo_match_a.copy()
#        cosmo_match_b['w0'] = -0.6
#        cosmo_match_b['wa'] = -1.5
#        cosmo_match_b['w'] = -0.6
#
#
#        wmatcher_params = defaults.wmatcher_params.copy()
#        C_start = cp.CosmoPie(cosmology=cosmo_start)
#        C_match_a = cp.CosmoPie(cosmology=cosmo_match_a)
#        C_match_b = cp.CosmoPie(cosmology=cosmo_match_b)
#
#        wm = WMatcher(C_start)
#        zs = np.arange(0.,1.51,0.05)
#        a_s = 1./(1.+zs)
#
#        ws_a = wm.match_w(C_match_a,zs)
#        ws_b = wm.match_w(C_match_b,zs)
#
#        weff_2 = np.loadtxt('test_inputs/wmatch/weff_2.dat')
#        ws_a_pred = weff_2[:,1]
#        zs_a_pred = weff_2[:,0]
#
#        ws_a_interp = InterpolatedUnivariateSpline(zs_a_pred,ws_a_pred,k=3)(zs)
#        print "mse a/point: "+str(np.linalg.norm((ws_a-ws_a_interp)/ws_a_interp)/ws_a_interp.size)
#
#        weff_1 = np.loadtxt('test_inputs/wmatch/weff_1.dat')
#        ws_b_pred = weff_1[:,1]
#        zs_b_pred = weff_1[:,0]
#
#        ws_b_interp = InterpolatedUnivariateSpline(zs_b_pred,ws_b_pred,k=3)(zs)
#        print "mse b/point: "+str(np.linalg.norm((ws_b-ws_b_interp)/ws_b_interp)/ws_b_interp.size)
#
#
#        #w1s,w2s = wm.match_w2(C_match,zs)
#        pow_mults_a = wm.match_growth(C_match_a,zs,ws_a)
#        pow_mults_b = wm.match_growth(C_match_b,zs,ws_b)
#        #print wm.match_w(C_start,np.array([1.0]))
#
#        sigma_a_in = np.loadtxt('test_inputs/wmatch/sigma_2.dat')
#        sigma_b_in = np.loadtxt('test_inputs/wmatch/sigma_1.dat')
#
#        sigma_a_interp = InterpolatedUnivariateSpline(sigma_a_in[:,0],sigma_a_in[:,1],k=3)(zs)
#        sigma_b_interp = InterpolatedUnivariateSpline(sigma_b_in[:,0],sigma_b_in[:,1],k=3)(zs)
#
#        sigma_a = np.sqrt(pow_mults_a)*0.83
#        sigma_b = np.sqrt(pow_mults_b)*0.83
#
#
#        print "mse sigma a/point: "+str(np.linalg.norm((sigma_a-sigma_a_interp)/sigma_a_interp)/sigma_a_interp.size)
#        print "mse sigma b/point: "+str(np.linalg.norm((sigma_b-sigma_b_interp)/sigma_b_interp)/sigma_b_interp.size)
#        import matplotlib.pyplot as plt
#        #plt.plot(a_s,ws)
#        #should match arXiv:1601.07230v3 figure 2
#        plt.plot(zs,(cosmo_match_a['w0']+(1-a_s)*cosmo_match_a['wa']))
#        plt.plot(zs,ws_a)
#        plt.plot(zs,ws_a_interp)
#        plt.ylim([-1.55,-0.4])
#        plt.xlim([1.5,0.])
#        plt.show()
#
#        plt.plot(zs,(cosmo_match_b['w0']+(1-a_s)*cosmo_match_b['wa']))
#        plt.plot(zs,ws_b)
#        plt.plot(zs,ws_b_interp)
#        plt.ylim([-1.55,-0.4])
#        plt.xlim([1.5,0.])
#        plt.show()
#
#        plt.plot(zs,np.sqrt(pow_mults_a)*0.83)
#        plt.plot(zs,np.sqrt(pow_mults_b)*0.83)
#        plt.plot(zs,sigma_a_interp)
#        plt.plot(zs,sigma_b_interp)
#        plt.xlim([1.5,0.])
#        plt.ylim([0.8,0.9])
#        plt.show()
#        #plt.plot(wm.w_Es)
#        #plt.show()
