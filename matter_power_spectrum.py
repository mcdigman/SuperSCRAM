"""
MatterPower class handles and wrap all matter power spectra related functionality
"""
from warnings import warn
from camb_power import camb_pow
from scipy.interpolate import RectBivariateSpline,InterpolatedUnivariateSpline

import numpy as np
import FASTPTcode.FASTPT as FASTPT
import cosmopie as cp

import defaults
import halofit
import w_matcher
#class for a matter power spectrum which can get both linear and nonlinear power spectra as needed
#TODO clean up
#TODO treat w0 and w consistently
class MatterPower(object):
    def __init__(self,C_in,P_lin=None,k_in=None,matter_power_params=defaults.matter_power_params,camb_params=None,wmatcher_params=defaults.wmatcher_params,halofit_params=defaults.halofit_params,fpt_params=defaults.fpt_params,wm_in=None,wm_safe=False,P_fid=None,camb_safe=False,de_perturbative=False):
        """Generate matter power spectrum for input cosmology
        linear power spectrum use camb, nonlinear can use halofit or FAST-PT
        Inputs:
        C_in: input CosmoPie
        P_lin: Input matter power spectrum. Optional.
        matter_power_params,camb_params,wmatcher_params,halofit_params,fpt_params: dictionaries of parameters
        wm_in: input WMatcher object. Optional.
        wm_safe: if True and wm_in is not None, use wm_in
        P_fid: Fiducial  power spectrum. Optional.
        camb_safe: If True and P_fid is not None, will borrow camb_grid from P_fid if possible. Useful if only linear growth factor different from P_fid.
        de_perturbative: If True, get power spectra for constant w even if w(z) in C_in is not constant
        """

        #save all the input parameter sets
        self.params = matter_power_params
        self.camb_params = camb_params.copy()
        self.camb_params['return_sigma8']=True
        self.wmatcher_params = wmatcher_params
        self.halofit_params = halofit_params
        self.fpt_params = fpt_params
        self.wm_safe=wm_safe
        self.de_perturbative=de_perturbative

        self.C = C_in
        self.cosmology = self.C.cosmology

        if P_lin is None or k_in is None:
            k_camb,self.P_lin,self.sigma8_in = camb_pow(self.cosmology,camb_params=self.camb_params)
            #TODO check handling fixing sigma8 right
            if k_in is None:
                self.k = k_camb
            else:
                if not np.allclose(k_camb,k_in):
                    #print "MatterPower: adjusting k grid"
                    #TODO extrapolate properly, not safe on edges now. Issue is k_camb is off by factor of self.cosmology['H0']/71.902712048990196
                    self.P_lin = InterpolatedUnivariateSpline(k_camb,self.P_lin,k=2)(k_in)
                self.k=k_in
        else:
            self.k = k_in
            self.P_lin = P_lin
            self.sigma8_in = C_in.get_sigma8()

        self.a_grid = np.arange(self.params['a_max'],self.params['a_min']-self.params['a_step']/10.,-self.params['a_step'])
        self.z_grid = 1./self.a_grid-1.
        self.n_a = self.a_grid.size

        self.de_model = self.cosmology['de_model']
        self.get_w_matcher = self.params['needs_wmatcher'] and not self.de_model=='constant_w' and not self.de_perturbative
        if self.get_w_matcher:
            if wm_in is not None and self.wm_safe:
                self.wm = wm_in
            else:
                self.wm = w_matcher.WMatcher(self.C,wmatcher_params=self.wmatcher_params)
            self.w_match_grid = self.wm.match_w(self.C,self.z_grid)
            #TODO is this grid acceptably fine?
            self.w_match_interp = InterpolatedUnivariateSpline(self.z_grid,self.w_match_grid,k=3,ext=2)
            self.pow_mult_grid = self.wm.match_growth(self.C,self.z_grid,self.w_match_grid)
            self.use_match_grid = True
        else:
            self.wm = None
            self.w_match_grid = np.zeros(self.n_a)+self.cosmology['w']
            self.w_match_interp = InterpolatedUnivariateSpline(self.z_grid,self.w_match_grid,k=3,ext=2)
            self.pow_mult_grid = np.zeros(self.n_a)+1.
            self.use_match_grid = False
        #figure out an error checking method here
        #Interpolated to capture possible k dependence in camb of w,#TODO see if sufficient
        #needed because field can cluster on very large scales, see arXiv:astro-ph/9906174
        if self.params['needs_camb_w_grid'] and self.use_match_grid:

            #get grid of camb power spectra for various w values
            #TODO found bug with force_sigma8 consistency
            #borrow some parameters from an input power spectrum if camb need not be called repeatedly
            #TODO check robustness ie do not need to also get alternate w_match_grid
            #TODO not sure all these camb calls are a good idea
            cache_usable = False
            if camb_safe and not P_fid is None:
                self.camb_w_interp = P_fid.camb_w_interp
                self.camb_sigma8_interp = P_fid.camb_sigma8_interp
                self.camb_w_grid = P_fid.camb_w_grid
                self.camb_w_pows = P_fid.camb_w_pows
                self.camb_sigma8s = P_fid.camb_sigma8s
                self.use_camb_grid = P_fid.use_camb_grid
                self.w_min = P_fid.w_min
                self.w_max = P_fid.w_max
                if self.use_match_grid and np.any(self.w_match_grid>self.w_max) or np.any(self.w_match_grid<self.w_min):
                    #TODO extending would be better than regenerating
                    #TODO bug is causing this to expand itself somehow
                    warn('Insufficient range in given camb w grid, needed min '+str(np.min(self.w_match_grid))+' max '+str(np.max(self.w_match_grid)))
                else:
                    cache_usable=True
                #self.camb_w_interp = RectBivariateSpline(self.k,self.camb_w_grid,self.camb_w_pows,kx=1,ky=1)
                #self.camb_sigma8_interp = InterpolatedUnivariateSpline(self.camb_w_grid,self.camb_sigma8s,k=2)
            if not cache_usable:
                #get a w grid that is no larger than it needs to be, keeping integer number of steps from w=-1 for convenience
                #TODO could refine edge criteria
                n_w_below = np.ceil((-1.-np.min(self.w_match_grid-self.params['w_edge']-self.params['w_step']))/self.params['w_step'])
                n_w_above = np.ceil((1.+np.max(self.w_match_grid+self.params['w_edge']+self.params['w_step']))/self.params['w_step'])
                self.w_min = -1.-n_w_below*self.params['w_step']
                self.w_max = -1.+n_w_above*self.params['w_step']
                if np.any(np.abs(np.diff(self.w_match_grid))<self.params['w_step']):
                    warn('given step size may '+str(self.params['w_step'])+' may be insufficient to resolve grid with min spacing '+str(np.min(np.abs(np.diff(self.w_match_grid)))))
                self.camb_w_grid = np.arange(self.w_min,self.w_max,self.params['w_step'])
                #make sure the grid has at least some minimum number of values
                if self.camb_w_grid.size<self.params['min_n_w']:
                    self.camb_w_grid = np.linspace(self.w_min,self.w_max,self.params['min_n_w'])
                print "matter_power_spectrum: camb grid",self.camb_w_grid
                print "matter_power_spectrum: w grid diffs",np.diff(self.w_match_grid)

                n_cw = self.camb_w_grid.size
                self.camb_w_pows = np.zeros((self.k.size,n_cw))
                self.camb_sigma8s = np.zeros(n_cw)
                camb_cosmos = self.cosmology.copy()
                camb_cosmos['de_model'] = 'constant_w'
                for i in xrange(0,n_cw):
                    camb_cosmos['w'] = self.camb_w_grid[i]
                    #TODO check the G input is correct
                    print "camb with w=",self.camb_w_grid[i]
                    k_i,self.camb_w_pows[:,i],self.camb_sigma8s[i] = camb_pow(camb_cosmos,camb_params=self.camb_params)
                    #This interpolation shift shouldn't really be needed because self.k is generated with the same value of H0
                    if not np.all(k_i==self.k):
                        self.camb_w_pows[:,i] = InterpolatedUnivariateSpline(k_i,self.camb_w_pows[:,i],k=2,ext=2)(self.k)
                    self.camb_w_interp = RectBivariateSpline(self.k,self.camb_w_grid,self.camb_w_pows,kx=3,ky=3)
                    self.camb_sigma8_interp = InterpolatedUnivariateSpline(self.camb_w_grid,self.camb_sigma8s,k=2,ext=2)
                self.use_camb_grid = True
        else:
            self.w_min = None
            self.w_max = None
            self.camb_w_grid = np.array([])
            self.camb_cosmos= np.array([])
            self.camb_w_pows = np.array([])
            self.camb_sigma8s = np.array([])
            self.camb_w_interp = None
            self.camb_sigma8_interp=None
            self.use_camb_grid = False

#        if self.params['needs_halofit']:
#            self.hf=halofit.HalofitPk(self.C,p_lin=self.P_lin,halofit_params=self.halofit_params)
#        else:
#            self.hf = None

        if self.params['needs_fpt']:
            self.fpt = FASTPT.FASTPT(self.k,self.fpt_params['nu'],low_extrap=self.fpt_params['low_extrap'],high_extrap=self.fpt_params['high_extrap'],n_pad=self.fpt_params['n_pad'])
            #self.one_loop = self.fpt.one_loop(self.P_lin,C_window=self.fpt_params['C_window'])
        else:
            self.fpt = None
            #self.one_loop=np.array([])

    def get_sigma8_eff(self,zs):
        """get effective sigma8(z) for matching power spectrum amplitude in w(z) models from WMatcher"""
        if self.use_match_grid:
            #w_match_grid = self.wm.match_w(self.C,zs)
            w_match_grid = self.w_match_interp(zs)
            pow_mult_grid = self.wm.match_growth(self.C,zs,w_match_grid)
            #pow_mult_grid = self.pow_mult_interp(zs)
            return self.camb_sigma8_interp(w_match_grid)*np.sqrt(pow_mult_grid)
        else:
            return self.sigma8_in+np.zeros(zs.size)

    #const_pow_mult allows adjusting sigma8 without creating a whole new power spectrum
    def get_matter_power(self,zs_in,pmodel=defaults.matter_power_params['nonlinear_model'],const_pow_mult=1.,get_one_loop=False):
        """get a matter power spectrum P(z)
        Inputs:
        pmodel: nonlinear power spectrum model to use, options are 'linear','halofit', and 'fastpt'
        const_pow_mult: multiplier to adjust sigma8 without creating a whole new power spectrum
        get_one_loop: If True and pmodel=='fastpt', return the one loop contribution in addition to the nonlinear power spectrum
        """
        if  isinstance(zs_in,np.ndarray):
            zs = zs_in
        else:
            zs = np.array([zs_in])

        n_z = zs.size

        if self.use_match_grid:
            #w_match_grid = self.wm.match_w(self.C,zs)
            #TODO maybe allow override of interpolation
            w_match_grid = self.w_match_interp(zs)
            pow_mult_grid = self.wm.match_growth(self.C,zs,w_match_grid)*const_pow_mult
        #else:
        #    w_match_grid = np.zeros(zs.size)+self.cosmology.w
        #    pow_mult_grid = np.zeros(zs.size)+1.*const_pow_mult

        G_norms = self.C.G_norm(zs)
        if self.use_match_grid:
            Pbases = np.zeros((self.k.size,n_z))
            if self.use_camb_grid:
                for i in xrange(0,n_z):
                    Pbases[:,i]=pow_mult_grid[i]*self.camb_w_interp(self.k,w_match_grid[i]).flatten()
            else:
                Pbases = np.outer(self.P_lin,pow_mult_grid)
        else:
            Pbases = np.outer(self.P_lin,np.full(n_z,1.))*const_pow_mult

        P_nonlin = np.zeros((self.k.size,n_z))
        if pmodel=='linear':
            P_nonlin = Pbases*G_norms**2
        elif pmodel=='halofit':
            if self.use_match_grid:
                for i in xrange(0,n_z):
                    cosmo_hf_i = self.cosmology.copy()
                    cosmo_hf_i['de_model'] = 'constant_w'
                    cosmo_hf_i['w'] = w_match_grid[i]
                    hf_C_calc = cp.CosmoPie(cosmology=cosmo_hf_i,silent=True,G_safe=True,G_in=InterpolatedUnivariateSpline(self.C.z_grid,self.wm.growth_interp(w_match_grid[i],self.C.a_grid),ext=2,k=2),needs_power=False)
                    hf_C_calc[i]=self.k
                    hf_calc = halofit.HalofitPk(hf_C_calc,p_lin=Pbases[:,i],halofit_params=self.halofit_params)
                    P_nonlin[:,i] = 2.*np.pi**2*(hf_calc.D2_NL(self.k,zs[i]).T/self.k**3)
            else:
                hf_calc = halofit.HalofitPk(self.C,p_lin=self.P_lin*const_pow_mult,halofit_params=self.halofit_params)
                P_nonlin = 2.*np.pi**2*(hf_calc.D2_NL(self.k,zs).T/self.k**3).T

        elif pmodel=='fastpt':
            if self.use_match_grid:
                one_loops = np.zeros((self.k.size,n_z))
                for i in xrange(0,n_z):
                    G_i = G_norms[i]
                    one_loops[:,i] = self.fpt.one_loop(Pbases[:,i],C_window=self.fpt_params['C_window'])*G_i**4
                    P_nonlin[:,i] =  Pbases[:,i]*G_i**2+one_loops[:,i]
            else:
                one_loops = np.outer(self.fpt.one_loop(self.P_lin,C_window=self.fpt_params['C_window']),G_norms**4)
                P_nonlin = np.outer(self.P_lin,G_norms**2)+one_loops
        if pmodel=='fastpt' and get_one_loop:
            return P_nonlin,one_loops
        elif get_one_loop:
            raise ValueError('could not get one loop power spectrum for pmodel '+str(pmodel))
        else:
            return P_nonlin



#if __name__ == '__main__':
#    cosmo_in = defaults.cosmology.copy()
#    camb_params = defaults.camb_params.copy()
#    camb_params['kmax']=10.
#    camb_params['maxkh']=10.
#    camb_params['npoints'] = 1000
#    camb_params['force_sigma8'] = False
#    #C_in = cp.CosmoPie(cosmo_in,camb_params=camb_params)
#    #mps = MatterPower(C_in)
#
#    do_constant_consistency_test = True
#    if do_constant_consistency_test:
#        w_use = -1.1
#        pmodel_use = 'halofit'
#
#        mp_params_const = defaults.matter_power_params.copy()
#        mp_params_const['w_step'] = 0.01
#        mp_params_const['w_edge'] = 0.02
#
#        cosmo_const = cosmo_in.copy()
#        cosmo_const['de_model'] = 'constant_w'
#        cosmo_const['w'] = w_use
#        cosmo_const['w0'] = w_use
#        cosmo_const['wa'] = 0.
#        for i in xrange(0,36):
#            cosmo_const['ws36_'+str(i)] = w_use
#
#        cosmo_w0wa = cosmo_const.copy()
#        cosmo_w0wa['de_model'] = 'w0wa'
#
#        cosmo_jdem = cosmo_const.copy()
#        cosmo_jdem['de_model'] = 'jdem'
#
#        C_const = cp.CosmoPie(cosmo_const,camb_params=camb_params)
#        C_w0wa = cp.CosmoPie(cosmo_w0wa,camb_params=camb_params)
#        C_jdem = cp.CosmoPie(cosmo_jdem,camb_params=camb_params)
#
#        mps_const = MatterPower(C_const,camb_params=camb_params,matter_power_params = mp_params_const)
#        mps_w0wa = MatterPower(C_w0wa,camb_params=camb_params,matter_power_params = mp_params_const)
#        mps_jdem = MatterPower(C_jdem,camb_params=camb_params,matter_power_params = mp_params_const)
#
#        zs = np.arange(0.,1.2,0.3)
#
#        P_const = mps_const.get_matter_power(zs,pmodel=pmodel_use)
#        P_w0wa = mps_w0wa.get_matter_power(zs,pmodel=pmodel_use)
#        P_jdem = mps_jdem.get_matter_power(zs,pmodel=pmodel_use)
#        #agreement ~5*10^-6
#        print "rms error fraction const w0wa ="+str(np.linalg.norm((P_const-P_w0wa)/P_const))
#        print "rms error fraction const jdem ="+str(np.linalg.norm((P_const-P_jdem)/P_const))
#        print "rms error fraction jdem w0wa ="+str(np.linalg.norm((P_jdem-P_w0wa)/P_w0wa))
#
#
#    do_w0wa_jdem_consistency_test = False
#
#    if do_w0wa_jdem_consistency_test:
#        w0_use = -1.1
#        wa_use = 0.2
#        pmodel_use = 'halofit'
#
#        mp_params_const = defaults.matter_power_params.copy()
#        mp_params_const['w_step'] = 0.01
#        mp_params_const['w_edge'] = 0.04
#
#        cosmo_const = cosmo_in.copy()
#        cosmo_const['de_model'] = 'constant_w'
#        #will really only work in limit where w small, given different w for matched models TODO is this a problem?
#        cosmo_const['w'] = w0_use+wa_use*(1.-0.1)
#        cosmo_const['w0'] = w0_use
#        cosmo_const['wa'] = wa_use
#        a_jdem = 1.-0.025*np.arange(0,36)
#        for i in xrange(0,36):
#            cosmo_const['ws36_'+str(i)] = w0_use+(1.-(a_jdem[i]-0.025/2.))*wa_use
#
#        cosmo_w0wa = cosmo_const.copy()
#        cosmo_w0wa['de_model'] = 'w0wa'
#
#        cosmo_jdem = cosmo_const.copy()
#        cosmo_jdem['de_model'] = 'jdem'
#
#        C_w0wa = cp.CosmoPie(cosmo_w0wa,camb_params=camb_params)
#        C_jdem = cp.CosmoPie(cosmo_jdem,camb_params=camb_params)
#
#        mps_w0wa = MatterPower(C_w0wa,camb_params=camb_params,matter_power_params = mp_params_const)
#        mps_jdem = MatterPower(C_jdem,camb_params=camb_params,matter_power_params = mp_params_const)
#
#        zs = np.arange(0.,1.2,0.3)
#
#        P_w0wa = mps_w0wa.get_matter_power(zs,pmodel=pmodel_use)
#        P_jdem = mps_jdem.get_matter_power(zs,pmodel=pmodel_use)
#        #agreement ~5*10^-6
#        print "rms error w0wa model fraction jdem w0wa ="+str(np.linalg.norm((P_jdem-P_w0wa)/P_w0wa))
#
#    do_jdem_convergence_test = False
#    if do_jdem_convergence_test:
#        w_use = -1.
#        eps_use = 0.5
#        pmodel_use = 'halofit'
#
#        mp_params_const1 = defaults.matter_power_params.copy()
#        mp_params_const1['w_step'] = 0.001
#        mp_params_const1['w_edge'] = 2.*mp_params_const1['w_step']
#
#        mp_params_const2 = mp_params_const1.copy()
#        mp_params_const2['w_step'] = 0.0001
#        mp_params_const2['w_edge'] = 2.*mp_params_const2['w_step']
#
#        cosmo_const = cosmo_in.copy()
#        cosmo_const['de_model'] = 'constant_w'
#        cosmo_const['w'] = w_use
#        cosmo_const['w0'] = w_use
#        cosmo_const['wa'] = 0.
#        a_jdem = 1.-0.025*np.arange(0,36)
#        for i in xrange(0,36):
#            cosmo_const['ws36_'+str(i)] = w_use
#
#        cosmo_pert = cosmo_const.copy()
#        cosmo_pert['de_model'] = 'jdem'
#        cosmo_pert['ws36_35']+=eps_use
#
#        C_const = cp.CosmoPie(cosmo_const,camb_params=camb_params)
#        C_pert = cp.CosmoPie(cosmo_pert,camb_params=camb_params)
#
#        mps_const = MatterPower(C_const,camb_params=camb_params,matter_power_params = mp_params_const1)
#        mps_pert1 = MatterPower(C_pert,camb_params=camb_params,matter_power_params = mp_params_const1)
#        mps_pert2 = MatterPower(C_pert,camb_params=camb_params,matter_power_params = mp_params_const2)
#
#        zs = np.arange(0.,1.2,0.3)
#
#        P_const = mps_const.get_matter_power(zs,pmodel=pmodel_use)
#        P_pert1 = mps_pert1.get_matter_power(zs,pmodel=pmodel_use)
#        P_pert2 = mps_pert2.get_matter_power(zs,pmodel=pmodel_use)
#        response_pert1 = (P_pert1-P_const)/eps_use
#        response_pert2 = (P_pert2-P_const)/eps_use
#        print "rms error pert2 pert1 ="+str(np.linalg.norm((response_pert1-response_pert2)/response_pert2))
#        print "rms deviation pert2 pert1 ="+str(np.linalg.norm((response_pert1-response_pert2)))
#
#    do_w0wa_convergence_test = False
#    if do_w0wa_convergence_test:
#        w_use = -1.
#        eps_use = 0.07
#        pmodel_use = 'halofit'
#
#        mp_params_const1 = defaults.matter_power_params.copy()
#        mp_params_const1['w_step'] = 0.005
#        mp_params_const1['w_edge'] = 2.*mp_params_const1['w_step']
#
#        mp_params_const2 = mp_params_const1.copy()
#        mp_params_const2['w_step'] = 0.0005
#        mp_params_const2['w_edge'] = 2.*mp_params_const2['w_step']
#
#        cosmo_const = cosmo_in.copy()
#        cosmo_const['de_model'] = 'constant_w'
#        cosmo_const['w'] = w_use
#        cosmo_const['w0'] = w_use
#        cosmo_const['wa'] = 0.
#
#        cosmo_pert = cosmo_const.copy()
#        cosmo_pert['de_model'] = 'w0wa'
#        cosmo_pert['wa']+=eps_use
#
#        C_const = cp.CosmoPie(cosmo_const,camb_params=camb_params)
#        C_pert = cp.CosmoPie(cosmo_pert,camb_params=camb_params)
#
#        mps_const = MatterPower(C_const,camb_params=camb_params,matter_power_params = mp_params_const1)
#        mps_pert1 = MatterPower(C_pert,camb_params=camb_params,matter_power_params = mp_params_const1)
#        mps_pert2 = MatterPower(C_pert,camb_params=camb_params,matter_power_params = mp_params_const2)
#
#        zs = np.arange(0.,1.2,0.3)
#
#        P_const = mps_const.get_matter_power(zs,pmodel=pmodel_use)
#        P_pert1 = mps_pert1.get_matter_power(zs,pmodel=pmodel_use)
#        P_pert2 = mps_pert2.get_matter_power(zs,pmodel=pmodel_use)
#        response_pert1 = (P_pert1-P_const)/eps_use
#        response_pert2 = (P_pert2-P_const)/eps_use
#        print "rms error pert2 pert1 ="+str(np.linalg.norm((response_pert1-response_pert2)/response_pert2))
#        print "rms deviation pert2 pert1 ="+str(np.linalg.norm((response_pert1-response_pert2)))
#
#    do_plottest=False
#    if do_plottest:
#        import matplotlib.pyplot as plt
#        zs = np.array([1.])
#        #Pzs = mps.get_matter_power(zs,pmodel='linear')
#        cosmo_2 = cosmo_in.copy()
#        cosmo_2['de_model']='w0wa'
#        cosmo_2['wa'] =0.#-0.5
#        cosmo_2['w0'] =-2.
#        cosmo_2['w'] =-2.
#        C_2 = cp.CosmoPie(cosmo_2,camb_params=camb_params)
#        mps2 = MatterPower(C_2,camb_params=camb_params)
#
#        P_lin2 = mps2.get_matter_power(zs,pmodel='linear')
#        P_hf2 = mps2.get_matter_power(zs,pmodel='halofit')
#        P_fpt2 = mps2.get_matter_power(zs,pmodel='fastpt')
#        P_hf2_alt = 2.*np.pi**2*((mps2.hf.D2_NL(mps2.k,zs).T)/mps2.k**3).T
#        #should be 1 for wa=0.
#        print "mean rat with hf alt ",np.average(P_hf2_alt/P_hf2)
#        plt.loglog(mps2.k,P_lin2)
#        plt.loglog(mps2.k,P_hf2)
#        plt.loglog(mps2.k,P_fpt2)
#        plt.loglog(mps2.k,P_hf2_alt)
#        plt.show()
#
#        #plt.semilogx(mps.k,(mps.camb_w_pows.T/mps.P_lin).T)
#        #plt.show()
#
#    do_timetest=False
#    if do_timetest:
#        camb_params['kmax']=10.
#        camb_params['maxkh']=10.
#        zs = np.arange(0.,2.0,0.001)
#        cosmo_2 = cosmo_in.copy()
#        cosmo_2['de_model']='w0wa'
#        cosmo_2['wa'] =-0.5#-0.5
#        cosmo_2['w0'] =-2.
#        cosmo_2['w'] =-2.
#        C_2 = cp.CosmoPie(cosmo_2,camb_params=camb_params)
#        mps2 = MatterPower(C_2,camb_params=camb_params)
#
#        P_lin2 = mps2.get_matter_power(zs,pmodel='linear')
#        P_hf2 = mps2.get_matter_power(zs,pmodel='halofit')
#        P_fpt2 = mps2.get_matter_power(zs,pmodel='fastpt')
#
#    do_cutoff_sensitivity_test = False
#    if do_cutoff_sensitivity_test:
#        camb_params_cut_1 = camb_params.copy()
#        camb_params_cut_1['maxk']=40.
#        camb_params_cut_1['maxkh']=400.
#        camb_params_cut_2 = camb_params_cut_1.copy()
#        camb_params_cut_2['maxk']=10.
#        camb_params_cut_2['maxkh']=10.
#
#        cosmo_in = cosmo_in.copy()
#        cosmo_in['de_model']='w0wa'
#        cosmo_in['wa'] =-0.5#-0.5
#        cosmo_in['w0'] =-2.
#        cosmo_in['w'] =-2.
#
#        zs = np.array([0.])
#
#
#        C_1 = cp.CosmoPie(cosmo_in,camb_params=camb_params_cut_1)
#        mps_1 = MatterPower(C_1,camb_params=camb_params_cut_1)
#        P_hf_1 = mps_1.get_matter_power(zs,pmodel='halofit')
#        P_fpt_1 = mps_1.get_matter_power(zs,pmodel='fastpt')
#        k_1 = mps_1.k
#
#        C_2 = cp.CosmoPie(cosmo_in,camb_params=camb_params_cut_2)
#        mps_2 = MatterPower(C_2,camb_params=camb_params_cut_2)
#        P_hf_2 = mps_2.get_matter_power(zs,pmodel='halofit')
#        P_fpt_2 = mps_2.get_matter_power(zs,pmodel='fastpt')
#        k_2 = mps_2.k
#
#        k_use = k_2
#        P_hf_1_shift = InterpolatedUnivariateSpline(k_1,P_hf_1,k=2,ext=2)(k_use)
#        P_fpt_1_shift = InterpolatedUnivariateSpline(k_1,P_fpt_1,k=2,ext=2)(k_use)
#        P_hf_2_shift = InterpolatedUnivariateSpline(k_2,P_hf_2,k=2,ext=2)(k_use)
#        P_fpt_2_shift = InterpolatedUnivariateSpline(k_2,P_fpt_2,k=2,ext=2)(k_use)
#
#        print "mean hf rat",np.average(P_hf_1_shift/P_hf_2_shift)
#        print "mean fpt rat",np.average(P_fpt_1_shift/P_fpt_2_shift)
