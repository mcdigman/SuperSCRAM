import numpy as np
import defaults
import FASTPTcode.FASTPT as FASTPT
import halofit 
from camb_power import camb_pow
import w_matcher 
from scipy.interpolate import RectBivariateSpline,InterpolatedUnivariateSpline
import cosmopie as cp
#class for a matter power spectrum which can get both linear and nonlinear power spectra as needed
#TODO clean up
#TODO treat w0 and w equivalently
class MatterPower:
    def __init__(self,C_in,P_lin=None,k_in=None,matter_power_params=defaults.matter_power_params,camb_params=defaults.camb_params,wmatcher_params=defaults.wmatcher_params,halofit_params=defaults.halofit_params,fpt_params=defaults.fpt_params,wm_in=None,wm_safe=False):
        #save all the input parameter sets
        self.params = matter_power_params
        self.camb_params = camb_params.copy()
        self.camb_params['return_sigma8']=True
        self.wmatcher_params = wmatcher_params
        self.halofit_params = halofit_params
        self.fpt_params = fpt_params
        self.wm_safe=wm_safe

        self.C = C_in
        self.cosmology = self.C.cosmology

        if P_lin is None or k_in is None:
            k_camb,self.P_lin,self.sigma8_in = camb_pow(self.cosmology,camb_params=self.camb_params)
            #TODO check handling fixing sigma8 right
            if k_in is None:
                self.k = k_camb
            else:
                if not np.all(k_camb==k_in):    
                    #TODO extrapolate properly
                    self.P_lin = InterpolatedUnivariateSpline(k_camb,self.P_lin,k=2)(k_in)
                self.k=k_in
        else:
            self.k = k_in
            self.P_lin = P_lin

        self.a_grid = np.arange(self.params['a_min'],self.params['a_max'],self.params['a_step'])
        self.z_grid = 1./self.a_grid-1.
        self.n_a = self.a_grid.size

        self.de_model = self.cosmology['de_model']  
        if self.params['needs_wmatcher']:
            if wm_in is not None and self.wm_safe:
                self.wm = wm_in
            else:
                self.wm = w_matcher.WMatcher(self.C,wmatcher_params=self.wmatcher_params)
            w_match_grid = self.wm.match_w(self.C,self.z_grid)
            #self.pow_mult_grid = self.wm.match_growth(self.C,self.z_grid,w_match_grid)
            self.use_match_grid = True
        else:
            self.wm = None
            w_match_grid = np.zeros(self.n_a)+self.cosmology['w']
            self.pow_mult_grid = np.zeros(self.n_a)+1.
            self.use_match_grid = False
        #figure out an error checking method here
        #if np.any(w_match_grid>0.1):
        #Interpolated to capture possible k dependence in camb of w,#TODO see if sufficient
        #needed because field can cluster on very large scales, see arXiv:astro-ph/9906174
        if self.params['needs_camb_w_grid'] and self.use_match_grid:
            #get a w grid that is no larger than it needs to be
            w_min = np.min(w_match_grid)-self.params['w_edge']
            w_max = np.max(w_match_grid)+self.params['w_edge']
            self.camb_w_grid = np.arange(w_min,w_max,self.params['w_step'])
            #make sure the grid has at least some minimum number of values
            if self.camb_w_grid.size<self.params['min_n_w']:
                self.camb_w_grid = np.linspace(w_min,w_max,self.params['min_n_w']) 

            self.n_cw = self.camb_w_grid.size
            self.camb_cosmos = np.zeros(self.n_cw,dtype=object)
            self.camb_Cs = np.zeros(self.n_cw,dtype=object)
            self.camb_w_pows = np.zeros((self.k.size,self.n_cw))
            self.camb_sigma8s = np.zeros(self.n_cw)
            print "camb grid",self.camb_w_grid
            #get grid of camb power spectra for various w values 
            for i in range(0,self.n_cw):
                self.camb_cosmos[i] = self.cosmology.copy()
                self.camb_cosmos[i]['de_model'] = 'constant_w'
                self.camb_cosmos[i]['w'] = self.camb_w_grid[i]
                #TODO check the G input is correct
                #print self.wm.growth_interp(self.camb_w_grid[i],C_in.a_grid)
                print "camb with w=",self.camb_w_grid[i]
                k_i,self.camb_w_pows[:,i],self.camb_sigma8s[i] = camb_pow(self.camb_cosmos[i],camb_params=self.camb_params)
                if not np.all(k_i==self.k):
                    self.camb_w_pows[:,i] = InterpolatedUnivariateSpline(k_i,self.camb_w_pows[:,i],k=2)(self.k)
            self.camb_w_interp = RectBivariateSpline(self.k,self.camb_w_grid,self.camb_w_pows,kx=1,ky=1)
            self.use_camb_grid = True
        else:
            self.camb_w_grid = np.array([])
            self.n_cw = 0
            self.camb_cosmos= np.array([])
            self.camb_w_pows = np.array([])
            self.camb_Cs = np.array([])
            self.camb_w_interp = None
            self.use_camb_grid = False
#        self.use_camb_grid = False
        


        if self.params['needs_halofit']:
            self.hf=halofit.halofitPk(self.C,self.k,p_lin=self.P_lin,halofit_params=self.halofit_params) 
            #if self.use_camb_grid:
            #    self.hfs = np.zeros(self.n_cw,dtype=object)
            #    self.camb_Cs = np.zeros(self.n_cw,dtype=object)
            #    self.hf_w_pows = np.zeros((self.k.size,self.n_cw))
            #    for i in range(0,self.n_cw):
            #        self.camb_Cs[i] = cp.CosmoPie(cosmology=self.camb_cosmos[i],silent=True,G_safe=True,G_in=InterpolatedUnivariateSpline(C_in.z_grid[::-1],self.wm.growth_interp(self.camb_w_grid[i],C_in.a_grid)[::-1],ext=2,k=2))
            #        self.hfs[i] = halofit.halofitPk(self.camb_Cs[i],self.k,self.camb_w_pows[:,i])
             #       self.hf_w_pows[:,i] = hfs[i].D2_NL(self.k,w_overwride=True,fixed_w=self.camb_w_grid[i])
           # else:
            #    self.hf_w_pows = np.array([])
          #      self.hfs = np.array([])
        else:
            self.hf = None
            #self.camb_Cs = np.array([])
           # self.hf_w_pows = np.array([])
         #   self.hfs = np.array([])

        if self.params['needs_fpt']:
            self.fpt = FASTPT.FASTPT(self.k,self.fpt_params['nu'],low_extrap=self.fpt_params['low_extrap'],high_extrap=self.fpt_params['high_extrap'],n_pad=self.fpt_params['n_pad'])
            self.one_loop = self.fpt.one_loop(self.P_lin,C_window=self.fpt_params['C_window'])
           # if self.use_camb_grid:
           #     self.one_loops = np.zeros((self.k,self.n_cw))
           #     for i in range(0,self.n_cw):
           #         self.one_loops[:,i] = self.fpt.one_loop(self.camb_w_pows[:,i],C_window=fpt_params['C_window'])
           # self.one_loops = np.array([])
        else:
            self.fpt = None
            self.one_loop=np.array([])
           # self.one_loops = np.array([])
    #get an effective sigma8 if appropriate
    def get_sigma8_eff(self,zs):
        if self.params['needs_wmatcher']:
            w_match_grid = self.wm.match_w(self.C,zs)
            pow_mult_grid = self.wm.match_growth(self.C,zs,w_match_grid)
            return InterpolatedUnivariateSpline(self.camb_w_grid,self.camb_sigma8s,k=2)(w_match_grid)*np.sqrt(pow_mult_grid)
        else:
            return self.sigma8_in

    def linear_power(self,zs,const_pow_mult=1.):
        if self.params['needs_wmatcher']:
            w_match_grid = self.wm.match_w(self.C,zs)
            pow_mult_grid = self.wm.match_growth(self.C,zs,w_match_grid)*const_pow_mult
        else:
            w_match_grid = np.zeros(zs.size)+self.cosmology.w
            pow_mult_grid = np.zeros(zs.size)+1.*const_pow_mult
        Gs = self.C.G_norm(zs)
        Ps = np.zeros((self.k.size,zs.size))
        if self.use_camb_grid:
            for i in range(0,zs.size):
                Ps[:,i]=Gs[i]**2*pow_mult_grid[i]*self.camb_w_interp(self.k,w_match_grid[i]).flatten()
        else:
            Ps = np.outer(self.P_lin,Gs**2*pow_mult_grid)
        return Ps
    #const_pow_mult allows adjusting sigma8 without creating a whole new power spectrum
    def nonlinear_power(self,zs,pmodel=defaults.matter_power_params['nonlinear_model'],const_pow_mult=1.,get_one_loop=False):
        if self.params['needs_wmatcher']:
            w_match_grid = self.wm.match_w(self.C,zs)
            pow_mult_grid = self.wm.match_growth(self.C,zs,w_match_grid)*const_pow_mult
        else:
            w_match_grid = np.zeros(zs.size)+self.cosmology.w
            pow_mult_grid = np.zeros(zs.size)+1.*const_pow_mult
            
        G_norms = self.C.G_norm(zs)
        Pbases = np.zeros((self.k.size,zs.size))
        if self.use_camb_grid:
            for i in range(0,zs.size):
                Pbases[:,i]=pow_mult_grid[i]*self.camb_w_interp(self.k,w_match_grid[i]).flatten()
        else:
            Pbases = np.outer(self.P_lin,pow_mult_grid)
        
        Plin = G_norms**2*Pbases 
        P_nonlin = np.zeros((self.k.size,zs.size))
        self.hf_C_calcs = np.zeros(zs.size,dtype=object)
        self.hf_calcs = np.zeros(zs.size,dtype=object)
        one_loops = np.zeros((self.k.size,zs.size))
        if pmodel=='halofit':
            for i in range(0,zs.size):
                cosmo_hf_i = self.cosmology.copy()
                cosmo_hf_i['de_model'] = 'constant_w'
                cosmo_hf_i['w'] = w_match_grid[i]
                self.hf_C_calcs[i] = cp.CosmoPie(cosmology=cosmo_hf_i,silent=True,G_safe=True,G_in=InterpolatedUnivariateSpline(self.C.z_grid[::-1],self.wm.growth_interp(w_match_grid[i],self.C.a_grid)[::-1],ext=2,k=2))
                self.hf_calcs[i] = halofit.halofitPk(self.hf_C_calcs[i],self.k,p_lin=Pbases[:,i],halofit_params=self.halofit_params)
                P_nonlin[:,i] = 2.*np.pi**2*(self.hf_calcs[i].D2_NL(self.k,zs[i]).T/self.k**3)
        elif pmodel=='fastpt':
            for i in range(0,zs.size):
                one_loops[:,i] = self.fpt.one_loop(Pbases[:,i],C_window=self.fpt_params['C_window'])
                G_i = G_norms[i]
                P_nonlin[:,i] =  Pbases[:,i]*G_i**2+one_loops[:,i]*G_i**4
        if pmodel=='fastpt' and get_one_loop:
            return P_nonlin,one_loops
        elif get_one_loop:
            raise ValueError('could not get one loop power spectrum for pmodel '+str(pmodel))
        else:
            return P_nonlin


                
if __name__ == '__main__':
    cosmo_in = defaults.cosmology.copy()
    camb_params = defaults.camb_params.copy()
    camb_params['kmax']=10.
    camb_params['maxkh']=400.
    #C_in = cp.CosmoPie(cosmo_in,camb_params=camb_params)
    #mps = MatterPower(C_in)
    do_plottest=False
    if do_plottest:
        import matplotlib.pyplot as plt
        zs = np.array([1.])
        #Pzs = mps.linear_power(zs)
        cosmo_2 = cosmo_in.copy()
        cosmo_2['de_model']='w0wa'
        cosmo_2['wa'] =0.#-0.5
        cosmo_2['w0'] =-2.
        cosmo_2['w'] =-2.
        C_2 = cp.CosmoPie(cosmo_2,camb_params=camb_params)
        mps2 = MatterPower(C_2,camb_params=camb_params)
        
        P_lin2 = mps2.linear_power(zs)
        P_hf2 = mps2.nonlinear_power(zs,pmodel='halofit')
        P_fpt2 = mps2.nonlinear_power(zs,pmodel='fastpt')
        P_hf2_alt = 2.*np.pi**2*((mps2.hf.D2_NL(mps2.k,zs).T)/mps2.k**3).T
        #should be 1 for wa=0.
        print "mean rat with hf alt ",np.average(P_hf2_alt/P_hf2)
        plt.loglog(mps2.k,P_lin2)
        plt.loglog(mps2.k,P_hf2)
        plt.loglog(mps2.k,P_fpt2)
        plt.loglog(mps2.k,P_hf2_alt)
        plt.show()

        #plt.semilogx(mps.k,(mps.camb_w_pows.T/mps.P_lin).T)
        #plt.show()
    do_timetest=True
    if do_timetest:
        camb_params['kmax']=10.
        camb_params['maxkh']=10.
        zs = np.arange(0.,2.0,0.001)
        cosmo_2 = cosmo_in.copy()
        cosmo_2['de_model']='w0wa'
        cosmo_2['wa'] =-0.5#-0.5
        cosmo_2['w0'] =-2.
        cosmo_2['w'] =-2.
        C_2 = cp.CosmoPie(cosmo_2,camb_params=camb_params)
        mps2 = MatterPower(C_2,camb_params=camb_params)
        
        P_lin2 = mps2.linear_power(zs)
        P_hf2 = mps2.nonlinear_power(zs,pmodel='halofit')
        P_fpt2 = mps2.nonlinear_power(zs,pmodel='fastpt')
    do_cutoff_sensitivity_test = False
    if do_cutoff_sensitivity_test:
        camb_params_cut_1 = camb_params.copy()
        camb_params_cut_1['maxk']=40.
        camb_params_cut_1['maxkh']=400.
        camb_params_cut_2 = camb_params_cut_1.copy()
        camb_params_cut_2['maxk']=10.
        camb_params_cut_2['maxkh']=10.

        cosmo_in = cosmo_in.copy()
        cosmo_in['de_model']='w0wa'
        cosmo_in['wa'] =-0.5#-0.5
        cosmo_in['w0'] =-2.
        cosmo_in['w'] =-2.

        zs = np.array([0.])
        

        C_1 = cp.CosmoPie(cosmo_in,camb_params=camb_params_cut_1)
        mps_1 = MatterPower(C_1,camb_params=camb_params_cut_1)
        P_hf_1 = mps_1.nonlinear_power(zs,pmodel='halofit')
        P_fpt_1 = mps_1.nonlinear_power(zs,pmodel='fastpt')
        k_1 = mps_1.k
        
        C_2 = cp.CosmoPie(cosmo_in,camb_params=camb_params_cut_2)
        mps_2 = MatterPower(C_2,camb_params=camb_params_cut_2) 
        P_hf_2 = mps_2.nonlinear_power(zs,pmodel='halofit')
        P_fpt_2 = mps_2.nonlinear_power(zs,pmodel='fastpt')
        k_2 = mps_2.k

        k_use = k_2
        P_hf_1_shift = InterpolatedUnivariateSpline(k_1,P_hf_1,k=2,ext=2)(k_use)
        P_fpt_1_shift = InterpolatedUnivariateSpline(k_1,P_fpt_1,k=2,ext=2)(k_use)
        P_hf_2_shift = InterpolatedUnivariateSpline(k_2,P_hf_2,k=2,ext=2)(k_use)
        P_fpt_2_shift = InterpolatedUnivariateSpline(k_2,P_fpt_2,k=2,ext=2)(k_use)

        print "mean hf rat",np.average(P_hf_1_shift/P_hf_2_shift)
        print "mean fpt rat",np.average(P_fpt_1_shift/P_fpt_2_shift)
