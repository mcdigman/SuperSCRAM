"""
MatterPower class handles and wrap all matter power spectra related functionality
"""
from __future__ import division,print_function,absolute_import
from builtins import range
from warnings import warn
import numpy as np
from scipy.interpolate import RectBivariateSpline,InterpolatedUnivariateSpline
from camb_power import camb_pow

import FASTPTcode.FASTPT as FASTPT
import cosmopie as cp

import halofit
import w_matcher
from extrap_utils import power_law_extend
#class for a matter power spectrum which can get both linear and nonlinear power spectra as needed
#TODO clean up
class MatterPower(object):
    """A generic matter power spectrum which can provide different types of linear or nonlinear (halofit, FAST-PT) power spectra as needed"""
    def __init__(self,C_in,power_params,k_in=None,wm_in=None,P_fid=None,camb_safe=False,de_perturbative=False):
        """Generate matter power spectrum for input cosmology
        linear power spectrum use camb, nonlinear can use halofit or FAST-PT
        Inputs:
        C_in: input CosmoPie
        matter_power_params,camb_params,wmatcher_params,halofit_params,fpt: dictionaries of parameters
        wm_in: input WMatcher object. Optional.
        P_fid: Fiducial  power spectrum. Optional.
        camb_safe: If True and P_fid is not None, will borrow camb_grid from P_fid if possible.
                Useful if only linear growth factor different from P_fid.
        de_perturbative: If True, get power spectra for constant w even if w(z) in C_in is not constant
        """

        #save all the input parameter sets
        self.power_params = power_params
        self.params = power_params.matter_power
        self.camb_params = power_params.camb.copy()
        self.camb_params['return_sigma8'] = True
        self.de_perturbative = de_perturbative

        self.C = C_in
        self.cosmology = self.C.cosmology

        #give error if extrapolation is more than offset between camb default H0 and our H0
        self.extend_limit = self.cosmology['H0']/71.902712048990196
        self.extend_limit = np.max([self.extend_limit,1./self.extend_limit])+0.001

        self.k_camb,self.P_lin,self.sigma8_in = camb_pow(self.cosmology,camb_params=self.camb_params)
        if k_in is None:
            self.k = self.k_camb
        else:
            self.P_lin = power_law_extend(self.k_camb,self.P_lin,k_in,k=3,extend_limit=self.extend_limit)
            self.k = k_in

        self.a_grid = np.arange(self.params['a_max'],self.params['a_min']-self.params['a_step']/10.,-self.params['a_step'])
        self.z_grid = 1./self.a_grid-1.
        self.n_a = self.a_grid.size

        self.de_model = self.cosmology['de_model']
        self.get_w_matcher = self.params['needs_wmatcher'] and not self.de_model=='constant_w' and not self.de_perturbative
        if self.get_w_matcher:
            if wm_in is not None:
                self.wm = wm_in
            else:
                self.wm = w_matcher.WMatcher(self.C,self.power_params.wmatcher)
            self.w_match_grid = self.wm.match_w(self.C,self.z_grid)
            self.growth_match_grid = self.wm.match_growth(self.C,self.z_grid,self.w_match_grid)
            self.w_match_interp = InterpolatedUnivariateSpline(self.z_grid,self.w_match_grid,k=3,ext=2)
            self.growth_match_interp = InterpolatedUnivariateSpline(self.z_grid,self.growth_match_grid,k=3,ext=2)
            self.use_match_grid = True
        else:
            self.wm = None
            self.w_match_grid = None
            self.w_match_interp = None
            self.growth_match_grid = None
            self.growth_match_interp = None
            self.use_match_grid = False


        #figure out an error checking method here
        #Interpolated to capture possible k dependence in camb of w
        #needed because field can cluster on very large scales, see arXiv:astro-ph/9906174
        if self.params['needs_camb_w_grid'] and self.use_match_grid:

            #get grid of camb power spectra for various w values
            #borrow some parameters from an input power spectrum if camb need not be called repeatedly
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
                    raise ValueError('Insufficient range in given camb w grid, needed min '+str(np.min(self.w_match_grid))+' max '+str(np.max(self.w_match_grid)))
                else:
                    cache_usable = True
            if not cache_usable:
                #get a w grid that is no larger than it needs to be, keeping integer number of steps from w=-1 for convenience
                n_w_below = np.ceil((-1.-np.min(self.w_match_grid-self.params['w_edge']-self.params['w_step']))/self.params['w_step'])
                n_w_above = np.ceil((1.+np.max(self.w_match_grid+self.params['w_edge']+self.params['w_step']))/self.params['w_step'])
                self.w_min = -1.-n_w_below*self.params['w_step']
                self.w_max = -1.+n_w_above*self.params['w_step']
                min_step = np.min(np.abs(np.diff(self.w_match_grid)))
                if min_step<self.params['w_step']:
                    warn('step size '+str(self.params['w_step'])+' may be insufficient for grid with min step '+str(min_step))
                self.camb_w_grid = np.arange(self.w_min,self.w_max,self.params['w_step'])
                #make sure the grid has at least some minimum number of values
                if self.camb_w_grid.size<self.params['min_n_w']:
                    self.camb_w_grid = np.linspace(self.w_min,self.w_max,self.params['min_n_w'])
                print("matter_power_spectrum: camb grid",self.camb_w_grid)
                print("matter_power_spectrum: w grid",self.w_match_grid)
                print("matter_power_spectrum: w grid diffs",np.diff(self.w_match_grid))

                n_cw = self.camb_w_grid.size
                self.camb_w_pows = np.zeros((self.k.size,n_cw))
                self.camb_sigma8s = np.zeros(n_cw)
                camb_cosmos = self.cosmology.copy()
                camb_cosmos['de_model'] = 'constant_w'
                for i in range(0,n_cw):
                    camb_cosmos['w'] = self.camb_w_grid[i]
                    print("camb with w=",self.camb_w_grid[i])
                    k_i,self.camb_w_pows[:,i],self.camb_sigma8s[i] = camb_pow(camb_cosmos,camb_params=self.camb_params)
                    self.camb_w_pows[:,i] = power_law_extend(k_i,self.camb_w_pows[:,i],self.k,k=3,extend_limit=self.extend_limit)
                self.camb_w_interp = RectBivariateSpline(self.k,self.camb_w_grid,self.camb_w_pows,kx=3,ky=3)
                self.camb_sigma8_interp = InterpolatedUnivariateSpline(self.camb_w_grid,self.camb_sigma8s,k=3,ext=2)
                self.use_camb_grid = True
        else:
            self.w_min = None
            self.w_max = None
            self.camb_w_grid = np.array([])
            self.camb_w_pows = np.array([])
            self.camb_sigma8s = np.array([])
            self.camb_w_interp = None
            self.camb_sigma8_interp = None
            self.use_camb_grid = False

        #don't bother actually using the match grid if it is constant anyway
        if (self.C.de_model=='w0wa' and self.cosmology['wa']==0.):
            self.use_match_grid = False

        if self.params['needs_fpt']:
            fpt_p = self.power_params.fpt
            self.fpt = FASTPT.FASTPT(self.k,fpt_p['nu'],None,fpt_p['low_extrap'],fpt_p['high_extrap'],fpt_p['n_pad'])
        else:
            self.fpt = None

    def get_sigma8_eff(self,zs):
        """get effective sigma8(z) for matching power spectrum amplitude in w(z) models from WMatcher"""
        if self.use_match_grid:
            w_match_grid = self.w_match_interp(zs)
            #pow_mult_grid = self.wm.match_growth(self.C,zs,w_match_grid)
            pow_mult_grid = self.growth_match_interp(zs)
            return self.camb_sigma8_interp(w_match_grid)*np.sqrt(pow_mult_grid)
        else:
            return self.sigma8_in+np.zeros(zs.size)

    #const_pow_mult allows adjusting sigma8 without creating a whole new power spectrum
    def get_matter_power(self,zs_in,pmodel='linear',const_pow_mult=1.,get_one_loop=False):
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

        G_norms = self.C.G_norm(zs)

        if self.use_match_grid:
            w_match_grid = self.w_match_interp(zs)
            pow_mult_grid = self.growth_match_interp(zs)*const_pow_mult

            Pbases = np.zeros((self.k.size,n_z))
            if self.use_camb_grid:
                for i in range(0,n_z):
                    Pbases[:,i] = pow_mult_grid[i]*self.camb_w_interp(self.k,w_match_grid[i]).flatten()
            else:
                Pbases = np.outer(self.P_lin,pow_mult_grid)
        else:
            Pbases = np.outer(self.P_lin,np.full(n_z,1.))*const_pow_mult

        P_nonlin = np.zeros((self.k.size,n_z))
        if pmodel=='linear':
            P_nonlin = Pbases*G_norms**2
        elif pmodel=='halofit':
            if self.use_match_grid:
                for i in range(0,n_z):
                    cosmo_hf_i = self.cosmology.copy()
                    cosmo_hf_i['de_model'] = 'constant_w'
                    cosmo_hf_i['w'] = w_match_grid[i]
                    G_hf = InterpolatedUnivariateSpline(self.C.z_grid,self.wm.growth_interp(w_match_grid[i],self.C.a_grid),ext=2,k=3)
                    hf_C_calc = cp.CosmoPie(cosmo_hf_i,self.C.p_space,silent=True,G_safe=True,G_in=G_hf)
                    hf_C_calc.k = self.k
                    hf_calc = halofit.HalofitPk(hf_C_calc,Pbases[:,i],self.power_params.halofit,self.camb_params['leave_h'])
                    P_nonlin[:,i] = 2.*np.pi**2*(hf_calc.D2_NL(self.k,zs[i]).T/self.k**3)
            else:
                hf_calc = halofit.HalofitPk(self.C,self.P_lin*const_pow_mult,self.power_params.halofit,self.camb_params['leave_h'])
                P_nonlin = 2.*np.pi**2*(hf_calc.D2_NL(self.k,zs).T/self.k**3).T

        elif pmodel=='fastpt':
            if self.use_match_grid:
                one_loops = np.zeros((self.k.size,n_z))
                for i in range(0,n_z):
                    G_i = G_norms[i]
                    one_loops[:,i] = self.fpt.one_loop(Pbases[:,i],C_window=self.power_params.fpt['C_window'])*G_i**4
                    P_nonlin[:,i] =  Pbases[:,i]*G_i**2+one_loops[:,i]
            else:
                one_loops = np.outer(self.fpt.one_loop(self.P_lin,C_window=self.power_params.fpt['C_window']),G_norms**4)
                P_nonlin = np.outer(self.P_lin,G_norms**2)+one_loops

        if pmodel=='fastpt' and get_one_loop:
            return P_nonlin,one_loops
        elif get_one_loop:
            raise ValueError('could not get one loop power spectrum for pmodel '+str(pmodel))
        else:
            return P_nonlin
