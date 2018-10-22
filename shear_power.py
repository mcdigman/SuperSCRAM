"""
Handles lensing observable power spectrum
"""
from __future__ import division,print_function,absolute_import
from builtins import range
from warnings import warn
import numpy as np

from power_response import dp_ddelta
from lensing_weight import QShear#,QMag,QNum,QK
from lensing_source_distribution import get_source_distribution
from algebra_utils import trapz2
from extrap_utils import power_law_extend

class ShearPower(object):
    """handles lensing power spectra"""
    def __init__(self,C,z_fine,f_sky,params,mode='power',nz_matcher=None):
        """
            inputs:
                C: CosmoPie object
                z_fine: z grid
                f_sky: angular area of window in sky fraction
                mode: whether to get power spectrum 'power' or dC/d\\bar{\\delta} 'dc_ddelta'
                nz_matcher: an NZMatcher object, for getting source distribution and n_gal if not None
                params: see line by line description in defaults.py
        """
        print("Getting ShearPower")
        self.k_in = C.k
        self.k_max = np.max(self.k_in)
        self.C = C
        #self.zs = zs
        self.zs = z_fine
        self.params = params
        self.nz_matcher = nz_matcher

        #self.l_starts = np.logspace(np.log(params['l_min']),np.log(params['l_max']),params['n_l'],base=np.exp(1.))
        self.dlogl = (np.log(params['l_max'])-np.log(params['l_min']))/(params['n_l'])
        self.log_ls = np.log(params['l_min'])+np.arange(0,params['n_l']+1)*self.dlogl
        self.l_starts = np.exp(self.log_ls[:-1])
        self.l_ends = np.exp(self.log_ls[1:])
        self.l_mids = np.exp(self.log_ls[:-1]+self.dlogl/2.)
        self.delta_ls = np.diff(np.exp(self.log_ls))

        #118000000 galaxies/rad^2 if 10/arcmin^2 and f_sky is area in radians of field
        self.f_sky = f_sky
        self.pmodel = params['pmodel']

        #assume input ls is constant log spaced and gives starts of bins, need middle of bins
        #log_ls = np.log(self.l_starts)
        #dl = np.diff(log_ls)[0]
        #self.l_mids = np.exp(log_ls[0]+dl*(1./2.+np.arange(0,log_ls.size)))
        #self.delta_ls = np.diff(np.exp(log_ls[0]+dl*np.arange(0,log_ls.size+1)))

        self.n_gal = self.params['n_gal']
        if self.nz_matcher is not None:
            self.n_gal = self.nz_matcher.get_N_projected(self.zs,self.f_sky*4.*np.pi)
        if self.n_gal is None:
            raise ValueError('must specify n_gal in params or give an nz_matcher object')

        self.n_map = {QShear:{QShear:(self.params['sigma2_e']/(2.*self.n_gal))}}
        self.n_l = self.l_starts.size
        self.n_z = self.zs.size

        self.p_dd_use = np.zeros((self.n_l,self.n_z))
        self.ps = np.zeros(self.n_z)

        self.mode = mode

        #some methods require special handling
        if self.mode=='power':
            p_grid = self.C.P_lin.get_matter_power(self.zs,pmodel=self.pmodel)
        elif self.mode=='dc_ddelta':
            p_grid = dp_ddelta(self.C.P_lin,self.zs,self.C,self.pmodel,self.params['epsilon'])[0]
        else:
            raise ValueError('unrecognized mode \''+str(self.mode)+'\'')
        self.rs = self.C.D_comov(self.zs)
        #NOTE if using Omegak not 0, make sure rs and r_As used consistently
        if self.C.Omegak==0:
            self.r_As = self.rs
        else:
            warn('ShearPower may not support nonzero curvature consistently')
            self.r_As = self.C.D_comov_A(self.zs)

        self.k_use = np.outer((self.l_mids+0.5),1./self.r_As)

        #loop appears necessary due to uneven grid spacing in k_use
        assert not np.any(p_grid==0.)
        for i in range(0,self.n_z):
            self.p_dd_use[:,i] = power_law_extend(self.k_in,p_grid[:,i],self.k_use[:,i],k=3)
#            continue
#            k_range = (self.k_use[:,i]<=self.k_max) & (self.k_use[:,i]>=self.k_min)
#            k_loc = self.k_use[:,i][k_range]
#            self.p_dd_use[:,i][k_range] = interp1d(self.k_in,p_grid[:,i])(k_loc)
#            #self.p_dd_use[:,i] = self.pow_interp(self.k_use[:,i],self.zs[i])[:,0]

        assert not np.any(self.p_dd_use==0.)
        self.sc_as = 1/(1+self.zs)
        #galaxy galaxy and galaxy dm power spectra. Set to matter power spectra for now, could input bias model
        self.p_gg_use = self.p_dd_use
        self.p_gd_use = self.p_dd_use

        self.source_dist = get_source_distribution(self.params['smodel'],self.zs,self.rs,self.C,self.params,None,self.nz_matcher)
        self.ps = self.source_dist.ps

        #used in getting Cll_q_q for a given ShearPower
        self.Cll_kernel = 1./self.r_As**2*self.p_dd_use

        self.ps.flags['WRITEABLE'] = False
        self.p_dd_use.flags['WRITEABLE'] = False
        self.p_gg_use.flags['WRITEABLE'] = False
        self.p_gd_use.flags['WRITEABLE'] = False
        self.Cll_kernel.flags['WRITEABLE'] = False
        self.rs.flags['WRITEABLE'] = False
        self.k_use.flags['WRITEABLE'] = False
        self.l_starts.flags['WRITEABLE'] = False
        self.l_mids.flags['WRITEABLE'] = False
        print("Got ShearPower")

    def r_corr(self):
        """get correlation r between galaxy and dm power spectra, will be 1 unless input model for bias"""
        return self.p_gd_use/np.sqrt(self.p_dd_use*self.p_gg_use)

    def get_n_shape(self,class_1,class_2):
        """get shape noise associated with 2 types of observables over entire z range"""
        try:
            return self.n_map[class_1][class_2]
        except KeyError:
            warn("unrecognized key for shape noise "+str(class_1)+" or "+str(class_2)+" using 0")
            return 0

    #take as an array of qs, ns, and rcs instead of the matrices themselves
    #ns is [n_ac,n_ad,n_bd,n_bc]
    def cov_g_diag(self,qs,ns_in,rcs=None):
        """Get diagonal elements of gaussian covariance between observables
            inputs:
                qs: a list of 4 QWeight objects [q1,q2,q3,q4]
                ns_in: an input list of shape noise [n13,n24,n14,n23] [0.,0.,0.,0.] if no noise
                rcs: correlation parameters, [r13,r24,r14,r23]. Optional.
        """
        if rcs is None:
            rcs = np.full(4,1.)
        ns = np.zeros(ns_in.size)
        ns[0] = ns_in[0]/trapz2((((self.zs>=qs[0].z_min)&(self.zs<=qs[0].z_max))*self.ps),self.rs)
        ns[1] = ns_in[1]/trapz2((((self.zs>=qs[0].z_min)&(self.zs<=qs[0].z_max))*self.ps),self.rs)
        ns[2] = ns_in[2]/trapz2((((self.zs>=qs[1].z_min)&(self.zs<=qs[1].z_max))*self.ps),self.rs)
        ns[3] = ns_in[3]/trapz2((((self.zs>=qs[1].z_min)&(self.zs<=qs[1].z_max))*self.ps),self.rs)
        #could exploit/cache symmetries to reduce time
        c_ac = Cll_q_q(self,qs[0],qs[2],rcs[0]).Cll()
        c_bd = Cll_q_q(self,qs[1],qs[3],rcs[1]).Cll()
        c_ad = Cll_q_q(self,qs[0],qs[3],rcs[2]).Cll()
        c_bc = Cll_q_q(self,qs[1],qs[2],rcs[3]).Cll()
        cov_diag = 1./(self.f_sky*(2.*self.l_mids+1.)*self.delta_ls)*((c_ac+ns[0])*(c_bd+ns[2])+(c_ad+ns[1])*(c_bc+ns[3]))
        #(C13*C24+ C13*N24+N13*C24 + C14*C23+C14*N23+N14*C23+N13*N24+N14*N23)
        return cov_diag

#    def tan_shear(self,thetas,with_limber=False):
#        """stacked tangential shear, note ls must be successive integers to work correctly
#        note that exact result is necessary if result must be self-consistent (ie tan_shear(theta)=tan_shear(theta+2*pi)) for theta not<<1
#        see Putter & Takada 2010 arxiv:1007.4809
#        not used by anything or tested"""
#        n_t = thetas.size
#        tans = np.zeros(n_t)
#        kg_pow = Cll_k_g(self).Cll()
#        for i in range(0,n_t):
#            if with_limber:
#                tans[i] = trapz2((2.*self.ls+1.)/(4.*np.pi*self.ls*(self.ls+1.))*kg_pow*spp.lpmv(2,ls,np.cos(thetas[i])),ls)
#            else:
#                tans[i] = trapz2(self.ls/(2.*np.pi)*kg_pow*spp.jn(2,thetas[i]*ls),ls)
#
#        return tans

class Cll_q_q(object):
    """class for a generic lensing power spectrum"""
    def __init__(self,sp,q1s,q2s,corr_param=1.):
        """
            inputs:
                sp: a ShearPower object
                q1s: a QWeight object
                q2s: a QWeight object
                corr_param: correlation parameter, could be an array
        """
        self.rs = sp.rs
        self.zs = sp.zs
        self.integrand = (corr_param*q1s.qs.T*q2s.qs.T*sp.Cll_kernel).T

    def Cll(self,z_min=0.,z_max=np.inf):
        """get lensing power spectrum integrated in a specified range"""
        if z_min==0. and z_max==np.inf:
            return trapz2(self.integrand,self.rs)
        else:
            mask = ((self.zs<=z_max)&(self.zs>=z_min))
            return trapz2((mask*self.integrand.T).T,self.rs)

    def Cll_integrand(self):
        """get the integrand of the lensing power spectrum, if want to multiply by something else before integrating"""
        return self.integrand


#class Cll_q_q_nolimber(Cll_q_q):
#    def __init__(self,sp,q1s,q2s,corr_param=None):
#        integrand1 = np.zeros((sp.n_z,sp.n_l))
#        #integrand2 = np.zeros((sp.n_z,sp.n_l))
#        integrand_total = np.zeros((sp.n_z,sp.n_l))
#        for i in range(0,sp.n_z):
#            window_int1 = np.zeros((sp.n_z,sp.n_l))
#        #    window_int2 = np.zeros((sp.n_z,sp.n_l))
#            for j in range(0,sp.n_z):
#                window_int1[j] = q1s.qs[j]/np.sqrt(sp.rs[j])*spp.jv(sp.ls+0.5,(sp.ls+0.5)/sp.rs[i]*sp.rs[j])
#         #       window_int2[j] = q2s.qs[j]/np.sqrt(sp.rs[j])*spp.jv(sp.ls+0.5,(sp.ls+0.5)/sp.rs[i]*sp.rs[j])
#            integrand1[i] = np.trapz(window_int1,sp.rs,axis=0)
#         #   integrand2[i] = np.trapz(window_int2,sp.rs,axis=0)
#            integrand_total[i] = integrand1[i]*integrand1[i]*sp.p_dd_use[:,i]*(sp.ls+0.5)**2/sp.rs[i]**3
#        self.integrand = integrand_total
#        self.rs = sp.rs

#class Cll_q_q_order2(Cll_q_q):
#    def __init__(self,sp,q1s,q2s,corr_param=None):
#        integrand1 = np.zeros((sp.n_z,sp.n_l))
#        integrand2 = np.zeros((sp.n_z,sp.n_l))
#        if corr_param.size!=0:
#            for i in range(0,sp.n_z):
#                integrand1[i] = q1s.qs[i]*q2s.qs[i]/sp.rs[i]**2*sp.p_dd_use[:,i]*corr_param[:,i]
#        else:
#            for i in range(0,sp.n_z):
#                integrand1[i] = q1s.qs[i]*q2s.qs[i]/sp.rs[i]**2*sp.p_dd_use[:,i]
#        for i in range(0,sp.n_z-1): #check edge case
#            term1 = sp.rs[i]**2/2.*(q1s.rcs_d2[i]/q1s.rcs[i]+q2s.rcs_d2[i]/q2s.rcs[i])
#            term2 = sp.rs[i]**3/6.*(q1s.rcs_d3[i]/q1s.rcs[i]+q2s.rcs_d3[i]/q2s.rcs[i])
#            integrand2[i] = -1./(sp.ls+0.5)**2*(term1+term2)*integrand1[i]
#        self.integrand = integrand1+integrand2
#        self.rs = sp.rs
