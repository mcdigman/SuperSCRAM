"""
Handles lensing observable power spectrum
"""
from time import time
from warnings import warn 
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline,RectBivariateSpline

from power_response import dp_ddelta
from lensing_weight import QShear,QMag,QNum,QK
from lensing_source_distribution import get_source_distribution
from algebra_utils import trapz2

import numpy as np
import cosmopie as cp
import scipy.special as spp
import defaults
import matter_power_spectrum as mps
#TODO create derivative class than can change whole cosmology
#TODO check integral boundaries ok
#TODO remove explicit cosmosis pmodel
class ShearPower(object):
    def __init__(self,C,zs,ls,omega_s,pmodel='halofit',ps=np.array([]),params=defaults.lensing_params,mode='power'):
        """
            Handles lensing power spectra
            inputs:
                C: CosmoPie object
                zs: z grid
                ls: l bins to use
                omega_s: angular area of window in radians 
                pmodel: nonlinear power spectrum model to use, options 'linear','halofit','fastpt','cosmosis'
                ps: lensing source distribution. Optional.
                mode: whether to get power spectrum 'power' or dC/d\bar{\delta} 'dc_ddelta'
        """
        self.k_in = C.k
        self.C = C
        self.zs = zs
        self.ls = ls
        self.epsilon = params['epsilon']
        
        #self.omega_s = np.pi/(3.*np.sqrt(2))
        #self.n_gal = 286401.
        #118000000 galaxies/rad^2 if 10/arcmin^2 and omega_s is area in radians of field
        self.omega_s = omega_s
        self.n_gal = params['n_gal']
        self.pmodel = pmodel
        self.sigma2_e = params['sigma2_e']
        self.sigma2_mu = params['sigma2_mu']
        

        self.n_map = {QShear:{QShear:(self.sigma2_e/(2.*self.n_gal))}}
        #print self.n_map[QShear][QShear]
        self.n_k = self.k_in.size
        self.n_l = self.ls.size
        self.n_z = self.zs.size
        
        self.p_dd_use=np.zeros((self.n_l,self.n_z))
        self.ps = np.zeros(self.n_z)

        self.params = params
        self.mode=mode
                    
        #some methods require special handling    
        if self.mode == 'power':
            if pmodel == 'cosmosis':
                z_bar = np.loadtxt('test_inputs/proj_2/z.txt')
                k_bar = np.loadtxt('test_inputs/proj_2/k_h.txt')*C.h
                self.pow_interp = RectBivariateSpline(k_bar,z_bar,(np.loadtxt('test_inputs/proj_2/p_k.txt')/C.h**3).T,kx=1,ky=1)
                self.chis = interp1d(z_bar,np.loadtxt('test_inputs/proj_2/d_m.txt')[::-1])(zs)
                self.chi_As = self.chis
            elif pmodel=='halofit':
                self.pow_interp = RectBivariateSpline(self.k_in,self.zs,self.C.P_lin.get_matter_power(self.zs,pmodel='halofit'),kx=2,ky=2)
            elif pmodel=='fastpt':
                self.pow_interp = RectBivariateSpline(self.k_in,self.zs,self.C.P_lin.get_matter_power(self.zs,pmodel='fastpt'),kx=2,ky=2)
            elif pmodel=='linear':
                self.pow_interp = RectBivariateSpline(self.k_in,self.zs,self.C.P_lin.get_matter_power(self.zs,pmodel='linear'),kx=2,ky=2)
            else:
                raise ValueError('unrecognized pmodel\''+str(pmodel)+'\' for mode \''+str(self.mode)+'\'')
        elif self.mode == 'dc_ddelta':
            self.pow_interp = RectBivariateSpline(self.k_in,zs,dp_ddelta(self.k_in,self.C.P_lin,zs,C=self.C,pmodel=pmodel,epsilon=self.epsilon)[0],kx=2,ky=2) 
        else:
            raise ValueError('unrecognized mode \''+str(self.mode)+'\'')

        if pmodel!='cosmosis':
            self.chis = self.C.D_comov(zs)
            #TODO if using Omegak not 0, make sure chis and chi_As used consistently
            if C.Omegak==0:
                self.chi_As = self.chis
            else:
                warn('ShearPower may not support nonzero curvature consistently')
                self.chi_As = self.C.D_comov_A(zs)

        self.k_use = np.outer((self.ls+0.5),1./self.chi_As)
        
        #loop appears necessary due to uneven grid spacing in k_use
        for i in xrange(0,self.n_z):
            self.p_dd_use[:,i] = self.pow_interp(self.k_use[:,i],zs[i])[:,0]

        self.sc_as = 1/(1+self.zs)

        #galaxy galaxy and galaxy dm power spectra. Set to matter power spectra for now, could input bias model
        self.p_gg_use = self.p_dd_use 
        self.p_gd_use = self.p_dd_use 

        self.source_dist = get_source_distribution(self.params['smodel'],self.zs,self.chis,self.C,self.params,ps=ps) 
        self.ps = self.source_dist.ps
        self.z_min_dist = params['z_min_dist']
        self.z_max_dist = params['z_max_dist']

        #used in getting Cll_q_q for a given ShearPower
        self.Cll_kernel = 1./self.chi_As**2*self.p_dd_use

    def r_corr(self):
        """get correlation r between galaxy and dm power spectra, will be 1 unless input model for bias"""
        return self.p_gd_use/np.sqrt(self.p_dd_use*self.p_gg_use)
    
    def get_n_shape(self,class_1,class_2):
        """get shape noise associated with 2 types of observables"""
        try:
            return self.n_map[class_1][class_2]
        except KeyError:
            warn("unrecognized key for shape noise "+str(class_1)+" or "+str(class_2)+" using 0")
            return 0 

    #take as an array of qs, ns, and rs instead of the matrices themselves
    #ns is [n_ac,n_ad,n_bd,n_bc]
    def cov_g_diag(self,qs,ns_in=[0.,0.,0.,0.],rs=np.full(4,1.),delta_ls=None,ls=None):
        """Get diagonal elements of gaussian covariance between observables
            inputs:
                qs: a list of 4 QWeight objects [q1,q2,q3,q4]
                ns_in: an input list of shape noise [n13,n24,n14,n23]
                rs: correlation parameters, [r13,r24,r14,r23]. Optional.
                ls: array of ls to use if not default. Optional.
                delta_ls: array of differences of ls. Optional
        """
        cov_diag = np.zeros(self.n_l)
        if ls is None:
            ls = self.ls
        if delta_ls is None:
            #calculate diff with a ghost cell
            delta_ls = InterpolatedUnivariateSpline(np.arange(0,ls.size),ls,ext=2).derivative(1)(np.arange(0,ls.size))
        ns = np.zeros(ns_in.size)
        #TODO handle varying bin sizes better
        ns[0] = ns_in[0]/trapz2((1.*(self.chis>=qs[0].chi_min)*(self.chis<=qs[0].chi_max)*self.ps),self.chis)
        ns[1] = ns_in[1]/trapz2((1.*(self.chis>=qs[0].chi_min)*(self.chis<=qs[0].chi_max)*self.ps),self.chis)
        ns[2] = ns_in[2]/trapz2((1.*(self.chis>=qs[1].chi_min)*(self.chis<=qs[1].chi_max)*self.ps),self.chis)
        ns[3] = ns_in[3]/trapz2((1.*(self.chis>=qs[1].chi_min)*(self.chis<=qs[1].chi_max)*self.ps),self.chis)

        c_ac = Cll_q_q(self,qs[0],qs[2],rs[0]).Cll()#*np.sqrt(4.*np.pi)
        c_bd = Cll_q_q(self,qs[1],qs[3],rs[1]).Cll()#*np.sqrt(4.*np.pi) 
        c_ad = Cll_q_q(self,qs[0],qs[3],rs[2]).Cll()#*np.sqrt(4.*np.pi) 
        c_bc = Cll_q_q(self,qs[1],qs[2],rs[3]).Cll()#*np.sqrt(4.*np.pi) 
        cov_diag = 1./(self.omega_s*(2.*self.ls+1.)*delta_ls)*((c_ac+ns[0])*(c_bd+ns[2])+(c_ad+ns[1])*(c_bc+ns[3]))
        #(C13*C24+ C13*N24+N13*C24 + C14*C23+C14*N23+N14*C23+N13*N24+N14*N23)
        return cov_diag

    def tan_shear(self,thetas,with_limber=False):
        """stacked tangential shear, note ls must be successive integers to work correctly
        note that exact result is necessary if result must be self-consistent (ie tan_shear(theta)=tan_shear(theta+2*pi)) for theta not <<1
        see Putter & Takada 2010 arxiv:1007.4809
        not used by anything or tested"""
        n_t = thetas.size
        tans = np.zeros(n_t)
        kg_pow = Cll_k_g(self).Cll()
        for i in xrange(0,n_t):
            if with_limber:
                tans[i] = trapz2((2.*self.ls+1.)/(4.*np.pi*self.ls*(self.ls+1.))*kg_pow*spp.lpmv(2,ls,np.cos(thetas[i])),ls)
            else:
                tans[i] = trapz2(self.ls/(2.*np.pi)*kg_pow*spp.jn(2,thetas[i]*ls),ls)

        return tans

class Cll_q_q(object):
    def __init__(self,sp,q1s,q2s,corr_param=1.):
        """
        class for a generic lensing power spectrum
            inputs:
                sp: a ShearPower object
                q1s: a QWeight object
                q2s: a QWeight object
                corr_param: correlation parameter, could be an array
        """
        self.integrand = np.zeros((sp.n_z,sp.n_l))
        self.chis = sp.chis
        self.integrand = (corr_param*q1s.qs.T*q2s.qs.T*sp.Cll_kernel).T

    def Cll(self,chi_min=0,chi_max=np.inf):
        """get lensing power spectrum integrated in a specified range"""
        high_mask = (self.chis<=chi_max)*1.
        low_mask = (self.chis>=chi_min)*1.
        return trapz2((high_mask*low_mask*self.integrand.T).T,self.chis)

    def Cll_integrand(self):
        """get the integrand of the lensing power spectrum, if want to multiply by something else before integrating"""
        return self.integrand

class Cll_sh_sh(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        """Shear shear lensing power spectrum"""
        Cll_q_q.__init__(self,sp,QShear(sp,chi_max=chi_max1,chi_min=chi_min1),QShear(sp,chi_max=chi_max2,chi_min=chi_min2))

class Cll_g_g(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        """Galaxy Galaxy lensing power spectrum"""
        Cll_q_q.__init__(self,sp,QNum(sp,chi_max=chi_max1,chi_min=chi_min1),QNum(sp,chi_max=chi_max2,chi_min=chi_min2))

class Cll_mag_mag(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        """Magnification magnification lensing power spectrum"""
        Cll_q_q.__init__(self,sp,QMag(sp,chi_max=chi_max1,chi_min=chi_min1),QMag(sp,chi_max=chi_max2,chi_min=chi_min2))

class Cll_k_k(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        """Convergence convergence lensing power spectrum"""
        Cll_q_q.__init__(self,sp,QK(sp,chi_max=chi_max1,chi_min=chi_min1),QK(sp,chi_max=chi_max2,chi_min=chi_min2))

class Cll_k_g(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        """Convergence galaxy lensing power spectrum"""
        Cll_q_q.__init__(self,sp,QK(sp,chi_max=chi_max1,chi_min=chi_min1),QNum(sp,chi_max=chi_max2,chi_min=chi_min2))

class Cll_sh_mag(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        """Shear magnification lensing power spectrum"""
        Cll_q_q.__init__(self,sp,QShear(sp,chi_max=chi_max1,chi_min=chi_min1),QMag(sp,chi_max=chi_max2,chi_min=chi_min2))

class Cll_mag_g(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        """Magnification galaxy lensing power spectrum"""
        Cll_q_q.__init__(self,sp,QMag(sp,chi_max=chi_max1,chi_min=chi_min1),QNum(sp,chi_max=chi_max2,chi_min=chi_min2))

class Cll_sh_g(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        """Shear galaxy lensing power spectrum"""
        Cll_q_q.__init__(self,sp,QShear(sp,chi_max=chi_max1,chi_min=chi_min1),QNum(sp,chi_max=chi_max2,chi_min=chi_min2),corr_param=sp.r_corr())


#class Cll_q_q_nolimber(Cll_q_q):
#    def __init__(self,sp,q1s,q2s,corr_param=np.array([])):
#        integrand1 = np.zeros((sp.n_z,sp.n_l))
#        #integrand2 = np.zeros((sp.n_z,sp.n_l))
#        integrand_total = np.zeros((sp.n_z,sp.n_l))
#        for i in xrange(0,sp.n_z):
#            window_int1 = np.zeros((sp.n_z,sp.n_l))
#        #    window_int2 = np.zeros((sp.n_z,sp.n_l))
#            for j in xrange(0,sp.n_z):
#                window_int1[j] = q1s.qs[j]/np.sqrt(sp.chis[j])*spp.jv(sp.ls+0.5,(sp.ls+0.5)/sp.chis[i]*sp.chis[j])
#         #       window_int2[j] = q2s.qs[j]/np.sqrt(sp.chis[j])*spp.jv(sp.ls+0.5,(sp.ls+0.5)/sp.chis[i]*sp.chis[j])
#            integrand1[i] = np.trapz(window_int1,sp.chis,axis=0)
#         #   integrand2[i] = np.trapz(window_int2,sp.chis,axis=0)
#            integrand_total[i] = integrand1[i]*integrand1[i]*sp.p_dd_use[:,i]*(sp.ls+0.5)**2/sp.chis[i]**3
#        self.integrand = integrand_total
#        self.chis = sp.chis

#TODO make this work with current q design
#class Cll_q_q_order2(Cll_q_q):
#    def __init__(self,sp,q1s,q2s,corr_param=np.array([])):
#        integrand1 = np.zeros((sp.n_z,sp.n_l))
#        integrand2 = np.zeros((sp.n_z,sp.n_l))
#        if corr_param.size !=0: 
#            for i in xrange(0,sp.n_z):
#                integrand1[i] = q1s.qs[i]*q2s.qs[i]/sp.chis[i]**2*sp.p_dd_use[:,i]*corr_param[:,i]
#        else:
#            for i in xrange(0,sp.n_z):
#                integrand1[i] = q1s.qs[i]*q2s.qs[i]/sp.chis[i]**2*sp.p_dd_use[:,i]
#        for i in xrange(0,sp.n_z-1): #check edge case
#            term1 = sp.chis[i]**2/2.*(q1s.rs_d2[i]/q1s.rs[i]+q2s.rs_d2[i]/q2s.rs[i])
#            term2 = sp.chis[i]**3/6.*(q1s.rs_d3[i]/q1s.rs[i]+q2s.rs_d3[i]/q2s.rs[i])
#            integrand2[i] = -1./(sp.ls+0.5)**2*(term1+term2)*integrand1[i]
#        self.integrand=integrand1+integrand2
#        self.chis=sp.chis


if __name__=='__main__':
    
        C=cp.CosmoPie(cosmology=defaults.cosmology,p_space='jdem')
#       d = np.loadtxt('Pk_Planck15.dat')
        #d = np.loadtxt('camb_m_pow_l.dat')
        #k_in = np.logspace(-5,2,200,base=10)
        #k_in = d[:,0]
    #    k_in = np.loadtxt('test_inputs/proj_1/k_h.txt')
#        k_in = np.logspace(-4,5,5000,base=10)
#       zs = np.logspace(-2,np.log10(3),50,base=10)
        zs = np.arange(0.1,1.0,0.1)
       # zs = np.loadtxt('test_inputs/proj_1/z.txt')
       # zs[0] = 10**-3
        #zs = np.arange(0.005,2,0.0005)
        #ls = np.unique(np.logspace(0,4,3000,dtype=int))*1.0
        ls = np.arange(1,5000)
        epsilon = 0.0001
        #ls = np.loadtxt('test_inputs/proj_1/ell.txt')

        #ls = np.logspace(np.log10(2),np.log10(10000),3000,base=10)
#       ls = revd[:,0]
        #t1 = time()
        #cosmo_a = defaults.cosmology.copy()
        #k_a = d[:,0]
        #P_a = d[:,1]
        #k_a,P_a = cpow.camb_pow(C.cosmology)
        P_a = mps.MatterPower(C)
        C.P_lin=P_a
        k_a = P_a.k
        C.k = k_a
       # P_a = hf.HalofitPk(k_a,C=C).D2_L(k_a,0)
        omega_s=np.pi/(3.*np.sqrt(2.))
      
        
        import matplotlib.pyplot as plt
        do_power_response1 = False
        do_power_response2 = True
        do_power_response3 = False

        if do_power_response1:
            ax = plt.subplot(111)
            zbar = np.array([0.1,1.0])
            dcalt1,p1a = dp_ddelta(k_a,P_a,zbar,C,pmodel='linear')
            dcalt2,p2a = dp_ddelta(k_a,P_a,zbar,C,pmodel='halofit')
            dcalt3,p3a = dp_ddelta(k_a,P_a,zbar,C,pmodel='fastpt')

            ax.set_xlabel('l',size=20)
            plt.title('gaussian source with $\sigma=0.4$ centered at $z=1.0$')
            ax.set_ylabel('|$\\frac{\partial ln(P(k))}{\partial \\bar{\delta}}$|')
            ax.plot(k_a,abs(dcalt1/p1a))
            ax.plot(k_a,abs(dcalt2/p2a))
            ax.plot(k_a,abs(dcalt3/p3a))
            plt.xlim([0.,0.4])
            plt.ylim([1.2,3.2])
            ax.legend(['Linear','Halofit','1 loop fpt'],loc=3)
            plt.grid()
            plt.show()
        if do_power_response2:
            ax = plt.subplot(111)
            #sp_dch2 = ShearPower(C,zs,ls,omega_s,pmodel='halofit',param_vary='Omegamh2',mode='dc_dpar') 
            #sp1 = ShearPower(C,zs,ls,omega_s,pmodel='halofit',param_vary='Omegamh2',mode='power') 
            chi_min = C.D_comov(0.)
            chi_max = C.D_comov(0.9)
            #dc_ss_dch2 = Cll_sh_sh(sp_dch2,chi_max,chi_max,chi_min,chi_min).Cll()
            #q1 = QShear(sp1,chi_min,chi_max)
            #ax.loglog(ls,np.abs(dc_ss_dch2))
            plt.show()
        if do_power_response3:
            ax = plt.subplot(111)
            #sp_dmh2 = ShearPower(C,zs,ls,omega_s,pmodel='halofit',param_vary='Omegamh2',mode='dc_dpar') 
            #sp_dbh2 = ShearPower(C,zs,ls,omega_s,pmodel='halofit',param_vary='Omegabh2',mode='dc_dpar') 
            #sp_dlh2 = ShearPower(C,zs,ls,omega_s,pmodel='halofit',param_vary='OmegaLh2',mode='dc_dpar') 
            #sp_dns = ShearPower(C,zs,ls,omega_s,pmodel='halofit',param_vary='ns',mode='dc_dpar') 
            #sp_dLogAs = ShearPower(C,zs,ls,omega_s,pmodel='halofit',param_vary='LogAs',mode='dc_dpar') 
            #dc_ss_dmh2 = Cll_sh_sh(sp_dmh2).Cll()  
            #dc_ss_dbh2 = Cll_sh_sh(sp_dbh2).Cll()  
            #dc_ss_dlh2 = Cll_sh_sh(sp_dlh2).Cll()  
            #dc_ss_dns = Cll_sh_sh(sp_dns).Cll()  
            #dc_ss_dLogAs = Cll_sh_sh(sp_dLogAs).Cll()  
            #ax.loglog(ls,np.abs(dc_ss_dmh2))
            #ax.loglog(ls,np.abs(dc_ss_dbh2))
            #ax.loglog(ls,np.abs(dc_ss_dlh2))
            #ax.loglog(ls,np.abs(dc_ss_dns))
            #ax.loglog(ls,np.abs(dc_ss_dLogAs))
            #dc_ss_dLogAs is nearly identical to dc_ss_dlh2, chris says it is not surprising unless possibly if they are exactly identical
            #they do look to be approximately numericaly identical
            plt.show()
