import numpy as np
import cosmopie as cp
import halofit as hf
from numpy import exp
from scipy.interpolate import interp1d,interp2d
from scipy.interpolate import InterpolatedUnivariateSpline,SmoothBivariateSpline,RectBivariateSpline
import scipy.special as sp
from time import time
import FASTPTcode.FASTPT as FASTPT
import defaults
import camb_power as cpow
import FASTPTcode.matter_power_spt as mps
from warnings import warn 
from power_response import dp_ddelta
from power_parameter_response import dp_dparameter
from lensing_weight import q_shear,q_mag,q_num,q_k
#from lensing_source_distribution import GaussianZSource,ConstantZSource,CosmolikeZSource #TODO link this in
from lensing_source_distribution import get_source_distribution
import sph
import re
class shear_power:
    def __init__(self,k_in,C,zs,ls,omega_s,pmodel='halofit',P_in=np.array([]),ps=np.array([]),P_select=np.array([]),Cs=[],params=defaults.lensing_params,fpt_params=defaults.fpt_params,param_vary=None,mode='power'):
	self.k_in = k_in
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
        

        self.n_map = {q_shear:{q_shear:(self.sigma2_e/(2.*self.n_gal))}}
        print self.n_map[q_shear][q_shear]
        self.n_k = self.k_in.size
        self.n_l = self.ls.size
        self.n_z = self.zs.size
        
        self.omegas = np.zeros(self.n_z)

        self.p_dd_use=np.zeros((self.n_l,self.n_z))
	self.ps = np.zeros(self.n_z)

        self.params = params
                    
        #some methods require special handling    
        if pmodel == 'cosmosis':
            z_bar = np.loadtxt('test_inputs/proj_2/z.txt')
            k_bar = np.loadtxt('test_inputs/proj_2/k_h.txt')*C.h
            self.pow_interp = RectBivariateSpline(k_bar,z_bar,(np.loadtxt('test_inputs/proj_2/p_k.txt')/C.h**3).T,kx=1,ky=1)
            self.chis = interp1d(z_bar,np.loadtxt('test_inputs/proj_2/d_m.txt')[::-1])(zs)
            self.chi_As = self.chis
        elif pmodel=='halofit':
            self.halo = hf.halofitPk(self.C,k_in,P_in)
            self.pow_interp = RectBivariateSpline(self.k_in,self.zs,2*np.pi**2*(self.halo.D2_NL(self.k_in,self.zs).T/self.k_in**3).T)
        elif pmodel=='fastpt' or pmodel =='dc_fastpt':
            fpt = FASTPT.FASTPT(k_in,fpt_params['nu'],low_extrap=-5,high_extrap=5,n_pad=800)
            if pmodel=='dc_fastpt':
                self.pow_interp = RectBivariateSpline(k_in,zs,dp_ddelta(k_in,P_in,zs,C=self.C,pmodel='halofit',epsilon=self.epsilon,fpt=fpt,fpt_params=fpt_params)[0],kx=1,ky=1)
        elif pmodel=='dc_halofit':
            self.halo_a = hf.halofitPk(C,k_in,P_in)
            self.halo_b = hf.halofitPk(C,k_in,P_in*(1.+self.epsilon/C.sigma8)**2)
            self.pow_interp = RectBivariateSpline(k_in,zs,dp_ddelta(k_in,P_in,zs,C=self.C,pmodel='halofit',epsilon=self.epsilon,halo_a=self.halo_a,halo_b=self.halo_b)[0],kx=1,ky=1)
        elif pmodel=='linear':
            self.pow_interp = RectBivariateSpline(self.k_in,self.zs,np.outer(P_in,self.Gs**2),kx=1,ky=1)
        elif pmodel=='dc_dparameter':
            self.param_vary = param_vary
            self.pow_interp = RectBivariateSpline(self.k_in,self.zs,dp_dparameter(self.k_in,P_in,self.zs,C=self.C,parameter=self.param_vary,pmodel='halofit',epsilon=self.epsilon)[0],kx=1,ky=1)
        elif pmodel=='dc_linear':
            self.pow_interp = RectBivariateSpline(k_in,zs,dp_ddelta(k_in,P_in,zs,C=self.C,pmodel='linear')[0],kx=1,ky=1) 
        else:
            raise ValueError('unrecognized option \''+str(pmodel)+'\'')

        if pmodel!='cosmosis':
            self.chis = self.C.D_c_array(zs)
            self.chi_As = self.C.D_c_A_array(zs)

        self.Gs = self.C.G_norm(zs)
        #TODO eliminate self.omegas, it is constant.
        self.omegas = self.C.Omegam

	self.k_use = np.outer((self.ls+0.5),1./self.chi_As)
        
        #loop appears necessary due to uneven grid spacing in k_use
        for i in range(0,self.n_z):
            self.p_dd_use[:,i] = self.pow_interp(self.k_use[:,i],zs[i])[:,0]

        self.dchis=self.C.D_comov_dz(zs)

	self.sc_as = 1/(1+self.zs)

        self.p_gg_use = self.p_dd_use #temporary for testing purposes
        self.p_gd_use = self.p_dd_use #temporary for testing purposes

        self.source_dist = get_source_distribution(self.params['smodel'],self.zs,self.chis,self.C,self.params,ps=ps) 
        self.ps = self.source_dist.ps
        self.z_min_dist = params['z_min_dist']
        self.z_max_dist = params['z_max_dist']

    def r_corr(self):
        return self.p_gd_use/np.sqrt(self.p_dd_use*self.p_gg_use)
    
    def get_n_shape(self,class_1,class_2):
        try:
            return self.n_map[class_1][class_2]
        except KeyError:
            return 0.

    
    def cov_g_diag(self,c_ac,c_ad,c_bd,c_bc,n_ac=0,n_ad=0,n_bd=0,n_bc=0): #only return diagonal because all else are zero by kronecker delta
        cov_diag = np.zeros(self.n_l) #assume same size for now
        for i in range(0,self.n_l):
            cov_diag[i] = 4.*np.pi/(self.omega_s*(2.*self.ls[i]+1.)*self.delta_l)*((c_ac[i]+n_ac)*(c_bd[i]+n_bd)+(c_ad[i]+n_ad)*(c_bc[i]+n_bc))
        return cov_diag
    
    
    #take as an array of qs, ns, and rs instead of the matrices themselves
    #ns is [n_ac,n_ad,n_bd,n_bc]
    def cov_g_diag2(self,qs,ns_in=[0,0,0,0],r_ac=np.array([]),r_ad=np.array([]),r_bd=np.array([]),r_bc=np.array([]),delta_ls=None,ls=None):
        cov_diag = np.zeros(self.n_l)
        if delta_ls is None:
            delta_ls = np.zeros(self.ls.size)
            delta_ls[0:(self.ls.size-1)] = np.diff(self.ls) #todo check extrapolation
            delta_ls[-1] = np.exp(2.*np.log(self.ls[-1])-np.log(self.ls[-2]))-self.ls[-1]
        if ls is None:
            ls = self.ls
        ns = np.zeros(ns_in.size)
        #TODO handle varying bin sizes better
        ns[0] = ns_in[0]/np.trapz((1.*(self.chis>=qs[0].chi_min)*(self.chis<=qs[0].chi_max)*self.ps),self.chis)
        ns[1] = ns_in[1]/np.trapz((1.*(self.chis>=qs[1].chi_min)*(self.chis<=qs[1].chi_max)*self.ps),self.chis)
        ns[2] = ns_in[2]/np.trapz((1.*(self.chis>=qs[2].chi_min)*(self.chis<=qs[2].chi_max)*self.ps),self.chis)
        ns[3] = ns_in[3]/np.trapz((1.*(self.chis>=qs[3].chi_min)*(self.chis<=qs[3].chi_max)*self.ps),self.chis)
        c_ac = Cll_q_q(self,qs[0],qs[2],r_ac).Cll()#*np.sqrt(4.*np.pi)
        c_bd = Cll_q_q(self,qs[1],qs[3],r_bd).Cll()#*np.sqrt(4.*np.pi) 
        c_ad = Cll_q_q(self,qs[0],qs[3],r_ad).Cll()#*np.sqrt(4.*np.pi) 
        c_bc = Cll_q_q(self,qs[1],qs[2],r_bc).Cll()#*np.sqrt(4.*np.pi) 
        #TODO put back delta l, removed the delta l because it should be one here
        for i in range(0,self.n_l): 
            cov_diag[i] = 1./(self.omega_s*(2.*self.ls[i]+1.)*delta_ls[i])*((c_ac[i]+ns[0])*(c_bd[i]+ns[2])+(c_ad[i]+ns[1])*(c_bc[i]+ns[3]))
            #(C13*C24+ C13*N24+N13*C24 + C14*C23+C14*N23+N14*C23+N13*N24+N14*N23)
            #cov_diag[i] = 1./(self.omega_s*(2.*ls[i]+1.)*delta_ls[i])*(c_ac[i]*c_bd[i]+c_ac[i]*ns[2]+ns[0]*c_bd[i]+c_ad[i]*c_bc[i]+ns[3]*c_ad[i]+ns[1]*c_bc[i]+ns[0]*ns[2]+ns[1]*ns[3])
        return cov_diag

    def bin_cov_by_l(self,cov_diag,l_starts): 
        binned_c = np.zeros(l_starts.size) #add fisher matrices then take 1/F_bin
        for i in range(0,l_starts.size):
            if i == l_starts.size-1:
                l_end = l_starts.size
            else:
                l_end = l_starts[i+1]
            fisher_mat=(np.sum(1./cov_diag[l_starts[i]:l_end]))/float(l_end-l_starts[i])
            #fisher matrix sum of covariance eigenvalues divided by delta l, not sure if this is correct
            binned_c[i] = 1./fisher_mat
        return binned_c
    
    #stacked tangential shear, note ls must be successive integers to work correctly
    #note that exact result is necessary if result must be self-consistent (ie tan_shear(theta)=tan_shear(theta+2*pi)) for theta not <<1
    #see Putter & Takada 2010 arxiv:1007.4809
    def tan_shear(self,thetas,with_limber=False):
        n_t = thetas.size
        tans = np.zeros(n_t)
        kg_pow = Cll_k_g(self).Cll()
        for i in range(0,n_t):
            if with_limber:
                tans[i] = np.trapz((2.*self.ls+1.)/(4.*np.pi*self.ls*(self.ls+1.))*kg_pow*sp.lpmv(2,ls,np.cos(thetas[i])),ls)
            else:
                tans[i] = np.trapz(self.ls/(2.*np.pi)*kg_pow*sp.jn(2,thetas[i]*ls),ls)

        return tans

    def cov_mats(self,z_bins,cname1='shear',cname2='shear'):
        covs = np.zeros((z_bins.size-1,z_bins.size-1,self.n_l,self.n_l))
        q1s = np.empty(z_bins.size-1,dtype=object)
        q2s = np.empty(z_bins.size-1,dtype=object)
        for i in range(0,z_bins.size-1):
            chi_min = self.C.D_comov(z_bins[i])
            chi_max = self.C.D_comov(z_bins[i+1])
            if cname1 == 'shear':
                q1s[i] = q_shear(self,chi_min,chi_max)
            elif cname1 == 'num':
                q1s[i] = q_num(self,chi_min,chi_max)
            else:
                warn('invalid value \''+cname1+'\' for cname1 using shear instead')
                q1s[i] = q_shear(self,chi_min,chi_max)

            if cname2 == 'shear':
                q2s[i] = q_shear(self,chi_min,chi_max)
            elif cname2 == 'num':
                q2s[i] = q_num(self,chi_min,chi_max)
            else:
                warn('invalid value \''+cname2+'\' for cname2 using shear instead')
                q2s[i] = q_shear(self,chi_min,chi_max)
        n_ss = self.sigma2_e/(2.*self.n_gal)
        for i in range(0,z_bins.size-1):
            for j in range(0,z_bins.size-1):
                covs[i,j] = np.diagflat(self.cov_g_diag2([q1s[i],q2s[i],q1s[j],q2s[j]],[n_ss,n_ss,n_ss,n_ss]))
        return covs
    
    def dc_ddelta_alpha(self,basis,z_bins,theta,phi,cname1='shear',cname2='shear'):
        if not re.match('^dc',self.pmodel):
            warn('pmodel: '+self.pmodel+' not recognized for derivative, dc_ddelta_alpha may give unexpected results')
        dcs = np.zeros(self.n_l)
        for i in range(0,z_bins.size-1):
            chi_min = self.C.D_comov(z_bins[i])
            chi_max = self.C.D_comov(z_bins[i+1])
            dcs[i] = basis.D_delta_bar_D_delta_alpha(chi_min,chi_max,theta,phi)
            if cname1 == 'shear' and cname2 == 'shear':
                dcs[i] *= self.Cll_sh_sh(chi_max,chi_max,chi_min,chi_min).Cll()
            elif cname1 == 'shear' and cname2 == 'num':
                dcs[i] *= self.Cll_sh_g(chi_max,chi_max,chi_min,chi_min).Cll()
            else:
                warn('invalid cname1 \''+cname1+'\' or cname2 \''+cname2+'\' using shear shear')
                cname1 = 'shear'
                cname2 = 'shear'
                dcs[i] *= self.Cll_sh_sh(chi_max,chi_max,chi_min,chi_min).Cll()
        return dcs


class Cll_q_q:
    def __init__(self,sp,q1s,q2s,corr_param=np.array([])):
        self.integrand = np.zeros((sp.n_z,sp.n_l))
        self.chis = sp.chis
        if corr_param.size !=0: 
           for i in range(0,sp.n_z):
                self.integrand[i] = q1s.qs[i]*q2s.qs[i]/sp.chi_As[i]**2*sp.p_dd_use[:,i]*corr_param[:,i]
        else:
            for i in range(0,sp.n_z):
                self.integrand[i] = q1s.qs[i]*q2s.qs[i]/sp.chi_As[i]**2*sp.p_dd_use[:,i]
    def Cll(self,chi_min=0,chi_max=np.inf):
        high_mask = (self.chis<=chi_max)*1.
        low_mask = (self.chis>=chi_min)*1.
        return np.trapz((high_mask*low_mask*self.integrand.T).T,self.chis,axis=0)

    def Cll_integrand(self):
        return self.integrand

class Cll_sh_sh(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        Cll_q_q.__init__(self,sp,q_shear(sp,chi_max=chi_max1,chi_min=chi_min1),q_shear(sp,chi_max=chi_max2,chi_min=chi_min2))

class Cll_g_g(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        Cll_q_q.__init__(self,sp,q_num(sp,chi_max=chi_max1,chi_min=chi_min1),q_num(sp,chi_max=chi_max2,chi_min=chi_min2))

class Cll_mag_mag(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        Cll_q_q.__init__(self,sp,q_mag(sp,chi_max=chi_max1,chi_min=chi_min1),q_mag(sp,chi_max=chi_max2,chi_min=chi_min2))

class Cll_k_k(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        Cll_q_q.__init__(self,sp,q_k(sp,chi_max=chi_max1,chi_min=chi_min1),q_k(sp,chi_max=chi_max2,chi_min=chi_min2))

class Cll_k_g(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        Cll_q_q.__init__(self,sp,q_k(sp,chi_max=chi_max1,chi_min=chi_min1),q_num(sp,chi_max=chi_max2,chi_min=chi_min2))

class Cll_sh_mag(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        Cll_q_q.__init__(self,sp,q_shear(sp,chi_max=chi_max1,chi_min=chi_min1),q_mag(sp,chi_max=chi_max2,chi_min=chi_min2))

class Cll_mag_g(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        Cll_q_q.__init__(self,sp,q_mag(sp,chi_max=chi_max1,chi_min=chi_min1),q_num(sp,chi_max=chi_max2,chi_min=chi_min2))

class Cll_sh_g(Cll_q_q):
    def __init__(self,sp,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        Cll_q_q.__init__(self,sp,q_shear(sp,chi_max=chi_max1,chi_min=chi_min1),q_num(sp,chi_max=chi_max2,chi_min=chi_min2),corr_param=sp.r_corr())
class Cll_q_q_nolimber(Cll_q_q):
    def __init__(self,sp,q1s,q2s,corr_param=np.array([])):
        integrand1 = np.zeros((sp.n_z,sp.n_l))
        #integrand2 = np.zeros((sp.n_z,sp.n_l))
        integrand_total = np.zeros((sp.n_z,sp.n_l))
        for i in range(0,sp.n_z):
            window_int1 = np.zeros((sp.n_z,sp.n_l))
        #    window_int2 = np.zeros((sp.n_z,sp.n_l))
            for j in range(0,sp.n_z):
                window_int1[j] = q1s.qs[j]/np.sqrt(sp.chis[j])*sp.jv(sp.ls+0.5,(sp.ls+0.5)/sp.chis[i]*sp.chis[j])
         #       window_int2[j] = q2s.qs[j]/np.sqrt(sp.chis[j])*sp.jv(sp.ls+0.5,(sp.ls+0.5)/sp.chis[i]*sp.chis[j])
            integrand1[i] = np.trapz(window_int1,sp.chis,axis=0)
         #   integrand2[i] = np.trapz(window_int2,sp.chis,axis=0)
            integrand_total[i] = integrand1[i]*integrand1[i]*sp.p_dd_use[:,i]*(sp.ls+0.5)**2/sp.chis[i]**3
        self.integrand = integrand_total
        self.chis = sp.chis

#TODO make this work with current q design
class Cll_q_q_order2(Cll_q_q):
    def __init__(self,sp,q1s,q2s,corr_param=np.array([])):
        integrand1 = np.zeros((sp.n_z,sp.n_l))
        integrand2 = np.zeros((sp.n_z,sp.n_l))
        if corr_param.size !=0: 
            for i in range(0,sp.n_z):
                integrand1[i] = q1s.qs[i]*q2s.qs[i]/sp.chis[i]**2*sp.p_dd_use[:,i]*corr_param[:,i]
        else:
            for i in range(0,sp.n_z):
                integrand1[i] = q1s.qs[i]*q2s.qs[i]/sp.chis[i]**2*sp.p_dd_use[:,i]
        for i in range(0,sp.n_z-1): #check edge case
            term1 = sp.chis[i]**2/2.*(q1s.rs_d2[i]/q1s.rs[i]+q2s.rs_d2[i]/q2s.rs[i])
            term2 = sp.chis[i]**3/6.*(q1s.rs_d3[i]/q1s.rs[i]+q2s.rs_d3[i]/q2s.rs[i])
            integrand2[i] = -1./(sp.ls+0.5)**2*(term1+term2)*integrand1[i]
        self.integrand=integrand1+integrand2
        self.chis=sp.chis

def dc_ddelta(zs,zmin,zmax,ls,C,cosmology=defaults.cosmology,epsilon=0.0001,model='halofit_var_redshift'):
    cosmo_a = cosmology.copy()
    k_a,P_a = cpow.camb_pow(cosmo_a)
    #P_a = hf.halofitPk(k_a,C=cp.CosmoPie(cosmo_a)).D2_L(k_a,0.)
    P_select = np.zeros(zs.size,dtype=np.int)
    sp1 = shear_power(k_a,C,zs,ls,pmodel=model,P_in=np.array([P_a]),P_select=P_select,Cs=[cp.CosmoPie(cosmo_a)])
    sh_pow1 = Cll_sh_sh(sp1,chi_min1=C.D_comov(zmin),chi_max1=C.D_comov(zmax),chi_min2=C.D_comov(zmin),chi_max2=C.D_comov(zmax)).Cll()
    domegam = dc_dtheta(zs,zmin,zmax,ls,'Omegam',cosmology,C,epsilon,P_a=P_a,k_a=k_a,model=model,sh_pow1=sh_pow1)
    domegal = dc_dtheta(zs,zmin,zmax,ls,'OmegaL',cosmology,C,epsilon,P_a=P_a,k_a=k_a,model=model,sh_pow1=sh_pow1)
    domegak = dc_dtheta(zs,zmin,zmax,ls,'Omegak',cosmology,C,epsilon,P_a=P_a,k_a=k_a,model=model,sh_pow1=sh_pow1)
    dh = dc_dtheta(zs,zmin,zmax,ls,'h',cosmology,C,epsilon,P_a=P_a,k_a=k_a,model=model,sh_pow1=sh_pow1)
    f0 = -1.-C.Omegam/2.+C.OmegaL+5./(2.*C.G(0))*C.Omegam #see arXiv:1106.5507v2 page 32, eq 106-107
    #https://arxiv.org/pdf/astro-ph/0006089v3.pdf eq 4
    return C.Omegam*(C.Omegam+2./3.*f0)*domegam-2./3.*f0*domegak+(1-C.Omegam)*(C.Omegam+2./3.*f0)*domegal-C.h*(C.Omegam/2.+f0/3.)*dh

def dc_dtheta(zs,zmin,zmax,ls,C,theta_name='Omegach2',cosmology=defaults.cosmology,epsilon=0.0001,P_a=np.array([]),k_a=np.array([]),model='halofit_var_redshift',sh_pow1=None):
    cosmo_a = cosmology.copy()
    cosmo_b = cosmology.copy()
    cosmo_b[theta_name]+= epsilon
    #needed for consistency with different ways of writing the same thing. Assume matter is cdm
    if theta_name=='Omegach2'or theta_name=='Omegabh2':
        cosmo_b['Omegamh2']+= epsilon
        cosmo_b['Omegam']+= epsilon/cosmology['h']**2
    elif theta_name =='Omegamh2':
        cosmo_b['Omegam']+= epsilon/cosmology['h']**2
        cosmo_b['Omegach2']+= epsilon
    elif theta_name == 'Omegam':
        cosmo_b['Omegamh2']+=epsilon*cosmology['h']**2
        cosmo_b['Omegach2']+=epsilon*cosmology['h']**2
    elif theta_name == 'H0':
        cosmo_b['h']+=epsilon/100.
    elif theta_name == 'h':
        cosmo_b['H0']+=epsilon*100.

    Cs = [cp.CosmoPie(cosmology=cosmo_a),cp.CosmoPie(cosmology=cosmo_b)]
    if k_a.size==0:
        k_a,P_a = cpow.camb_pow(cosmo_a)
    #P_b = hf.halofitPk(k_a,C=Cs[1]).D2_L(k_a,0) 
    k_b,P_b = cpow.camb_pow(cosmo_b)

    P_select = np.zeros(zs.size,dtype=np.int)
    if sh_pow1 is None:
        sh_pow1 = Cll_sh_sh(self,chi_min1=C.D_comov(zmin),chi_max1=C.D_comov(zmax),chi_min2=C.D_comov(zmin),chi_max2=C.D_comov(zmax)).Cll()

    P_select[(zs>=zmin)&(zs<=zmax)] = 1

    sp2 = shear_power(k_a,C,zs,ls,pmodel='halofit_var_redshift',P_in=np.array([P_a,P_b]),P_select=P_select,Cs=Cs)

    chi_max = C.D_comov(zmax)
    chi_min = C.D_comov(zmin)
    return (Cll_sh_sh(sp2,chi_max,chi_max,chi_min,chi_min)-sh_pow1).Cll()/epsilon

if __name__=='__main__':
    
	C=cp.CosmoPie(cosmology=defaults.cosmology)
        #d_t = np.loadtxt('chomp_pow_nlin.dat')
#	d = np.loadtxt('Pk_Planck15.dat')
	#d = np.loadtxt('camb_m_pow_l.dat')
#	xiaod = np.loadtxt('power_spectrum_1.dat')
#	intd = pickle.load(open('../CosmicShearPython/Cll_camb.pkl',mode='r'))
        #chompd = np.loadtxt('sh_sh_comp.dat')
#        revd = np.loadtxt('len_pow1.dat')
#        k_in = chompd[:,0]
        #k_in = np.logspace(-5,2,200,base=10)
        #k_in = d[:,0]
    #    k_in = np.loadtxt('test_inputs/proj_1/k_h.txt')
#        k_in = np.logspace(-4,5,5000,base=10)
#	zs = np.logspace(-2,np.log10(3),50,base=10)
	zs = np.arange(0.1,1.0,0.1)
       # zs = np.loadtxt('test_inputs/proj_1/z.txt')
       # zs[0] = 10**-3
        #zs = np.arange(0.005,2,0.0005)
        #ls = np.unique(np.logspace(0,4,3000,dtype=int))*1.0	
        ls = np.arange(1,5000)
        epsilon = 0.0001
        #ls = np.loadtxt('test_inputs/proj_1/ell.txt')

        #ls = np.logspace(np.log10(2),np.log10(10000),3000,base=10)
	#ls = chompd[:,0]	
#	ls = revd[:,0]	
	#t1 = time()
        #cosmo_a = defaults.cosmology.copy()
        #k_a = d[:,0]
        #P_a = d[:,1]
        k_a,P_a = cpow.camb_pow(C.cosmology)
       # P_a = hf.halofitPk(k_a,C=C).D2_L(k_a,0)
        omega_s=np.pi/(3.*np.sqrt(2.))
      
        
        import matplotlib.pyplot as plt
        do_power_response1 = True
        do_power_response2 = False

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
            sp_dch2 = shear_power(k_a,C,zs,ls,omega_s,pmodel='dc_dparameter',P_in=P_a,param_vary='Omegamh2') 
            chi_min = C.D_comov(0.)
            chi_max = C.D_comov(0.4)
            dc_ss_dch2 = Cll_sh_sh(sp_dch2,chi_max,chi_max,chi_min,chi_min).Cll()
            ax.loglog(ls,np.abs(dc_ss_dch2))
            plt.show()
