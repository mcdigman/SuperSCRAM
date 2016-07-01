import numpy as np
import cosmopie as cp
import halofit as hf
from numpy import exp
#from scipy.integrate import quad
from scipy.interpolate import interp1d,interp2d
from scipy.interpolate import InterpolatedUnivariateSpline,SmoothBivariateSpline
import scipy.special as sp
from time import time
import pickle 
#import ps2
import FASTPTcode.FASTPT as FASTPT
import defaults
import camb_power as cpow
import FASTPTcode.matter_power_spt as mps
class shear_power:
    def __init__(self,k_in,C,zs,ls,pmodel='halofit_linear',P_in=np.array([]),cosmology_in={},ps=np.array([]),P_select=np.array([]),Cs=[]):
	self.k_in = k_in
        self.C = C
        self.zs = zs
        self.ls = ls
        self.epsilon = 0.0001
        self.delta_l = ls[-1]-ls[0] #maybe not right
        self.omega_s = 5000 #filler, from eifler, in deg^2
        self.n_gal = 26000000. #filler, from krause & eifler in galaxies/arcmin^2

        self.sigma2_e = 0.32 #from eifler
        self.sigma2_mu = 1.2 #eifler says this is uncertain
        
        self.n_k = self.k_in.size
        self.n_l = self.ls.size
        self.n_z = self.zs.size
        
        self.chis = np.zeros(self.n_z)
        self.chi_As = np.zeros(self.n_z)
        self.omegas = np.zeros(self.n_z)

        self.k_use=np.zeros((self.n_l,self.n_z))
        self.p_dd_use=np.zeros((self.n_l,self.n_z))
	self.sc_as = np.zeros(self.n_z)
	self.dchis = np.zeros(self.n_z)
	self.ps = np.zeros(self.n_z)

        self.cosmology = cosmology_in
            
        #some methods require special handling    
        if pmodel=='halofit_redshift_nonlinear':
            p_bar = hf.halofitPk(0.42891513142857135,self.C).D2_NL(self.k_in)
        elif pmodel == 'cosmosis_nonlinear':
            z_bar = np.loadtxt('test_inputs/proj_1/z.txt')
            k_bar = np.loadtxt('test_inputs/proj_1/k_h.txt')
            p_bar = interp2d(k_bar,z_bar,np.loadtxt('test_inputs/proj_1/p_k.txt'))
            self.chis = interp1d(z_bar,np.loadtxt('test_inputs/proj_1/d_m.txt')[::-1])(zs)
            self.chi_As = self.chis
        elif pmodel=='halofit_nonlinear' or pmodel=='halofit_linear':
            self.halo = hf.halofitPk(k_in,P_in,self.C)
        elif pmodel=='halofit_var_redshift':
            self.halos = []
            for i in range(0,(P_in.shape)[0]):
                self.halos.append(hf.halofitPk(k_in,P_in[i],C=Cs[i]))
        elif pmodel=='halofit_var_redshift_lin':
            self.halos = []
            for i in range(0,(P_in.shape)[0]):
                self.halos.append(hf.halofitPk(k_in,P_in[i],C=Cs[i]))

        elif pmodel=='fastpt_nonlin' or pmodel =='dc_fastpt':
            fpt = FASTPT.FASTPT(k_in,-2,low_extrap=-5,high_extrap=5,n_pad=800)
        #elif pmodel=='deltabar_fpt':
        #    fpt = FASTPT.FASTPT(k_in,-2,low_extrap=-5,high_extrap=5,n_pad=800)
        #    self.halo = hf.halofitPk(k_in,P_in,self.C)
        elif pmodel=='dc_halofit':
            self.halo_a = hf.halofitPk(k_in,P_in,C=C)
            self.halo_b = hf.halofitPk(k_in,P_in*(1.+epsilon/C.sigma8)**2,C=C)
        elif pmodel=='dc_fastpt':
            fpt = FASTPT.FASTPT(k_a,-2,low_extrap=-10,high_extrap=8,n_pad=800)

        for i in range(0,self.n_z):
            if pmodel!='cosmosis_nonlinear':
                self.chis[i] = self.C.D_comov(zs[i])
                self.chi_As[i] = self.C.D_comov_A(zs[i])
            self.omegas[i] = self.C.Omegam
       #     self.omegas[i] = self.C.Omegam_z(zs[i])

	    self.k_use[:,i] = (self.ls+0.5)/self.chi_As[i] 
            if pmodel=='halofit_linear':
                self.p_dd_use[:,i] = 2*np.pi**2*interp1d(self.k_in,self.halo.D2_L(self.k_in,self.zs[i]))(self.k_use[:,i])/self.k_use[:,i]**3
            elif pmodel=='halofit_nonlinear':
                self.p_dd_use[:,i] = 2*np.pi**2*interp1d(self.k_in,self.halo.D2_NL(self.k_in,self.zs[i]))(self.k_use[:,i])/self.k_use[:,i]**3
            elif pmodel=='halofit_redshift_nonlinear':
                self.p_dd_use[:,i] = 2*np.pi**2*interp1d(self.k_in,p_bar*self.C.G_norm(self.zs[i])**2/(self.C.G_norm(0.42891513142857135)**2))(self.k_use[:,i])/self.k_use[:,i]**3
            elif pmodel=='redshift_linear':
                self.p_dd_use[:,i] = interp1d(self.k_in,P_in*self.C.G_norm(self.zs[i])**2)(self.k_use[:,i])
            elif pmodel=='cosmosis_nonlinear':
                self.p_dd_use[:,i] = p_bar(self.k_use[:,i],zs[i])
            elif pmodel=='fastpt_nonlin':
                self.p_dd_use[:,i] = interp1d(self.k_in,(fpt.one_loop(P_in*self.C.G_norm(self.zs[i])**2,C_window=0.75)+P_in*self.C.G_norm(self.zs[i])**2))(self.k_use[:,i])
            elif pmodel=='halofit_var_redshift':
                self.p_dd_use[:,i] = 2*np.pi**2*interp1d(self.k_in,self.halos[P_select[i]].D2_NL(self.k_in,self.zs[i]))(self.k_use[:,i])/self.k_use[:,i]**3
            elif pmodel=='halofit_var_redshift_lin':
                self.p_dd_use[:,i] = 2*np.pi**2*interp1d(self.k_in,self.halos[P_select[i]].D2_L(self.k_in,self.zs[i]))(self.k_use[:,i])/self.k_use[:,i]**3
            elif pmodel=='dc_halofit':
                self.p_dd_use[:,i] = interp1d(k_in,dp_ddelta(k_in,P_in,zs[i],pmodel='halofit',epsilon=epsilon,halo_a=self.halo_a,halo_b=self.halo_b)[0])(self.k_use[:,i])
            elif pmodel=='dc_linear':
                self.p_dd_use[:,i] = interp1d(k_in,dp_ddelta(k_in,P_in,zs[i],pmodel='linear')[0])(self.k_use[:,i])
            elif pmodel=='dc_fastpt':
                self.p_dd_use[:,i] = interp1d(k_in,dp_ddelta(k_in,P_in,zs[i],pmodel='fastpt',fpt=fpt)[0])(self.k_use[:,i])

         #   elif pmodel=='deltabar':
         #       lin_pow= 2*np.pi**2*interp1d(self.k_in,self.halo.D2_L(self.k_in,self.zs[i]))(self.k_use[:,i])/self.k_use[:,i]**3
         #       dpdk =(InterpolatedUnivariateSpline(self.k_use[:,i],lin_pow,ext=2).derivative(1))(self.k_use[:,i]) 
         #       self.p_dd_use[:,i] = 47./21.*lin_pow-1./3.*self.k_use[:,i]*dpdk
         #   elif pmodel=='deltabar_hnl':
         #       lin_pow= 2*np.pi**2*interp1d(self.k_in,self.halo.D2_NL(self.k_in,self.zs[i]))(self.k_use[:,i])/self.k_use[:,i]**3
         #       dpdk =(InterpolatedUnivariateSpline(self.k_use[:,i],lin_pow,ext=2).derivative(1))(self.k_use[:,i]) 
         #       self.p_dd_use[:,i] = lin_pow-1./3.*self.k_use[:,i]*dpdk
         #   elif pmodel=='deltabar_fpt':
         #       lin_pow= 2*np.pi**2*interp1d(self.k_in,self.halo.D2_L(self.k_in,self.zs[i]))(self.k_use[:,i])/self.k_use[:,i]**3
         #       dpdk =(InterpolatedUnivariateSpline(self.k_use[:,i],lin_pow,ext=2).derivative(1))(self.k_use[:,i]) 
         #       self.p_dd_use[:,i] = 47./21.*lin_pow-1./3.*self.k_use[:,i]*dpdk
         #       self.p_dd_use[:,i] += 26./21.*interp1d(self.k_in,(fpt.one_loop_sep(P_in*self.C.G_norm(self.zs[i])**2,C_window=0.75)))(self.k_use[:,i])
            #elif pmodel=='deltabar_nl':
                #lin_pow= 2*np.pi**2*interp1d(self.k_in,self.halo.D2_NL(self.k_in,self.zs[i]))(self.k_use[:,i])/self.k_use[:,i]**3
                #dpdk =(InterpolatedUnivariateSpline(self.k_use[:,i],self.k_use[:,i]**3*lin_pow,ext=2).derivative(1))(self.k_use[:,i]) 
                #self.p_dd_use[:,i] = 13./21.*lin_pow-1./3.*self.k_use[:,i]*dpdk

            else:
                if i==0:
                    print("invalid pmodel value \'"+pmodel+"\' using halofit_linear instead")
                self.p_dd_use[:,i] = 2*np.pi**2*interp1d(self.k_in,hf.halofitPk(self.zs[i],self.cosmology).D2_L(self.k_in))(self.k_use[:,i])/self.k_use[:,i]**3


	    self.dchis[i] = (self.C.D_comov(self.zs[i]+self.epsilon)-self.C.D_comov(self.zs[i]))/self.epsilon
	    self.sc_as[i] = 1/(1+self.zs[i])

        self.p_gg_use = self.p_dd_use #temporary for testing purposes
        self.p_gd_use = self.p_dd_use #temporary for testing purposes
        if ps.size==0:
         #   self.ps[-3] = 1./(self.chis[-2]-self.chis[-3])  #make option, this makes it act like a proper delta function
            self.ps = np.exp(-(self.zs-1.)**2/(2.*(0.4)**2))
       #     self.ps[-1] = 1./(self.C.D_comov(2*zs[-1]-zs[-2])-self.chis[-1])  #make option, this makes it act like a proper delta function
        #self.ps[30] = 1./(self.chis[31]-self.chis[30])  #make option, this makes it act like a proper delta function
        for i in range(0,self.n_z-1): #compensate for different bin sizes
           self.ps[i] = self.ps[i]/(self.chis[i+1]-self.chis[i])
        self.ps[-1] = self.ps[-1]/(self.C.D_comov(2*zs[-1]-zs[-2])-self.chis[-1]) #patch last value so it isn't just 0
       # for i in range(0,self.n_z-1): #assume constant distribution of galaxies
      #     self.ps[i] = 1/(self.chis[i+1]-self.chis[i])
     #   self.ps[-1] = 1/(self.C.D_comov(2*zs[-1]-zs[-2])-self.chis[-1]) #patch last value so it isn't just 0
       # for i in range(0,self.n_z-1): # a model from the cosmolike paper
      #      self.ps[i] = zs[i]**1.24*np.exp(-(zs[i]/0.51)**1.01)/(self.chis[i+1]-self.chis[i])
        self.ps = self.ps/np.trapz(self.ps,self.chis) #normalize galaxy probability distribution

    def r_corr(self):
        return self.p_gd_use/np.sqrt(self.p_dd_use*self.p_gg_use)
    

    def Cll_sh_sh(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        return self.Cll_q_q(q_shear(self,chi_max=chi_max1,chi_min=chi_min1),q_shear(self,chi_max=chi_max2,chi_min=chi_min2))

    def Cll_g_g(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        return self.Cll_q_q(q_num(self,chi_max=chi_max1,chi_min=chi_min1),q_num(self,chi_max=chi_max2,chi_min=chi_min2))
    
    def Cll_mag_mag(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        return self.Cll_q_q(q_mag(self,chi_max=chi_max1,chi_min=chi_min1),q_mag(self,chi_max=chi_max2,chi_min=chi_min2))
    
    def Cll_k_k(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        return self.Cll_q_q(q_k(self,chi_max=chi_max1,chi_min=chi_min1),q_k(self,chi_max=chi_max2,chi_min=chi_min2))

    def Cll_k_g(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        return self.Cll_q_q(q_k(self,chi_max=chi_max1,chi_min=chi_min1),q_num(self,chi_max=chi_max2,chi_min=chi_min2))

    def Cll_sh_mag(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        return self.Cll_q_q(q_shear(self,chi_max=chi_max1,chi_min=chi_min1),q_mag(self,chi_max=chi_max2,chi_min=chi_min2))

    def Cll_mag_g(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        return self.Cll_q_q(q_mag(self,chi_max=chi_max1,chi_min=chi_min1),q_num(self,chi_max=chi_max2,chi_min=chi_min2))

    def Cll_sh_g(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        return self.Cll_q_q(q_shear(self,chi_max=chi_max1,chi_min=chi_min2),q_num(self,chi_max=chi_max2,chi_min=chi_min2),corr_param=self.r_corr())


    def Cll_q_q(self,q1s,q2s,corr_param=np.array([])):
        integrand = np.zeros((self.n_z,self.n_l))
        if corr_param.size !=0: 
            for i in range(0,self.n_z):
                integrand[i] = q1s.qs[i]*q2s.qs[i]/self.chi_As[i]**2*self.p_dd_use[:,i]*corr_param[:,i]
        else:
            for i in range(0,self.n_z):
                integrand[i] = q1s.qs[i]*q2s.qs[i]/self.chi_As[i]**2*self.p_dd_use[:,i]
        return np.trapz(integrand,self.chis,axis=0)

    #very slow, requires extremely fine z grid to work 
    def Cll_q_q_nolimber(self,q1s,q2s,corr_param=np.array([])):
        integrand1 = np.zeros((self.n_z,self.n_l))
        #integrand2 = np.zeros((self.n_z,self.n_l))
        integrand_total = np.zeros((self.n_z,self.n_l))
        for i in range(0,self.n_z):
            window_int1 = np.zeros((self.n_z,self.n_l))
        #    window_int2 = np.zeros((self.n_z,self.n_l))
            for j in range(0,self.n_z):
                window_int1[j] = q1s.qs[j]/np.sqrt(self.chis[j])*sp.jv(self.ls+0.5,(self.ls+0.5)/self.chis[i]*self.chis[j])
         #       window_int2[j] = q2s.qs[j]/np.sqrt(self.chis[j])*sp.jv(self.ls+0.5,(self.ls+0.5)/self.chis[i]*self.chis[j])
            integrand1[i] = np.trapz(window_int1,self.chis,axis=0)
         #   integrand2[i] = np.trapz(window_int2,self.chis,axis=0)
            integrand_total[i] = integrand1[i]*integrand1[i]*self.p_dd_use[:,i]*(self.ls+0.5)**2/self.chis[i]**3
        return np.trapz(integrand_total,self.chis,axis=0)


    def Cll_q_q_order2(self,q1s,q2s,corr_param=np.array([])):
        integrand1 = np.zeros((self.n_z,self.n_l))
        integrand2 = np.zeros((self.n_z,self.n_l))
        if corr_param.size !=0: 
            for i in range(0,self.n_z):
                integrand1[i] = q1s.qs[i]*q2s.qs[i]/self.chis[i]**2*self.p_dd_use[:,i]*corr_param[:,i]
        else:
            for i in range(0,self.n_z):
                integrand1[i] = q1s.qs[i]*q2s.qs[i]/self.chis[i]**2*self.p_dd_use[:,i]
        for i in range(0,self.n_z-1): #check edge case
            term1 = self.chis[i]**2/2.*(q1s.rs_d2[i]/q1s.rs[i]+q2s.rs_d2[i]/q2s.rs[i])
            term2 = self.chis[i]**3/6.*(q1s.rs_d3[i]/q1s.rs[i]+q2s.rs_d3[i]/q2s.rs[i])
            integrand2[i] = -1./(self.ls+0.5)**2*(term1+term2)*integrand1[i]
        return np.trapz(integrand1+integrand2,self.chis,axis=0)
    
    def cov_g_diag(self,c_ac,c_ad,c_bd,c_bc,n_ac=0,n_ad=0,n_bd=0,n_bc=0): #only return diagonal because all else are zero by kronecker delta
        cov_diag = np.zeros(self.n_l) #assume same size for now
        for i in range(0,self.n_l):
            cov_diag[i] = 2*np.pi/(self.omega_s*self.ls[i]*self.delta_l)*((c_ac[i]+n_ac)*(c_bd[i]+n_bd)+(c_ad[i]+n_ad)*(c_bc[i]+n_bc))
        return cov_diag

    #take as an array of qs, ns, and rs instead of the matrices themselves
    #ns is [n_ac,n_ad,n_bd,n_bc]
    def cov_g_diag2(self,qs,ns,r_ac=np.array([]),r_ad=np.array([]),r_bd=np.array([]),r_bc=np.array([])):
        cov_diag = np.zeros(self.n_l)
        c_ac = self.Cll_q_q(qs[0],qs[2],r_ac)
        c_bd = self.Cll_q_q(qs[1],qs[3],r_bd)
        c_ad = self.Cll_q_q(qs[0],qs[3],r_ad)
        c_bc = self.Cll_q_q(qs[1],qs[2],r_bc)
        
        #removed the delta l because it should be one here
        for i in range(0,self.n_l): 
            cov_diag[i] = 2*np.pi/(self.omega_s*self.ls[i])*((c_ac[i]+ns[0])*(c_bd[i]+ns[2])+(c_ad[i]+ns[1])*(c_bc[i]+ns[3]))
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
        kg_pow = self.Cll_k_g()
        for i in range(0,n_t):
            if with_limber:
                tans[i] = np.trapz((2.*self.ls+1.)/(4.*np.pi*self.ls*(self.ls+1.))*kg_pow*sp.lpmv(2,ls,np.cos(thetas[i])),ls)
            else:
                tans[i] = np.trapz(self.ls/(2.*np.pi)*kg_pow*sp.jn(2,thetas[i]*ls),ls)

        return tans
class q_weight:
    def __init__(self,chis,qs,chi_min=0.,chi_max=np.inf):
        self.chi_min = chi_min
        self.chi_max = chi_max
        self.qs = qs
       # self.q_spline = InterpolatedUnivariateSpline(chis,self.qs,ext=2)
       # self.r_spline = InterpolatedUnivariateSpline(chis,self.qs/np.sqrt(chis),ext=2) #for calculating to second order
class q_shear(q_weight):
    def __init__(self,sp,chi_min=0.,chi_max=np.inf,mult=1.):
        qs = 3./2.*(sp.C.H0/sp.C.c)**2*sp.omegas*sp.chi_As/sp.sc_as*self.gs(sp,chi_max=chi_max,chi_min=chi_min)*mult
       # r_spline = InterpolatedUnivariateSpline(sp.chis,qs/np.sqrt(sp.chis),ext=2) #for calculating to second order
       # self.rs = r_spline(sp.chis)
       # self.rs_d1 = r_spline.derivative(1)(sp.chis)
       # self.rs_d2 = r_spline.derivative(2)(sp.chis)
       # self.rs_d3 = r_spline.derivative(3)(sp.chis)
        q_weight.__init__(self,sp.chis,qs,chi_min=chi_min,chi_max=chi_max) 

    def gs(self,sp,chi_max=np.inf,chi_min=0):
        g_vals = np.zeros(sp.n_z)
        low_mask = (sp.chis>=chi_min)*1. #so only integrate from max(chi,chi_min)
	for i in range(0,sp.n_z):
            if chi_max<sp.chis[i]:
                break
            if sp.C.Omegak==0.0:
                g_vals[i] =np.trapz(low_mask[i:sp.n_z]*sp.ps[i:sp.n_z]*(sp.chis[i:sp.n_z]-sp.chis[i])/sp.chis[i:sp.n_z],sp.chis[i:sp.n_z])
            elif sp.C.Omegak>0.0: #TODO handle curvature
                sqrtK = np.sqrt(sp.C.K)
                g_vals[i] =np.trapz(low_mask[i:sp.n_z]*sp.ps[i:sp.n_z]*sp.chis[i]*1./sqrtK(1./np.tan(sqrtK*sp.chis[i])-1./np.tan(sqrtK*sp.chis[i:sp.n_z])),sp.chis[i:sp.n_z])
            else:
                sqrtK = np.sqrt(abs(sp.C.K))
                g_vals[i] =np.trapz(low_mask[i:sp.n_z]*sp.ps[i:sp.n_z]*sp.chis[i]*1./sqrtK(1./np.tanh(sqrtK*sp.chis[i])-1./np.tanh(sqrtK*sp.chis[i:sp.n_z])),sp.chis[i:sp.n_z])
        return g_vals

class q_mag(q_shear):
    def __init__(self,sp,chi_max=np.inf,chi_min=0.):
        q_shear.__init__(self,sp,chi_max=chi_max,chi_min=chi_min,mult=2.)


class q_k(q_shear):
    def __init__(self,sp,chi_max=np.inf,chi_min=0.):
        q_shear.__init__(self,sp,chi_max=chi_max,chi_min=chi_min,mult=1.)

class q_num(q_weight):
    def __init__(self,sp,chi_max=np.inf,chi_min=0.):
        q = np.zeros((sp.n_z,sp.n_l))
        self.b = self.bias(sp)
        for i in range(0,sp.n_z):
            if sp.chis[i]>chi_max:
                break
            elif sp.chis[i]<chi_min:
                continue
            else:
                q[i] = sp.ps[i]*self.b[:,i]
      #  self.rs = np.zeros((sp.n_z,sp.n_l))

       # self.rs_d1 = np.zeros((sp.n_z,sp.n_l))
       # self.rs_d2 = np.zeros((sp.n_z,sp.n_l))
       # self.rs_d3 = np.zeros((sp.n_z,sp.n_l))
       # for i in range(0,sp.n_l):
       #     r_spline = InterpolatedUnivariateSpline(sp.chis,q[:,i]/np.sqrt(sp.chis))
       #     self.rs[:,i] = r_spline(sp.chis)
       #     self.rs_d1[:,i] = r_spline.derivative(1)(sp.chis)
       #     self.rs_d2[:,i] = r_spline.derivative(2)(sp.chis)
       #     self.rs_d3[:,i] = r_spline.derivative(3)(sp.chis)
        q_weight.__init__(self,sp.chis,q,chi_min=chi_min,chi_max=chi_max)

    def bias(self,sp):
        return np.sqrt(sp.p_gg_use/sp.p_dd_use)

def dc_ddelta(zs,zmin,zmax,ls,cosmology=defaults.cosmology,C=cp.CosmoPie(),epsilon=0.0001,model='halofit_var_redshift'):
    cosmo_a = cosmology.copy()
    k_a,P_a = cpow.camb_pow(cosmo_a)
    #P_a = hf.halofitPk(k_a,C=cp.CosmoPie(cosmo_a)).D2_L(k_a,0.)
    P_select = np.zeros(zs.size,dtype=np.int)
    sp1 = shear_power(k_a,C,zs,ls,pmodel=model,P_in=np.array([P_a]),cosmology_in=cosmology,P_select=P_select,Cs=[cp.CosmoPie(cosmo_a)])
    sh_pow1 = sp1.Cll_sh_sh(chi_min1=C.D_comov(zmin),chi_max1=C.D_comov(zmax),chi_min2=C.D_comov(zmin),chi_max2=C.D_comov(zmax))
    domegam = dc_dtheta(zs,zmin,zmax,ls,'Omegam',cosmology,C,epsilon,P_a=P_a,k_a=k_a,model=model,sh_pow1=sh_pow1)
    domegal = dc_dtheta(zs,zmin,zmax,ls,'OmegaL',cosmology,C,epsilon,P_a=P_a,k_a=k_a,model=model,sh_pow1=sh_pow1)
    domegak = dc_dtheta(zs,zmin,zmax,ls,'Omegak',cosmology,C,epsilon,P_a=P_a,k_a=k_a,model=model,sh_pow1=sh_pow1)
    dh = dc_dtheta(zs,zmin,zmax,ls,'h',cosmology,C,epsilon,P_a=P_a,k_a=k_a,model=model,sh_pow1=sh_pow1)
    f0 = -1.-C.Omegam/2.+C.OmegaL+5./(2.*C.G(0))*C.Omegam #see arXiv:1106.5507v2 page 32, eq 106-107
    #https://arxiv.org/pdf/astro-ph/0006089v3.pdf eq 4
    return C.Omegam*(C.Omegam+2./3.*f0)*domegam-2./3.*f0*domegak+(1-C.Omegam)*(C.Omegam+2./3.*f0)*domegal-C.h*(C.Omegam/2.+f0/3.)*dh

def dc_dtheta(zs,zmin,zmax,ls,theta_name='Omegach2',cosmology=defaults.cosmology,C=cp.CosmoPie(),epsilon=0.0001,P_a=np.array([]),k_a=np.array([]),model='halofit_var_redshift',sh_pow1=None):
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
        sh_pow1 = sp1.Cll_sh_sh(chi_min1=C.D_comov(zmin),chi_max1=C.D_comov(zmax),chi_min2=C.D_comov(zmin),chi_max2=C.D_comov(zmax))

    P_select[(zs>=zmin)&(zs<=zmax)] = 1

    sp2 = shear_power(k_a,C,zs,ls,pmodel='halofit_var_redshift',P_in=np.array([P_a,P_b]),cosmology_in=cosmology,P_select=P_select,Cs=Cs)

    chi_max = C.D_comov(zmax)
    chi_min = C.D_comov(zmin)
    return (sp2.Cll_sh_sh(chi_max,chi_max,chi_min,chi_min)-sh_pow1)/epsilon

def dp_ddelta(k_a,P_a,zbar,C=cp.CosmoPie(),pmodel='linear',epsilon=0.0001,halo_a=None,halo_b=None,fpt=None):
    if pmodel=='linear':
        pza = P_a**C.G_norm(zbar)**2
        dpdk =(InterpolatedUnivariateSpline(k_a,pza,ext=2).derivative(1))(k_a) 
        dp = 47./21.*pza-1./3.*k_a*dpdk
    elif pmodel=='halofit':
        if halo_a is None:
            halo_a = hf.halofitPk(k_a,P_a,C=C)
        if halo_b is None:
            halo_b = hf.halofitPk(k_a,P_a*(1.+epsilon/C.sigma8)**2,C=C)
        pza = halo_a.D2_NL(k_a,zbar)*2.*np.pi**2/k_a**3
        pzb = halo_b.D2_NL(k_a,zbar)*2.*np.pi**2/k_a**3
        dpdk =(InterpolatedUnivariateSpline(k_a,pza,ext=2).derivative(1))(k_a) 
        dp = 13./21.*C.sigma8*(pzb-pza)/epsilon+pza-1./3.*k_a*dpdk
    elif pmodel=='fastpt':
        if fpt is None:
            fpt = FASTPT.FASTPT(k_a,-2,low_extrap=-10,high_extrap=8,n_pad=4000)
        plin = P_a**C.G_norm(zbar)**2
        pza = plin+fpt.one_loop(plin,C_window=0.75)
        dpdk =(InterpolatedUnivariateSpline(k_a,pza,ext=2).derivative(1))(k_a) 
        dp = 47./21.*pza-1./3.*k_a*dpdk+26./21.*one_loop_sep(fpt,plin,C_window=0.75)
    else:
        print('invalid pmodel option \''+str(pmodel)+'\' using linear')
        pza = P_a**C.G_norm(zbar)**2
        dpdk =(InterpolatedUnivariateSpline(k_a,pza,ext=2).derivative(1))(k_a) 
        dp = 47./21.*pza-1./3.*k_a*dpdk
    return dp,pza

def one_loop_sep(fpt,P,C_window=0.75):
    Ps,P22=fpt.P22(P,None,C_window)
    P13=mps.P_13_reg(fpt.k_old,Ps)
    if (fpt.extrap):
        _,P=fpt.EK.PK_orginal(P22+2.*P13) #TODO check if this is right
        return P
    return P22+2.*P13

    #maybe can get bias and r_corr directly from something else
if __name__=='__main__':
    
	C=cp.CosmoPie(cosmology=defaults.cosmology)
        #d_t = np.loadtxt('chomp_pow_nlin.dat')
#	d = np.loadtxt('Pk_Planck15.dat')
	d = np.loadtxt('camb_m_pow_l.dat')
#	xiaod = np.loadtxt('power_spectrum_1.dat')
#	intd = pickle.load(open('../CosmicShearPython/Cll_camb.pkl',mode='r'))
        #chompd = np.loadtxt('sh_sh_comp.dat')
#        revd = np.loadtxt('len_pow1.dat')
#        k_in = chompd[:,0]
        #k_in = np.logspace(-5,2,200,base=10)
        k_in = d[:,0]
    #    k_in = np.loadtxt('test_inputs/proj_1/k_h.txt')
#        k_in = np.logspace(-4,5,5000,base=10)
#	zs = np.logspace(-2,np.log10(3),50,base=10)
	zs = np.arange(0.01,1.0,0.01)
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
	t1 = time()
        cosmo_a = defaults.cosmology.copy()
        #k_a = d[:,0]
        #P_a = d[:,1]
        k_a,P_a = cpow.camb_pow(cosmo_a)
       # P_a = hf.halofitPk(k_a,C=C).D2_L(k_a,0)
        zmin = zs[40]
        zmax = zs[41]
        chimin = C.D_comov(zmin)
        chimax = C.D_comov(zmax)
      #  sp1 = shear_power(k_a,C,zs,ls,pmodel='halofit_linear',P_in=P_a,cosmology_in=defaults.cosmology)
      #  sh_pow1 = sp1.Cll_sh_sh(chi_max,chi_max,chi_min,chi_min)
      #  dpdl =(InterpolatedUnivariateSpline(ls,sh_pow1,ext=2).derivative(1))(ls)
      #  deltabar3 = 47./21.*sh_pow1-1./3.*dpdl*ls
        #for i in range(0,zs.size):
        #    sp.sph_jn(ls,zs[i])
#        sp1 = shear_power(k_in,C,zs,ls,pmodel='redshift_linear',P_in=d[:,1],cosmology_in=defaults.cosmology)
 #       sh_pow1 = sp1.Cll_sh_sh()
      #  sp2 = shear_power(k_in,C,zs,ls,pmodel='halofit_linear',P_in=d[:,1],cosmology_in=defaults.cosmology)
        #sp1 = shear_power(k_in,C,zs,ls,pmodel='halofit_linear',P_in=d[:,1],cosmology_in=defaults.cosmology)
      #  sh_pow2 = sp2.Cll_sh_sh()
        #sp3 = shear_power(k_in,C,zs,ls,pmodel='halofit_nonlinear',P_in=d[:,1],cosmology_in=defaults.cosmology)
        #sh_pow_limber = sp3.Cll_sh_sh()
        #sh_pow_nolimber = sp3.Cll_q_q_nolimber(q_shear(sp3),q_shear(sp3))
        #sh_pow3 = sp3.Cll_sh_sh()
        #cosmo_a = defaults.cosmology.copy()
        #cosmo_b = defaults.cosmology.copy()
        #cosmo_c = defaults.cosmology.copy()
        #epsilon = 0.001
        #sp_alt = shear_power(k_a,C,zs,ls,pmodel='deltabar',P_in=P_a,cosmology_in=defaults.cosmology)
        #dcalt = sp_alt.Cll_sh_sh(chimax,chimax,chimin,chimin)
        #sp3a = shear_power(k_a,C,zs,ls,pmodel='halofit_nonlinear',P_in=P_a,cosmology_in=defaults.cosmology)
        #sp3b = shear_power(k_a,C,zs,ls,pmodel='halofit_nonlinear',P_in=(1.+epsilon/C.sigma8)**2*P_a,cosmology_in=defaults.cosmology)
        #sh_pow3a = sp3a.Cll_sh_sh(chimax,chimax,chimin,chimin)
        #sh_pow3b = sp3b.Cll_sh_sh(chimax,chimax,chimin,chimin)
        #p3a = sp3a.p_dd_use[:,40]
        #p3b = sp3b.p_dd_use[:,40]
        zbar = 0.
        #dcalt1,p1a = dp_ddelta(k_a,P_a,zbar,pmodel='linear')
        #dcalt2,p2a = dp_ddelta(k_a,P_a,zbar,pmodel='halofit')
        #dcalt3,p3a = dp_ddelta(k_a,P_a,zbar,pmodel='fastpt')
#        p1a = P_a**C.G_norm(zbar)**2
        #p1b = hf.halofitPk(k_a,P_a*(1.+epsilon/C.sigma8)**2,C=C).D2_NL(k_a,0.4)
#        dpdk1 =(InterpolatedUnivariateSpline(k_a,p1a,ext=2).derivative(1))(k_a) 
#        dcalt1 = 47./21.*p1a-1./3.*k_a*dpdk1
    
#        p2a = hf.halofitPk(k_a,P_a,C=C).D2_NL(k_a,zbar)*2.*np.pi**2/k_a**3
#        p2b = hf.halofitPk(k_a,P_a*(1.+epsilon/C.sigma8)**2,C=C).D2_NL(k_a,zbar)*2.*np.pi**2/k_a**3
#        dpdk2 =(InterpolatedUnivariateSpline(k_a,p2a,ext=2).derivative(1))(k_a) 
#        dcalt2 = 13./21.*C.sigma8*(p2b-p2a)/epsilon+p2a-1./3.*k_a*dpdk2
        sp_hf = shear_power(k_in,C,zs,ls,pmodel='dc_halofit',P_in=d[:,1],cosmology_in=defaults.cosmology) 
        dc_hf = sp_hf.Cll_sh_sh(chimax,chimax,chimin,chimin)
        sp_fpt = shear_power(k_in,C,zs,ls,pmodel='dc_fastpt',P_in=d[:,1],cosmology_in=defaults.cosmology) 
        dc_fpt = sp_fpt.Cll_sh_sh(chimax,chimax,chimin,chimin)
        sp_lin = shear_power(k_in,C,zs,ls,pmodel='dc_linear',P_in=d[:,1],cosmology_in=defaults.cosmology) 
        dc_lin = sp_lin.Cll_sh_sh(chimax,chimax,chimin,chimin)


#        fpt = FASTPT.FASTPT(k_a,-2,low_extrap=-5,high_extrap=8,n_pad=800)
#        p3a = p1a+fpt.one_loop(p1a,C_window=0.75)
#        dpdk3 =(InterpolatedUnivariateSpline(k_a,p3a,ext=2).derivative(1))(k_a) 
#        dcalt3 = 47./21.*p3a-1./3.*k_a*dpdk3+26./21.*fpt.one_loop_sep(p1a,C_window=0.75)

       # dcalt1 = 47./21.*p1a-1./3.*k_a*dpdk1
        #dcalt2 = sh_pow3a-1./3.*ls*dpdk
        #sp_alt3 = shear_power(k_a,C,zs,ls,pmodel='deltabar_fpt',P_in=P_a,cosmology_in=defaults.cosmology)
        #dcalt3 = sp_alt3.Cll_sh_sh(chimax,chimax,chimin,chimin)
        #sp_alt4 = shear_power(k_a,C,zs,ls,pmodel='deltabar_hnl',P_in=P_a,cosmology_in=defaults.cosmology)
        #dcalt4 = sp_alt4.Cll_sh_sh(chimax,chimax,chimin,chimin)
        #dcnl = dc_ddelta(zs,zmin,zmax,ls,model='halofit_var_redshift')
        #dclin = dc_ddelta(zs,zmin,zmax,ls,model='halofit_var_redshift_lin')
        #cosmo_b['Omegach2']+=epsilon
        #cosmo_c['Omegach2']-=epsilon
        #k_a,P_a = cpow.camb_pow(cosmo_a)
        #k_b,P_b = cpow.camb_pow(cosmo_b)
        #k_c,P_c = cpow.camb_pow(cosmo_c)
        
#        cosmo_b['Omegach2']-=25*epsilon
#        P_select = np.zeros(zs.size,dtype=np.int)
#        P_select[0:10] = 1
#        nbin = 50
#        p10s = np.zeros(nbin)
#        p100s = np.zeros(nbin)
#        p1000s = np.zeros(nbin)
#        omegachs = np.zeros(nbin)
#        for i in range(0,nbin):
#            cosmo_b['Omegach2']+=epsilon
#            omegachs[i] = cosmo_b['Omegach2']
#            k_b,P_b = cpow.camb_pow(cosmo_b)
#            sp8 = shear_power(k_a,C,zs,ls,pmodel='halofit_var_redshift',P_in=np.array([P_a,P_b]),cosmology_in=defaults.cosmology,P_select=P_select)
#            sh_pow8 = sp8.Cll_sh_sh()
#            p10s[i] = sh_pow8[10]
#            p100s[i] = sh_pow8[100]
#            p1000s[i] = sh_pow8[1000]
           # sp8 = shear_power(k_a,C,zs,ls,pmodel='halofit_var_redshift',P_in=np.array([P_a,P_b]),cosmology_in=defaults.cosmology,P_select=P_select)
            #sh_pow8 = sp8.Cll_sh_sh()
#        sp9 = shear_power(k_a,C,zs,ls,pmodel='halofit_var_redshift',P_in=np.array([P_a,P_b]),cosmology_in=defaults.cosmology,P_select=P_select)
#        sh_pow9 = sp9.Cll_sh_sh()
#        sp9c = shear_power(k_a,C,zs,ls,pmodel='halofit_var_redshift',P_in=np.array([P_a,P_c]),cosmology_in=defaults.cosmology,P_select=P_select)
#        sh_pow9c = sp9c.Cll_sh_sh()
#        div_c1 = abs(sh_pow9-sh_pow9c)/(2.*epsilon)
#        P_select[0:10] = 0
#        P_select[94:104] = 1
#        sp10 = shear_power(k_a,C,zs,ls,pmodel='halofit_var_redshift',P_in=np.array([P_a,P_b]),cosmology_in=defaults.cosmology,P_select=P_select)
#        sh_pow10 = sp10.Cll_sh_sh()
#        sp10c = shear_power(k_a,C,zs,ls,pmodel='halofit_var_redshift',P_in=np.array([P_a,P_c]),cosmology_in=defaults.cosmology,P_select=P_select)
#        sh_pow10c = sp10c.Cll_sh_sh()
#        div_c2 = abs(sh_pow10-2.*sh_pow8+sh_pow10c)/(2.*epsilon)
#        P_select[94:104] = 0
#        P_select[188:198] = 1
#        sp11 = shear_power(k_a,C,zs,ls,pmodel='halofit_var_redshift',P_in=np.array([P_a,P_b]),cosmology_in=defaults.cosmology,P_select=P_select)
#        sh_pow11 = sp11.Cll_sh_sh()
#        sp11c = shear_power(k_a,C,zs,ls,pmodel='halofit_var_redshift',P_in=np.array([P_a,P_c]),cosmology_in=defaults.cosmology,P_select=P_select)
#        sh_pow11c = sp11c.Cll_sh_sh()
#        div_c3 = abs(sh_pow11-2.*sh_pow8+sh_pow11c)/(2.*epsilon)
#        P_select[188:198] = 0
#        P_select[45:55] = 1
#        sp12 = shear_power(k_a,C,zs,ls,pmodel='halofit_var_redshift',P_in=np.array([P_a,P_b]),cosmology_in=defaults.cosmology,P_select=P_select)
#        sh_pow12 = sp12.Cll_sh_sh()
#        sp12c = shear_power(k_a,C,zs,ls,pmodel='halofit_var_redshift',P_in=np.array([P_a,P_c]),cosmology_in=defaults.cosmology,P_select=P_select)
#        sh_pow12c = sp12c.Cll_sh_sh()
#        div_c4 = abs(sh_pow12-2.*sh_pow8+sh_pow12c)/(2.*epsilon)
       # sp7 = shear_power(k_in,C,zs,ls,pmodel='cosmosis_nonlinear',P_in=d[:,1],cosmology_in=defaults.cosmology)
      #  sh_pow7 = sp7.Cll_sh_sh()
   #     sh_pow7_gg = sp7.Cll_g_g()
   #     sh_pow7_sg = sp7.Cll_sh_g()
   #     sh_pow7_mm = sp7.Cll_mag_mag()
  #      sp5 = shear_power(k_in,C,zs[0:50],ls,pmodel='halofit_nonlinear',P_in=d[:,1],cosmology_in=defaults.cosmology)
   #     sh_pow5 = sp5.Cll_sh_sh()
    #    sp6 = shear_power(k_in,C,zs,ls,pmodel='halofit_nonlinear',P_in=d[:,1],cosmology_in=defaults.cosmology)
     #   sh_pow6 = sp6.Cll_sh_sh()
        #sp4 = shear_power(k_in,C,zs,ls,pmodel='fastpt_nonlin',P_in=d[:,1],cosmology_in=defaults.cosmology)
        #sh_pow4 = sp4.Cll_sh_sh()
        
        t2 = time()

	print(t2-t1)
        #import projected_power as prj
        #pp=prj.projected_power(k_in,d[:,1],C,3)
        #C_EE1=pp.C_EE(3,ls)
#        sh_sh_pow = sp7.Cll_sh_sh()
#        sh_g_pow = sp7.Cll_sh_g()
#        g_g_pow = sp7.Cll_g_g()
#
#        n_ss = sp7.sigma2_e/(2.*sp7.n_gal)
#        n_gg = 1/sp7.n_gal
#        
#        #n_ss = 0
#        #n_gg = 0
#        #ac,ad,bd,bc
#        cov_ss_gg = sp7.cov_g_diag(sh_g_pow,sh_g_pow,sh_g_pow,sh_g_pow)
#        cov_sg_sg = sp7.cov_g_diag(sh_sh_pow,sh_g_pow,g_g_pow,sh_g_pow,n_ss,0,n_gg,0)
#      #  cov_sg_sg2 = sp7.cov_g_diag2([q_shear(sp7),q_num(sp7),q_shear(sp7),q_num(sp7)],[n_ss,0,n_gg,0],r_bd=sp7.r_corr(),r_bc=sp7.r_corr())
#        cov_sg_ss = sp7.cov_g_diag(sh_sh_pow,sh_sh_pow,sh_g_pow,sh_g_pow,n_ss,n_ss,0,0)
#        cov_sg_gg = sp7.cov_g_diag(sh_g_pow,sh_g_pow,g_g_pow,g_g_pow,0,0,n_gg,n_gg)
#        cov_gg_gg = sp7.cov_g_diag(g_g_pow,g_g_pow,g_g_pow,g_g_pow,n_gg,n_gg,n_gg,n_gg)
#        cov_ss_ss = sp7.cov_g_diag(sh_sh_pow,sh_sh_pow,sh_sh_pow,sh_sh_pow,n_ss,n_ss,n_ss,n_ss)

        
        import matplotlib.pyplot as plt
	ax = plt.subplot(111)
        #ax.loglog(ls,sh_pow_limber)
        #ax.loglog(ls,sh_pow_nolimber)
        #ax.loglog(ls,abs(dc))
    	ax.set_xlabel('l',size=20)
        plt.title('gaussian source with $\sigma=0.4$ centered at $z=1.0$')
        ax.set_ylabel('|$\\frac{\partial ln(P(k))}{\partial \\bar{\delta}}$|')
        ax.loglog(ls,dc_lin)
        ax.loglog(ls,dc_hf)
        ax.loglog(ls,dc_fpt)
        #ax.loglog(ls,abs(dcnl))
        #ax.loglog(ls,abs(deltabar3))
        #ax.loglog(ls,abs(dclin))
        #ax.loglog(ls,abs(dcalt))
        #ax.plot(k_a,abs(dcalt1/p1a))
        #ax.plot(k_a,abs(dcalt2/p2a))
        #ax.plot(k_a,abs(dcalt3/p3a))
        #plt.xlim([0.,0.4])
        #plt.ylim([1,4.])
        #ax.loglog(ls,abs(dcalt3))
        ax.legend(['Linear','Halofit','1 loop fpt'],loc=3)
#	ax.set_ylabel('l(l+1)$C^{\gamma\gamma}(2\pi)^{-1}$')
#	ax.set_ylabel('|$\\frac{\partial C^{\gamma\gamma}}{\partial\Omega_{m}(z)}(2\pi)^{-1}$|')
        #ax.set_xlabel('$\Omega_m h^2$')
        #ax.set_ylabel('$C^{\gamma\gamma}(l)$')
        #ax.plot(omegachs,p10s)
        #ax.plot(omegachs,p100s)
        #ax.plot(omegachs,p1000s)
        #ax.legend(['l=100'],loc=3)
    #    ax.loglog(ls,div_c1)
    #    ax.loglog(ls,div_c4)
    #    ax.loglog(ls,div_c2)
    #    ax.loglog(ls,div_c3)
    #    ax.legend(['z=0.0-0.1','z=0.45-0.55','z=0.95-1.05','z=1.9-2.0'],loc=3)
       # thetas = np.linspace(0.,2.*np.pi,100)
       # tans = sp3.tan_shear(thetas)
       # ax.plot(thetas,tans)
        #ax.loglog(k_in,d[:,1])
        #ax.loglog(k_in,hf.halofitPk(0,defaults.cosmology).D2_L2(k_in)/k_in**3*2.*np.pi**2)
       # ax.loglog(sp7.k_use[:,0],sp7.p_dd_use[:,0])
     #   sh_pow_cosm = np.loadtxt('test_inputs/proj_1/ss_pow.txt')
  #      lin1=ax.loglog(ls,ls*(ls+1.)*sp1.Cll_sh_sh()/(2.*np.pi))
      #  lin1=ax.loglog(ls,ls*(ls+1.)*sp2.Cll_sh_sh()/(2.*np.pi))
        #lin1=ax.loglog(ls,ls*(ls+1.)*sp3.Cll_sh_sh()/(2.*np.pi))
       # lin1=ax.loglog(ls,ls*(ls+1.)*sp7.Cll_sh_sh()/(2.*np.pi))
       # lin1=ax.loglog(ls,ls*(ls+1.)*sp4.Cll_sh_sh()/(2.*np.pi))
       # lin1=ax.loglog(ls,sp2.Cll_sh_sh()/(2.*np.pi))
       # lin1=ax.loglog(ls,sp3.Cll_sh_sh()/(2.*np.pi))
       # lin1=ax.loglog(ls,sp7.Cll_sh_sh()/(2.*np.pi))
   #     lin1=ax.loglog(ls,ls*(ls+1.)*sh_pow_cosm/(2.*np.pi))
       # lin1=ax.loglog(ls,ls*(ls+1.)*(interp1d(ls/C.h,sh_pow1,bounds_error=False,fill_value=-np.inf)(ls))/(2.*np.pi))
    #    lin1=ax.loglog(ls,ls*(ls+1.)*sp4.Cll_sh_sh()/(2.*np.pi))
      #  ax.legend(["halofit linear","halofit nonlinear","cosmosis nonlinear","fastpt nonlinear"],loc=2)
        #sh_pow1_order2 = sp7.Cll_q_q_order2(q_shear(sp7),q_shear(sp7))
        #gg_pow1_order2 = sp7.Cll_q_q_order2(q_num(sp7),q_num(sp7))
        #lin1=ax.loglog(ls,ls*(ls+1.)*sp7.Cll_sh_sh()/(2.*np.pi),label='Shear Power Spectrum linear redshifted')
        #lin1=ax.loglog(ls,ls*(ls+1.)*sh_pow1_order2/(2.*np.pi),label='Shear Power Spectrum linear redshifted')
        #lin1=ax.loglog(ls,ls*(ls+1.)*sp7.Cll_g_g()/(2.*np.pi),label='Shear Power Spectrum linear redshifted')
        #lin1=ax.loglog(ls,ls*(ls+1.)*gg_pow1_order2/(2.*np.pi),label='Shear Power Spectrum linear redshifted')
       # lin1=ax.loglog(ls,sp1.Cll_sh_sh(chi_max1=sp1.chis[10]),label='Shear Power Spectrum linear redshifted')
        #ax.loglog(ls,sp1.Cll_g_g())
        #ax.loglog(ls,sp1.Cll_sh_g())
#        ax.legend(["sh_sh","g_g","sh_g"])
        #ax.loglog(ls,cov_ss_gg) 
        #ax.loglog(ls,cov_sg_sg) 
        #ax.loglog(ls,cov_sg_ss) 
        #ax.loglog(ls,cov_sg_gg) 
        #ax.loglog(ls,cov_ss_ss) 
        #ax.loglog(ls,cov_gg_gg) 
	#ax.legend(["ss_gg","sg_sg","sg_ss","sg_gg","ss_ss","gg_gg"])
        #lin2=ax.loglog(ls,ls*(ls+1.)*sh_pow2/(2.*np.pi),label='Shear Power Spectrum halofit linear')
        #lin3=ax.loglog(ls,ls*(ls+1.)*sh_pow3/(2.*np.pi),label='Shear Power Spectrum halofit nonlinear')

        #ax.loglog(ls,sh_pow_cosm/sh_pow7)
        #lin3=ax.loglog(ls,ls*(ls+1.)*sh_pow5/(2.*np.pi),label='Shear Power Spectrum halofit nonlinear')
       # ax.plot(ls[0:(ls.size-112)],interp1d(ls*C.h,sh_pow7/C.h**3)(ls[0:(ls.size-112)])/sh_pow_cosm[0:(ls.size-112)])
#	lin3=ax.loglog(ls,ls*(ls+1.)*sh_pow6/(2.*np.pi),label='Shear Power Spectrum halofit nonlinear')
        #ax.legend(["1","0.5","2"],loc=2)
       # ax.loglog(ls,ls*(ls+1.)*revd[:,1]/(2.*np.pi),label='Shear Power Spectrum halofit nonlinear')
#	lin4=ax.loglog(ls,ls*(ls+1.)*sh_pow4/(2*np.pi),label='Shear Power Spectrum fastpt nonlinear')
        #ax.loglog(ls,ls*(ls+1.)*chompd[:,1]/(2.*np.pi))
       # ax.legend(["linear redshifted","halofit linear","halofit nonlinear","fastpt nonlinear"])
      #  ax.loglog(ls,np.abs(chompd[:,1]/sh_pow3))

#	ax.loglog(xiaod[:,0],xiaod[:,2])
#	ax.loglog(intd[1],intd[0][:,3])
        #ax.loglog(revd[:,0],revd[:,1])
        #ax.loglog(ls,C_EE1)
       # ax.loglog(ls,C_EE1*revd[0,1]/C_EE1[0])
#        ax.loglog(sp3.k_use[:,0],sp3.p_dd_use[:,0])
        #ax.loglog(d_t[0],hf.halofitPk(0.429,sp3.cosmology).D2_NL(d_t[0])/(d_t[0]**3)*2*np.pi**2)
#        ax.loglog(d_t[0],d_t[1])
 #       ax.loglog(d[:,0],d[:,1]*C.G_norm(0.429)**2)
       # ax.loglog(d_t[:,0],sp2.q_shear()*d_t[0,1]/sp2.q_shear()[0])
        plt.grid()
	plt.show()
       # np.savetxt('chis_save.dat',sp3.chis)
