from scipy.integrate import simps
import numpy as np
import cosmopie as cp
import halofit as hf
from numpy import exp
#from scipy.integrate import quad
from scipy.interpolate import interp1d,interp2d
from scipy.interpolate import InterpolatedUnivariateSpline,SmoothBivariateSpline
from time import time
import pickle 
#import ps2
import FASTPTcode.FASTPT as FASTPT
import defaults

class shear_power:
    def __init__(self,k_in,C,zs,ls,pmodel='halofit_linear',P_in=np.array([]),cosmology_in={},ps=np.array([])):
	self.k_in = k_in
        self.C = C
        self.zs = zs
        self.ls = ls
        self.epsilon = 0.0000001
        self.delta_l = ls[-1]-ls[0] #maybe not right
        self.omega_s = 5000 #filler, from eifler, in deg^2
        self.n_gal = 26000000. #filler, from krause & eifler in galaxies/arcmin^2

        self.sigma2_e = 0.32 #from eifler
        self.sigma2_mu = 1.2 #eifler says this is uncertain
        
        self.n_k = self.k_in.size
        self.n_l = self.ls.size
        self.n_z = self.zs.size
        
        self.chis = np.zeros(self.n_z)
        self.omegas = np.zeros(self.n_z)

        self.k_use=np.zeros((self.n_l,self.n_z))
        self.p_dd_use=np.zeros((self.n_l,self.n_z))
	self.sc_as = np.zeros(self.n_z)
	self.dchis = np.zeros(self.n_z)
	self.ps = np.zeros(self.n_z)

        self.cosmology = cosmology_in
            
        #some methods require special handling    
        if pmodel=='halofit_redshift_nonlinear':
            p_bar = hf.halofitPk(0.42891513142857135,self.cosmology).D2_NL(self.k_in)
        elif pmodel == 'cosmosis_nonlinear':
            z_bar = np.loadtxt('test_inputs/proj_1/z.txt')
            p_bar = interp2d(k_in,z_bar,np.loadtxt('test_inputs/proj_1/p_k.txt'))
            self.chis = interp1d(z_bar,np.loadtxt('test_inputs/proj_1/d_m.txt')[::-1])(zs)

        for i in range(0,self.n_z):
            if pmodel!='cosmosis_nonlinear':
                self.chis[i] = self.C.D_comov(zs[i])
            self.omegas[i] = self.C.Omegam
       #     self.omegas[i] = self.C.Omegam_z(zs[i])

	    self.k_use[:,i] = (self.ls+0.5)/self.chis[i] 
            if pmodel=='halofit_linear':
                self.p_dd_use[:,i] = 2*np.pi**2*interp1d(self.k_in,hf.halofitPk(self.zs[i],self.cosmology).D2_L(self.k_in))(self.k_use[:,i])/self.k_use[:,i]**3
            elif pmodel=='halofit_nonlinear':
                self.p_dd_use[:,i] = 2*np.pi**2*interp1d(self.k_in,hf.halofitPk(self.zs[i],self.cosmology).D2_NL(self.k_in))(self.k_use[:,i])/self.k_use[:,i]**3
            elif pmodel=='halofit_redshift_nonlinear':
                #self.p_dd_use[:,i] = 2*np.pi**2*interp1d(self.k_in,p_bar*self.C.G_norm(self.zs[i])**2/(self.C.G_norm(0.42891513142857135)**2))(self.k_use[:,i])/self.k_use[:,i]**3
                self.p_dd_use[:,i] = interp1d(self.k_in,P_in*self.C.G_norm(self.zs[i])**2/self.C.G_norm(0.42891513142857135)**2)(self.k_use[:,i])
            elif pmodel=='redshift_linear':
                self.p_dd_use[:,i] = interp1d(self.k_in,P_in*self.C.G_norm(self.zs[i])**2)(self.k_use[:,i])
            elif pmodel=='cosmosis_nonlinear':
                #self.p_dd_use[:,i] = interp1d(self.k_in,P_in*self.C.G_norm(self.zs[i])**2)(self.k_use[:,i])
               # self.p_dd_use[:,i] = interp1d(self.k_in,p_bar[i])(self.k_use[:,i])
                self.p_dd_use[:,i] = p_bar(self.k_use[:,i],zs[i])
                #self.p_dd_use[:,i] = p_bar[i]
            elif pmodel=='fastpt_nonlin':
                fpt = FASTPT.FASTPT(k_in,-2,low_extrap=-5,high_extrap=5,n_pad=800)
                self.p_dd_use[:,i] = interp1d(self.k_in,(fpt.one_loop(P_in*self.C.G_norm(self.zs[i])**2,C_window=0.75)+P_in*self.C.G_norm(self.zs[i])**2))(self.k_use[:,i])
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
           # self.ps[-2] = 1./(self.chis[-1]-self.chis[-2])  #make option, this makes it act like a proper delta function
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

    def Cll_sh_mag(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        return self.Cll_q_q(q_shear(self,chi_max=chi_max1,chi_min=chi_min1),q_mag(self,chi_max=chi_max2,chi_min=chi_min2))

    def Cll_mag_g(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        return self.Cll_q_q(q_mag(self,chi_max=chi_max1,chi_min=chi_min1),q_num(self,chi_max=chi_max2,chi_min=chi_min2))

    def Cll_sh_g(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        return self.Cll_q_q(q_shear(self,chi_max=chi_max1,chi_min=chi_min2),q_num(self,chi_max=chi_max2,chi_min=chi_min2),corr_param=self.r_corr())

    #def q_shear(self,chi_max=np.inf,chi_min=0.):
    #    return 3./2.*(self.C.H0/self.C.c)**2*self.omegas*self.chis/self.sc_as*self.gs(chi_max=chi_max,chi_min=chi_min)


    def Cll_q_q(self,q1s,q2s,corr_param=np.array([])):
        integrand = np.zeros((self.n_z,self.n_l))
        if corr_param.size !=0: 
            for i in range(0,self.n_z):
                integrand[i] = q1s.qs[i]*q2s.qs[i]/self.chis[i]**2*self.p_dd_use[:,i]*corr_param[:,i]
        else:
            for i in range(0,self.n_z):
                integrand[i] = q1s.qs[i]*q2s.qs[i]/self.chis[i]**2*self.p_dd_use[:,i]
        return np.trapz(integrand,self.chis,axis=0)

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
        
class q_weight:
    def __init__(self,chis,qs,chi_min=0.,chi_max=np.inf):
        self.chi_min = chi_min
        self.chi_max = chi_max
        self.qs = qs
       # self.q_spline = InterpolatedUnivariateSpline(chis,self.qs,ext=2)
       # self.r_spline = InterpolatedUnivariateSpline(chis,self.qs/np.sqrt(chis),ext=2) #for calculating to second order
class q_shear(q_weight):
    def __init__(self,sp,chi_min=0.,chi_max=np.inf,mult=1.):
        qs = 3./2.*(sp.C.H0/sp.C.c)**2*sp.omegas*sp.chis/sp.sc_as*self.gs(sp,chi_max=chi_max,chi_min=chi_min)*mult
        r_spline = InterpolatedUnivariateSpline(sp.chis,qs/np.sqrt(sp.chis),ext=2) #for calculating to second order
        self.rs = r_spline(sp.chis)
        self.rs_d1 = r_spline.derivative(1)(sp.chis)
        self.rs_d2 = r_spline.derivative(2)(sp.chis)
        self.rs_d3 = r_spline.derivative(3)(sp.chis)
        q_weight.__init__(self,sp.chis,qs,chi_min=chi_min,chi_max=chi_max) 

    def gs(self,sp,chi_max=np.inf,chi_min=0):
        g_vals = np.zeros(sp.n_z)
        low_mask = (sp.chis>=chi_min)*1. #so only integrate from max(chi,chi_min)
	for i in range(0,sp.n_z):
            if chi_max<sp.chis[i]:
                break
            g_vals[i] =np.trapz(low_mask[i:sp.n_z]*sp.ps[i:sp.n_z]*(sp.chis[i:sp.n_z]-sp.chis[i])/sp.chis[i:sp.n_z],sp.chis[i:sp.n_z])
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
        self.rs = np.zeros((sp.n_z,sp.n_l))

        self.rs_d1 = np.zeros((sp.n_z,sp.n_l))
        self.rs_d2 = np.zeros((sp.n_z,sp.n_l))
        self.rs_d3 = np.zeros((sp.n_z,sp.n_l))
        for i in range(0,sp.n_l):
            r_spline = InterpolatedUnivariateSpline(sp.chis,q[:,i]/np.sqrt(sp.chis))
            self.rs[:,i] = r_spline(sp.chis)
            self.rs_d1[:,i] = r_spline.derivative(1)(sp.chis)
            self.rs_d2[:,i] = r_spline.derivative(2)(sp.chis)
            self.rs_d3[:,i] = r_spline.derivative(3)(sp.chis)
        q_weight.__init__(self,sp.chis,q,chi_min=chi_min,chi_max=chi_max)

    def bias(self,sp):
        return np.sqrt(sp.p_gg_use/sp.p_dd_use)

     
    #maybe can get bias and r_corr directly from something else
if __name__=='__main__':
    
        #d_t = np.loadtxt('chomp_pow_nlin.dat')
	d = np.loadtxt('Pk_Planck15.dat')
#	xiaod = np.loadtxt('power_spectrum_1.dat')
#	intd = pickle.load(open('../CosmicShearPython/Cll_camb.pkl',mode='r'))
        #chompd = np.loadtxt('sh_sh_comp.dat')
#        revd = np.loadtxt('len_pow1.dat')
#        k_in = chompd[:,0]
	C=cp.CosmoPie(cosmology=defaults.cosmology_cosmosis)
        #k_in = np.logspace(-5,2,200,base=10)
        k_in = np.loadtxt('test_inputs/proj_1/k_h.txt')
#        k_in = np.logspace(-4,5,5000,base=10)
#	zs = np.logspace(-2,np.log10(3),50,base=10)
#	zs = np.arange(0.001,1,0.001)
        zs = np.loadtxt('test_inputs/proj_1/z.txt')
        zs[0] = 10**-3
        #zs = np.arange(0.005,2,0.0005)
	#ls = np.arange(2,3000)
        ls = np.loadtxt('test_inputs/proj_1/ell.txt')
        #ls = np.logspace(np.log10(2),np.log10(10000),3000,base=10)
	#ls = chompd[:,0]	
#	ls = revd[:,0]	
	t1 = time()
        
        #sp1 = shear_power(k_in,C,zs[0:100],ls,pmodel='redshift_linear',P_in=d[:,1],cosmology_in=defaults.cosmology)
        #sh_pow1 = sp1.Cll_sh_sh()
        #sp2 = shear_power(k_in,C,zs,ls,pmodel='halofit_linear',P_in=d[:,1],cosmology_in=defaults.cosmology)
        #sp1 = shear_power(k_in,C,zs,ls,pmodel='halofit_linear',P_in=d[:,1],cosmology_in=defaults.cosmology)
        #sh_pow2 = sp2.Cll_sh_sh()
      #  sp3 = shear_power(k_in,C,zs,ls,pmodel='halofit_nonlinear',P_in=d[:,1],cosmology_in=defaults.cosmology)
      #  sh_pow3 = sp3.Cll_sh_sh()
        sp7 = shear_power(k_in,C,zs,ls,pmodel='cosmosis_nonlinear',P_in=d[:,1],cosmology_in=defaults.cosmology_cosmosis)
        sh_pow7 = sp7.Cll_sh_sh()
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
        sh_sh_pow = sp7.Cll_sh_sh()
        sh_g_pow = sp7.Cll_sh_g()
        g_g_pow = sp7.Cll_g_g()

        n_ss = sp7.sigma2_e/(2.*sp7.n_gal)
        n_gg = 1/sp7.n_gal
        
        #n_ss = 0
        #n_gg = 0
        #ac,ad,bd,bc
        cov_ss_gg = sp7.cov_g_diag(sh_g_pow,sh_g_pow,sh_g_pow,sh_g_pow)
        cov_sg_sg = sp7.cov_g_diag(sh_sh_pow,sh_g_pow,g_g_pow,sh_g_pow,n_ss,0,n_gg,0)
      #  cov_sg_sg2 = sp7.cov_g_diag2([q_shear(sp7),q_num(sp7),q_shear(sp7),q_num(sp7)],[n_ss,0,n_gg,0],r_bd=sp7.r_corr(),r_bc=sp7.r_corr())
        cov_sg_ss = sp7.cov_g_diag(sh_sh_pow,sh_sh_pow,sh_g_pow,sh_g_pow,n_ss,n_ss,0,0)
        cov_sg_gg = sp7.cov_g_diag(sh_g_pow,sh_g_pow,g_g_pow,g_g_pow,0,0,n_gg,n_gg)
        cov_gg_gg = sp7.cov_g_diag(g_g_pow,g_g_pow,g_g_pow,g_g_pow,n_gg,n_gg,n_gg,n_gg)
        cov_ss_ss = sp7.cov_g_diag(sh_sh_pow,sh_sh_pow,sh_sh_pow,sh_sh_pow,n_ss,n_ss,n_ss,n_ss)

        
        import matplotlib.pyplot as plt
	ax = plt.subplot(111)
	ax.set_xlabel('l',size=20)
	ax.set_ylabel('l(l+1)$C^{\gamma\gamma}(2\pi)^{-1}$')
        
        sh_pow1_order2 = sp7.Cll_q_q_order2(q_shear(sp7),q_shear(sp7))
        gg_pow1_order2 = sp7.Cll_q_q_order2(q_num(sp7),q_num(sp7))
        lin1=ax.loglog(ls,ls*(ls+1.)*sp7.Cll_sh_sh()/(2.*np.pi),label='Shear Power Spectrum linear redshifted')
        lin1=ax.loglog(ls,ls*(ls+1.)*sh_pow1_order2/(2.*np.pi),label='Shear Power Spectrum linear redshifted')
        lin1=ax.loglog(ls,ls*(ls+1.)*sp7.Cll_g_g()/(2.*np.pi),label='Shear Power Spectrum linear redshifted')
        lin1=ax.loglog(ls,ls*(ls+1.)*gg_pow1_order2/(2.*np.pi),label='Shear Power Spectrum linear redshifted')
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
