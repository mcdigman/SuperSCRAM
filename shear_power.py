from scipy.integrate import simps
import numpy as np
import cosmopie as cp
import halofit as hf
from numpy import exp
#from scipy.integrate import quad
from scipy.interpolate import interp1d
from time import time
import pickle 
import ps2
import FASTPTcode.FASTPT as FASTPT
import defaults

class shear_power:
    def __init__(self,k_in,C,zs,ls,pmodel='halofit_linear',P_in=[],cosmology_in={}):
	self.k_in = k_in
        self.C = C
        self.zs = zs
        self.ls = ls

        self.epsilon = 0.0000001
        self.delta_l = ls[-1]-ls[0] #maybe not right
        self.omega_s = 5000 #filler, from eifler, in deg^2
        self.n_gal = 26. #filler, from krause & eifler in galaxies/arcmin^2

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

        for i in range(0,self.n_z):
            self.chis[i] = self.C.D_comov(zs[i])
            self.omegas[i] = self.C.Omegam_z(zs[i])

	    self.k_use[:,i] = self.ls/self.chis[i] 
            if pmodel=='halofit_linear':
                self.p_dd_use[:,i] = 2*np.pi**2*interp1d(self.k_in,hf.PowerSpectrum(self.zs[i],self.cosmology).D2_L(self.k_in))(self.k_use[:,i])/self.k_use[:,i]**3
            elif pmodel=='halofit_nonlinear':
                self.p_dd_use[:,i] = 2*np.pi**2*interp1d(self.k_in,hf.PowerSpectrum(self.zs[i],self.cosmology).D2_NL(self.k_in))(self.k_use[:,i])/self.k_use[:,i]**3
            elif pmodel=='redshift_linear':
                self.p_dd_use[:,i] = interp1d(self.k_in,P_in*self.C.G_norm(self.zs[i])**2)(self.k_use[:,i])
            elif pmodel=='fastpt_nonlin':
                fpt = FASTPT.FASTPT(k_in,-2,low_extrap=-5,high_extrap=5,n_pad=800)
                self.p_dd_use[:,i] = interp1d(self.k_in,(fpt.one_loop(P_in*self.C.G_norm(self.zs[i])**2,C_window=0.75)+P_in*self.C.G_norm(self.zs[i])**2))(self.k_use[:,i])
            else:
                if i==0:
                    print("invalid pmodel value \'"+pmodel+"\' using halofit_linear instead")
                self.p_dd_use[:,i] = 2*np.pi**2*interp1d(self.k_in,hf.PowerSpectrum(self.zs[i],self.cosmology).D2_L(self.k_in))(self.k_use[:,i])/self.k_use[:,i]**3


	    self.dchis[i] = (self.C.D_comov(self.zs[i]+self.epsilon)-self.C.D_comov(self.zs[i]))/self.epsilon
	    self.sc_as[i] = 1/(1+self.zs[i])

        self.p_gg_use = self.p_dd_use #temporary for testing purposes
        self.p_gd_use = self.p_dd_use #temporary for testing purposes

        #self.ps[-2] = 1./(self.chis[-1]-self.chis[-2])  #make option, this makes it act like a proper delta function
        #self.ps[20] = 1./(self.chis[21]-self.chis[20])  #make option, this makes it act like a proper delta function
        #self.ps[30] = 1./(self.chis[31]-self.chis[30])  #make option, this makes it act like a proper delta function
        for i in range(0,self.n_z-1): #assume constant distribution of galaxies
            self.ps[i] = 1/(self.chis[i+1]-self.chis[i])
        self.ps[-1] = 1/(self.C.D_comov(2*zs[-1]-zs[-2])-self.chis[-1]) #patch last value so it isn't just 0
        #for i in range(0,self.n_z-1): # a model from the cosmolike paper
        #    self.ps[i] = zs[i]**1.24*np.exp(-(zs[i]/0.51)**1.01)/(self.chis[i+1]-self.chis[i])
        self.ps = self.ps/np.trapz(self.ps,self.chis) #normalize galaxy probability distribution

    def gs(self,chi_max=np.inf,chi_min=0):
        g_vals = np.zeros(self.n_z)
        low_mask = (self.chis>=chi_min)*1. #so only integrate from max(chi,chi_min)
	for i in range(0,self.n_z):
            if chi_max<self.chis[i]:
                break
            g_vals[i] =np.trapz(low_mask[i:self.n_z]*self.ps[i:self.n_z]*(self.chis[i:self.n_z]-self.chis[i])/self.chis[i:self.n_z],self.chis[i:self.n_z])#*self.dchis[i:self.n_z],dx=0.01)
        return g_vals
     
    #maybe can get bias and r_corr directly from something else
    def bias(self):
        return np.sqrt(self.p_gg_use/self.p_dd_use)

    def r_corr(self):
        return self.p_gd_use/np.sqrt(self.p_dd_use*self.p_gg_use)

    def Cll_sh_sh(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
      #  g_vals = self.gs()
#	sh_pow = np.zeros(self.n_l)	
#	sh_pow =  9./4.*(self.C.H0/self.C.c)**4*np.trapz(self.C.Omegam_z(zs)**2*g_vals**2/self.sc_as**2*(self.p_dd_use),self.chis,axis=1)#*self.dchis,dx=0.01,axis=1)
 #       return sh_pow
        return self.Cll_q_q(self.q_shear(chi_max=chi_max1,chi_min=chi_min1),self.q_shear(chi_max=chi_max2,chi_min=chi_min2))

    def Cll_g_g(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        #integrand = np.zeros((self.n_z,self.n_l))
        #b = self.bias()
        #for i in range(0,self.n_z):
        #    integrand[i] = self.ps[i]**2/self.chis[i]**2*b[:,i]**2*self.p_dd_use[:,i] 
        #return np.trapz(integrand,self.chis,axis=0)
        return self.Cll_q_q(self.q_num(chi_max=chi_max1,chi_min=chi_min1),self.q_num(chi_max=chi_max2,chi_min=chi_min2))
    
    def Cll_mag_mag(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        return self.Cll_q_q(self.q_mag(chi_max=chi_max1,chi_min=chi_min1),self.q_mag(chi_max=chi_max2,chi_min=chi_min2))
    
    def Cll_k_k(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        return self.Cll_q_q(self.q_k(chi_max=chi_max1,chi_min=chi_min1),self.q_k(chi_max=chi_max2,chi_min=chi_min2))

    def Cll_sh_mag(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        return self.Cll_q_q(self.q_shear(chi_max=chi_max1,chi_min=chi_min1),self.q_mag(chi_max=chi_max2,chi_min=chi_min2))

    def Cll_mag_g(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
        return self.Cll_q_q(self.q_mag(chi_max=chi_max1,chi_min=chi_min1),self.q_num(chi_max=chi_max2,chi_min=chi_min2))

    def Cll_sh_g(self,chi_max1=np.inf,chi_max2=np.inf,chi_min1=0.,chi_min2=0.):
#        g = self.gs()
#        b = self.bias()
#        r = self.r_corr()
#        integrand = np.zeros((self.n_z,self.n_l))
#        for i in range(0,self.n_z):
#            integrand[i]=3./2.*(self.C.H0/self.C.c)**2*self.omegas[i]*g[i]*self.ps[i]*b[:,i]*r[:,i]/(self.sc_as[i]*self.chis[i])*self.p_dd_use[:,i]
#        return np.trapz(integrand,self.chis,axis=0)
        return self.Cll_q_q(self.q_shear(chi_max=chi_max1,chi_min=chi_min2),self.q_num(chi_max=chi_max2,chi_min=chi_min2),corr_param=self.r_corr())

    def q_shear(self,chi_max=np.inf,chi_min=0.):
        return 3./2.*(self.C.H0/self.C.c)**2*self.omegas*self.chis/self.sc_as*self.gs(chi_max=chi_max,chi_min=chi_min)

    def q_mag(self,chi_max=np.inf,chi_min=0.):
        return self.q_shear(chi_max=chi_max,chi_min=chi_min)*2.

    def q_k(self,chi_max=np.inf,chi_min=0.):
        return self.q_shear(chi_max=chi_max,chi_min=chi_min)

    def q_num(self,chi_max=np.inf,chi_min=0.):
        q = np.zeros((self.n_z,self.n_l))
        b = self.bias()
        for i in range(0,self.n_z):
            if self.chis[i]>chi_max:
                break
            elif self.chis[i]<chi_min:
                continue
            else:
                q[i] = self.ps[i]*b[:,i]
        return q

    def Cll_q_q(self,q1s,q2s,corr_param=np.array([])):
        integrand = np.zeros((self.n_z,self.n_l))
        if corr_param.size !=0: 
            for i in range(0,self.n_z):
                integrand[i] = q1s[i]*q2s[i]/self.chis[i]**2*self.p_dd_use[:,i]*corr_param[:,i]
        else:
            for i in range(0,self.n_z):
                integrand[i] = q1s[i]*q2s[i]/self.chis[i]**2*self.p_dd_use[:,i]
        return np.trapz(integrand,self.chis,axis=0)

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

        for i in range(0,self.n_l):
            cov_diag[i] = 2*np.pi/(self.omega_s*self.ls[i]*self.delta_l)*((c_ac[i]+ns[0])*(c_bd[i]+ns[2])+(c_ad[i]+ns[1])*(c_bc[i]+ns[3]))
        return (cov_diag,c_ac,c_bd,c_ad,c_bc)

    

if __name__=='__main__':
    
	d = np.loadtxt('Pk_Planck15.dat')
#	xiaod = np.loadtxt('power_spectrum_1.dat')
#	intd = pickle.load(open('../CosmicShearPython/Cll_camb.pkl',mode='r'))
        revd = np.loadtxt('len_pow1.dat')
        k_in = d[:,0]
#	zs = np.logspace(-2,np.log10(3),50,base=10)
	zs = np.arange(0.01,3,0.01)
	C=cp.CosmoPie()
	ls = np.arange(2,3000)
#	ls = revd[:,0]	
	t1 = time()
        
        sp1 = shear_power(k_in,C,zs,ls,pmodel='redshift_linear',P_in=d[:,1],cosmology_in=defaults.cosmology)
        #sh_pow1 = sp1.Cll_sh_sh()
        #sp2 = shear_power(k_in,C,zs,ls,pmodel='halofit_linear',P_in=d[:,1])
        #sh_pow2 = sp2.Cll_sh_sh()
        #sp3 = shear_power(k_in,C,zs,ls,pmodel='halofit_nonlinear',P_in=d[:,1])
        #sh_pow3 = sp3.Cll_sh_sh()
        #sp4 = shear_power(k_in,C,zs,ls,pmodel='fastpt_nonlin',P_in=d[:,1])
        #sh_pow4 = sp4.Cll_sh_sh()
        
        t2 = time()

	print(t2-t1)
        #import projected_power as prj
        #pp=prj.projected_power(k_in,d[:,1],C,3)
        #C_EE1=pp.C_EE(3,ls)
        sh_sh_pow = sp1.Cll_sh_sh()
        sh_g_pow = sp1.Cll_sh_g()
        g_g_pow = sp1.Cll_g_g()

        n_ss = sp1.sigma2_e/(2.*sp1.n_gal)
        n_gg = 1/sp1.n_gal
        
        #n_ss = 0
        #n_gg = 0
        #ac,ad,bd,bc
        cov_ss_gg = sp1.cov_g_diag(sh_g_pow,sh_g_pow,sh_g_pow,sh_g_pow)
        cov_sg_sg = sp1.cov_g_diag(sh_sh_pow,sh_g_pow,g_g_pow,sh_g_pow,n_ss,0,n_gg,0)
        #cov_sg_sg2 = sp1.cov_g_diag2([sp1.q_shear(),sp1.q_num(),sp1.q_shear(),sp1.q_num()],[n_ss,0,n_gg,0],r_bd=sp1.r_corr(),r_bc=sp1.r_corr())
        cov_sg_ss = sp1.cov_g_diag(sh_sh_pow,sh_sh_pow,sh_g_pow,sh_g_pow,n_ss,n_ss,0,0)
        cov_sg_gg = sp1.cov_g_diag(sh_g_pow,sh_g_pow,g_g_pow,g_g_pow,0,0,n_gg,n_gg)
        cov_gg_gg = sp1.cov_g_diag(g_g_pow,g_g_pow,g_g_pow,g_g_pow,n_gg,n_gg,n_gg,n_gg)
        cov_ss_ss = sp1.cov_g_diag(sh_sh_pow,sh_sh_pow,sh_sh_pow,sh_sh_pow,n_ss,n_ss,n_ss,n_ss)

        import matplotlib.pyplot as plt
	ax = plt.subplot(111)
	ax.set_xlabel('l',size=20)
	ax.set_ylabel('C')
        
      #  lin1=ax.loglog(ls,sp1.Cll_sh_sh(),label='Shear Power Spectrum linear redshifted')
       # lin1=ax.loglog(ls,sp1.Cll_sh_sh(chi_max1=sp1.chis[10]),label='Shear Power Spectrum linear redshifted')
     #   ax.loglog(ls,sp1.Cll_g_g())
     #   ax.loglog(ls,sp1.Cll_sh_g())
#        ax.legend(["sh_sh","g_g","sh_g"])
        ax.loglog(ls,cov_ss_gg) 
        ax.loglog(ls,cov_sg_sg) 
        #ax.loglog(ls,cov_sg_sg2) 
        ax.loglog(ls,cov_sg_ss) 
        ax.loglog(ls,cov_sg_gg) 
        ax.loglog(ls,cov_ss_ss) 
        ax.loglog(ls,cov_gg_gg) 
	ax.legend(["ss_gg","sg_sg","sg_ss","sg_gg","ss_ss","gg_gg"])
        #lin2=ax.loglog(ls,sh_pow2,label='Shear Power Spectrum halofit linear')
	#lin3=ax.loglog(ls,sh_pow3,label='Shear Power Spectrum halofit nonlinear')
	#lin4=ax.loglog(ls,sh_pow4,label='Shear Power Spectrum fastpt nonlinear')

 #       ax.legend(["linear redshifted","halofit linear","halofit nonlinear","fastpt nonlinear"])

#	ax.loglog(xiaod[:,0],xiaod[:,2])
#	ax.loglog(intd[1],intd[0][:,3])
        #ax.loglog(revd[:,0],revd[:,1])
        #ax.loglog(ls,C_EE1)
       # ax.loglog(ls,C_EE1*revd[0,1]/C_EE1[0])
	plt.grid()
	plt.show()

