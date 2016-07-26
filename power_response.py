import numpy as np
import FASTPTcode.FASTPT as FASTPT
import halofit as hf
import cosmopie as cp
from scipy.interpolate import InterpolatedUnivariateSpline
import defaults
import camb_power as cpow

def dp_ddelta(k_a,P_a,zbar,C=cp.CosmoPie(),pmodel='linear',epsilon=0.0001,halo_a=None,halo_b=None,fpt=None):
    if pmodel=='linear':
       # pza = P_a
        pza = P_a*C.G_norm(zbar)**2
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
            fpt = FASTPT.FASTPT(k_a,-2,low_extrap=None,high_extrap=None,n_pad=1000)
        plin = P_a*C.G_norm(zbar)**2

        pza = plin+fpt.one_loop(plin,C_window=0.75)
        dpdk =(InterpolatedUnivariateSpline(k_a,pza,ext=2).derivative(1))(k_a) 
        dp = 47./21.*pza-1./3.*k_a*dpdk+26./21.*fpt.one_loop(plin,C_window=0.75)
    else:
        print('invalid pmodel option \''+str(pmodel)+'\' using linear')
        pza = P_a**C.G_norm(zbar)**2
        dpdk =(InterpolatedUnivariateSpline(k_a,pza,ext=2).derivative(1))(k_a) 
        dp = 47./21.*pza-1./3.*k_a*dpdk
    return dp,pza

#def derv(x,f):
#    return np.gradient(f)/np.gradient(x)
#
#class power_response:
#    def __init__(self,k,P_lin,cosmopie):
#    
#        self.CosmoPie=cosmopie
#        
#        self.k=k
#        self.P_lin=P_lin
#        
#    def linear_response(self,z):
#        # first get d ln P/ d bar{\delta}
#        P=self.CosmoPie.G_norm(z)**2*self.P_lin
#        dp=68/21. - 1/3.*derv(np.log(self.k), np.log(self.k**3*P))
#        return dp/P
#        
#    def SPT_response(self,z):
#        P=self.P_lin
#        G=self.CosmoPie.G_norm(z)
#        fastpt=FASTPT.FASTPT(self.k,-2,low_extrap=-5,high_extrap=5,n_pad=500) 
#        P_spt=fastpt.one_loop(P,C_window=.65)*G**4
#        spt=G**2*P + P_spt
#        
#        dp=68/21. - 1/3.*derv(np.log(self.k), np.log(self.k**3*P)) + 26/21.*P_spt/spt
#        # also return P for a check 
#        return dp/spt, spt 
#    
#    def halofit_response(self,z,delta_bar_0=0.01):
#        # sig_8=self.CosmoPie.sigma_r(0,8)
##         sig_8_new=(1+13/21.*delta_bar_0)*sig_8
##         
##         P_new=np.sqrt(sig_8_new/sig_8)*self.P_lin
##         
##         hf1=halofit(self.k,p_lin=self.P_lin,C=self.CosmoPie)
##         hf2=halofit(self.k,p_lin=P_new,C=self.CosmoPie)
##         
##         P_hf=hf1.P_NL(k,z)
##         
##         dphf=(np.log(hf2.P_NL(k,z))-np.log(P_hf))/np.log(13/21.*delta_bar_0)
##         
##         dp=(13/21.*dphf + 2 - 1/3.*derv(np.log(k**3*P_hf),np.log(k)))/P_hf
##         
##         return dp, P_hf
##         
#        
if __name__=="__main__":
    
        C=cp.CosmoPie(cosmology=defaults.cosmology)
        d = np.loadtxt('camb_m_pow_l.dat')
        k_in = d[:,0]
        zs = np.arange(0.1,1.0,0.1)
        ls = np.arange(1,5000)
        epsilon = 0.0001

        cosmo_a = defaults.cosmology.copy()
        k_a,P_a = cpow.camb_pow(cosmo_a)
        
        import matplotlib.pyplot as plt
        
        d=np.loadtxt('test_pkdbar.dat')

        zbar = 3.
        ax = plt.subplot(221)
        plt.xlim([0.,0.4])
        plt.ylim([1.2,3.2])
        plt.grid()
        plt.title('z=3.0')
        dcalt1,p1a = dp_ddelta(k_a,P_a,zbar,pmodel='linear')
        dcalt2,p2a = dp_ddelta(k_a,P_a,zbar,pmodel='halofit')
        dcalt3,p3a = dp_ddelta(k_a,P_a,zbar,pmodel='fastpt')
        ax.plot(k_a,abs(dcalt1/p1a))
        ax.plot(k_a,abs(dcalt2/p2a))
        ax.plot(k_a,abs(dcalt3/p3a))

        zbar = 2.
        ax = plt.subplot(222)
        plt.xlim([0.,0.4])
        plt.ylim([1.2,3.2])
        plt.grid()
        plt.title('z=2.0')
        dcalt1,p1a = dp_ddelta(k_a,P_a,zbar,pmodel='linear')
        dcalt2,p2a = dp_ddelta(k_a,P_a,zbar,pmodel='halofit')
        dcalt3,p3a = dp_ddelta(k_a,P_a,zbar,pmodel='fastpt')
        ax.plot(k_a,abs(dcalt1/p1a))
        ax.plot(k_a,abs(dcalt2/p2a))
        ax.plot(k_a,abs(dcalt3/p3a))
        

        zbar = 1.
        ax = plt.subplot(223)
        plt.xlim([0.,0.4])
        plt.ylim([1.2,3.2])
        plt.grid()
        plt.title('z=1.0')
        dcalt1,p1a = dp_ddelta(k_a,P_a,zbar,pmodel='linear')
        dcalt2,p2a = dp_ddelta(k_a,P_a,zbar,pmodel='halofit')
        dcalt3,p3a = dp_ddelta(k_a,P_a,zbar,pmodel='fastpt')
        ax.plot(k_a,abs(dcalt1/p1a))
        ax.plot(k_a,abs(dcalt2/p2a))
        ax.plot(k_a,abs(dcalt3/p3a))
        h=.6774
        ax.plot(d[:,0], d[:,1], color='black')
        

        zbar = 0.
        ax = plt.subplot(224)
        plt.xlim([0.,0.4])
        plt.ylim([1.2,3.2])
        plt.grid()
        plt.title('z=0.0')
        dcalt1,p1a = dp_ddelta(k_a,P_a,zbar,pmodel='linear')
        dcalt2,p2a = dp_ddelta(k_a,P_a,zbar,pmodel='halofit')
        dcalt3,p3a = dp_ddelta(k_a,P_a,zbar,pmodel='fastpt')
        ax.plot(k_a,abs(dcalt1/p1a))
        ax.plot(k_a,abs(dcalt2/p2a))
        ax.plot(k_a,abs(dcalt3/p3a))
        
        plt.legend(['linear','halofit','fastpt'],loc=4)
        plt.show()
#    d=np.loadtxt('Pk_Planck15.dat')
#    k=d[:,0]; P=d[:,1]
#    
#    from cosmopie import CosmoPie
#    
#    CosmoPie=CosmoPie(k=k, P_lin=P)
##    print k[-1], P[-1]
#
#    
##    PR=power_response(k,P,CosmoPie)
##    dp_lin=PR.linear_response(0)
##    dp_spt,spt=PR.SPT_response(0)
##    dp_hf, P_hf=PR.halofit_response(0)
##    print dp_hf, P_hf
#    
#    import matplotlib.pyplot as plt
#    from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
#    
#    fig=plt.figure(figsize=(14,10))
#    ax=fig.add_subplot(111)
#    ax.set_xlim(0,1)
#    ax.set_xlabel(r'$k$ [$h$/Mpc]', size=30)
#    ax.set_ylabel(r'$ d\; \ln P/d\bar{\delta}$', size=30)
#    ax.tick_params(axis='both', which='major', labelsize=20)
#    ax.tick_params(axis='both', width=2, length=10)
#    ax.tick_params(axis='both', which='minor', width=1, length=5)
#    ax.xaxis.set_major_formatter(FormatStrFormatter('%2.2f'))
#    ax.xaxis.labelpad = 20
#
#    
#   
#    ax.plot(k,dp_lin*P, lw=4, color='black', label='linear')
#    ax.plot(k,dp_spt*spt,'--', lw=4, color='black', label='SPT')
#    ax.plot(k,dp_hf*P_hf,lw=2, color='red', label='halofit')
#    
#    
#    plt.legend(loc=4, fontsize=30)
#    plt.grid()
#    plt.show()
        
    
        
