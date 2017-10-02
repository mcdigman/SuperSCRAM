import numpy as np
import cosmopie as cp
from scipy.interpolate import InterpolatedUnivariateSpline,RectBivariateSpline
import defaults
import matter_power_spectrum as mps

#supports either vector or scalar zbar, if zbar is vector returns (k_a.size,zbar.size) np array, otherwise k_a.size np array
def dp_ddelta(k_a,P_a,zbar,C,pmodel='linear',epsilon=0.0001):
    #equations can be found in Chiang & Wagner arXiv:
    if pmodel=='linear':
        #TODO evaluate if redshift dependence (anywhere) needs to include Chiang & Wagner correction 4.17 
        #support vector zbar
        if isinstance(zbar,np.ndarray) and zbar.size>1:
            pza = P_a.get_matter_power(zbar,pmodel='linear')
            #degree must be at least 2 for derivative, apparently
            #TODO no particular reason to use rectbivariate spline instead of a bunch of univariate splines here, maybe just make a method for getting the derivative without splines
            dpdk = RectBivariateSpline(k_a,zbar,pza,kx=2,ky=1)(k_a,zbar,dx=1) 
            dp = 47./21.*pza-1./3.*(k_a*dpdk.T).T
        #TODO support scalar zbar everywhere
        else:
            pza = P_a.get_matter_power(zbar,pmodel='linear')[:,0]
            dpdk = (InterpolatedUnivariateSpline(k_a,pza,ext=2,k=2).derivative(1))(k_a)
            dp = 47./21.*pza-1./3.*(k_a*dpdk)
    elif pmodel=='halofit':
        if isinstance(zbar,np.ndarray) and zbar.size>1:
            #TODO insert prescription to handle the spike when this switches to using the linear matter power spectrum
            pza = P_a.get_matter_power(zbar,pmodel='halofit',const_pow_mult=1.)
            pzb = P_a.get_matter_power(zbar,pmodel='halofit',const_pow_mult=(1.+epsilon/C.get_sigma8())**2)
            pzc = P_a.get_matter_power(zbar,pmodel='halofit',const_pow_mult=(1.-epsilon/C.get_sigma8())**2)
            dpdk = RectBivariateSpline(k_a,zbar,pza,kx=2,ky=1)(k_a,zbar,dx=1) 
            dp = 13./21.*C.get_sigma8()*(pzb-pzc)/(2.*epsilon)+pza-1./3.*(k_a*dpdk.T).T
            #dp = 13./21.*C.get_sigma8()*(pzb-pza)/(epsilon)+pza-1./3.*(k_a*dpdk.T).T
        else:
            pza = P_a.get_matter_power(zbar,pmodel='halofit',const_pow_mult=1.)[:,0]
            pzb = P_a.get_matter_power(zbar,pmodel='halofit',const_pow_mult=(1.+epsilon/C.get_sigma8())**2)[:,0]
            pzc = P_a.get_matter_power(zbar,pmodel='halofit',const_pow_mult=(1.-epsilon/C.get_sigma8())**2)[:,0]
            dpdk =(InterpolatedUnivariateSpline(k_a,pza,ext=2,k=2).derivative(1))(k_a) 
            dp = 13./21.*C.get_sigma8()*(pzb-pzc)/(2.*epsilon)+pza-1./3.*k_a*dpdk
            #dp = 13./21.*C.get_sigma8()*(pzb-pza)/(epsilon)+pza-1./3.*k_a*dpdk
    elif pmodel=='fastpt':
        if isinstance(zbar,np.ndarray) and zbar.size>1:
            pza,one_loop = P_a.get_matter_power(zbar,pmodel='fastpt',get_one_loop=True)
            dpdk = RectBivariateSpline(k_a,zbar,pza,kx=2,ky=1)(k_a,zbar,dx=1) 
            dp = 47./21.*pza-1./3.*(k_a*(dpdk.T)).T+26./21.*one_loop
        else:
            pza,one_loop = P_a.get_matter_power(zbar,pmodel='fastpt',get_one_loop=True)
            pza=pza[:,0]
            one_loop=one_loop[:,0]
            dpdk = (InterpolatedUnivariateSpline(k_a,pza,ext=2,k=2).derivative(1))(k_a) 
            dp = 47./21.*pza-1./3.*k_a*dpdk+26./21.*one_loop
    else:
        raise ValueError('invalid pmodel option \''+str(pmodel)+'\'')
    return dp,pza

if __name__=="__main__":
    
        C=cp.CosmoPie(cosmology=defaults.cosmology)
        d = np.loadtxt('camb_m_pow_l.dat')
        k_in = d[:,0]
        zs = np.arange(0.1,1.0,0.1)
        ls = np.arange(1,5000)
        epsilon = 0.0001

        cosmo_a = defaults.cosmology.copy()
        #k_a,P_a = cpow.camb_pow(cosmo_a)
    
        P_a = mps.MatterPower(C)
        k_a = P_a.k
        
        import matplotlib.pyplot as plt
        
        #d=np.loadtxt('test_pkdbar.dat')
        zbar = np.array([3.])
        ax = plt.subplot(221)
        plt.xlim([0.,0.4])
        plt.ylim([1.2,3.2])
        plt.grid()
        plt.title('z=3.0')
        dcalt1,p1a = dp_ddelta(k_a,P_a,zbar,C,pmodel='linear')
        dcalt2,p2a = dp_ddelta(k_a,P_a,zbar,C,pmodel='halofit')
        dcalt3,p3a = dp_ddelta(k_a,P_a,zbar,C,pmodel='fastpt')
        ax.plot(k_a,abs(dcalt1/p1a))
        ax.plot(k_a,abs(dcalt2/p2a))
        ax.plot(k_a,abs(dcalt3/p3a))

        zbar = np.array([2.])
        ax = plt.subplot(222)
        plt.xlim([0.,0.4])
        plt.ylim([1.2,3.2])
        plt.grid()
        plt.title('z=2.0')
        dcalt1,p1a = dp_ddelta(k_a,P_a,zbar,C,pmodel='linear')
        dcalt2,p2a = dp_ddelta(k_a,P_a,zbar,C,pmodel='halofit')
        dcalt3,p3a = dp_ddelta(k_a,P_a,zbar,C,pmodel='fastpt')
        ax.plot(k_a,abs(dcalt1/p1a))
        ax.plot(k_a,abs(dcalt2/p2a))
        ax.plot(k_a,abs(dcalt3/p3a))
        

        zbar = np.array([1.])
        ax = plt.subplot(223)
        plt.xlim([0.,0.4])
        plt.ylim([1.2,3.2])
        plt.grid()
        plt.title('z=1.0')
        dcalt1,p1a = dp_ddelta(k_a,P_a,zbar,C,pmodel='linear')
        dcalt2,p2a = dp_ddelta(k_a,P_a,zbar,C,pmodel='halofit')
        dcalt3,p3a = dp_ddelta(k_a,P_a,zbar,C,pmodel='fastpt')
        ax.plot(k_a,abs(dcalt1/p1a))
        ax.plot(k_a,abs(dcalt2/p2a))
        ax.plot(k_a,abs(dcalt3/p3a))
        h=.6774
        ax.plot(d[:,0], d[:,1], color='black')
        

        zbar = np.array([0.])
        ax = plt.subplot(224)
        plt.xlim([0.,0.4])
        plt.ylim([1.2,3.2])
        plt.grid()
        plt.title('z=0.0')
        dcalt1,p1a = dp_ddelta(k_a,P_a,zbar,C,pmodel='linear')
        dcalt2,p2a = dp_ddelta(k_a,P_a,zbar,C,pmodel='halofit')
        dcalt3,p3a = dp_ddelta(k_a,P_a,zbar,C,pmodel='fastpt')
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
        
    
        
