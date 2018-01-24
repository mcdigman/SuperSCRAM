"""get response of matter power spectrum to an overall density perturbation"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline,RectBivariateSpline

def dp_ddelta(P_a,zbar,C,pmodel='linear',epsilon=0.0001):
    """get separate universe response of the power spectrum to a density fluctuation
        inputs:
            k_a: the k vector
            P_a: a MatterPowerSpectrum object
            zbar: the z bins to get, if zbar is vector return (P_a.k.size,zbar.size) np array, otherwise P_a.k.size np array
            C: a CosmoPie object
            pmodel: the power spectrum model to use, options 'linear','halofit','fastpt'
            epsilon: the epsilon to use for finite difference derivatives needed to get 'halofit' response
    """
    k_a = P_a.k
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
            dpdk = (InterpolatedUnivariateSpline(k_a,pza,ext=2,k=2).derivative(1))(k_a)
            dp = 13./21.*C.get_sigma8()*(pzb-pzc)/(2.*epsilon)+pza-1./3.*k_a*dpdk
            #dp = 13./21.*C.get_sigma8()*(pzb-pza)/(epsilon)+pza-1./3.*k_a*dpdk
    elif pmodel=='fastpt':
        if isinstance(zbar,np.ndarray) and zbar.size>1:
            pza,one_loop = P_a.get_matter_power(zbar,pmodel='fastpt',get_one_loop=True)
            dpdk = RectBivariateSpline(k_a,zbar,pza,kx=2,ky=1)(k_a,zbar,dx=1)
            dp = 47./21.*pza-1./3.*(k_a*(dpdk.T)).T+26./21.*one_loop
        else:
            pza,one_loop = P_a.get_matter_power(zbar,pmodel='fastpt',get_one_loop=True)
            pza = pza[:,0]
            one_loop = one_loop[:,0]
            dpdk = (InterpolatedUnivariateSpline(k_a,pza,ext=2,k=2).derivative(1))(k_a)
            dp = 47./21.*pza-1./3.*k_a*dpdk+26./21.*one_loop
    else:
        raise ValueError('invalid pmodel option \''+str(pmodel)+'\'')
    return dp,pza
