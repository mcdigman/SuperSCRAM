"""get response of matter power spectrum to an overall density perturbation"""
from __future__ import division,print_function,absolute_import
from builtins import range
import numpy as np

def dp_ddelta(P_a,zbar,C,pmodel,epsilon=0.0001):
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
    dlnk = np.log(k_a[1])-np.log(k_a[0])
    #equations can be found in Chiang & Wagner arXiv:
    if pmodel=='linear':
        #TODO evaluate if redshift dependence (anywhere) needs to include Chiang & Wagner correction 4.17
        pza = P_a.get_matter_power(zbar,pmodel='linear')
        dpdlnk = np.gradient(pza,dlnk,axis=0)
        dp = 47./21.*pza-1./3.*dpdlnk
    elif pmodel=='halofit':
        #TODO insert prescription to handle the spike when this switches to using the linear matter power spectrum
        pza = P_a.get_matter_power(zbar,pmodel='halofit',const_pow_mult=1.)
        pzb = P_a.get_matter_power(zbar,pmodel='halofit',const_pow_mult=(1.+epsilon/C.get_sigma8())**2)
        pzc = P_a.get_matter_power(zbar,pmodel='halofit',const_pow_mult=(1.-epsilon/C.get_sigma8())**2)

        dpdlnk = np.gradient(pza,dlnk,axis=0)
        dp = 13./21.*C.get_sigma8()*(pzb-pzc)/(2.*epsilon)+pza-1./3.*dpdlnk
    elif pmodel=='fastpt':
        pza,one_loop = P_a.get_matter_power(zbar,pmodel='fastpt',get_one_loop=True)
        dpdlnk = np.gradient(pza,dlnk,axis=0)
        dp = 47./21.*pza-1./3.*dpdlnk+26./21.*one_loop
    else:
        raise ValueError('invalid pmodel option \''+str(pmodel)+'\'')

    if not isinstance(zbar,np.ndarray):
        dp = dp[:,0]
        pza = pza[:,0]

    return dp,pza
