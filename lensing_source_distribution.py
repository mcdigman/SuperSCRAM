"""Provides various lensing source distributions"""
#TODO handle photo z uncertainties
import numpy as np
import defaults
from algebra_utils import trapz2

class SourceDistribution(object):
    def __init__(self,ps,zs,chis,C,params=defaults.lensing_params):
        """generic input source distribution class"""
        self.ps = ps
        self.zs = zs
        self.chis = chis
        self.C = C
        self.params=params

class GaussianZSource(SourceDistribution):
    def __init__(self,zs,chis,C,zbar=1.0,sigma=0.4,params=defaults.lensing_params):
        """gaussian source distribution in z space
            inputs:
                zbar: mean z of sources
                sigma: width of source distribution
        """
        ps = np.exp(-(zs-zbar)**2/(2.*(sigma)**2))
        ps = dz_to_dchi(ps,zs,chis,C,params)
        SourceDistribution.__init__(self,ps,zs,chis,C,params)

class ConstantZSource(SourceDistribution):
    def __init__(self,zs,chis,C,params=defaults.lensing_params):
        """constant source distribution"""
        ps = np.zeros(zs.size)+1.
        ps = dz_to_dchi(ps,zs,chis,C,params)
        SourceDistribution.__init__(self,ps,zs,chis,C,params)

class CosmoLikeZSource(SourceDistribution):
    def __init__(self,zs,chis,C,alpha=1.24,beta=1.01,z0=0.51,params=defaults.lensing_params):
        """source distribution from cosmolike paper
        cosmolike uses alpha=1.3, beta=1.5, z0=0.56"""
        ps = zs**alpha*np.exp(-(zs/z0)**beta)
        ps = dz_to_dchi(ps,zs,chis,C,params)
        SourceDistribution.__init__(self,ps,zs,chis,C,params)

class NZMatcherZSource(SourceDistribution):
    def __init__(self,zs,chis,C,nz_match,params=defaults.lensing_params):
        """source distribution using an NZMatcher object"""
        ps = nz_match.get_dN_dzdOmega(zs)
        ps = dz_to_dchi(ps,zs,chis,C,params)
        SourceDistribution.__init__(self,ps,zs,chis,C,params)

def get_source_distribution(smodel,zs,chis,C,params=defaults.lensing_params,ps=np.array([]),nz_matcher=None):
    """generate a source distribution from among the list of available source distributions"""
    if smodel == 'gaussian':
        dist = GaussianZSource(zs,chis,C,zbar=params['zbar'],sigma=params['sigma'])
    elif smodel == 'constant':
        dist = ConstantZSource(zs,chis,C,params)
    elif smodel == 'cosmolike':
        dist = CosmoLikeZSource(zs,chis,C,params)
    elif smodel=='custom_z':
        if ps.size == zs.size:
            dist = SourceDistribution(dz_to_dchi(ps,zs,chis,C,params),zs,chis,C,params)
        else:
            raise ValueError('input zs.size='+str(zs.size)+'and ps.size='+str(ps.size)+' do not match')
    elif smodel=='nzmatcher':
        dist = NZMatcherZSource(zs,chis,C,nz_matcher,params)
    else:
        raise ValueError('invalid smodel value\''+str(smodel)+'\'')
    return dist

def dz_to_dchi(p_in,zs,chis,C,params):
    """put a z distribution into a distribution in comoving distance"""
    z_min_dist = params['z_min_dist']
    z_max_dist = params['z_max_dist']
    ps = np.zeros(p_in.size)
    for i in xrange(0,zs.size-1): #compensate for different bin sizes
        ps[i] = p_in[i]/(chis[i+1]-chis[i])
    ps[-1] = p_in[-1]/(C.D_comov(2*zs[-1]-zs[-2])-chis[-1]) #patch for last value
    ps = ps*(zs<=z_max_dist)*(zs>=z_min_dist) #cutoff outside dist limits
    return ps/trapz2(ps,chis) #normalize galaxy probability distribution


