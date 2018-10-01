"""Provides various lensing source distributions"""
#TODO handle photo z uncertainties
from __future__ import division,print_function,absolute_import
from builtins import range
import numpy as np
from algebra_utils import trapz2

class SourceDistribution(object):
    """generic input source distribution class"""
    def __init__(self,ps,zs,rs,C,params):
        """ inputs:
                ps: probability density of sources
                zs: z grid of sources
                rs: comoving distances corresponding to zs
                C: CosmoPie object
                params: parameters as needed
        """
        self.ps = ps
        self.zs = zs
        self.rs = rs
        self.C = C
        self.params = params
        #lock variables
        self.ps.flags['WRITEABLE'] = False
        self.zs.flags['WRITEABLE'] = False
        self.rs.flags['WRITEABLE'] = False

class GaussianZSource(SourceDistribution):
    """gaussian source distribution in z space"""
    def __init__(self,zs,rs,C,params,zbar=1.0,sigma=0.4):
        """inputs:
                zbar: mean z of sources
                sigma: width of source distribution
        """
        ps = np.exp(-(zs-zbar)**2/(2.*(sigma)**2))
        ps = _dz_to_dchi(ps,zs,rs,C,params)
        SourceDistribution.__init__(self,ps,zs,rs,C,params)

class ConstantZSource(SourceDistribution):
    """constant source distribution"""
    def __init__(self,zs,rs,C,params):
        """see SourceDistribution"""
        ps = np.zeros(zs.size)+1.
        ps = _dz_to_dchi(ps,zs,rs,C,params)
        SourceDistribution.__init__(self,ps,zs,rs,C,params)

class CosmoLikeZSource(SourceDistribution):
    """source distribution from cosmolike paper"""
    def __init__(self,zs,rs,C,params,alpha=1.24,beta=1.01,z0=0.51):
        """cosmolike uses alpha=1.3, beta=1.5, z0=0.56"""
        ps = zs**alpha*np.exp(-(zs/z0)**beta)
        ps = _dz_to_dchi(ps,zs,rs,C,params)
        SourceDistribution.__init__(self,ps,zs,rs,C,params)

class NZMatcherZSource(SourceDistribution):
    """source distribution using an NZMatcher object"""
    def __init__(self,zs,rs,C,params,nz_match):
        """nz_match: an NZMatcher object"""
        self.nz_match = nz_match
        ps = nz_match.get_dN_dzdOmega(zs)
        ps = _dz_to_dchi(ps,zs,rs,C,params)
        SourceDistribution.__init__(self,ps,zs,rs,C,params)

def get_source_distribution(smodel,zs,rs,C,params,ps=None,nz_matcher=None):
    """generate a source distribution from among the list of available source distributions"""
    if smodel=='gaussian':
        dist = GaussianZSource(zs,rs,C,params,zbar=params['zbar'],sigma=params['sigma'])
    elif smodel=='constant':
        dist = ConstantZSource(zs,rs,C,params)
    elif smodel=='cosmolike':
        dist = CosmoLikeZSource(zs,rs,C,params)
    elif smodel=='custom_z':
        if ps is not None and ps.size==zs.size:
            dist = SourceDistribution(_dz_to_dchi(ps,zs,rs,C,params),zs,rs,C,params)
        else:
            raise ValueError('input zs.size='+str(zs.size)+'and ps.size='+str(ps.size)+' do not match')
    elif smodel=='nzmatcher':
        dist = NZMatcherZSource(zs,rs,C,params,nz_matcher)
    else:
        raise ValueError('invalid smodel value\''+str(smodel)+'\'')
    return dist

def _dz_to_dchi(p_in,zs,rs,C,params):
    """put a z distribution into a distribution in comoving distance"""
    #z_min_dist = params['z_min_dist']
    #z_max_dist = params['z_max_dist']
######
#    ps = np.zeros(p_in.size)
#    for i in range(0,zs.size-1): #compensate for different bin sizes
#        ps[i] = p_in[i]/(rs[i+1]-rs[i])
#    ps[-1] = p_in[-1]/(C.D_comov(2*zs[-1]-zs[-2])-rs[-1]) #patch for last value
######
    ps = p_in*C.Ez(zs)/C.DH

    ps = ps/trapz2(ps,rs) #normalize galaxy probability distribution
    #ps = ps*(zs<=z_max_dist)*(zs>=z_min_dist) #cutoff outside dist limits
    return ps  
