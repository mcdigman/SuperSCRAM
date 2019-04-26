"""
contains various weight functions used by lensing observables
"""
from __future__ import division,print_function,absolute_import
from builtins import range
import numpy as np
from scipy.integrate import cumtrapz
from algebra_utils import trapz2

class QWeight(object):
    """abstract class for weight function"""
    def __init__(self,rs,qs,z_min=0.,z_max=np.inf):
        """ inputs:
                rs: an array of comoving distances
                qs: the weight function at each comoving distance
                z_min: the minimum redshift to use in integrations
                z_max: the maximum redshift to use in integrations
        """
        #assert z_min<5.
        #assert z_max<5. or z_max==np.inf
        self.z_min = z_min
        self.z_max = z_max
        self.rs = rs
        self.qs = qs

class QShear(QWeight):
    """weight function for a shear observable as in ie arXiv:1302.2401v2"""
    def __init__(self,sp,z_min=0.,z_max=np.inf,mult=1.):
        """ inputs:
                sp: a ShearPower object
                mult: a constant multiplier for the weight function
                z_min,z_max: minimum and maximum comoving distances
        """
        qs = 3./2.*(sp.C.H0/sp.C.c)**2*sp.C.Omegam*sp.r_As/sp.sc_as*_gs(sp,z_min,z_max)*mult
        QWeight.__init__(self,sp.rs,qs,z_min,z_max)

def _gs(sp,z_min=0.,z_max=np.inf):
    """helper function for QShear"""
    g_vals = np.zeros(sp.n_z)
    low_mask = (sp.zs>=z_min)*1. #so only integrate from max(z,z_min)
    high_mask = (sp.zs<=z_max)*1. #so only integrate to min(z,z_max)
    ps_mask = sp.ps*high_mask*low_mask
    ps_norm = ps_mask/trapz2(ps_mask,sp.rs)
    #norm1 = trapz2(ps_mask,sp.rs)
    #norm2 = trapz2(ps_mask[(sp.zs>=z_min) & (sp.zs<=z_max)],sp.rs[(sp.zs>=z_min) & (sp.zs<=z_max)])

    if sp.C.Omegak==0.0:
        g_vals = -cumtrapz(ps_norm[::-1],sp.rs[::-1],initial=0.)[::-1]+sp.rs*cumtrapz((ps_norm/sp.rs)[::-1],sp.rs[::-1],initial=0.)[::-1]
    else:
        for i in range(0,sp.n_z):
            if z_max<sp.zs[i]:
                break
            if sp.C.Omegak>0.0:
                sqrtK = np.sqrt(sp.C.K)
                g_vals[i] = trapz2(ps_norm[i:sp.n_z]*sp.rs[i]*1./sqrtK*(1./np.tan(sqrtK*sp.rs[i])-1./np.tan(sqrtK*sp.rs[i:sp.n_z])),sp.rs[i:sp.n_z])
            else:
                sqrtK = np.sqrt(abs(sp.C.K))
                g_vals[i] = trapz2(ps_norm[i:sp.n_z]*sp.rs[i]*1./sqrtK*(1./np.tanh(sqrtK*sp.rs[i])-1./np.tanh(sqrtK*sp.rs[i:sp.n_z])),sp.rs[i:sp.n_z])
    return g_vals

class QMag(QShear):
    """weight function for magnification, just shear weight function*2 now"""
    def __init__(self,sp,z_min=0.,z_max=np.inf):
        """see QShear"""
        QShear.__init__(self,sp,z_min,z_max,mult=2.)


class QK(QShear):
    """weight function for convergence, just shear weight function now"""
    def __init__(self,sp,z_min=0.,z_max=np.inf):
        """see QShear"""
        QShear.__init__(self,sp,z_min,z_max,mult=1.)

class QNum(QWeight):
    """weight function for galaxy lensing, as in ie arXiv:1302.2401v2"""
    def __init__(self,sp,z_min=0.,z_max=np.inf):
        """see QShear"""
        q = np.zeros((sp.n_z,sp.n_l))
        self.b = _bias(sp)
        for i in range(0,sp.n_z):
            if sp.zs[i]>z_max:
                break
            elif sp.zs[i]<z_min:
                continue
            else:
                q[i] = sp.ps[i]*self.b[:,i]
        QWeight.__init__(self,sp.rs,q,z_min,z_max)

def _bias(sp):
    """bias as a function of k and r, used in galaxy lensing helper for QNUM"""
    return np.sqrt(sp.p_gg_use/sp.p_dd_use)
