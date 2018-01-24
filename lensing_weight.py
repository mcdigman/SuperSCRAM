"""
contains various weight functions used by lensing observables
"""
import numpy as np
from algebra_utils import trapz2

class QWeight(object):
    """abstract class for weight function"""
    def __init__(self,chis,qs,chi_min=0.,chi_max=np.inf):
        """ inputs:
                chis: an array of comoving distances
                qs: the weight function at each comoving distance
                chi_min: the minimum comoving distance to use in integrations
                chi_max: the maximum comoving distance to use in integrations
        """
        self.chi_min = chi_min
        self.chi_max = chi_max
        self.chis = chis
        self.qs = qs

class QShear(QWeight):
    """weight function for a shear observable as in ie arXiv:1302.2401v2"""
    def __init__(self,sp,chi_min=0.,chi_max=np.inf,mult=1.):
        """ inputs:
                sp: a ShearPower object
                mult: a constant multiplier for the weight function
                chi_min,chi_max: minimum and maximum comoving distances
        """
        qs = 3./2.*(sp.C.H0/sp.C.c)**2*sp.C.Omegam*sp.chi_As/sp.sc_as*self.gs(sp,chi_max=chi_max,chi_min=chi_min)*mult
        QWeight.__init__(self,sp.chis,qs,chi_min=chi_min,chi_max=chi_max)

    def gs(self,sp,chi_max=np.inf,chi_min=0):
        """helper function for QShear"""
        g_vals = np.zeros(sp.n_z)
        low_mask = (sp.chis>=chi_min)*1. #so only integrate from max(chi,chi_min)
        high_mask = (sp.chis<=chi_max)*1. #so only integrate from max(chi,chi_min)

        ps_norm = sp.ps*high_mask*low_mask/trapz2(sp.ps*low_mask*high_mask,sp.chis) #TODO check normalization
        for i in xrange(0,sp.n_z):
            if chi_max<sp.chis[i]:
                break
            if sp.C.Omegak==0.0:
                g_vals[i] = trapz2(low_mask[i:sp.n_z]*ps_norm[i:sp.n_z]*(sp.chis[i:sp.n_z]-sp.chis[i])/sp.chis[i:sp.n_z],sp.chis[i:sp.n_z])
            elif sp.C.Omegak>0.0: #TODO handle curvature
                sqrtK = np.sqrt(sp.C.K)
                g_vals[i] = trapz2(low_mask[i:sp.n_z]*ps_norm[i:sp.n_z]*sp.chis[i]*1./sqrtK(1./np.tan(sqrtK*sp.chis[i])-1./np.tan(sqrtK*sp.chis[i:sp.n_z])),sp.chis[i:sp.n_z])
            else:
                sqrtK = np.sqrt(abs(sp.C.K))
                g_vals[i] = trapz2(low_mask[i:sp.n_z]*ps_norm[i:sp.n_z]*sp.chis[i]*1./sqrtK(1./np.tanh(sqrtK*sp.chis[i])-1./np.tanh(sqrtK*sp.chis[i:sp.n_z])),sp.chis[i:sp.n_z])
        return g_vals

class QMag(QShear):
    """weight function for magnification, just shear weight function*2 now"""
    def __init__(self,sp,chi_max=np.inf,chi_min=0.):
        """see QShear"""
        QShear.__init__(self,sp,chi_max=chi_max,chi_min=chi_min,mult=2.)


class QK(QShear):
    """weight function for convergence, just shear weight function now"""
    def __init__(self,sp,chi_max=np.inf,chi_min=0.):
        """see QShear"""
        QShear.__init__(self,sp,chi_max=chi_max,chi_min=chi_min,mult=1.)

class QNum(QWeight):
    """weight function for galaxy lensing, as in ie arXiv:1302.2401v2"""
    def __init__(self,sp,chi_max=np.inf,chi_min=0.):
        """see QShear"""
        q = np.zeros((sp.n_z,sp.n_l))
        self.b = self.bias(sp)
        for i in xrange(0,sp.n_z):
            if sp.chis[i]>chi_max:
                break
            elif sp.chis[i]<chi_min:
                continue
            else:
                q[i] = sp.ps[i]*self.b[:,i]
        QWeight.__init__(self,sp.chis,q,chi_min=chi_min,chi_max=chi_max)

    def bias(self,sp):
        """bias as a function of k and r, used in galaxy lensing"""
        return np.sqrt(sp.p_gg_use/sp.p_dd_use)
