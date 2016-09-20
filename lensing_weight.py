import numpy as np

class q_weight:
    def __init__(self,chis,qs,chi_min=0.,chi_max=np.inf):
        self.chi_min = chi_min
        self.chi_max = chi_max
        self.qs = qs

class q_shear(q_weight):
    def __init__(self,sp,chi_min=0.,chi_max=np.inf,mult=1.):
        qs = 3./2.*(sp.C.H0/sp.C.c)**2*sp.omegas*sp.chi_As/sp.sc_as*self.gs(sp,chi_max=chi_max,chi_min=chi_min)*mult
        q_weight.__init__(self,sp.chis,qs,chi_min=chi_min,chi_max=chi_max) 

    def gs(self,sp,chi_max=np.inf,chi_min=0):
        g_vals = np.zeros(sp.n_z)
        low_mask = (sp.chis>=chi_min)*1. #so only integrate from max(chi,chi_min)
        high_mask = (sp.chis<=chi_max)*1. #so only integrate from max(chi,chi_min)
        print chi_min,chi_max
        
        ps_norm = sp.ps*high_mask*low_mask/np.trapz(sp.ps*low_mask*high_mask,sp.chis) #TODO check normalization
	for i in range(0,sp.n_z):
            if chi_max<sp.chis[i]:
                break
            if sp.C.Omegak==0.0:
                g_vals[i] =np.trapz(low_mask[i:sp.n_z]*ps_norm[i:sp.n_z]*(sp.chis[i:sp.n_z]-sp.chis[i])/sp.chis[i:sp.n_z],sp.chis[i:sp.n_z])
            elif sp.C.Omegak>0.0: #TODO handle curvature
                sqrtK = np.sqrt(sp.C.K)
                g_vals[i] =np.trapz(low_mask[i:sp.n_z]*ps_norm[i:sp.n_z]*sp.chis[i]*1./sqrtK(1./np.tan(sqrtK*sp.chis[i])-1./np.tan(sqrtK*sp.chis[i:sp.n_z])),sp.chis[i:sp.n_z])
            else:
                sqrtK = np.sqrt(abs(sp.C.K))
                g_vals[i] =np.trapz(low_mask[i:sp.n_z]*ps_norm[i:sp.n_z]*sp.chis[i]*1./sqrtK(1./np.tanh(sqrtK*sp.chis[i])-1./np.tanh(sqrtK*sp.chis[i:sp.n_z])),sp.chis[i:sp.n_z])
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
                q_weight.__init__(self,sp.chis,q,chi_min=chi_min,chi_max=chi_max)
    def bias(self,sp):
        return np.sqrt(sp.p_gg_use/sp.p_dd_use)
