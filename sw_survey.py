import numpy as np
import cosmopie as cp
import defaults
import re
from warnings import warn
import lensing_observables as lo
from sw_cov_mat import SWCovMat
class SWSurvey:
    def __init__(self,geo,survey_id,ls = np.array([]),C = cp.CosmoPie(),params=defaults.sw_survey_params,observable_list=defaults.sw_observable_list,len_params=defaults.lensing_params):
        self.geo = geo
        self.params = params
        self.needs_lensing = params['needs_lensing']
        self.C = C
        self.ls = ls
        self.survey_id = survey_id
        if self.needs_lensing:
            self.len_pow = lo.LensingPowerBase(self.geo,self.ls,survey_id,len_params,C)
            self.len_params = len_params
        else:
            self.len_pow = None
            self.len_params = None

        self.observable_names = generate_observable_names(self.geo,observable_list,params['cross_bins'])
        self.observables = self.names_to_observables(self.observable_names)
    def get_N_O_I(self):
        return self.observables.size

    def get_O_I_list(self):
        O_I_list = np.zeros(self.observables.size,dtype=object)
        for i in range(0,self.observables.size):
            O_I_list[i] = self.observables[i].get_O_I()
        return O_I_list

    def get_dO_I_ddelta_bar_list(self):
        dO_I_ddelta_bar_list = np.zeros(self.observables.size,dtype=object)
        for i in range(self.observables.size):
            dO_I_ddelta_bar_list[i] = self.observables[i].get_dO_I_ddelta_bar()
        return dO_I_ddelta_bar_list

    def get_covars(self):
        cov_mats = np.zeros((self.get_N_O_I(),self.get_N_O_I()),dtype=object)
        for i in range(0,cov_mats.shape[0]):
            for j in range(0,cov_mats.shape[1]):
                cov = SWCovMat(self.observables[i],self.observables[j])
                cov_mats[i,j] = cov.get_total_covar()
        return cov_mats
    def names_to_observables(self,names):
        observables = np.zeros(len(names.keys()),dtype=object)
        itr = 0 
        for key in names:
            if self.params and re.match('^len',key):
                r1 = names[key]['r1']
                r2 = names[key]['r2']
                if re.match('^len_shear_shear',key):
                    observables[itr] = lo.ShearShearLensingObservable(self.len_pow,r1,r2,params=self.len_params)
                elif re.match('^len_galaxy_galaxy',key):
                    observables[itr] = lo.GalaxyGalaxyLensingObservable(self.len_pow,r1,r2,params=self.len_params)
                else:
                    warn('unrecognized or unprocessable observable: \'',key,'\', skipping')
                    observables[itr] = None
            else:
                warn('unrecognized or unprocessable observable: \'',key,'\', skipping')
                observables[itr] = None
            itr+=1
        return observables
            
def generate_observable_names(geo,observable_list,cross_bins=defaults.sw_survey_params['cross_bins']):
    rbins = geo.rbins
    names = {}
    for name in observable_list:
        if re.match('^len',name):
            for i in range(0,rbins.shape[0]):
                r1 = rbins[i]
                if cross_bins:
                    for j in range(0,rbins.shape[0]):
                        r2 = rbins[j]
                        name_str = name+'_'+str(i)+'_'+str(j)
                        names[name_str] = {'r1':r1,'r2':r2}
                else:
                    name_str = name+'_'+str(i)+'_'+str(i)
                    names[name_str] = {'r1':r1,'r2':r1}
        else:
            warn('observable name \'',name,'\' unrecognized, ignoring')
    return names

if __name__=='__main__':
    from geo import rect_geo
    Theta = [np.pi/4.,np.pi/2.]
    Phi = [0,np.pi/3.]
    d=np.loadtxt('camb_m_pow_l.dat')
    k=d[:,0]; P=d[:,1]
    C=cp.CosmoPie(k=k,P_lin=P)
    zbins = np.array([0.1,0.8])
    ls = np.arange(2,500)
    geo = rect_geo(zbins,Theta,Phi,C)
    sw_survey = SWSurvey(geo,'survey1',ls,C)
    O_I_list = sw_survey.get_O_I_list()
    dO_I_ddelta_bar_list = sw_survey.get_dO_I_ddelta_bar_list()
    import matplotlib.pyplot as plt
    ax  = plt.subplot(111)
    ax.loglog(ls,O_I_list[0])
    ax.loglog(ls,dO_I_ddelta_bar_list[0])
    plt.xlabel('ls')
    plt.legend(['O_I','dO_I_ddelta_bar'])
    plt.show()


