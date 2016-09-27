import numpy as np
import cosmopie as cp
import defaults
import re
from warnings import warn
from sw_cov_mat import SWCovMat
from Dn import DNumberDensityObservable

class LWSurvey:
    def __init__(self,geos,survey_id,basis,C,ls = np.array([]),params=defaults.lw_survey_params,observable_list=defaults.lw_observable_list,dn_params=defaults.dn_params):
        self.geos = geos
        self.params = params
        self.C = C
        self.ls = ls
        self.survey_id = survey_id
        self.basis = basis
        self.ddelta_bar_ddelta_alpha_list = np.zeros(self.geos.size,dtype = object)
        for i in range(0,self.geos.size):
            self.ddelta_bar_ddelta_alpha_list[i] = self.basis.D_delta_bar_D_delta_alpha(self.geos[i],tomography=True)
        self.dn_params = defaults.dn_params
        self.observable_names = generate_observable_names(self.geos,observable_list,params['cross_bins'])
        self.observables = self.names_to_observables(self.observable_names)

    def get_ddelta_bar_ddelta_alpha_list(self):
        return self.ddelta_bar_ddelta_alpha_list

    def get_N_O_a(self):
        return self.observables.size

    def get_dO_a_ddelta_bar_list(self):
        dO_a_ddelta_bar_list = np.zeros(self.observables.size,dtype=object)
        for i in range(self.observables.size):
            dO_a_ddelta_bar_list[i] = self.observables[i].get_dO_a_ddelta_bar()
        return dO_a_ddelta_bar_list

    def fisher_accumulate(self,fisher_0):
        for i in range(0,self.get_N_O_a()):
            
            #print "fisher", np.linalg.eigvals(self.observables[i].get_F_alpha_beta())[0]
            fisher_0.add_fisher(self.observables[i].get_F_alpha_beta())

    def names_to_observables(self,names):
        observables = np.zeros(len(names.keys()),dtype=object)
        itr = 0 
        for key in names:
            if re.match('^d_number_density',key):
                bin1 = names[key]['bin1']
                bin2 = names[key]['bin2']
                observables[itr] = DNumberDensityObservable(np.array([bin1,bin2]),self.geos,self.dn_params,self.survey_id,self.C,self.ddelta_bar_ddelta_alpha_list)
            else:
                warn('unrecognized or unprocessable observable: \'',key,'\', skipping')
                observables[itr] = None
            itr+=1
        return observables
     
def generate_observable_names(geos,observable_list,cross_bins=defaults.lw_survey_params['cross_bins']):
    rbins = geos[0].rbins
    names = {}
    for name in observable_list:
        if re.match('^d_number_density',name):
            for i in range(0,rbins.shape[0]):
                r1 = rbins[i]
                if cross_bins:
                    for j in range(0,rbins.shape[0]):
                        r2 = rbins[j]
                        name_str = name+'_'+str(i)+'_'+str(j)
                        names[name_str] = {'bin1':i,'bin2':j}
                else:
                    name_str = name+'_'+str(i)+'_'+str(i)
                    names[name_str] = {'bin1':i,'bin2':i}
        else:
            warn('observable name \'',name,'\' unrecognized, ignoring')
    return names

if __name__=='__main__':
    from geo import rect_geo
    import sph_klim
    Theta1 = [np.pi/4.,np.pi/2.]
    Phi1 = [0,np.pi/3.]
    Theta2 = [np.pi/4.,np.pi/2.]
    Phi2 = [np.pi/3.,2.*np.pi/3.]
    d=np.loadtxt('camb_m_pow_l.dat')
    k=d[:,0]; P=d[:,1]
    C=cp.CosmoPie(k=k,P_lin=P)
    zs = np.array([0.1,0.8])
    ls = np.arange(2,500)
    geo1 = rect_geo(zs,Theta1,Phi1,C)
    geo2 = rect_geo(zs,Theta2,Phi2,C)
    k_cut = 0.005
    l_ceil = 100
    r_max = 4000.
    basis = sph_klim.sph_basis_k(r_max,C,k_cut,l_ceil)
    geos = np.array([geo1,geo2])
    lw_survey = LWSurvey(geos,'survey1',basis,ls,C=C)
    dO_a_ddelta_bar_list = lw_survey.get_dO_a_ddelta_bar_list()
    import matplotlib.pyplot as plt
    ax  = plt.subplot(111)
    fisher = basis.get_fisher()
    T = lw_survey.get_ddelta_bar_ddelta_alpha_list()[0][0]
    a1 = fisher.contract_covar(T,T)
    print a1
    lw_survey.fisher_accumulate(fisher)
    a2 = fisher.contract_covar(T,T)
    print a2
    #ax.loglog(np.diag(fisher.get_F_alpha_beta()))
    #ax.loglog(dO_a_ddelta_bar_list[0])
    #plt.xlabel('ls')
    #plt.legend(['dO_I_ddelta_bar'])
    #plt.show()

