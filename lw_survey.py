"""Class for handling a long wavelength survey, used for testing mitigation strategies"""
import numpy as np
import cosmopie as cp
import defaults
import re
from warnings import warn
from Dn import DNumberDensityObservable

class LWSurvey(object):
    def __init__(self,geos,survey_id,basis,C,params=defaults.lw_survey_params,observable_list=defaults.lw_observable_list,dn_params=defaults.dn_params):
        """handle getting long wavelength observables and their fisher matrices for mitigation
            inputs:
                geos: an array of Geo objects, fo the survey windows of different long wavelength surveys
                survey_id: an id for the survey
                basis: an LWBasis object
                C: a Cosmopie object
                params: necessary parameters
                observable_list: a list of observables required
                dn_params: parameters needed by DNumberDensityObservable specifically
        """ 
        print "lw_survey: began initializing long wavelength survey: "+str(survey_id)
        self.geos = geos
        self.params = params
        self.C = C
        self.survey_id = survey_id
        self.basis = basis
        self.dn_params = dn_params
        self.observable_names = generate_observable_names(observable_list)
        self.observables = self.names_to_observables(self.observable_names)
        print "lw_survey: finished initializing long wavelength survey: "+str(survey_id)

    def get_N_O_a(self):
        """get number of long wavelength observables"""
        return self.observables.size

    def get_dO_a_ddelta_bar_list(self):
        """get list of arrays of long wavelength observables"""
        dO_a_ddelta_bar_list = np.zeros(self.observables.size,dtype=object)
        for i in xrange(self.observables.size):
            dO_a_ddelta_bar_list[i] = self.observables[i].get_dO_a_ddelta_bar()
        return dO_a_ddelta_bar_list

    def fisher_accumulate(self,fisher_0):
        """add the fisher matrices for all available lw observables to the FisherMatrix object fisher_0"""
        for i in xrange(0,self.get_N_O_a()): 
            if self.observables[i].fisher_type:
                fisher_0.add_fisher(self.observables[i].get_fisher())
            else:
                fisher_0.perturb_fisher(self.observables[i].get_perturbing_vector())

            #fisher_0.internal_mat = self.observables[i].add_fisher(fisher_0.internal_mat)

    def get_total_rank(self):
        """get the total rank of perturbations that will be added to the SSC contribution, for testing interlace theorems"""
        rank = 0
        for itr in xrange(0,self.observables.size):
            if not self.observables[itr] is None:
                rank+=self.observables[itr].get_rank()
        return rank

    def names_to_observables(self,names):
        """get the list of long wavelength observables corresponding to a given dictionary of names
            only currently recognized name is d_number_density"""
        observables = np.zeros(len(names.keys()),dtype=object)
        itr = 0 
        for key in names:
            if re.match('^d_number_density',key):
                observables[itr] = DNumberDensityObservable(self.geos,self.dn_params,self.survey_id,self.C,self.basis,defaults.nz_params_wfirst,defaults.nz_params_lsst)
            else:
                warn('unrecognized or unprocessable observable: \'',key,'\', skipping')
                observables[itr] = None
            itr+=1
        return observables
     
def generate_observable_names(observable_list):
    """get a dictionary of names from the given list of names
        can include parameters but names_to_observables does not currently need that functionality"""
    names = {}
    for name in observable_list:
        if re.match('^d_number_density',name):
            names[name] = {}
        else:
            warn('observable name \'',name,'\' unrecognized, ignoring')
    return names

if __name__=='__main__':
    from geo import RectGeo
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
    geo1 = RectGeo(zs,Theta1,Phi1,C)
    geo2 = RectGeo(zs,Theta2,Phi2,C)
    k_cut = 0.005
    l_ceil = 100
    r_max = 4000.
    basis = sph_klim.SphBasisK(r_max,C,k_cut,l_ceil)
    geos = np.array([geo1,geo2])
    lw_survey = LWSurvey(geos,'survey1',basis,ls,C=C)
    dO_a_ddelta_bar_list = lw_survey.get_dO_a_ddelta_bar_list()
    import matplotlib.pyplot as plt
    ax  = plt.subplot(111)
    fisher = basis.get_fisher()
    T = lw_survey.get_ddelta_bar_ddelta_alpha_list()[0][0]
    a1 = fisher.project_covar(T)
    print a1
    lw_survey.fisher_accumulate(fisher)
    a2 = fisher.project_covar(T)
    print a2
    #ax.loglog(np.diag(fisher.get_fisher()))
    #ax.loglog(dO_a_ddelta_bar_list[0])
    #plt.xlabel('ls')
    #plt.legend(['dO_I_ddelta_bar'])
    #plt.show()


