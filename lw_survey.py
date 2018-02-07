"""Class for handling a long wavelength survey, used for testing mitigation strategies"""

import re

from warnings import warn
import numpy as np
from Dn import DNumberDensityObservable

class LWSurvey(object):
    """handle getting long wavelength observables and their fisher matrices for mitigation"""
    def __init__(self,geos,survey_id,basis,C,params,observable_list,param_list):
        """ inputs:
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
        self.param_list = param_list

        self.observable_names = generate_observable_names(observable_list,self.param_list)
        self.observables = self.names_to_observables(self.observable_names)
        print "lw_survey: finished initializing long wavelength survey: "+str(survey_id)

    def get_N_O_a(self):
        """get number of long wavelength observables"""
        return self.observables.size

#    def get_dO_a_ddelta_bar_list(self):
#        """get list of arrays of long wavelength observables"""
#        dO_a_ddelta_bar_list = np.zeros(self.observables.size,dtype=object)
#        for i in xrange(self.observables.size):
#            dO_a_ddelta_bar_list[i] = self.observables[i].get_dO_a_ddelta_bar()
#        return dO_a_ddelta_bar_list

    def fisher_accumulate(self,fisher_0):
        """add the fisher matrices for all available lw observables to the FisherMatrix object fisher_0"""
        for i in xrange(0,self.get_N_O_a()):
            if self.observables[i].fisher_type:
                fisher_0.add_fisher(self.observables[i].get_fisher())
            else:
                vs,sigma2s = self.observables[i].get_perturbing_vector()
                fisher_0.perturb_fisher(vs,sigma2s)

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
                params = names[key]
                observables[itr] = DNumberDensityObservable(self.geos,params['dn_params'],self.survey_id,self.C,self.basis,params['n_params'],params['mf_params'])
            else:
                warn('unrecognized or unprocessable observable: \'',key,'\', skipping')
                observables[itr] = None
            itr+=1
        return observables

def generate_observable_names(observable_list,param_list):
    """get a dictionary of full names from the given list of names
        param_list, of dicts of parameters as needed for names_to_observables"""
    names = {}

    for itr in xrange(0,observable_list.size):
        name = observable_list[itr]
        if re.match('^d_number_density',name):
            names[name] = param_list[itr]
        else:
            warn('observable name \'',name,'\' unrecognized, ignoring')
    return names
