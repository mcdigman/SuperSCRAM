"""
long wavelength observable abstract class
Methods must be implemented
"""
from __future__ import division,print_function,absolute_import
from builtins import range
class LWObservable(object):
    """abstract class for lw obserables."""
    def __init__(self,geos,params,survey_id,C):
        """inputs:
            geos: a numpy array of geos
            params: a dictionary of params
            survey_id: an id from the associated LWSurvey
            C: a CosmoPie object
            fisher_type is True, will get_fisher from this object, otherwise use get_perturbing_vector"""
        self.geos = geos
        self.C = C
        self.params = params
        self.survey_id = survey_id
        self.fisher_type = True

#    def get_dO_a_ddelta_bar(self):
#        """
#        get_dO_a_ddelta_bar must be implemented by subclass, should return a numpy array with the first axis being z bins of geo
#        return array of vectors representing derivative of observable wrt \bar{delta}
#        """
#        raise NotImplementedError('Subclasses of LWObservable must implement get_dO_I_ddelta_bar')

    def get_fisher(self):
        """return a FisherMatrix object associated with the observable"""
        raise NotImplementedError('Subclasses of LWOobservable must implement get_fisher')

    def get_survey_id(self):
        """return the id string associated with the survey"""
        return self.survey_id
