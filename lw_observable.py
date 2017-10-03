"""
long wavelength observable abstract class
Methods must be implemented
"""
class LWObservable(object):
    def __init__(self,geos,params,survey_id,C):
        """abstract class for lw obserables.
            inputs:
                geos: a numpy array of geos
                params: a dictionary of params
                survey_id: an id from the associated LWSurvey
                C: a CosmoPie object"""
        self.geos = geos
        self.C = C
        self.params = params
        self.survey_id = survey_id
        #TODO formalize fisher_type
        self.fisher_type=True
    #TODO assess if actually needs to be mandatory for an LWObservable
    def get_dO_a_ddelta_bar(self):
        """
        get_dO_a_ddelta_bar must be implemented by subclass, should return a numpy array with the first axis being z bins of geo
        return array of vectors representing derivative of observable wrt \bar{delta}
        """
        raise NotImplementedError('Subclasses of LWObservable must implement get_dO_I_ddelta_bar')

    def get_fisher(self):
        """return a FisherMatrix object associated with the observable"""
        raise NotImplementedError('Subclasses of LWOobservable must implement get_fisher')

    def get_survey_id(self):
        """return the id string associated with the survey"""
        return self.survey_id
    #TODO maybe add get_perturbing vector
