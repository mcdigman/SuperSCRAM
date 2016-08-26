import cosmopie as cp
#long wavelength observable abstract class
#Methods must be implemented 
class LWObservable:
    #Takes a numpy array of geos, dictionary of params, and a cosmopie
    def __init__(self,geos,params,survey_id,C):
        self.geos = geos
        self.C = C
        self.params = params
        self.survey_id = survey_id
    #get_dO_a_ddelta_bar must be implemented by subclass, should return a numpy array with the first axis being z bins of geo
    #return array of vectors representing derivative of observable wrt \bar{delta}
    def get_dO_a_ddelta_bar(self):
        raise NotImplementedError('Subclasses of sw_observable must implement get_dO_I_ddelta_bar')
    def get_survey_id(self):
        return self.survey_id
