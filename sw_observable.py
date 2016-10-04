import cosmopie as cp
#short wavelength observable abstract class
#Methods must be implemented 
class SWObservable:
    #Takes a geo, dictionary of params, a cosmopie,dimension of vector returned
    def __init__(self,geo,params,survey_id,C,dim=0):
        self.geo = geo
        self.C = C
        self.params = params
        self.dim = dim
        self.survey_id = survey_id
    #get_O_I must be implemented by subclass, should return a numpy array with the first axis being z bins of geo
    #return array of vectors representing the observable itself
    def get_O_I(self):
        raise NotImplementedError('Subclasses of sw_observable must implement get_O_I')
    #get_dO_I_ddelta_bar must be implemented by subclass, should return a numpy array with the first axis being z bins of geo
    #return array of vectors representing derivative of observable wrt \bar{delta}
    def get_dO_I_ddelta_bar(self):
        raise NotImplementedError('Subclasses of sw_observable must implement get_dO_I_ddelta_bar')
    def get_survey_id(self):
        return self.survey_id
    def get_dimension(self):
        return self.dim
