"""
Abstract class for a short wavelength observable
"""

class SWObservable(object):
    """short wavelength observable abstract class"""
    def __init__(self,survey_id,dim=0):
        """
            inputs:
                survey_id: an id for the survey this observable is associated with
                dim: the number of dimensions for this observable, optional if subclass overrides get_dimension
        """
        self.dim = dim
        self.survey_id = survey_id

    #TODO not actually needed by anythin
    def get_O_I(self):
        """Must be implemented by subclass, should return a numpy array with the first axis being z bins
        return array of vectors representing the observable itself"""
        raise NotImplementedError('Subclasses of sw_observable must implement get_O_I')

    def get_dO_I_ddelta_bar(self):
        """Must be implemented by subclass, should return a numpy array with the first axis being z bins
        return array of vectors representing derivative of observable wrt \bar{delta}"""
        raise NotImplementedError('Subclasses of sw_observable must implement get_dO_I_ddelta_bar')

    def get_survey_id(self):
        """Get survey_id associated with the observable"""
        return self.survey_id

    def get_dimension(self):
        """Get number of dimensions of observable"""
        return self.dim
