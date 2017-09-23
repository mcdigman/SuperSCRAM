"""Abstract class for a long wavelength basis"""

class LWBasis(object): 
    def __init__(self,C):
        self.C = C

    def get_fisher(self):
        """get FisherMatrix object"""
        return NotImplementedError('subclasses of LWBasis must implement get_fisher')

    def get_size(self):
        """get number of basis elements"""
        return NotImplementedError('subclasses of LWBasis must implement get_size')

    def D_O_I_D_delta_alpha(self,geo,integrand):
        """get partial derivative of observable in the basis
        inputs:
            geo: a Geo object
            integrand: an array including r dependence to integrate over"""
        return NotImplementedError('subclasses of LWBasis must implement D_O_I_D_delta_alpha')
