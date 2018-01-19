"""Abstract class for a long wavelength basis"""

class LWBasis(object):
    """Abstract class for a long wavelength basis"""
    def __init__(self,C):
        """create the long wavelength basis"""
        self.C = C

    def get_fisher(self,initial_state):
        """get FisherMatrix object,initial_state is initial_state for FisherMatrix"""
        raise NotImplementedError('subclasses of LWBasis must implement get_fisher')

    def get_size(self):
        """get number of basis elements"""
        raise NotImplementedError('subclasses of LWBasis must implement get_size')

    def D_O_I_D_delta_alpha(self,geo,integrand):
        """get partial derivative of observable in the basis
        inputs:
            geo: a Geo object
            integrand: an array including r dependence to integrate over"""
        raise NotImplementedError('subclasses of LWBasis must implement D_O_I_D_delta_alpha')
