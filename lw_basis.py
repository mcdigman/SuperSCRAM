"""Abstract class for a long wavelength basis"""
from __future__ import division,print_function,absolute_import
from builtins import range

class LWBasis(object):
    """Abstract class for a long wavelength basis"""
    def __init__(self,C):
        """create the long wavelength basis"""
        self.C = C

    def get_fisher(self,initial_state,silent):
        """get FisherMatrix object,initial_state is initial_state for FisherMatrix"""
        raise NotImplementedError('subclasses of LWBasis must implement get_fisher')

    def get_size(self):
        """get number of basis elements"""
        raise NotImplementedError('subclasses of LWBasis must implement get_size')

    def get_dO_I_ddelta_alpha(self,geo,integrand):
        """get partial derivative of observable in the basis
        inputs:
            geo: a Geo object
            integrand: an array including r dependence to integrate over"""
        raise NotImplementedError('subclasses of LWBasis must implement get_dO_I_ddelta_alpha')
