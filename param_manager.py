"""wrapper for parameters needed by MatterPower to prevent excessively long arguments"""
from __future__ import division,print_function,absolute_import
from builtins import range
class PowerParamManager(object):
    """wrapper for parameters needed by MatterPower to prevent excessively long arguments"""
    def __init__(self,matter_power,wmatcher,halofit,camb,fpt):
        """params needed by MatterPower,WMatcher,HalofitPK,camb,FASTPT"""
        self.matter_power = matter_power
        self.wmatcher = wmatcher
        self.halofit = halofit
        self.camb = camb
        self.fpt = fpt

    def copy(self):
        """deep copy self"""
        return PowerParamManager(self.matter_power.copy(),self.wmatcher.copy(),self.halofit.copy(),self.camb.copy(),self.fpt.copy())
