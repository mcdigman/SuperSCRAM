import cosmopie as cp
import numpy as np
from sw_observable import SWObservable
import defaults
import shear_power as sp
#Handle a lensing power observable as implemented in shear_power.py, get methods must be specified in subclasses
class LensingPowerBase():
    def __init__(self,geo,ls,params=defaults.lensing_params,C=cp.CosmoPie()):
        self.C=C
        self.geo = geo
        self.params = params
        self.ls = ls
        z_step = params['z_resolution']
        z_min = params['z_min_integral']

        zs = np.arange(z_step,geo.zbins[-1],z_step)
        self.C_pow = sp.shear_power(C.k,C,zs,ls,pmodel=params['pmodel_O'],P_in=C.P_lin)
        self.dC_ddelta = sp.shear_power(C.k,C,zs,ls,pmodel=params['pmodel_dO_ddelta'],P_in=C.P_lin)

#Generic lensing signal, subclass must define a function handle self.len_handle for which obserable to use
class LensingObservable(SWObservable):
    def __init__(self,len_pow,r1,r2,params=defaults.lensing_params):
        self.len_pow = len_pow
        self.r1 = r1
        self.r2= r2
        SWObservable.__init__(self,len_pow.geo,params,len_pow.C)
    def get_O_I(self):
        return self.len_handle(self.len_pow.C_pow,self.r1[1],self.r2[1],self.r1[0],self.r2[0]).Cll()
    def get_dO_I_ddelta_bar(self):
        return self.len_handle(self.len_pow.dC_ddelta,self.r1[1],self.r2[1],self.r1[0],self.r2[0]).Cll()

#Shear shear lensing signal
class ShearShearLensingObservable(LensingObservable):
    def __init__(self,len_pow,r1,r2,params=defaults.lensing_params):
        self.len_handle = sp.Cll_sh_sh
        LensingObservable.__init__(self,len_pow,r1,r2,params)

#galaxy galaxy lensing signal
class GalaxyGalaxyLensingObservable(LensingObservable):
    def __init__(self,len_pow,r1,r2,params=defaults.lensing_params):
        self.len_handle = sp.Cll_g_g
        LensingObservable.__init__(self,len_pow,r1,r2,params)
