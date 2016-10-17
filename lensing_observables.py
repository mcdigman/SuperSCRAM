import cosmopie as cp
import numpy as np
from sw_observable import SWObservable
import defaults
import shear_power as sp
import lensing_weight as len_w
#Handle a lensing power observable as implemented in shear_power.py, get methods must be specified in subclasses
class LensingPowerBase():
    def __init__(self,geo,ls,survey_id,C,params=defaults.lensing_params):
        self.C=C
        self.geo = geo
        self.params = params
        self.ls = ls
        self.survey_id = survey_id
 #       z_step = params['z_resolution']
 #       z_min = params['z_min_integral']
    
  #      zs = np.arange(z_min,geo.zs[-1],z_step)
        #TODO consider letting shear_power interact with geo directly
        omega_s = self.geo.angular_area()/(4.*np.pi**2)
        self.C_pow = sp.shear_power(C.k,C,self.geo.z_fine,ls,omega_s=omega_s,pmodel=params['pmodel_O'],P_in=C.P_lin,params=params)
        self.dC_ddelta = sp.shear_power(C.k,C,self.geo.z_fine,ls,omega_s=omega_s,pmodel=params['pmodel_dO_ddelta'],P_in=C.P_lin,params=params)
        

#Generic lensing signal, subclass must define a function handle self.len_handle for which obserable to use
class LensingObservable(SWObservable):
    def __init__(self,len_pow,r1,r2,params=defaults.lensing_params):
        self.len_pow = len_pow
        self.r1 = r1
        self.r2= r2
        self.q1_pow = self.q1_handle(self.len_pow.C_pow,self.r1[0],self.r1[1])
        self.q2_pow = self.q2_handle(self.len_pow.C_pow,self.r2[0],self.r2[1])
        self.q1_dC = self.q1_handle(self.len_pow.dC_ddelta,self.r1[0],self.r1[1])
        self.q2_dC = self.q2_handle(self.len_pow.dC_ddelta,self.r2[0],self.r2[1])
        SWObservable.__init__(self,len_pow.geo,params,len_pow.survey_id,len_pow.C,dim=self.len_pow.ls.size)
    def get_O_I(self):
        return sp.Cll_q_q(self.len_pow.C_pow,self.q1_pow,self.q2_pow).Cll()
        #return self.len_handle(self.len_pow.C_pow,self.r1[1],self.r2[1],self.r1[0],self.r2[0]).Cll()
    def get_dO_I_ddelta_bar(self):
        #return self.len_handle(self.len_pow.dC_ddelta,self.r1[1],self.r2[1],self.r1[0],self.r2[0]).Cll()
        return sp.Cll_q_q(self.len_pow.dC_ddelta,self.q1_dC,self.q2_dC).Cll_integrand()

#Shear shear lensing signal
class ShearShearLensingObservable(LensingObservable):
    def __init__(self,len_pow,r1,r2,params=defaults.lensing_params):
        #self.len_handle = sp.Cll_sh_sh
        self.q1_handle = len_w.q_shear
        self.q2_handle = len_w.q_shear
        LensingObservable.__init__(self,len_pow,r1,r2,params)

#galaxy galaxy lensing signal
class GalaxyGalaxyLensingObservable(LensingObservable):
    def __init__(self,len_pow,r1,r2,params=defaults.lensing_params):
        #self.len_handle = sp.Cll_g_g
        self.q1_handle = len_w.q_num
        self.q2_handle = len_w.q_num
        LensingObservable.__init__(self,len_pow,r1,r2,params)
