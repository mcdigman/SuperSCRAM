import cosmopie as cp
import numpy as np
from sw_observable import SWObservable
import defaults
import shear_power as sp
import lensing_weight as len_w
import power_parameter_response as ppr
#Handle a lensing power observable as implemented in shear_power.py, get methods must be specified in subclasses
class LensingPowerBase():
    def __init__(self,geo,ls,survey_id,C,cosmo_par_list,cosmo_par_epsilons,Cs_pert=None,params=defaults.lensing_params,log_param_derivs=np.array([]),ps=np.array([])):
        self.C=C
        self.geo = geo
        self.params = params
        self.cosmo_par_list = cosmo_par_list
        self.cosmo_par_epsilons = cosmo_par_epsilons
        self.ls = ls
        self.survey_id = survey_id
        self.Cs_pert = Cs_pert
 #       z_step = params['z_resolution']
 #       z_min = params['z_min_integral']
    
  #      zs = np.arange(z_min,geo.zs[-1],z_step)
        #TODO consider letting shear_power interact with geo directly
        #omega_s = self.geo.angular_area()/(4.*np.pi**2)
        omega_s = self.geo.angular_area()/(4.*np.pi)#removed square
        self.C_pow = sp.shear_power(C.k,C,self.geo.z_fine,ls,omega_s=omega_s,pmodel=params['pmodel_O'],mode='power',P_in=C.P_lin,params=params,ps=ps)
        self.dC_ddelta = sp.shear_power(C.k,C,self.geo.z_fine,ls,omega_s=omega_s,pmodel=params['pmodel_dO_ddelta'],mode='dc_ddelta',P_in=C.P_lin,params=params,ps=ps)
        self.dC_dpars = np.zeros((cosmo_par_list.size,2),dtype=object)
        if self.Cs_pert is None:
            self.Cs_pert = ppr.get_perturbed_cosmopies(C,cosmo_par_list,cosmo_par_epsilons,log_param_derivs) 
        for i in xrange(0,cosmo_par_list.size):
            self.dC_dpars[i,0] = sp.shear_power(self.Cs_pert[i,0].k,self.Cs_pert[i,0],self.geo.z_fine,ls,omega_s=omega_s,pmodel=params['pmodel_O'],mode='power',P_in=self.Cs_pert[i,0].P_lin,params=params)
            self.dC_dpars[i,1] = sp.shear_power(self.Cs_pert[i,1].k,self.Cs_pert[i,1],self.geo.z_fine,ls,omega_s=omega_s,pmodel=params['pmodel_O'],mode='power',P_in=self.Cs_pert[i,1].P_lin,params=params)
        

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
        self.q1_dpars = np.zeros((len_pow.cosmo_par_list.size,2),dtype=object)
        self.q2_dpars = np.zeros((len_pow.cosmo_par_list.size,2),dtype=object)
        for itr in xrange(0,len_pow.cosmo_par_list.size): 
            self.q1_dpars[itr,0] = self.q1_handle(self.len_pow.dC_dpars[itr,0],self.r1[0],self.r1[1])
            self.q1_dpars[itr,1] = self.q1_handle(self.len_pow.dC_dpars[itr,1],self.r1[0],self.r1[1])
            self.q2_dpars[itr,0] = self.q2_handle(self.len_pow.dC_dpars[itr,0],self.r2[0],self.r2[1])
            self.q2_dpars[itr,1] = self.q2_handle(self.len_pow.dC_dpars[itr,1],self.r2[0],self.r2[1])
        SWObservable.__init__(self,len_pow.geo,params,len_pow.survey_id,len_pow.C,dim=self.len_pow.ls.size)

    def get_O_I(self):
        return sp.Cll_q_q(self.len_pow.C_pow,self.q1_pow,self.q2_pow).Cll()

    def get_dO_I_ddelta_bar(self):
        return sp.Cll_q_q(self.len_pow.dC_ddelta,self.q1_dC,self.q2_dC).Cll_integrand()

    def get_dO_I_dpars(self):
        dO_dpars = np.zeros((self.get_dimension(),self.len_pow.cosmo_par_list.size))
        for itr in xrange(0,dO_dpars.shape[1]):
            Cll_low = sp.Cll_q_q(self.len_pow.dC_dpars[itr,0],self.q1_dpars[itr,0],self.q2_dpars[itr,0]).Cll()
            Cll_high = sp.Cll_q_q(self.len_pow.dC_dpars[itr,1],self.q1_dpars[itr,1],self.q2_dpars[itr,1]).Cll()
            dO_dpars[:,itr] = (Cll_high-Cll_low)/(2.*self.len_pow.cosmo_par_epsilons[itr])
        return dO_dpars

#Shear shear lensing signal
class ShearShearLensingObservable(LensingObservable):
    def __init__(self,len_pow,r1,r2,params=defaults.lensing_params):
        self.q1_handle = len_w.q_shear
        self.q2_handle = len_w.q_shear
        LensingObservable.__init__(self,len_pow,r1,r2,params)

#galaxy galaxy lensing signal
class GalaxyGalaxyLensingObservable(LensingObservable):
    def __init__(self,len_pow,r1,r2,params=defaults.lensing_params):
        self.q1_handle = len_w.q_num
        self.q2_handle = len_w.q_num
        LensingObservable.__init__(self,len_pow,r1,r2,params)
