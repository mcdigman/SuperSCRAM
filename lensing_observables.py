"""
Contains lensing observable classes and functions
"""
import numpy as np
from sw_observable import SWObservable
import defaults
import shear_power as sp
import lensing_weight as len_w
import power_parameter_response as ppr
class LensingPowerBase(object):
    def __init__(self,geo,ls,survey_id,C,cosmo_par_list,cosmo_par_epsilons,params=defaults.lensing_params,log_param_derivs=np.array([]),ps=np.array([])):
        """Shared object to generate and store things that will be needed my multiple LensingObservable objects, such as the lensing weight functions
            inputs:
                geo: a Geo object for the geometry
                ls: the set of l bins to compute the sw observables on
                survey_id: an identifier for the SWSurvey this object is associated with
                cosmo_par_list: a list of the names of cosmological parameters to compute \partial{O_I}\partial{\Theta_i} with respect to
                cosmo_par_epsilons: array of epsilons to use when calculating partial derivatives specified by cosmo_par_list
                log_param_derivs: a boolean array with dimension matching cosmo_par_list, will do \partial{O_I}\partial{ln(\Theta_i)} for any True element. Optional.
                ps: an in input lensing source distribution. Optional.
                params: parameters
        """
        self.C=C
        self.geo = geo
        self.params = params
        self.cosmo_par_list = cosmo_par_list
        self.cosmo_par_epsilons = cosmo_par_epsilons
        self.ls = ls
        self.survey_id = survey_id

        #omega_s = self.geo.angular_area()/(4.*np.pi**2)
        omega_s = self.geo.angular_area()/(4.*np.pi)#removed square
        self.C_pow = sp.ShearPower(C,self.geo.z_fine,ls,omega_s=omega_s,pmodel=params['pmodel_O'],mode='power',params=params,ps=ps)
        self.dC_ddelta = sp.ShearPower(C,self.geo.z_fine,ls,omega_s=omega_s,pmodel=params['pmodel_dO_ddelta'],mode='dc_ddelta',params=params,ps=ps)

        self.dC_dpars = np.zeros((cosmo_par_list.size,2),dtype=object)
        self.Cs_pert = ppr.get_perturbed_cosmopies(C,cosmo_par_list,cosmo_par_epsilons,log_param_derivs)

        for i in xrange(0,cosmo_par_list.size):
            self.dC_dpars[i,0] = sp.ShearPower(self.Cs_pert[i,0],self.geo.z_fine,ls,omega_s=omega_s,pmodel=params['pmodel_O'],mode='power',params=params)
            self.dC_dpars[i,1] = sp.ShearPower(self.Cs_pert[i,1],self.geo.z_fine,ls,omega_s=omega_s,pmodel=params['pmodel_O'],mode='power',params=params)

#TODO observables should know their name
class LensingObservable(SWObservable):
    def __init__(self,len_pow,r1,r2,q1_handle,q2_handle):
        """Generic lensing observable, subclass only need to define a function handle self.len_handle for which obserable to use
            inputs:
                len_pow: a LensingPowerBase object
                r1,r2: [min r, max r] for the lensing weights to integrate over
                q1_handle,q2_handle: function handles for QWeight objects, ie len_w.QShear
        """
        self.len_pow = len_pow
        self.r1 = r1
        self.r2= r2
        self.q1_handle = q1_handle
        self.q2_handle = q2_handle
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
        SWObservable.__init__(self,len_pow.survey_id,dim=self.len_pow.ls.size)

    def get_O_I(self):
        """Get the actual observable associated with the object"""
        return sp.Cll_q_q(self.len_pow.C_pow,self.q1_pow,self.q2_pow).Cll()

    def get_dO_I_ddelta_bar(self):
        """Get \frac{\partial{O_I}}{\partial\bar{\delta}(z)} as a function of the z grid, to be integrated by the long wavelength basis"""
        return sp.Cll_q_q(self.len_pow.dC_ddelta,self.q1_dC,self.q2_dC).Cll_integrand()

    def get_dO_I_dpars(self):
        """Get \partial{O_I}\partial{\Theta_i} for the set of observables as set up in LensingPowerBase"""
        dO_dpars = np.zeros((self.get_dimension(),self.len_pow.cosmo_par_list.size))
        for itr in xrange(0,dO_dpars.shape[1]):
            Cll_low = sp.Cll_q_q(self.len_pow.dC_dpars[itr,0],self.q1_dpars[itr,0],self.q2_dpars[itr,0]).Cll()
            Cll_high = sp.Cll_q_q(self.len_pow.dC_dpars[itr,1],self.q1_dpars[itr,1],self.q2_dpars[itr,1]).Cll()
            dO_dpars[:,itr] = (Cll_high-Cll_low)/(2.*self.len_pow.cosmo_par_epsilons[itr])
        return dO_dpars

class ShearShearLensingObservable(LensingObservable):
    def __init__(self,len_pow,r1,r2):
        """Shear shear lensing signal LensingObservable"""
        LensingObservable.__init__(self,len_pow,r1,r2,len_w.QShear,len_w.QShear)

class GalaxyGalaxyLensingObservable(LensingObservable):
    def __init__(self,len_pow,r1,r2):
        """Galaxy galaxy lensing signal LensingObservable"""
        LensingObservable.__init__(self,len_pow,r1,r2,len_w.QNum,len_w.QNum)
