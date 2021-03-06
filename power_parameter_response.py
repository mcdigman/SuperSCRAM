"""module gets perturbed sets of cosmologies and CosmoPie objects for varying cosmological parameters"""
from __future__ import division,print_function,absolute_import
from builtins import range
from warnings import warn

import numpy as np
import cosmopie as cp
import matter_power_spectrum as mps
def get_perturbed_cosmopies(C_fid,pars,epsilons,log_par_derivs=None,override_safe=False):
    """get set of 2 perturbed cosmopies, above and below (including camb linear power spectrum) for getting partial derivatives
        using central finite difference method
        inputs:
            C_fid: the fiducial CosmoPie
            pars: an array of the names of parameters to change
            epsilons: an array of step sizes correspondings to pars
            log_par_derivs: if True for a given element of pars will do log derivative in the parameter
            override_safe: if True do not borrow the growth factor or power spectrum from C_fid even if we could
    """
    cosmo_fid = C_fid.cosmology.copy()
    P_fid = C_fid.P_lin
    k_fid = C_fid.k

    power_params = P_fid.power_params.copy()

    #default assumption is ordinary derivative, can do log deriv in parameter also
    #if log_par_derivs[i]==True, will do log deriv
    if log_par_derivs is not None and log_par_derivs.size!=pars.size:
        raise ValueError('invalid input log_par_derivs '+str(log_par_derivs))
    elif log_par_derivs is None:
        log_par_derivs = np.zeros(pars.size,dtype=bool)

    Cs_pert = np.zeros((pars.size,2),dtype=object)

    for i in range(0,pars.size):
        cosmo_a = get_perturbed_cosmology(cosmo_fid,pars[i],epsilons[i],log_par_derivs[i])
        cosmo_b = get_perturbed_cosmology(cosmo_fid,pars[i],-epsilons[i],log_par_derivs[i])

        #set cosmopie power spectrum appropriately
        #avoid unnecessarily recomputing growth factors if they won't change. If growth factors don't change neither will w matching
        if pars[i] in cp.GROW_SAFE and not override_safe:
            C_a = cp.CosmoPie(cosmo_a,p_space=cosmo_fid['p_space'],G_safe=True,G_in=C_fid.G_p)
            C_b = cp.CosmoPie(cosmo_b,p_space=cosmo_fid['p_space'],G_safe=True,G_in=C_fid.G_p)
            P_a = mps.MatterPower(C_a,power_params,k_in=k_fid,wm_in=P_fid.wm,de_perturbative=True)
            P_b = mps.MatterPower(C_b,power_params,k_in=k_fid,wm_in=P_fid.wm,de_perturbative=True)
        else:
            C_a = cp.CosmoPie(cosmo_a,p_space=cosmo_fid['p_space'])
            C_b = cp.CosmoPie(cosmo_b,p_space=cosmo_fid['p_space'])
            #avoid unnecessarily recomputing WMatchers for dark energy related parameters, and unnecessarily calling camb
            if pars[i] in cp.DE_SAFE and not override_safe:
                P_a = mps.MatterPower(C_a,power_params,k_in=k_fid,wm_in=P_fid.wm,P_fid=P_fid,camb_safe=True)
                P_b = mps.MatterPower(C_b,power_params,k_in=k_fid,wm_in=P_fid.wm,P_fid=P_fid,camb_safe=True)
            else:
                P_a = mps.MatterPower(C_a,power_params,k_in=k_fid,de_perturbative=True)
                P_b = mps.MatterPower(C_b,power_params,k_in=k_fid,de_perturbative=True)
        #k_a = P_a.k
        #k_b = P_b.k

        C_a.set_power(P_a)
        C_b.set_power(P_b)

        Cs_pert[i,0] = C_a
        Cs_pert[i,1] = C_b
    return Cs_pert


#get new cosmology with 1 parameter self consistently (ie enforce exact relations) perturbed
def get_perturbed_cosmology(cosmo_old,parameter,epsilon,log_param_deriv=False):
    """get a perturbed cosmology dictionary needed for get_perturbed_cosmopie
        inputs:
            cosmo_old: fiducial cosmology
            paramter: name of parameter to change
            epsilon: amount to change parameter by
            log_param_deriv: if True, do log change in parameter"""
    cosmo_new = cosmo_old.copy()
    if cosmo_new.get(parameter) is not None:

        if log_param_deriv:
            cosmo_new[parameter] = np.exp(np.log(cosmo_old[parameter])+epsilon)
        else:
            cosmo_new[parameter]+=epsilon

        if cosmo_old['de_model']=='w0wa':
            if parameter=='w':
                cosmo_new['w0'] = cosmo_new['w']
            elif parameter=='w0':
                cosmo_new['w'] = cosmo_new['w0']

        if parameter not in cp.P_SPACES.get(cosmo_new.get('p_space')) and parameter not in cp.DE_METHODS[cosmo_new.get('de_model')]:
            warn('parameter \''+str(parameter)+'\' may not support derivatives with the parameter set \''+str(cosmo_new.get('p_space'))+'\'')
        return cp.add_derived_pars(cosmo_new)
    else:
        raise ValueError('undefined parameter in cosmology \''+str(parameter)+'\'')
