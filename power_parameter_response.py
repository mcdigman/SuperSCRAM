"""module gets perturbed sets of cosmologies and CosmoPie objects for varying cosmological parameters"""
from warnings import warn

import numpy as np
import cosmopie as cp
import matter_power_spectrum as mps
#TODO there may be a little h handling bug somewhere, see test case 2
def get_perturbed_cosmopies(C_fid,pars,epsilons,log_param_derivs=np.array([],dtype=bool),override_safe=False):
    """get set of 2 perturbed cosmopies, above and below (including camb linear power spectrum) for getting partial derivatives
        using central finite difference method
        inputs:
            C_fid: the fiducial CosmoPie
            pars: an array of the names of parameters to change
            epsilons: an array of step sizes correspondings to pars
            log_param_derivs: if True for a given element of pars will do log derivative in the parameter
            override_safe: if True do not borrow the growth factor or power spectrum from C_fid even if we could
    """
    cosmo_fid = C_fid.cosmology.copy()
    P_fid = C_fid.P_lin
    k_fid = C_fid.k

    power_params = P_fid.power_params.copy()

    #default assumption is ordinary derivative, can do log deriv in parameter also
    #if log_param_derivs[i] == True, will do log deriv
    if not log_param_derivs.size==pars.size:
        log_param_derivs = np.zeros(pars.size,dtype=bool)

    Cs_pert = np.zeros((pars.size,2),dtype=object)

    for i in xrange(0,pars.size):
        cosmo_a = get_perturbed_cosmology(cosmo_fid,pars[i],epsilons[i],log_param_derivs[i])
        cosmo_b = get_perturbed_cosmology(cosmo_fid,pars[i],-epsilons[i],log_param_derivs[i])

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
        k_a = P_a.k
        k_b = P_b.k

        C_a.P_lin = P_a
        C_a.k = k_a
        C_b.P_lin = P_b
        C_b.k = k_b

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

        if cosmo_old['de_model'] == 'w0wa' and parameter=='w':
            warn('given parameter should not be used in w0wa parameterization, use w0 instead')
            cosmo_new['w0'] = cosmo_new['w']

        if parameter not in cp.P_SPACES.get(cosmo_new.get('p_space')) and parameter not in cp.DE_METHODS[cosmo_new.get('de_model')]:
            warn('parameter \''+str(parameter)+'\' may not support derivatives with the parameter set \''+str(cosmo_new.get('p_space'))+'\'')
        return cp.add_derived_pars(cosmo_new)
    else:
        raise ValueError('undefined parameter in cosmology \''+str(parameter)+'\'')

#TODO test if needed
def dp_dpar(C_fid,parameter,log_param_deriv,pmodel,epsilon,log_deriv=False):
    """get derivative of power spectrum with respect to an observable, may not currently work"""
    Cs = get_perturbed_cosmopies(C_fid,np.array([parameter]),np.array([epsilon]),np.array([log_param_deriv]),override_safe=True)
    if log_deriv:
        pza = C_fid.k**3/(2.*np.pi**2.)*Cs[0][0].P_lin.get_matter_power(np.array([0.]),pmodel=pmodel)
        pzb = C_fid.k**3/(2.*np.pi**2.)*Cs[0][1].P_lin.get_matter_power(np.array([0.]),pmodel=pmodel)
        result = (np.log(pza)-np.log(pzb))/(2*epsilon)
    else:
        result = (Cs[0][0].P_lin.get_matter_power(np.array([0.]),pmodel=pmodel)-Cs[0][1].P_lin.get_matter_power(np.array([0.]),pmodel=pmodel))/(2*epsilon)
    return result
