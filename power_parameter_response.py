import re
import numpy as np
import FASTPTcode.FASTPT as FASTPT
import halofit as hf
import cosmopie as cp
from scipy.interpolate import InterpolatedUnivariateSpline,interp1d
import defaults
import camb_power as cpow
from warnings import warn
import matter_power_spectrum as mps

def get_perturbed_cosmopies(C_fid,pars,epsilons,log_param_derivs=np.array([],dtype=bool)):
    """get set of 2 perturbed cosmopies, above and below (including camb linear power spectrum) for getting partial derivatives
        using central finite difference method 
        inputs:
            C_fid: the fiducial CosmoPie
            pars: an array of the names of parameters to change
            epsilons: an array of step sizes correspondings to pars
            log_param_derivs: if True for a given element of pars will do log derivative in the parameter
    """
    cosmo_fid = C_fid.cosmology.copy()
    P_fid = C_fid.P_lin
    k_fid = C_fid.k
    camb_params=C_fid.camb_params.copy()

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
        if pars[i] in cp.GROW_SAFE:
            C_a = cp.CosmoPie(cosmo_a,p_space=cosmo_fid['p_space'],G_safe=True,G_in=C_fid.G_p)
            C_b = cp.CosmoPie(cosmo_b,p_space=cosmo_fid['p_space'],G_safe=True,G_in=C_fid.G_p)
            P_a = mps.MatterPower(C_a,k_in=k_fid,camb_params=camb_params,wm_in=P_fid.wm,wm_safe=True,de_perturbative=True)
            P_b = mps.MatterPower(C_b,k_in=k_fid,camb_params=camb_params,wm_in=P_fid.wm,wm_safe=True,de_perturbative=True)
        else:
            C_a = cp.CosmoPie(cosmo_a,p_space=cosmo_fid['p_space'])
            C_b = cp.CosmoPie(cosmo_b,p_space=cosmo_fid['p_space'])
            #avoid unnecessarily recomputing WMatchers for dark energy related parameters, and unnecessarily calling camb
            if pars[i] in cp.DE_SAFE:
                P_a = mps.MatterPower(C_a,k_in=k_fid,camb_params=camb_params,wm_in=P_fid.wm,wm_safe=True,P_fid=P_fid,camb_safe=True)
                P_b = mps.MatterPower(C_b,k_in=k_fid,camb_params=camb_params,wm_in=P_fid.wm,wm_safe=True,P_fid=P_fid,camb_safe=True)
            else:
                P_a = mps.MatterPower(C_a,k_in=k_fid,camb_params=camb_params,de_perturbative=True)
                P_b = mps.MatterPower(C_b,k_in=k_fid,camb_params=camb_params,de_perturbative=True)
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

if __name__=="__main__":
    
        #d = np.loadtxt('camb_m_pow_l.dat')
        #k_in = d[:,0]
        zs = np.arange(0.1,1.0,0.1)



        do_plot1=False
        do_plot2=False
        do_plot3=True
        import matplotlib.pyplot as plt

        if do_plot1:
            camb_params = defaults.camb_params.copy()
            pmodel_use = 'linear'
            camb_params['minkh'] = 0.02
            camb_params['maxkh'] = 10.0
            camb_params['kmax'] = 10.0
            camb_params['leave_h'] = False
            camb_params['force_sigma8'] = True

            epsilon = 0.01

            C=cp.CosmoPie(defaults.cosmology.copy(),p_space='basic')
            cosmo_fid=C.cosmology
            k_fid,P_fid = cpow.camb_pow(cosmo_fid,camb_params=camb_params)


            dp_params = defaults.dp_params.copy()
            dp_params['use_k3p']=False
            dp_params['log_deriv_direct']=True

            #attempt to replicate leftmost subplot at top of page 6 in Neyrinck 2011 arxiv:1105.2955v2
            dp_ch2,pza,pzb = dp_dpar(k_fid,np.array([0.]),C,'Omegamh2',log_param_deriv=True,pmodel=pmodel_use,epsilon=epsilon,camb_params=camb_params,dp_params=dp_params)
            dp_bh2,pza,pzb = dp_dpar(k_fid,np.array([0.]),C,'Omegabh2',log_param_deriv=True,pmodel=pmodel_use,epsilon=epsilon,camb_params=camb_params,dp_params=dp_params)
            dp_s8,pza,pzb = dp_dpar(k_fid,np.array([0.]),C,'sigma8',log_param_deriv=True,pmodel=pmodel_use,epsilon=epsilon,camb_params=camb_params,dp_params=dp_params)
            dp_ns,pza,pzb = dp_dpar(k_fid,np.array([0.]),C,'ns',pmodel=pmodel_use,epsilon=epsilon,camb_params=camb_params,dp_params=dp_params)
            min_index = 0
            max_index = np.argmin(k_fid<0.5)
         
            plt.semilogx(k_fid[min_index:max_index],dp_ns[min_index:max_index,0])
            plt.semilogx(k_fid[min_index:max_index],dp_s8[min_index:max_index,0]/2.)
            plt.semilogx(k_fid[min_index:max_index],dp_ch2[min_index:max_index,0])
            plt.semilogx(k_fid[min_index:max_index],dp_bh2[min_index:max_index,0])
            plt.legend(['ns','log(sigma8^2)','log(Omegach2)','log(Omegabh2)'],loc=4)
            plt.xlabel('k')
            plt.xlim([0.02,0.5])
            plt.ylim([-2.5,1.5])
            plt.ylabel('|dln(P)/dtheta|')
            plt.title('camb linear power spectrum response functions')
            plt.grid()
            plt.show()
                 
        if do_plot2:
            camb_params = defaults.camb_params.copy()
            pmodel_use = 'halofit'
            camb_params['minkh'] = 0.01
            camb_params['maxkh'] = 50.0
            camb_params['kmax'] = 100.0
            camb_params['leave_h'] = True
            camb_params['force_sigma8'] = False
            dp_params = defaults.dp_params.copy()
            dp_params['use_k3p']=True
            dp_params['log_deriv_direct']=True
            epsilon = 0.0001
            cosmo_fid = defaults.cosmology.copy()
            cosmo_fid['h']=0.7;cosmo_fid['ns']=0.96;cosmo_fid['Omegamh2']=0.14014;cosmo_fid['Omegabh2']=0.02303;cosmo_fid['Omegakh2']=0.
            cosmo_fid['sigma8']=0.82
            cosmo_fid['LogAs']=-19.9229844537
            cosmo_fid = cp.strip_cosmology(cosmo_fid,'lihu',overwride=['tau','Yp'])
            cosmo_fid['skip_sigma8']=True
            cosmo_fid = cp.add_derived_pars(cosmo_fid)
            C=cp.CosmoPie(cosmology=cosmo_fid,p_space='lihu')
            k_fid,P_fid = cpow.camb_pow(cosmo_fid,camb_params=camb_params)
            #attempt to replicate leftmost subplot at top of page 6 in Neyrinck 2011 arxiv:1105.2955v2
            epsilon_h =0.02
            dp_h,pza,pzb = dp_dpar(k_fid,np.array([0.]),C,'h',pmodel=pmodel_use,epsilon=epsilon_h,camb_params=camb_params,dp_params=dp_params)
            epsilon_LogAs = 0.03 
            dp_LogAs,pza,pzb = dp_dpar(k_fid,np.array([0.]),C,'LogAs',pmodel=pmodel_use,epsilon=epsilon_LogAs,camb_params=camb_params,dp_params=dp_params)
            epsilon_ns = 0.01
            dp_ns,pza,pzb = dp_dpar(k_fid,np.array([0.]),C,'ns',pmodel=pmodel_use,epsilon=epsilon_ns,camb_params=camb_params,dp_params=dp_params)
            min_index = 0
            #max_index = np.argmin(k_fid<1.1)
            max_index = k_fid.size
         
            #plt.semilogx(k_fid[min_index:max_index],dp_h[min_index:max_index,0])
            plt.loglog(k_fid[min_index:max_index],np.abs(dp_h[min_index:max_index,0]))
            plt.loglog(k_fid[min_index:max_index],np.abs(dp_ns[min_index:max_index,0]))
            #plt.loglog(k_fid[min_index:max_index]/cosmo_fid['h'],np.abs(dp_h[min_index:max_index,0]/P_fid[min_index:max_index]))
            #plt.loglog(k_fid[min_index:max_index]/cosmo_fid['h'],np.abs(dp_ns[min_index:max_index,0]/P_fid[min_index:max_index]))
            plt.loglog(k_fid[min_index:max_index],np.abs(dp_LogAs[min_index:max_index,0]))
            plt.legend(['h','ns','As'],loc=4)
            plt.xlabel('k h mpc^-1')
            plt.xlim([0.01,2.])
            plt.ylim([0.2,3.5])
            plt.ylabel('|dln(P)/dtheta|')
            plt.title('camb+halofit power spectrum response functions')
            plt.grid()
            plt.show()
        if do_plot3:
            camb_params = defaults.camb_params.copy()
            pmodel_use = 'linear'
            #camb_params['minkh'] = 0.02
            camb_params['maxkh'] = 10.0
            camb_params['kmax'] = 10.0
            camb_params['leave_h'] = False
            camb_params['force_sigma8'] = False#True

            epsilon = 0.01

            cosmo_fid=defaults.cosmology.copy()
            #k_fid,P_fid = cpow.camb_pow(cosmo_fid,camb_params=camb_params)
            C=cp.CosmoPie(defaults.cosmology.copy(),p_space='jdem',camb_params=camb_params)
            P_fid = mps.MatterPower(C,camb_params=camb_params)
            C.P_lin=P_fid
            C.k = P_fid.k
            Cs_pert = get_perturbed_cosmopies(C,np.array(['Omegamh2']),np.array([epsilon]),np.array([False]))
