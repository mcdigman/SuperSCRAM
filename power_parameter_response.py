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

#P_a is fiducial power spectrum 
#parameter can be H0,Omegabh2,Omegach2,Omegak,ns,sigma8,h
#include others such as omegam, w
#TODO need to change G_norm or does not work properly
def dp_dpar(k_fid,zs,C,parameter,pmodel='linear',epsilon=0.0001,log_param_deriv=False,fpt=None,fpt_params=defaults.fpt_params,camb_params=defaults.camb_params,dp_params=defaults.dp_params):
    cosmo_fid = C.cosmology.copy()
    #print cosmo_fid
    #skip getting sigma8 if possible because it is slow
    if not parameter=='sigma8' and ('sigma8' not in cp.P_SPACES[cosmo_fid['p_space']]):
        cosmo_fid['skip_sigma8'] = True
        camb_params_use = camb_params.copy()
        camb_params_use['force_sigma8'] = False
    cosmo_a = get_perturbed_cosmology(cosmo_fid,parameter,epsilon,log_param_deriv=log_param_deriv)
    #print "cosmo_a",cosmo_a
    cosmo_b = get_perturbed_cosmology(cosmo_fid,parameter,-epsilon,log_param_deriv=log_param_deriv)
    #print "cosmo_b",cosmo_b
    #get perturbed linear power spectra from camb
    k_a,P_a = cpow.camb_pow(cosmo_a,camb_params=camb_params_use) 
    k_b,P_b = cpow.camb_pow(cosmo_b,camb_params=camb_params_use) 
    if not np.all(k_a==k_fid) or not(np.all(k_b==k_fid)):
        print "adapting k grid"
        P_a = InterpolatedUnivariateSpline(k_a,P_a,k=1)(k_fid)
        P_b = InterpolatedUnivariateSpline(k_b,P_b,k=1)(k_fid)
    if pmodel=='linear':
        pza = np.outer(P_a,C.G_norm(zs)**2) 
        pzb = np.outer(P_b,C.G_norm(zs)**2) 
    elif pmodel=='halofit':
        halo_a = hf.halofitPk(C,k_fid,P_a)
        halo_b = hf.halofitPk(C,k_fid,P_b)
        #TODO make D2_NL support vector z syntax
        pza = (2.*np.pi**2/k_fid**3*halo_a.D2_NL(k_fid,zs).T).T
        pzb = (2.*np.pi**2/k_fid**3*halo_b.D2_NL(k_fid,zs).T).T
    elif pmodel=='fastpt':
        if fpt is None:
            fpt = FASTPT.FASTPT(k_fid,fpt_params['nu'],low_extrap=fpt_params['low_extrap'],high_extrap=fpt_params['high_extrap'],n_pad=fpt_params['n_pad'])
        #TODO maybe make fastpt support vector z if it doesn't already
        p_lin_a = np.outer(P_a,C.G_norm(zs)**2) #TODO C needs to change properly
        p_lin_b = np.outer(P_b,C.G_norm(zs)**2) 
        one_loop_a = fpt.one_loop(P_a,C_window=fpt_params['C_window'])
        one_loop_b = fpt.one_loop(P_b,C_window=fpt_params['C_window'])
        pza = p_lin_a+np.outer(one_loop_a,C.G_norm(zs)**4)
        pzb = p_lin_b+np.outer(one_loop_b,C.G_norm(zs)**4)
        #pza = np.zeros_like(p_lin_a)
        #pzb = np.zeros_like(p_lin_b)
        #for itr in xrange(0,zs.size):
        #    pza[:,itr] = p_lin_a[:,itr] + fpt.one_loop(p_lin_a[:,itr],C_window=fpt_params['C_window'])
        #    pzb[:,itr] = p_lin_b[:,itr] + fpt.one_loop(p_lin_b[:,itr],C_window=fpt_params['C_window'])            
    else:
        raise ValueError('invalid pmodel option \''+str(pmodel)+'\'')

    if dp_params['use_k3p']:
        pza = k_a**3*pza/(2.*np.pi**2)
        pzb = k_b**3*pzb/(2.*np.pi**2)

    #TODO I'm not sure if interpolation like this is correct here. 
    if dp_params['log_deriv_direct']:
        dp = (np.log(pza)-np.log(pzb))/(2.*epsilon)
#    elif dp_params['log_deriv_indirect']:
#        dp = (pza-pzb)/(2.*epsilon*P_fid)
    else:
        dp = (pza-pzb)/(2.*epsilon)
    return dp,pza,pzb 
    
#get set of perturbed cosmopies (including camb linear power spectrum) for getting derivatives, assuming using central finite difference method
#TODO sigma8 handling is a mess
def get_perturbed_cosmopies(C_fid,pars,epsilons,log_param_derivs=np.array([])):
    cosmo_fid = C_fid.cosmology.copy()
    P_fid = C_fid.P_lin
    k_fid = C_fid.k
    camb_params=C_fid.camb_params.copy()

    #default assumption is ordinary derivative, can do log deriv in parameter also
    #if log_param_derivs[i] == True, will do log deriv
    if not log_param_derivs.size==pars.size:
        log_param_derivs = np.zeros(pars.size,dtype=bool)
    if epsilons is None:
        epsilons = np.zeros(pars.size)+0.0001
    
    Cs_pert = np.zeros((pars.size,2),dtype=object) 

    #de_model = cosmo_fid.get('de_model')
    #wmatch = C_fid.wmatch

    for i in xrange(0,pars.size):

        #skip evaulating sigma8 unless now unless it is necessary (it is slow), will get it later 
        if (not pars[i]=='sigma8') and ('sigma8' not in cp.P_SPACES[cosmo_fid['p_space']]):
            cosmo_fid['skip_sigma8'] = True
            camb_params['force_sigma8'] = False
            camb_params['return_sigma8'] = True

        cosmo_a = get_perturbed_cosmology(cosmo_fid,pars[i],epsilons[i],log_param_derivs[i])
        cosmo_b = get_perturbed_cosmology(cosmo_fid,pars[i],-epsilons[i],log_param_derivs[i])

        #handle dark energy
        #use wmatcher for all dark energy for now TODO consider actually calculating power in some way, because of possible ns degeneracy issue
        
        #if pars[i] in cp.DE_METHODS[de_model] and not de_model=='constant_w':
        #    k_a=k_fid
        #    k_b=k_fid
        #    P_a=P_fid
        #    P_b=P_fid
        #else:
          #  k_a,P_a,sigma8_a = cpow.camb_pow(cosmo_a,camb_params=camb_params) 
          #  k_b,P_b,sigma8_b = cpow.camb_pow(cosmo_b,camb_params=camb_params) 
        #attach a sigma8 to the resulting cosmology in case something needs it later

        #TODO not sure if interpolation is needed/correct
        #if not np.all(k_a==k_fid) or not(np.all(k_b==k_fid)):
        #    print "adapting k grid"
        #    P_a = InterpolatedUnivariateSpline(k_a,P_a,k=1)(k_fid)
        #    P_b = InterpolatedUnivariateSpline(k_b,P_b,k=1)(k_fid)
        #    k_a = k_fid
        #    k_b = k_fid

        #set cosmopie power spectrum appropriately
        if pars[i] in cp.GROW_SAFE:
            C_a = cp.CosmoPie(cosmo_a,p_space=cosmo_fid['p_space'],safe_sigma8=True,G_safe=True,G_in=C_fid.G_p)
            C_b = cp.CosmoPie(cosmo_b,p_space=cosmo_fid['p_space'],safe_sigma8=True,G_safe=True,G_in=C_fid.G_p)
            P_a = mps.MatterPower(C_a,k_in=k_fid,camb_params=camb_params,wm_in=P_fid.wm,wm_safe=True)
            P_b = mps.MatterPower(C_b,k_in=k_fid,camb_params=camb_params,wm_in=P_fid.wm,wm_safe=True)
        else:
            C_a = cp.CosmoPie(cosmo_a,p_space=cosmo_fid['p_space'],safe_sigma8=True)
            C_b = cp.CosmoPie(cosmo_b,p_space=cosmo_fid['p_space'],safe_sigma8=True)
            P_a = mps.MatterPower(C_a,k_in=k_fid,camb_params=camb_params)
            P_b = mps.MatterPower(C_b,k_in=k_fid,camb_params=camb_params)
        k_a = P_a.k
        k_b = P_b.k

        C_a.P_lin = P_a
        C_a.k = k_a
        C_b.P_lin = P_b
        C_b.k = k_b

        if not pars[i]=='sigma8' and ('sigma8' not in cp.P_SPACES[cosmo_fid['p_space']]):
            cosmo_a['sigma8'] = P_a.get_sigma8_eff(zs=np.array([0.]))[0]
            cosmo_b['sigma8'] = P_b.get_sigma8_eff(zs=np.array([0.]))[0]
            cosmo_a.pop('skip_sigma8',None)
            cosmo_b.pop('skip_sigma8',None)
        C_a.sigma8=cosmo_a['sigma8']
        C_b.sigma8=cosmo_b['sigma8']
        Cs_pert[i,0] = C_a 
        Cs_pert[i,1] = C_b
    return Cs_pert
        

#get new cosmology with 1 parameter self consistently (ie enforce exact relations) perturbed
def get_perturbed_cosmology(cosmo_old,parameter,epsilon=0.0001,log_param_deriv=False):
    cosmo_new = cosmo_old.copy()
    if cosmo_new.get(parameter) is not None:

        if log_param_deriv:
            cosmo_new[parameter] = np.exp(np.log(cosmo_old[parameter])+epsilon)   
        else:
            cosmo_new[parameter]+=epsilon   
        #TODO reconcile
        if cosmo_old['de_model'] == 'w0wa' and parameter=='w':
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
