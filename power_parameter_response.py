import numpy as np
import FASTPTcode.FASTPT as FASTPT
import halofit as hf
import cosmopie as cp
from scipy.interpolate import InterpolatedUnivariateSpline,interp1d
import defaults
import camb_power as cpow
from warnings import warn

#P_a is fiducial power spectrum 
#parameter can be H0,Omegabh2,Omegach2,Omegak,ns,sigma8,h
#include others such as omegam, w
#use central finite difference because I think chris said to, check if it is actually necessary.
def dp_dparameter(k_fid,P_fid,zs,C,parameter,pmodel='linear',epsilon=0.0001,fpt=None,fpt_params=defaults.fpt_params,camb_params=defaults.camb_params,dp_params=defaults.dp_params):
    cosmo_fid = C.cosmology.copy()
    cosmo_a = get_perturbed_cosmology(cosmo_fid,parameter,epsilon)
    cosmo_b = get_perturbed_cosmology(cosmo_fid,parameter,-epsilon)
    print cosmo_a
    print cosmo_b
    #get perturbed linear power spectra from camb
    k_a,P_a = cpow.camb_pow(cosmo_a,camb_params=camb_params) 
    k_b,P_b = cpow.camb_pow(cosmo_b,camb_params=camb_params) 
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
        p_lin_a = np.outer(P_a,C.G_norm(zs)**2) 
        p_lin_b = np.outer(P_a,C.G_norm(zs)**2) 
        pza = np.zeros_like(p_lin_a)
        pzb = np.zeros_like(p_lin_b)
        for itr in range(0,zs.size):
            pza[:,itr] = p_lin_a[:,itr] = fpt.one_loop(p_lin_a[:,itr],C_window=fpt_params['C_window'])
            pzb[:,itr] = p_lin_b[:,itr] = fpt.one_loop(p_lin_b[:,itr],C_window=fpt_params['C_window'])            
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
    
#get new cosmology with 1 parameter self consistently (ie enforce exact relations) perturbed
def get_perturbed_cosmology(cosmo_old,parameter,epsilon=0.0001,hold_omegah2=True):
    cosmo_new = cosmo_old.copy()
    if cosmo_new.get(parameter) is not None:
        cosmo_new[parameter]+=epsilon   
        if parameter not in cp.P_SPACES.get(cosmo_new.get('p_space')):
            warn('parameter '+str(parameter)+' may not support derivatives with this parameter set')
        return cp.add_derived_parameters(cosmo_new)
    else:
        raise ValueError('undefined parameter in cosmology \''+str(parameter)+'\'')

    if parameter=='Omegabh2':
        #there is no Omegab or it would change. ignore change to Omegam for now
        #TODO do I need to change omega m here?
        cosmo_new['Omegabh2'] = cosmo_old['Omegabh2']+epsilon  
        cosmo_new['Omegab'] = cosmo_new['Omegabh2']/cosmo_old['h']**2
        cosmo_new['Omegamh2'] = cosmo_old['Omegamh2']+epsilon 
        cosmo_new['Omegam'] = cosmo_new['Omegamh2']/cosmo_old['h']**2
    elif parameter=='Omegach2':
        cosmo_new['Omegach2'] = cosmo_old['Omegach2']+epsilon
        cosmo_new['Omegac'] = cosmo_new['Omegach2']/cosmo_old['h']**2
        cosmo_new['Omegamh2'] = cosmo_old['Omegamh2']+epsilon 
        cosmo_new['Omegam'] = cosmo_new['Omegamh2']/cosmo_old['h']**2
        #added
        cosmo_new['OmegaL'] = 1.-cosmo_new['Omegab']-cosmo_new['Omegac']
    elif parameter=='Omegamh2':
        #treat change in Omegam as change in Omegac
        cosmo_new['Omegamh2'] = cosmo_old['Omegamh2']+epsilon
        cosmo_new['Omegam'] = cosmo_new['Omegamh2']/cosmo_old['h']**2
        cosmo_new['Omegach2'] = cosmo_old['Omegach2']+epsilon
        cosmo_new['Omegac'] = cosmo_new['Omegach2']/cosmo_old['h']**2
    elif parameter=='OmegaL':
        cosmo_new['OmegaL'] = cosmo_old['OmegaL']+epsilon
    elif parameter=='Omegam':
        #treat change in Omegam as change in Omegac
        cosmo_new['Omegam'] = cosmo_old['Omegam']+epsilon
        cosmo_new['Omegamh2'] = cosmo_new['Omegam']*cosmo_old['h']**2
        cosmo_new['Omegac'] = cosmo_old['Omegac']+epsilon
        cosmo_new['Omegach2'] = cosmo_new['Omegac']*cosmo_old['h']**2
    elif parameter=='H0':
        #TODO think about if this is allowed, what we want to keep fixed
        #for now hold Omegab,Omegam,Omegac constant
        cosmo_new['H0'] = cosmo_old['H0']+epsilon
        cosmo_new['h'] = cosmo_old['h']+cosmo_old['h']/cosmo_old['H0']*epsilon
        cosmo_new['Omegabh2'] = cosmo_old['Omegabh2']/cosmo_old['h']**2*cosmo_new['h']**2
        cosmo_new['Omegach2'] = cosmo_old['Omegach2']/cosmo_old['h']**2*cosmo_new['h']**2
        cosmo_new['Omegamh2'] = cosmo_old['Omegamh2']/cosmo_old['h']**2*cosmo_new['h']**2
    elif parameter=='h':
        cosmo_new['h'] = cosmo_old['h']+epsilon
        cosmo_new['H0'] = cosmo_old['H0']+cosmo_old['H0']/cosmo_old['h']*epsilon
        if hold_omegah2:
            cosmo_new['Omegab'] = cosmo_old['Omegab']*cosmo_old['h']**2/cosmo_new['h']**2
            cosmo_new['Omegac'] = cosmo_old['Omegac']*cosmo_old['h']**2/cosmo_new['h']**2
            cosmo_new['Omegam'] = cosmo_old['Omegam']*cosmo_old['h']**2/cosmo_new['h']**2
            #TODO need to force universe to stay flat consistently
            cosmo_new['OmegaL'] = 1.-cosmo_new['Omegab']-cosmo_new['Omegac']
            #cosmo_new['OmegaL'] = cosmo_old['OmegaL']*cosmo_old['h']**2/cosmo_new['h']**2
        else:
            cosmo_new['Omegabh2'] = cosmo_old['Omegabh2']/cosmo_old['h']**2*cosmo_new['h']**2
            cosmo_new['Omegach2'] = cosmo_old['Omegach2']/cosmo_old['h']**2*cosmo_new['h']**2
            cosmo_new['Omegamh2'] = cosmo_old['Omegamh2']/cosmo_old['h']**2*cosmo_new['h']**2
    elif parameter=='sigma8':
        cosmo_new['sigma8']=cosmo_old['sigma8']+epsilon
    elif parameter=='ns':
        cosmo_new['ns'] = cosmo_old['ns']+epsilon
    elif parameter=='Omegak':
        cosmo_new['Omegak'] = cosmo_old['Omegak']+epsilon
    elif parameter=='Omegar':
        #TODO check relationships
        cosmo_new['Omegar'] = cosmo_old['Omegar']+epsilon
    elif parameter=='tau':
        cosmo_new['tau'] = cosmo_old['tau']+epsilon
    elif parameter=='100thetamc':
        warn('power_parameter_response: unsupported parameter '+parameter)
        cosmo_new['100thetamc'] = cosmo_old['100thetamc']+epsilon
    elif parameter=='As':
        cosmo_new['As'] = cosmo_old['As']+epsilon
    else:
        raise ValueError('unrecognized paramater \''+str(parameter)+'\'')
    return cosmo_new


if __name__=="__main__":
    
        #d = np.loadtxt('camb_m_pow_l.dat')
        #k_in = d[:,0]
        zs = np.arange(0.1,1.0,0.1)
        epsilon = 0.00001



        do_plot1=False
        do_plot2=True
        import matplotlib.pyplot as plt

        if do_plot1:
            camb_params = defaults.camb_params.copy()
            pmodel_use = 'linear'
            camb_params['minkh'] = 0.02
            camb_params['maxkh'] = 10.0
            camb_params['kmax'] = 10.0
            camb_params['leave_h'] = False

            cosmo_fid = defaults.cosmology.copy()
            C=cp.CosmoPie(cosmology=cosmo_fid)
            k_fid,P_fid = cpow.camb_pow(cosmo_fid,camb_params=camb_params)

            #attempt to replicate leftmost subplot at top of page 6 in Neyrinck 2011 arxiv:1105.2955v2
            dp_ch2,pza,pzb = dp_dparameter(k_fid,P_fid,np.array([0.]),C,'Omegach2',pmodel=pmodel_use,epsilon=epsilon,camb_params=camb_params)
            dp_bh2,pza,pzb = dp_dparameter(k_fid,P_fid,np.array([0.]),C,'Omegabh2',pmodel=pmodel_use,epsilon=epsilon,camb_params=camb_params)
            dp_s8,pza,pzb = dp_dparameter(k_fid,P_fid,np.array([0.]),C,'sigma8',pmodel=pmodel_use,epsilon=epsilon,camb_params=camb_params)
            dp_ns,pza,pzb = dp_dparameter(k_fid,P_fid,np.array([0.]),C,'ns',pmodel=pmodel_use,epsilon=epsilon,camb_params=camb_params)
            min_index = 0
            max_index = np.argmin(k_fid<0.5)
         
            plt.semilogx(k_fid[min_index:max_index],dp_ns[min_index:max_index,0]/P_fid[min_index:max_index])
            plt.semilogx(k_fid[min_index:max_index],dp_s8[min_index:max_index,0]/P_fid[min_index:max_index]*cosmo_fid['sigma8']/2.)
            plt.semilogx(k_fid[min_index:max_index],dp_ch2[min_index:max_index,0]/P_fid[min_index:max_index]*cosmo_fid['Omegach2'])
            plt.semilogx(k_fid[min_index:max_index],dp_bh2[min_index:max_index,0]/P_fid[min_index:max_index]*cosmo_fid['Omegabh2'])
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
            dp_params = defaults.dp_params.copy()
            dp_params['use_k3p']=True
            epsilon = 0.0001
            cosmo_fid = defaults.cosmology.copy()
            cosmo_fid['h']=0.7;cosmo_fid['ns']=0.96;cosmo_fid['Omegamh2']=0.14014;cosmo_fid['Omegabh2']=0.02303;cosmo_fid['Omegakh2']=0.
            cosmo_fid['sigma8']=0.82
            cosmo_fid['LogAs']=-19.9229844537
            cosmo_fid = cp.strip_cosmology(cosmo_fid,'lihu',overwride=['tau','Yp'])
            cosmo_fid = cp.add_derived_parameters(cosmo_fid)
            C=cp.CosmoPie(cosmology=cosmo_fid)
            k_fid,P_fid = cpow.camb_pow(cosmo_fid,camb_params=camb_params)
            #TODO investigate why As makes no difference at all
            #attempt to replicate leftmost subplot at top of page 6 in Neyrinck 2011 arxiv:1105.2955v2
            epsilon_h =0.02
            dp_h,pza,pzb = dp_dparameter(k_fid,P_fid,np.array([0.]),C,'h',pmodel=pmodel_use,epsilon=epsilon_h,camb_params=camb_params,dp_params=dp_params)
            epsilon_LogAs = 0.03 
            dp_LogAs,pza,pzb = dp_dparameter(k_fid,P_fid,np.array([0.]),C,'LogAs',pmodel=pmodel_use,epsilon=epsilon_LogAs,camb_params=camb_params,dp_params=dp_params)
            epsilon_ns = 0.01
            dp_ns,pza,pzb = dp_dparameter(k_fid,P_fid,np.array([0.]),C,'ns',pmodel=pmodel_use,epsilon=epsilon_ns,camb_params=camb_params,dp_params=dp_params)
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
                 
