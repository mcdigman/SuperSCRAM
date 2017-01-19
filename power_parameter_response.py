import numpy as np
import FASTPTcode.FASTPT as FASTPT
import halofit as hf
import cosmopie as cp
from scipy.interpolate import InterpolatedUnivariateSpline,interp1d
import defaults
import camb_power as cpow

def dp_ddelta(k_a,P_a,zbar,C,pmodel='linear',epsilon=0.0001,halo_a=None,halo_b=None,fpt=None,fpt_params=defaults.fpt_params):
    if pmodel=='linear':
       # pza = P_a
        pza = P_a*C.G_norm(zbar)**2
        #checked equality
        #dp1 = 68./21.*pza-1./3.*(InterpolatedUnivariateSpline(np.log(k_a),np.log(pza*k_a**3)).derivative(1))(np.log(k_a))*pza
        dpdk =(InterpolatedUnivariateSpline(k_a,pza,ext=2,k=1).derivative(1))(k_a) 
        dp = 47./21.*pza-1./3.*k_a*dpdk
    elif pmodel=='halofit':
        if halo_a is None:
            halo_a = hf.halofitPk(C,k_a,P_a)
        if halo_b is None:
            halo_b = hf.halofitPk(C,k_a,P_a*(1.+epsilon/C.sigma8)**2)
        pza = halo_a.D2_NL(k_a,zbar)*2.*np.pi**2/k_a**3
        pzb = halo_b.D2_NL(k_a,zbar)*2.*np.pi**2/k_a**3
        dpdk =(InterpolatedUnivariateSpline(k_a,pza,ext=2,k=1).derivative(1))(k_a) 
        dp = 13./21.*C.sigma8*(pzb-pza)/epsilon+pza-1./3.*k_a*dpdk
    elif pmodel=='fastpt':
        if fpt is None:
            fpt = FASTPT.FASTPT(k_a,fpt_params['nu'],low_extrap=fpt_params['low_extrap'],high_extrap=fpt_params['high_extrap'],n_pad=fpt_params['n_pad'])
        plin = P_a*C.G_norm(zbar)**2

        pza = plin+fpt.one_loop(plin,C_window=fpt_params['C_window'])
        dpdk =(InterpolatedUnivariateSpline(k_a,pza,ext=2,k=1).derivative(1))(k_a) 
        dp = 47./21.*pza-1./3.*k_a*dpdk+26./21.*fpt.one_loop(plin,C_window=0.75)
    else:
        raise ValueError('invalid pmodel option \''+str(pmodel)+'\'')
    return dp,pza
#TODO make power_response support fpt params
#P_a is fiducial power spectrum 
#parameter can be H0,Omegabh2,Omegach2,Omegak,ns,sigma8,h
#include others such as omegam, w
#use central finite difference because I think chris said to, check if it is actually necessary.
def dp_dparameter(k_fid,P_fid,zs,C,parameter,pmodel='linear',epsilon=0.0001,fpt=None,fpt_params=defaults.fpt_params):
    cosmo_fid = C.cosmology.copy()
    cosmo_a = get_perturbed_cosmology(cosmo_fid,parameter,epsilon/2.)
    cosmo_b = get_perturbed_cosmology(cosmo_fid,parameter,-epsilon/2.)
    print cosmo_a
    print cosmo_b
    #get perturbed linear power spectra from camb
    k_a,P_a = cpow.camb_pow(cosmo_a) 
    k_b,P_b = cpow.camb_pow(cosmo_b) 
    if not np.all(k_a==k_fid):
        P_a = np.interp1d(k_a,P_a)(k_fid)
        P_b = np.interp1d(k_b,P_a)(k_fid)

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
    dp = (0.5*pza-0.5*pzb)/epsilon
    return dp,pza,pzb 
    
#get new cosmology with 1 parameter self consistently (ie enforce exact relations) perturbed
def get_perturbed_cosmology(cosmo_old,parameter,epsilon=0.0001):
    cosmo_new = cosmo_old.copy()
    if parameter=='Omegabh2':
        #there is no Omegab or it would change. ignore change to Omegam for now
        #TODO do I need to change omega m here?
       cosmo_new['Omegabh2'] = cosmo_old['Omegabh2']+epsilon 
    elif parameter=='Omegach2':
        cosmo_new['Omegach2'] = cosmo_old['Omegach2']+epsilon
    elif parameter=='Omegamh2':
        cosmo_new['Omegamh2'] = cosmo_old['Omegamh2']+epsilon
        cosmo_new['Omegam'] = cosmo_new['Omegamh2']/cosmo_new['h']**2
    elif parameter=='OmegaL':
        cosmo_new['OmegaL'] = cosmo_old['OmegaL']+epsilon
    elif parameter=='Omegam':
        cosmo_new['Omegam'] = cosmo_old['Omegam']+epsilon
        cosmo_new['Omegamh2'] = cosmo_new['Omegam']*cosmo_new['h']**2
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
        cosmo_new['100thetamc'] = cosmo_old['100thetamc']+epsilon
    elif parameter=='As':
        cosmo_new['As'] = cosmo_old['As']+epsilon
    else:
        raise ValueError('unrecognized paramater \''+str(parameter)+'\'')
    return cosmo_new


if __name__=="__main__":
    
        C=cp.CosmoPie(cosmology=defaults.cosmology)
        d = np.loadtxt('camb_m_pow_l.dat')
        k_in = d[:,0]
        zs = np.arange(0.1,1.0,0.1)
        ls = np.arange(1,5000)
        epsilon = 0.0001

        cosmo_fid = C.cosmology.copy()
        k_fid,P_fid = cpow.camb_pow(cosmo_fid)
        dp_ch2_lin,pza,pzb = dp_dparameter(k_fid,P_fid,np.array([0.5]),C,'Omegach2',pmodel='linear',epsilon=epsilon)
        dp_ch2_hf,pza,pzb = dp_dparameter(k_fid,P_fid,np.array([0.5]),C,'Omegach2',pmodel='halofit',epsilon=epsilon)
        #dp_bh2,pza,pzb = dp_dparameter(k_fid,P_fid,np.array([0.5]),C,'Omegabh2',pmodel='linear',epsilon=epsilon)
        #dp_ns,pza,pzb = dp_dparameter(k_fid,P_fid,np.array([0.5]),C,'ns',pmodel='linear',epsilon=epsilon)
        dp_s8,pza,pzb = dp_dparameter(k_fid,P_fid,np.array([0.5]),C,'sigma8',pmodel='halofit',epsilon=epsilon)
        #print get_perturbed_cosmology(cosmo_a,'Omegamh2',epsilon)
        #print cosmo_a

        
        import matplotlib.pyplot as plt
        plt.loglog(k_fid,np.abs(dp_ch2_lin[:,0]/P_fid))
        plt.loglog(k_fid,np.abs(dp_ch2_hf[:,0]/P_fid))
        #plt.loglog(k_fid,np.abs(dp_bh2[:,0]/P_fid))
        #plt.loglog(k_fid,np.abs(dp_ns[:,0]/P_fid))
        plt.loglog(k_fid,np.abs(dp_s8[:,0]/P_fid))
        plt.show()
        
            
        
