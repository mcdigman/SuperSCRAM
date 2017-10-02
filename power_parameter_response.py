import numpy as np
import cosmopie as cp
import defaults
import camb_power as cpow
from warnings import warn
import matter_power_spectrum as mps
import halofit as hf
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
        if pars[i] in cp.GROW_SAFE and not override_safe:
            C_a = cp.CosmoPie(cosmo_a,p_space=cosmo_fid['p_space'],G_safe=True,G_in=C_fid.G_p,camb_params=camb_params)
            C_b = cp.CosmoPie(cosmo_b,p_space=cosmo_fid['p_space'],G_safe=True,G_in=C_fid.G_p,camb_params=camb_params)
            P_a = mps.MatterPower(C_a,k_in=k_fid,camb_params=camb_params,wm_in=P_fid.wm,wm_safe=True,de_perturbative=True)
            P_b = mps.MatterPower(C_b,k_in=k_fid,camb_params=camb_params,wm_in=P_fid.wm,wm_safe=True,de_perturbative=True)
        else:
            C_a = cp.CosmoPie(cosmo_a,p_space=cosmo_fid['p_space'],camb_params=camb_params)
            C_b = cp.CosmoPie(cosmo_b,p_space=cosmo_fid['p_space'],camb_params=camb_params)
            #avoid unnecessarily recomputing WMatchers for dark energy related parameters, and unnecessarily calling camb
            if pars[i] in cp.DE_SAFE and not override_safe:
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

def dp_dpar(C_fid,parameter,log_param_deriv,pmodel,epsilon,log_deriv=False):
    Cs = get_perturbed_cosmopies(C_fid,np.array([parameter]),np.array([epsilon]),np.array([log_param_deriv]),override_safe=True)
    if log_deriv:
#        pza_obj = mps.MatterPower(Cs[0][0],camb_params=C_fid.camb_params)
#        pzb_obj = mps.MatterPower(Cs[0][1],camb_params=C_fid.camb_params)
#        k_a = pza_obj.k
#        k_b = pzb_obj.k

        #k_a,pza = cpow.camb_pow(Cs[0,0].cosmology,camb_params=C_fid.camb_params)
        #k_b,pzb = cpow.camb_pow(Cs[0,1].cosmology,camb_params=C_fid.camb_params)
        
#        pza = pza_obj.get_matter_power(np.array([0.]),pmodel=pmodel)[:,0]
#        pzb = pzb_obj.get_matter_power(np.array([0.]),pmodel=pmodel)[:,0]

#        if not np.all(k_a==C_fid.k) or not np.all(k_b==C_fid.k):
#            pza = InterpolatedUnivariateSpline(k_a,pza,k=3)(C_fid.k)
#            pzb = InterpolatedUnivariateSpline(k_b,pzb,k=3)(C_fid.k)

#        halo_a = hf.HalofitPk(Cs[0,0],C_fid.k,pza)
#        halo_b = hf.HalofitPk(Cs[0,1],C_fid.k,pzb)

#        #pza = (2.*np.pi**2/C_fid.k**3*halo_a.D2_NL(k_fid,0.).T).T
#        #pzb = (2.*np.pi**2/C_fid.k**3*halo_b.D2_NL(k_fid,0.).T).T
        
#        pza = halo_a.D2_NL(k_fid,0.)
#        pzb = halo_b.D2_NL(k_fid,0.)
#        print pza.shape


        #pza = k_a**3/(2*np.pi**2.)*pza
        #pzb = k_b**3/(2*np.pi**2.)*pzb
        pza = C_fid.k**3/(2.*np.pi**2.)*Cs[0][0].P_lin.get_matter_power(np.array([0.]),pmodel=pmodel)
        pzb = C_fid.k**3/(2.*np.pi**2.)*Cs[0][1].P_lin.get_matter_power(np.array([0.]),pmodel=pmodel)
        result = (np.log(pza)-np.log(pzb))/(2*epsilon)
    else:
        result = (Cs[0][0].P_lin.get_matter_power(np.array([0.]),pmodel=pmodel)-Cs[0][1].P_lin.get_matter_power(np.array([0.]),pmodel=pmodel))/(2*epsilon)
    return result
#P_a is fiducial power spectrum 
#parameter can be H0,Omegabh2,Omegach2,Omegak,ns,sigma8,h
#include others such as omegam, w
#TODO need to change G_norm or does not work properly
#def dp_dpar(k_fid,zs,C,parameter,pmodel='linear',epsilon=0.0001,log_param_deriv=False,fpt=None,fpt_params=defaults.fpt_params,camb_params=defaults.camb_params,dp_params=defaults.dp_params):
#    cosmo_fid = C.cosmology.copy()
#    #print cosmo_fid
#    #skip getting sigma8 if possible because it is slow
#    if not parameter=='sigma8' and ('sigma8' not in cp.P_SPACES[cosmo_fid['p_space']]):
#        cosmo_fid['skip_sigma8'] = True
#        camb_params_use = camb_params.copy()
#        camb_params_use['force_sigma8'] = False
#    cosmo_a = get_perturbed_cosmology(cosmo_fid,parameter,epsilon,log_param_deriv=log_param_deriv)
#    #print "cosmo_a",cosmo_a
#    cosmo_b = get_perturbed_cosmology(cosmo_fid,parameter,-epsilon,log_param_deriv=log_param_deriv)
#    #print "cosmo_b",cosmo_b
#    #get perturbed linear power spectra from camb
#    k_a,P_a = cpow.camb_pow(cosmo_a,camb_params=camb_params_use) 
#    k_b,P_b = cpow.camb_pow(cosmo_b,camb_params=camb_params_use) 
#    if not np.all(k_a==k_fid) or not(np.all(k_b==k_fid)):
#        print "adapting k grid"
#        P_a = InterpolatedUnivariateSpline(k_a,P_a,k=1)(k_fid)
#        P_b = InterpolatedUnivariateSpline(k_b,P_b,k=1)(k_fid)
#    if pmodel=='linear':
#        pza = np.outer(P_a,C.G_norm(zs)**2) 
#        pzb = np.outer(P_b,C.G_norm(zs)**2) 
#    elif pmodel=='halofit':
#        halo_a = hf.HalofitPk(C,k_fid,P_a)
#        halo_b = hf.HalofitPk(C,k_fid,P_b)
#        #TODO make D2_NL support vector z syntax
#        pza = (2.*np.pi**2/k_fid**3*halo_a.D2_NL(k_fid,zs).T).T
#        pzb = (2.*np.pi**2/k_fid**3*halo_b.D2_NL(k_fid,zs).T).T
#    elif pmodel=='fastpt':
#        if fpt is None:
#            fpt = FASTPT.FASTPT(k_fid,fpt_params['nu'],low_extrap=fpt_params['low_extrap'],high_extrap=fpt_params['high_extrap'],n_pad=fpt_params['n_pad'])
#        #TODO maybe make fastpt support vector z if it doesn't already
#        p_lin_a = np.outer(P_a,C.G_norm(zs)**2) #TODO C needs to change properly
#        p_lin_b = np.outer(P_b,C.G_norm(zs)**2) 
#        one_loop_a = fpt.one_loop(P_a,C_window=fpt_params['C_window'])
#        one_loop_b = fpt.one_loop(P_b,C_window=fpt_params['C_window'])
#        pza = p_lin_a+np.outer(one_loop_a,C.G_norm(zs)**4)
#        pzb = p_lin_b+np.outer(one_loop_b,C.G_norm(zs)**4)
#        #pza = np.zeros_like(p_lin_a)
#        #pzb = np.zeros_like(p_lin_b)
#        #for itr in xrange(0,zs.size):
#        #    pza[:,itr] = p_lin_a[:,itr] + fpt.one_loop(p_lin_a[:,itr],C_window=fpt_params['C_window'])
#        #    pzb[:,itr] = p_lin_b[:,itr] + fpt.one_loop(p_lin_b[:,itr],C_window=fpt_params['C_window'])            
#    else:
#                raise ValueError('invalid pmodel option \''+str(pmodel)+'\'')
#
#    if dp_params['use_k3p']:
#        pza = k_a**3*pza/(2.*np.pi**2)
#        pzb = k_b**3*pzb/(2.*np.pi**2)
#
#    #TODO I'm not sure if interpolation like this is correct here. 
#    if dp_params['log_deriv_direct']:
#        dp = (np.log(pza)-np.log(pzb))/(2.*epsilon)
##    elif dp_params['log_deriv_indirect']:
##        dp = (pza-pzb)/(2.*epsilon*P_fid)
#    else:
#        dp = (pza-pzb)/(2.*epsilon)
#    return dp,pza,pzb 
if __name__=="__main__":
    
        #d = np.loadtxt('camb_m_pow_l.dat')
        #k_in = d[:,0]
        zs = np.arange(0.1,1.0,0.1)


        from scipy.interpolate import InterpolatedUnivariateSpline
        do_plot1=False
        do_plot2=True
        do_plot3=False
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

            C=cp.CosmoPie(defaults.cosmology.copy(),p_space='basic',camb_params=camb_params)
            cosmo_fid=C.cosmology
            #k_fid,P_fid = cpow.camb_pow(cosmo_fid,camb_params=camb_params)
            P_fid = mps.MatterPower(C,camb_params=camb_params)
            C.P_lin = P_fid
            C.k= P_fid.k
            k_fid = P_fid.k

            dp_params = defaults.dp_params.copy()
            dp_params['use_k3p']=False
            dp_params['log_deriv_direct']=True

            #attempt to replicate leftmost subplot at top of page 6 in Neyrinck 2011 arxiv:1105.2955v2
            dp_ch2 = dp_dpar(C,'Omegamh2',log_param_deriv=True,pmodel=pmodel_use,epsilon=epsilon)
            dp_bh2 = dp_dpar(C,'Omegabh2',log_param_deriv=True,pmodel=pmodel_use,epsilon=epsilon)
            dp_s8 = dp_dpar(C,'sigma8',log_param_deriv=True,pmodel=pmodel_use,epsilon=epsilon)
            dp_ns = dp_dpar(C,'ns',log_param_deriv=False,pmodel=pmodel_use,epsilon=epsilon)
            min_index = 0
            max_index = np.argmin(k_fid<0.5)

            dlnp_ch2 = dp_ch2/P_fid.get_matter_power(np.array([0.]),pmodel=pmodel_use)
            dlnp_ns = dp_ns/P_fid.get_matter_power(np.array([0.]),pmodel=pmodel_use)
            dlnp_s8 = dp_s8/P_fid.get_matter_power(np.array([0.]),pmodel=pmodel_use)
            dlnp_bh2 = dp_bh2/P_fid.get_matter_power(np.array([0.]),pmodel=pmodel_use)

            plt.semilogx(k_fid[min_index:max_index],dlnp_ns[min_index:max_index,0])
            plt.semilogx(k_fid[min_index:max_index],dlnp_s8[min_index:max_index,0]/2.)
            plt.semilogx(k_fid[min_index:max_index],dlnp_ch2[min_index:max_index,0])
            plt.semilogx(k_fid[min_index:max_index],dlnp_bh2[min_index:max_index,0])
            plt.legend(['ns','log(sigma8^2)','log(Omegach2)','log(Omegabh2)'],loc=4)
            plt.xlabel('k')
            plt.xlim([0.02,0.5])
            plt.ylim([-2.5,1.5])
            plt.ylabel('dln(P)/dtheta')
            plt.title('camb linear power spectrum response functions')
            plt.grid()
            plt.show()
                 
        if do_plot2:
            camb_params = defaults.camb_params.copy()
            pmodel_use = 'halofit'
            #camb_params['minkh'] = 0.01
            #camb_params['maxkh'] = 50.0
            #camb_params['kmax'] = 100.0
            camb_params['leave_h'] = False
            camb_params['force_sigma8'] = False
            dp_params = defaults.dp_params.copy()
            dp_params['use_k3p']=True
            dp_params['log_deriv_direct']=True
            epsilon = 0.0001
            cosmo_fid = defaults.cosmology.copy()
            cosmo_fid['h']=0.7;cosmo_fid['ns']=0.96;cosmo_fid['Omegamh2']=0.14014;cosmo_fid['Omegabh2']=0.02303;cosmo_fid['Omegakh2']=0.
            cosmo_fid['sigma8']=0.82
            cosmo_fid['LogAs']=-19.9229844537
            cosmo_fid['mnu'] = 0.06
            cosmo_fid = cp.strip_cosmology(cosmo_fid,'lihu',overwride=['tau','Yp','mnu'])
            cosmo_fid['skip_sigma8']=True
            cosmo_fid = cp.add_derived_pars(cosmo_fid)
            C=cp.CosmoPie(cosmology=cosmo_fid,p_space='lihu',camb_params=camb_params)
            #k_fid,P_fid = cpow.camb_pow(cosmo_fid,camb_params=camb_params)
            P_fid = mps.MatterPower(C,camb_params=camb_params)
            C.P_lin = P_fid
            C.k= P_fid.k
            k_fid = P_fid.k
            #attempt to replicate leftmost subplot at top of page 6 in Neyrinck 2011 arxiv:1105.2955v2
            epsilon_h =0.001
            dlnp_h = dp_dpar(C,'h',log_param_deriv=False,pmodel=pmodel_use,epsilon=epsilon_h,log_deriv=True)
            #dlnp_h,pza,pzb = dp_dpar(k_fid,np.array([0.]),C,'h',pmodel=pmodel_use,epsilon=epsilon_h,camb_params=camb_params,dp_params=dp_params)
            #dp_h = k_fid**3*dp_h/(2*np.pi**2)
            epsilon_LogAs = 0.001 
            dlnp_LogAs = dp_dpar(C,'LogAs',log_param_deriv=False,pmodel=pmodel_use,epsilon=epsilon_LogAs,log_deriv=True)
            #dlnp_LogAs,pza,pzb = dp_dpar(k_fid,np.array([0.]),C,'LogAs',pmodel=pmodel_use,epsilon=epsilon_h,camb_params=camb_params,dp_params=dp_params)
            epsilon_ns = 0.001
            dlnp_ns = dp_dpar(C,'ns',log_param_deriv=False,pmodel=pmodel_use,epsilon=epsilon_ns,log_deriv=True)
            #dlnp_ns,pza,pzb = dp_dpar(k_fid,np.array([0.]),C,'ns',pmodel=pmodel_use,epsilon=epsilon_h,camb_params=camb_params,dp_params=dp_params)
            min_index = 0
            #max_index = np.argmin(k_fid<1.1)
            max_index = k_fid.size
         
            #dlnp_h = dp_h/P_fid.get_matter_power(np.array([0.]),pmodel=pmodel_use)
            #dlnp_LogAs = dp_LogAs/P_fid.get_matter_power(np.array([0.]),pmodel=pmodel_use)
            #dlnp_ns = dp_ns/P_fid.get_matter_power(np.array([0.]),pmodel=pmodel_use)

            match_h = np.loadtxt('test_inputs/dp_dpar/lihu_match_h.dat')
            dlnp_h_lihu = 10.**match_h[:,1] 
            k_h_lihu = 10.**match_h[:,0]
            dlnp_h_interp = InterpolatedUnivariateSpline(k_h_lihu,dlnp_h_lihu,k=3,ext=0)(k_fid)

            match_LogAs = np.loadtxt('test_inputs/dp_dpar/lihu_match_lnAs.dat')
            dlnp_LogAs_lihu = 10.**match_LogAs[:,1] 
            k_LogAs_lihu = 10.**match_LogAs[:,0]
            dlnp_LogAs_interp = InterpolatedUnivariateSpline(k_LogAs_lihu,dlnp_LogAs_lihu,k=3,ext=0)(k_fid)

            match_ns = np.loadtxt('test_inputs/dp_dpar/lihu_match_ns.dat')
            dlnp_ns_lihu = 10.**match_ns[:,1] 
            k_ns_lihu = 10.**match_ns[:,0]
            dlnp_ns_interp = InterpolatedUnivariateSpline(k_ns_lihu,dlnp_ns_lihu,k=3,ext=0)(k_fid)
            
            #plt.semilogx(k_fid[min_index:max_index],dp_h[min_index:max_index,0])
            plt.loglog(k_fid[min_index:max_index]/C.cosmology['h'],np.abs(dlnp_h[min_index:max_index]))
            plt.loglog(k_fid[min_index:max_index]/C.cosmology['h'],np.abs(dlnp_ns[min_index:max_index]))
            #plt.loglog(k_fid[min_index:max_index]/cosmo_fid['h'],np.abs(dp_h[min_index:max_index,0]/P_fid[min_index:max_index]))
            #plt.loglog(k_fid[min_index:max_index]/cosmo_fid['h'],np.abs(dp_ns[min_index:max_index,0]/P_fid[min_index:max_index]))
            plt.loglog(k_fid[min_index:max_index]/C.cosmology['h'],np.abs(dlnp_LogAs[min_index:max_index]))
            plt.loglog(k_fid[min_index:max_index],dlnp_h_interp)
            plt.loglog(k_fid[min_index:max_index],dlnp_LogAs_interp)
            plt.loglog(k_fid[min_index:max_index],dlnp_ns_interp)
            plt.legend(['h','ns','As','h pred','ns pred','As pred'],loc=4)
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
