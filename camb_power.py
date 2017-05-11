from time import time
import numpy as np
import camb
from camb import model, initialpower
import defaults
#TODO may want to investigate possible halofit discrepancy
def camb_pow(cosmology,zbar=0.,camb_params=defaults.camb_params,nonlinear_model=model.NonLinear_none):
#Now get matter power spectra and sigma8 at redshift 0 and 0.8
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=cosmology['H0'], ombh2=cosmology['Omegabh2'], omch2=cosmology['Omegach2'],omk=cosmology['Omegak'],mnu=cosmology['mnu']) #ignores tau and YHe for now

        if cosmology.get('As') is not None:
            pars.InitPower.set_params(ns=cosmology['ns'],As=cosmology['As'])
        else:
            pars.InitPower.set_params(ns=cosmology['ns'])
  
        #pars.set_for_lmax(2500,lens_potential_accuracy=1)

        #pars.omegav=cosmology['OmegaL']

        pars.set_dark_energy(cosmology['w']) #re-set defaults

        #Not non-linear corrections couples to smaller scales than you want
        #TODO kmax here creates problems above this scale but it is very slow to increase it.
        pars.set_matter_power(redshifts=[zbar], kmax=camb_params['kmax'])
        #Linear spectra
        if not nonlinear_model==model.NonLinear_none:
            camb.set_halofit_version('takahashi')
        #camb.set_halofit_version('original')
        pars.NonLinear = nonlinear_model
        results = camb.get_results(pars)
        kh, z, pk = results.get_matter_power_spectrum(minkh=camb_params['minkh'], maxkh=camb_params['maxkh'], npoints = camb_params['npoints'])
        if camb_params['force_sigma8']:
            sigma8 = cosmology['sigma8']
            pk[0] = pk[0]*sigma8**2/results.get_sigma8()**2
        elif camb_params['return_sigma8']:
            sigma8=results.get_sigma8()[0]
        else:
            sigma8=None

        pars.set_dark_energy() #reset defaults
        
       # if camb_params['leave_h']:
       #     return kh,pk[0],sigma8,results
       # else:
       #     return kh*cosmology['h'],pk[0]/cosmology['h']**3,sigma8,results
        #TODO check little h again
        if camb_params['leave_h']:
            if camb_params['return_sigma8']:
                return kh,pk[0],sigma8
            else:
                return kh,pk[0]
        else:
            if camb_params['return_sigma8']:
                return kh*cosmology['h'],pk[0]/cosmology['h']**3,sigma8
            else:
                return kh*cosmology['h'],pk[0]/cosmology['h']**3

#just get sigma8 for an input cosmology
def camb_sigma8(cosmology,camb_params=defaults.camb_params):
    pars = camb.CAMBparams()
    #TODO see about omega r, including tau and YHe
    pars.set_cosmology(H0=cosmology['H0'], ombh2=cosmology['Omegabh2'], omch2=cosmology['Omegach2'],omk=cosmology['Omegak'])
    pars.InitPower.set_params(ns=cosmology['ns'],As=cosmology['As'])
  
    pars.omegav=cosmology['OmegaL']
    pars.set_dark_energy() #re-set defaults
    pars.set_matter_power(redshifts=[0.], kmax=camb_params['kmax'])
    results = camb.get_results(pars)
    return results.get_sigma8()[0]

    
if __name__=='__main__':
    import cosmopie as cp
    transfer_test = False
    if transfer_test:
        cosmo_a = defaults.cosmology.copy()
        camb_params = defaults.camb_params.copy()
        camb_params['force_sigma8']=False
        camb_params['leave_h']=True
        camb_params['kmax'] = 200.
        camb_params['maxkh'] = 200.

        transfer_initial_test=False
        if transfer_initial_test:
            cosmo_b = defaults.cosmology.copy()
            cosmo_b['ns']+=0.01
            cosmo_c = defaults.cosmology.copy()
            cosmo_c['As']+=0.01*cosmo_a['As']
            k_a,P_a,_,res_a=camb_pow(cosmo_a,camb_params=camb_params)
            k_b,P_b,_,res_b=camb_pow(cosmo_b,camb_params=camb_params)

            inflation_params = initialpower.InitialPowerParams()
            inflation_params.set_params(ns=cosmo_b['ns'],As=cosmo_b['As'])
            res_b.power_spectra_from_transfer(inflation_params)
            k_b_alt, _, P_b_alt=res_b.get_matter_power_spectrum(minkh=camb_params['minkh'], maxkh=camb_params['maxkh'], npoints = camb_params['npoints'])

            k_c,P_c,_,res_c=camb_pow(cosmo_c,camb_params=camb_params)

            inflation_params = initialpower.InitialPowerParams()
            inflation_params.set_params(ns=cosmo_c['ns'],As=cosmo_c['As'])
            res_b.power_spectra_from_transfer(inflation_params)
            k_c_alt, _, P_c_alt=res_c.get_matter_power_spectrum(minkh=camb_params['minkh'], maxkh=camb_params['maxkh'], npoints = camb_params['npoints'])

            assert(np.all(P_b_alt==P_b))
            assert(np.all(P_c_alt==P_c))

        cosmo_d = cosmo_a.copy()
        cosmo_d['w'] =-0.99
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=cosmo_a['H0'], ombh2=cosmo_a['Omegabh2'], omch2=cosmo_a['Omegach2'],omk=cosmo_a['Omegak'])
        pars.InitPower.set_params(ns=cosmo_a['ns'],As=cosmo_a['As'])
  
        z_grid = np.arange(0.,6.,0.3)
        pars.omegav=cosmo_a['OmegaL']
        pars.set_dark_energy(cosmo_a['w']) #re-set defaults
        pars.set_matter_power(redshifts=z_grid, kmax=camb_params['kmax'])
        pars.NonLinear = model.NonLinear_none
        results = camb.get_results(pars)
        kh_as, z_as, P_as = results.get_matter_power_spectrum(minkh=camb_params['minkh'], maxkh=camb_params['maxkh'], npoints = camb_params['npoints'])
        C_a = cp.CosmoPie(cosmology=cosmo_a,k=kh_as,P_lin=P_as[0],camb_params=camb_params,z_max=20.01,z_space=0.001)
        P_arecs = np.outer(C_a.G_norm(z_grid)**2/C_a.G_norm(z_grid[0])**2,P_as[0])
        P_agrow=P_as/P_as[0]
        do_d=False
        if do_d:
            pars.omegav=cosmo_d['OmegaL']
            pars.set_dark_energy(cosmo_d['w']) #re-set defaults
            pars.set_matter_power(redshifts=z_grid, kmax=camb_params['kmax'])
            pars.NonLinear = model.NonLinear_none
            results = camb.get_results(pars)
            kh_ds, z_ds, P_ds = results.get_matter_power_spectrum(minkh=camb_params['minkh'], maxkh=camb_params['maxkh'], npoints = camb_params['npoints'])
            pars.set_dark_energy()
            C_d = cp.CosmoPie(cosmology=cosmo_d,k=kh_as,P_lin=P_ds[0],camb_params=camb_params,z_max=40.01)
            P_drecs = np.outer(C_d.G_norm(z_grid)**2/C_d.G_norm(z_grid[0])**2,P_ds[0])
            P_dgrow=P_ds/P_ds[0]

        import matplotlib.pyplot as plt
        #assert(np.all(P_dgrow==P_agrow))
        #plt.semilogx(kh_ds,1.-(P_dgrow/P_agrow).T)
        plt.semilogx(kh_as,(1.-P_arecs/P_as).T)
        #plt.semilogx(kh_as,(1.-P_drecs/P_ds).T)
        plt.show()
        

    pow_test = False
    if pow_test:
        kh1,pk1 = camb_pow(defaults.cosmology,zbar=9.)
        kh2,pk2 = camb_pow(defaults.cosmology,zbar=6.)
        kh3,pk3 = camb_pow(defaults.cosmology,zbar=3.)
        kh4,pk4 = camb_pow(defaults.cosmology,zbar=0.)
        import matplotlib.pyplot as plt
        plt.loglog(kh1,pk1/pk4)
        plt.loglog(kh1,pk2/pk4)
        plt.loglog(kh1,pk3/pk4)

    fit_test = False
    if fit_test:
        from cosmopie import add_derived_parameters,strip_cosmology
        cosmo_start = defaults.cosmology.copy()
        cosmo_start['h']=0.7;cosmo_start['ns']=0.96;cosmo_start['Omegamh2']=0.14014;cosmo_start['Omegabh2']=0.02303;cosmo_start['Omegakh2']=0.;cosmo_start['OmegaL']=0.67
        cosmo_start['sigma8']=0.82;cosmo_start['OmegaLh2']=0.32683
        nA = 3
        As = np.zeros(nA)
        s8s = np.zeros(nA)
        epsilon = 0.04
        for itr in range(0,nA):
            cosmo_strip = strip_cosmology(cosmo_start,p_space='jdem')
            cosmo_fid = add_derived_parameters(cosmo_strip)
            As[itr] = cosmo_fid['As'] 
            s8s[itr] = camb_sigma8(cosmo_fid)
            cosmo_start['LogAs']+=epsilon

        print s8s
        from scipy.interpolate import interp1d
        print np.log(interp1d(s8s,As)(0.82))

    
    w_test=False
    if w_test:
        camb_params = defaults.camb_params.copy()
        camb_params['force_sigma8']=False
        camb_params['leave_h']=False
        camb_params['kmax'] = 400.
        camb_params['maxkh'] = 400.
        camb_params['npoints'] = 10000
        camb_params['return_sigma8'] = True

        cosmo_start = defaults.cosmology.copy()
        import cosmopie as cp
        cosmo_start = cp.add_derived_parameters(cosmo_start,p_space='jdem')
        cosmo_start['de_model']='constant_w'
        cosmo_start['w']=-1
        cosmo_start['mnu'] = 0.
        w_step = -0.4
        ws = np.arange(-0.6,-1.9,w_step)
        zs = np.array([0.1])
        kls= np.zeros((ws.size,camb_params['npoints']))
        Pls= np.zeros((ws.size,camb_params['npoints']))
        knls= np.zeros((ws.size,camb_params['npoints']))
        Pnls= np.zeros((ws.size,camb_params['npoints']))
        Peffs= np.zeros((ws.size,camb_params['npoints']))
        Phfs= np.zeros((ws.size,camb_params['npoints']))
        Pfpts= np.zeros((ws.size,camb_params['npoints']))
        Pfpt2s= np.zeros((ws.size,camb_params['npoints']))
        Pfpt3s= np.zeros((ws.size,camb_params['npoints']))
        Phf2s= np.zeros((ws.size,camb_params['npoints']))
        Phf3s= np.zeros((ws.size,camb_params['npoints']))
        Phf4s= np.zeros((ws.size,camb_params['npoints']))
        sigma8ls= np.zeros((ws.size))
        sigma8nls= np.zeros((ws.size))
        Gs = np.zeros(ws.size)
        Gnorms = np.zeros(ws.size)
        Cs = np.zeros(ws.size,dtype=object)
        C4s = np.zeros(ws.size,dtype=object)
        hfs = np.zeros(ws.size,dtype=object)
        hf2s = np.zeros(ws.size,dtype=object)
        hf4s = np.zeros(ws.size,dtype=object)
        import halofit as hf
        import FASTPTcode.FASTPT as FASTPT
        import w_matcher
        fpt_params = defaults.fpt_params
        fpt_params['n_pad']=3000
        fpt_params['high_extrap']=8
        fpt_params['low_extrap']=-8
        k_0,P_0,sigma8_0 =camb_pow(cosmo_start,camb_params=camb_params,nonlinear_model=model.NonLinear_none) 
        C_0=cp.CosmoPie(cosmo_start,needs_power=False)
        G_0 = C_0.G(0.)
        halofit_params =defaults.halofit_params.copy()
        halofit_params['cutoff']=False
        hf_0 = hf.halofitPk(C_0,k_0,P_0,halofit_params=halofit_params)

        n_w = ws.size
        wm = w_matcher.WMatcher(C_0)
        w4s = np.zeros((n_w,zs.size))
        mult4s = np.zeros((n_w,zs.size))
        scales = np.zeros((n_w,zs.size))
        kq2s = np.zeros((n_w))
        sqs = np.zeros((n_w))
        alphaqs = np.zeros((n_w))
        transfer_mods = np.zeros((n_w,k_0.size))

        print "begin loop"
        for itr in range(0,ws.size):
            cosmo_start['w'] = ws[itr]
            Cs[itr]=cp.CosmoPie(cosmo_start,needs_power=False)
            Gs[itr] = Cs[itr].G(0.)
            Gnorms[itr] = Cs[itr].G_norm(0.)
            kls[itr],Pls[itr],sigma8ls[itr] = camb_pow(cosmo_start,camb_params=camb_params,nonlinear_model=model.NonLinear_none)
            if itr==0:
                fpt = FASTPT.FASTPT(kls[0],fpt_params['nu'],low_extrap=fpt_params['low_extrap'],high_extrap=fpt_params['high_extrap'],n_pad=fpt_params['n_pad'])
            knls[itr],Pnls[itr],sigma8nls[itr] = camb_pow(cosmo_start,zbar=1.,camb_params=camb_params,nonlinear_model=model.NonLinear_both)
            hfs[itr] = hf.halofitPk(Cs[itr],kls[itr],Pls[itr],halofit_params=halofit_params)
            Phfs[itr] = 2*np.pi**2*(hfs[itr].D2_NL(kls[itr],1.0).T/kls[itr]**3).T
            #Pfpts[itr] =Pls[0]*((Gs[itr]/Gs[0])**2)*Gnorms[itr]**2+fpt.one_loop(Pls[0]*(Gs[itr]/Gs[0])**2,C_window=fpt_params['C_window'])*Gnorms[itr]**4
            Pfpts[itr] =P_0*(Gs[itr]/G_0)**2+fpt.one_loop(P_0,C_window=fpt_params['C_window'])*(Gs[itr]/G_0)**4
            Pfpt2s[itr] =Pls[itr]*(Gnorms[itr])**2+fpt.one_loop(Pls[itr],C_window=fpt_params['C_window'])*(Gnorms[itr])**4
            w4s[itr] = wm.match_w(Cs[itr],zs)
            mult4s[itr] = wm.match_growth(Cs[itr],zs,w4s[itr])
            scales[itr] = wm.match_scale(zs,w4s[itr])
            cosmo_eff = cosmo_start.copy()
            cosmo_eff['w'] = w4s[itr,0]
            cosmo_eff['de_model'] = 'constant_w'
            cosmo_eff['sigma8']*=np.sqrt(mult4s[itr,0])*np.sqrt(scales[itr])
            C4s[itr] = cp.CosmoPie(cosmo_eff,needs_power=False) 
            kq2s[itr] = np.abs((3.*C_0.H0/C_0.c)**2*(1.-w4s[itr])*(2.+2.*w4s[itr]-w4s[itr]*C_0.Omegam))
            sqs[itr] = (0.012-0.036*w4s[itr]-0.017/w4s[itr])*(1.-C_0.Omegam)+(0.098+0.029*w4s[itr]-0.085/w4s[itr])*np.log(C_0.Omegam)
            alphaqs[itr] = (-w4s[itr])**sqs[itr]
            transfer_mods[itr] = (alphaqs[itr]+alphaqs[itr]*(k_0**2/kq2s[itr]))/(1.+alphaqs[itr]*(k_0**2/kq2s[itr])) 
            #P_eff = P_0*mult4s[itr,0]*transfer_mods[itr]**2*sigma8ls[itr]**2/sigma8_0**2 #(sigma8ls[itr]**2/sigma8_0**2) should be interchangeable with scales[itr]
            Peffs[itr] = P_0*mult4s[itr,0]*scales[itr]*transfer_mods[itr]**2#*sigma8ls[itr]**2/sigma8_0**2 #(sigma8ls[itr]**2/sigma8_0**2) should be interchangeable with scales[itr]
            Pfpt3s[itr] =Peffs[itr]+fpt.one_loop(Peffs[itr],C_window=fpt_params['C_window'])
            #if itr>0:
            hf2s[itr] = hf.halofitPk(C4s[itr],kls[itr],Peffs[itr],halofit_params=halofit_params)
            hf4s[itr] = hf.halofitPk(C4s[itr],kls[itr],P_0*mult4s[itr,0],halofit_params=halofit_params)
            Phf2s[itr] = 2*np.pi**2*(hf2s[itr].D2_NL(kls[itr],1.0,w_overwride=True,fixed_w=w4s[itr],grow_overwride=False,fixed_growth=mult4s[itr]*C_0.G_norm(0.)**2).T/kls[itr]**3).T
            Phf3s[itr] = 2*np.pi**2*(hfs[itr].D2_NL(kls[itr],0.,w_overwride=True,fixed_w=w4s[itr]).T/kls[itr]**3).T
            Phf4s[itr] = 2*np.pi**2*(hf4s[itr].D2_NL(kls[itr],0.,w_overwride=False).T/kls[itr]**3).T
            #else:
            #    hf2s[itr] = hf.halofitPk(Cs[itr],kls[itr],P_0*mul)
            #    hf4s[itr] = hf.halofitPk(C4s[itr],kls[itr],P_0*mult4s[itr,0])
            #    Phf2s[itr] = 2*np.pi**2*(hf2s[itr].D2_NL(kls[itr],0.,w_overwride=False,fixed_w=w4s[itr],grow_overwride=True,fixed_growth=C_0.G_norm(0.)**2).T/kls[itr]**3).T
            #    Phf3s[itr] = 2*np.pi**2*(hfs[itr].D2_NL(kls[itr],0.,w_overwride=True,fixed_w=w4s[itr]).T/kls[itr]**3).T
            #    Phf4s[itr] = 2*np.pi**2*(hf4s[itr].D2_NL(kls[itr],0.,w_overwride=False).T/kls[itr]**3).T
        import matplotlib.pyplot as plt
        #ax = plt.subplot(111)
        #ax.set_ylim([0.95,(np.max((Ps/Ps[0]).T)-1)*1.1+1])
        #ax.semilogx(knls.T,(Pnls/Pnls[0]).T*Gs[0]/Gs)
        plt.semilogx(kls.T,(Phf2s/Phfs).T)
        #ax.semilogx(kls.T,(Pnls/Phfs).T)
        from scipy.integrate import cumtrapz
        print (cumtrapz(np.abs(Pnls-Phfs),kls,initial=0))[:,-1]
        print (cumtrapz(np.abs(Phf4s-Phfs),kls,initial=0))[:,-1]
        print (cumtrapz(np.abs(Pfpts-Phfs),kls,initial=0))[:,-1]
        from scipy.interpolate import InterpolatedUnivariateSpline

        InterpolatedUnivariateSpline(np.log(kls[0]),np.log(Pls[0])).derivative()(np.log(kls[0]))
        plt.show()
        
#plt.loglog(kls.T,(cumtrapz(np.abs(Pnls-Phfs),kls,initial=0)).T)
    #k_in,P_in=camb_pow(C.cosmology)
    #camb_params = {'npoints':10000,
                #'minkh':1.1e-4,
                #'maxkh':1e5,
                #'kmax':100.0, #may need to be higher for some purposes,like 100, but makes things slower
                #'leave_h':False,
                #'force_sigma8':True,
                #'return_sigma8':False
                #}
