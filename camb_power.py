import sys, platform, os
from time import time
import numpy as np
import camb
from camb import model, initialpower
import defaults

def camb_pow(cosmology,zbar=0.,camb_params=defaults.camb_params):
#Now get matter power spectra and sigma8 at redshift 0 and 0.8
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=cosmology['H0'], ombh2=cosmology['Omegabh2'], omch2=cosmology['Omegach2'],omk=cosmology['Omegak'],tau=cosmology.get('tau'),YHe=cosmology.get('Yp'))
        if cosmology.get('As') is not None:
            pars.InitPower.set_params(ns=cosmology['ns'],As=cosmology['As'])
        else:
            pars.InitPower.set_params(ns=cosmology['ns'])
  
        pars.omegav=cosmology['OmegaL']
        #pars.set_for_lmax(2500,lens_potential_accuracy=1)

        pars.set_dark_energy() #re-set defaults

        #Not non-linear corrections couples to smaller scales than you want
        #TODO kmax here creates problems above this scale but it is very slow to increase it.
        pars.set_matter_power(redshifts=[zbar], kmax=camb_params['kmax'])
        #Linear spectra
        pars.NonLinear = model.NonLinear_none
        results = camb.get_results(pars)
        kh, z, pk = results.get_matter_power_spectrum(minkh=camb_params['minkh'], maxkh=camb_params['maxkh'], npoints = camb_params['npoints'])

        if camb_params['force_sigma8']:
            sigma8 = cosmology['sigma8']
            pk[0] = pk[0]*sigma8**2/results.get_sigma8()**2
        elif camb_params['return_sigma8']:
            sigma8=results.get_sigma8()[0]
        else:
            sigma8=None
        
        
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
    transfer_test = True
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
        import cosmopie as cp
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
        do_d=True
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

        import cosmopie as cp
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
    #k_in,P_in=camb_pow(C.cosmology)
    #np.savetxt('P_default.csv',np.array([k_in,P_in]),delimiter=",")
    #camb_params = {'npoints':10000,
                #'minkh':1.1e-4,
                #'maxkh':1e5,
                #'kmax':100.0, #may need to be higher for some purposes,like 100, but makes things slower
                #'leave_h':False,
                #'force_sigma8':True,
                #'return_sigma8':False
                #}
