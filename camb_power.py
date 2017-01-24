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
        pars.InitPower.set_params(ns=cosmology['ns'],As=cosmology['As'])
  
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
        if camb_params['leave_h']:
            return kh,pk[0]*cosmology['sigma8']**2/results.get_sigma8()**2
        else:
            return kh*cosmology['h'],pk[0]*cosmology['sigma8']**2/results.get_sigma8()**2/cosmology['h']**3

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
    pow_test = True
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
