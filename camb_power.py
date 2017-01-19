import sys, platform, os
from time import time
import numpy as np
import camb
from camb import model, initialpower
import defaults

def camb_pow(cosmology,camb_params=defaults.camb_params):
#Now get matter power spectra and sigma8 at redshift 0 and 0.8
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=cosmology['H0'], ombh2=cosmology['Omegabh2'], omch2=cosmology['Omegach2'],omk=cosmology['Omegak'],tau=cosmology['tau'],YHe=cosmology['Yp'])

        pars.omegav=cosmology['OmegaL']
        pars.set_dark_energy() #re-set defaults
        pars.InitPower.set_params(ns=cosmology['ns'],As=cosmology['As'])

        #Not non-linear corrections couples to smaller scales than you want
        #TODO kmax here creates problems above this scale but it is very slow to increase it.
        pars.set_matter_power(redshifts=[0.], kmax=defaults.camb_params['kmax'])
        #Linear spectra
        pars.NonLinear = model.NonLinear_none
        results = camb.get_results(pars)
        kh, z, pk = results.get_matter_power_spectrum(minkh=camb_params['minkh'], maxkh=camb_params['maxkh'], npoints = camb_params['npoints'])

        return kh*cosmology['h'],pk[0]*cosmology['sigma8']**2/results.get_sigma8()**2/cosmology['h']**3
if __name__=='__main__':
    kh,pk = camb_pow(defaults.cosmology)
