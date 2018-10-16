"""wrapper for camb to interface with our cosmology conventions"""
from __future__ import division,print_function,absolute_import
from builtins import range
from warnings import warn
import camb

#Note force_sigma8 forces sigma8_nl(z) to be the sigma8 in the cosmology, not sigma8_lin(z=0.), which may not be ideal behavior
def camb_pow(cosmology,camb_params,zbar=0.,nonlinear_model=camb.model.NonLinear_none):
    """get camb linear power spectrum
        inputs:
            cosmology: can vary H0, Omegabh2,Omegach2,Omegak,mnu,As,ns,OmegaL and w
            zbar: zs at which to get the power spectrum
            camb_params: some parameters controling k
            nonlinear_model: overwride and do something other than linear if set
    """
    if camb_params is None:
        raise ValueError('need camb params')
    pars = camb.CAMBparams()
    #ignores tau and YHe for now
    pars.set_cosmology(H0=cosmology['H0'],ombh2=cosmology['Omegabh2'],omch2=cosmology['Omegach2'],omk=cosmology['Omegak'],mnu=cosmology['mnu']) 
    pars.set_accuracy(camb_params['accuracy'],camb_params['accuracy'],camb_params['accuracy'])
    if cosmology.get('As') is not None:
        pars.InitPower.set_params(ns=cosmology['ns'],As=cosmology['As'],pivot_scalar=camb_params['pivot_scalar'])
    else:
        pars.InitPower.set_params(ns=cosmology['ns'])

    #pars.set_for_lmax(2500,lens_potential_accuracy=1)

    #pars.omegav=cosmology['OmegaL']
    if cosmology['de_model'] == 'w0wa':
        assert cosmology['w0']==cosmology['w']
    pars.set_dark_energy(cosmology['w']) #re-set defaults

    #Not non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=[zbar], kmax=camb_params['kmax'])
    #Linear spectra
    if nonlinear_model!=camb.model.NonLinear_none:
        camb.set_halofit_version('takahashi')
    #camb.set_halofit_version('original')
    pars.NonLinear = nonlinear_model
    results = camb.get_results(pars)
    kh, _, pk = results.get_matter_power_spectrum(minkh=camb_params['minkh'], maxkh=camb_params['maxkh'], npoints=camb_params['npoints'])
    if camb_params['force_sigma8']:
        sigma8 = cosmology['sigma8']
        pk[0] = pk[0]*sigma8**2/results.get_sigma8()**2
    elif camb_params['return_sigma8']:
        sigma8 = results.get_sigma8()[0]
    else:
        sigma8 = None

    pars.set_dark_energy() #reset defaults

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

def camb_sigma8(cosmology,camb_params=None):
    """get camb sigma8
        inputs:
            cosmology: can vary H0, Omegabh2,Omegach2,Omegak,mnu,As,ns,OmegaL and w
            camb_params: some parameters controling k
    """
    if camb_params is None:
        raise ValueError('need camb params')
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmology['H0'], ombh2=cosmology['Omegabh2'], omch2=cosmology['Omegach2'],omk=cosmology['Omegak'],mnu=cosmology['mnu'])
    if 'As' in cosmology:
        pars.InitPower.set_params(ns=cosmology['ns'],As=cosmology['As'])
    else:
        warn('cannot compute sigma8 reliably without input As')
        pars.InitPower.set_params(ns=cosmology['ns'])
    pars.set_dark_energy(cosmology['w']) #re-set defaults
    pars.omegav = cosmology['OmegaL']
    pars.set_matter_power(redshifts=[0.], kmax=camb_params['kmax'])
    results = camb.get_results(pars)
    sigma8 = results.get_sigma8()[0]
    pars.set_dark_energy() #re-set defaults
    return sigma8
