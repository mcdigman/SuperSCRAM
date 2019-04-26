"""rotate between cosmological parametrizations"""
from __future__ import division,print_function,absolute_import
from builtins import range
import numpy as np

def rotate_jdem_to_lihu(f_set_in,C_in):
    """rotate between jdem and lihu"""
    if C_in.de_model == 'constant_w':
        jdem_pars = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w'])

        lihu_pars = np.array(['ns','Omegach2','Omegabh2','h','LogAs','w'])
        response_pars = np.array(['ns','Omegach2','Omegabh2','Omegamh2','OmegaLh2','h','LogAs','w'])
        response_derivs_lihu = np.array([[1.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,1.,-1.,0.,0.,0.],[0.,0.,1.,1.,-1.,0.,0.,0.],[0.,0.,0.,0.,2.*C_in.cosmology['h'],1.,0.,0.],[0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,1.]]).T
    elif C_in.de_model == 'w0wa':
        jdem_pars = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w0','wa'])

        lihu_pars = np.array(['ns','Omegach2','Omegabh2','h','LogAs','w0','wa'])
        response_pars = np.array(['ns','Omegach2','Omegabh2','Omegamh2','OmegaLh2','h','LogAs','w0','wa'])
        response_derivs_lihu = np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,1.,-1.,0.,0.,0.,0.],[0.,0.,1.,1.,-1.,0.,0.,0.,0.],[0.,0.,0.,0.,2.*C_in.cosmology['h'],1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.]]).T
    else:
        raise ValueError('unsupported de model '+str(C_in.de_model))


    project_jdem_to_lihu = np.zeros((lihu_pars.size,jdem_pars.size))

    for itr1 in range(0,lihu_pars.size):
        for itr2 in range(0,response_pars.size):
            if response_pars[itr2] in jdem_pars:
                name = response_pars[itr2]
                i = np.argwhere(jdem_pars==name)[0,0]
                project_jdem_to_lihu[itr1,i] = response_derivs_lihu[itr2,itr1]

    f_jdem_to_lihu = np.zeros_like(f_set_in)
    for i in range(0,f_jdem_to_lihu.size):
        f_jdem_to_lihu[i] = np.dot(project_jdem_to_lihu,np.dot(f_set_in[i],project_jdem_to_lihu.T))
    return f_jdem_to_lihu

def rotate_lihu_to_jdem(f_set_in,C_in):
    """rotate between lihu and jdem"""
    if C_in.de_model == 'constant_w':
        jdem_pars = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w'])

        lihu_pars = np.array(['ns','Omegach2','Omegabh2','h','LogAs','w'])
        response_pars = np.array(['ns','Omegach2','Omegabh2','Omegamh2','OmegaLh2','h','LogAs','w'])
        response_derivs_jdem = np.array([[1.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,1.,0.,1./(2.*C_in.cosmology['h']),0.,0.],[0.,-1.,1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,1./(2.*C_in.cosmology['h']),0.,0.],[0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,1.]]).T
    elif C_in.de_model == 'w0wa':
        jdem_pars = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w0','wa'])

        lihu_pars = np.array(['ns','Omegach2','Omegabh2','h','LogAs','w0','wa'])
        response_pars = np.array(['ns','Omegach2','Omegabh2','Omegamh2','OmegaLh2','h','LogAs','w0','wa'])
        response_derivs_jdem = np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,1.,0.,1./(2.*C_in.cosmology['h']),0.,0.,0.],[0.,-1.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,1./(2.*C_in.cosmology['h']),0.,0.,0.],[0.,0.,0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.]]).T
    else:
        raise ValueError('unsupported de model '+str(C_in.de_model))

    project_lihu_to_jdem = np.zeros((jdem_pars.size,lihu_pars.size))

    for itr1 in range(0,jdem_pars.size):
        for itr2 in range(0,response_pars.size):
            if response_pars[itr2] in lihu_pars:
                name = response_pars[itr2]
                i = np.argwhere(lihu_pars==name)[0,0]
                project_lihu_to_jdem[itr1,i] = response_derivs_jdem[itr2,itr1]

    f_lihu_to_jdem = np.zeros_like(f_set_in)
    for i in range(0,f_lihu_to_jdem.size):
        f_lihu_to_jdem[i] = np.dot(project_lihu_to_jdem,np.dot(f_set_in[i],project_lihu_to_jdem.T))
    return f_lihu_to_jdem
