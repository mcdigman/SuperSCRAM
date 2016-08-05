import numpy as np
# default cosmology is Planck 2015 
cosmology={'Omegabh2' :0.02227,
				'Omegach2' :0.1184,
				'Omegamh2' : 0.1413,
				'OmegaL'   : 0.6935,
				'Omegam'   : .3065,
				'H0'       : 67.90, 
				'sigma8'   : .8154, 
				'h'        :.6790, 
				'Omegak'   : 0.0, # check on this value 
				'Omegar'   : 0.0,
                                'ns'       : 0.9681,
                                'tau'      : 0.067,
                                '100thetamc': 1.04106,
                                'As'        : 2.143*10**-9}
cosmology_cosmosis={'Omegabh2' :0.049,
				'Omegach2' :0.1188,
				'Omegamh2' : 0.14170,
				'OmegaL'   : .641,
				'Omegam'   : .31,
				'H0'       : 67, 
				'sigma8'   : .81, 
				'h'        :.67, 
				'Omegak'   : 0.0, # check on this value 
				'Omegar'   : 0.0}
lensing_params = {  'z_resolution'    :0.1,
                    'z_min_integral'  :0.1,
                    'pmodel_O'        :'halofit_nonlinear',
                    'pmodel_dO_ddelta':'dc_halofit',
                    'n_gal'           :1180000000,
                    'omega_s'         :np.pi/(3.*np.sqrt(2.)),
                    'delta_l'         :1.,
                    'sigma2_e'        :0.32,
                    'sigma2_mu'       :1.2,
                    'smodel'          :'gaussian',
                    'zbar'            :0.55,
                    'sigma'           :0.4,
                    'epsilon'         :0.0001} 
sw_observable_list = ['len_shear_shear']
sw_survey_params = {    'needs_lensing'     : True,
                        'cross_bins'        : False,
                        'observable_list'   : sw_observable_list}


