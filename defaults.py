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
cosmology_cosmolike={'Omegabh2' :0.02227,
				'Omegach2' :0.1204,
				'Omegamh2' : 0.14267,
				'OmegaL'   : 0.685,
				'Omegam'   : .315,
				'H0'       : 67.3, 
				'sigma8'   : .829, 
				'h'        :.673, 
				'Omegak'   : 0.0, # check on this value 
				'Omegar'   : 0.0,
                                'ns'       : 0.9603,
                                'tau'      : 0.067,
                                '100thetamc': 1.04106,
                                'As'        : 2.143*10**-9}
cosmology_chiang={'Omegabh2' :0.023,
				'Omegach2' :0.1093,
				'Omegamh2' : 0.1323,
				'OmegaL'   : 0.73,
				'Omegam'   : .27,
				'H0'       : 70., 
				'sigma8'   : .7913, 
				'h'        :0.7, 
				'Omegak'   : 0.0, # check on this value 
				'Omegar'   : 0.0,
                                'ns'       : 0.95,
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
				'Omegar'   : 0.0,
                                'ns'       : 0.9681} #guess
lensing_params = {  'z_resolution'    :0.05, #fine resolution
                    'z_min_integral'  :0.01, #lowest z
                    'z_max_integral'  :2,#highes z
                    'pmodel_O'        :'halofit_nonlinear', #default method for finding p grid
                    'pmodel_dO_ddelta':'dc_halofit', #default method for finding dp/ddeltabar grid
                    'n_gal'           :118000000*4.,#118000000 galaxies/rad^2=10 galaxies/arcmin^2
                    'delta_l'         :1., #binning window
                    'sigma2_e'        :0.27**2*2, #other noise term
                    'sigma2_mu'       :1.2, #noise term for magnification
                    'smodel'          :'constant', #type, current options are 'gaussian','constant','cosmolike'
                    'z_min_dist'      :0.,
                    'z_max_dist'      :np.inf,
                    'zbar'            :0.1, #mean of gaussian source distribution
                    'sigma'           :0.4, #stdev of gaussian source distribution
                    'epsilon'         :0.0001} #small parameter 
sw_observable_list = ['len_shear_shear']
sw_survey_params = {    'needs_lensing'     : True,
                        'cross_bins'        : True}
dn_params = {   'M_cut'  : 10**12.5}#(12.5)}
lw_observable_list = ['d_number_density']
lw_survey_params = {    'cross_bins': False} 
basis_params = {    'allow_caching'         :True,
                    'n_bessel_oversample'   :100000,
                    'k_max'                 :100.,
                    'k_min'                 :10**-4,
                    'n_radial_sample':100000 }
polygon_params = {'res_healpix':6,'l_max':100}
