import numpy as np
# default cosmology is Planck 2015 TT+lowP+lensing+ext (arxiv 1502.01589v3 page 32) 
#TODO set not derived paramters to none and derive
cosmology={'Omegabh2' :0.02227,
				'Omegach2' :0.1184,
                                'Omegab'   :0.0483037,#calculated
                                'Omegac'   : 0.25681, #calculated
				'Omegamh2' : 0.1413,
				'OmegaL'   : 0.6935,
                                'OmegaLh2' : 0.319732,#calculated
				'Omegam'   : .3065,
				'H0'       : 67.90, 
				'sigma8'   : .8154, 
				'h'        :.6790,#calculated 
				'Omegak'   : 0.0, # check on this value 
                                'Omegakh2' : 0.0,
				'Omegar'   : 0.0,
                                'ns'       : 0.9681,
                                'tau'      : 0.067,
                                '100thetamc': 1.04106,
                                'Yp'        :0.251,
                                'As'        : 2.143*10**-9,
                                'LogAs'     :-19.9619, #calculated
                                'w'         :-1.0,#not from planck
                                'de_model'  :'constant_w',#dark energy model
                                'zstar'     :1089.90#redshift at last scattering
                                }
#cosmology from jdem 2008 working group paper arxiv:0901.0721v1
cosmology_jdem={                'ns'       : 0.963,
				'Omegamh2' : 0.1326,
                                'Omegabh2' : 0.0227,
                                'Omegakh2' : 0.,
                                'OmegaLh2' : 0.3844,
                                'dGamma'    : 0.,#parameter we don't need
                                'dM'        :0.,#parameter we don't need
                                'LogG0'     :0.,#parameter we don't need
                                'LogAs'     :-19.9628,
                                'w'         :-1,
                                'de_model'  :'constant_w'
                                }
cosmology_cosmolike={'Omegabh2' :0.02227,
				'Omegach2' :0.1204,
				'Omegamh2' : 0.14267,
				'OmegaL'   : 0.685,
				'OmegaLh2'   : 0.310256,#calculdated
				'Omegam'   : .315,
				'H0'       : 67.3, 
				'sigma8'   : .829, 
				'h'        :.673, 
				'Omegak'   : 0.0, # check on this value 
                                'Omegakh2' : 0.0,
				'Omegar'   : 0.0,
                                'ns'       : 0.9603,
                                'tau'      : 0.067,
                                '100thetamc': 1.04106,
                                'Yp'        :None,
                                'As'        : 2.143*10**-9,
                                'LogAs'     : np.log(2.143*10**-9)}
cosmology_chiang={'Omegabh2' :0.023,
				'Omegach2' :0.1093,
				'Omegamh2' : 0.1323,
				'OmegaL'   : 0.73,
                                'OmegaLh2' : 0.3577,
				'Omegam'   : .27,
				'H0'       : 70., 
				'sigma8'   : .7913, 
				'h'        :0.7, 
				'Omegak'   : 0.0, # check on this value 
                                'Omegakh2' : 0.0,
				'Omegar'   : 0.0,
                                'ns'       : 0.95,
                                'tau'      : None,
                                '100thetamc':None,
                                'Yp'        :None,
                                'As'        : None,
                                'LogAs'   : None}
cosmology_cosmosis={'Omegabh2' :0.049,
				'Omegach2' :0.1188,
				'Omegamh2' : 0.14170,
				'OmegaL'   : .641,
                                'OmegaLh2' : 0.287745,
				'Omegam'   : .31,
				'H0'       : 67, 
				'sigma8'   : .81, 
				'h'        :.67, 
				'Omegak'   : 0.0, # check on this value 
                                'Omegakh2' : 0.0,
				'Omegar'   : 0.0,
                                'tau'      : None, # eventually fix to real cosmosis values
                                'Yp'       : None,
                                'As'        : 2.143*10**-9,
                                'ns'       : 0.9681,
                                'LogAs'   : np.log(2.143*10**-9)} #guess
lensing_params = {  'z_resolution'    :0.005, #fine resolution
                    'z_min_integral'  :0.005, #lowest z
                    'z_max_integral'  :2,#highest z
                    'pmodel_O'        :'halofit', #default method for finding p grid
                    'pmodel_dO_ddelta':'halofit', #default method for finding dp/ddeltabar grid
                    'pmodel_dO_dparameter':'halofit',#default method for finding dp/dparameter grid
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
dn_params = {   'M_cut'         : 10**12.5,
                'variable_cut'  : True}#(12.5)}
lw_observable_list = ['d_number_density']
lw_survey_params = {    'cross_bins': False} 
basis_params = {    'allow_caching'         :True,
                    'n_bessel_oversample'   :100000,
                    'k_max'                 :100.,
                    'k_min'                 :10**-4,
                    'n_radial_sample':100000 }
polygon_params = {'res_healpix':6,'n_double':30}
nz_params = {   'data_source'   :'./data/CANDELS-GOODSS2.dat',
                'i_cut'         :24,
                'area_sterad'   :0.0409650328530259/3282.80635,
                'smooth_sigma'  :0.04,
                'n_right_extend':4,
                'z_resolution'  :0.0001,
                'mirror_boundary':True}

hmf_params = {      'log10_min_mass'    :   6,
                    'log10_max_mass'    :   18,
                    'n_grid'            :   3000,
                    'z_resolution'      : 0.001,
                    'z_min'             : 0.,
                    'z_max'             : 3.}
fpt_params = {   'C_window':0.75,
                    'n_pad':1000,
                    'low_extrap':None,
                    'high_extrap':None,
                    'nu' :-2}
camb_params = {'npoints':10000,
                'minkh':1.1e-4,
                'maxkh':1e5,
                'kmax':100.0, #may need to be higher for some purposes,like 100, but makes things slower
                'leave_h':False,
                'force_sigma8':True,
                'return_sigma8':False
                }
dp_params = {'use_k3p':False,
            'log_deriv_direct':False,
            'log_deriv_indirect':False}
#amara refregier 2006 parameter forecast stuff
cosmopie_params = {'p_space':'overwride'}
