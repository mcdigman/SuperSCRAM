"""provides some useful default values"""
import numpy as np
from param_manager import PowerParamManager
# default cosmology is Planck 2015 TT+lowP+lensing+ext (arxiv 1502.01589v3 page 32)
cosmology = {   'Omegabh2':0.02227,
                'Omegach2':0.1190390665,#calculated, paper give 0.1184
                'Omegab'  :0.0483037300370249,#calculated
                'Omegac'  : 0.258196269962975, #calculated
                'Omegamh2': 0.1413090665,#calculated
                'OmegaL'  : 0.6935,
                'OmegaLh2': 0.3197319335,#calculated
                'Omegam'  : .3065,
                'H0'      : 67.90,
                'sigma8'  : 0.8296437384606381,#calculated
                'h'       :.6790,
                'Omegak'  : 0.0,
                'Omegakh2': 0.0,
                'Omegar'  : 0.0,
                'Omegarh2': 0.0,
                'ns'      : 0.9681,
                'tau'     : 0.067,
                'Yp'      :0.251,
                'As'      : 2.143*10**-9,
                'LogAs'   :-19.9619, #calculated
                'w'       :-1.0,#not from planck
                'de_model':'constant_w',#dark energy model
                #'zstar'   :1089.90,#redshift at last scattering
                'mnu'     :0.
            }
#cosmology from jdem 2008 working group paper arxiv:0901.0721v1
cosmology_jdem = {  'ns'      : 0.963,
                    'Omegamh2': 0.1326,
                    'Omegabh2': 0.0227,
                    'Omegakh2': 0.,
                    'OmegaLh2': 0.3844,
                    'dGamma'  : 0.,#parameter we don't need
                    'dM'      :0.,#parameter we don't need
                    'LogG0'   :0.,#parameter we don't need
                    'LogAs'   :-19.9628,
                    'w'       :-1.,
                    'de_model':'constant_w',
                    'mnu'     :0.
                 }
lensing_params = {  #'z_resolution'    :0.002, #fine resolution
                    #'z_min_integral'  :0.0005, #lowest z
                    #'z_max_integral'  :2,#highest z
                    'pmodel'        :'halofit', #default method for finding p grid
                    'n_gal'           :None,#118000000*6.,#118000000 galaxies/rad^2=10 galaxies/arcmin^2
                    'delta_l'         :1., #binning window
                    'sigma2_e'        :0.27**2*2, #other noise term
                    'sigma2_mu'       :1.2, #noise term for magnification
                    'smodel'          :'constant', #type, current options are 'gaussian','constant','cosmolike'
                    'z_min_dist'      :0.,
                    'z_max_dist'      :np.inf,
                    'zbar'            :0.1, #mean of gaussian source distribution
                    'sigma'           :0.4, #stdev of gaussian source distribution
                    'epsilon'         :0.0001,#small paramater
                    'l_min'           :20,#start of minimum l bin
                    'l_max'           :3000,#start of maximum l bin
                    'n_l'             :20,#number of l bins
                 }
sw_observable_list = np.array(['len_shear_shear'])
sw_survey_params = {    'needs_lensing'     : True,
                        'cross_bins'        : True}
dn_params = {'nz_select': 'CANDELS'}#'M_cut':(12.5)}
lw_observable_list = np.array(['d_number_density'])
lw_survey_params = {    'cross_bins': False}
basis_params = {   'n_bessel_oversample'   :400000,
                   #TODO n_bessel_ovsersample default may not be enough
                   # 'k_max'                 :10.,#TODO check
                   # 'k_min'                 :10**-4,
                   'x_grid_size':100000}#convergence related
polygon_params = {'res_healpix':6,'n_double':30}
nz_params = {   'data_source'   :'./data/CANDELS-GOODSS2.dat',
                'i_cut'         :24,
                'area_sterad'   :0.0409650328530259/3282.80635,
                'smooth_sigma'  :0.04,
                'n_right_extend':4,
                'z_resolution'  :0.0001,
                'mirror_boundary':True}
nz_params_wfirst_gal = {    'data_source'   :'./data/CANDELS-GOODSS2.dat',
                            'i_cut'         :26,
                            'area_sterad'   :0.0409650328530259/3282.80635,
                            'smooth_sigma'  :0.04,
                            'n_right_extend':4,
                            'z_resolution'  :0.001,
                            'mirror_boundary':True
                       }
nz_params_wfirst_lens = {   'data_source'   :'./data/H-5x140s.dat',
                            'area_sterad'   : 0.040965*np.pi**2/180**2,
                            'smooth_sigma'  :0.01,
                            'n_right_extend':16,
                            'z_resolution'  :0.001,
                            'mirror_boundary':True
                        }
nz_params_lsst = {  'data_source'   :'./data/CANDELS-GOODSS2.dat',
                    'i_cut'         :25.3,#lsst assumes they will get 10 billion galaxies with i<26 in 20000 deg^2, 4 billion lensing quality with i<25.3
                    'area_sterad'   :0.0409650328530259/3282.80635,
                    'smooth_sigma'  :0.05,
                    'n_right_extend':4,
                    'z_resolution'  :0.0001,
                    'mirror_boundary':True
                 }

hmf_params = {      'log10_min_mass'    :   6,
                    'log10_max_mass'    :   18,
                    'n_grid'            :   500,
                    'z_resolution'      : 0.01,
                    'z_min'             : 0.,
                    'z_max'             : 4.05,
                    'n_z'               : 405,
                    'b_norm_overwride'    : False,
                    'f_norm_overwride'    : False}
fpt_params = {  'C_window':0.75,
                'n_pad':1000,
                'low_extrap':-5,
                'high_extrap':5,
                'nu' :-2}
camb_params = { 'npoints':1000,
                'minkh':1.1e-4,
                'maxkh':10., #1e5
                'kmax':10.0,#20 #may need to be higher for some purposes,like 100, but makes things slower
                'leave_h':False,
                'force_sigma8':False,
                'return_sigma8':False
              }
#amara refregier 2006 parameter forecast stuff
prior_fisher_params = { 'row_strip'     :np.array([3,5,6,7]),
                        'fisher_source' :'data/F_Planck_tau0.01.dat',
                        'n_full'        :45,
                        'n_de'          :36,
                        'z_step'        :0.025
                      }
halofit_params = {  'r_max':6,#5
                    'r_min':0.05,#0.05
                    'r_step':0.01,
                    'max_extend':10,
                    'extrap_wint':True,#
                    'k_fix'     :500, #require integral in wint to go up this far if extrapolating
                    'min_kh_nonlinear' :0.005, #transition between linear and nonlinear power camb power spectrum
                    'smooth_width': 0.002 #scale over which to smooth linear to nonlinear transition to avoid spikes in derivatives
                 }

wmatcher_params = { 'w_step':0.01,
                    'w_min':-3.50,
                    'w_max':0.1,
                    'a_step':0.001,
                    'a_min':0.001,
                    'a_max':1.00
                  }

matter_power_params = { 'needs_halofit'  :True,
                        'needs_fpt'     :True,
                        'needs_wmatcher':True,
                        'needs_camb_w_grid':True,
                        'w_edge' : 0.08, #0.08
                        'w_step' : 0.002, #0.0005
                        'min_n_w' : 3,
                        'a_min' : 0.05,
                        'a_max' : 1.0,
                        'a_step' : 0.05,
                      }
lw_param_list = np.array([{'dn_params':dn_params,'n_params':nz_params_wfirst_gal,'mf_params':hmf_params}])

power_params = PowerParamManager(matter_power_params,wmatcher_params,halofit_params,camb_params,fpt_params)
