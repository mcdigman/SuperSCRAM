import copy 
import planck_fisher
import defaults

import fisher_matrix as fm
import numpy as np

from warnings import warn

f_spec_mit={'lw_base':True,'lw_mit':True,'sw_g':True,'sw_ng':True,'par_prior':True}
f_spec_no_mit={'lw_base':True,'lw_mit':False,'sw_g':True,'sw_ng':True,'par_prior':True}
f_spec_g={'lw_base':False,'lw_mit':False,'sw_g':True,'sw_ng':False,'par_prior':True}
f_spec_g_pure={'lw_base':False,'lw_mit':False,'sw_g':True,'sw_ng':False,'par_prior':False}
f_spec_SSC_mit={'lw_base':True,'lw_mit':True,'sw_g':False,'sw_ng':False,'par_prior':False}
f_spec_SSC_no_mit={'lw_base':True,'lw_mit':False,'sw_g':False,'sw_ng':False,'par_prior':False}
f_return_par = {'lw':False,'sw':False,'par':True}
f_return_sw_par = {'lw':False,'sw':True,'par':True}
f_return_sw = {'lw':False,'sw':True,'par':False}
f_return_lw = {'lw':True,'sw':False,'par':False}
#master class for managing fisher matrix manipulations
class multi_fisher:
    #input: basis, an sph_basis_k object or compatible standard
    #input: sw_survey, ans sw_survey objects
    def __init__(self,basis,sw_survey,lw_surveys, prior_params=defaults.planck_fisher_params):
        self.lw_F_no_mit=basis.get_fisher()
        self.basis=basis
        self.sw_survey=sw_survey
        self.lw_surveys = lw_surveys
        self.prior_params = prior_params

        #accumulate lw covariances onto fisher_tot
        self.lw_F_mit = copy.deepcopy(self.lw_F_no_mit)
        for i in xrange(0,self.lw_surveys.size):
            self.lw_surveys[i].fisher_accumulate(self.lw_F_mit)

        #prepare to project lw basis to sw basis
        #self.dO_I_ddelta_bar_list = self.sw_survey.get_dO_I_ddelta_bar_list()
        self.n_o = self.sw_survey.get_N_O_I()
        #self.lw_to_sw_array =np.zeros((self.basis.get_size(),self.sw_survey.get_total_dimension()))
        #self.d_sizes=self.sw_survey.get_dimension_list()
        #itr=0
        #for i in xrange(0,self.n_o):
        #    self.lw_to_sw_array[:,itr:itr+self.d_sizes[i]] = self.basis.D_O_I_D_delta_alpha(self.sw_survey.geo,self.dO_I_ddelta_bar_list[i])
        #    itr+=self.d_sizes[i]

        self.lw_to_sw_array = self.basis.D_O_I_D_delta_alpha(self.sw_survey.geo,self.sw_survey.get_dO_I_ddelta_bar_array())

        self.n_sw = self.sw_survey.get_total_dimension()

        self.sw_to_par_array=sw_survey.get_dO_I_dpar_array()
        
        #sw covariances to add
        self.sw_non_SSC_covars = self.sw_survey.get_non_SSC_sw_covar_arrays()
        self.sw_g_covar = fm.fisher_matrix(self.sw_non_SSC_covars[0],input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR,silent=True)
        self.sw_ng_covar = fm.fisher_matrix(self.sw_non_SSC_covars[1],input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR,silent=True)

#        self.sw_SSC_no_mit_covar = self.lw_F_no_mit.project_covar(self.lw_to_sw_array)
#        self.sw_SSC_mit_covar = self.lw_F_mit.project_covar(self.lw_to_sw_array)

        #get total sw covariances
#        self.sw_tot_no_mit = copy.deepcopy(self.sw_SSC_no_mit_covar)
#        self.sw_tot_no_mit.add_covar(self.sw_g_covar)
#        self.sw_tot_no_mit.add_covar(self.sw_ng_covar)

#        self.sw_tot_mit = copy.deepcopy(self.sw_SSC_mit_covar)
#        self.sw_tot_mit.add_covar(self.sw_g_covar)
#        self.sw_tot_mit.add_covar(self.sw_ng_covar)
        
        #get parameter fisher matrices
#        self.par_tot_no_mit_no_prior = self.sw_tot_no_mit.project_fisher(self.sw_to_par_array) 
#        self.par_tot_mit_no_prior = self.sw_tot_mit.project_fisher(self.sw_to_par_array) 
        
        #get prior fisher matrix
        #TODO make possible to choose
        if self.sw_survey.C.de_model=='w0wa':
            self.fisher_priors = fm.fisher_matrix(planck_fisher.get_w0wa_projected(params=self.prior_params), input_type=fm.REP_FISHER,initial_state=fm.REP_FISHER,silent=True)
        elif self.sw_survey.C.de_model=='constant_w':
            self.fisher_priors = fm.fisher_matrix(planck_fisher.get_w0_projected(params=self.prior_params), input_type=fm.REP_FISHER,initial_state=fm.REP_FISHER,silent=True)
        elif self.sw_survey.C.de_model=='jdem':
            self.fisher_priors = fm.fisher_matrix(planck_fisher.get_jdem_projected(params=self.prior_params), input_type=fm.REP_FISHER,initial_state=fm.REP_FISHER,silent=True)
        else:
            warn('unknown prior parametrization: '+str(self.sw_survey.C.de_model)+' will not use priors for de')
            self.fisher_priors = fm.fisher_matrix(planck_fisher.get_no_de_projected(params=self.prior_params), input_type=fm.REP_FISHER,initial_state=fm.REP_FISHER,silent=True)



        #get full parameter fisher matrices with priors  
#        self.par_tot_no_mit = copy.deepcopy(self.par_tot_no_mit_no_prior)
#        self.par_tot_no_mit.add_fisher(self.fisher_priors)
#        self.par_tot_mit = copy.deepcopy(self.par_tot_mit_no_prior)
#        self.par_tot_mit.add_fisher(self.fisher_priors)

    #get a fisher matrix as specificed by f_spec and f_return
    #for example the following combination would return the parameter fisher matrix including lw mitigation, sw gaussian and nonguassian covariance
    #f_spec={'lw_base':True,'lw_mit':True,'sw_g':True,'sw_ng':True,'par_prior':True}
    #f_return = {'lw':False,'sw':False,'par':True}
    def get_fisher(self,f_spec,f_return):
        if f_return['lw'] or f_return['sw'] or f_return['par']:
            lw_fisher = self.get_lw_fisher(f_spec)
            if f_return['sw'] or f_return['par']:
                if lw_fisher is None:
                    fisher_from_lw=None
                else:
                    fisher_from_lw = lw_fisher.project_covar(self.lw_to_sw_array)
                sw_fisher = self.get_sw_fisher(f_spec,fisher_from_lw)
                if f_return['par']:
                    if sw_fisher is None:
                        fisher_from_sw=None
                    else:
                        fisher_from_sw = sw_fisher.project_fisher(self.sw_to_par_array)
                    par_fisher = self.get_par_fisher(f_spec,fisher_from_sw)

        #avoid returning unwanted arrays to allow garbage collection
        results = np.array([None,None,None])
        if f_return['lw']:
            results[0] = lw_fisher
        if f_return['sw']:
            results[1] = sw_fisher
        if f_return['par']:
            results[2] = par_fisher
        return results

    #get a parameter fisher matrix with a given projected sw matrix with or without priors 
    def get_par_fisher(self,f_spec,fisher_from_sw):
        if fisher_from_sw is None:
            if f_spec['par_prior']:
                return copy.deepcopy(self.fisher_priors)
            else:
                return None

        result = copy.deepcopy(fisher_from_sw)

        if f_spec['par_prior']:
            result.add_fisher(self.fisher_priors)

        return result
        
    #get a lw fisher with or without mitigation
    def get_lw_fisher(self,f_spec):
        if f_spec['lw_base'] and not f_spec['lw_mit']:
            return self.lw_F_no_mit 
        elif f_spec['lw_base'] and f_spec['lw_mit']:
            return self.lw_F_mit
        elif not f_spec['lw_base'] and f_spec['lw_mit']:
            raise ValueError('multi_fisher does not support mitigation without ssc contamination')
        else:
            return None

    #get a sw fisher with given projected lw matrix and with or without gaussian and nongaussian components
    def get_sw_fisher(self,f_spec,fisher_from_lw):
        if fisher_from_lw is None:
            if f_spec['sw_g'] or f_spec['sw_ng']:
                sw_result = fm.fisher_matrix(np.zeros((self.n_sw,self.n_sw)),input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR,silent=True)
            else:
                return None
        else:
            sw_result = copy.deepcopy(fisher_from_lw)
        if f_spec['sw_g']:
            sw_result.add_covar(self.sw_g_covar)
        if f_spec['sw_ng']:
            sw_result.add_covar(self.sw_ng_covar)
        return sw_result
        
    def get_fisher_set(self):
        result = np.zeros(3,dtype=object)
        result[0] = self.get_fisher(f_spec_g,f_return_sw_par)
        result[1] = self.get_fisher(f_spec_no_mit,f_return_sw_par)
        result[2] = self.get_fisher(f_spec_mit,f_return_sw_par)
        return result

    def get_a_lw(self):
        project_lw_a = self.basis.D_delta_bar_D_delta_alpha(self.sw_survey.geo,tomography=True)[0]
        return np.array([self.lw_F_no_mit.project_covar(project_lw_a).get_covar(),self.lw_F_mit.project_covar(project_lw_a).get_covar()])


