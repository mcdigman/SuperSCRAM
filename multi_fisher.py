import fisher_matrix as fm
import numpy as np
import copy as copy
import planck_fisher
import defaults

fisher_spec_mit={'lw_base':True,'lw_mit':True,'sw_g':True,'sw_ng':True,'par_prior':True}
fisher_spec_no_mit={'lw_base':True,'lw_mit':False,'sw_g':True,'sw_ng':True,'par_prior':True}
fisher_spec_g={'lw_base':False,'lw_mit':False,'sw_g':True,'sw_ng':False,'par_prior':True}
fisher_returns_par = {'lw':False,'sw':False,'par':True}
#master class for managing fisher matrix manipulations
class multi_fisher:
    #input: fisher_base, which should be the fisher_matrix object F_0 in the basis, specified by basis
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
        self.dO_I_ddelta_bar_list = self.sw_survey.get_dO_I_ddelta_bar_list()
        self.n_o = self.dO_I_ddelta_bar_list.size
        self.lw_to_sw_array =np.zeros((self.basis.get_size(),self.sw_survey.get_total_dimension()))
        self.d_sizes=self.sw_survey.get_dimension_list()
        itr=0
        for i in xrange(0,self.n_o):
            self.lw_to_sw_array[:,itr:itr+self.d_sizes[i]] = self.basis.D_O_I_D_delta_alpha(self.sw_survey.geo,self.dO_I_ddelta_bar_list[i])
            itr+=self.d_sizes[i]
        self.n_sw = self.sw_survey.get_total_dimension()

        #TODO fix transpose issue
        self.sw_to_parameter_array=sw_survey.get_dO_I_dparameter_array()
        
        #sw covariances to add
        self.sw_g_covar = fm.fisher_matrix(self.sw_survey.get_gaussian_cov(),input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR,silent=True)
        self.sw_ng_covar = fm.fisher_matrix(self.sw_survey.get_nongaussian_cov(),input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR,silent=True)

#        self.sw_SSC_no_mit_covar = self.lw_F_no_mit.project_covar(self.lw_to_sw_array)
#        self.sw_SSC_mit_covar = self.lw_F_mit.project_covar(self.lw_to_sw_array)

        #get total sw covariances
        #TODO maybe make more general, ie just add list of covariance matrices
#        self.sw_tot_no_mit = copy.deepcopy(self.sw_SSC_no_mit_covar)
#        self.sw_tot_no_mit.add_covar(self.sw_g_covar)
#        self.sw_tot_no_mit.add_covar(self.sw_ng_covar)

#        self.sw_tot_mit = copy.deepcopy(self.sw_SSC_mit_covar)
#        self.sw_tot_mit.add_covar(self.sw_g_covar)
#        self.sw_tot_mit.add_covar(self.sw_ng_covar)
        
        #get parameter fisher matrices
#        self.par_tot_no_mit_no_prior = self.sw_tot_no_mit.project_fisher(self.sw_to_parameter_array) 
#        self.par_tot_mit_no_prior = self.sw_tot_mit.project_fisher(self.sw_to_parameter_array) 
        
        #get prior fisher matrix
        #TODO make possible to choose
        self.fisher_priors = fm.fisher_matrix(planck_fisher.get_w0wa_projected(params=self.prior_params), input_type=fm.REP_FISHER,initial_state=fm.REP_FISHER,silent=True)
        #get full parameter fisher matrices with priors  
#        self.par_tot_no_mit = copy.deepcopy(self.par_tot_no_mit_no_prior)
#        self.par_tot_no_mit.add_fisher(self.fisher_priors)
#        self.par_tot_mit = copy.deepcopy(self.par_tot_mit_no_prior)
#        self.par_tot_mit.add_fisher(self.fisher_priors)

    #get a fisher matrix as specificed by fisher_spec and fisher_returns
    #for example the following combination would return the parameter fisher matrix including lw mitigation, sw gaussian and nonguassian covariance
    #fisher_spec={'lw_base':True,'lw_mit':True,'sw_g':True,'sw_ng':True,'par_prior':True}
    #fisher_returns = {'lw':False,'sw':False,'par':True}
    def get_fisher(self,fisher_spec,fisher_returns):
        lw_fisher=None;sw_fisher=None;par_fisher=None
        if fisher_returns['lw'] or fisher_returns['sw'] or fisher_returns['par']:
            lw_fisher = self.get_lw_fisher(fisher_spec)
            if (fisher_returns['sw'] or fisher_returns['par']):
                if lw_fisher is None:
                    fisher_from_lw=None
                else:
                    fisher_from_lw = lw_fisher.project_covar(self.lw_to_sw_array)
                sw_fisher = self.get_sw_fisher(fisher_spec,fisher_from_lw)
                if fisher_returns['par']:
                    if sw_fisher is None:
                        fisher_from_sw=None
                    else:
                        fisher_from_sw = sw_fisher.project_fisher(self.sw_to_parameter_array)
                    par_fisher = self.get_par_fisher(fisher_spec,fisher_from_sw)

        #avoid returning unwanted arrays to allow garbage collection
        results = np.array([None,None,None])
        if fisher_returns['lw']:
            results[0] = lw_fisher
        if fisher_returns['sw']:
            results[1] = sw_fisher
        if fisher_returns['par']:
            results[2] = par_fisher
        return results

    #get a parameter fisher matrix with a given projected sw matrix with or without priors 
    def get_par_fisher(self,fisher_spec,fisher_from_sw):
        if fisher_from_sw is None:
            if fisher_spec['par_prior']:
                return copy.deepcopy(self.fisher_priors)
            else:
                return None

        result = copy.deepcopy(fisher_from_sw)

        if fisher_spec['par_prior']:
            result.add_fisher(self.fisher_priors)

        return result
        
    #get a lw fisher with or without mitigation
    def get_lw_fisher(self,fisher_spec):
        if fisher_spec['lw_base'] and not fisher_spec['lw_mit']:
            return self.lw_F_no_mit 
        elif fisher_spec['lw_base'] and fisher_spec['lw_mit']:
            return self.lw_F_mit
        elif not fisher_spec['lw_base'] and fisher_spec['lw_mit']:
            raise ValueError('multi_fisher does not support mitigation with ssc contamination')
        else:
            return None

    #get a sw fisher with given projected lw matrix and with or without gaussian and nongaussian components
    def get_sw_fisher(self,fisher_spec,fisher_from_lw):
        if fisher_from_lw is None:
            if fisher_spec['sw_g'] or fisher_spec['sw_ng']:
                sw_result = fm.fisher_matrix(np.zeros((self.n_sw,self.n_sw)),input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR,silent=True)
            else:
                return None
        else:
            sw_result = copy.deepcopy(fisher_from_lw)
        if fisher_spec['sw_g']:
            sw_result.add_covar(self.sw_g_covar)
        if fisher_spec['sw_ng']:
            sw_result.add_covar(self.sw_ng_covar)
        return sw_result
        


