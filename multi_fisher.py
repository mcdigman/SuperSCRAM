"""class for combining the different fisher and covariance matrices in the code
and getting results in cosmological parameter space"""

import copy
import numpy as np

import prior_fisher

import fisher_matrix as fm

f_spec_mit = {'lw_base':True,'lw_mit':True,'sw_g':True,'sw_ng':True,'par_prior':True}
f_spec_no_mit = {'lw_base':True,'lw_mit':False,'sw_g':True,'sw_ng':True,'par_prior':True}
f_spec_g = {'lw_base':False,'lw_mit':False,'sw_g':True,'sw_ng':False,'par_prior':True}

f_spec_mit_noprior = {'lw_base':True,'lw_mit':True,'sw_g':True,'sw_ng':True,'par_prior':False}
f_spec_no_mit_noprior = {'lw_base':True,'lw_mit':False,'sw_g':True,'sw_ng':True,'par_prior':False}
f_spec_g_noprior = {'lw_base':False,'lw_mit':False,'sw_g':True,'sw_ng':False,'par_prior':False}

f_spec_SSC_mit = {'lw_base':True,'lw_mit':True,'sw_g':False,'sw_ng':False,'par_prior':False}
f_spec_SSC_no_mit = {'lw_base':True,'lw_mit':False,'sw_g':False,'sw_ng':False,'par_prior':False}
f_return_par = {'lw':False,'sw':False,'par':True}
f_return_sw_par = {'lw':False,'sw':True,'par':True}
f_return_sw = {'lw':False,'sw':True,'par':False}
f_return_lw = {'lw':True,'sw':False,'par':False}
class MultiFisher(object):
    """master class for managing fisher matrix manipulations between bases"""
    def __init__(self,basis,sw_survey,lw_surveys, prior_params,needs_a=False,do_mit=True):
        """
        master class for managing fisher matrix manipulations between bases
        inputs:
            basis: an LWBasis object
            sw_survey: an SWSurvey object
            lw_surveys: a list of LWSurvey objects to be combined into mitigation strategy
            prior_params: params for the prior fisher matrix to use in cosmological parameter space
        """
        print "MultiFisher: began initialization"
        self.basis = basis
        self.sw_survey = sw_survey
        self.lw_surveys = lw_surveys
        self.prior_params = prior_params
        self.needs_a = needs_a
        self.do_mit = do_mit

        #prepare to project lw basis to sw basis
        self.n_sw = self.sw_survey.get_total_dimension()

        self.lw_F_no_mit = None
        self.lw_F_mit = None

        print "MultiFisher: getting projection matrices"
        self.lw_to_sw_array = self.basis.D_O_I_D_delta_alpha(self.sw_survey.geo,self.sw_survey.get_dO_I_ddelta_bar_array())
        self.sw_to_par_array = sw_survey.get_dO_I_dpar_array()


        print "MultiFisher: getting lw no mit covariance"
        #get the variance if needed but don't always do this because it adds time
        if self.needs_a:
            print "MultiFisher: getting lw no mit variance"
            self.a_vals = np.zeros(2)
            self.lw_F_no_mit = self.get_lw_fisher(f_spec_SSC_no_mit,initial_state=fm.REP_COVAR)
            self.project_lw_a = self.basis.D_delta_bar_D_delta_alpha(self.sw_survey.geo,tomography=True)[0]
            self.a_vals[0] = self.lw_F_no_mit.project_covar(self.project_lw_a,destructive=True).get_covar()
            self.lw_F_no_mit = None
        else:
            self.a_vals = None
            self.project_lw_a = None


        self.lw_F_no_mit = self.get_lw_fisher(f_spec_SSC_no_mit,initial_state=fm.REP_COVAR)

        print "MultiFisher: projecting lw no mit covariance"
        self.sw_f_ssc_no_mit = self.lw_F_no_mit.project_covar(self.lw_to_sw_array,destructive=True)
        self.lw_F_no_mit = None

        if do_mit:
            print "MultiFisher: getting lw mit covariance"
            #self.lw_F_mit = self.get_lw_fisher(f_spec_SSC_mit,initial_state=fm.REP_FISHER)
            self.lw_F_mit = self.get_lw_fisher(f_spec_SSC_mit,initial_state=fm.REP_COVAR)

            if self.needs_a:
                print "MultiFisher: getting lw mit variance "
                self.a_vals[1] = self.lw_F_mit.project_covar(self.project_lw_a).get_covar()

            print "MultiFisher: projecting lw mit covariance"
            self.sw_f_ssc_mit = self.lw_F_mit.project_covar(self.lw_to_sw_array,destructive=True)
            self.lw_F_mit = None
        else:
            self.sw_f_ssc_mit = None
        #accumulate lw covariances onto fisher_tot

        #for i in xrange(0,self.lw_surveys.size):
        #    self.lw_surveys[i].fisher_accumulate(self.lw_F_mit)
        #self.lw_F_mit.switch_rep(fm.REP_CHOL_INV)
        #self.lw_F_no_mit.switch_rep(fm.REP_CHOL_INV)


        self.lw_F_mit = None

        #sw covariances to add
        print "MultiFisher: getting sw covariance matrices"
        self.sw_non_SSC_covars = self.sw_survey.get_non_SSC_sw_covar_arrays()
        self.sw_g_covar = fm.FisherMatrix(self.sw_non_SSC_covars[0],input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR,silent=True)
        self.sw_ng_covar = fm.FisherMatrix(self.sw_non_SSC_covars[1],input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR,silent=True)

        self.fisher_prior_obj = prior_fisher.PriorFisher(self.sw_survey.C.de_model,self.prior_params)
        self.fisher_priors = self.fisher_prior_obj.get_fisher()

        print "MultiFisher: finished initialization"

    def get_fisher(self,f_spec,f_return):
        """get a list of 3 FisherMatrix objects, [long wavelength, short wavelength, cosmological parameters]
        inputs:
            f_spec: a dictionary with keys lw, sw, and par. If value at a key is False, return None instead of a FisherMatrix (to save memory)
            f_return: specification of which long wavelength fisher, short wavelength covariance, and cosmological prior fisher matrices to include
                for example the following combination would return the parameter fisher matrix including lw mitigation, sw gaussian and nonguassian covariance
                f_spec = {'lw_base':True,'lw_mit':True,'sw_g':True,'sw_ng':True,'par_prior':True}
                f_return = {'lw':False,'sw':False,'par':True}
        """
        if f_return['lw'] or f_return['sw'] or f_return['par']:
            if f_return['lw']:
                lw_fisher = self.get_lw_fisher(f_spec)
            if f_return['sw'] or f_return['par']:
                if f_spec['lw_mit'] and f_spec['lw_base']:
                    fisher_from_lw = self.sw_f_ssc_mit
                elif f_spec['lw_base']:
                    fisher_from_lw = self.sw_f_ssc_no_mit
                else:
                    fisher_from_lw = None
                sw_fisher = self.get_sw_fisher(f_spec,fisher_from_lw)
                if f_return['par']:
                    if sw_fisher is None:
                        fisher_from_sw = None
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

    def get_par_fisher(self,f_spec,fisher_from_sw):
        """helper for get_fisher, get a parameter fisher matrix with a given projected sw matrix with or without priors"""
        if fisher_from_sw is None:
            if f_spec['par_prior']:
                return copy.deepcopy(self.fisher_priors)
            else:
                return None

        result = copy.deepcopy(fisher_from_sw)

        if f_spec['par_prior']:
            result.add_fisher(self.fisher_priors)

        return result

    def get_lw_fisher(self,f_spec,initial_state=fm.REP_CHOL):
        """helper for get_fisher, get a lw fisher with or without mitigation"""
        if f_spec['lw_base'] and not f_spec['lw_mit']:
            return self.basis.get_fisher(initial_state=initial_state)
        elif f_spec['lw_base'] and f_spec['lw_mit']:
            if self.lw_F_mit is None:
                result = self.basis.get_fisher(initial_state=fm.REP_COVAR)
                #result = self.basis.get_fisher(initial_state=fm.REP_FISHER)
                print "got fisher 1 in state "+str(result.internal_state)
                for i in xrange(0,self.lw_surveys.size):
                    self.lw_surveys[i].fisher_accumulate(result)
                print "accumulated fisher"
                result.switch_rep(initial_state)
                print "switched rep"
                return result
            else:
                return self.lw_F_mit
        elif not f_spec['lw_base'] and f_spec['lw_mit']:
            raise ValueError('MultiFisher does not support mitigation without ssc contamination')
        else:
            return None

    def get_sw_fisher(self,f_spec,fisher_from_lw):
        """helper for get_fisher, get a sw fisher with given projected lw matrix and with or without gaussian and nongaussian components"""
        if fisher_from_lw is None:
            if f_spec['sw_g'] or f_spec['sw_ng']:
                sw_result = fm.FisherMatrix(np.zeros((self.n_sw,self.n_sw)),input_type=fm.REP_COVAR,silent=True)
            else:
                return None
        else:
            sw_result = copy.deepcopy(fisher_from_lw)
        if f_spec['sw_g']:
            sw_result.add_covar(self.sw_g_covar)
        if f_spec['sw_ng']:
            sw_result.add_covar(self.sw_ng_covar)
        return sw_result

    def get_fisher_set(self,include_priors=True):
        """get a 2d array of FisherMatrix objects, 1st dimension is [gaussian, no mitigation, with mitigation], 2nd dimension is [lw,sw,par]
            inputs: include_priors: if True, include cosmological priors in the cosmological parameter FisherMatrix objects"""
        result = np.zeros(3,dtype=object)
        if include_priors:
            result[0] = self.get_fisher(f_spec_g,f_return_sw_par)
            result[1] = self.get_fisher(f_spec_no_mit,f_return_sw_par)
            if self.do_mit:
                result[2] = self.get_fisher(f_spec_mit,f_return_sw_par)
        else:
            result[0] = self.get_fisher(f_spec_g_noprior,f_return_sw_par)
            result[1] = self.get_fisher(f_spec_no_mit_noprior,f_return_sw_par)
            if self.do_mit:
                result[2] = self.get_fisher(f_spec_mit_noprior,f_return_sw_par)
        return result

    #TODO handle caching better to avoid this logic
    def get_eig_set(self,fisher_set,ssc_metric=False):
        """Get 2d array of eigensystems for C^{ij}metric^{-1 ij}v=lambda v with 1st dimension [no mitigation, with mitigation] 2nd dimension [sw,par]
            inputs:
                fisher_set: an output from get_fisher_set
                ssc_metric: if True, use the no mitigation SSC covariance as the metric instead of the gaussian covariance"""
        result = np.zeros((2,2),dtype=object)
        f_set_par = np.zeros(3,dtype=object)
        for i in xrange(0,3):
            f_set_par[i] = fisher_set[i][2]
        if ssc_metric:
            metrics = np.array([fisher_set[1][1],f_set_par[1]])
        else:
            metrics = np.array([fisher_set[0][1],f_set_par[0]])

        #result[0,0] = fisher_set[1][1].get_cov_eig_metric(metrics[0])
        #result[0,1] = fisher_set[2][1].get_cov_eig_metric(metrics[0])
        result[1,0] = f_set_par[1].get_cov_eig_metric(metrics[1])
        result[1,1] = f_set_par[2].get_cov_eig_metric(metrics[1])
        return result

    def get_a_lw(self):
        r"""get (v.T).C_lw.v where v=\frac{\partial\bar{\delta}}{\delta_\alpha} for [no mitigation, with mitigation] lw covariance matrices"""
        if self.needs_a:
            return self.a_vals
        else:
            project_lw_a = self.basis.D_delta_bar_D_delta_alpha(self.sw_survey.geo,tomography=True)[0]
            lw_proj_no_mit = self.get_lw_fisher(f_spec_SSC_no_mit,initial_state=fm.REP_COVAR).project_covar(project_lw_a).get_covar()
            if self.do_mit:
                lw_proj_mit = self.get_lw_fisher(f_spec_SSC_mit,initial_state=fm.REP_COVAR).project_covar(project_lw_a).get_covar()
                #lw_proj_mit = self.get_lw_fisher(f_spec_SSC_mit,initial_state=fm.REP_CHOL_INV).project_covar(project_lw_a).get_covar()
            else:
                lw_proj_mit = 0.
            return np.array([lw_proj_no_mit,lw_proj_mit])
