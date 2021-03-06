"""class for combining the different fisher and covariance matrices in the code
and getting results in cosmological parameter space"""
from __future__ import division,print_function,absolute_import
from builtins import range
from warnings import warn

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
    def __init__(self,basis,sw_survey,lw_surveys, prior_params,needs_a=False,do_mit=True,do_fast=True):
        """
        master class for managing fisher matrix manipulations between bases
        inputs:
            basis: an LWBasis object
            sw_survey: an SWSurvey object
            lw_surveys: a list of LWSurvey objects to be combined into mitigation strategy
            prior_params: params for the prior fisher matrix to use in cosmological parameter space
            do_fast: allow using woodbury matrix identity to never creates the lw fisher matrix in memory
        """
        print("MultiFisher: began initialization")
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
        self.lw_to_sw_array = None

        print("MultiFisher: getting projection matrices")
        self.lw_to_sw_array = self.get_lw_to_sw_array()
        self.sw_to_par_array = sw_survey.get_dO_I_dpar_array()



        if self.needs_a:
            print("MultiFisher: getting lw no mit variance")
            self.a_vals = np.zeros(2,dtype=object)
            self.project_lw_a = self.basis.get_ddelta_bar_ddelta_alpha(self.sw_survey.geo,tomography=True)
        else:
            self.a_vals = None
            self.project_lw_a = None




        if do_fast and do_mit and self.lw_surveys.size==1 and self.lw_surveys[0].observables.size==1:
            #
            print("MultiFisher: projecting lw covariance")
            vs_perturb,sigma2s_perturb = self.lw_surveys[0].observables[0].get_perturbing_vector()
            sw_cov_ssc,sw_cov_ssc_mit = self.basis.perturb_and_project_covar(vs_perturb,self.get_lw_to_sw_array(),sigma2s_perturb)
            if self.needs_a:
                self.a_vals[0],self.a_vals[1] = self.basis.perturb_and_project_covar(vs_perturb,self.project_lw_a.T,sigma2s_perturb)
                self.project_lw_a = None
            self.lw_to_sw_array = None
            vs_perturb = None
            sigma2s_perturb=None
            self.sw_f_ssc_no_mit = fm.FisherMatrix(sw_cov_ssc,fm.REP_COVAR,fm.REP_COVAR)
            sw_cov_ssc = None
            self.sw_f_ssc_mit = fm.FisherMatrix(sw_cov_ssc_mit,fm.REP_COVAR,fm.REP_COVAR)
            sw_cov_ssc_mit = None
        else:
            print("MultiFisher: projecting lw no mit covariance")
            sw_cov_ssc = self.basis.project_covar(self.get_lw_to_sw_array())
            self.sw_f_ssc_no_mit = fm.FisherMatrix(sw_cov_ssc,fm.REP_COVAR,fm.REP_COVAR)
            sw_cov_ssc = None
            if self.needs_a:
                self.a_vals[0] = self.basis.project_covar(vs_perturb,self.project_lw_a.T)
            if do_mit:
                print("MultiFisher: getting lw mit covariance")
                self.lw_F_mit = self.get_lw_fisher(f_spec_SSC_mit,initial_state=fm.REP_COVAR)
                print("MultiFisher: projecting lw mit covariance")
                self.sw_f_ssc_mit = self.lw_F_mit.project_covar(self.get_lw_to_sw_array(),destructive=self.needs_a)
                if self.needs_a:
                    self.a_vals[1] = self.lw_F_mit.project_covar(self.project_lw_a.T).get_covar()
            else:
                self.sw_f_ssc_mit = None
            self.project_lw_a = None
            self.lw_F_mit = None
            self.lw_to_sw_array = None

        #sw covariances to add
        print("MultiFisher: getting sw covariance matrices")
        self.sw_non_SSC_covars = self.sw_survey.get_non_SSC_sw_covar_arrays()
        self.sw_g_covar = fm.FisherMatrix(self.sw_non_SSC_covars[0],fm.REP_COVAR,fm.REP_COVAR,silent=True)
        self.sw_ng_covar = fm.FisherMatrix(self.sw_non_SSC_covars[1],fm.REP_COVAR,fm.REP_COVAR,silent=True)

        if self.sw_survey.C.p_space=='jdem':
            self.fisher_prior_obj = prior_fisher.PriorFisher(self.sw_survey.C.de_model,self.prior_params)
            self.fisher_priors = self.fisher_prior_obj.get_fisher()
        else:
            warn('Current priors do not support p_space '+str(self.sw_survey.C.p_space)+', defaulting to 0 priors')
            self.fisher_prior_obj = None
            self.fisher_priors = fm.FisherMatrix(np.zeros((self.sw_to_par_array.shape[1],self.sw_to_par_array.shape[1])),fm.REP_FISHER,fm.REP_FISHER,silent=True)

        print("MultiFisher: finished initialization")

    def get_fisher(self,f_spec,f_return):
        """get a list of 3 FisherMatrix objects, [long wavelength, short wavelength, cosmological parameters]
        inputs:
            f_spec: a dictionary with keys lw, sw, and par.
                    If value at a key is False, return None instead of a FisherMatrix (to save memory)
            f_return:   specification of which long wavelength fisher, short wavelength covariance,
                        and cosmological prior fisher matrices to include
                        for example the following combination would return the parameter fisher matrix
                        including lw mitigation, sw gaussian and nonguassian covariance
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

    def get_lw_fisher(self,f_spec,initial_state=fm.REP_COVAR):
        """helper for get_fisher, get a lw fisher with or without mitigation"""
        if f_spec['lw_base'] and not f_spec['lw_mit']:
            return self.basis.get_fisher(initial_state=initial_state,silent=True)
        elif f_spec['lw_base'] and f_spec['lw_mit']:
            if self.lw_F_mit is None:
                result = self.basis.get_fisher(initial_state=fm.REP_COVAR,silent=True)
                #result = self.basis.get_fisher(initial_state=fm.REP_FISHER)
                for i in range(0,self.lw_surveys.size):
                    self.lw_surveys[i].fisher_accumulate(result)
                result.switch_rep(initial_state)
                return result
            else:
                return self.lw_F_mit
        elif not f_spec['lw_base'] and f_spec['lw_mit']:
            raise ValueError('MultiFisher does not support mitigation without ssc contamination')
        else:
            return None

    def get_sw_fisher(self,f_spec,fisher_from_lw):
        """ helper for get_fisher, get a sw fisher with given projected lw matrix
            with or without gaussian and nongaussian components"""
        if fisher_from_lw is None:
            if f_spec['sw_g'] or f_spec['sw_ng']:
                sw_result = fm.FisherMatrix(np.zeros((self.n_sw,self.n_sw)),fm.REP_COVAR,silent=True)
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
        """ get a 2d array of FisherMatrix objects,
            1st dimension is [gaussian, no mitigation, with mitigation], 2nd dimension is [lw,sw,par]
            include_priors: if True, include cosmological priors in the cosmological parameter FisherMatrix objects"""
        result = np.zeros(3,dtype=object)
        if include_priors:
            result[0] = self.get_fisher(f_spec_g,f_return_sw_par)
            result[1] = self.get_fisher(f_spec_no_mit,f_return_sw_par)
            if self.do_mit:
                result[2] = self.get_fisher(f_spec_mit,f_return_sw_par)
            else:
                result[2] = result[1]
        else:
            result[0] = self.get_fisher(f_spec_g_noprior,f_return_sw_par)
            result[1] = self.get_fisher(f_spec_no_mit_noprior,f_return_sw_par)
            if self.do_mit:
                result[2] = self.get_fisher(f_spec_mit_noprior,f_return_sw_par)
            else:
                result[2] = result[1]
        return result

    def get_eig_set(self,fisher_set,ssc_metric=False,include_sw=False):
        """Get 2d array of eigensystems for C^{ij}metric^{-1 ij}v=lambda v
            with 1st dimension [no mitigation, with mitigation] 2nd dimension [sw,par]
            inputs:
                fisher_set: an output from get_fisher_set
                ssc_metric: if True, use the no mitigation SSC covariance as the metric instead of the gaussian covariance"""
        return get_eig_set(fisher_set,ssc_metric,include_sw)

    def get_lw_to_sw_array(self):
        """get the matrix for projecting long wavelength observables to sw basis"""
        if self.lw_to_sw_array is None:
            lw_to_sw_array = self.basis.get_dO_I_ddelta_alpha(self.sw_survey.geo,self.sw_survey.get_dO_I_ddelta_bar_array())
        else:
            lw_to_sw_array = self.lw_to_sw_array
        return lw_to_sw_array

    def get_a_lw(self,destructive=False):
        r"""get (v.T).C_lw.v where v=\frac{\partial\bar{\delta}}{\delta_\alpha}
        for [no mitigation, with mitigation] lw covariance matrices
        per tomographic bin"""
        if self.needs_a:
            return self.a_vals
        else:
            d_no_mit = destructive or self.lw_F_no_mit is None
            project_lw_a = self.basis.get_ddelta_bar_ddelta_alpha(self.sw_survey.geo,tomography=True)
            a_no_mit = self.get_lw_fisher(f_spec_SSC_no_mit,fm.REP_COVAR).project_covar(project_lw_a.T,destructive=d_no_mit).get_covar()
            if d_no_mit:
                self.lw_F_no_mit = None
            if self.do_mit:
                d_mit = destructive or self.lw_F_mit is None
                a_mit = self.get_lw_fisher(f_spec_SSC_mit,fm.REP_COVAR).project_covar(project_lw_a.T,destructive=d_mit).get_covar()
                if d_mit:
                    self.lw_F_mit = None
            else:
                a_mit = 0.
            return np.array([a_no_mit,a_mit])

def get_eig_set(fisher_set,ssc_metric=False,include_sw=False):
    """Get 2d array of eigensystems for C^{ij}metric^{-1 ij}v=lambda v
        with 1st dimension [no mitigation, with mitigation] 2nd dimension [sw,par]
        inputs:
            fisher_set: an output from get_fisher_set
            ssc_metric: if True, use the no mitigation SSC covariance as the metric instead of the gaussian covariance"""
    result = np.zeros((2,2),dtype=object)
    f_set_par = np.zeros(3,dtype=object)
    for i in range(0,3):
        f_set_par[i] = fisher_set[i][2]
    if ssc_metric:
        metrics = np.array([fisher_set[1][1],f_set_par[1]])
    else:
        metrics = np.array([fisher_set[0][1],f_set_par[0]])
    if include_sw:
        result[0,0] = fisher_set[1][1].get_cov_eig_metric(metrics[0])
        result[0,1] = fisher_set[2][1].get_cov_eig_metric(metrics[0])
    result[1,0] = f_set_par[1].get_cov_eig_metric(metrics[1])
    result[1,1] = f_set_par[2].get_cov_eig_metric(metrics[1])
    return result
