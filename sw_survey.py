"""
Handle a short wavelength survey
"""
import re
from warnings import warn

import numpy as np

import lensing_observables as lo
from sw_cov_mat import SWCovMat

DEBUG = False

#TODO evaluate if param_list as used by LWSurvey more elegant
class SWSurvey(object):
    """Short wavelength survey: manage short wavelength observables and get their non SSC covariances and derivatives"""
    def __init__(self,geo,survey_id,C,params,cosmo_par_list=None,cosmo_par_eps=None,observable_list=None,len_params=None,ps=None,nz_matcher=None):
        """ inputs:
                geo: a Geo object
                survey_id: some identifier for the survey
                C: a CosmoPie object
                cosmo_par_list: list of cosmological parameters that should be varied
                cosmo_par_eps: amount to vary cosmological paramters by when getting partial derivatives
                params, len_params: parameters
                observable_list: list of observable names to get
                ps: lensing source distribution. optional
                nz_matcher: NZMatcher object. optional
        """

        print "sw_survey: began initializing survey: "+str(survey_id)
        self.geo = geo
        self.params = params
        self.needs_lensing = self.params['needs_lensing']
        self.C = C # cosmopie
        self.survey_id = survey_id
        if cosmo_par_list is None:
            self.cosmo_par_list = np.array([])
            self.cosmo_par_eps = np.array([])
        else:
            if cosmo_par_eps is None or cosmo_par_eps.size!=cosmo_par_list.size:
                raise ValueError('cosmo_par_eps must be set and same size if cosmo_par_list is set')

            self.cosmo_par_list = cosmo_par_list
            self.cosmo_par_eps = cosmo_par_eps
        self.nz_matcher = nz_matcher
        self.len_params = len_params
        if self.needs_lensing:
            self.len_pow = lo.LensingPowerBase(self.geo,survey_id,C,self.cosmo_par_list,self.cosmo_par_eps,self.len_params,None,ps,self.nz_matcher)
        else:
            self.len_pow = None
        self.n_param = self.cosmo_par_list.size

        self.observable_names = generate_observable_names(self.geo,observable_list,params['cross_bins'])
        self.observables = self.names_to_observables(self.observable_names)
        print "sw_survey: finished initializing survey: "+str(survey_id)


    def get_survey_id(self):
        """return the survey id"""
        return self.survey_id

    def get_N_O_I(self):
        """get the number of available observables"""
        return self.observables.size

    def get_total_dimension(self):
        """get the total number of dimensions in the output arrays, sum of dimensions of observables"""
        return np.sum(self.get_dimension_list())

    def get_dimension_list(self):
        """get an array listing the number of dimensions of the individual observables in order"""
        dim_list = np.zeros(self.get_N_O_I(),dtype=np.int_)
        for i in xrange(dim_list.size):
            dim_list[i] = self.observables[i].get_dimension()
        return dim_list

#    def get_O_I_array(self):
#        """Get the sw observables as a concatenated array"""
#        O_I_array = np.zeros(self.get_total_dimension())
#        itr = 0
#        ds = self.get_dimension_list()
#        for i in xrange(self.observables.size):
#            O_I_array[itr:itr+ds[i]] = self.observables[i].get_O_I()
#            itr+=ds[i]
#        return O_I_array

    def get_dO_I_ddelta_bar_array(self):
        r"""Get \frac{\partial O_I}{\partial\bar{\delta}} of the observables
            is a function of z, which must be integrated over"""
        dO_I_ddelta_bar_array = np.zeros((self.geo.z_fine.size,self.get_total_dimension()))
        itr = 0
        ds = self.get_dimension_list()
        for i in xrange(self.observables.size):
            dO_I_ddelta_bar_array[:,itr:itr+ds[i]] = self.observables[i].get_dO_I_ddelta_bar()
            itr+=ds[i]
        return dO_I_ddelta_bar_array

    def get_dO_I_dpar_array(self):
        r"""Get \frac{\partial O_I}{\partial\Theta_i} of the observables wrt cosmological parameters"""
        dO_I_dpar_array = np.zeros((self.get_total_dimension(),self.n_param))
        ds = self.get_dimension_list()
        itr = 0
        for i in xrange(0,self.get_N_O_I()):
            dO_I_dpar_array[itr:itr+ds[i],:] = self.observables[i].get_dO_I_dpars()
            itr+=ds[i]
        return dO_I_dpar_array

    def get_non_SSC_sw_covar_arrays(self):
        """get 2 arrays, the total gaussian, nongaussian sw covariance matrices"""
        cov_mats = np.zeros((2,self.get_total_dimension(),self.get_total_dimension()))
        ds = self.get_dimension_list()
        #n1 and n2 are to track indices so cov_mats can be a float array instead of an array of objects
        n1 = 0
        for i in xrange(0,self.get_N_O_I()):
            n2 = 0
            for j in xrange(0,i+1):
                #if time consumption here is a problem can exploit symmetries to avoid getting same Cll multiple times
                cov = SWCovMat(self.observables[i],self.observables[j],silent=True)
                cov_mats[0,n1:n1+ds[i],n2:n2+ds[j]] = cov.get_gaussian_covar_array()
                cov_mats[0,n2:n2+ds[j],n1:n1+ds[i]] = cov_mats[0,n1:n1+ds[i],n2:n2+ds[j]]
                cov_mats[1,n1:n1+ds[i],n2:n2+ds[j]] = cov.get_nongaussian_covar_array()
                cov_mats[1,n2:n2+ds[j],n1:n1+ds[i]] = cov_mats[1,n1:n1+ds[i],n2:n2+ds[j]]
                n2+=ds[j]
            n1+=ds[i]

        assert np.all(cov_mats[0]==cov_mats[0].T)
        if DEBUG:
            n1 = 0
            for i in xrange(0,self.get_N_O_I()):
                n2 = 0
                for j in xrange(0,self.get_N_O_I()):
                    assert np.all(cov_mats[0,n2:n2+ds[j],n1:n1+ds[i]] == cov_mats[0,n1:n1+ds[i],n2:n2+ds[j]])
                n2+=ds[j]
            n1+=ds[i]

        return cov_mats

    def names_to_observables(self,names):
        """convert a dictionary of full names and tomographic bins of requested observable to the correct observable objects,
            full names are, for example, len_shear_shear_1_2, which is the shear shear lensing power spectrum between tomographic bins 1 and 2
            inputs:
                names: observable names as generated by generate_observable_names
        """
        observables = np.zeros(len(names.keys()),dtype=object)
        itr = 0
        for key in sorted(names.keys()):
            if self.params['needs_lensing'] and re.match('^len',key):
                r1 = names[key]['r1']
                r2 = names[key]['r2']

                if re.match('^len_shear_shear',key):
                    observables[itr] = lo.ShearShearLensingObservable(self.len_pow,r1,r2)
                elif re.match('^len_galaxy_galaxy',key):
                    observables[itr] = lo.GalaxyGalaxyLensingObservable(self.len_pow,r1,r2)
                else:
                    warn('unrecognized or unprocessable observable: \'',key,'\', skipping')
                    observables[itr] = None
            else:
                warn('unrecognized or unprocessable observable: \'',key,'\', skipping')
                observables[itr] = None
            itr+=1
        return observables

def generate_observable_names(geo,observable_list,cross_bins):
    """generate a list of full observable names for each tomographic bin in the geometry
        short names start with len_ if they are lensing related, then the rest of the name
        for example len_shear_shear is a shear shear lensing power spectrum
        inputs:
            geo: a Geo object to extract tomographic bins from
            observable_list: is a list of short names
            cross_bins: if False, don't do tomography with different bins"""
    rbins = geo.rbins
    names = {}
    for name in observable_list:
        if re.match('^len',name):
            n_fill = np.int(np.ceil(np.log10(rbins.shape[0]))) #pad with zeros so names sort correctly
            for i in xrange(0,rbins.shape[0]):
                r1 = rbins[i]
                if cross_bins:
                    for j in xrange(0,rbins.shape[0]):
                        #Only take r1<=r2
                        if i>j:
                            pass
                        else:
                            r2 = rbins[j]
                            name_str = name+'_'+str(i).zfill(n_fill)+'_'+str(j).zfill(n_fill)
                            names[name_str] = {'r1':r1,'r2':r2}

                else:
                    name_str = name+'_'+str(i)+'_'+str(i)
                    names[name_str] = {'r1':r1,'r2':r1}
        else:
            warn('observable name \'',name,'\' unrecognized, ignoring')
    return names
