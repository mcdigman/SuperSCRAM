"""Basis and covariances for long wavelength fluctuations"""
from __future__ import division,print_function,absolute_import
from builtins import range
from time import time
from warnings import warn
from scipy.special import jv
from scipy.integrate import quad,odeint
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.linalg as spl

import numpy as np

from sph_functions import j_n, jn_zeros_cut
from lw_basis import LWBasis
from algebra_utils import cholesky_inplace

import fisher_matrix as fm

# the smallest value
eps = np.finfo(float).eps


#TODO test analytic result for power law power spectrum
class SphBasisK(LWBasis):
    """ get the long wavelength basis with spherical bessel functions described in the paper
        basis modes are selected by the k value of the bessel function zero, which approximates order of importance"""
    def __init__(self,r_max,C,k_cut,params,l_ceil=100):#,geometry,CosmoPie):
        """ inputs:
                r_max: the maximum radius of the sector
                C: CosmoPie object
                k_cut: cutoff k value for mode selection
                l_ceil: maximum l value allowed independent of k_cut, mainly because limited table of bessel function zeros available
                important! no little h in any of the calculations
        """
        print("sph_klim: begin init basis id: "+str(id(self))+" with r_max="+str(r_max)+" k_cut="+str(k_cut))
        LWBasis.__init__(self,C)
        #TODO check correct power spectrum
        self.r_max = r_max
        P_lin_in = self.C.P_lin.get_matter_power(np.array([0.]),pmodel='linear')[:,0]
        camb_params2 = self.C.P_lin.camb_params.copy()
        camb_params2['npoints'] = params['n_bessel_oversample']
        #P_lin_in,k_in = camb_pow(self.C.cosmology,camb_params=camb_params2)
        k_in = self.C.k
        #k = np.logspace(np.log10(np.min(k_in)),np.log10(np.max(k_in))-0.00001,6000000)
        #k = k_in
        self.params = params
        self.k_cut = k_cut
        self.kmin = np.min(k_in)
        self.kmax = np.max(k_in)

        self.ddelta_bar_cache = {}

        #do not change from linspace for fast
        k = np.linspace(self.kmin,self.kmax,params['n_bessel_oversample'])
        k2 = k**2
        dk = k[1]-k[0]
        #InterpolatedUnivariateSpline is better than interp1d here
        #because it better approximates the spline used by camb. 
        #Increasing camb's accuracy would also change this functions convergence
        #The lw covariance has 2 convergence concerns: 
        #increasing n_bessel_oversample converges to P_lin better, 
        #increasing npoints gives better P_lin
        P_lin = InterpolatedUnivariateSpline(k_in,P_lin_in,k=3,ext=2)(k)

        #P_lin = interp1d(k_in,k_in)(k)

        # define the super mode wave vector k alpha
        # and also make the map from l_alpha to k_alpha
        t1 = time()
        self.k_num = np.zeros(l_ceil+1,dtype=np.int)
        self.k_zeros = np.zeros(l_ceil+1,dtype=object)
        self.n_l = 0
        for ll in range(0,self.k_num.size):
            k_alpha = jn_zeros_cut(ll,self.k_cut*self.r_max)/self.r_max
            #once there are no zeros above the cut, skip higher l
            if k_alpha.size==0:
                print("sph_klim: cutting off all l>=",ll)
                break
            else:
                self.k_num[ll] = k_alpha.size
                self.k_zeros[ll] = k_alpha
                self.n_l += 1
        l_alpha = np.arange(0,self.n_l)

        self.lm_map = np.zeros((l_alpha.size,3),dtype=object)
#        self.lm = np.zeros((l_alpha.size,2),dtype=object)
        C_size = 0
        for i in range(l_alpha.size):
            m = np.arange(-l_alpha[i], l_alpha[i]+1)
            self.lm_map[i,0] = l_alpha[i]
            self.lm_map[i,1] = self.k_zeros[i]
            self.lm_map[i,2] = m
            C_size = self.lm_map[i,1].size*self.lm_map[i,2].size + C_size
#            self.lm[i,0] = l_alpha[i]
#            self.lm[i,1] = m
        self.C_size = C_size
        print("sph_klim: basis size: ",self.C_size)
        self.C_id = np.zeros((C_size,3))

        self.C_compact = np.zeros(self.n_l,dtype=object)
        print("sph_klim: begin constructing covariance matrix. basis id: ",id(self))
        itr = 0
        for a in range(self.n_l):
            ll = self.lm_map[a,0]
            kk = self.lm_map[a,1]
            mm = self.lm_map[a,2]

            print("sph_klim: calculating covar for l=",ll)
            for c in range(mm.size):
                for b in range(kk.size):
                    self.C_id[itr,0] = ll
                    self.C_id[itr,1] = kk[b]
                    self.C_id[itr,2] = mm[c]
                    itr = itr+1
            #calculate I integrals and make table
            self.C_compact[ll] = np.zeros((kk.size,kk.size))
            integrand1 = k*P_lin*jv(ll+0.5,k*self.r_max)**2
            inv_k = 1./(k2-np.expand_dims(kk**2,1))
            integrand_b = (integrand1*inv_k)
            integrand1 = None
            for b in range(0,kk.size):
                #trapezoidal rule using inner products to avoid temporary arrays
                integrated_bd = np.inner(inv_k,integrand_b[b])*dk
                integrated_bd-=(inv_k[:,0]*integrand_b[b][0]+inv_k[:,-1]*integrand_b[b][-1])/2.*dk

                for d in range(0,b+1):
                    coeff = 8.*np.sqrt(kk[b]*kk[d])*kk[b]*kk[d]/(np.pi*self.r_max**2*jv(ll+1.5,kk[b]*self.r_max)*jv(ll+1.5,kk[d]*self.r_max))
                    #TODO convergence test
                    #note: this integrand is highly oscillatory, and some integration methods may produce inaccurate results,
                    #especially for off diagonal elements. Tightly sampled trapezoidal rule is at least stable, be careful switching to anything else
                    self.C_compact[ll][b,d] = coeff*integrated_bd[d]
                    self.C_compact[ll][d,b] = self.C_compact[ll][b,d]
                integrated_bd = None
            integrand_b = None

        print("sph_klim: finished calculating covars")
#        x_grid = np.linspace(0.,np.max(self.C_id[:,1])*self.r_max,self.params['x_grid_size'])
#        self.rints = np.zeros(self.n_l,dtype=object)

#        for ll in range(0,self.n_l):
#            result_ll = odeint(lambda y,x: x**2*j_n(ll,x),0.,x_grid,atol=1e-20,rtol=1e-13)[:,0]
#            self.rints[ll] = InterpolatedUnivariateSpline(x_grid,result_ll,ext=2,k=3)
#
        t2 = time()
        print("sph_klim: basis time: ",t2-t1)
        print("sph_klim: finished init basis id: ",id(self))

    def get_size(self):
        """Get number of basis elements"""
        return self.C_size

    def get_n_l(self):
        """get the number of l values"""
        return self.n_l

    def get_covar_array(self):
        """get the covariance matrix for the basis as an array"""
        result = np.zeros((self.C_id.shape[0],self.C_id.shape[0]),order='F')
        itr_ll = 0
        for ll in range(0,self.n_l):
            n_k = self.C_compact[ll].shape[0]
            for _m_itr in range(0,2*ll+1):
                result[itr_ll:itr_ll+n_k,itr_ll:itr_ll+n_k] = self.C_compact[ll]
                itr_ll+=n_k
        return result

    def get_fisher_array(self):
        """get the fisher matrix for the basis as an array"""
        result = np.zeros((self.C_id.shape[0],self.C_id.shape[0]),order='F')
        itr_ll = 0
        for ll in range(0,self.n_l):
            n_k = self.C_compact[ll].shape[0]
            res = spl.solve(self.C_compact[ll],np.identity(n_k),sym_pos=True,lower=True,check_finite=False,overwrite_b=True)
            res = (res+res.T)/2.
            for _m_itr in range(0,2*ll+1):
                result[itr_ll:itr_ll+n_k,itr_ll:itr_ll+n_k] = res
                itr_ll+=n_k
        return result

    def get_cov_cholesky_array(self):
        """get cholesky decomposition of covariance matrix as an array"""
        result = np.zeros((self.C_id.shape[0],self.C_id.shape[0]),order='F')
        itr_ll = 0
        for ll in range(0,self.n_l):
            n_k = self.C_compact[ll].shape[0]
            res = cholesky_inplace(self.C_compact[ll],inplace=False,lower=True)
            for _m_itr in range(0,2*ll+1):
                result[itr_ll:itr_ll+n_k,itr_ll:itr_ll+n_k] = res
                itr_ll+=n_k
        return result

    def get_fisher(self,initial_state=fm.REP_COVAR,silent=True):
        """Get FisherMatrix object for the covariance matrix computed by the basis."""
        if initial_state==fm.REP_FISHER:
            return fm.FisherMatrix(self.get_fisher_array(),input_type=fm.REP_FISHER,initial_state=fm.REP_FISHER,silent=silent)
        elif initial_state==fm.REP_COVAR:
            return fm.FisherMatrix(self.get_covar_array(),input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR,silent=silent)
        elif initial_state==fm.REP_CHOL:
            return fm.FisherMatrix(self.get_cov_cholesky_array(),input_type=fm.REP_CHOL,initial_state=fm.REP_CHOL,silent=silent)
        else:
            return fm.FisherMatrix(self.get_covar_array(),input_type=fm.REP_COVAR,initial_state=initial_state,silent=silent)

    def get_variance(self,geo,k_cut_in=None,itr_in=0):
        r"""get the variance  (v.T).C_lw.v where v=\frac{\partial\bar{\delta}}{\delta_\alpha} in the given geometry"""
        #v = np.array([self.D_delta_bar_D_delta_alpha(geo,tomography=True)[itr_in]]).T
        v = self.D_delta_bar_D_delta_alpha(geo,tomography=True).T
        variance = np.zeros((v.shape[1],v.shape[1]))

        if k_cut_in is None:
            k_cut_use = self.k_cut
        else:
            k_cut_use = k_cut_in

        itr_ll = 0
        for ll in range(0,self.n_l):
            n_k = self.C_compact[ll].shape[0]
            res = self.C_compact[ll]
            n_break = n_k
            for k_itr in range(0,n_k):
                if self.C_id[itr_ll+k_itr,1]>k_cut_use:
                    n_break = k_itr
                    break
            #if n_break==0:
            #    continue
            for _m_itr in range(0,2*ll+1):
                variance+=np.dot(v[itr_ll:itr_ll+n_break].T,np.dot(res[0:n_break,0:n_break],v[itr_ll:itr_ll+n_break]))
                itr_ll+=n_k
            res = None
        return variance


    def k_LW(self):
        """return the long-wavelength wave vector k"""
        return self.C_id[:,1]

    #get the partial derivatives of an sw observable wrt the basis given an integrand
    #with elements at each r_fine i.e.  \frac{\partial O_i}{\partial \bar(\delta)(r_{fine})}
    #ignoring z>1.5 gives ~0.03% eig accuracy without mitigation,~0.3% accuracy with
    #ignoring z>0.6 gives ~3.6% eig accuracy without mit, ~1.8% accuracy with
    #ignoring z>0.4 gives ~20% eig accuracy without mit, ~3% accuracy with
    #interpretation: information adds info at higher redshift but not much variance there.
    def D_O_I_D_delta_alpha(self,geo,integrand,use_r=True,range_spec=None):
        r"""Get \frac{\partial O^I}{\partial \delta_\alpha} for an observable.
            inputs:
                geo: a Geo object for the geometry
                integrand: \frac{\partial O^I}{\partial \bar{\delta}} as a function of geo.r_fine, which must be integrated over
                use_r: if False, use geo.z_fine instead of geo.r_fine for integration
                range_spec: array slice specification for geo.r_fine to limit range of integration for tomographic bins
        """
        print("sph_klim: calculating D_O_I_D_delta_alpha")
        d_delta_bar = self.D_delta_bar_D_delta_alpha(geo,tomography=False)
#        d_delta_bar = (d_delta_bar.T*(geo.z_fine<0.4)).T
#        result = np.zeros((d_delta_bar.shape[1],integrand.shape[1]))
        #the variable to integrate over
        if use_r:
            x = geo.r_fine
        else:
            x = geo.z_fine
        #allow a restriction to the range of r_fine or z_fine
        if x.size != integrand.shape[0]:
            raise ValueError('invalid input x with shape '+str(x.shape)+' does not match '+str(integrand.shape))

        if range_spec is not None:
            x_cut = x[range_spec]
            d_delta_bar_cut = d_delta_bar[range_spec,:]
            integrand_cut = integrand[range_spec]
        else:
            x_cut = x
            d_delta_bar_cut = d_delta_bar
            integrand_cut = integrand

        delta_x = np.diff(x_cut)
        dx1s = (delta_x*d_delta_bar_cut[1::].T)/2.
        dx2s = (delta_x*d_delta_bar_cut[:-1:].T)/2.
        dxs = np.hstack([np.zeros((dx1s.shape[0],1)),dx1s])+np.hstack([dx2s,np.zeros((dx2s.shape[0],1))])
        result = np.dot(dxs,integrand_cut)
        print("sph_klim: got D_O_I_D_delta_alpha")
        return result

    #Note: Cacheing is potentially dangerous if the geo changes, so it should not be allowed to.
    def D_delta_bar_D_delta_alpha(self,geo,tomography=True):
        r"""Calculate \frac{\partial\bar{\delta}}{\partial\delta_\alpha}
            inputs:
                geo: an input Geo object for the geometry
                tomography: if True use tomographic (coarse) bins, otherwise use resolution (fine) bins for r integrals
        """
        #TODO Check this
        print("sph_klim: begin D_delta_bar_D_delta_alpha with geo id: ",id(geo))

        #Caching implements significant speedup, check caches
        result_cache = self.ddelta_bar_cache.get(str(id(geo)))
        if result_cache is not None:
            if tomography and ('tomo' in result_cache):
                print("sph_klim: tomographic bins retrieved from cache")
                return result_cache['tomo']
            elif (not tomography) and ('fine' in result_cache):
                print("sph_klim: fine bins retrieved from cache")
                return result_cache['fine']
            else:
                print("sph_klim: cache miss with nonempty cache")
        else:
            print("sph_klim: cache miss with empty cache")
            self.ddelta_bar_cache[str(id(geo))] = {}


        a_00 = geo.a_lm(0,0)
        print("sph_klim: a_00="+str(a_00))

        if tomography:
            rbins = geo.rbins
            result = np.zeros((rbins.shape[0],self.C_id.shape[0]))
            r_cache = self.ddelta_bar_cache.get(str(id(geo.zs)))
            print("sph_klim: calculating with tomographic (coarse) bins")
        else:
            rbins = geo.rbins_fine
            result = np.zeros((rbins.shape[0],self.C_id.shape[0]))
            r_cache = self.ddelta_bar_cache.get(str(id(geo.z_fine)))
            print("sph_klim: calculating with resolution (fine) slices")

        if a_00<=0:
            if a_00<-1.e-14:
                raise ValueError('a_00 '+str(a_00)+' cannot be negative')
            else:
                warn('geo has 0 area, all responses must be 0')
                return result

        if r_cache is None:
            r_cache = self.gen_R_cache(rbins)#{}

        #norm = 3./(rbins[:,1]**3 - rbins[:,0]**3)#/(a_00*2.*np.sqrt(np.pi))

        for itr in range(self.C_id.shape[0]):
            ll = int(self.C_id[itr,0])
            kk = self.C_id[itr,1]
            mm = int(self.C_id[itr,2])
            r_part = r_cache[(kk,ll)]
            result[:,itr] = r_part*(geo.a_lm(ll,mm)/(a_00*2.*np.sqrt(np.pi)))

        if tomography:
            self.ddelta_bar_cache[str(id(geo))]['tomo'] = result
            self.ddelta_bar_cache[str(id(geo.zs))] = r_cache
        else:
            self.ddelta_bar_cache[str(id(geo))]['fine'] = result
            self.ddelta_bar_cache[str(id(geo.z_fine))] = r_cache
        print("sph_klim: finished d_delta_bar_d_delta_alpha for geo id: ",id(geo))
        return result

    def gen_R_cache(self,rbins):
        """generate the r_cache dict needed by  D_delta_bar_D_delta_alpha"""
        x_grid = np.linspace(0.,np.max(self.C_id[:,1])*self.r_max,self.params['x_grid_size'])
        ll_old = -1
        r_cache = {}

        norm = 3./(rbins[:,1]**3 - rbins[:,0]**3)
        for itr in range(self.C_id.shape[0]):
            ll = int(self.C_id[itr,0])
            kk = self.C_id[itr,1]
            if ll!=ll_old:
                result_ll = odeint(lambda y,x,l_in=ll: x**2*j_n(l_in,x),0.,x_grid,atol=1e-20,rtol=1e-13)[:,0]
                rint_ll = InterpolatedUnivariateSpline(x_grid,result_ll,ext=2,k=3)
                ll_old=ll
            if (kk,ll) not in r_cache:
                r_part = (rint_ll(rbins[:,1]*kk)-rint_ll(rbins[:,0]*kk))/kk**3*norm
                r_cache[(kk,ll)] = r_part
        return r_cache

def R_int(r_range,k,ll):
    r""" returns \int R_n(rk_alpha) r2 dr
        inputs:
            r_range: [min r, max r] r range to integrate over
            k: k_alpha, zero of bessel function
            ll: index of bessel function to use"""
    # I am using the spherical Bessel function for R_n, but that might change
    def _integrand(r):
        return r**2*j_n(ll,r*k)
    result = quad(_integrand,r_range[0],r_range[1],epsabs=10e-20,epsrel=1e-10)[0]
    return result

#I_alpha checked
def I_alpha(k_alpha,k,r_max,l_alpha):
    r"""return the integral \int_0^r_{max} dr r^2 j_{\l_alpha}(k_\alpha r)j_{l_\alpha}(k r)
    needed to calculate long wavelength covariance matrix."""
    a = k_alpha*r_max
    b = k*r_max
    ll = l_alpha+.5
    return np.pi/2./np.sqrt(k_alpha*k)/(k_alpha**2 - k**2)*r_max*(-k_alpha*jv(ll-1,a)*jv(ll,b))

def norm_factor(ka,la,r_max):
    """Get normalization factor, which is I_\alpha(k_\alpha,r_{max}) simplified"""
    return -np.pi*r_max**2/(4.*ka)*jv(la+1.5,ka*r_max)*jv(la-0.5,ka*r_max)
