from time import time
from scipy.special import jv
from scipy.integrate import quad,odeint
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.linalg as spl

import numpy as np

from sph_functions import j_n, jn_zeros_cut
from lw_basis import LWBasis
from algebra_utils import trapz2,cholesky_inplace

import fisher_matrix as fm
import defaults

# the smallest value
eps=np.finfo(float).eps


#TODO test analytic result for power law power spectrum
class SphBasisK(LWBasis):
    def __init__(self,r_max,C,k_cut=2.0,l_ceil=100,params=defaults.basis_params):#,geometry,CosmoPie):

        """ get the long wavelength basis with spherical bessel functions described in the paper
            basis modes are selected by the k value of the bessel function zero, which approximates order of importance
            inputs:
                r_max: the maximum radius of the sector
                C: CosmoPie object
                k_cut: cutoff k value for mode selection
                l_ceil: maximum l value allowed independent of k_cut, mainly because limited table of bessel function zeros available
                important! no little h in any of the calculations
        """
        print "sph_klim: begin init basis id: ",id(self)
        LWBasis.__init__(self,C)
        #TODO check correct power spectrum
        self.r_max = r_max
        P_lin_in=self.C.P_lin.get_matter_power(np.array([0.]),pmodel='linear')[:,0]
        camb_params2 = self.C.camb_params.copy()
        camb_params2['npoints'] = params['n_bessel_oversample']
        #P_lin_in,k_in = camb_pow(self.C.cosmology,camb_params=camb_params2)
        k_in = self.C.k
        #k = np.logspace(np.log10(np.min(k_in)),np.log10(np.max(k_in))-0.00001,6000000)
        #k = k_in
        self.params = params
        self.k_cut = k_cut
        self.kmin = np.min(k_in)
        self.kmax = np.max(k_in)

        self.allow_caching = params['allow_caching']
        if self.allow_caching:
            self.ddelta_bar_cache = {}
        #do not change from linspace for fast
        k = np.linspace(self.kmin,self.kmax,params['n_bessel_oversample'])
        self.dk = k[1]-k[0]
        #InterpolatedUnivariateSpline is better than interp1d here, because it better approximates the spline used by camb. Increasing camb's accuracy would also change this functions convergence
        #The lw covariance has 2 convergence concerns: increasing n_bessel_oversample converges to P_lin better, increasing npoints gives better P_lin
        P_lin = InterpolatedUnivariateSpline(k_in,P_lin_in,k=3,ext=2)(k)

        #P_lin = interp1d(k_in,k_in)(k)

        # define the super mode wave vector k alpha
        # and also make the map from l_alpha to k_alpha
        t1 = time()
        self.k_num = np.zeros(l_ceil+1,dtype=np.int)
        self.k_zeros = np.zeros(l_ceil+1,dtype=object)
        self.n_l = 0
        for ll in xrange(0,self.k_num.size):
            k_alpha = jn_zeros_cut(ll,self.k_cut*self.r_max)/self.r_max
            #once there are no zeros above the cut, skip higher l
            if k_alpha.size == 0:
                print "sph_klim: cutting off all l>=",ll
                break
            else:
                self.k_num[ll] = k_alpha.size
                self.k_zeros[ll] = k_alpha
                self.n_l += 1
        l_alpha = np.arange(0,self.n_l)

        self.l=l_alpha #TODO unnecessary
        self.lm_map=np.zeros((l_alpha.size,3),dtype=object)
        self.lm=np.zeros((l_alpha.size,2),dtype=object)
        C_size=0
        for i in xrange(l_alpha.size):
            m=np.arange(-l_alpha[i], l_alpha[i]+1)
            self.lm_map[i,0]=l_alpha[i]
            self.lm_map[i,1]=self.k_zeros[i]
            self.lm_map[i,2]=m
            C_size=self.lm_map[i,1].size*self.lm_map[i,2].size + C_size
            self.lm[i,0]=l_alpha[i]
            self.lm[i,1]=m
        self.C_size = C_size
        print "sph_klim: basis size: ",self.C_size
        self.C_id=np.zeros((C_size,3))
        #self.C_alpha_beta=np.zeros((self.C_id.shape[0],self.C_id.shape[0]),order='F')

        self.C_compact = np.zeros(self.n_l,dtype=object)
        print "sph_klim: begin constructing covariance matrix. basis id: ",id(self)
        itr=0
        for a in xrange(self.n_l):
            ll=self.lm_map[a,0]
            kk=self.lm_map[a,1]
            mm=self.lm_map[a,2]

            #itr_m1 = itr
            print "sph_klim: calculating covar for l=",ll
            for c in xrange(mm.size):
            #    itr_k1 = itr
                for b in xrange(kk.size):
                    self.C_id[itr,0]=ll
                    self.C_id[itr,1]=kk[b]
                    self.C_id[itr,2]=mm[c]
                    itr=itr+1
            #calculate I integrals and make table
            self.C_compact[ll] = np.zeros((kk.size,kk.size))
            integrand1 = k*P_lin*jv(ll+0.5,k*self.r_max)**2
            #TODO can make somewhat more efficient if necessary
            for b in xrange(0,kk.size):
                for d in xrange(b,kk.size):
                    coeff = 8.*np.sqrt(kk[b]*kk[d])*kk[b]*kk[d]/(np.pi*self.r_max**2*jv(ll+1.5,kk[b]*self.r_max)*jv(ll+1.5,kk[d]*self.r_max))
                    #TODO convergence test
                    #note: this integrand is highly oscillatory, and some integration methods may produce inaccurate results,
                    #especially for off diagonal elements. Tightly sampled trapezoidal rule is at least stable, be careful switching to anything else
                    self.C_compact[ll][b,d]=coeff*trapz2(integrand1/((k**2-kk[b]**2)*(k**2-kk[d]**2)),dx=self.dk,given_dx=True) #check coefficient
                    self.C_compact[ll][d,b]=self.C_compact[ll][b,d]

        print "sph_klim: finished calculating covars"
        x_grid = np.linspace(0.,np.max(self.C_id[:,1])*self.r_max,self.params['x_grid_size'])
        self.rints = np.zeros(self.n_l,dtype=object)

        for ll in xrange(0,self.n_l):
            r_I = lambda y,x: x**2*j_n(ll,x)
            result_ll = odeint(r_I,0.,x_grid,atol=10e-20,rtol=10e-10)[:,0]
            self.rints[ll] = InterpolatedUnivariateSpline(x_grid,result_ll,ext=2,k=3)

        t2 = time()
        print "sph_klim: basis time: ",t2-t1
        print "sph_klim: finished init basis id: ",id(self)

    def get_size(self):
        """Get number of basis elements"""
        return self.C_size

    def get_covar_array(self):
        result = np.zeros((self.C_id.shape[0],self.C_id.shape[0]),order='F')
        itr_ll = 0
        for ll in xrange(0,self.n_l):
            n_k = self.C_compact[ll].shape[0]
            for m_itr in xrange(0,2*ll+1):
                result[itr_ll:itr_ll+n_k,itr_ll:itr_ll+n_k]=self.C_compact[ll]
                itr_ll+=n_k
        return result

    def get_fisher_array(self):
        result = np.zeros((self.C_id.shape[0],self.C_id.shape[0]),order='F')
        itr_ll = 0
        for ll in xrange(0,self.n_l):
            n_k = self.C_compact[ll].shape[0]
            res = spl.solve(self.C_compact[ll],np.identity(n_k),sym_pos=True,lower=True,check_finite=False,overwrite_b=True)
            res = (res+res.T)/2.
            for m_itr in xrange(0,2*ll+1):
                result[itr_ll:itr_ll+n_k,itr_ll:itr_ll+n_k] = res
                itr_ll+=n_k
        return result

    def get_cov_cholesky_array(self):
        result = np.zeros((self.C_id.shape[0],self.C_id.shape[0]),order='F')
        itr_ll = 0
        for ll in xrange(0,self.n_l):
            n_k = self.C_compact[ll].shape[0]
            res = cholesky_inplace(self.C_compact[ll],inplace=False,lower=True)
            for m_itr in xrange(0,2*ll+1):
                result[itr_ll:itr_ll+n_k,itr_ll:itr_ll+n_k]=res
                itr_ll+=n_k
        return result

    #TODO storing packed representation of self.C_alpha_beta and generating the FisherMatrix object here would be more memory efficient, safer so does not mutate self.C_alpha_beta
    def get_fisher(self,initial_state=fm.REP_COVAR):
        """Get FisherMatrix object for the covariance matrix computed by the basis."""
        if initial_state==fm.REP_FISHER:
            return fm.FisherMatrix(self.get_fisher_array(),input_type=fm.REP_FISHER,initial_state=fm.REP_FISHER,silent=True)
        elif initial_state==fm.REP_COVAR:
            return fm.FisherMatrix(self.get_covar_array(),input_type=fm.REP_COVAR,initial_state=fm.REP_COVAR,silent=True)
        elif initial_state==fm.REP_CHOL:
            return fm.FisherMatrix(self.get_cov_cholesky_array(),input_type=fm.REP_CHOL,initial_state=fm.REP_CHOL,silent=True)
        else:
            return fm.FisherMatrix(self.get_covar_array(),input_type=fm.REP_COVAR,initial_state=initial_state,silent=True)

    def k_LW(self):
        """return the long-wavelength wave vector k"""
        return self.C_id[:,1]

    #get the partial derivatives of an sw observable wrt the basis given an integrand with elements at each r_fine i.e.  \frac{\partial O_i}{\partial \bar(\delta)(r_{fine})}
    def D_O_I_D_delta_alpha(self,geo,integrand,force_recompute = False,use_r=True,range_spec=None):
        """Get \frac{\partial O^I}{\partial \delta_\alpha} for an observable.
            inputs:
                geo: a Geo object for the geometry
                integrand: the observable as a function of geo.r_fine, which must be integrated over
                use_r: if False, use geo.z_fine instead of geo.r_fine for integration
                range_spec: array slice specification for geo.r_fine to limit range of integration for tomographic bins
        """
        print "sph_klim: calculating D_O_I_D_delta_alpha"
        d_delta_bar = self.D_delta_bar_D_delta_alpha(geo,force_recompute,tomography=False)
        result = np.zeros((d_delta_bar.shape[1],integrand.shape[1]))
        #the variable to integrate over
        if use_r:
            x = geo.r_fine
        else:
            x = geo.z_fine
        #allow a restriction to the range of r_fine or z_fine,TODO check if can default
        if range_spec is not None:
            x = x[range_spec]
            d_delta_bar = d_delta_bar[range_spec,:]
       # dxs = np.diff(x,axis=0)
        dx1s = (np.diff(x)*d_delta_bar[1::].T)/2.
        dx2s = (np.diff(x)*d_delta_bar[:-1:].T)/2.
        for ll in xrange(0, integrand.shape[1]):
            #TODO can be sped up if dx is constant
            #note this is just the trapezoidal rule with d_delta_bar absorbed into dx for a minor speedup
            result[:,ll] = (np.dot(dx2s,integrand[:-1:,ll])+np.dot(dx1s,integrand[1::,ll]))
            #result[:,ll] = trapz2((d_delta_bar.T*integrand[:,ll]).T,dx=dxs,given_dx=True)
        print "sph_klim: got D_O_I_D_delta_alpha"
        return result

    #Note: the basis has an allow_caching setting for whether to allow cacheing between function calls for the same geo
    #Cacheing is potentially dangerous if the geo changes, so it should not be allowed to. #TODO add cache_good flag to geo
    #Note that this function will cache alm and R_int internally as needed regardless of cache_alm or allow_caching, but it won't save the results for future function calls if caching is prohibited
    def D_delta_bar_D_delta_alpha(self,geo,force_recompute = False,tomography=True):
        """Calculate \frac{\partial\bar{\delta}}{\partial\delta_\alpha}
            inputs:
                geo: an input Geo object for the geometry
                tomography: if True use tomographic (coarse) bins, otherwise use resolution (fine) bins for r integrals
                force_recompute: clears the cache: use this if geo has changed.
        """
        #r=np.array([r_min,r_max])
        #TODO Check this
        print "sph_klim: begin D_delta_bar_D_delta_alpha with geo id: ",id(geo)

        #Caching implements significant speedup, check caches
        if self.allow_caching and not force_recompute:
            result_cache = self.ddelta_bar_cache.get(str(id(geo)))
            if result_cache is not None:
                if tomography and ('tomo' in result_cache):
                    print "sph_klim: tomographic bins retrieved from cache"
                    return result_cache['tomo']
                elif (not tomography) and ('fine' in result_cache):
                    print "sph_klim: fine bins retrieved from cache"
                    return result_cache['fine']
                else:
                    print "sph_klim: cache miss with nonempty cache"
            else:
                print "sph_klim: cache miss with empty cache"
                self.ddelta_bar_cache[str(id(geo))] = {}


        a_00=geo.a_lm(0,0)
        print "sph_klim: a_00="+str(a_00)
       # Omega=a_00*np.sqrt(4*np.pi)
        r_cache = {}

        if tomography:
            rbins = geo.rbins
            result=np.zeros((rbins.shape[0],self.C_id.shape[0]))
            print "sph_klim: calculating with tomographic (coarse) bins"
        else:
            rbins = geo.rbins_fine
            result=np.zeros((rbins.shape[0],self.C_id.shape[0]))
            print "sph_klim: calculating with resolution (fine) slices"

        norm=3./(rbins[:,1]**3 - rbins[:,0]**3)/(a_00*2.*np.sqrt(np.pi))

        for itr in xrange(self.C_id.shape[0]):
            ll=int(self.C_id[itr,0])
            kk=self.C_id[itr,1]
            mm=int(self.C_id[itr,2])
            #TODO just precompute the r_parts
            if (kk,ll) in r_cache:
                r_part = r_cache[(kk,ll)]
            else:
                #r_part = np.zeros(rbins.shape[0])
#                r_part2 = np.zeros(rbins.shape[0])
#                for i in xrange(0,rbins.shape[0]):
#                    r_part2[i] = R_int(rbins[i],kk,ll)*norm[i]
                r_part = (self.rints[ll](rbins[:,1]*kk)-self.rints[ll](rbins[:,0]*kk))/kk**3*norm
#                atol_loc = np.max(r_part)*1e-5
#                assert(np.allclose(r_part,r_part2,atol=atol_loc,rtol=1e-5))
#                if not np.allclose(r_part,r_part2,atol=atol_loc,rtol=1e-3):
#                    print r_part/r_part2
#                    r_grid3 = np.linspace(0.,self.r_max,self.params['x_grid_size'])
#                    integrand3= r_grid3**2*j_n(ll,r_grid3*kk)
#                    integrated3 = InterpolatedUnivariateSpline(r_grid3,cumtrapz(integrand3,r_grid3,initial=0.),k=3,ext=2)
#                    r_part3 = (integrated3(rbins[:,1])-integrated3(rbins[:,0]))*norm
#                    print r_part/r_part3
#                    print r_part2/r_part3
#                    print ll,kk
#                    raise RuntimeError('values do not match')
                r_cache[(kk,ll)] = r_part
            result[:,itr] = r_part*geo.a_lm(ll,mm)

        if self.allow_caching:
            if tomography:
                self.ddelta_bar_cache[str(id(geo))]['tomo'] = result
            else:
                self.ddelta_bar_cache[str(id(geo))]['fine'] = result
        print "sph_klim: finished d_delta_bar_d_delta_alpha for geo id: ",id(geo)
        return result

def R_int(r_range,k,ll):
    """ returns \int R_n(rk_alpha) r2 dr
        inputs:
            r_range: [min r, max r] r range to integrate over
            k: k_alpha, zero of bessel function
            ll: index of bessel function to use"""
    # I am using the spherical Bessel function for R_n, but that might change
    #TODO change name of j_n to sph_j_n or something
    def integrand(r):
        return r**2*j_n(ll,r*k)
    #TODO can be done with trapz
    I = quad(integrand,r_range[0],r_range[1],epsabs=10e-20,epsrel=10-7)[0]
    #TODO check if eps logic needed
    return I

#I_alpha checked
def I_alpha(k_alpha,k,r_max,l_alpha):
    """return the integral \int_0^r_{max} dr r^2 j_{\l_alpha}(k_\alpha r)j_{l_\alpha}(k r)
    needed to calculate long wavelength covariance matrix."""
    a=k_alpha*r_max
    b=k*r_max
    l=l_alpha+.5
    return np.pi/2./np.sqrt(k_alpha*k)/(k_alpha**2 - k**2)*r_max*(-k_alpha*jv(l-1,a)*jv(l,b))

def norm_factor(ka,la,r_max):
    """Get normalization factor, which is I_\alpha(k_\alpha,r_{max}) simplified"""
    return -np.pi*r_max**2/(4.*ka)*jv(la+1.5,ka*r_max)*jv(la-0.5,ka*r_max)

#if __name__=="__main__":
#    import geo
#
#    d=np.loadtxt('Pk_Planck15.dat')
#    k=d[:,0]; P=d[:,1]
#
#    zs=np.array([.1,.2,.3])
#    z_fine = np.arange(0.01,np.max(zs),0.01)
#    Theta=[np.pi/4,np.pi/2.]
#    Phi=[0,np.pi/3.]
#
#
#    from cosmopie import CosmoPie
#    C=CosmoPie(k=k,P_lin=P)
#
#    geometry=geo.RectGeo(zs,Theta,Phi,C,z_fine)
#
#    r_max=C.D_comov(0.5)
#
#    k_cut = 0.010
#    l_ceil = 100
#    R=SphBasisK(r_max,C,k_cut,l_ceil)
#    print R.C_size
#
#    r_min=C.D_comov(.1)
#    r_max=C.D_comov(.2)
#
#    print 'this is r range', r_min, r_max
