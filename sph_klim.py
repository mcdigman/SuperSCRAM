import numpy as np
from numpy import pi, sqrt, sin, cos
from sph_functions import j_n, jn_zeros_cut, dJ_n, Y_r
from scipy.special import jv
from scipy.integrate import trapz, quad,dblquad
from scipy.interpolate import InterpolatedUnivariateSpline,interp1d
from algebra_utils import cholesky_inv
import scipy as sp
import sys 
from time import time
import fisher_matrix as fm
import defaults

# the smallest value 
eps=np.finfo(float).eps	
#I_alpha checked
def I_alpha(k_alpha,k,r_max,l_alpha):
			# return the integral \int_0^r_{max} dr r^2 j_{\l_alpha}(k_\alpha r)j_{l_\alpha}(k r)
			
			a=k_alpha*r_max; b=k*r_max 
			l=l_alpha+.5
          #              print pi/2./sqrt(k_alpha*k)/(k_alpha**2 - k**2)*r_max*(k*jv(l-1,b)*jv(l,a)-k_alpha*jv(l-1,a)*jv(l,b))
                        return pi/2./sqrt(k_alpha*k)/(k_alpha**2 - k**2)*r_max*(-k_alpha*jv(l-1,a)*jv(l,b))
		#	return pi/2./sqrt(k_alpha*k)/(k_alpha*82 - k**2)*r_max*(k*jv(l-1,b)*jv(l,a)-k_alpha*jv(l-1,a)*jv(l,b))
	#		return pi/2./sqrt(k_alpha*k)/(k_alpha**2 - k**2)*r_max*(k*jv(l-1,b)*jv(l,a)-k_alpha*jv(l-1,a)*jv(l,b))
		#	print  pi/2.*r_max**2/sqrt(k_alpha*k)*(a*jv(l,b)*dJ_n(l,a)-b*jv(l,a)*dJ_n(l,b))/(b**2-a**2)
		#	print pi/2./sqrt(k_alpha*k)*r_max*(k_alpha*jv(l,b)*dJ_n(l,a)-k*jv(l,a)*dJ_n(l,b))/(k**2-k_alpha**2)
			
                #        sys.exit()


class sph_basis_k(object): 
	
	def __init__(self,r_max,C,k_cut = 2.0,l_ceil = 100,params = defaults.basis_params):#,geometry,CosmoPie):
		
		''' inputs:
			r_max = the maximum radius of the sector 
			l_alpha = the l in j_l(k_alpha r) 
			n_zeros = the number of zeros 
			k = the wave vector k of P(k)
			P = the linear power specturm 
			important! no little h in any of the calculations 
		''' 
	        print "sph_klim: begin init basis id: ",id(self)	
		k_in,P_lin_in=C.get_P_lin()
	        #k = np.logspace(np.log10(np.min(k_in)),np.log10(np.max(k_in))-0.00001,6000000)
                #k = k_in
                self.params = params
                kmin = params['k_min']
                kmax = params['k_max']

                self.allow_caching = params['allow_caching']
                if self.allow_caching:
                    self.ddelta_bar_cache = {}
	        k = np.linspace(kmin,kmax,params['n_bessel_oversample'])
                P_lin = interp1d(k_in,P_lin_in)(k)

	        self.r_max = r_max	
		# define the super mode wave vector k alpha
		# and also make the map from l_alpha to k_alpha 	
                t1 = time()
		self.k_num = np.zeros(l_ceil+1,dtype=np.int)
                self.k_zeros = np.zeros(l_ceil+1,dtype=object)
                n_l = 0
                for ll in range(0,self.k_num.size):
                    k_alpha = jn_zeros_cut(ll,k_cut*r_max)/r_max
                    #once there are no zeros above the cut, skip higher l
                    if k_alpha.size == 0:
                        print "sph_klim: cutting off all l>=",ll
                        break
                    else:
                        self.k_num[ll] = k_alpha.size
                        self.k_zeros[ll] = k_alpha
                        n_l += 1
                l_alpha = np.arange(0,n_l)
                
		self.l=l_alpha #TODO unnecessary
	#	self.map_l_to_alpha=np.zeros(self.k_alpha.size) 
		self.lm_map=np.zeros((l_alpha.size,3),dtype=object)  
		self.lm=np.zeros((l_alpha.size,2),dtype=object)
		m_size=np.zeros_like(l_alpha)   
		C_size=0
		for i in range(l_alpha.size):
		    m=np.arange(-l_alpha[i], l_alpha[i]+1)    
		    #j=i*n_zeros
		    self.lm_map[i,0]=l_alpha[i]
		    self.lm_map[i,1]=self.k_zeros[i]
		    self.lm_map[i,2]=m
		    C_size=self.lm_map[i,1].size*self.lm_map[i,2].size + C_size
		    #self.k_alpha[j:j+self.k_num[i]]=k_alpha
		   # self.map_l_to_alpha[j:j+self.k_num[i]]=l_alpha[i]
		    self.lm[i,0]=l_alpha[i]
		    self.lm[i,1]=m
                    #print "l,n_zeros",l_alpha[i],self.k_num[i]
	        self.C_size = C_size	
                print "sph_klim: basis size: ",self.C_size
		self.C_id=np.zeros((C_size,3))
		self.C_alpha_beta=np.zeros((self.C_id.shape[0],self.C_id.shape[0]))


                print "sph_klim: begin constructing covariance matrix. basis id: ",id(self)
		itr=0
		for a in range(self.lm_map.shape[0]):
		    ll=self.lm_map[a,0]
		    kk=self.lm_map[a,1]
		    mm=self.lm_map[a,2]

                    itr_m1 = itr

                    self.norms = np.zeros(k.size)
                    for b in range(kk.size):
                        self.norms[b] = self.norm_factor(kk[b],ll)
                    print "sph_klim: calculating covar for l=",ll
		    for c in range(mm.size):
                        itr_k1 = itr
		        for b in range(kk.size):
		            self.C_id[itr,0]=ll
		            self.C_id[itr,1]=kk[b]
		            self.C_id[itr,2]=mm[c]
		            itr=itr+1
                        #print itr,ll,mm[c],itr_k1
                        #calculate I integrals and make table
                        if c==0:
                            integrand1 = k*P_lin*jv(ll+0.5,k*self.r_max)**2
                            for b in range(0,kk.size):
                                #print b
                                for d in range(b,kk.size):
                                    coeff = 8.*np.sqrt(kk[b]*kk[d])*kk[b]*kk[d]/(np.pi*self.r_max**2*jv(ll+1.5,kk[b]*self.r_max)*jv(ll+1.5,kk[d]*self.r_max))
                                    self.C_alpha_beta[itr_k1+b,itr_k1+d]=coeff*trapz(integrand1/((k**2-kk[b]**2)*(k**2-kk[d]**2)),k); #check coefficient
                                    self.C_alpha_beta[itr_k1+d,itr_k1+b]=self.C_alpha_beta[itr_k1+b,itr_k1+d];
                        else:
                            for b in range(0,kk.size):
                                for d in range(b,kk.size):
                                    self.C_alpha_beta[itr_k1+b,itr_k1+d] = self.C_alpha_beta[itr_m1+b,itr_m1+d] 
                                    self.C_alpha_beta[itr_k1+d,itr_k1+b] = self.C_alpha_beta[itr_m1+d,itr_m1+b] 
	        #TODO can make more efficient if necessary	
                t2 = time()
                print "sph_klim: basis time: ",t2-t1
                #print "sph_klim: maximum element of covariance matrix: ",np.max(self.C_alpha_beta)
	        print "sph_klim: finished init basis id: ",id(self)	

        #I_\alpha(k_\alpha,r_{max}) simplified
        def norm_factor(self,ka,la):
            return -np.pi*self.r_max**2/(4.*ka)*jv(la+1.5,ka*self.r_max)*jv(la-0.5,ka*self.r_max)

	def Cov_alpha_beta(self):
		return self.C_alpha_beta
	
	def get_F_alpha_beta(self):
	    return cholesky_inv(self.C_alpha_beta)

        def get_fisher(self,allow_caching=False):
            return fm.fisher_matrix(cholesky_inv(self.C_alpha_beta),allow_caching)
		
	def k_LW(self):
		# return the long-wavelength wave vector k 
                return self.C_id[:,1]

        #get the partial derivatives of an sw observable wrt the basis given an integrand with elements at each r_fine i.e.  \frac{\partial O_i}{\partial \bar(\delta)(r_{fine})}
        #TODO this may not be very efficient
        def D_O_I_D_delta_alpha(self,geo,integrand,force_recompute = False):
            d_delta_bar = self.D_delta_bar_D_delta_alpha(geo,force_recompute,tomography=False)
            result = np.zeros((d_delta_bar.shape[1],integrand.shape[1]))
            for alpha in range(0,d_delta_bar.shape[1]):
                for ll in range(0, integrand.shape[1]):
                    result[alpha,ll] = trapz(d_delta_bar[:,alpha]*integrand[:,ll],geo.r_fine) 
            return result

        #Calculate D_delta_bar_D_delta_alpha. 
        #tomography: if true use tomographic (coarse) bins, otherwise use resolution (fine) bins for r integrals
        #force_recompute: clears the cache: use this if geo has changed.
        #Note: the basis has an allow_caching setting for whether to allow cacheing between function calls for the same geo
        #Caching is potentially dangerous if the geo changes, so it should not be allowed to. #TODO add cache_good flag to geo
        #cache_alm permits caching alm for future function calls. 
        #Note that this function will cache alm and R_int internally as needed regardless of cache_alm or allow_caching, but it won't save the results for future function calls if caching is prohibited
	def D_delta_bar_D_delta_alpha(self,geo,force_recompute = False,tomography=True,cache_alm=True):
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
                    elif cache_alm and ('alm' in result_cache):
                        #Can get here if only ever run with the other value of tomography before.
                        #Note: Storing alm from different values of tomography wouldn't work if alm were r dependent 
                        alm_cache = result_cache['alm']
                        print "sph_klim: alm retrieved from cache"
                    elif cache_alm:
                        #If it gets here, a previous run didn't store alm, which is not necessarily an error, but isn't the way the code works now.
                        warn("sph_klim: alm cache missed but cache miss not expected here. Calculating alm as usual. geo id: "+str(id(geo))+" basis id: "+str(id(basis)))
                        alm_cache = {}
                    else:
                        #Get here if never run before or cache has been cleared
                        print "sph_klim: cache miss with nonempty cache"
                        alm_cache = {}
                        
                else:
                    print "sph_klim: cache miss with empty cache"
                    self.ddelta_bar_cache[str(id(geo))] = {}
                    alm_cache = {}
            else:
                alm_cache = {}


	    a_00=a_lm(geo,0,0)
            print "sph_klim: a_00="+str(a_00)
            print "sph_klim: theta: "+str(geo.Theta)
            print "sph_klim: phi: "+str(geo.Phi)
	   # Omega=np.sqrt(a_00)*4.*np.pi
            #CHANGED
	   # Omega=a_00*np.sqrt(4*pi) 
	   # norm=3./(r_max**3 - r_min**3)/Omega
	    #result=np.zeros((geo.rs.size-1,self.C_id.shape[0]))
            r_cache = {} 
            #store alms for current l, m pair because computing them is slow
            alm_last = 0.
            ll_last = -1
            mm_last = 0

            #TODO move r binning to geo 
            if tomography:
                rs = geo.rs
	        result=np.zeros((geo.rs.size-1,self.C_id.shape[0]))
                print "sph_klim: calculating with tomographic (coarse) bins"
            else:
                rs = np.hstack((0.,geo.r_fine))
	        result=np.zeros((geo.r_fine.size,self.C_id.shape[0]))
                print "sph_klim: calculating with resolution (fine) slices"

	    for itr in range(self.C_id.shape[0]):
	        ll=self.C_id[itr,0]
	        kk=self.C_id[itr,1]
	        mm=self.C_id[itr,2]
                #TODO clean up these conditionals
                if (str(kk)+","+str(ll)) in r_cache:
                    r_part = r_cache[(str(kk)+","+str(ll))]
                else:
                    r_part = np.zeros(rs.size-1)
                    for i in range(0,rs.size-1):
                        r = np.array([rs[i],rs[i+1]])
	                norm=3./(r[1]**3 - r[0]**3)/(a_00*2.*np.sqrt(np.pi))   
                        r_part[i] = R_int(r,kk,ll)*norm
                    r_cache[(str(kk)+","+str(ll))] = r_part 
	        if ll_last == ll and mm_last == mm:
                    result[:,itr] = r_part*alm_last
                else:
                    alm_last = alm_cache.get(str(ll)+","+str(mm))
                    if alm_last is None:
                        alm_last = a_lm(geo,ll,mm)
                        alm_cache[str(ll)+","+str(mm)] = alm_last
                    ll_last = ll
                    mm_last = mm
                    result[:,itr] = r_part*alm_last
            if self.allow_caching:
                if tomography:
                    self.ddelta_bar_cache[str(id(geo))]['tomo'] = result
                else:
                    self.ddelta_bar_cache[str(id(geo))]['fine'] = result
                if cache_alm: 
                    self.ddelta_bar_cache[str(id(geo))]['alm'] = alm_cache
            print "sph_klim: finished d_delta_bar_d_delta_alpha for geo id: ",id(geo)
	    return result
def R_int(r_range,k,ll):
    # returns \int R_n(rk_alpha) r2 dr
    # I am using the spherical Bessel function for R_n, but that might change  
    r_min,r_max=r_range 
		
    #TODO change name to sph_j_n or something
    def integrand(r):
    	return r**2*j_n(ll,r*k)
    #CHANGED
    #I= quad(integrand,r_min,r_max)[0]
    I= quad(integrand,r_min,r_max)[0]
	
#	if (np.absolute(I) <= eps):
#		I=0.0
			
    return I 

#slow part, TODO consider caching results	
def a_lm(geo,l,m):    
	# returns \int d theta d phi \sin(theta) Y_lm(theta, phi)
	# theta is an array of 2 numbers representing the max and min of theta
	# phi is an array of 2 numbers representing the max and min of phi
	# l and m are the indices for the spherical harmonics 
        Theta = geo.Theta
        Phi = geo.Phi
	theta_min, theta_max=Theta
	phi_min, phi_max=Phi
	#def integrand(theta,phi):
	def integrand(phi,theta):
        #print theta, phi 
	    return sin(theta)*Y_r(l,m,theta,phi)
	
	#I = dblquad(integrand,theta_min,theta_max, lambda phi: phi_min, lambda phi :phi_max)[0]
	#I1 = dblquad(integrand,theta_min,theta_max, lambda phi: phi_min, lambda phi :phi_max)[0]
        #TODO I1 tsting purposes only
        I2 = geo.surface_integral(integrand)
        #if np.absolute(I1) <= eps:
                #    I1 = 0.0
                #if not I1==I2:
		#    print "I1,I2,eps",I1,I2,eps
                #    sys.exit()

	return I2
		 
if __name__=="__main__":
        import geo

	d=np.loadtxt('Pk_Planck15.dat')
	k=d[:,0]; P=d[:,1]
	
	zs=np.array([.1,.2,.3])
        z_fine = np.arange(0.01,np.max(zs),0.01)
	Theta=[np.pi/4,np.pi/2.]
	Phi=[0,np.pi/3.]
	

	from cosmopie import CosmoPie
	C=CosmoPie(k=k,P_lin=P)

	geometry=geo.rect_geo(zs,Theta,Phi,C,z_fine)
	
	r_max=C.D_comov(0.5)
	
        k_cut = 0.010
        l_ceil = 100
	R=sph_basis_k(r_max,C,k_cut,l_ceil)
        print R.C_size	
	
	r_min=C.D_comov(.1)
	r_max=C.D_comov(.2)
	
	print 'this is r range', r_min, r_max 
	#X=R.D_delta_bar_D_delta_alpha(r_min,r_max,geometry)
	

	
	
	#for i in range(4):
	    #norm=3./(r_max**3 - r_min**3)/(a_00*2.*np.sqrt(np.pi))
	#	print 'l',X[i,0]
	#	print 'm',X[i,1]
	#	print 'deriv',X[i,2]
	
	#a,b=R.Cov_alpha_beta()
	#print b 
	
	
		
