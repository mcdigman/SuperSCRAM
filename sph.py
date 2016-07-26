import numpy as np
from numpy import pi, sqrt, sin, cos
from sph_functions import j_n, jn_zeros, dJ_n, Y_r
from scipy.special import jv
from scipy.integrate import trapz, quad,dblquad
from scipy.interpolate import InterpolatedUnivariateSpline,interp1d
import sys 
from time import time

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


class sph_basis(object): 
	
	def __init__(self,r_max,l_alpha,n_zeros,CosmoPie):#,geometry,CosmoPie):
		
		''' inputs:
			r_max = the maximum radius of the sector 
			l_alpha = the l in j_l(k_alpha r) 
			n_zeros = the number of zeros 
			k = the wave vector k of P(k)
			P = the linear power specturm 
			important! no little h in any of the calculations 
		''' 
		
		k_in,P_lin_in=CosmoPie.get_P_lin()
	        #k = np.logspace(np.log10(np.min(k_in)),np.log10(np.max(k_in))-0.00001,6000000)
                #k = k_in
                kmin = 10**-4
                kmax = 100.0

	        k = np.linspace(kmin,kmax,1000000)
                P_lin = interp1d(k_in,P_lin_in)(k)

		self.l=l_alpha
	        self.r_max = r_max	
		# define the super mode wave vector k alpha
		# and also make the map from l_alpha to k_alpha 
		
		# this needs to be made faster 
                t1 = time()
		self.k_alpha=np.zeros(l_alpha.size*n_zeros) 
		
		
		self.map_l_to_alpha=np.zeros(self.k_alpha.size) 
		self.lm_map=np.zeros((l_alpha.size,3),dtype=object)  
		self.lm=np.zeros((l_alpha.size,2),dtype=object)
		m_size=np.zeros_like(l_alpha)   
		C_size=0
		for i in range(l_alpha.size):
		    m=np.arange(-l_alpha[i], l_alpha[i]+1)    
		    j=i*n_zeros
		    self.lm_map[i,0]=l_alpha[i]
		    self.lm_map[i,1]=jn_zeros(l_alpha[i],n_zeros)/r_max
		    self.lm_map[i,2]=m
		    C_size=self.lm_map[i,1].size*self.lm_map[i,2].size + C_size
		    self.k_alpha[j:j+n_zeros]=jn_zeros(l_alpha[i],n_zeros)/r_max
		    self.map_l_to_alpha[j:j+n_zeros]=l_alpha[i]
		    self.lm[i,0]=l_alpha[i]
		    self.lm[i,1]=m
	        self.C_size = C_size	
		self.C_id=np.zeros((C_size,3))
		self.C_alpha_beta=np.zeros((self.C_id.shape[0],self.C_id.shape[0]))
		id=0

		for a in range(self.lm_map.shape[0]):
		    ll=self.lm_map[a,0]
		    kk=self.lm_map[a,1]
		    mm=self.lm_map[a,2]
                    id_m1 = id
                    #self.Is = np.zeros((kk.size,k.size))
                    self.norms = np.zeros(k.size)
		    for c in range(mm.size):
                        id_k1 = id
		        for b in range(kk.size):
		            self.C_id[id,0]=ll
		            self.C_id[id,1]=kk[b]
		            self.C_id[id,2]=mm[c]
                            if c==0:
                                #self.Is[b] = I_alpha(kk[b],k,r_max,ll)
                                self.norms[b] = self.norm_factor(kk[b],ll)
		            id=id+1
                        print id,id_m1,id_k1
                        #calculate I integrals and make table
                        if c==0:
                            integrand1 = k*P_lin*jv(ll+0.5,k*self.r_max)**2
                            for b in range(0,kk.size):
                                #A = self.Is[b]
                                print b
                                for d in range(b,kk.size):
                                    #B = self.Is[d]
                                    #coeff = np.pi/(2.*self.norms[b]*self.norms[d])*self.r_max**2*np.sqrt(kk[b]*kk[d])*jv(ll-0.5,kk[b]*self.r_max)*jv(ll-0.5,kk[d]*self.r_max)
                                    coeff = 8.*np.sqrt(kk[b]*kk[d])*kk[b]*kk[d]/(np.pi*self.r_max**2*jv(ll+1.5,kk[b]*self.r_max)*jv(ll+1.5,kk[d]*self.r_max))
                                    #self.C_alpha_beta[id_k1+b,id_k1+d]=2./(np.pi*self.norm_factor(kk[b],ll)*self.norm_factor(kk[d],ll))*trapz(k**2*P_lin*A*B,k); #check coefficient
                                    self.C_alpha_beta[id_k1+b,id_k1+d]=coeff*trapz(integrand1/((k**2-kk[b]**2)*(k**2-kk[d]**2)),k); #check coefficient
                                    self.C_alpha_beta[id_k1+d,id_k1+b]=self.C_alpha_beta[id_k1+b,id_k1+d];
                        else:
                            for b in range(0,kk.size):
                                for d in range(b,kk.size):
                                    self.C_alpha_beta[id_k1+b,id_k1+d] = self.C_alpha_beta[id_m1+b,id_m1+d] 
                                    self.C_alpha_beta[id_k1+d,id_k1+b] = self.C_alpha_beta[id_m1+d,id_m1+b] 
	        #TODO can make more efficient if necessary	
                t2 = time()
                print "basis time: ",t2-t1
		#for alpha in range(self.C_id.shape[0]):
		#    for beta in range(max(0,alpha-n_zeros),min(alpha+n_zeros,self.C_id.shape[0])):
		#        l1=self.C_id[alpha,0]; l2=self.C_id[beta,0]
		#        m1=self.C_id[alpha,2]; m2=self.C_id[beta,2]
		#        k1=self.C_id[alpha,1]; k2=self.C_id[beta,1]
		#        if ( (l1==l2) & (m1==m2)):
		#            #A=I_alpha(k1,k,r_max,l1)
		#            #B=I_alpha(k2,k,r_max,l2)
                #            A = self.Is[alpha]
                #            B = self.Is[beta]
		#            #self.C_alpha_beta[alpha,beta]=8.*trapz(k**2*P_lin*A*B,k)
		#            self.C_alpha_beta[alpha,beta]=2./np.pi*trapz(k**2*P_lin*A*B,k)*(-1)**l1;
		#            
                print "max c 1",np.max(self.C_alpha_beta)
		self.F_alpha_beta=np.linalg.inv(self.C_alpha_beta)

        #I_\alpha(k_\alpha,r_{max}) simplified
        def norm_factor(self,ka,la):
            return -np.pi*self.r_max**2/(4.*ka)*jv(la+1.5,ka*self.r_max)*jv(la-0.5,ka*self.r_max)

	def Cov_alpha_beta(self):
		return self.C_alpha_beta
	
	def get_F_alpha_beta(self):
	    return self.F_alpha_beta 
		
	def k_LW(self):
		# return the long-wavelength wave vector k 
		return self.k_alpha
        #slow part, TODO consider caching results	
	def a_lm(self,geo,l,m):    
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
	
	def R_int(self,r_range,k,n):
	# returns \int R_n(rk_alpha) r2 dr
	# I am using the spherical Bessel function for R_n, but that might change  
		r_min,r_max=r_range 
		

		def integrand(r):
			return r**2*j_n(n,r*k)
                #CHANGED
		#I= quad(integrand,r_min,r_max)[0]
		I= quad(integrand,r_min,r_max)[0]
		
		if (np.absolute(I) <= eps):
			I=0.0
			
		return I 
	
	def D_delta_bar_D_delta_alpha(self,r_min,r_max,geo):
	    	    
	    r=np.array([r_min,r_max])
            #TODO Check this
	    a_00=self.a_lm(geo,0,0)
            print a_00
            print "theta",geo.Theta
            print "phi",geo.Phi
	   # Omega=np.sqrt(a_00)*4.*np.pi
            #CHANGED
	   # Omega=a_00*np.sqrt(4*pi) 
	   # norm=3./(r_max**3 - r_min**3)/Omega
	    norm=3./(r_max**3 - r_min**3)/(a_00*2.*np.sqrt(np.pi))
	
	    result=np.zeros(self.C_id.shape[0])
            #store alms for current l, m pair because computing them is slow
            alm_last = 0.
            ll_last = -1
            mm_last = 0
	    for id in range(self.C_id.shape[0]):
	        ll=self.C_id[id,0]
	        kk=self.C_id[id,1]
	        mm=self.C_id[id,2]
                #TESTING
                if self.norm_factor(kk,ll)<0:
                    warn("critical issue, negative norm factor")
                    sys.exit()
	        if ll_last == ll and mm_last == mm:
                    result[id] = self.R_int(r,kk,ll)*alm_last*norm#*self.norm_factor(kk,ll)
                else:
                    alm_last = self.a_lm(geo,ll,mm)
                    ll_last = ll
                    mm_last = mm
                    result[id]=self.R_int(r,kk,ll)*alm_last*norm#*self.norm_factor(kk,ll)

	    return result
	    
		 
if __name__=="__main__":

	d=np.loadtxt('Pk_Planck15.dat')
	k=d[:,0]; P=d[:,1]
	
	z_bins=np.array([.1,.2,.3])
	Theta=[np.pi/4,np.pi/2.]
	Phi=[0,np.pi/3.]
	geometry=np.array([z_bins,Theta,Phi])
	
	from cosmopie import CosmoPie
	cp=CosmoPie(k=k,P_lin=P)
	
	r_max=cp.D_comov(4)
	
	R=sph_basis(r_max,np.array([0,1,2,3]),3,cp)
	
	
	r_min=cp.D_comov(.1)
	r_max=cp.D_comov(.2)
	
	print 'this is r range', r_min, r_max 
	X=R.D_delta_bar_D_delta_alpha(r_min,r_max,Theta,Phi)
	
	
	
	for i in range(4):
	    #norm=3./(r_max**3 - r_min**3)/(a_00*2.*np.sqrt(np.pi))
		print 'l',X[i,0]
		print 'm',X[i,1]
		print 'deriv',X[i,2]
	
	a,b=R.Cov_alpha_beta()
	print b 
	
	
		
