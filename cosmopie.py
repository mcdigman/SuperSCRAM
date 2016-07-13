''' 
	This class computes various cosmological parameters.
	The default cosmology is Planck 2015. 
	Joseph E. McEwen (c) 2016
	mcewen.24@osu.edu 
	
''' 

import numpy as np
from numpy import pi 
from scipy.integrate import romberg, quad, trapz,cumtrapz
#from talk_to_class import class_objects
import sys
from scipy.interpolate import interp1d

eps=np.finfo(float).eps

class CosmoPie :
	
	def __init__(self,cosmology=None, P_lin=None, k=None,lin_type='class',precompute=True,z_max=4.0,z_space=0.01):
		# default to Planck 2015 values 
		
		if cosmology is None:
		
			self.Omegabh2 = 0.02230
			self.Omegach2 = 0.1188
			self.Omegamh2 = 0.14170
			self.OmegaL   = .6911
			self.Omegam   = .3089
			self.H0       = 67.74 
			self.sigma8   = .8159 
			self.h        = .6774 
			self.Omegak   = 0.0 # check on this value 
			self.Omegar   = 0.0 # check this value too
		else :
			self.Omegabh2 = cosmology['Omegabh2']
			self.Omegach2 = cosmology['Omegach2']
			self.Omegamh2 = cosmology['Omegamh2']
			self.OmegaL   = cosmology['OmegaL']
			self.Omegam   = cosmology['Omegam']
			self.H0       = cosmology['H0']
			self.sigma8   = cosmology['sigma8']
			self.h        = cosmology['h']
			self.Omegak   = cosmology['Omegak']
			self.Omegar   = cosmology['Omegar']

		
		self.P_lin=P_lin
		self.k=k    
		
		# if lin_type=='class':
# 		    X=class_objects(cosmology)
# 		    P=X.linear_power(self.k*self.h)/self.h**3
# 		    self.P_lin=P
		     
		# solar mass
		self.M_sun=1.9885*1e30 # kg
		
		# parsec 
		self.pc=3.08567758149*1e16 # m 
		
		# Newton's constant 
		self.GN=6.67408*10**(-11) # m^3/kg/s^2
		
		# speed of light 
		self.c        = 2.997924580*1e5 #km/s
		
		self.DH       = self.c/self.H0
		
		self.tH= 1/self.H0
			
                #curvature
                self.K = -self.Omegak*(self.H0/self.c)**2

                #precompute some things for speedups if desired (initially use no precompute)
	        self.precompute=False
                if precompute:
                    z_grid = np.arange(0.,z_max,z_space)
                    #there are massive savings from precomputing G because of the integral it contains
                    G_arr = np.zeros(z_grid.size)
                    #first compute the integral in G for the whole range \int_0^{1e4}(integrand)
		    integrand1 = lambda zp : (1+zp)*self.H0**3/self.H(zp)**3
		    G_base = quad(integrand1,0.,1e4)[0]
                    #subtract the cumulative result from G_base so final result is \int_z^{1e4}(integrand)
                    integrand2 = (1.+z_grid)*self.H0**3/self.H(z_grid)**3
                    self.G_p = interp1d(z_grid,2.5*self.Omegam/self.H0*self.H(z_grid)*(G_base-cumtrapz(integrand2,z_grid,initial=0.)))
               #     for i in range(0,z_grid.size):
               #         G_arr[i] = self.G(z_grid[i])
               #     self.G_p = interp1d(z_grid, G_arr)
                self.precompute=precompute
                    
	def Ez(self,z):
		zp1=z + 1
		return np.sqrt(self.Omegam*zp1**3 + self.Omegar*zp1**4 + self.Omegak*zp1**2 + self.OmegaL) 
	  
	def H(self,z):
		return self.H0*self.Ez(z)  
	
	def dH_da(self,z):
		# the derivative of H with respect to a 
		zp1=z + 1
		
		return -(1+z)**2*self.H0/2./self.Ez(z)*(3*self.Omegam*zp1**2 +4*self.Omegar*zp1**3  +2*self.Omegak*zp1 )
	
	# distances, volumes, and time
	# -----------------------------------------------------------------------------
	def D_comov(self,z):
		# the line of sight comoving distance 
		I = lambda zp : 1/self.Ez(zp)
		return self.DH*quad(I,0,z)[0]
	
        def D_comov_A(self,z):
		# the comoving angular diameter distance
                if self.K ==0:
                    return self.D_comov(z)
                else:
                    sqrtK = np.sqrt(abs(self.K))
		    return 1./sqrtK*np.sin(sqrtK*self.D_comov(z))
	
	def D_comov_dz(self,z):
	    return self.DH/self.Ez(z) 
		
	def D_comov_T(self,z):
		# the transverse comoving distance 
		
		if (self.Omegak==0):   
			return self.D_comov(z)
		
		if (self.Omegak > 0):
			sq=np.sqrt(self.Omegak)
			return self.DH/sq*np.sinh(sq*self.D_comov(z)/self.DH)
		
		if (self.Omegak < 0): 
			sq=np.sqrt(self.Omegak)
			return self.DH/sq*np.sin(sq*self.D_comov(z)/self.DH)
			
		
	def D_A(self,z):
		# angular diameter distance
		return self.D_comov_T(z)/(1+z)
		
	def D_L(self,z):
		# luminosity distance 
		return (1+z)**2*self.D_A(z)
	
	def DV(self,z):
		# comoving volume element, with out the d\Omega 
		# dV/dz
		return self.DH*(1+z)**2*self.D_A(z)**2/self.Ez(z)
		
	def D_A_array(self,z):

	    d=np.zeros_like(z)
	    for i in range(z.size):
	        d[i]=self.D_A(z[i])
	    return d 
	
	def D_c_array(self,z):

	    d=np.zeros_like(z)
	    for i in range(z.size):
	        d[i]=self.D_comov(z[i])
	    return d 
	    
	
	def look_back(self,z):
		I = lambda z : 1/self.Ez(z)/(1+z)
		return self.tH*quad(I,0,z)[0]
	# -----------------------------------------------------------------------------   
	
	# Growth functions
	# -----------------------------------------------------------------------------
# 	def G(self,z):
# 		# linear Growth factor (Eqn. 7.77 in Dodelson)
# 		# 1 + z = 1/a 
# 		# G = 5/2 Omega_m H(z)/H_0 \int_0^a da'/(a'H(a')/H_0)^3
# 		# Omega_m=Omega_m_z/a^3/H(z)^2
# 		a=1/float(1+z)
# 		def Integrand(ap):
# 		    zp=1/float(ap)-1
# 		   
# 		    denominator=float((ap*self.H(zp))**3)
# 		    
# 		    return 1/denominator 
# 	
# 		return 5/2.*self.Omegam_z(z)*a**2*self.H(z)**3*quad(Integrand,1e-5,a)[0]
# 		#return 5/2.*self.Omegam_z(z)*a**2*self.H(z)**3*romberg(Integrand,eps,a)

	#TODO support vector z	
	def G(self,z):
            if self.precompute:
                return self.G_p(z)
            else:
		integrand = lambda zp : (1+zp)*self.H0**3/self.H(zp)**3
		#return 2.5*self.Omegam/self.H0*self.H(z)*romberg(integrand,z,1e4)
		return 2.5*self.Omegam/self.H0*self.H(z)*quad(integrand,z,1e4)[0]

# 	def G(self,z):
# 		integrand = lambda zp : 1/(1 + zp)**2/self.H(z)**3 
# 		return 2.5*self.Omegam_z(z)/(1+z)**2*self.H(z)**3*quad(integrand, z,1e5)[0]
	def G_norm(self,z):
		# the normalized linear growth factor
		# normalized so the G(0) =1 
                G_0=self.G(0.)
		return self.G(z)/G_0 
		
		
	def log_growth(self,z):
		# using equation 3.2 from Baldauf 2015 
		a=1/(1+z)
		print 'what I think it is', a/self.H(z)*self.dH_da(z) + 5/2.*self.Omegam*self.G_norm(0)/self.H(z)**2/a**2/self.G_norm(z)
		return -3/2.*self.Omegam/a**3*self.H0**2/self.H(z)**2 + 1/self.H(z)**2/a**2/self.G_norm(z)
	
	def G_array(self,z):
		# the normalized linear growth factor 
		# for an array 
		if (type(z)==float):
			return np.array([self.G_norm(z)])
		result=np.zeros(z.size)
		for i in range(z.size):
			result[i]=self.G_norm(z[i])
		return result 
			
	# ----------------------------------------------------------------------------
	 
	# halo and matter stuff 
	# -----------------------------------------------------------------------------   
	def delta_c(self,z):
		# critical threshold for spherical collapse, as given 
		# in the appendix of NFW 1997 
		A=0.15*(12.*pi)**(2/3.)
		
		if ( (self.Omegam ==1) & (self.OmegaL==0)):
			d_crit=A
		if ( (self.Omegam < 1) & (self.OmegaL ==0)):
			d_crit=A*self.Omegam**(0.0185)
		if ( (self.Omegam + self.OmegaL)==1.0):
			d_crit=A*self.Omegam**(0.0055)
			   
		d_c=d_crit#/self.G_norm(z)
		return d_c
		
	def delta_v(self,z):
		# over density for virialized halo
		
		A=178.0
		if ( (self.Omegam_z(z) ==1) & (self.OmegaL_z(z)==0)):
			d_v=A
		if ( (self.Omegam_z(z) < 1) & (self.OmegaL_z(z) ==0)):
			d_v=A/self.Omegam_z(z)**(0.7)
		if ( (self.Omegam_z(z) + self.OmegaL_z(z))==1.0):
			d_v=A/self.Omegam_z(z)**(0.55)
			   
		return d_v/self.G_norm(z)
		
		
	def nu(self,z,mass):
		# calculates nu=(delta_c/sigma(M))^2
		# delta_c is the overdensity for collapse 
		return (self.delta_c(z)/self.sigma_m(mass,z))**2
	
	def sigma_m(self,mass,z):
		# RMS power on a scale of R(mass)
		# rho=mass/volume=mass
		R=3/4.*mass/self.rho_bar(z)/pi
		R=R**(1/3.)	   
		return self.sigma_r(z,R)
	
	def sigma_r(self,z,R):
		# returns RMS power on scale R
		# sigma^2(R)= \int dlogk/(2 pi^2) W^2 P(k) k^3 
		# user needs to adjust for growth factor upon return 
		if self.P_lin is None:
			raise ValueError('You need to provide a linear power spectrum and k to get sigma valeus')
		if self.k is None:
			raise ValueError('You need to provide a linear power spectrum and k to get sigma valeus')
		W=3.0*(np.sin(self.k*R)/self.k**3/R**3-np.cos(self.k*R)/self.k**2/R**2)
		#P=self.G_norm(z)**2*self.P_lin
		P=self.P_lin
		I=trapz(W*W*P*self.k**3,np.log(self.k))/2./pi**2
		
		return np.sqrt(I)
		
		
	# densities as a function of redshift 
	def Omegam_z(self,z):
		# matter density as a function of redshift 
		return self.Omegam*(1+z)**3/self.Ez(z)**2
	
	def OmegaL_z(self,z):
		# dark energy density as function of redshift 
		return self.OmegaL/self.Ez(z)**2

	def Omegak_z(self,z):
		return 1-self.Omegam_z(z)-self.OmegaL_z(z)
	
	def Omega_tot(self,z):
		return self.Omegak_z(z) + self.OmegaL_z(z) + self.Omegam_L(z)
	
	def rho_bar(self,z):
		# return average density in units of solar mass and h^2 
		return self.rho_crit(z)*self.Omegam_z(z)
	
	def rho_crit(self,z):
		# return critical density in units of solar mass and h^2 
		factor=1e12/self.M_sun*self.pc	   
		#print 'rho crit [g/cm^3] at z =', z, 3*self.H(z)**2/8./pi/self.GN*1e9/(3.086*10**24)**2/10**2
		return 3*self.H(z)**2/8./pi/self.GN*factor/self.h**2
	
	def get_P_lin(self):
	    return self.k, self.P_lin
	   
	# -----------------------------------------------------------------------------
		
	 
if __name__=="__main__": 

	C=CosmoPie()
	z=3.5
	z=.1
	print('Comoving distance',C.D_comov(z))
	print('Angular diameter distance',C.D_A(z))
	print('Luminosity distance', C.D_L(z)) 
	print('Growth factor', C.G(z)) 
	print('Growth factor for really small z',C.G(1e-10))
	z=0.0
	print('logrithmic growth factor', C.log_growth(z))
	print('compare logrithmic growth factor to approxiamtion', C.Omegam**(-.6), C.Omegam)
	print('cirtical overdensity ',C.delta_c(0)  ) 
		
	z=np.linspace(0,5,80) 
	D1=np.zeros(80)
	D2=np.zeros(80)
	D3=np.zeros(80)
	for i in range(80):
		D1[i]=C.D_A(z[i])/C.DH
		D2[i]=C.D_L(z[i])/C.DH
		D3[i]=C.DV(z[i])/C.DH
		
	import matplotlib.pyplot as plt
	
	ax=plt.subplot(121)
	ax.set_xlabel(r'$z$', size=20)
	
	ax.plot(z, D1, label='Angular Diameter distance [Mpc]') 
	#ax.plot(z, D2, label='Luminosity distance [Mpc]') 
	plt.grid()
	
	ax=plt.subplot(122)
	ax.set_ylim(0,1)
	ax.plot(z, D3/C.DH**3, label='Angular Diameter distance [Mpc]') 
	#ax.plot(z, D2, label='Luminosity distance [Mpc]') 
	plt.grid()
	
	plt.show()
	
	   
	
	
