''' 
	This class computes various cosmological parameters.
	The default cosmology is Planck 2015. 
	Joseph E. McEwen (c) 2016
	mcewen.24@osu.edu 
	
''' 

import numpy as np
from numpy import pi 
from scipy.integrate import romberg, quad

eps=np.finfo(float).eps

class CosmoPie :
	
	def __init__(self,cosmology=None):
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
		
		
		self.c        = 2.998*1e5
		
		self.DH       = self.c/self.H0
			
		
	def Ez(self,z):
		zp1=z + 1
		return np.sqrt(self.Omegam*zp1**3 + self.Omegar*zp1**4 + self.Omegak*zp1**2 + self.OmegaL) 
	  
	def H(self,z):
		return self.H0*self.Ez(z)  
	
	def dH_da(self,z):
	    # the derivative of H with respect to a 
	    zp1=z + 1
	    
	    return -(1+z)**2*self.H0/2./self.Ez(z)*(3*self.Omegam*zp1**2 +4*self.Omegar*zp1**3  +2*self.Omegak*zp1 )
	
	# distances 
	# -----------------------------------------------------------------------------
	def D_comov(self,z):
		# the line of sight comoving distance 
		I = lambda z : 1/self.Ez(z)
		return self.DH*quad(I,0,z)[0]
		
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
	# -----------------------------------------------------------------------------   
	
	# Growth functions
	# -----------------------------------------------------------------------------
	def G(self,z):
		# linear Growth factor (Eqn. 7.77 in Dodelson)
		# 1 + z = 1/a 
		# G = 5/2 Omega_m H(z)/H_0 \int_0^a da'/(a'H(a')/H_0)^3
		def Integrand(a):
			return 1/(a*self.Ez(1/a-1))**3
	
		return 5/2.*self.Omegam*self.Ez(z)*quad(Integrand,eps,1/(1+z))[0]
		
	def G_norm(self,z):
		# the normalized linear growth factor
		# normalized so the G(0) =1 
		
		G_0=self.G(0)
		return self.G(z)/G_0   
		
	def log_growth(self,z):
		# using equation 3.2 from Baldauf 2015 
		a=1/(1+z)
		print 'what I think it is', a/self.H(z)*self.dH_da(z) + 5/2.*self.Omegam*self.G_norm(0)/self.H(z)**2/a**2/self.G_norm(z)
		return -3/2.*self.Omegam/a**3*self.H0**2/self.H(z)**2 + 1/self.H(z)**2/a**2/self.G_norm(z)
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
			   
		d_c=d_crit/self.G_norm(z)
		
		return d_c
	
	#def average_matter(self, z): 
	
		
   
	# -----------------------------------------------------------------------------
		
	 
if __name__=="__main__": 

	C=CosmoPie()
	z=3.5
	print('Comoving distance',C.D_comov(z))
	print('Angular diameter distance',C.D_A(z))
	print('Luminosity distance', C.D_L(z)) 
	print('Growth factor', C.G(z)) 
	print('Growth factor for really small z',C.G(1e-10))
	z=0.0
	print('logrithmic growth factor', C.log_growth(z))
	print('compare logrithmic growth factor to approxiamtion', C.Omegam**(-.6), C.Omegam)
	print('cirtical overdensity ',C.delta_c(0)  ) 
		
	z=np.linspace(0,20,80) 
	D1=np.zeros(80)
	D2=np.zeros(80)
	for i in range(80):
		D1[i]=C.D_A(z[i])
		D2[i]=C.D_L(z[i])
		
		
	import matplotlib.pyplot as plt
	
	ax=plt.subplot(111)
	ax.set_xlabel(r'$z$', size=20)
	
	ax.plot(z, D1, label='Angular Diameter distance [Mpc]') 
	#ax.plot(z, D2, label='Luminosity distance [Mpc]') 
	
	plt.grid()
	plt.show()
	
	   
	
	