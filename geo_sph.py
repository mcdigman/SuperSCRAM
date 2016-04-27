import numpy as np
from numpy import sin,pi 
from scipy.special import sph_harm as Y_lm, jv
from scipy.integrate import nquad, dblquad, quad

# make sure you understand this 
eps=np.finfo(float).eps
	
def a_lm(theta,phi,l,m):

	# returns \int d theta d phi \sin(theta) Y_lm(theta, phi)
	# theta is an array of 2 numbers representing the max and min of theta
	# phi is an array of 2 numbers representing the max and min of phi
	# l and m are the indices for the spherical harmonics 
	
	theta_min, theta_max=theta
	phi_min, phi_max=phi
	
	def integrand(theta,phi):
		# the scipy spherical harmonic inputs are ordered as 
		# Y_lm(m,l,phi,theta)
		return sin(theta)*Y_lm(m,l,phi,theta)
	
	def real_I(theta,phi):
		return np.real(integrand(theta,phi))
	
	def imag_I(theta,phi):
		return np.imag(integrand(theta,phi)) 
	
	R= dblquad(real_I,phi_min, phi_max, lambda theta: theta_min, lambda theta :theta_max)[0]
	I= dblquad(imag_I,phi_min, phi_max, lambda theta: theta_min, lambda theta :theta_max)[0]
	
	if (np.absolute(R) <= eps):
		R=0.0
	if (np.absolute(I) <= eps):
		I=0.0 
  
	return R + 1j*I

def j(n,z):
	return jv(n + .5,z)*np.sqrt(pi/2./z)
	
def R_int(r,k,n):
	# returns \int R_n(r) r2 dr 
	r_min,r_max=r 
	
	def real_I(r):
		return np.real(r**2*j(n,r*k))
	
	def imag_I(r):
		return np.imag(r**2*j(n,r*k))
	
	
	R= quad(real_I,r_min,r_max)[0]
	I= quad(imag_I,r_min,r_max)[0]
	
	if (np.absolute(R) <= eps):
		R=0.0
	if (np.absolute(I) <= eps):
		I=0.0 
  
	return R + 1j*I 
	
def delta_alpha_avg(delta_alpha,coords):
	n_alpha=delta_alpha.size
	r,theta,phi=coords
	r_min,r_max=r 
	# where does the l and m come from ? 
	return 1 
	
if __name__=="__main__":
	
	
	# test against some solutions from mathematica 
	l=1
	m=0
	theta=[0,pi/2.]
	phi=[0,pi]
	
	# second value is from mathematica 
	print a_lm(theta,phi,l,m), .767495
	
	l=2
	m=1
	theta=[0,pi/2.]
	phi=[0,pi]
	
	# second value is from mathematica 
	print a_lm(theta,phi,l,m), -.515032*1j
	
	l=3
	m=-2
	theta=[0,pi/2.]
	phi=[0,pi]
	
	# second value is from mathematica 
	print a_lm(theta,phi,l,m), 0
	
	l=3
	m=0
	theta=[0,pi/2.]
	phi=[0,pi]
	
	# second value is from mathematica 
	print a_lm(theta,phi,l,m), -0.293092
	
	print j(0,10)
	print j(3,50)
	
	
   