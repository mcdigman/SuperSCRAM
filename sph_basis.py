import numpy as np
from numpy import sin,pi 
from scipy.special import sph_harm as Y_lm
from sph_functions import Y_r, j_n
from scipy.integrate import dblquad


eps=np.finfo(float).eps	
def a_lm(theta,phi,l,m):

	# returns \int d theta d phi \sin(theta) Y_lm(theta, phi)
	# theta is an array of 2 numbers representing the max and min of theta
	# phi is an array of 2 numbers representing the max and min of phi
	# l and m are the indices for the spherical harmonics 
	
	theta_min, theta_max=theta
	phi_min, phi_max=phi
	
	def integrand(theta,phi):
		# use the real spherical harmonics from sph_functions
		return sin(theta)*Y_r(l,m,theta,phi)
	
	I= dblquad(integrand,phi_min, phi_max, lambda theta: theta_min, lambda theta :theta_max)[0]

	if (np.absolute(I) <= eps):
		I=0.0 
  
	return I

	
def R_int(k,r_range,n):
	# returns \int R_n(rk_alpha) r2 dr
	# I am using the spherical Bessel function for R_n, but that might change  
	r_min,r_max=r_range 
	
	def integrand(r):
		return r**2*j_n(n,r*k)
	
	I= quad(integrand,r_min,r_max)[0]
	
	if (np.absolute(I) <= eps):
		I=0.0 
  
	return I
	
def delta_alpha_bar(mat,coords):
	
	# send in matrix that has 
	# delta_alpha,n_alpha l_alpha, m_alpha
	delta_alpha,n_alpha,l_alpha,m_alpha=mat
	n_alpha=delta_alpha.size
	
	r,theta,phi=coords
	r_min,r_max=r 
	
	a_00=a_lm(theta,phi,0,0)
	Omega=a_00*np.sqrt(4*pi) 
	
	delta_bar=0
	for i in range(n_alpha):
	    delta_bar=delta_bar+R_int(r,n_alpha[i])*a_lm(theta,phi,l_alpha[i],m_alpha[i])
	
	delta_bar=3/(r_max**3 - r_min**3)/Omega 

	return delta_bar 
	
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
	
	
	
   