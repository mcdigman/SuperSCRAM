''' 
	Code to calculate spherical functions, such as:
	spherical Bessel functions
	the zeros of the Bessel functions
	the real spherical harmonics 
	the derivatives of the Bessel function
	J. E. McEwen (c) 2016
'''

import numpy as np
from numpy import sin, cos, exp, pi, sqrt
from scipy.misc import factorial2
from scipy.optimize import newton
from scipy.special import sph_harm as Y_lm, jv, sph_jn
import sys
from the_mad_house import e_message

eps =np.finfo(float).eps
z_cut=1e-2

data=np.loadtxt('spherical_bessel_zeros_long.dat')

def sph_Bessel_down(n,z):
	n_start=n + int(np.sqrt(n*40)/2.) 
	
	j2=0
	j1=1 
	
	for i in range(n_start,-1,-1):
	# for loop until n=0, so that you can renormalize 
		j0=(2*i+3)/z*j1 - j2 
		j2=j1
		j1=j0
		if (i==n):
			result=j0
		
		j0_true=sin(z)/z 
	# renormalize and return result 
	return result*j0_true/j0 
		
		
#print 'this is eps', eps 
#print 'check eps', 1+eps
def sph_Bessel_array(n,z):
	z=np.asarray(z, dtype=float)
	result=np.zeros(z.size)
	
	# id for limiting cases 
	id1=np.where( z <= z_cut)[0]
	id2=np.where( z > z_cut)[0] 
	
	if (id1.size !=0):
		result[id1]= z[id1]**n/factorial2(2*n+1, exact=True)
	if n==0:
		result[id2]=sin(z[id2])/z[id2]
	if n==1: 
		result[id2]=sin(z[id2])/z[id2]**2 - cos(z[id2])/z[id2]
	if n==2:
		result[id2]=(3/z[id2]**3 -1/z[id2])*sin(z[id2]) -3/z[id2]**2*cos(z[id2]) 
	if n >2:  
	
		if n > np.real(z):
			return sph_Bessel_down(n,z)

		else: 
			j0=sin(z[id2])/z[id2]; j1=sin(z[id2])/z[id2]**2 - cos(z[id2])/z[id2] 
   
			j=np.zeros(id2.size)
			for i in range(1,n):
				j=(2*i+1)/z[id2]*j1-j0
				j0=j1
				j1=j
		
			result[id2]=j
		 
		return result 
	
def sph_Bessel(n,z):
  
	# limiting case for z near zero 
	if ( z <= z_cut):
		return z**n/factorial2(2*n+1, exact=True)
	if n==0:
		return sin(z)/z
	if n==1: 
		return sin(z)/z**2 - cos(z)/z
	if n==2:
		return (3/z**3 -1/z)*sin(z) -3/z**2*cos(z) 
	if n >2: 
	
		if n > np.real(z):
			return sph_Bessel_down(n,z)
		
		else :
		# do upward recursion 
			j0=sin(z)/z; j1=sin(z)/z**2 - cos(z)/z
   
			for i in range(1,n):
				j=(2*i+1)/z*j1-j0
				j0=j1
				j1=j
			return j
	
def j_n(n,z):
		
	z=np.asarray(z)
	
	if(z.size==1):
		return sqrt(pi/(2*z))*jv(n+0.5,z)
	else:  
		result=np.zeros_like(z)
		for i in range(z.size):
			result[i]=sqrt(pi/(2*z[i]))*jv(n+0.5,z[i])
		return result
		
def dj_n(n,z):

	z=np.asarray(z)
	
	if(z.size==1):
		a=j_n(n,z)
		b=j_n(n+1,z)
		c=j_n(n-1,z)
		return (c-b)/2. -a/2./z
	
	else:  
		a=j_n(n,z)
		b=j_n(n+1,z)
		c=j_n(n-1,z)
		return (c-b)/2. -a/2./z
		
#return all zeros in a row smaller than a cut 		
def jn_zeros_cut(l,q_lim):
    return data[l,data[l]<q_lim]

def jn_zeros(l,n_zeros): 
# 	print 'this is n', n 
# 	# fixed minimum and maximum range for zero finding 
# 	min=3; max=1e5
# 	
# 	# tolerance
# 	tol=1e-15
# 	zeros=np.zeros(n_zeros)
# 	
# 	def func(z):
# 	    if z < pi:
# 	        return np.infty
# 		return j_n(n,z)
# 		
# 	def J_func(z):
# 		return dj_n(n,z)
# 		
# 	for i in range(n_zeros):
# 		guess=1 + np.sqrt(2) + i*pi + n + n**(0.4)
# 		
# 		zeros[i]=newton(func, guess,tol=tol)	
    
	return data[l,:n_zeros]

def dJ_n(n,z): 
	# check this returns correct value for derivative of Big J_n(z)   
	return (n/z)*jv(n,z)-jv(n+1,z)
	
def Y_r(l,m,theta,phi):
	# real spherical harmonics, using scipy spherical harmonic functions
	# the scipy spherical harmonic inputs are ordered as 
	# Y_lm(m,l,phi,theta)
	if (np.abs(m) > l):
		print e_message()
		print('You must have |m| <=l for spherical harmonics.')
		raise ValueError('Please check the function values you sent into Y_r in module sph_functions.py')
		#sys.exit()

	if m==0.0:
		return np.real(Y_lm(m,l,phi,theta))
		#if np.real(result) < eps:
		#    result =0
		#return result
		#print 'check', result 
		#if result < eps:
		#    result=0
		#return result
	if m<0:
		result =1j/np.sqrt(2)*(Y_lm(m,l,phi,theta) - (-1)**m*Y_lm(-m,l,phi,theta))
		if np.absolute(np.real(result)) < eps:
			result=0
		#return np.sqrt(2)*(-1)**m*np.imag(Y_lm(m,l,phi,theta))
		return np.real(result)
	if m>0: 
		result =1/np.sqrt(2)*(Y_lm(-m,l,phi,theta)  + (-1)**m*Y_lm(m,l,phi,theta))
		#return np.sqrt(2)*(-1)**m*np.real(Y_lm(m,l,phi,theta))
		if np.absolute(np.real(result)) < eps:
			result=0
		return np.real(result)
		

#Daniel added - radial part of tidal force		
def S_rr(l,m,theta,phi,k,r):
	return (Y_r(l,m,theta,phi)/4.) * (2*jn(l,k*r) - jn(l+2,k*r) - jn(l-2,k*r))
	
	
if __name__=="__main__": 
	
	print('check spherical Bessel against mathematica output')
	print('function values',j_n(0,2.))
	print('mathematica value', 0.454649)
	print('function values',j_n(1,2.))
	print('mathematica value', 0.435398)
	print('function values',j_n(2,2.))
	print('mathematica value', 0.198448)
	print('function values',j_n(3,2.))
	print('mathematica value', 0.0607221)
	print('function values',j_n(10,2.))
	print('mathematica value', 6.8253e-8)
	print('function values',j_n(50,101.5))
	print('mathematica value', -0.0100186)
	
	print('check derivative of Bessel against keisan.casio.com')
	print('function values', dJ_n(0,1))
	print('true value', -0.4400505857449335159597)
	print('function values', dJ_n(3,11.5))
	print('true value', -0.0341759332779211515933)
	print('function values', dJ_n(5,3.145))
	print('true value', 0.0686374928139798052691)

	y=lambda phi : sin(phi)
	x=lambda phi : cos(phi) 
	z=lambda theta : cos(theta)
	theta=3*np.pi/2; phi=3*np.pi/4
	
	# check the values for the real spherical harmonics
	# checking against values on https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#l_.3D_0.5B2.5D.5B3.5D
	# check against wiki values
	print 'check l=0 case'
	print 'function value', Y_r(0,0,theta,phi)
	print 'wiki value', .5*np.sqrt(1/pi)
	print '------------------------------------------'
	# l=1 case 
	print 'check l=1, m=-1 case'
	print 'function value', Y_r(1,-1,theta,phi)
	print 'wiki value', np.sqrt(3/4./pi)*y(phi)
	print '------------------------------------------'
	print 'check l=1, m=0 case'
	print 'function value', Y_r(1,0,theta,phi)
	print 'wiki value', np.sqrt(3/4./pi)*z(theta)
	print '------------------------------------------'
	print 'check l=1, m=1 case'
	print 'function value', Y_r(1,1,theta,phi)
	print 'wiki value', np.sqrt(3/4./pi)*x(phi)
	print '------------------------------------------'
	print 'check l=2, m=-2 case'
	print 'function value', Y_r(2,-2,theta,phi)
	print 'wiki value', .5*np.sqrt(15/pi)*x(phi)*y(phi)
	print 'check l=2, m=-1 case'
	print 'function value', Y_r(2,-1,theta,phi)
	print 'wiki value', .5*np.sqrt(15/pi)*z(theta)*y(phi)
	print 'check l=2, m=0 case'
	print 'function value', Y_r(2,0,theta,phi)
	print 'wiki value', .25*np.sqrt(5/pi)*(2*z(theta)**2 - y(phi)**2 - x(phi)**2)
	print 'check l=2, m=1 case'
	print 'function value', Y_r(2,1,theta,phi)
	print 'wiki value', .5*np.sqrt(15/pi)*z(theta)*x(phi)
	print 'check l=2, m=2 case'
	print 'function value', Y_r(2,2,theta,phi)
	print 'wiki value', .25*np.sqrt(15/pi)*(x(phi)**2-y(phi)**2)
	print '------------------------------------------'
	#print 'check an incorrect m and l value'
	#print Y_r(2,3,theta,phi)
	print '------------------------------------------'
	print 'check normalization'
	from scipy.integrate import nquad
	def norm_check(m1,l1,m2,l2):
		def func(theta,phi):
			return sin(theta)*Y_r(l1,m1,theta,phi)*Y_r(l2,m2,theta,phi)
		
		def funcII(theta,phi):
			return sin(theta)*Y_lm(m1,l1,phi,theta)*np.conjugate(Y_lm(m2,l2,phi,theta))
		 
		  
		I=nquad(func,[[0,pi],[0,2*pi]])[0]
		#print 'check against spherical harmonics',nquad(funcII,[[0,pi],[0,2*pi]])[0]
		if I < eps:
			I=0
		return I 
		
	print 'check normalization, 1,1,1,1:', norm_check(1,1,1,1)
	print 'check normalization, 0,1,0,1:', norm_check(0,1,0,1)
	print 'check normalization, 0,2,0,2:', norm_check(0,2,0,2)
	print 'check normalization, 1,2,1,2:', norm_check(1,2,1,2)
	print 'check normalization, 2,2,2,2:', norm_check(2,2,2,2)
	print 'check normalization, -1,2,-1,2:', norm_check(-1,2,-1,2)
	print 'check normalization, -1,2,0,2:', norm_check(-1,2,0,2)
	print 'check normalization, -1,2,-1,3:', norm_check(-1,2,-1,3)
	
	import matplotlib.pyplot as plt
	
	z=np.linspace(0,10,200)
	
	
	ax=plt.subplot(111)
	ax.set_xlim(0,10)
	ax.set_ylim(-1,1)
	ax.set_ylabel(r'$j_n(z)$', size=30)
	ax.set_xlabel(r'$z$', size=30)
	
	ax.plot(z, j_n(0,z))
	ax.plot(z, j_n(1,z))
	ax.plot(z, j_n(2,z))
	ax.plot(z, j_n(3,z))
	ax.plot(z, j_n(4,z))
	
	x=jn_zeros(0,3)
	print 'this is x', x 
	ax.plot(x,np.zeros(x.size),'o')
	x=jn_zeros(1,3)
	ax.plot(x,np.zeros(x.size),'o')
	x=jn_zeros(2,3)
	ax.plot(x,np.zeros(x.size),'o')
	x=jn_zeros(3,3)
	ax.plot(x,np.zeros(x.size),'o')
	x=jn_zeros(4,3)
	ax.plot(x,np.zeros(x.size),'o')
	x=jn_zeros(9,3)
	ax.plot(x,np.zeros(x.size),'o', label='l=9')
	ax.plot(z, j_n(10,z))
	
	
	plt.legend(loc=2)
	plt.grid()
	plt.show()

	
