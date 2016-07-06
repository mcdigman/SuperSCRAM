import numpy as np
from numpy import pi, sqrt, sin, cos
from sph_functions import j_n, jn_zeros, dJ_n, Y_r
from scipy.special import jv
from scipy.integrate import trapz, quad,dblquad
import sys 


# the smallest value 
eps=np.finfo(float).eps	

def I_alpha(k_alpha,k,r_max,l_alpha):
			# return the integral \int_0^r_max dr r^2 j_laplpha(k_alphar)j_lalpha(kr)
			
			a=k_alpha*r_max; b=k*r_max 
			l=l_alpha+.5
			return pi/2./sqrt(k_alpha*k)/(k_alpha*82 - k**2)*r_max*(k*jv(l-1,b)*jv(l,a)-k_alpha*jv(l-1,a)*jv(l,b))
			#return pi/2.*r_max**2/sqrt(k_alpha*k)*(a*jv(l,b)*dJ_n(l,a)-b*jv(l,a)*dJ_n(l,b))/(b**2-a**2)
			#return pi/2./sqrt(k_alpha*k)*r_max*(k_alpha*jv(l,b)*dJ_n(l,a)-k*jv(l,a)*dJ_n(l,b))/(k**2-k_alpha**2)
			


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
		
		k,P_lin=CosmoPie.get_P_lin()
		
		self.l=l_alpha
		
		# define the super mode wave vector k alpha
		# and also make the map from l_alpha to k_alpha 
		
		# this needs to be made faster 
		self.k_alpha=np.zeros(l_alpha.size*n_zeros) 
		
		
		self.map_l_to_alpha=np.zeros(self.k_alpha.size)      
		for i in range(l_alpha.size):
			j=i*n_zeros
			#print 'this is i , and j', i, j 
			#print 'this is l_alpha', l_alpha[i]
			self.k_alpha[j:j+n_zeros]=jn_zeros(l_alpha[i],n_zeros)/r_max
			self.map_l_to_alpha[j:j+n_zeros]=l_alpha[i]
			
	   # define the covariance matrix C_{\alpha \beta} and fill with values 
		self.C_alpha_beta=np.zeros((self.k_alpha.size,self.k_alpha.size))
		for alpha in range(self.k_alpha.size):
			for beta in range(self.k_alpha.size):
				A=I_alpha(self.k_alpha[alpha],k,r_max,self.map_l_to_alpha[alpha])
				B=I_alpha(self.k_alpha[beta],k,r_max,self.map_l_to_alpha[beta])
				self.C_alpha_beta[alpha,beta]=8*trapz(k**2*P_lin*A*B,k)
		
		# self.z_bins=geometry[0]
# 		self.r_bins=np.zeros_like(self.z_bins)
# 		self.Theta=geometry[1]
# 		self.Phi=geometry[2]
		
	def Cov_alpha_beta(self):
		return self.map_l_to_alpha, self.C_alpha_beta
	
	def k_LW(self):
		# return the long-wavelength wave vector k 
		return self.k_alpha
	
	def a_lm(self,Theta,Phi,l,m):    
		# returns \int d theta d phi \sin(theta) Y_lm(theta, phi)
		# theta is an array of 2 numbers representing the max and min of theta
		# phi is an array of 2 numbers representing the max and min of phi
		# l and m are the indices for the spherical harmonics 
		
		theta_min, theta_max=Theta
		phi_min, phi_max=Phi
		
		def integrand(theta,phi):
			#print theta, phi 
			return sin(theta)*Y_r(l,m,theta,phi)
	
		I = dblquad(integrand,theta_min,theta_max, lambda phi: phi_min, lambda phi :phi_max)[0]
				
		if (np.absolute(I) <= eps):
			I=0.0 
		return I
	
	def R_int(self,r_range,k,n):
	# returns \int R_n(rk_alpha) r2 dr
	# I am using the spherical Bessel function for R_n, but that might change  
		r_min,r_max=r_range 
		

		def integrand(r):
			return r**2*j_n(n,r*k)

		I= quad(integrand,r_min,r_max)[0]
		
		if (np.absolute(I) <= eps):
			I=0.0
			
		return I 
	
	def D_delta_bar_D_delta_alpha(self,r_min,r_max,Theta,Phi):
	
		r=np.array([r_min,r_max])
	
		a_00=self.a_lm(Theta,Phi,0,0)
		#aa=np.array([0,2*pi]);bb=np.array([0,pi])
		#print 'a_00 check', self.a_lm(aa,bb,0,0), 2*sqrt(pi)
		#print 'sph check', Y_r(0,0,0,0), .5*1/sqrt(pi)
		
		Omega=a_00*np.sqrt(4*pi) 
		norm=3/(r_max**3 - r_min**3)/Omega 
		print 'this is Omega', Omega, norm, 3/(r_max**3 - r_min**3)
	
		result=np.zeros((self.l.size,3),dtype=object)
		# result will hold l_alpha,{m_\alpha}, {\partial \bar{\delta} /\partial \delta_alpha}
		for i in range(self.l.size):
			m=np.asarray(np.arange(-self.l[i],self.l[i]+1))
			#print m
			#print self.l[i]
			result[i,0]=self.l[i]
			result[i,1]=m
			
			hold=np.zeros(m.size)
		
			for j in range(m.size):
			    hold[j]=self.R_int(r,self.k_alpha[i],self.l[i])*self.a_lm(Theta,Phi,self.l[i],m[j])*norm
			result[i,2]=hold
		
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
		print 'l',X[i,0]
		print 'm',X[i,1]
		print 'deriv',X[i,2]
	
	a,b=R.Cov_alpha_beta()
	print b 
	
	
		
