'''
	Halo mass function classes
'''
import numpy as np
from numpy import log, log10, exp, pi
from scipy.interpolate import interp1d
from scipy.integrate import trapz 
import sys

class ST_hmf():
	def __init__(self,CosmoPie, delta_bar=None):
		
		# log 10 of the minimum and maximum halo mass
		self.min=6
		self.max=18
		# number of grid points in M 
		n_grid=1000
		# I add an additional point because I will take derivatives latter
		# and the finite difference scheme misses that last grid point. 
		dlog_M=(self.max-self.min)/float(n_grid)
		self.M_grid=10**np.linspace(self.min,self.max+dlog_M, n_grid+1)
		self.mass_grid=self.M_grid[0:-1]
		
		
		self.nu_array=np.zeros(n_grid+1)
		self.sigma=np.zeros(n_grid+1)
		#self.Omegam_0=cosmology['Omegam']
		#self.OmegaL=cosmology['OmegaL']
		
		
		#self.delta_v=CosmoPie.delta_v(z)
		self.delta_c=CosmoPie.delta_c(0)
		if delta_bar is not None:
			self.delta_c=self.delta_c - delta_bar 
		self.rho_bar=CosmoPie.rho_bar(0)
		self.Growth=CosmoPie.G_norm
		print 'rhos', 0,self.rho_bar/(1e10), CosmoPie.rho_crit(0)/(1e10)
		
		#print 'hubble', CosmoPie.H(z)
		
		for i in range(self.M_grid.size):
			
			R=3./4.*self.M_grid[i]/self.rho_bar/pi
			R=R**(1./3.)
			self.sigma[i]=CosmoPie.sigma_r(0.,R)
	
		
		# calculated at z=0        								
		self.nu_array=(self.delta_c/self.sigma)**2
		sigma_inv=self.sigma**(-1)
		dsigma_dM=np.diff(log(sigma_inv))/(np.diff(self.M_grid))
		
		
		
		self.sigma_of_M=interp1d(self.M_grid[:-1],self.sigma[:-1])
		self.nu_of_M=interp1d(self.M_grid[:-1], self.nu_array[:-1])
		self.M_of_nu=interp1d(self.nu_array[:-1],self.M_grid[:-1])
		self.dsigma_dM_of_M=interp1d(self.M_grid[:-1],dsigma_dM)
		
	def f_norm(self,z,G):
		A=0.3222; a=0.707; p=0.3
		nu=self.nu_array/G**2
		
		f=A*np.sqrt(2*a/np.pi)*(1 + (1/a/nu)**p)*np.sqrt(nu)*exp(-a*nu/2.)
		sigma_inv=(G**2*self.sigma)**(-1)
		norm=trapz(f,log(sigma_inv))
		
# 		d=np.loadtxt('test_data/hmf_test.dat')
# 		import matplotlib.pyplot as plt
# 		ax=plt.subplot(111)
# 		ax.set_xscale('log')
# 		ax.set_yscale('log')
# 		
# 		ax.plot(self.M_grid,f/norm*(.6774**2))
# 		ax.plot(d[:,0],d[:,2])
# 		ax.plot(d[:,0],d[:,4])
# 		plt.grid()
# 		plt.show()
		
		return norm
	
	def bias_norm(self,z,G):
	   
		A=0.3222; a=0.707; p=0.3
		nu=self.nu_array/G**2
		
		bias=(1 + (a*nu-1)/self.delta_c + 2*p/(self.delta_c*(1+(a*nu)**p)))
		f=A*np.sqrt(2*a/np.pi)*(1 + (1/a/nu)**p)*np.sqrt(nu)*exp(-a*nu/2.)
		
		norm=trapz(f*bias,nu)
		return norm
		
# 		d=np.loadtxt('test_data/hmf_test.dat')
# 		import matplotlib.pyplot as plt
# 		ax=plt.subplot(111)
# 		ax.set_xscale('log')
# 		ax.set_yscale('log')
# 		
# 		ax.plot(self.M_grid,f/norm*(.6774**2))
# 		ax.plot(d[:,0],d[:,2])
# 		ax.plot(d[:,0],d[:,4])
# 		plt.grid()
# 		plt.show()
		
		return norm

	def f_sigma(self,M,z):
		A=0.3222; a=0.707; p=0.3
		G=self.Growth(z)
		norm=self.f_norm(z,G)
		nu=self.nu_of_M(M)/G**2
		f=A*np.sqrt(2*a/np.pi)*(1 + (1/a/nu)**p)*np.sqrt(nu)*exp(-a*nu/2.)
# 		
# 		print 'hello G', G
# 		d=np.loadtxt('test_data/hmf_test.dat')
# 		import matplotlib.pyplot as plt
# 		ax=plt.subplot(111)
# 		ax.set_xscale('log')
# 		ax.set_yscale('log')
		
# 		ax.plot(M,f*(.6774**2))
# 		ax.plot(d[:,0],d[:,2])
# 		ax.plot(d[:,0],d[:,4])
# 		plt.show()
		
		return f/norm

	def mass_func(self,M,z):
		f=self.f_sigma(M,z)
		sigma=self.sigma_of_M(M)
		dsigma_dM=self.dsigma_dM_of_M(M)
		mf=self.rho_bar/M*dsigma_dM*f
		# return sigma,f,and the mass function dn/dM
		return sigma,f,mf 
	
	def dndM(self,M,z):
		_,_,mf=self.mass_func(M,z)
		return mf 

	def M_star(self):
		return self.M_of_nu(1.0)
		
	def bias(self,M,z):
		A=0.3222; a=0.707; p=0.3
		G=self.Growth(z)
		nu=self.nu_of_M(M)/G**2
		norm=self.bias_norm(z,G)
		bias=(1 + (a*nu-1)/self.delta_c + 2*p/(self.delta_c*(1+(a*nu)**p)))
		return bias/norm
	
	def bias_avg(self,min_mass,z):
	    mass=self.mass_grid[ self.mass_grid >= min_mass]
	    b_array=np.zeros_like(mass)
	    for i in range(mass.size):
	        b_array[i]=self.bias(mass[i],z)
	    
	    mf=self.dnDm(mass,z)	    
	    
	    return trapz(b_array*mf,mass) 
	    
	    
		
	def n_avg(self,M,z):
		mass=self.mass_grid[ self.mass_grid >= M]
		mf=self.dndM(mass,z)
		return trapz(mf,mass)
		
	
if __name__=="__main__":

	cosmology={'Omegabh2' :0.02230,
				'Omegach2' :0.1188,
				'Omegamh2' : 0.14170,
				'OmegaL'   : .6911,
				'Omegam'   : .3089,
				'H0'       : 67.74, 
				'sigma8'   : .8159, 
				'h'        :.6774, 
				'Omegak'   : 0.0, # check on this value 
				'Omegar'   : 0.0}

	d=np.loadtxt('Pk_Planck15.dat')
	k=d[:,0]; P=d[:,1]
	
	d=np.loadtxt('test_data/hmf_z0_test1.dat')
	d=np.loadtxt('test_data/hmf_test.dat')
				
	from cosmopie import CosmoPie
	CP=CosmoPie(cosmology,P_lin=P,k=k)	
	
	#z=1.1; h=.6774
	#rho_bar=CP.rho_bar(z)
	hmf=ST_hmf(CP)
	
	
	z1=0;z2=1.1;z3=.1
	#M=np.logspace(8,15,150)
	M=d[:,0]
	_,f1,mf1=hmf.mass_func(M,z1)
	_,f2,mf2=hmf.mass_func(M,z2)
	_,f2,mf3=hmf.mass_func(M,z3)
	
	
	h=.6774
	import matplotlib.pyplot as plt
	
	ax=plt.subplot(131)
	ax.set_xscale('log')
	ax.set_yscale('log')
	#ax.set_xlim(9.6,15)
	#ax.set_ylim(-5.4,0)
	ax.set_ylabel(r'$\frac{d n}{d \log_{10}(M_h)}$', size=20)
	ax.plot(M,mf1*M)
	ax.plot(d[:,0], d[:,1]/h, '--')
	
	ax.plot(M,mf2*M)
	ax.plot(d[:,0], d[:,3]/h**2, '--')
	
	zz=np.array([.1,.2,.3,.5,1,2])
	for i in range(zz.size):
		_,_,y=hmf.mass_func(M,zz[i])
		ax.plot(M,y*M)
		
	
	ax.plot(M,mf3*M)
	#plt.axvline(M_star)
	
	
	plt.grid()
	
	ax=plt.subplot(132)
	ax.set_xscale('log')
	#ax.set_xlim(-.2,.4)
	ax.set_yscale('log')
	ax.set_ylabel(r'$f(M_h)}$', size=20)
		
	
	ax.plot(M,f1)
	ax.plot(M,d[:,2]/h, color='red')
	
	ax.plot(M,f2)
	ax.plot(M,d[:,4]/h, color='red')
	
	plt.grid()
	
	ax=plt.subplot(133)
	ax.set_xscale('log')
	#ax.set_xlim(-.2,.4)
	ax.set_yscale('log')
	ax.set_ylabel(r'$f(M_h)}$', size=20)
	
	b1=hmf.bias(M,0)
	b2=hmf.bias(M,1)
		
	ax.plot(M,b1)
	ax.plot(M,b2)
	
	plt.grid()
	plt.show()
	
