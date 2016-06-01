'''
	Halo mass function classes
'''
import numpy as np
from numpy import log, log10, exp
from scipy.interpolate import interp1d
from scipy.integrate import trapz 
import sys

class ST_hmf():
	def __init__(self,z,cosmology,CosmoPie):
		self.z=z    # redshift 
		
		# log 10 of the minimum and maximum halo mass
		self.min=6
		self.max=16
		# number of grid points in M 
		n_grid=1000
        # I add an additional point because I will take derivatives latter
        # and the finite difference scheme misses that last grid point. 
		dlog_M=(self.max-self.min)/float(n_grid)
		self.M_grid=10**np.linspace(self.min,self.max+dlog_M, n_grid+1)
		
		
		self.nu_array=np.zeros(n_grid+1)
		self.sigma=np.zeros(n_grid+1)
		self.Omegam_0=cosmology['Omegam']
		self.OmegaL=cosmology['OmegaL']
		
		
		self.delta_v=CosmoPie.delta_v(z)
		self.delta_c=CosmoPie.delta_c(z)
		rho_bar=CosmoPie.rho_bar(z)
		
		for i in range(self.M_grid.size):
			self.sigma[i]=CosmoPie.sigma_m(self.M_grid[i],z)
			
						
		
		self.nu_array=(self.delta_c/self.sigma)**2
		
		print 'delta_c', self.delta_c

		
		#self.nu_of_M=interp1d(self.M_grid, self.nu_array)
		self.M_of_nu=interp1d(self.nu_array,self.M_grid)
		
		# Sheth-Tormen parameters
		A=0.3222; a=0.707; p=0.3
		
		f=A*np.sqrt(2*a/np.pi)*(1 + 1/(a*self.nu_array)**p) \
			*np.sqrt(self.nu_array)*exp(-a*self.nu_array/2.)
		
		sigma_inv=self.sigma**(-1)
		
		# need to normalize so that \int f dlog(sigma^-1) =1
		# Since f corresponds to the fraction of collapsed objects in per unit 
		# interval in dlog(sigma^1) and the halo model assumes all mass is in collapsed 
		# objects, then the integral must be one. 
		norm=trapz(f,log(sigma_inv))
		f=f/norm
		self.norm=norm
        # the derivative dlog(sigma^-1)/dM
		dsigma_dM=np.diff(log(sigma_inv))/(np.diff(self.M_grid))
		
		MassFunc=rho_bar/self.M_grid[:-1]*dsigma_dM*f[:-1]
		
		# functions for functions below 
		self.diff_mass_func=interp1d(self.M_grid[:-1],MassFunc)
		self.mass_func=interp1d(self.M_grid[:-1],f[:-1])
		self.sigma_inv=interp1d(self.M_grid[:-1],self.sigma[:-1])

		
	def dn_dM(self,M):
		return self.diff_mass_func(M)
	
	def f_of_M(self,M):
		return self.mass_func(M)
	
	def f_of_sigma_inv(self,M):
	    f=self.mass_func(M)
	    return self.sigma_inv(M), f
	
	def M_star(self):
	    return self.M_of_nu(1.0)
		
		
		

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
	
	z=0; h=.6774
	rho_bar=CP.rho_bar(z)
	hmf=ST_hmf(z,cosmology,CP)
	
	#M=np.logspace(8,15,150)
	M=d[:,0]
	dndM=hmf.dn_dM(M)
	M_star=hmf.M_star()
	a,b=hmf.f_of_sigma_inv(M)
	
	z=5
	hmf=ST_hmf(z,cosmology,CP)
	
	dndM2=hmf.dn_dM(M)	
	c,e=hmf.f_of_sigma_inv(M)
	
	#sys.exit()
	import matplotlib.pyplot as plt
	
	ax=plt.subplot(121)
	ax.set_xscale('log')
	ax.set_yscale('log')
	#ax.set_xlim(9.6,15)
	#ax.set_ylim(-5.4,0)
	ax.set_ylabel(r'$\frac{d n}{d \log_{10}(M_h)}$', size=20)
	ax.plot(M,dndM*M)
	ax.plot(d[:,0], d[:,1]/h, color='red')
	
	ax.plot(M,dndM2*M)
	ax.plot(d[:,0], d[:,3]/h, color='red')
	plt.axvline(M_star)
	
	
	plt.grid()
	
	ax=plt.subplot(122)
	ax.set_xscale('log')
	#ax.set_xlim(-.2,.4)
	ax.set_yscale('log')
	ax.set_ylabel(r'$f(M_h)}$', size=20)
		
	ax.plot(M,b)
	ax.plot(M,d[:,2]/h, color='red')
	
	ax.plot(M,e)
	ax.plot(M,d[:,4]/h, color='red')
	
	plt.grid()
	plt.show()
	