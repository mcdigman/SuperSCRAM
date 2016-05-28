'''
	Halo mass function classes
'''
import numpy as np
from numpy import log, log10, exp
from scipy.interpolate import interp1d

class ST_hmf():
	def __init__(self,z,cosmology,CosmoPie):
		self.z=z    # redshift 
		
		self.min=9
		self.max=16
		n_grid=1000
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
		
		#self.nu_of_M=interp1d(self.M_grid, self.nu_array)
		self.M_of_nu=interp1d(self.nu_array,self.M_grid)
		# characteristic mass
		self.M_star=self.M_of_nu(1.0)
		
		# Sheth-Tormen parameters
		A=0.3222; a=0.707; p=0.3
		
		f=A*np.sqrt(2*a/np.pi)*(1 + 1/(a*self.nu_array)**p) \
		    *np.sqrt(self.nu_array)*exp(-a*self.nu_array/2.)
		
		sigma_inv=self.sigma**(-1)
		
		d_sigma_inv_d_mass=np.diff(sigma_inv)/(np.diff(self.M_grid))
		
		mass_func=rho_bar/self.M_grid[:-1]*d_sigma_inv_d_mass*f[:-1]
		
		self.diff_mass_func=interp1d(self.M_grid[:-1],mass_func)

		
	def dn_dM(self,M):
	    return self.diff_mass_func(M)
	    
	    
	    

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
    
				
    from cosmopie import CosmoPie
    CP=CosmoPie(cosmology,P_lin=P,k=k)	
    print 'Omegam', CP.Omegam_z(1)
    hmf=ST_hmf(0,cosmology,CP)
    
    M=np.logspace(10,14,100)
    N=hmf.dn_dM(M)
    
    import matplotlib.pyplot as plt
    
    ax=plt.subplot(111)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    
    ax.plot(M,N)
    plt.grid()
    plt.show()
    