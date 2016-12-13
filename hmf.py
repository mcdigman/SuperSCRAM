'''
	Halo mass function classes
'''
import numpy as np
from numpy import log, log10, exp, pi
from scipy.interpolate import interp1d,RectBivariateSpline
from scipy.integrate import trapz,cumtrapz 
import sys
import defaults

class ST_hmf():
	def __init__(self,CosmoPie, delta_bar=None,params=defaults.hmf_params):
		self.params = params
		# log 10 of the minimum and maximum halo mass
		self.min=6
		self.max=18
		# number of grid points in M 
		n_grid=self.params['n_grid']
		# I add an additional point because I will take derivatives latter
		# and the finite difference scheme misses that last grid point. 
		dlog_M=(self.max-self.min)/float(n_grid)
		self.M_grid=10**np.linspace(self.min,self.max+dlog_M, n_grid+1)
		self.mass_grid=self.M_grid[0:-1]
		
		
		self.nu_array=np.zeros(n_grid+1)
		self.sigma=np.zeros(n_grid+1)
		
		
		self.delta_c=CosmoPie.delta_c(0)
		if delta_bar is not None:
			self.delta_c=self.delta_c - delta_bar 
		self.rho_bar=CosmoPie.rho_bar(0)

		self.Growth=CosmoPie.G_norm
                

		print 'rhos', 0,self.rho_bar/(1e10), CosmoPie.rho_crit(0)/(1e10)
		
		
		for i in range(self.M_grid.size):
			
			R=3./4.*self.M_grid[i]/self.rho_bar/pi
			R=R**(1./3.)
                        #TODO evaluate if faster way to get sigma_r
			self.sigma[i]=CosmoPie.sigma_r(0.,R)
	
		
		# calculated at z=0        								
		self.nu_array=(self.delta_c/self.sigma)**2
		sigma_inv=self.sigma**(-1)
		dsigma_dM=np.diff(log(sigma_inv))/(np.diff(self.M_grid))
				
		self.sigma_of_M=interp1d(self.M_grid[:-1],self.sigma[:-1])
		self.nu_of_M=interp1d(self.M_grid[:-1], self.nu_array[:-1])
		self.M_of_nu=interp1d(self.nu_array[:-1],self.M_grid[:-1])
		self.dsigma_dM_of_M=interp1d(self.M_grid[:-1],dsigma_dM)

                #grid for precomputing z dependent quantities
                self.z_grid = np.arange(self.params['z_min'],self.params['z_max'],self.params['z_resolution'])
                self.G_grid = CosmoPie.G_norm(self.z_grid)
                
                self.b_norm_grid = self.bias_norm(self.G_grid)
                self.b_norm_cache = interp1d(self.z_grid,self.b_norm_grid)
                

		
	def f_norm(self,G):
		A=0.3222; a=0.707; p=0.3
		nu=self.nu_array/G**2
		f=A*np.sqrt(2*a/np.pi)*(1 + (1/a/nu)**p)*np.sqrt(nu)*exp(-a*nu/2.)
		sigma_inv=1./self.sigma*1./G**2
		norm=trapz(f,log(sigma_inv))
		return norm

	def f_norm_array(self,G):
		A=0.3222; a=0.707; p=0.3
		nu=np.outer(self.nu_array,1./G**2)	
		f=A*np.sqrt(2*a/np.pi)*(1 + (1/a/nu)**p)*np.sqrt(nu)*exp(-a*nu/2.)
		sigma_inv=np.outer(1./self.sigma,1./G**2)
		norm=trapz(f,log(sigma_inv),axis=0)
		return norm
        #Now supports vector arguments	
        #TODO eliminate redundant z argument
	def bias_norm(self,G):
	   
		A=0.3222; a=0.707; p=0.3
		nu=np.outer(self.nu_array,1./G**2)
	
		bias=(1 + (a*nu-1)/self.delta_c + 2*p/(self.delta_c*(1+(a*nu)**p)))
		f=A*np.sqrt(2*a/np.pi)*(1 + (1/a/nu)**p)*np.sqrt(nu)*exp(-a*nu/2.)
		
		norm=trapz(f*bias,nu,axis=0)
		return norm
		
	def f_sigma(self,M,G):
		A=0.3222; a=0.707; p=0.3
		norm=self.f_norm(G)
		nu=self.nu_of_M(M)/G**2
		f=A*np.sqrt(2*a/np.pi)*(1 + (1/a/nu)**p)*np.sqrt(nu)*exp(-a*nu/2.)
		return f/norm

	def f_sigma_array(self,M,G):
		A=0.3222; a=0.707; p=0.3
		norm=self.f_norm_array(G)
		nu=np.outer(self.nu_of_M(M),1./G**2)
		f=A*np.sqrt(2*a/np.pi)*(1 + (1/a/nu)**p)*np.sqrt(nu)*exp(-a*nu/2.)
		return f/norm


	def mass_func(self,M,G):
		f=self.f_sigma(M,G)
		sigma=self.sigma_of_M(M)
		dsigma_dM=self.dsigma_dM_of_M(M)
		mf=self.rho_bar/M*dsigma_dM*f
		# return sigma,f,and the mass function dn/dM
		return sigma,f,mf 

	def mass_func_array(self,M,G):
		f=self.f_sigma_array(M,G)
		sigma=self.sigma_of_M(M)
		dsigma_dM=self.dsigma_dM_of_M(M)
		mf=(self.rho_bar/M*dsigma_dM*f.T).T
		# return sigma,f,and the mass function dn/dM
		return sigma,f,mf 
        
	def dndM_G_array(self,M,G):
	    _,_,mf=self.mass_func_array(M,G)
	    return mf 

	def dndM_G(self,M,G):
	    _,_,mf=self.mass_func(M,G)
	    return mf 

	def dndM(self,M,z):
            G=self.Growth(z)
	    _,_,mf=self.mass_func(M,G)
	    return mf 

	def M_star(self):
		return self.M_of_nu(1.0)
		
	def bias(self,M,z,norm=None):
		A=0.3222; a=0.707; p=0.3
		G=self.Growth(z)
		nu=self.nu_of_M(M)/G**2
                if norm is None:
		    norm=self.bias_norm(G)
		bias=(1 + (a*nu-1)/self.delta_c + 2*p/(self.delta_c*(1+(a*nu)**p)))
		return bias/norm

	#TODO vectorize,rename,interpolate
	def bias_n_avg(self,min_mass,z):
	    mass=self.mass_grid[ self.mass_grid >= min_mass]
	    b_array=np.zeros_like(mass)

            norm = self.b_norm_cache(z)
	    for i in range(mass.size):
	        b_array[i]=self.bias(mass[i],z,norm=norm)
	    G = self.Growth(z) 
	    mf=self.dndM_G(mass,G)	    
	    
	    return trapz(b_array*mf,mass) 
	
	def bias_array(self,M,zs,G,norm):
		A=0.3222; a=0.707; p=0.3
		nu=np.outer(self.nu_of_M(M),1./G**2)
		bias=(1 + (a*nu-1)/self.delta_c + 2*p/(self.delta_c*(1+(a*nu)**p)))
		return bias/norm

	def bias_n_avg_array(self,min_mass,zs):
            result = np.zeros(zs.size)
            norms = self.b_norm_cache(zs)
            Gs = self.Growth(zs)
            b_array = self.bias_array(self.mass_grid,zs,Gs,norms)
	    mf=self.dndM_G_array(self.mass_grid,Gs)
            integrated = RectBivariateSpline(self.mass_grid,zs,-np.vstack((np.zeros((1,mf.shape[1])),cumtrapz((b_array*mf)[::-1,:],self.mass_grid[::-1],axis=0)))[::-1,:],kx=1,ky=1)
	    #TODO could unroll loop, if speedup is necessary
            for itr in range(0,zs.size):
                #restrict = self.mass_grid>= min_mass[itr]
                result[itr] = integrated(min_mass[itr],zs[itr])#trapz(b_array[restrict,itr]*mf[restrict,itr],self.mass_grid[restrict],axis=0) 	
            return result
	    
		
	def n_avg(self,M,z):
		mass=self.mass_grid[ self.mass_grid >= M]
                G = self.Growth(z)
		mf=self.dndM_G(mass,G)
		return trapz(mf,mass)

	def n_avg_array(self,M,zs):
                G = self.Growth(zs)
		mf=self.dndM_G_array(self.mass_grid,G)
                integrated = RectBivariateSpline(self.mass_grid,zs,-np.vstack((np.zeros((1,mf.shape[1])),cumtrapz(mf[::-1,:],self.mass_grid[::-1],axis=0)))[::-1,:],kx=1,ky=1)
                result = np.zeros(zs.size)
                for itr in range(0,zs.size):
                    result[itr] = integrated(M[itr],zs[itr])
                #for itr in range(0,zs.size):
                #    restrict = self.mass_grid>=M[itr]
                #    result[itr] = trapz(mf[restrict,itr],self.mass_grid[restrict],axis=0) 
		return result
		
		
	
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
	
	#z=1.1; h=.677el
	#rho_bar=CP.rho_bar(z)
	hmf=ST_hmf(CP)
	
	
	z1=0;z2=1.1;z3=.1
        G1=hmf.Growth(z1);G2=hmf.Growth(z2);G3=hmf.Growth(z3)
	#M=np.logspace(8,15,150)
	M=d[:,0]
	_,f1,mf1=hmf.mass_func(M,G1)
	_,f2,mf2=hmf.mass_func(M,G2)
	_,f2,mf3=hmf.mass_func(M,G3)
	
	
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
        GG=hmf.Growth(zz)
	for i in range(zz.size):
		_,_,y=hmf.mass_func(M,GG[i])
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
	
