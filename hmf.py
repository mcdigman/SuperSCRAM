'''
    Halo mass function classes
'''
import numpy as np
from numpy import log, log10, exp, pi
from scipy.interpolate import interp1d,RectBivariateSpline,InterpolatedUnivariateSpline
from scipy.integrate import trapz,cumtrapz 
import sys
import defaults

#TODO determine if useful to add    arXiv:astro-ph/9907024 functionality
class ST_hmf():
        def __init__(self,CosmoPie, delta_bar=None,params=defaults.hmf_params):
            self.params = params
            # log 10 of the minimum and maximum halo mass
            self.min=params['log10_min_mass']
            self.max=params['log10_max_mass']
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
            

            #print 'rhos', 0,self.rho_bar/(1e10), CosmoPie.rho_crit(0)/(1e10)
            
            R = (3./4.*self.M_grid/self.rho_bar/pi)**(1./3.)
            self.sigma=CosmoPie.sigma_r(0.,R)
            
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
            self.b_norm_cache = InterpolatedUnivariateSpline(self.G_grid[::-1],self.b_norm_grid[::-1],k=3,ext=2)

            self.f_norm_grid = self.f_norm(self.G_grid)
            self.f_norm_cache = InterpolatedUnivariateSpline(self.G_grid[::-1],self.f_norm_grid[::-1],k=3,ext=2)
                
        #NOTE most of these methods would benefit from caching their results for a grid of G and mass, if time becomes a problem

        #get normalization factor to ensure all mass is in a halo
        #accepts either a numpy array or a scalar input for G
        def f_norm(self,G):
            A=0.3222; a=0.707; p=0.3
            if isinstance(G,np.ndarray):
                nu=np.outer(self.nu_array,1./G**2)  
                sigma_inv=np.outer(1./self.sigma,1./G**2)
            else:
                nu=self.nu_array/G**2
                sigma_inv=1./self.sigma*1./G**2
            f=A*np.sqrt(2*a/np.pi)*(1 + (1/a/nu)**p)*np.sqrt(nu)*exp(-a*nu/2.)
            norm=trapz(f,log(sigma_inv),axis=0)
            return norm

        #gets the normalization for bias b(r) for Sheth-Tormen mass function
        #supports numpy array or scalar for G
        def bias_norm(self,G):
            if isinstance(G,np.ndarray):
                nu=np.outer(self.nu_array,1./G**2)
            else:
                nu=self.nu_array/G**2
    
            bias = self.bias_nu(nu)
            f = self.f_nu(nu)
            norm=trapz(f*bias,nu,axis=0)
            return norm
            
        #return bias as a function of a numpy matrix nu 
        def bias_nu(self,nu):
            A=0.3222; a=0.707; p=0.3
            return (1 + (a*nu-1)/self.delta_c + 2*p/(self.delta_c*(1+(a*nu)**p)))

        #return f as a function of a numpy matrix nu
        def f_nu(self,nu):
            A=0.3222; a=0.707; p=0.3
            return A*np.sqrt(2*a/np.pi)*(1 + (1/a/nu)**p)*np.sqrt(nu)*exp(-a*nu/2.)
            

        #gets Sheth-Tormen mass function normalized so all mass is in a halo
        #implements equation 7 in arXiv:astro-ph/0005260
        #both M and G can be numpy arrays
        def f_sigma(self,M,G,norm=None):
            #TODO make norm optional
            if norm is None:
                norm=self.f_norm_cache(G)

            #note  nu = delta_c**2/sigma**2
            if isinstance(G,np.ndarray) and isinstance(M,np.ndarray):
                nu=np.outer(self.nu_of_M(M),1./G**2)
            else:
                nu=self.nu_of_M(M)/G**2

            f = self.f_nu(nu)
            return f/norm

        #get Sheth-Tormen mass function
        #implements equation 3 in   arXiv:astro-ph/0607150
        #supports M and G to be numpy arrays or scalars
        def mass_func(self,M,G):
            f=self.f_sigma(M,G)
            sigma=self.sigma_of_M(M)
            dsigma_dM=self.dsigma_dM_of_M(M)
            if isinstance(G,np.ndarray) and isinstance(M,np.ndarray):
                mf=(self.rho_bar/M*dsigma_dM*f.T).T
            else:
                mf=self.rho_bar/M*dsigma_dM*f
            # return sigma,f,and the mass function dn/dM
            return sigma,f,mf 


        #gets Sheth-Tormen dn/dM as a function of G
        #both M and G can be numpy arrays or scalars
        def dndM_G(self,M,G):
            _,_,mf=self.mass_func(M,G)
            return mf 

        #gets Sheth-Tormen dn/dM as a function of z
        #both M and z can be numpy arrays or scalars
        def dndM(self,M,z):
            G=self.Growth(z)
            _,_,mf=self.mass_func(M,G)
            return mf 

        def M_star(self):
            return self.M_of_nu(1.0)
    
        #get Sheth-Tormen bias as a function of z    
        def bias(self,M,z,norm=None):
            G=self.Growth(z)
            nu=self.nu_of_M(M)/G**2
            if norm is None:
                norm=self.b_norm_cache(G)
            bias = self.bias_nu(nu)
            return bias/norm
        #get bias as a function of G
        #implements arXiv:astro-ph/9901122 eq 12 up to norm
        #TODO optionally includes normalization as an argument
        def bias_G(self,M,G,norm=None):

            if norm is None:
                norm=self.b_norm_cache(G)

            if isinstance(M,np.ndarray) and isinstance(G,np.ndarray):
                nu=np.outer(self.nu_of_M(M),1./G**2)
            else:
                nu=self.nu_of_M(M)/G**2
            bias = self.bias_nu(nu)
            return bias/norm

        #get b(z)n(z) for Sheth Tormen mass function 
        #min_mass can be a function of z or a constant
        #if min_mass is a function of z it should be an array of the same size as z
        #z can be scalar or an array
        def bias_n_avg(self,min_mass,z):
            if isinstance(min_mass,np.ndarray):
                if not min_mass.size == z.size:
                    raise ValueError('min_mass and z must be the same size')

            G = self.Growth(z) 
            norm = self.b_norm_cache(G)
            

            #TODO could probably consolidate these conditions somewhat
            if isinstance(min_mass,np.ndarray):
                mf=self.dndM_G(self.mass_grid,G)

                b_array = self.bias_G(self.mass_grid,G,norm)

                result = np.zeros(G.size)

                #TODO fix in case where min_mass size is 1 
                integrated = RectBivariateSpline(self.mass_grid,G[::-1],-np.vstack((np.zeros((1,mf.shape[1])),cumtrapz((b_array*mf)[::-1,:],self.mass_grid[::-1],axis=0)))[::-1,::-1],kx=1,ky=1)
                for itr in range(0,G.size):
                    result[itr] = integrated(min_mass[itr],G[itr])
                return result
            else:
                mass=self.mass_grid[ self.mass_grid >= min_mass]
                mf=self.dndM_G(mass,G)  

                b_array=np.zeros_like(mass)

                for i in range(mass.size):
                    b_array[i]=self.bias_G(mass[i],G,norm=norm)
            
                return trapz(b_array*mf,mass) 

        #M is lower cutoff mass
        #min_mass can be a function of z or a constant
        #if min_mass is a function of z it should be an array of the same size as z
        #z can be scalar or an array
        def n_avg(self,min_mass,z):
            if isinstance(min_mass,np.ndarray):
                if not min_mass.size == z.size:
                    raise ValueError('min_mass and z must be the same size')
            G = self.Growth(z)
            #TODO this does not handle z being of length 1 correctly
            if isinstance(min_mass,np.ndarray):
                mf=self.dndM_G(self.mass_grid,G)
                integrated = RectBivariateSpline(self.mass_grid,G[::-1],-np.vstack((np.zeros((1,mf.shape[1])),cumtrapz(mf[::-1,:],self.mass_grid[::-1],axis=0)))[::-1,::-1],kx=1,ky=1)
                result = np.zeros(G.size)
                for itr in range(0,G.size):
                    result[itr] = integrated(min_mass[itr],G[itr])
                return result
            else:
                mass=self.mass_grid[ self.mass_grid >= min_mass]
                mf=self.dndM_G(mass,G)
                return trapz(mf,mass)
            
    
    
if __name__=="__main__":

        #cosmology={'Omegabh2' :0.02230,
        #                       'Omegach2' :0.1188,
        #                       'Omegamh2' : 0.14170,
        #                       'OmegaL'   : .6911,
        #                       'Omegam'   : .3089,
        #                       'H0'       : 67.74, 
        #                       'sigma8'   : .8159, 
        #                       'h'        :.6774, 
        #                       'Omegak'   : 0.0, # check on this value 
        #                       'Omegar'   : 0.0}

        #d=np.loadtxt('Pk_Planck15.dat')
        #k=d[:,0]; P=d[:,1]
        
        #d=np.loadtxt('test_data/hmf_z0_test1.dat')
        #d=np.loadtxt('test_data/hmf_test.dat')
            
        from cosmopie import CosmoPie
        import cosmopie
        import matter_power_spectrum as mps
        cosmo_fid = defaults.cosmology
        cosmo_fid['Omegamh2']=0.3*0.7**2
        cosmo_fid['Omegabh2']=0.05*0.7**2
        cosmo_fid['OmegaLh2']=0.7*0.7**2
        cosmo_fid['sigma8'] = 0.9
        cosmo_fid['h']=0.7
        cosmo_fid = cosmopie.add_derived_parameters(cosmo_fid,p_space='basic')
        CP=CosmoPie(cosmo_fid)
        P=mps.MatterPower(CP)
        CP.P_lin = P
        CP.k = P.k
        #z=1.1; h=.677el
        #rho_bar=CP.rho_bar(z)
        params = defaults.hmf_params.copy()
        params['z_min'] = 0.0
        params['z_max'] = 5.5
        params['z_resolution'] = 0.001
        params['log10_min_mass'] = -30
        params['log10_max_mass'] = 18.63
        hmf=ST_hmf(CP,params=params)
        
        
        #M=np.logspace(8,15,150)
        
        
        do_plot_check1 = False
        if do_plot_check1:

            h=.6774
            z1=0;z2=1.1;z3=.1
            G1=hmf.Growth(z1);G2=hmf.Growth(z2);G3=hmf.Growth(z3)

            M = hmf.mass_grid
            #M=d[:,0]
            _,f1,mf1=hmf.mass_func(M,G1)
            _,f2,mf2=hmf.mass_func(M,G2)
            import matplotlib.pyplot as plt
            _,f2,mf3=hmf.mass_func(M,G3)
            
            ax=plt.subplot(131)
            ax.set_xscale('log')
            ax.set_yscale('log')
            #ax.set_xlim(9.6,15)
            #ax.set_ylim(-5.4,0)
            ax.set_ylabel(r'$\frac{d n}{d \log_{10}(M_h)}$', size=20)
            ax.plot(M,mf1*M)
            #ax.plot(d[:,0], d[:,1]/h, '--')
            
            ax.plot(M,mf2*M)
            #ax.plot(d[:,0], d[:,3]/h**2, '--')
            
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
                    

            #TODO reintroduce this test
            ax.plot(M,f1)
            #ax.plot(M,d[:,2]/h, color='red')
            
            ax.plot(M,f2)
            #ax.plot(M,d[:,4]/h, color='red')
            
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

        do_sanity_checks=True
        if do_sanity_checks:
            #some sanity checks 
            #Gs = np.arange(0.01,1.0,0.01)
            zs = np.arange(0.,5.,0.1)
            Gs = CP.G_norm(zs)
            #check normalized to unity (all dark matter is in some halo)
            assert(np.allclose(np.zeros(Gs.size)+1.,trapz(hmf.f_sigma(hmf.mass_grid,Gs).T,np.log(hmf.sigma[:-1:]**-1),axis=1)))
            _,_,dndM = hmf.mass_func(hmf.mass_grid,Gs)
            n_avgs_alt = np.zeros(zs.size)
            bias_n_avgs_alt = np.zeros(zs.size)
            dndM_G_alt = np.zeros((hmf.mass_grid.size,zs.size))
            #arbitrary input M(z) cutoff
            m_z_in = np.linspace(np.min(hmf.mass_grid),np.max(hmf.mass_grid),zs.size) 
            for i in range(0,zs.size):
                n_avgs_alt[i] = hmf.n_avg(m_z_in[i],zs[i])
                bias_n_avgs_alt[i] = hmf.bias_n_avg(m_z_in[i],zs[i])
                dndM_G_alt[:,i] = hmf.dndM_G(hmf.mass_grid,Gs[i])
            dndM_G=hmf.dndM_G(hmf.mass_grid,Gs)
            #consistency checks for vector method
            assert(np.allclose(dndM_G,dndM_G_alt)) 
            n_avgs = hmf.n_avg(m_z_in,zs)
            assert(np.allclose(n_avgs,n_avgs_alt)) 
            bias_n_avgs = hmf.bias_n_avg(m_z_in,zs)
            assert(np.allclose(bias_n_avgs,bias_n_avgs_alt))
            #check integrating dn/dM over all M actually gives n
            assert(np.allclose(trapz(dndM.T,hmf.mass_grid,axis=1),hmf.n_avg(np.zeros(zs.size)+hmf.mass_grid[0],zs)))
            #not sure why, but this is true
            assert(np.allclose(np.zeros(zs.size)+1.,np.trapz(hmf.f_sigma(hmf.mass_grid,Gs)*hmf.bias_G(hmf.mass_grid,Gs,hmf.bias_norm(Gs)),np.outer(hmf.nu_of_M(hmf.mass_grid),1./Gs**2),axis=0)*hmf.f_norm(Gs)))
        do_plot_test2 = False    
        if do_plot_test2: 
            import matplotlib.pyplot as plt
            #Ms = 10**(np.linspace(11,14,500))
            Ms = hmf.mass_grid
            zs = np.array([0.])
            Gs = CP.G_norm(zs)
            bias = hmf.bias_G(Ms,Gs,1.)
           # dndM_G=hmf.dndM_G(hmf.mass_grid,Gs)
            dndM_G=hmf.f_sigma(hmf.mass_grid,Gs)*hmf.f_norm(Gs)
            plt.plot(np.log(1./hmf.sigma[:-1:]),np.log(dndM_G))
            plt.xlim([-1.3,1.25])
            plt.ylim([-10.5,-0.5])
            #plt.loglog(Ms,bias**2) 
            #plt.xlim([10**11,10**14])
            #plt.ylim([0.1,100])
            plt.grid()
            plt.show()
