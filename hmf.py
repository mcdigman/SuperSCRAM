'''
    Halo mass function classes
'''

from scipy.interpolate import interp1d,RectBivariateSpline,InterpolatedUnivariateSpline
from scipy.integrate import cumtrapz
import numpy as np
from algebra_utils import trapz2

#NOTE if useful could add arXiv:astro-ph/9907024 functionality
class ST_hmf(object):
    """Sheth Tormen Halo Mass Function"""
    def __init__(self,C,params,delta_bar=None):
        """ C: a CosmoPie
            delta_bar: mean density fluctuation
            params:
                log10_min_mass,log10_max_mass: log10(minimum/maximum mass to go to)
                n_grid: number of grid points for mass
        """
        self.A = 0.3222
        self.a = 0.707
        self.p = 0.3

        self.params = params
        # log 10 of the minimum and maximum halo mass
        self.min = params['log10_min_mass']
        self.max = params['log10_max_mass']
        # number of grid points in M
        n_grid = self.params['n_grid']
        # I add an additional point because I will take derivatives latter
        # and the finite difference scheme misses that last grid point.
        dlog_M = (self.max-self.min)/float(n_grid)
        self.M_grid = 10**np.linspace(self.min,self.max+dlog_M, n_grid+1)
        self.mass_grid = self.M_grid[0:-1]


        self.nu_array = np.zeros(n_grid+1)
        self.sigma = np.zeros(n_grid+1)
        self.C = C

        #self.delta_c=C.delta_c(0)
        #TODO consider having option to overwride delta_c
        self.delta_c = 1.686
        if delta_bar is not None:
            self.delta_c = self.delta_c - delta_bar
        self.rho_bar = C.rho_bar(0)

        self.Growth = C.G_norm


        #print 'rhos', 0,self.rho_bar/(1e10), C.rho_crit(0)/(1e10)

        self.R = (3./4.*self.M_grid/self.rho_bar/np.pi)**(1./3.)
        #manually fixed h factor to agree with convention
        self.sigma = C.sigma_r(0.,self.R/C.h)

        # calculated at z=0
        self.nu_array = (self.delta_c/self.sigma)**2
        sigma_inv = self.sigma**(-1)
        dsigma_dM = np.diff(np.log(sigma_inv))/(np.diff(self.M_grid))

        self.sigma_of_M = interp1d(self.M_grid[:-1],self.sigma[:-1])
        self.nu_of_M = interp1d(self.M_grid[:-1], self.nu_array[:-1])
        self.M_of_nu = interp1d(self.nu_array[:-1],self.M_grid[:-1])
        self.dsigma_dM_of_M = interp1d(self.M_grid[:-1],dsigma_dM)

        #grid for precomputing z dependent quantities
        self.z_grid = np.arange(self.params['z_min'],self.params['z_max'],self.params['z_resolution'])
        self.G_grid = C.G_norm(self.z_grid)


        self.b_norm_overwride = self.params['b_norm_overwride']
        if self.b_norm_overwride:
            self.b_norm_grid = self.bias_norm(self.G_grid)
            self.b_norm_cache = InterpolatedUnivariateSpline(self.G_grid[::-1],self.b_norm_grid[::-1],k=3,ext=2)

        self.f_norm_overwride = self.params['f_norm_overwride']
        if self.f_norm_overwride:
            self.f_norm_grid = self.f_norm(self.G_grid)
            self.f_norm_cache = InterpolatedUnivariateSpline(self.G_grid[::-1],self.f_norm_grid[::-1],k=3,ext=2)

    def f_norm(self,G):
        """ get normalization factor to ensure all mass is in a halo
            accepts either a numpy array or a scalar input for G"""
        if isinstance(G,np.ndarray):
            nu = np.outer(self.nu_array,1./G**2)
            sigma_inv = np.outer(1./self.sigma,1./G**2)
        else:
            nu = self.nu_array/G**2
            sigma_inv = 1./self.sigma*1./G**2
        f = self.A*np.sqrt(2*self.a/np.pi)*(1. + (1./self.a/nu)**self.p)*np.sqrt(nu)*np.exp(-self.a*nu/2.)
        norm = trapz2(f,np.log(sigma_inv))
        return norm

    def bias_norm(self,G):
        """ gets the normalization for bias b(r) for Sheth-Tormen mass function
            supports numpy array or scalar for G"""
        if isinstance(G,np.ndarray):
            nu = np.outer(self.nu_array,1./G**2)
        else:
            nu = self.nu_array/G**2

        bias = self.bias_nu(nu)
        f = self.f_nu(nu)
        norm = trapz2(f*bias,nu)
        return norm

    def bias_nu(self,nu):
        """return bias as a function of a numpy matrix nu"""
        return 1. + (self.a*nu-1.)/self.delta_c + 2.*self.p/(self.delta_c*(1.+(self.a*nu)**self.p))

    def f_nu(self,nu):
        """return f as a function of a numpy matrix nu"""
        return self.A*np.sqrt(2.*self.a/np.pi)*(1. + (1./self.a/nu)**self.p)*np.sqrt(nu)*np.exp(-self.a*nu/2.)


    def f_sigma(self,M,G,norm_in=None):
        """ gets Sheth-Tormen mass function normalized so all mass is in a halo
            implements equation 7 in arXiv:astro-ph/0005260
            both M and G can be numpy arrays
            renormalizing for input mass range is optional"""
        if norm_in is None and self.f_norm_overwride:
            norm = self.f_norm_cache(G)
        else:
            norm = 1.

        #note  nu = delta_c**2/sigma**2
        if isinstance(G,np.ndarray) and isinstance(M,np.ndarray):
            nu = np.outer(self.nu_of_M(M),1./G**2)
        else:
            nu = self.nu_of_M(M)/G**2

        f = self.f_nu(nu)
        if self.f_norm_overwride or norm_in is not None:
            return f/norm
        else:
            return f

    def mass_func(self,M,G):
        """ get Sheth-Tormen mass function
            implements equation 3 in   arXiv:astro-ph/0607150
            supports M and G to be numpy arrays or scalars"""
        f = self.f_sigma(M,G)
        sigma = self.sigma_of_M(M)
        dsigma_dM = self.dsigma_dM_of_M(M)
        if isinstance(G,np.ndarray) and isinstance(M,np.ndarray):
            mf = (self.rho_bar/M*dsigma_dM*f.T).T
        else:
            mf = self.rho_bar/M*dsigma_dM*f
        # return sigma,f,and the mass function dn/dM
        return sigma,f,mf


    def dndM_G(self,M,G):
        """ gets Sheth-Tormen dn/dM as a function of G
            both M and G can be numpy arrays or scalars"""
        _,_,mf = self.mass_func(M,G)
        return mf

    def dndM(self,M,z):
        """ gets Sheth-Tormen dn/dM as a function of z
            both M and z can be numpy arrays or scalars"""
        G = self.Growth(z)
        _,_,mf = self.mass_func(M,G)
        return mf

    def M_star(self):
        """get M(nu=1.0)"""
        return self.M_of_nu(1.0)

    def bias(self,M,z,norm_in=None):
        """get Sheth-Tormen bias as a function of z"""
        G = self.Growth(z)
        self.bias_G(M,G,norm_in=norm_in)

    def bias_G(self,M,G,norm_in=None):
        """get bias as a function of G
        implements arXiv:astro-ph/9901122 eq 12"""
        if norm_in is None and self.b_norm_overwride:
            norm = self.b_norm_cache(G)
        else:
            norm = 1.

        if isinstance(M,np.ndarray) and isinstance(G,np.ndarray):
            nu = np.outer(self.nu_of_M(M),1./G**2)
        else:
            nu = self.nu_of_M(M)/G**2
        bias = self.bias_nu(nu)
        if self.b_norm_overwride or norm_in is not None:
            return bias/norm
        else:
            return bias

    def bias_n_avg(self,min_mass,z):
        """ get b(z)n(z) for Sheth Tormen mass function
            min_mass can be a function of z or a constant
            if min_mass is a function of z it should be an array of the same size as z
            z can be scalar or an array"""
        if isinstance(min_mass,np.ndarray):
            if not min_mass.size==z.size:
                raise ValueError('min_mass and z must be the same size')
            elif min_mass.size==1:
                #spline method fails in case where min_mass size is 1 so treat as scalar
                min_mass = min_mass[0]
                z = z[0]
            return_array=True
        else:
            return_array=False

        G = self.Growth(z)
        if self.b_norm_overwride:
            norm = self.b_norm_cache(G)
        else:
            norm = None

        #TODO could probably consolidate these conditions somewhat
        if isinstance(min_mass,np.ndarray):
            mf = self.dndM_G(self.mass_grid,G)

            b_array = self.bias_G(self.mass_grid,G,norm)

            result = np.zeros(G.size)

            grid_integrated = -np.vstack((np.zeros((1,mf.shape[1])),cumtrapz((b_array*mf)[::-1,:],self.mass_grid[::-1],axis=0)))[::-1,::-1]
            integrated = RectBivariateSpline(self.mass_grid,G[::-1],grid_integrated,kx=1,ky=1)
            for itr in xrange(0,G.size):
                result[itr] = integrated(min_mass[itr],G[itr])
            return result
        else:
            mass = self.mass_grid[ self.mass_grid>=min_mass]
            mf = self.dndM_G(mass,G)

            b_array = np.zeros_like(mass)

            for i in xrange(mass.size):
                b_array[i] = self.bias_G(mass[i],G,norm_in=norm)

            if return_array:
                return np.array([trapz2(b_array*mf,mass)])
            else:
                return trapz2(b_array*mf,mass)

    def n_avg(self,min_mass,z):
        """ M is lower cutoff mass
            min_mass can be a function of z or a constant
            if min_mass is a function of z it should be an array of the same size as z
            z can be scalar or an array"""
        if isinstance(z,np.ndarray):
            if isinstance(min_mass,np.ndarray) and not min_mass.size==z.size:
                raise ValueError('min_mass and z must be the same size')
            if z.size==1:
                #spline method fails in case where z size is 1 so treat as scalar
                min_mass = min_mass[0]
                z = z[0]
            return_array=True
        else:
            return_array=False
        G = self.Growth(z)

        if isinstance(min_mass,np.ndarray) and isinstance(z,np.ndarray):
            mf = self.dndM_G(self.mass_grid,G)
            integrated = RectBivariateSpline(self.mass_grid,G[::-1],-np.vstack((np.zeros((1,mf.shape[1])),cumtrapz(mf[::-1,:],self.mass_grid[::-1],axis=0)))[::-1,::-1],kx=1,ky=1)
            result = np.zeros(G.size)
            for itr in xrange(0,G.size):
                result[itr] = integrated(min_mass[itr],G[itr])
            return result
        else:
            if isinstance(min_mass,np.ndarray):
                result = np.zeros(min_mass.size)
                for i in xrange(0,min_mass.size):
                    mass = self.mass_grid[ self.mass_grid>=min_mass[i]]
                    mf = self.dndM_G(mass,G)
                    result[i] = trapz2(mf,mass)
                return result
            else:
                mass = self.mass_grid[ self.mass_grid>=min_mass]
                mf = self.dndM_G(mass,G)
                if return_array:
                    return np.array([trapz2(mf,mass)])
                else:
                    return trapz2(mf,mass)
#    def sigma_m(self,mass,z):
#        """ RMS power on a scale of R(mass)
#         rho = mass/volume=mass"""
#        R = 3/4.*mass/self.rho_bar(z)/np.pi
#        R = R**(1/3.)
#        return self.C.sigma_r(z,R)
#    def delta_c(self,z):
#        """ critical threshold for spherical collapse, as given
#        in the appendix of NFW 1997"""
#        A = 0.15*(12.*np.pi)**(2/3.)
#
#        if (self.C.Omegam==1) and (self.C.OmegaL==0):
#            d_crit = A
#        elif (self.C.Omegam < 1) and (self.C.OmegaL==0):
#            d_crit = A*self.Omegam**(0.0185)
#        elif (self.C.Omegam + self.C.OmegaL)==1.0:
#            d_crit = A*self.C.Omegam**(0.0055)
#        else:
#            d_crit = A*self.C.Omegam**(0.0055)
#            warn('inexact equality to 1~='+str(self.C.Omegam)+"+"+str(self.C.OmegaL)+"="+str(self.C.OmegaL+self.C.Omegam))
#        d_c = d_crit#/self.C.G_norm(z)
#        return d_c
#
#    def delta_v(self,z):
#        """over density for virialized halo"""
#        A = 178.0
#        if (self.C.Omegam_z(z)==1) and (self.C.OmegaL_z(z)==0):
#            d_v = A
#        if (self.C.Omegam_z(z) < 1) and (self.C.OmegaL_z(z)==0):
#            d_v = A/self.C.Omegam_z(z)**(0.7)
#        if (self.C.Omegam_z(z) + self.C.OmegaL_z(z))==1.0:
#            d_v = A/self.C.Omegam_z(z)**(0.55)
#
#        return d_v/self.G_norm(z)
#
#
#    def nu(self,z,mass):
#        """calculates nu=(delta_c/sigma(M))^2
#         delta_c is the overdensity for collapse"""
#        return (self.delta_c(z)/self.sigma_m(mass,z))**2

#if __name__=="__main__":
#
#    #cosmology={'Omegabh2' :0.02230,
#    #                       'Omegach2' :0.1188,
#    #                       'Omegamh2' : 0.14170,
#    #                       'OmegaL'   : .6911,
#    #                       'Omegam'   : .3089,
#    #                       'H0'       : 67.74,
#    #                       'sigma8'   : .8159,
#    #                       'h'        :.6774,
#    #                       'Omegak'   : 0.0, # check on this value
#    #                       'Omegar'   : 0.0}
#    from scipy.integrate import trapz
#    #d=np.loadtxt('Pk_Planck15.dat')
#    #k=d[:,0]; P=d[:,1]
#    #d=np.loadtxt('test_data/hmf_z0_test1.dat')
#    #d=np.loadtxt('test_data/hmf_test.dat')
#
#    import cosmopie as cp
#    import matter_power_spectrum as mps
#    cosmo_fid = defaults.cosmology
#
#    cosmo_fid['h'] = 0.65
#    cosmo_fid['Omegamh2'] = 0.148
#    cosmo_fid['Omegabh2'] = 0.02
#    cosmo_fid['OmegaLh2'] = 0.65*0.65**2
#    cosmo_fid['sigma8'] = 0.92
#    cosmo_fid['ns'] = 1.
#    cosmo_fid = cp.add_derived_pars(cosmo_fid,p_space='basic')
#    C = cp.CosmoPie(cosmo_fid)
#    P = mps.MatterPower(C)
#    C.P_lin = P
#    C.k = P.k
#    #z=1.1; h=.677el
#    #rho_bar=C.rho_bar(z)
#    params = defaults.hmf_params.copy()
#    params['z_min'] = 0.0
#    params['z_max'] = 5.0
#    params['z_resolution'] = 0.5
#    params['log10_min_mass'] = 6
#    params['log10_max_mass'] = 18.63
#    params['n_grid'] = 50000
#    hmf = ST_hmf(C,params=params)
#
#    #M=np.logspace(8,15,150)
#
#    import matplotlib.pyplot as plt
#
#    do_plot_check1 = False
#    if do_plot_check1:
#
#        h = .6774
#        z1 = 0;z2=1.1;z3=.1
#        G1 = hmf.Growth(z1);G2=hmf.Growth(z2);G3=hmf.Growth(z3)
#
#        M = hmf.mass_grid
#        #M=d[:,0]
#        _,f1,mf1=hmf.mass_func(M,G1)
#        _,f2,mf2=hmf.mass_func(M,G2)
#        _,f2,mf3=hmf.mass_func(M,G3)
#
#        ax = plt.subplot(131)
#        ax.set_xscale('log')
#        ax.set_yscale('log')
#        #ax.set_xlim(9.6,15)
#        #ax.set_ylim(-5.4,0)
#        ax.set_ylabel(r'$\frac{d n}{d \log_{10}(M_h)}$', size=20)
#        ax.plot(M,mf1*M)
#        #ax.plot(d[:,0], d[:,1]/h, '--')
#
#        ax.plot(M,mf2*M)
#        #ax.plot(d[:,0], d[:,3]/h**2, '--')
#
#        zz = np.array([.1,.2,.3,.5,1,2])
#        GG = hmf.Growth(zz)
#        for i in xrange(zz.size):
#            _,_,y=hmf.mass_func(M,GG[i])
#            ax.plot(M,y*M)
#
#
#        ax.plot(M,mf3*M)
#        #plt.axvline(M_star)
#
#
#        plt.grid()
#
#        ax = plt.subplot(132)
#        ax.set_xscale('log')
#        #ax.set_xlim(-.2,.4)
#        ax.set_yscale('log')
#        ax.set_ylabel(r'$f(M_h)}$', size=20)
#
#
#        ax.plot(M,f1)
#        #ax.plot(M,d[:,2]/h, color='red')
#
#        ax.plot(M,f2)
#        #ax.plot(M,d[:,4]/h, color='red')
#
#        plt.grid()
#
#        ax = plt.subplot(133)
#        ax.set_xscale('log')
#        #ax.set_xlim(-.2,.4)
#        ax.set_yscale('log')
#        ax.set_ylabel(r'$f(M_h)}$', size=20)
#
#        b1 = hmf.bias(M,0)
#        b2 = hmf.bias(M,1)
#
#        ax.plot(M,b1)
#        ax.plot(M,b2)
#
#        plt.grid()
#        plt.show()
#
#    do_sanity_checks = True
#    if do_sanity_checks:
#        #some sanity checks
#        #Gs = np.arange(0.01,1.0,0.01)
#        zs = np.arange(0.,4.,0.5)
#        Gs = C.G_norm(zs)
#        #check normalized to unity (all dark matter is in some halo)
#        f_norm_residual = trapz(hmf.f_sigma(hmf.mass_grid,Gs).T,np.log(hmf.sigma[:-1:]**-1),axis=1)
#        #assert np.allclose(np.zeros(Gs.size)+1.,norm_residual)
#        _,_,dndM = hmf.mass_func(hmf.mass_grid,Gs)
#        n_avgs_alt = np.zeros(zs.size)
#        bias_n_avgs_alt = np.zeros(zs.size)
#        dndM_G_alt = np.zeros((hmf.mass_grid.size,zs.size))
#        #arbitrary input M(z) cutoff
#        m_z_in = np.linspace(np.min(hmf.mass_grid),np.max(hmf.mass_grid),zs.size)
#        for i in xrange(0,zs.size):
#            n_avgs_alt[i] = hmf.n_avg(m_z_in[i],zs[i])
#            bias_n_avgs_alt[i] = hmf.bias_n_avg(m_z_in[i],zs[i])
#            dndM_G_alt[:,i] = hmf.dndM_G(hmf.mass_grid,Gs[i])
#        dndM_G = hmf.dndM_G(hmf.mass_grid,Gs)
#        #consistency checks for vector method
#        assert np.allclose(dndM_G,dndM_G_alt)
#        n_avgs = hmf.n_avg(m_z_in,zs)
#        assert np.allclose(n_avgs,n_avgs_alt)
#        bias_n_avgs = hmf.bias_n_avg(m_z_in,zs)
#        assert np.allclose(bias_n_avgs,bias_n_avgs_alt)
#        #check integrating dn/dM over all M actually gives n
#        assert np.allclose(trapz(dndM.T,hmf.mass_grid,axis=1),hmf.n_avg(np.zeros(zs.size)+hmf.mass_grid[0],zs))
#        #not sure why, but this is true
#        #assert np.allclose(np.zeros(zs.size)+1.,np.trapz(hmf.f_sigma(hmf.mass_grid,Gs)*hmf.bias_G(hmf.mass_grid,Gs,hmf.bias_norm(Gs)),np.outer(hmf.nu_of_M(hmf.mass_grid),1./Gs**2),axis=0)*hmf.f_norm(Gs))
#        b_norm_residual = np.trapz(hmf.f_sigma(hmf.mass_grid,Gs)*hmf.bias_G(hmf.mass_grid,Gs,hmf.bias_norm(Gs)),np.outer(hmf.nu_of_M(hmf.mass_grid),1./Gs**2),axis=0)
#        b_norm_residual_alt = np.trapz(hmf.f_sigma(hmf.mass_grid,Gs)*hmf.bias_G(hmf.mass_grid,Gs,1.),np.outer(hmf.nu_of_M(hmf.mass_grid),1./Gs**2),axis=0)
#        #assert np.allclose(np.zeros(zs.size)+1.,b_norm_residual)
#
#    do_plot_test2 = True
#    if do_plot_test2:
#        #Ms = 10**(np.linspace(11,14,500))
#        Ms = hmf.mass_grid
#       # dndM_G=hmf.dndM_G(hmf.mass_grid,Gs)
#        do_jenkins_comp = False
#        #should look like dotted line in figure 3 of    arXiv:astro-ph/0005260
#        if do_jenkins_comp:
#            zs = np.array([0.])
#            Gs = C.G_norm(zs)
#            dndM_G = hmf.f_sigma(hmf.mass_grid,Gs)
#            input_dndm = np.loadtxt('test_inputs/hmf/dig_jenkins_fig3.csv',delimiter=',')
#            plt.plot(np.log(1./hmf.sigma[:-1:]),np.log(dndM_G))
#            plt.plot(input_dndm[:,0],input_dndm[:,1])
#            plt.xlim([-1.3,1.25])
#            plt.ylim([-10.5,-0.5])
#            plt.grid()
#            plt.show()
#
#        do_sheth_bias_comp = True
#        #agrees pretty well, maybe as well as it should
#        if do_sheth_bias_comp:
#            cosmo_fid2 = cosmo_fid.copy()
#            cosmo_fid2['Omegamh2'] = 0.3*0.7**2
#            cosmo_fid2['Omegabh2'] = 0.05*0.7**2
#            cosmo_fid2['OmegaLh2'] = 0.7*0.7**2
#            cosmo_fid2['sigma8'] = 0.9
#            cosmo_fid2['h'] = 0.7
#            cosmo_fid2['ns'] = 1.0
#            cosmo_fid2 = cp.add_derived_pars(cosmo_fid2,p_space='basic')
#            C2 = cp.CosmoPie(cosmo_fid2)
#            P2 = mps.MatterPower(C2)
#            C2.P_lin = P2
#            C2.k = P2.k
#            hmf2 = ST_hmf(C2,params=params)
#            zs = np.array([0.,1.,2.,4.])
#            Gs = C2.G_norm(zs)
#            bias = hmf2.bias_G(Ms,Gs)
#            input_bias = np.loadtxt('test_inputs/hmf/dig_sheth_fig3.csv',delimiter=',')
#            plt.loglog(Ms,bias**2)
#            plt.loglog(10**input_bias[:,0],10**input_bias[:,1:5])
#            plt.xlim([10**11,10**14])
#            plt.ylim([0.2,80])
#            plt.grid()
#            plt.show()
#        #might agree
#        do_hu_bias_comp1 = False
#        if do_hu_bias_comp1:
#            zs = np.array([0.])
#            Gs = C.G_norm(zs)
#            bias = hmf.bias_G(Ms,Gs)[:,0]
#            input_bias = np.loadtxt('test_inputs/hmf/dig_hu.csv',delimiter=',')
#            plt.semilogx(Ms,bias)
#            plt.semilogx(10**input_bias[:,0],input_bias[:,1])
#            plt.xlim([3*10**14,2*10**15])
#            plt.ylim([1,8])
#            plt.grid()
#            plt.show()
#            masses = 10**np.linspace(np.log10(3*10**14),np.log10(2*10**15),100)
#            bias_hu_i = InterpolatedUnivariateSpline(10**input_bias[:,0],input_bias[:,1])(masses)
#            bias_i = InterpolatedUnivariateSpline(Ms,bias)(masses)
#        #should agree, does
#        do_hu_bias_comp2 = False
#        if do_hu_bias_comp2:
#            zs = np.array([0.])
#            Gs = C.G_norm(zs)
#            bias = hmf.bias_G(Ms,Gs)[:,0]
#            input_bias = np.loadtxt('test_inputs/hmf/dig_hu_bias2.csv',delimiter=',')
#            plt.loglog(Ms,bias)
#            plt.loglog(10**input_bias[:,0],10**input_bias[:,1])
#            plt.xlim([10**11,10**16])
#            plt.ylim([0.01,10])
#            plt.grid()
#            plt.show()
#            masses = 10**np.linspace(np.log10(10**11),np.log10(10**16),100)
#            bias_hu_i = InterpolatedUnivariateSpline(10**input_bias[:,0],10**input_bias[:,1])(masses)
#            bias_i = InterpolatedUnivariateSpline(Ms,bias)(masses)
#
#        #might agree
#        do_hu_navg_comp1 = False
#        if do_hu_navg_comp1:
#            cosmo_fid2 = cosmo_fid.copy()
#            cosmo_fid2['h'] = 0.65
#            cosmo_fid2['Omegamh2'] = 0.15
#            cosmo_fid2['Omegabh2'] = 0.02
#            cosmo_fid2['OmegaLh2'] = (1-cosmo_fid2['Omegamh2']/cosmo_fid2['h']**2)*cosmo_fid2['h']**2
#            cosmo_fid2['sigma8'] = 1.07
#            cosmo_fid2['ns'] = 1.
#            cosmo_fid2 = cp.add_derived_pars(cosmo_fid2,p_space='basic')
#            C2 = cp.CosmoPie(cosmo_fid2)
#            P2 = mps.MatterPower(C2)
#            C2.P_lin = P2
#            C2.k = P2.k
#            zs = np.array([0.])
#            Gs = C2.G_norm(zs)
#            hmf2 = ST_hmf(C2,params=params)
#            input_navg = np.loadtxt('test_inputs/hmf/dig_hu_navg.csv',delimiter=',')
#            navg = hmf2.n_avg(10**input_navg[:,0],0.)
#            plt.loglog(10**input_navg[:,0],navg)
#            plt.loglog(10**input_navg[:,0],10**input_navg[:,1])
#            plt.xlim([10**13.6,10**16])
#            plt.ylim([10**-9,10**-4])
#            plt.grid()
#            plt.show()
#            masses = 10**np.linspace(np.log10(10**13.6),np.log10(10**16),100)
#            navg_hu_i = InterpolatedUnivariateSpline(10**input_navg[:,0],10**input_navg[:,1])(masses)
#            navg_i = InterpolatedUnivariateSpline(10**input_navg[:,0],navg)(masses)
#        #should agree, does
#        do_hu_sigma_comp = False
#        if do_hu_sigma_comp:
#            input_sigma = np.loadtxt('test_inputs/hmf/dig_hu_sigma.csv',delimiter=',')
#            plt.loglog(hmf.R,hmf.sigma)
#            plt.loglog(10**input_sigma[:,0],10**input_sigma[:,1])
#            plt.xlim([0.1,200])
#            plt.ylim([0.001,10])
#            plt.grid()
#            plt.show()
