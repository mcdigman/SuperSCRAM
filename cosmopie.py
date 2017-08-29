''' 
        This class computes various cosmological parameters.
        The default cosmology is Planck 2015. 
        Joseph E. McEwen (c) 2016
        mcewen.24@osu.edu 
        
''' 

import numpy as np
from numpy import pi 
from scipy.integrate import romberg, quad, trapz,cumtrapz,odeint
import sys
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline
import defaults
import camb_power
from warnings import warn
eps=np.finfo(float).eps
import matter_power_spectrum as mps

class CosmoPie :
        
    def __init__(self,cosmology=defaults.cosmology, P_lin=None, k=None,p_space=defaults.cosmopie_params['p_space'],needs_power=False,camb_params=defaults.camb_params,safe_sigma8=False,a_step=0.0008,a_step_de = 0.0001,a_extra=10,a_extra_de=100,G_in=None,G_safe=False,silent=False,wmatch=None):
        # default to Planck 2015 values 
        self.silent=silent
        if not silent:
            print "cosmopie "+str(id(self))+": begin initialization"        
        #a wmatcher object is needed to match a variable dark energy equation of state
        self.wmatch=wmatch
        #define parameterization
        self.p_space=p_space
        #fill out cosmology
        self.cosmology = cosmology.copy()
        self.cosmology = add_derived_parameters(self.cosmology,self.p_space,safe_sigma8)

        self.Omegabh2 = self.cosmology['Omegabh2']
        self.Omegach2 = self.cosmology['Omegach2']
        self.Omegamh2 = self.cosmology['Omegamh2']
        self.OmegaL   = self.cosmology['OmegaL']
        self.Omegam   = self.cosmology['Omegam']
        self.ns =self.cosmology['ns']
        self.H0       = self.cosmology['H0']
        self.sigma8   = self.cosmology['sigma8']
        self.h        = self.cosmology['h']
        self.Omegak   = self.cosmology['Omegak']
        self.Omegar   = self.cosmology['Omegar']
        
        #get multipliers in H(z) for OmegaL with an equation of state w(z)
        self.de_model = self.cosmology.get('de_model')
        #default to assuming dark energy is constant
        if self.de_model is None:
            warn('no dark energy model specified, assuming w=-1')
            self.de_model = 'constant_w'
            self.cosmology['de_model']=self.de_model
            self.cosmology['w']=-1
        #z grid for dark energy interpolation
        a_de = np.arange(1.+a_step_de*a_extra_de*2,a_step_de,-a_step_de)
        z_de = 1./a_de-1.
        #z_de = np.arange(0.,z_max,z_space)
        #TODO interpolation may introduce unnecessary error in exact cases
        if self.de_model=='constant_w':
            self.de_mult = InterpolatedUnivariateSpline(z_de,(z_de+1.)**(3.*(1.+self.cosmology['w'])),k=1,ext=2)
            self.ws = np.zeros(z_de.size)+self.cosmology['w']
            self.w_interp = InterpolatedUnivariateSpline(z_de,self.ws,k=1,ext=2) 
        elif self.de_model=='w0wa':
            #self.ws = self.cosmology['w0']+(1.-1./(1.-z_grid_de))*self.cosmology['wa']
            #Chevallier-Polarski-Linder model, solution can be found in ie arXiv:1605.01475
            self.de_mult=InterpolatedUnivariateSpline(z_de,np.exp(-3.*self.cosmology['wa']*z_de/(1.+z_de))*(1.+z_de)**(3.*(1.+self.cosmology['w0']+self.cosmology['wa'])),k=1,ext=2)
            self.ws = self.cosmology['w0']+(1.-1./(1.+z_de))*self.cosmology['wa'] 
            self.w_interp = InterpolatedUnivariateSpline(z_de,self.ws,k=1,ext=2) 
        elif self.de_model=='grid_w':
            #ws = InterpolatedUnivariateSpline(self.cosmology['zs_de'],self.cosmology['ws_de'])(z_de)
            #cf ie https://ned.ipac.caltech.edu/level5/March08/Frieman/Frieman2.html#note1, arXiv:1605.01475
            de_integrand = (1.+self.cosmology['ws'])/(1.+self.cosmology['zs_de'])
            de_mults_in = np.exp(3.*cumtrapz(de_integrand,self.cosmology['zs_de'],initial=0.)) #TODO check initial should actually be zero
            self.de_mult = InterpolatedUnivariateSpline(self.cosmology['zs_de'],de_mults_in,k=2,ext=2) #k=2 so smooths some, check if this is a good idea.
            self.w_interp = InterpolatedUnivariateSpline(self.cosmology['zs_de'],self.cosmology['ws'],k=2,ext=2)
        elif self.de_model=='jdem':
            #piecewise constant approximation over 36 values with z=0.025 spacing
            ws_in = self.cosmology['ws36']
            zs_in = 0.025*np.arange(0,36)/(1-0.025*np.arange(0,36))
            ws_set = np.zeros(zs_in.size)
            itr = 0
            for i in xrange(0,z_de.size):
                if itr<zs_in.size-1:
                    #just extend last z value as far as needed if necessary
                    if zs_in[itr]>=z_de[i]:
                        itr+=1
                ws_set[i] = ws_in[itr] 

            de_integrand = (1.+ws_set)/(1.+zs_in)
            de_mults_in = np.exp(3.*cumtrapz(de_integrand,zs_in,initial=0.))
            self.de_mult = InterpolatedUnivariateSpline(zs_in,de_mults_in,k=2,ext=2) #smoothing may help differential equations
            self.w_interp = InterpolatedUnivariateSpline(zs_in,ws_set,k=2,ext=2)
        else:
            raise ValueError('unrecognized dark energy model \''+str(self.de_model)+'\'')

        self.camb_params = camb_params.copy()

        
        # solar mass
        self.M_sun=1.9885*1e30 # kg
        
        # parsec 
        self.pc=3.08567758149*1e16 # m 
        
        # Newton's constant (CODATA value) 
        self.GN=6.67408*10**(-11) # m^3/kg/s^2
        
        # speed of light 
        self.c        = 2.997924580*1e5 #km/s
        
        self.DH       = self.c/self.H0
        
        self.tH= 1/self.H0
                
        #curvature
        self.K = -self.Omegak*(self.H0/self.c)**2

    
        #precompute some things for speedups if desired (initially use no precompute)
#                self.precompute=False
#                if precompute:
#                    z_grid = np.arange(0.,z_max,z_space)
#                    #there are massive savings from precomputing G because of the integral it contains
#                    G_arr = np.zeros(z_grid.size)
#                    #first compute the integral in G for the whole range \int_0^{1e4}(integrand)
#                    integrand1 = lambda zp : (1+zp)*self.H0**3/self.H(zp)**3
#                    G_base = quad(integrand1,0.,1e4)[0]
#                    #subtract the cumulative result from G_base so final result is \int_z^{1e4}(integrand)
#                    integrand2 = (1.+z_grid)*self.H0**3/self.H(z_grid)**3
#                    self.G_p = interp1d(z_grid,2.5*self.Omegam/self.H0*self.H(z_grid)*(G_base-cumtrapz(integrand2,z_grid,initial=0.)))
#               #     for i in xrange(0,z_grid.size):
#               #         G_arr[i] = self.G(z_grid[i])
#               #     self.G_p = interp1d(z_grid, G_arr)
#                self.precompute=precompute
        #precompute the normalization factor from the differential equation
        self.a_step=a_step
        self.G_safe=G_safe

        a_min=a_step
        a_max=1.+a_step*a_extra
        a_list = np.arange(a_min,a_max,a_step)
        z_list = 1./a_list-1.

        self.z_grid = z_list
        self.a_grid = a_list
        if not G_safe:
            dp_multiplier = -1./a_list*(7./2.*self.Omegam_z(z_list)+3*self.Omegar_z(z_list)+(7./2.-3./2.*self.w_interp(z_list))*self.OmegaL_z(z_list)+4.*self.Omegak_z(z_list))
            d_multiplier = -1./a_list**2*(self.Omegar_z(z_list)+3./2.*(1.-self.w_interp(z_list))*self.OmegaL_z(z_list)+2*self.Omegak_z(z_list))
            #TODO check splines
            dp_mult_interp = InterpolatedUnivariateSpline(a_list,dp_multiplier)
            d_mult_interp = InterpolatedUnivariateSpline(a_list,d_multiplier)
            d_ics = np.array([1.,0.])
            d_args = (np.array([d_mult_interp,dp_mult_interp]),)
            def _d_evolve_eqs(ys,a,g_args):
                yps = np.zeros(2)
                yps[0]=ys[1]
                yps[1]=g_args[0](a)*ys[0]+g_args[1](a)*ys[1]
                return yps
            #drop ghost cell
            integ_result = odeint(_d_evolve_eqs,d_ics,a_list,d_args)
            self.G_p = InterpolatedUnivariateSpline(z_list[::-1],(a_list*integ_result[:,0])[::-1],k=2,ext=2)
        else:
            self.G_p=G_in

        #self.wmatch_overwride=wmatch_overwride
        #get matching factors for dark energy via the casarini arXiv:1601.07230v3 inspired method 
        #TODO create a separate power spectrum object to handle this stuff
#        if not self.wmatch_overwride:
#            if self.de_model is 'constant_w':
#                self.matched_ws = self.w_interp(self.z_grid)
#                self.matched_growth = np.zeros(self.z_grid.size)+1.
#            elif self.wmatch is None:
#                warn('no wmatcher given to cosmopie but dark energy model is not constant_w '+str(self.de_model)+' will simply use linear growth factor')
#                self.matched_ws = self.w_interp(self.z_grid)
#                self.matched_growth = np.zeros(self.z_grid.size)+1.
#            else:
#                #TODO feeding self to match_w seems dangerous
#                self.matched_ws = self.wmatch.match_w(self,self.z_grid)
#                self.matched_growth = self.wmatch.match_growth(self,self.z_grid,matched_ws)
#
#            self.matched_ws_interp = InterpolatedUnivariateSpline(self.z_grid,self.matched_ws)
#            self.matched_growth_interp = InterpolatedUnivariateSpline(self.z_grid,self.matched_growth)
#        else:
#            self.matched_ws = None
#            self.matched_growth = None
#            self.matched_ws_interp = None
#            self.matched_growth_interp = None
        if (P_lin is None or k is None) and needs_power:
            #self.k,self.P_lin=camb_power.camb_pow(self.cosmology,camb_params=camb_params)
            self.P_lin = mps.MatterPower(self)
            self.k=self.P_lin.k
        else:
            self.P_lin=P_lin
            self.k=k    

        if not silent: 
            print "cosmopie "+str(id(self))+": finished initialization"        
                
    def Ez(self,z):
        zp1=z + 1
        #need to use interpolating function for a dependence of OmegaL term because could be time dependent
        return np.sqrt(self.Omegam*zp1**3 + self.Omegar*zp1**4 + self.Omegak*zp1**2 + self.OmegaL*self.de_mult(z)) 
     
    def H(self,z):
        return self.H0*self.Ez(z)  
    
    def dH_da(self,z):
        # the derivative of H with respect to a 
        #TODO check this if anything actually uses it
        return -(1+z)*self.H0/2.*(3.*self.Omegam_z(z)+4.*self.Omegar_z(z)+2.*self.Omegak_z(z)+3.*(1.+self.w_interp(z))*self.OmegaL_z(z))                
#                return -(1+z)**2*self.H0/2./self.Ez(z)*(3*self.Omegam*zp1**2 +4*self.Omegar*zp1**3  +2*self.Omegak*zp1 )
    
    # distances, volumes, and time
    # -----------------------------------------------------------------------------
    def D_comov(self,z):
        # the line of sight comoving distance 
        I = lambda zp : 1/self.Ez(zp)
        return self.DH*quad(I,0,z)[0]
    
    def D_comov_A(self,z):
        # the comoving angular diameter distance, as defined in Bartelmann and Schneider arXiv:astro-px/9912508v1
        if self.K ==0:
            return self.D_comov(z)
        elif self.K>0:
            sqrtK = np.sqrt(abs(self.K))
            return 1./sqrtK*np.sin(sqrtK*self.D_comov(z))
        else:
            sqrtK = np.sqrt(abs(self.K))
            return 1./sqrtK*np.sinh(sqrtK*self.D_comov(z))

        #TODO IMPT switch to sinh/handle k!=0 correctly
    
    def D_comov_dz(self,z):
        return self.DH/self.Ez(z) 
            
    def D_comov_T(self,z):
        # the transverse comoving distance 
        
        if (self.Omegak==0):   
                return self.D_comov(z)
        
        if (self.Omegak > 0):
                sq=np.sqrt(self.Omegak)
                return self.DH/sq*np.sinh(sq*self.D_comov(z)/self.DH)
        
        if (self.Omegak < 0): 
                sq=np.sqrt(self.Omegak)
                return self.DH/sq*np.sin(sq*self.D_comov(z)/self.DH)
                    
            
    def D_A(self,z):
        # angular diameter distance
        return self.D_comov_T(z)/(1+z)
            
    def D_L(self,z):
        # luminosity distance 
        return (1+z)**2*self.D_A(z)
    
    def DV(self,z):
        # comoving volume element, with out the d\Omega 
        # dV/dz
        return self.DH*(1+z)**2*self.D_A(z)**2/self.Ez(z)
            
    def D_A_array(self,z):

        d=np.zeros_like(z)
        for i in xrange(z.size):
            d[i]=self.D_A(z[i])
        return d 

    def D_c_A_array(self,z):

        d=np.zeros_like(z)
        for i in xrange(z.size):
            d[i]=self.D_comov_A(z[i])
        return d 
    
    def D_c_array(self,z):

        d=np.zeros_like(z)
        for i in xrange(z.size):
            d[i]=self.D_comov(z[i])
        return d 
        
    
    def look_back(self,z):
        f1=3.086e19  # conversion  Mpc to km 
        f2=3.154e7   # conversion seconds to years
        I = lambda z : 1/self.Ez(z)/(1+z)
        return self.tH*quad(I,0,z)[0]*f1/f2
    # -----------------------------------------------------------------------------   
    
    # Growth functions
    # -----------------------------------------------------------------------------
#         def G(self,z):
#                 # linear Growth factor (Eqn. 7.77 in Dodelson)
#                 # 1 + z = 1/a 
#                 # G = 5/2 Omega_m H(z)/H_0 \int_0^a da'/(a'H(a')/H_0)^3
#                 # Omega_m=Omega_m_z/a^3/H(z)^2
#                 a=1/float(1+z)
#                 def Integrand(ap):
#                     zp=1/float(ap)-1
#                    
#                     denominator=float((ap*self.H(zp))**3)
#                     
#                     return 1/denominator 
#         
#                 return 5/2.*self.Omegam_z(z)*a**2*self.H(z)**3*quad(Integrand,1e-5,a)[0]
#                 #return 5/2.*self.Omegam_z(z)*a**2*self.H(z)**3*romberg(Integrand,eps,a)

    #TODO support vector z        
    def G(self,z):
        #if self.precompute:
        return self.G_p(z)
        #else:
        #    integrand = lambda zp : (1+zp)*self.H0**3/self.H(zp)**3
            #return 2.5*self.Omegam/self.H0*self.H(z)*romberg(integrand,z,1e4)
        #    return 2.5*self.Omegam/self.H0*self.H(z)*quad(integrand,z,1e4)[0]
    #solve the linear diffential equation to get G
    #TODO precompute
#    def G_Direct(self,z):
#        a_step=0.001
#        a_min=0.001
#        a_max=1.+a_step
#        a_list = np.arange(a_min,a_max,a_step)
#        z_list = 1./a_list-1.
#        dp_multiplier = -1./a_list*(7./2.*self.Omegam_z(z_list)+3*self.Omegar_z(z_list)+(7./2.-3./2.*self.w_interp(z_list))*self.OmegaL_z(z_list)+4.*self.Omegak_z(z_list))
#        d_multiplier = -1./a_list**2*(self.Omegar_z(z_list)+3./2.*(1.-self.w_interp(z_list))*self.OmegaL_z(z_list)+2*self.Omegak_z(z_list))
#        #TODO check splines
#        dp_mult_interp = InterpolatedUnivariateSpline(a_list,dp_multiplier)
#        d_mult_interp = InterpolatedUnivariateSpline(a_list,d_multiplier)
#        d_ics = np.array([1.,0.])
#        d_args = (np.array([d_mult_interp,dp_mult_interp]),)
#        def _d_evolve_eqs(ys,a,g_args):
#            yps = np.zeros(2)
#            yps[0]=ys[1]
#            yps[1]=g_args[0](a)*ys[0]+g_args[1](a)*ys[1]
#            return yps
#        integ_result = odeint(_d_evolve_eqs,d_ics,a_list,d_args)
#        g_interp = InterpolatedUnivariateSpline(z_list[::-1],(a_list*integ_result[:,0])[::-1],k=2,ext=2)
#        return g_interp(z),g_interp,d_mult_interp,dp_mult_interp,integ_result
        
         

#         def G(self,z):
#                 integrand = lambda zp : 1/(1 + zp)**2/self.H(z)**3 
#                 return 2.5*self.Omegam_z(z)/(1+z)**2*self.H(z)**3*quad(integrand, z,1e5)[0]
    
    def G_norm(self,z):
        # the normalized linear growth factor
        # normalized so the G(0) =1 
        G_0=self.G(0.)
        return self.G(z)/G_0
#        #G does not currently support vector inputs unless precomputed TODO could fix that
#        if (not self.precompute) and isinstance(z,np.ndarray):
#            result = np.zeros(z.size)
#            for i in xrange(z.size):
#                result[i] = self.G_norm(z[i])
#        else:
#            return self.G(z)/G_0 
            
            
    def log_growth(self,z):
        # using equation 3.2 from Baldauf 2015 
        a=1/(1+z)
        print 'what I think it is', a/self.H(z)*self.dH_da(z) + 5/2.*self.Omegam*self.G_norm(0)/self.H(z)**2/a**2/self.G_norm(z)
        return -3/2.*self.Omegam/a**3*self.H0**2/self.H(z)**2 + 1/self.H(z)**2/a**2/self.G_norm(z)
    #TODO probably obsolete        
#    def G_array(self,z):
#        # the normalized linear growth factor 
#        # for an array 
#        if isinstance(z,float):
#            return np.array([self.G_norm(z)])
#        elif self.precompute and isinstance(z,np.ndarray):
#            return G_norm(z)
#        else:
#            result=np.zeros(z.size)
#            for i in xrange(z.size):
#                result[i]=self.G_norm(z[i])
#            return result 
                    
    # ----------------------------------------------------------------------------
     
    # halo and matter stuff 
    # -----------------------------------------------------------------------------   
    def delta_c(self,z):
        # critical threshold for spherical collapse, as given 
        # in the appendix of NFW 1997 
        #TODO fitting formula, probably not appropriate for the code anymore
        #TODO why no z dependence anywhere?
        A=0.15*(12.*pi)**(2/3.)
        
        if ( (self.Omegam ==1) & (self.OmegaL==0)):
            d_crit=A
        elif ( (self.Omegam < 1) & (self.OmegaL ==0)):
            d_crit=A*self.Omegam**(0.0185)
        elif ( (self.Omegam + self.OmegaL)==1.0):
            d_crit=A*self.Omegam**(0.0055)
        else:      
            d_crit=A*self.Omegam**(0.0055)
            warn('inexact equality to 1~='+str(self.Omegam)+"+"+str(self.OmegaL)+"="+str(self.OmegaL+self.Omegam))
        d_c=d_crit#/self.G_norm(z)
        return d_c
            
    def delta_v(self,z):
        # over density for virialized halo
        
        A=178.0
        if ( (self.Omegam_z(z) ==1) & (self.OmegaL_z(z)==0)):
            d_v=A
        if ( (self.Omegam_z(z) < 1) & (self.OmegaL_z(z) ==0)):
            d_v=A/self.Omegam_z(z)**(0.7)
        if ( (self.Omegam_z(z) + self.OmegaL_z(z))==1.0):
            d_v=A/self.Omegam_z(z)**(0.55)
                   
        return d_v/self.G_norm(z)
            
            
    def nu(self,z,mass):
        # calculates nu=(delta_c/sigma(M))^2
        # delta_c is the overdensity for collapse 
        return (self.delta_c(z)/self.sigma_m(mass,z))**2
    
    def sigma_m(self,mass,z):
        # RMS power on a scale of R(mass)
        # rho=mass/volume=mass
        R=3/4.*mass/self.rho_bar(z)/pi
        R=R**(1/3.)           
        return self.sigma_r(z,R)
    
    def sigma_r(self,z,R):
        # returns RMS power on scale R
        # sigma^2(R)= \int dlogk/(2 pi^2) W^2 P(k) k^3 
        # user needs to adjust for growth factor upon return 
        if self.P_lin is None:
            raise ValueError('You need to provide a linear power spectrum and k to get sigma valeus')
        if self.k is None:
            raise ValueError('You need to provide a linear power spectrum and k to get sigma valeus')
        if isinstance(R,np.ndarray):
            kr = np.outer(self.k,R)
        else:
            kr = self.k*R
        W=3.0*(np.sin(kr)/kr**3-np.cos(kr)/kr**2)
        #P=self.G_norm(z)**2*self.P_lin
        #TODO should be z dependence?
        P=self.P_lin.linear_power(np.array([0.]))[:,0]
        I=trapz((W*W).T*P*self.k**3,np.log(self.k))/2./pi**2
        
        return np.sqrt(I)
            
    def Omegar_z(self,z):
        return self.Omegar*(1.+z)**4/self.Ez(z)**2

    # densities as a function of redshift 
    def Omegam_z(self,z):
        # matter density as a function of redshift 
        return self.Omegam*(1.+z)**3/self.Ez(z)**2

    def OmegaL_z(self,z):
        # dark energy density as function of redshift 
        #TODO check de_mult
        return self.OmegaL*self.de_mult(z)/self.Ez(z)**2

    def Omegak_z(self,z):
        return 1-self.Omegam_z(z)-self.OmegaL_z(z)-self.Omegar_z(z)
    
    def Omega_tot(self,z):
        return self.Omegak_z(z) + self.OmegaL_z(z) + self.Omegam_z(z)+self.Omegar_z(z)
    
    def rho_bar(self,z):
        # return average density in units of solar mass and h^2 
        return self.rho_crit(z)*self.Omegam_z(z)
    #critical density should be about 2.77536627*10^11 h^2 M_sun/Mpc^-3 at z=0 according to pdg table 
    def rho_crit(self,z):
        # return critical density in units of solar mass and h^2 
        factor=1e12/self.M_sun*self.pc           
        #print 'rho crit [g/cm^3] at z =', z, 3*self.H(z)**2/8./pi/self.GN*1e9/(3.086*10**24)**2/10**2
        return 3*self.H(z)**2/8./pi/self.GN*factor/self.h**2
    
    def get_P_lin(self):
        return self.k, self.P_lin
       
    # -----------------------------------------------------------------------------
    
#get cosmology with the required value set without enforcing consistency. Use defaults if unknown.
#def get_complete_cosmology(cosmo_old,required=defaults.cosmo_consistency_params['required'],cosmo_d = defaults.cosmology):
#    cosmo_new = {}
#
#    for param in required:
#        if param in cosmo_old:
#            continue
#        if param=='h':
#            if 'H0' in cosmo_old: 
#               cosmo_new['h']=cosmo_old['H0']/100.
#            else: 
#               cosmo_new['h']=cosmo_d['h']
#        elif param=='H0':
#            if 'h' in cosmo_old: 
#               cosmo_new['H0']=cosmo_old['h']*100.
#            else: 
#               cosmo_new['H0']=cosmo_d['H0']
#        elif param=='Omegac':
            
#remove all nonessential attributes for a cosmology unless they are in overwride list,
#leaving only the necessary elements of the parameter space
#possible parameter spaces (with required elements) are:
# 'jdem':parameter space proposed in joint dark energy mission figure of merit working group paper, arxiv:0901.0721v1
# 'lihu':parameter space used in Li, Hu & Takada 2013, arxiv:1408.1081v2
P_SPACES ={'jdem': ['ns','Omegamh2','Omegabh2','Omegakh2','OmegaLh2','dGamma','dM','LogG0','LogAs'],
            'lihu' : ['ns','Omegach2','Omegabh2','Omegakh2','h','LogAs'],
            'basic': ['ns','Omegamh2','Omegabh2','Omegakh2','h','sigma8']}
DE_METHODS = {'constant_w':['w'],
              'w0wa'      :['w','w0','wa'],
              'jdem'      :['ws36'],
              'grid_w'    :['ws']}
#parameters guaranteed not to affect linear growth factor (G_norm)
GROW_SAFE = ['ns','LogAs','sigma8']
def strip_cosmology(cosmo_old,p_space,overwride=[]):
    cosmo_new = cosmo_old.copy()
    if p_space in P_SPACES:
        param_need = P_SPACES[p_space]
        #delete unneeded values
        for key in cosmo_old:
            if key not in P_SPACES[p_space] and key not in overwride:
                cosmo_new.pop(key,None)
        #insert 0 for missing values
        #TODO consider better defaults
        for req in P_SPACES[p_space]:
            if req not in cosmo_new:
                cosmo_new[req] = defaults.cosmology_jdem[req]

    else:
        raise ValueError('unrecognized p_space \''+str(p_space)+'\'')


    #mark this cosmology with its parameter space
    cosmo_new['p_space']=p_space
    return cosmo_new
         
#use relations to add all known derived parameters to a cosmology starting from its parameter space, including:
#Omegam,Omegac,Omegab,Omegak,OmegaL,Omegar
#Omegamh2,Omegach2,Omegabh2,Omegakh2,OmegaLh2,Omegarh2
#H0,h
#LogAs,As,sigma8 #TODO check
def add_derived_parameters(cosmo_old,p_space=None,safe_sigma8=False):
        cosmo_new = cosmo_old.copy()
        if p_space is None:
            p_space = cosmo_old.get('p_space')
        if p_space is 'jdem':
            cosmo_new['Omegach2'] = cosmo_old['Omegamh2']-cosmo_old['Omegabh2']  
            cosmo_new['h'] = np.sqrt(cosmo_old['Omegamh2']+cosmo_old['OmegaLh2']+cosmo_old['Omegakh2'])
            cosmo_new['H0'] = cosmo_new['h']*100.
            cosmo_new['Omegab'] = cosmo_old['Omegabh2']/cosmo_new['h']**2
            cosmo_new['Omegac'] = cosmo_new['Omegach2']/cosmo_new['h']**2
            cosmo_new['Omegam'] = cosmo_old['Omegamh2']/cosmo_new['h']**2
            cosmo_new['OmegaL'] = cosmo_old['OmegaLh2']/cosmo_new['h']**2
            cosmo_new['Omegak'] = cosmo_old['Omegakh2']/cosmo_new['h']**2
            #define omega radiation
            cosmo_new['Omegar'] = 1.-cosmo_new['Omegam']-cosmo_new['OmegaL']-cosmo_new['Omegak']
            cosmo_new['Omegarh2'] = cosmo_new['Omegar']*cosmo_new['h']**2
     
            cosmo_new['As']=np.exp(cosmo_old['LogAs'])
            if cosmo_old.get('skip_sigma8') is None:
                cosmo_new['sigma8'] = camb_power.camb_sigma8(cosmo_new)
            elif safe_sigma8:
                cosmo_new['sigma8'] = cosmo_old['sigma8']
            else:
                cosmo_new['sigma8'] = None
        elif p_space is 'lihu':
            cosmo_new['Omegamh2'] = cosmo_old['Omegach2']+cosmo_old['Omegabh2']  
            cosmo_new['H0'] = cosmo_old['h']*100.
            cosmo_new['Omegab'] = cosmo_old['Omegabh2']/cosmo_new['h']**2
            cosmo_new['Omegac'] = cosmo_old['Omegach2']/cosmo_new['h']**2
            cosmo_new['Omegam'] = cosmo_new['Omegamh2']/cosmo_new['h']**2
            cosmo_new['Omegak'] = cosmo_old['Omegakh2']/cosmo_new['h']**2
            cosmo_new['Omegar'] = 0. #I think just set this to 0?

            cosmo_new['OmegaL'] = 1.-cosmo_new['Omegam']-cosmo_new['Omegar']-cosmo_new['Omegak']
            cosmo_new['OmegaLh2'] = cosmo_new['OmegaL']*cosmo_new['h']**2
            cosmo_new['Omegarh2'] = cosmo_new['Omegar']*cosmo_new['h']**2

            cosmo_new['As']=np.exp(cosmo_old['LogAs'])
            if cosmo_old.get('skip_sigma8') is  None:
                cosmo_new['sigma8'] = camb_power.camb_sigma8(cosmo_new)
            elif safe_sigma8:
                cosmo_new['sigma8'] = cosmo_old['sigma8']
            else:
                cosmo_new['sigma8'] = None
        elif p_space is 'basic':
            cosmo_new['Omegach2'] = cosmo_old['Omegamh2']-cosmo_old['Omegabh2']  
            cosmo_new['H0'] = cosmo_old['h']*100.
            cosmo_new['Omegab'] = cosmo_old['Omegabh2']/cosmo_new['h']**2
            cosmo_new['Omegac'] = cosmo_new['Omegach2']/cosmo_new['h']**2
            cosmo_new['Omegam'] = cosmo_old['Omegamh2']/cosmo_new['h']**2
            cosmo_new['Omegak'] = cosmo_old['Omegakh2']/cosmo_new['h']**2
            cosmo_new['Omegar'] = 0. #I think just set this to 0?

            cosmo_new['OmegaL'] = 1.-cosmo_new['Omegam']-cosmo_new['Omegar']-cosmo_new['Omegak']
            cosmo_new['OmegaLh2'] = cosmo_new['OmegaL']*cosmo_new['h']**2
            cosmo_new['Omegarh2'] = cosmo_new['Omegar']*cosmo_new['h']**2
            
            #don't know how to relate sigma8
           # cosmo_new['As']=np.exp(cosmo_old['LogAs'])

        elif p_space is 'overwride':
            #option which does nothing
            pass

        else:
            raise ValueError('unrecognized p_space \''+str(p_space)+'\'')

        cosmo_new['p_space'] = p_space
        return cosmo_new
if __name__=="__main__": 

        C=CosmoPie(cosmology=defaults.cosmology)
        z=3.5
        z=.1
        print('Comoving distance',C.D_comov(z))
        print('Angular diameter distance',C.D_A(z))
        print('Luminosity distance', C.D_L(z)) 
        print('Growth factor', C.G(z)) 
        print('Growth factor for really small z',C.G(1e-10))
        z=0.0
        print('logrithmic growth factor', C.log_growth(z))
        print('compare logrithmic growth factor to approxiamtion', C.Omegam**(-.6), C.Omegam)
        print('critical overdensity ',C.delta_c(0)  ) 
                
        z=np.linspace(0,5,80) 
        D1=np.zeros(80)
        D2=np.zeros(80)
        D3=np.zeros(80)
        for i in xrange(80):
                D1[i]=C.D_A(z[i])/C.DH
                D2[i]=C.D_L(z[i])/C.DH
                D3[i]=C.DV(z[i])/C.DH
                
        import matplotlib.pyplot as plt
        
        ax=plt.subplot(121)
        ax.set_xlabel(r'$z$', size=20)
        
        ax.plot(z, D1, label='Angular Diameter distance [Mpc]') 
        #ax.plot(z, D2, label='Luminosity distance [Mpc]') 
        plt.grid()
        
        ax=plt.subplot(122)
        ax.set_ylim(0,1)
        ax.plot(z, D3/C.DH**3, label='Angular Diameter distance [Mpc]') 
        #ax.plot(z, D2, label='Luminosity distance [Mpc]') 
        plt.grid()
        
        plt.show()
        
           
        
        
