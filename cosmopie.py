"""
        Computes and manages various parameters associated with a given cosmology
        Joseph E. McEwen (c) 2016
        Matthew C. Digman
        mcewen.24@osu.edu
"""

from warnings import warn
from scipy.integrate import odeint
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline
import numpy as np

from algebra_utils import trapz2
import matter_power_spectrum as mps

import defaults
import camb_power

eps=np.finfo(float).eps

class CosmoPie(object):
    """stores and calculates various parameters for a cosmology"""
    #TODO convergence test a grid values
    #TODO reduce possibility of circular reference to MatterPower
    def __init__(self,cosmology=defaults.cosmology, P_lin=None, k=None,p_space=defaults.cosmopie_params['p_space'],needs_power=False,camb_params=None,a_step=0.001,G_in=None,G_safe=False,silent=False):
        """
        Set up for cosmological parameters in input cosmology.
        Inputs:
        cosmology: a dictionary of cosmological parameters,
        P_lin: a MatterPower object. Optional.
        needs_power: If True and input P_lin is None, call camb to get MatterPower.
        p_space: The cosmological parameter space. available options listed in P_SPACES.
        G_in: Function to get linear growth factor. Optional.
        G_safe: If True and G_in is not None, use G_in instead of solving differential equation.
        silent: If True, less print statements
        """


        # default to Planck 2015 values
        self.silent=silent
        if not silent:
            print "cosmopie "+str(id(self))+": begin initialization"

        #define parameterization
        self.p_space=p_space
        #fill out cosmology
        self.cosmology = cosmology.copy()
        self.cosmology = add_derived_pars(self.cosmology,self.p_space)


        self.Omegabh2 = self.cosmology['Omegabh2']
        self.Omegach2 = self.cosmology['Omegach2']
        self.Omegamh2 = self.cosmology['Omegamh2']
        self.OmegaL   = self.cosmology['OmegaL']
        self.Omegam   = self.cosmology['Omegam']
        self.ns =self.cosmology['ns']
        self.H0       = self.cosmology['H0']
        #self.sigma8   = self.cosmology['sigma8']
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

        if self.de_model=='constant_w':
            #self.de_mult = InterpolatedUnivariateSpline(z_de,(z_de+1.)**(3.*(1.+self.cosmology['w'])),k=1,ext=2)
            self.de_mult = self.de_mult_const
            #ws = np.zeros(z_de.size)+self.cosmology['w']
            #self.w_interp = InterpolatedUnivariateSpline(z_de,ws,k=1,ext=2)
            self.w_interp = self.de_w_const
        elif self.de_model=='w0wa':
            #Chevallier-Polarski-Linder model, solution can be found in ie arXiv:1605.01475
            #self.de_mult=InterpolatedUnivariateSpline(z_de,np.exp(-3.*self.cosmology['wa']*z_de/(1.+z_de))*(1.+z_de)**(3.*(1.+self.cosmology['w0']+self.cosmology['wa'])),k=1,ext=2)
            self.de_mult = self.de_mult_w0wa
            #ws = self.cosmology['w0']+(1.-1./(1.+z_de))*self.cosmology['wa']
            #self.w_interp = InterpolatedUnivariateSpline(z_de,ws,k=1,ext=2)
            self.w_interp = self.de_w_w0wa
#        elif self.de_model=='grid_w':
#            #TODO storing grid in cosmology dangerous, make this work if anything actually uses it
#            #cf ie https://ned.ipac.caltech.edu/level5/March08/Frieman/Frieman2.html#note1, arXiv:1605.01475
#            z_de = self.cosmology['zs_de']
#            de_integrand = (1.+self.cosmology['ws'])/(1.+z_de)
#            de_exponent = cumtrapz(de_integrand,z_de,initial=0.)
#            if z_de[0]<0:
#                itr_first = np.where(z_de>=0.)[0][0]
#                val_0 = z_de[itr_first-1]-z_de[itr_first-1]*(de_mults_in[itr_first]-de_mults_in[itr_first-1])/(z_de[itr_first-1]-z_de[itr_first])
#                de_exponent-=val_0
#            de_mults_in = np.exp(3.*de_exponent)
#            #self.de_mult = InterpolatedUnivariateSpline(self.cosmology['zs_de'],de_mults_in,k=2,ext=2) #k=2 so smooths some, check if this is a good idea.
#            self.w_interp = InterpolatedUnivariateSpline(self.cosmology['zs_de'],self.cosmology['ws'],k=2,ext=2)
        elif self.de_model=='jdem':
            #piecewise constant approximation over 36 values with z=0.025 spacing

            #z grid for dark energy interpolation
            a_de = np.arange(1.,a_step,-a_step)
            z_de = 1./a_de-1.
            z_de[0] = 0. #enforce to avoid numeric precision issue

            ws_in = np.array([self.cosmology['ws36_'+str(i)] for i in range(0,36)])
            zs_max = 0.025*np.arange(1,37)/(1-0.025*np.arange(1,37))
            zs_min = 0.025*np.arange(0,36)/(1-0.025*np.arange(0,36))
            #default to value of w over edge
            #TODO enforce consistent use of default w, ie defaulting so at higher z just use last z given
            ws = np.full(z_de.size,self.cosmology['w'])
            itr = 0
            for i in xrange(0,z_de.size):
                if z_de[i]>=zs_max[itr]:
                    itr+=1
                    if itr>=zs_max.size:
                        break
                if z_de[i]>=zs_min[itr] and z_de[i]<zs_max[itr]:
                    ws[i] = ws_in[itr]

            self.w_interp_jdem = interp1d(z_de,ws)
            def de_w_jdem(z):
                z = np.asanyarray(z)
                if not (np.any(z>=9.) or np.any(z<0)):
                    result = self.w_interp_jdem(z)
                else:
                    result = np.zeros_like(z)
                    result[z>=9.] = self.cosmology['w']
                    result[z<0.] = self.cosmology['w']
                    result[(z>=0.)*(z<9.)] = self.w_interp_jdem(z[(z>=0.)*(z<9.)])
                return result
            #self.w_interp = InterpolatedUnivariateSpline(z_de,ws,k=1,ext=2)
            self.w_interp = de_w_jdem
            #self.w_interp = self.w_interp_jdem
            def de_exp_const_w(z,w):
                return np.log((z+1.)**(3.*(1.+w)))/3.
            de_step = de_exp_const_w(zs_max,ws_in)-de_exp_const_w(zs_min,ws_in)
            de_step_sum = np.full(de_step.size+1,0.)
            de_step_sum[1::] += np.cumsum(de_step)
           #def de_mult_jdem2(z):
           #    for i in xrange(0,z)
            de_exponent_true = de_exp_const_w(z_de,ws)
            for itr in xrange(0,36):
                de_exponent_true[(zs_min[itr]<=z_de)*(z_de<zs_max[itr])]+=de_step_sum[itr]-de_exp_const_w(zs_min[itr],ws_in[itr])
            de_exponent_true[z_de>=zs_max[-1]]+=de_step_sum[-1]-de_exp_const_w(zs_max[-1],self.cosmology['w'])
            self.de_true_interp = InterpolatedUnivariateSpline(z_de,np.exp(3.*de_exponent_true),k=3,ext=2)
 #           self.de_true_interp = interp1d(z_de,np.exp(3.*de_exponent_true))
            def de_mult_jdem(z):
                z = np.asanyarray(z)
                if not (np.any(z<0) or np.any(z>=9.)):
                    return self.de_true_interp(z)
                result = np.zeros_like(z)
                result[z<0.] = (z[z<0.]+1.)**(3.*(1.+self.cosmology['w']))
                result[(z>=0.)*(z<9.)] = self.de_true_interp(z[(z>=0.)*(z<9.)])
                result[z>=9.] = np.exp(3.*(de_exp_const_w(z[z>=9.],self.cosmology['w'])-de_exp_const_w(9.,self.cosmology['w'])+np.log(self.de_true_interp(9.))/3.))
                return result
 #           de_integrand = (1.+ws)/(1.+z_de)
 #           #self.de_integrand_interp = InterpolatedUnivariateSpline(z_de,de_integrand,k=3,ext=2)
 #           #value of exponent should be 0 at z=0, correct for z_de values <0
 #           #TODO do this integral piecewise exactly
 #           de_exponent = cumtrapz(de_integrand,z_de,initial=0.)
 #           de_exponent -= de_exponent[a_extra_de*2]

 #           self.de_exponent_jdem_interp =  InterpolatedUnivariateSpline(z_de,de_exponent,k=1)
 #           #extrapolate using constant
 #           def de_exponent_jdem(z):
 #               z = np.asanyarray(z)
 #               result = np.zeros_like(z)
 #               result[z<0.] = (z[z<0.]+1.)**(3.*(1.+self.cosmology['w']))
 #               result[z>=9.] = self.de_exponent_jdem_interp(9.)+np.log((z[z>=9.]+1.)**(3.*(1.+self.cosmology['w'])))/3.-np.log((9.+1.)**(3.*(1.+self.cosmology['w'])))/3.
 #               result[(0.<=z)*(z<9.)] = self.de_exponent_jdem_interp(z[(0<=z)*(z<9.)])
 #               return result
 #           def de_mult_jdem(z):
 #               z = np.asanyarray(z)
 #               return np.exp(3.*de_exponent_jdem(z))


 #           de_mults_in = np.exp(3.*de_exponent)
 #           de_mults_in+=1.-de_mults_in[a_extra_de*2]
 #           if z_de[0]<0:
 #               itr_first = np.where(z_de>=0.)[0][0]
 #               val_0 = z_de[itr_first-1]-z_de[itr_first-1]*(de_mults_in[itr_first]-de_mults_in[itr_first-1])/(z_de[itr_first-1]-z_de[itr_first])
 #               de_mults_in =de_mults_in-val_0
            #self.de_mult = integ_funct
            #TODO function instead of spline would be better
 #           self.de_mult = InterpolatedUnivariateSpline(z_de,de_mults_in,k=1) #smoothing may help differential equations
 #           self.de_mult_true = self.de_true_interp
 #           self.de_mult = de_mult_jdem
            #self.de_mult=de_mult_jdem
            self.de_mult = de_mult_jdem
 #           self.de_mult = self.de_true_interp
        else:
            raise ValueError('unrecognized dark energy model \''+str(self.de_model)+'\'')

        self.camb_params = camb_params


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


        #precompute the normalization factor from the differential equation
        self.a_step=a_step
        self.G_safe=G_safe

        a_min=a_step
        a_max=1.
        self.a_grid = np.arange(a_max,a_min-a_step/10.,-a_step)
        #self.a_grid2 = np.arange(a_min,a_max,a_step)[::-1]
        self.z_grid = 1./self.a_grid-1.

        if not G_safe:
#            dp_multiplier = -1./self.a_grid*(7./2.*self.Omegam_z(self.z_grid)+3*self.Omegar_z(self.z_grid)+(7./2.-3./2.*self.w_interp(self.z_grid))*self.OmegaL_z(self.z_grid)+4.*self.Omegak_z(self.z_grid))
#            d_multiplier = -1./self.a_grid**2*(self.Omegar_z(self.z_grid)+3./2.*(1.-self.w_interp(self.z_grid))*self.OmegaL_z(self.z_grid)+2*self.Omegak_z(self.z_grid))
            #TODO check splines

            def _dp_mult_interp(a):
                z = 1./a-1.
                return -1./a*(7./2.*self.Omegam_z(z)+3*self.Omegar_z(z)+(7./2.-3./2.*self.w_interp(z))*self.OmegaL_z(z)+4.*self.Omegak_z(z))
            def _d_mult_interp(a):
                z = 1./a-1.
                return -1./a**2*(self.Omegar_z(z)+3./2.*(1.-self.w_interp(z))*self.OmegaL_z(z)+2*self.Omegak_z(z))
#            dp_mult_interp = InterpolatedUnivariateSpline(self.a_grid[::-1],dp_multiplier[::-1])
#            d_mult_interp = InterpolatedUnivariateSpline(self.a_grid[::-1],d_multiplier[::-1])
#            self.dp_mult_interp = dp_mult_interp
#            self.d_mult_interp = d_mult_interp
            d_ics = np.array([1.,0.])
            d_args = (np.array([_d_mult_interp,_dp_mult_interp]),)
            def _d_evolve_eqs(ys,a,g_args):
                yps = np.zeros(2)
                yps[0]=ys[1]
                yps[1]=g_args[0](a)*ys[0]+g_args[1](a)*ys[1]
                return yps
            #drop ghost cell
            integ_result = odeint(_d_evolve_eqs,d_ics,self.a_grid[::-1],d_args)
            self.G_p = InterpolatedUnivariateSpline(self.z_grid,(self.a_grid*integ_result[:,0][::-1]),k=2,ext=2)
        else:
            self.G_p=G_in

        if (P_lin is None or k is None) and needs_power:
            self.P_lin = mps.MatterPower(self,camb_params=self.camb_params)
            self.k=self.P_lin.k
        else:
            self.P_lin=P_lin
            self.k=k

        if not silent:
            print "cosmopie "+str(id(self))+": finished initialization"


    def de_mult_const(self,z):
        """multiplier for de term if w constant"""
        return (z+1.)**(3.*(1.+self.cosmology['w']))
    def de_w_const(self,z):
        """w function for de term if w constant"""
        return np.full_like(z,self.cosmology['w'])
    def de_mult_w0wa(self,z):
        """multiplier for de term if w0wa"""
        return np.exp(-3.*self.cosmology['wa']*z/(1.+z))*(1.+z)**(3.*(1.+self.cosmology['w0']+self.cosmology['wa']))
    def de_w_w0wa(self,z):
        """w(z) for de term if w0wa"""
        return self.cosmology['w0']+(1.-1./(1.+z))*self.cosmology['wa']

    def Ez(self,z):
        """
        Get E(z)=H(z)/H0
        """
        zp1=z + 1
        #need to use interpolating function for a dependence of OmegaL term because could be time dependent if w is not constant
        return np.sqrt(self.Omegam*zp1**3 + self.Omegar*zp1**4 + self.Omegak*zp1**2 + self.OmegaL*self.de_mult(z))

    def H(self,z):
        """
        Hubble parameter H(z)
        """
        return self.H0*self.Ez(z)

    def dH_da(self,z):
        """
        Derivative of H with respect to a
        """
        #TODO check this if anything actually uses it
        return -(1+z)*self.H0/2.*(3.*self.Omegam_z(z)+4.*self.Omegar_z(z)+2.*self.Omegak_z(z)+3.*(1.+self.w_interp(z))*self.OmegaL_z(z))

    # distances, volumes, and time
    # -----------------------------------------------------------------------------
    def D_comov(self,z):
        """the line of sight comoving distance"""
        #I = lambda zp : 1/self.Ez(zp)
        #return self.DH*quad(I,0,z)[0]
        I = lambda y,zp : 1/self.Ez(zp)
        #make sure value at 0 is 0 to fix initial conditions
        z_use = np.hstack([0.,z])
        d = self.DH*odeint(I,0.,z_use,atol=10e-20,rtol=10e-10)[1::,0]
        if np.isscalar(z):
            return d[0]
        else:
            return d

    def D_comov_A(self,z):
        """the comoving angular diameter distance, as defined in ie Bartelmann and Schneider arXiv:astro-px/9912508v1"""
        if self.K ==0:
            return self.D_comov(z)
        elif self.K>0:
            sqrtK = np.sqrt(abs(self.K))
            return 1./sqrtK*np.sin(sqrtK*self.D_comov(z))
        else:
            sqrtK = np.sqrt(abs(self.K))
            return 1./sqrtK*np.sinh(sqrtK*self.D_comov(z))


    def D_comov_dz(self,z):
        """Integrand of D_comov"""
        return self.DH/self.Ez(z)

    def D_comov_T(self,z):
        """Transverse comoving distance"""
        if self.Omegak > 0:
            sq=np.sqrt(self.Omegak)
            return self.DH/sq*np.sinh(sq*self.D_comov(z)/self.DH)
        elif self.Omegak < 0:
            sq=np.sqrt(self.Omegak)
            return self.DH/sq*np.sin(sq*self.D_comov(z)/self.DH)
        else:
            return self.D_comov(z)

    def D_A(self,z):
        """angular diameter distance"""
        return self.D_comov_T(z)/(1+z)

    def D_L(self,z):
        """luminosity distance"""
        return (1+z)**2*self.D_A(z)

    def DV(self,z):
        r""" comoving volume element, with out the d\Omega
         dV/dz"""
        return self.DH*(1+z)**2*self.D_A(z)**2/self.Ez(z)
    # -----------------------------------------------------------------------------
    # Growth functions
    # -----------------------------------------------------------------------------
    def G(self,z):
        """Linear growth factor"""
        return self.G_p(z)

    def G_norm(self,z):
        """linear growth factor normalized so the G(z=0)=1"""
        G_0=self.G(0.)
        return self.G(z)/G_0


    def log_growth(self,z):
        """using equation 3.2 from Baldauf 2015, not currently consistent with G(z)"""
        a=1/(1+z)
        print 'what I think it is', a/self.H(z)*self.dH_da(z) + 5/2.*self.Omegam*self.G_norm(0)/self.H(z)**2/a**2/self.G_norm(z)
        return -3/2.*self.Omegam/a**3*self.H0**2/self.H(z)**2 + 1/self.H(z)**2/a**2/self.G_norm(z)

    # ----------------------------------------------------------------------------
    # halo and matter stuff
    # -----------------------------------------------------------------------------
    def delta_c(self,z):
        """ critical threshold for spherical collapse, as given
        in the appendix of NFW 1997"""
        #TODO fitting formula, probably not appropriate for the code anymore
        #TODO should have z dependence
        A=0.15*(12.*np.pi)**(2/3.)

        if (self.Omegam ==1) and (self.OmegaL==0):
            d_crit=A
        elif (self.Omegam < 1) and (self.OmegaL ==0):
            d_crit=A*self.Omegam**(0.0185)
        elif (self.Omegam + self.OmegaL)==1.0:
            d_crit=A*self.Omegam**(0.0055)
        else:
            d_crit=A*self.Omegam**(0.0055)
            warn('inexact equality to 1~='+str(self.Omegam)+"+"+str(self.OmegaL)+"="+str(self.OmegaL+self.Omegam))
        d_c=d_crit#/self.G_norm(z)
        return d_c

    def delta_v(self,z):
        """over density for virialized halo"""
        A=178.0
        if (self.Omegam_z(z) ==1) and (self.OmegaL_z(z)==0):
            d_v=A
        if (self.Omegam_z(z) < 1) and (self.OmegaL_z(z) ==0):
            d_v=A/self.Omegam_z(z)**(0.7)
        if (self.Omegam_z(z) + self.OmegaL_z(z))==1.0:
            d_v=A/self.Omegam_z(z)**(0.55)

        return d_v/self.G_norm(z)


    def nu(self,z,mass):
        """calculates nu=(delta_c/sigma(M))^2
         delta_c is the overdensity for collapse"""
        return (self.delta_c(z)/self.sigma_m(mass,z))**2

    def sigma_m(self,mass,z):
        """ RMS power on a scale of R(mass)
         rho=mass/volume=mass"""
        R=3/4.*mass/self.rho_bar(z)/np.pi
        R=R**(1/3.)
        return self.sigma_r(z,R)

    def sigma_r(self,z,R):
        r""" returns RMS power on scale R
         sigma^2(R)= \int dlogk/(2 np.pi^2) W^2 P(k) k^3
         user needs to adjust for growth factor upon return """
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
        P=self.P_lin.get_matter_power(np.array([0.]),pmodel='linear')[:,0]
        I=trapz2(((W*W).T*P*self.k**3).T,np.log(self.k)).T/2./np.pi**2

        return np.sqrt(I)

    # densities as a function of redshift

    def Omegar_z(self,z):
        """Radiation density as a function of redshift"""
        return self.Omegar*(1.+z)**4/self.Ez(z)**2

    def Omegam_z(self,z):
        """matter density as a function of redshift"""
        return self.Omegam*(1.+z)**3/self.Ez(z)**2

    def OmegaL_z(self,z):
        """dark energy density as function of redshift, considers variable w(z)"""
        return self.OmegaL*self.de_mult(z)/self.Ez(z)**2

    def Omegak_z(self,z):
        """curvature density as a function of redshift"""
        return 1.-self.Omegam_z(z)-self.OmegaL_z(z)-self.Omegar_z(z)

    def Omega_tot(self,z):
        """Total density as a function of redshift, should be 1"""
        return self.Omegak_z(z) + self.OmegaL_z(z) + self.Omegam_z(z)+self.Omegar_z(z)

    def rho_bar(self,z):
        """return average density in units of solar mass and h^2 """
        return self.rho_crit(z)*self.Omegam_z(z)

    #critical density should be about 2.77536627*10^11 h^2 M_sun/Mpc^-3 at z=0 according to pdg table
    def rho_crit(self,z):
        """return critical density in units of solar mass and h^2 """
        factor=1e12/self.M_sun*self.pc
        #print 'rho crit [g/cm^3] at z =', z, 3*self.H(z)**2/8./np.pi/self.GN*1e9/(3.086*10**24)**2/10**2
        return 3*self.H(z)**2/8./np.pi/self.GN*factor/self.h**2

    def get_P_lin(self):
        """Get stored linear power spectrum"""
        return self.k, self.P_lin

    #use this instead of storing sigma8 because isn't simple analytic formula
    def get_sigma8(self):
        """get sigma8, from cosmology or camb"""
        if 'sigma8' in P_SPACES[self.p_space]:
            return self.cosmology['sigma8']
        elif self.P_lin is None:
            warn('CosmoPie should use MatterPower to get camb computed sigma8')
            return camb_power.camb_sigma8(self.cosmology,self.camb_params)
        else:
            return self.P_lin.get_sigma8_eff(np.array([0.]))[0]

    #TODO be consistent about use of this function
    def set_power(self,P_in):
        """set the matter power spectrum for the CosmoPie
            inputs:
                P_in: a MatterPower object
        """
        self.P_lin = P_in
        self.k = P_in.k

    # -----------------------------------------------------------------------------

JDEM_LIST = ['ws36_'+str(itr_36) for itr_36 in xrange(0,36)]
P_SPACES ={'jdem': ['ns','Omegamh2','Omegabh2','Omegakh2','OmegaLh2','dGamma','dM','LogG0','LogAs'],
           'lihu' : ['ns','Omegach2','Omegabh2','Omegakh2','h','LogAs'],
           'basic': ['ns','Omegamh2','Omegabh2','Omegakh2','h','sigma8'],
           'overwride':[]}
DE_METHODS = {'constant_w':['w'],
              'w0wa'      :['w','w0','wa'],
              'jdem'      :JDEM_LIST,
              'grid_w'    :['ws']}
#parameters guaranteed not to affect linear growth factor (G_norm)
GROW_SAFE = ['ns','LogAs','sigma8']
#parameters guaranteed to not require generating a new WMatcher object
DE_SAFE = np.unique(np.concatenate(DE_METHODS.values())).tolist()
def strip_cosmology(cosmo_old,p_space,overwride=[]):
    """
        remove all nonessential attributes for a cosmology unless they are in overwride list,
        leaving only the necessary elements of the parameter space
        possible parameter spaces (with required elements) are:
        'jdem':parameter space proposed in joint dark energy mission figure of merit working group paper, arxiv:0901.0721v1
        'lihu':parameter space used in Li, Hu & Takada 2013, arxiv:1408.1081v2
    """
    cosmo_new = cosmo_old.copy()
    if p_space in P_SPACES:
        #delete unneeded values
        for key in cosmo_old:
            if key not in P_SPACES[p_space] and key not in overwride:
                cosmo_new.pop(key,None)
        #TODO consider better defaults
        for req in P_SPACES[p_space]:
            if req not in cosmo_new:
                cosmo_new[req] = defaults.cosmology_jdem[req]

    else:
        raise ValueError('unrecognized p_space \''+str(p_space)+'\'')


    #mark this cosmology with its parameter space
    cosmo_new['p_space']=p_space
    return cosmo_new

def add_derived_pars(cosmo_old,p_space=None):
    """
    use relations to add all known derived parameters to a cosmology starting from its parameter space, including:
    Omegam,Omegac,Omegab,Omegak,OmegaL,Omegar
    Omegamh2,Omegach2,Omegabh2,Omegakh2,OmegaLh2,Omegarh2
    H0,h
    LogAs,As,sigma8 #TODO check
    """
    cosmo_new = cosmo_old.copy()
    if p_space is None:
        p_space = cosmo_old.get('p_space')
    if p_space == 'jdem':
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
    elif p_space == 'lihu':
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
    elif p_space == 'basic':
        cosmo_new['Omegach2'] = cosmo_old['Omegamh2']-cosmo_old['Omegabh2']
        cosmo_new['H0'] = cosmo_old['h']*100.
        cosmo_new['Omegab'] = cosmo_old['Omegabh2']/cosmo_new['h']**2
        cosmo_new['Omegac'] = cosmo_new['Omegach2']/cosmo_new['h']**2
        cosmo_new['Omegam'] = cosmo_old['Omegamh2']/cosmo_new['h']**2
        cosmo_new['Omegak'] = cosmo_old['Omegakh2']/cosmo_new['h']**2
        cosmo_new['Omegar'] = 0. #just set to 0 for now

        cosmo_new['OmegaL'] = 1.-cosmo_new['Omegam']-cosmo_new['Omegar']-cosmo_new['Omegak']
        cosmo_new['OmegaLh2'] = cosmo_new['OmegaL']*cosmo_new['h']**2
        cosmo_new['Omegarh2'] = cosmo_new['Omegar']*cosmo_new['h']**2
       #don't know how to get As if don't have
       # cosmo_new['As']=np.exp(cosmo_old['LogAs'])

    elif p_space == 'overwride':
        #option which does nothing
        pass

    else:
        raise ValueError('unrecognized p_space \''+str(p_space)+'\'')

    cosmo_new['p_space'] = p_space
    return cosmo_new
