"""
        Computes and manages various parameters associated with a given cosmology
        Joseph E. McEwen (c) 2016
        Matthew C. Digman
        mcewen.24@osu.edu
"""

from warnings import warn
from scipy.integrate import odeint
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np

from algebra_utils import trapz2

from dark_energy_model import DarkEnergyConstant,DarkEnergyW0Wa,DarkEnergyJDEM

DEBUG=True

eps = np.finfo(float).eps
#TODO ensure safe sigma8
class CosmoPie(object):
    """stores and calculates various parameters for a cosmology"""
    #TODO convergence test a grid values
    #TODO reduce possibility of circular reference to MatterPower
    def __init__(self,cosmology,p_space,P_lin=None,k=None,a_step=0.001,G_in=None,G_safe=False,silent=False):
        """
        Set up for cosmological parameters in input cosmology.
        Inputs:
        cosmology: a dictionary of cosmological parameters,
        P_lin: a MatterPower object. Optional.
        p_space: The cosmological parameter space. available options listed in P_SPACES.
        G_in: Function to get linear growth factor. Optional.
        G_safe: If True and G_in is not None, use G_in instead of solving differential equation.
        silent: If True, less print statements
        """


        # default to Planck 2015 values
        self.silent = silent
        if not silent:
            print "cosmopie "+str(id(self))+": begin initialization"

        #define parameterization
        self.p_space = p_space
        #fill out cosmology
        self.cosmology = cosmology.copy()
        self.cosmology = add_derived_pars(self.cosmology,self.p_space)
        if DEBUG:
            sanity_check_pars(self.cosmology)


        self.Omegabh2 = self.cosmology['Omegabh2']
        self.Omegach2 = self.cosmology['Omegach2']
        self.Omegamh2 = self.cosmology['Omegamh2']
        self.OmegaLh2 = self.cosmology['OmegaLh2']
        self.Omegakh2 = self.cosmology['Omegakh2']
        self.Omegab   = self.cosmology['Omegab']
        self.Omegac   = self.cosmology['Omegac']
        self.Omegam   = self.cosmology['Omegam']
        self.Omegar   = self.cosmology['Omegar']
        self.OmegaL   = self.cosmology['OmegaL']
        self.Omegak   = self.cosmology['Omegak']
        self.ns = self.cosmology['ns']
        self.H0       = self.cosmology['H0']
        self.h        = self.cosmology['h']

        #get multipliers in H(z) for OmegaL with an equation of state w(z)
        self.de_model = self.cosmology.get('de_model')
        #default to assuming dark energy is constant
        if self.de_model is None:
            warn('no dark energy model specified, assuming w=-1')
            self.de_model = 'constant_w'
            self.cosmology['de_model'] = self.de_model
            self.cosmology['w'] = -1

        if self.de_model=='constant_w':
            self.de_object = DarkEnergyConstant(self.cosmology['w'])
        elif self.de_model=='w0wa':
            #Chevallier-Polarski-Linder model, solution can be found in ie arXiv:1605.01475
            self.de_object = DarkEnergyW0Wa(self.cosmology['w0'],self.cosmology['wa'])
#            #cf ie https://ned.ipac.caltech.edu/level5/March08/Frieman/Frieman2.html#note1, arXiv:1605.01475
        elif self.de_model=='jdem':
            ws_in = np.array([self.cosmology['ws36_'+str(i).zfill(2)] for i in range(0,36)])
            self.de_object = DarkEnergyJDEM(ws_in,a_step,self.cosmology['w'])
        else:
            raise ValueError('unrecognized dark energy model \''+str(self.de_model)+'\'')


        # solar mass
        #self.M_sun = 1.9885*1e30 # kg
        self.M_sun = 1.98847541534*1e30 

        # parsec
        self.pc = 3.08567758147*1e16 # m

        # Newton's constant (CODATA value)
        self.GN = 6.67408*10**(-11) # m^3/kg/s^2

        # speed of light
        self.c        = 2.997924580*1e5 #km/s

        self.DH       = self.c/self.H0

        #curvature
        self.K = -self.Omegak*(self.H0/self.c)**2

        #precompute the normalization factor from the differential equation
        self.a_step = a_step
        self.G_safe = G_safe

        a_min = a_step
        a_max = 1.
        self.a_grid = np.arange(a_max,a_min-a_step/10.,-a_step)
        self.z_grid = 1./self.a_grid-1.

        if not self.G_safe:
            def _dp_mult_interp(a):
                z = 1./a-1.
                return -1./a*(7./2.*self.Omegam_z(z)+3*self.Omegar_z(z)+(7./2.-3./2.*self.de_object.w_of_z(z))*self.OmegaL_z(z)+4.*self.Omegak_z(z))
            def _d_mult_interp(a):
                z = 1./a-1.
                return -1./a**2*(self.Omegar_z(z)+3./2.*(1.-self.de_object.w_of_z(z))*self.OmegaL_z(z)+2*self.Omegak_z(z))
            d_ics = np.array([1.,0.])
            d_args = (np.array([_d_mult_interp,_dp_mult_interp]),)
            def _d_evolve_eqs(ys,a,g_args):
                yps = np.zeros(2)
                yps[0] = ys[1]
                yps[1] = g_args[0](a)*ys[0]+g_args[1](a)*ys[1]
                return yps
            #drop ghost cell
            integ_result = odeint(_d_evolve_eqs,d_ics,self.a_grid[::-1],d_args)
            self.G_p = InterpolatedUnivariateSpline(self.z_grid,(self.a_grid*integ_result[:,0][::-1]),k=2,ext=2)
        else:
            self.G_p = G_in

        self.P_lin = P_lin
        self.k = k

        if not silent:
            print "cosmopie "+str(id(self))+": finished initialization"

    def Ez(self,z):
        """
        Get E(z)=H(z)/H0
        """
        zp1 = z + 1
        #need to use interpolating function for a dependence of OmegaL term because could be time dependent if w is not constant
        return np.sqrt(self.Omegam*zp1**3 + self.Omegar*zp1**4 + self.Omegak*zp1**2 + self.OmegaL*self.de_object.de_mult(z))

    def H(self,z):
        """
        Hubble parameter H(z)
        """
        return self.H0*self.Ez(z)

#    def dH_da(self,z):
#        """
#        Derivative of H with respect to a
#        """
#        return -(1+z)*self.H0/2.*(3.*self.Omegam_z(z)+4.*self.Omegar_z(z)+2.*self.Omegak_z(z)+3.*(1.+self.de_object.w_of_z(z))*self.OmegaL_z(z))

    # distances, volumes, and time
    # -----------------------------------------------------------------------------
    def D_comov(self,z):
        """the line of sight comoving distance, Mpc"""
        #I = lambda zp : 1/self.Ez(zp)
        #return self.DH*quad(I,0,z)[0]
        def _integrand(_,zp):
            """integrand for comoving distance"""
            return 1/self.Ez(zp)
        #make sure value at 0 is 0 to fix initial conditions
        z_use = np.hstack([0.,z])
        d = self.DH*odeint(_integrand,0.,z_use,atol=10e-20,rtol=10e-10)[1::,0]
        if np.isscalar(z):
            return d[0]
        else:
            return d

    def D_comov_A(self,z):
        """the comoving angular diameter distance, as defined in ie Bartelmann and Schneider arXiv:astro-px/9912508v1"""
        if self.K==0:
            return self.D_comov(z)
        elif self.K>0:
            sqrtK = np.sqrt(abs(self.K))
            return 1./sqrtK*np.sin(sqrtK*self.D_comov(z))
        else:
            sqrtK = np.sqrt(abs(self.K))
            return 1./sqrtK*np.sinh(sqrtK*self.D_comov(z))


#    def D_comov_dz(self,z):
#        """Integrand of D_comov"""
#        return self.DH/self.Ez(z)

    def D_comov_T(self,z):
        """Transverse comoving distance"""
        if self.Omegak > 0:
            sq = np.sqrt(self.Omegak)
            return self.DH/sq*np.sinh(sq*self.D_comov(z)/self.DH)
        elif self.Omegak < 0:
            sq = np.sqrt(-self.Omegak)
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
         dV/dz, in Mpc^3"""
        return self.DH*(1+z)**2*self.D_A(z)**2/self.Ez(z)
    # -----------------------------------------------------------------------------
    # Growth functions
    # -----------------------------------------------------------------------------
    def G(self,z):
        """Linear growth factor"""
        return self.G_p(z)

    def G_norm(self,z):
        """linear growth factor normalized so the G(z=0)=1"""
        G_0 = self.G(0.)
        return self.G(z)/G_0


#    def log_growth(self,z):
#        """using equation 3.2 from Baldauf 2015, not currently consistent with G(z)"""
#        a = 1/(1+z)
#        print 'what I think it is', a/self.H(z)*self.dH_da(z) + 5/2.*self.Omegam*self.G_norm(0)/self.H(z)**2/a**2/self.G_norm(z)
#        return -3/2.*self.Omegam/a**3*self.H0**2/self.H(z)**2 + 1/self.H(z)**2/a**2/self.G_norm(z)

    # ----------------------------------------------------------------------------
    # halo and matter stuff
    # -----------------------------------------------------------------------------

    #could move this to MatterPower or something
    def sigma_r(self,z,R):
        r""" returns RMS power on scale R
         sigma^2(R)= \int dlogk/(2 np.pi^2) W^2 P(k) k^3
         user needs to adjust for growth factor upon return """
        if self.P_lin is None:
            raise ValueError('You need to provide a linear power spectrum through set_power to get sigma valeus')
        z = np.asanyarray(z)
        k = self.P_lin.k 
        if isinstance(R,np.ndarray):
            kr = np.outer(k,R)
        else:
            kr = self.P_lin.k*R
        W = 3.0*(np.sin(kr)/kr**3-np.cos(kr)/kr**2)
        #P=self.G_norm(z)**2*self.P_lin
        #TODO should scale to input sigma8 if set?
        P = self.P_lin.get_matter_power(z,pmodel='linear')[:,0]
        result = np.trapz(((W*W).T*P*k**3).T,np.log(k),axis=0).T/2./np.pi**2

        return np.sqrt(result)

    # densities as a function of redshift

    def Omegar_z(self,z):
        """Radiation density as a function of redshift"""
        return self.Omegar*(1.+z)**4/self.Ez(z)**2

    def Omegam_z(self,z):
        """matter density as a function of redshift"""
        return self.Omegam*(1.+z)**3/self.Ez(z)**2

    def OmegaL_z(self,z):
        """dark energy density as function of redshift, considers variable w(z)"""
        return self.OmegaL*self.de_object.de_mult(z)/self.Ez(z)**2

    def Omegak_z(self,z):
        """curvature density as a function of redshift"""
        return 1.-self.Omegam_z(z)-self.OmegaL_z(z)-self.Omegar_z(z)

#    def Omega_tot(self,z):
#        """Total density as a function of redshift, should be 1"""
#        return self.Omegak_z(z) + self.OmegaL_z(z) + self.Omegam_z(z)+self.Omegar_z(z)

    def rho_bar(self,z):
        """return average density in units of solar mass and h^2 """
        return self.rho_crit(z)*self.Omegam_z(z)

    #critical density should be about 2.77536627*10^11 h^2 M_sun/Mpc^-3 at z=0 according to pdg table
    def rho_crit(self,z):
        """return critical density in units of solar mass and h^2 """
        factor = 1e12/self.M_sun*self.pc
        #print 'rho crit [g/cm^3] at z =', z, 3*self.H(z)**2/8./np.pi/self.GN*1e9/(self.pc*10**8)**2/10**2
        return 3*self.H(z)**2/8./np.pi/self.GN*factor/self.h**2

    def get_P_lin(self):
        """Get stored linear power spectrum"""
        return self.P_lin.k, self.P_lin

    #use this instead of storing sigma8 because isn't simple analytic formula
    def get_sigma8(self):
        """get sigma8, from cosmology or camb"""
        if 'sigma8' in P_SPACES[self.p_space]:
            return self.cosmology['sigma8']
        elif self.P_lin is None:
            raise ValueError('CosmoPie needs to have a MatterPower object through set_power to get camb computed sigma8')
            #return camb_power.camb_sigma8(self.cosmology,self.camb_params)
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

JDEM_LIST = ['ws36_'+str(itr_36).zfill(2) for itr_36 in xrange(0,36)]
P_SPACES = {'jdem': ['ns','Omegamh2','Omegabh2','Omegakh2','OmegaLh2','dGamma','dM','LogG0','LogAs'],
            'lihu' : ['ns','Omegach2','Omegabh2','Omegakh2','h','LogAs'],
            'basic': ['ns','Omegamh2','Omegabh2','Omegakh2','h','sigma8'],
            'overwride':[]}
DE_METHODS = {'constant_w':['w'],
              'w0wa'      :['w','w0','wa'],
              'jdem'      :JDEM_LIST,
              'grid_w'    :['ws']}
#parameters guaranteed not to affect linear growth factor (G_norm)
GROW_SAFE = ['ns','LogAs','sigma8','p_space','de_model']
#parameters guaranteed to not require generating a new WMatcher object
DE_SAFE = np.unique(np.concatenate(DE_METHODS.values())).tolist()
def strip_cosmology(cosmo_old,p_space,overwride=None):
    """
        remove all nonessential attributes for a cosmology unless they are in overwride list,
        leaving only the necessary elements of the parameter space
        possible parameter spaces (with required elements) are:
        'jdem':parameter space proposed in joint dark energy mission figure of merit working group paper, arxiv:0901.0721v1
        'lihu':parameter space used in Li, Hu & Takada 2013, arxiv:1408.1081v2
    """
    if overwride is None:
        overwride = []
    cosmo_new = cosmo_old.copy()
    if p_space in P_SPACES:
        #delete unneeded values
        for key in cosmo_old:
            if key not in P_SPACES[p_space] and key not in overwride:
                cosmo_new.pop(key,None)

        for req in P_SPACES[p_space]:
            if req not in cosmo_new:
                raise ValueError('cosmology is missing required argument '+str(req))
        #        cosmo_new[req] = defaults.cosmology_jdem[req]

    else:
        raise ValueError('unrecognized p_space \''+str(p_space)+'\'')


    #mark this cosmology with its parameter space
    cosmo_new['p_space'] = p_space
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
    if p_space=='jdem':
        cosmo_new['Omegach2'] = cosmo_old['Omegamh2']-cosmo_old['Omegabh2']
        cosmo_new['h'] = np.sqrt(cosmo_old['Omegamh2']+cosmo_old['OmegaLh2']+cosmo_old['Omegakh2'])
        cosmo_new['H0'] = cosmo_new['h']*100.
        cosmo_new['Omegab'] = cosmo_old['Omegabh2']/cosmo_new['h']**2
        cosmo_new['Omegac'] = cosmo_new['Omegach2']/cosmo_new['h']**2
        cosmo_new['Omegam'] = cosmo_old['Omegamh2']/cosmo_new['h']**2
        cosmo_new['OmegaL'] = cosmo_old['OmegaLh2']/cosmo_new['h']**2
        cosmo_new['Omegak'] = cosmo_old['Omegakh2']/cosmo_new['h']**2
        #define omega radiation
        cosmo_new['Omegar'] = 0.
        #cosmo_new['Omegar'] = 1.-cosmo_new['Omegam']-cosmo_new['OmegaL']-cosmo_new['Omegak']
        cosmo_new['Omegarh2'] = cosmo_new['Omegar']*cosmo_new['h']**2

        cosmo_new['As'] = np.exp(cosmo_old['LogAs'])
    elif p_space=='lihu':
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

        cosmo_new['As'] = np.exp(cosmo_old['LogAs'])
    elif p_space=='basic':
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

    elif p_space=='overwride':
        #option which does nothing
        pass

    else:
        raise ValueError('unrecognized p_space \''+str(p_space)+'\'')

    cosmo_new['p_space'] = p_space
    return cosmo_new

def sanity_check_pars(cosmo_in):
    """run a few sanity checks of the cosmological paramaters"""
    assert np.isclose(cosmo_in['Omegamh2'],cosmo_in['Omegam']*cosmo_in['h']**2)
    assert np.isclose(cosmo_in['OmegaLh2'],cosmo_in['OmegaL']*cosmo_in['h']**2)
    assert np.isclose(cosmo_in['Omegach2'],cosmo_in['Omegac']*cosmo_in['h']**2)
    assert np.isclose(cosmo_in['Omegabh2'],cosmo_in['Omegab']*cosmo_in['h']**2)
    assert np.isclose(cosmo_in['Omegakh2'],cosmo_in['Omegak']*cosmo_in['h']**2)
    assert np.isclose(cosmo_in['Omegarh2'],cosmo_in['Omegar']*cosmo_in['h']**2)
    assert np.isclose(cosmo_in['h'],cosmo_in['H0']/100.)
    assert np.isclose(cosmo_in['Omegak']+cosmo_in['Omegar']+cosmo_in['Omegam']+cosmo_in['OmegaL'],1.)
    assert np.isclose(cosmo_in['Omegakh2']+cosmo_in['Omegarh2']+cosmo_in['Omegamh2']+cosmo_in['OmegaLh2'],cosmo_in['h']**2)
    assert np.isclose(cosmo_in['Omegam'],cosmo_in['Omegab']+cosmo_in['Omegac'])
    assert np.isclose(cosmo_in['Omegamh2'],cosmo_in['Omegabh2']+cosmo_in['Omegach2'])
    assert cosmo_in['h']>0.
    assert cosmo_in['Omegam']>=0.
    assert cosmo_in['Omegac']>=0.
    assert cosmo_in['Omegab']>=0.
    assert cosmo_in['Omegar']==0. #TODO fix if use Omegar
    assert cosmo_in['Omegarh2']==0.
    assert cosmo_in['Omegak']==0.
    assert cosmo_in['Omegakh2']==0.
