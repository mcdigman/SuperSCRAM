import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import cosmopie as cp
import defaults
from warnings import warn
from numpy.core.umath_tests import inner1d
#p_h = np.loadtxt('camb_m_pow_l.dat')#
#p_interp = interp1d(p_h[:,0],p_h[:,1]*p_h[:,0]**3/(2*np.pi**2))

class HalofitPk(object):
    ''' python version of Halofit, orginally presented in Smith 2003. Bibtex entry for Smith 2003
                    et al. is below. 
                    
                    @MISC{2014ascl.soft02032P,
                    author = {{Peacock}, J.~A. and {Smith}, R.~E.},
                    title = "{HALOFIT: Nonlinear distribution of cosmological mass and galaxies}",
                    howpublished = {Astrophysics Source Code Library},
                    year = 2014,
                    month = feb,
                    archivePrefix = "ascl",
                    eprint = {1402.032},
                    adsurl = {http://adsabs.harvard.edu/abs/2014ascl.soft02032P},
                    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
                    }
    
    '''
    #take an input linear power spectrum at z=0, d2_l and d2_nl will output power spectrum at desired redshit
    def __init__(self,C,k_in,p_lin=np.array([]),halofit_params=defaults.halofit_params):

        self.C = C
        self.k_max = max(k_in)
        self.k_in = k_in
        self.params = halofit_params
        self.k_fix = self.params['k_fix']
        self.linear_cutoff = self.params['min_kh_nonlinear'] 
        self.smooth_width=self.params['smooth_width']
        if not self.C.camb_params['leave_h']:
            self.linear_cutoff = self.linear_cutoff*self.C.cosmology['h']
        #extrapolated internal power spectrum to avoid spurious dependence on input k_max
        k_max_in = np.max(k_in)
        #TODO probably should low extrap as well
        if self.params['extrap_wint'] and k_max_in<self.k_fix:
            #assume log spaced k_in
            log_k_space = np.average(np.diff(np.log(k_in)))
            k_diff = (np.log(self.k_fix)-np.log(k_max_in))
            self.n_k_extend = np.ceil(k_diff/log_k_space)
            k_extend = np.exp(np.log(k_max_in)+(np.arange(0,self.n_k_extend)+1.0)*log_k_space)
            self.k = np.hstack((k_in,k_extend))
            self.p_use = np.hstack((p_lin,p_lin[-1]*(k_extend/k_in[-1])**(C.ns-4.))) #use ns-4 extrapolation on high end
        else:
            self.n_k_extend=0
            self.k = k_in
            self.p_use = p_lin
        self.k_max = np.max(self.k)
            
        self.gams=.21# what is the number ?

        if self.p_use.size==0:
            self.p_use=self.p_cdm(self.k)*(2.*np.pi**2)/k_in**3
        self.p_input = self.p_use*self.k**3/(2.*np.pi**2)
        #spline might be more accurate, although slower 
        self.p_interp = interp1d(self.k,self.p_input)

       #TODO check necessary r_min,r_max,n_r nint in wint are good
        r_min = halofit_params['r_min']
        r_max = halofit_params['r_max']
        r_step = halofit_params['r_step']

        #obsolete
        #self.c_threshold = halofit_params['c_threshold']
        #self.cutoff = halofit_params['cutoff']

        #n_r = halofit_params['n_r']
        #array should use generously sized r_max initially
        rs = np.arange(r_min,r_max,r_step)
        n_r = rs.size
#        sigs = np.zeros(n_r)
#        sig_d1s = np.zeros(n_r)
#        sig_d2s = np.zeros(n_r)
        
        sigs,sig_d1s,sig_d2s=self.wint(rs)
        #can safely cutoff if 1/sig is greater than 1, because that corresponds to a normalized growth factor>1 and 1/sig monotonically increases with higher r
#        n_r_max = n_r
#        for i in xrange(0,n_r):
#            sig,d1,d2 = self.wint(rs[i])
#            sigs[i] = sig
#            sig_d1s[i] = d1
#            sig_d2s[i] = d2
#            if self.cutoff and 1./sig>1.:
#                n_r_max = i+1
#                sigs = sigs[0:n_r_max]
#                rs = rs[0:n_r_max]
#                sig_d1s = sig_d1s[0:n_r_max]
#                sig_d2s = sig_d2s[0:n_r_max]
#                break
        #extend range automatically if input r_max was not large enough to get to G=1 to prevent crashes
        if 1./sigs[-1]<1.:
            warn('with given parameters,halofit will only work up G_norm='+str(1./sigs[-1])+', try increasing r_max: extending')
            itr = 1
            r_range = r_max-r_min
            while 1./sigs[-1]<1. and itr<halofit_params['max_extend']:
                rs = np.hstack((rs,np.arange(rs[-1]+r_step,rs[-1]+r_step+r_range,r_step)))
                sigs,sig_d1s,sig_d2s=self.wint(rs)
                itr+=1

        self.n_r = n_r
        self.sigs = sigs 
        self.g_max = (1./sigs[-1])
        self.g_min = (1./sigs[0])
        #splines might give somewhat better results than interp1d for a given grid size, which is what this originally used, at the expense of time
        self.r_grow = interp1d(1./sigs,rs)
        self.sig_d1 = interp1d(rs,sig_d1s)
        self.sig_d2 = interp1d(rs,sig_d2s)
#        self.z = np.array([0.])
#        self.spectral_parameters()


#    #way of getting parameters without doing the interpolation, old
#    def spectral_parameters(self):
#
#        growth=self.C.G_norm(self.z)
#        self.amp=growth
#        xlogr1=-2.0
#        xlogr2=3.5 
#        
#        ''' iterate to determine the wavenumber where
#                        nonlinear effects become important (rknl), 
#                        the effectice spectral index (rneff),
#                        the second derivative of the power spectrum at rknl (rncur) 
#        '''
#
#        diff_last = np.inf
#        while True:
#                rmid = 10**( 0.5*(xlogr2+xlogr1) )
#
#                sig,d1,d2 = self.wint(rmid)
#                sig = sig*growth
#                diff = sig-1.0
#                diff_frac = abs((diff_last-diff)/diff)
#                diff_last = diff    
#                if diff > 0.0001 and diff_frac>self.c_threshold:
#                    xlogr1 = np.log10(rmid)
#                    continue
#                elif diff < -0.0001 and diff_frac>self.c_threshold:
#                    xlogr2 = np.log10(rmid)
#                    continue
#                else:
#                    self.rknl = 1./rmid
#                    self.rneff = -3-d1
#                    self.rncur = -d2
#                    if abs(diff) > 0.0001:
#                        warn('halofit may not have converged sufficiently')
#                    break
#                       #self.amp = growth

    def wint(self,r):
        '''
        fint the effectice spectral quantities
        rkn;, rneff, rncurr.  Uses a Gaussian filter at the 
        to determine the scale at which the variance is unity, to 
        fing rknl. rneff is defined as the first derivative of the 
        variance, calculated at the nonlinear wavenumber. rncur si the 
        second derivative of variance at rknl. 
        '''
        #TODO consider handling upper limit differently
        #current method introduces dependence of k_max of input power spectrum, which is not ideal.
        #nint = 1000
        #nint=np.floor((self.k_max)/2.)
        #print self.k_max/2.
        #TODO figure out what to do if k_max<2
        nint =min(np.int(self.k_max/2.),np.int(self.k_fix/2.))
        #nint = np.int(self.k_fix/2.)
        t = ( np.arange(nint)+0.5 )/nint
        y = 1./t - 1.
        #rk = y
        #TODO check this is correct way to get d2
        d2 = self.D2_L(y,0.)
        mult = d2/y/t**2/nint
        x2 = np.outer(r**2,y**2)
        w1=mult*np.exp(-x2)
        w2=2.*x2*w1
        #w3=4.*x2*(1.-x2)*w1
        #w3=2.*w2*(1.-x2)

        #sum1 = np.inner(mult,np.exp(-x2))   
        sum1 = np.sum(w1,axis=1)
        #sum2 = 2.*inner1d(w1,x2)
        sum2 = np.sum(w2,axis=1)
        sum3 = 2.*(sum2-inner1d(w2,x2))
        #sum3 = np.sum(w3,axis=1)
                
        sig = np.sqrt(sum1)
        d1  = -sum2/sum1        
        #d2  = -sum2**2/sum1**2 - sum3/sum1
        d2  = -d1**2 - sum3/sum1
        return sig,d1,d2



    def D2_L(self,rk,z):
        # the linear power spectrum 
        if not isinstance(z,np.ndarray):
            return self.C.G_norm(z)**2*self.p_interp(rk)
        else:
            return np.outer(self.p_interp(rk),self.C.G_norm(z)**2)
    #def D2_L(self,rk,z):
    #   rk=np.asarray(rk)
    #   return self.C.G_norm(z)**2*self.p_cdm(rk)
    
    #old way,obsolete 
    def p_cdm(self,rk):
        # unormalized power spectrum 
        # cf Annu. Rev. Astron. Astrophys. 1994. 32: 319-70 
        rk = np.asarray(rk)
        p_index = 1.
        rkeff=0.172+0.011*np.log(self.gams/0.36)*np.log(self.gams/0.36)
        q=1.e-20 + rk/self.gams
        q8=1.e-20 + rkeff/self.gams
        tk=1./(1.+(6.4*q+(3.0*q)**1.5+(1.7*q)**2)**1.13)**(1./1.13)
        tk8=1./(1.+(6.4*q8+(3.0*q8)**1.5+(1.7*q8)**2)**1.13)**(1./1.13)
        return self.C.get_sigma8()*self.C.get_sigma8()*((q/q8)**(3.+p_index))*tk*tk/tk8/tk8

    def D2_NL_smith(self,rk,z,return_components = False):
        """
        halo model nonlinear fitting formula as described in 
        Appendix C of Smith et al. (2002)
        modified by Takahashi arXiv:1208.2701v2 as in cosmosis
        """
        rk = np.asarray(rk)
        #rn    = self.rneff
        
        growth = self.C.G_norm(z)
        
        rmid = self.r_grow(growth)
        d1 = self.sig_d1(rmid)
        d2 = self.sig_d2(rmid)

        rknl = 1./rmid
        rn = -3-d1
        rncur = -d2
        #rncur = self.rncur
        #rknl  = self.rknl
        plin  = self.D2_L(rk,z)
        om_m  = self.C.Omegam_z(z)
        om_v  = self.C.OmegaL_z(z)

        #cf Bird, Viel, Haehnelt 2011 for extragam explanation (cosmosis)
        extragam = 0.3159-0.0765*rn-0.8350*rncur
        #extragam=0.
        
            
        gam=extragam+0.86485+0.2989*rn+0.1631*rncur
        a=10**(1.4861+1.83693*rn+1.67618*rn*rn+0.7940*rn*rn*rn+\
             0.1670756*rn*rn*rn*rn-0.620695*rncur)
        b=10**(0.9463+0.9466*rn+0.3084*rn*rn-0.940*rncur)
        c=10**(-0.2807+0.6669*rn+0.3214*rn*rn-0.0793*rncur)
        xmu=10**(-3.54419+0.19086*rn)
        xnu=10**(0.95897+1.2857*rn)
        alpha=1.38848+0.3701*rn-0.1452*rn*rn
        fnu = 0.0 #for neutrinos later, in cosmosis
        beta=0.8291+0.9854*rn+0.3400*rn**2+fnu*(-6.4868+1.4373*rn**2)

        
        if abs(1-om_m) > 0.01: #omega evolution
            f1a=om_m**(-0.0732)
            f2a=om_m**(-0.1423)
            f3a=om_m**(0.0725)
            f1b=om_m**(-0.0307)
            f2b=om_m**(-0.0585)
            f3b=om_m**(0.0743)       
            frac=om_v/(1.-om_m) 
            f1=frac*f1b + (1-frac)*f1a
            f2=frac*f2b + (1-frac)*f2a
            f3=frac*f3b + (1-frac)*f3a
        else:         
            f1=1.0
            f2=1.0
            f3=1.0
        
        y=(rk/rknl)
        
        ph = a*y**(f1*3.)/(1.+b*y**(f2)+(f3*c*y)**(3.-gam))
        ph /= (1.+xmu*y**(-1)+xnu*y**(-2))
        pq = plin*(1.+plin)**beta/(1.+plin*alpha)*np.exp(-y/4.0-y**2/8.0)
        
        pnl=pq+ph

        if return_components:
            return pnl,pq,ph,plin
        else:
            return pnl
                    
    def D2_NL(self,rk,z,return_components = False,w_overwride=False,fixed_w=-1.,grow_overwride=False,fixed_growth=1.):
        """
        halo model nonlinear fitting formula as described in 
        Appendix C of Smith et al. (2002)
        """
        rk = np.asarray(rk)
        #rn    = self.rneff
        if not grow_overwride: 
            growth = self.C.G_norm(z)
        else:
            growth = fixed_growth
        if np.max(growth)>self.g_max or np.min(growth)<self.g_min:
            raise ValueError('Growth factor exceeds precomputed range. Try increasing r_max to increase G max or decreasing r_min to decrease G min.')

        rmid = self.r_grow(growth)
        d1 = self.sig_d1(rmid)
        d2 = self.sig_d2(rmid)

        rknl = 1./rmid
        rn = -3.-d1
        rncur = -d2
        #rncur = self.rncur
        #rknl  = self.rknl
        plin  = self.D2_L(rk,z)
        om_m  = self.C.Omegam_z(z)
        om_v  = self.C.OmegaL_z(z)
        #w = -0.758
        if w_overwride:
            w=fixed_w
        else:
            w=self.C.w_interp(z) #not sure if z dependent w is appropriate in halofit, possible answer in https://arxiv.org/pdf/0911.2454.pdf
       # #cf Bird, Viel, Haehnelt 2011 for extragam explanation (cosmosis)
        #extragam = 0.3159-0.0765*rn-0.8350*rncur

        #gam=extragam+0.86485+0.2989*rn+0.1631*rncur
        gam = 0.1971-0.0843*rn+0.8460*rncur
        a=10**(1.5222+2.8553*rn+2.3706*rn*rn+0.9903*rn*rn*rn+\
             0.2250*rn*rn*rn*rn-0.6038*rncur+0.1749*om_v*(1.+w))
        b=10**(-0.5642+0.5864*rn+0.5716*rn*rn-1.5474*rncur+0.2279*om_v*(1.+w))
        c=10**(0.3698+2.0404*rn+0.8161*rn*rn+0.5869*rncur)
        xmu=0.
        xnu = 10**(5.2105+3.6902*rn)
        alpha=abs(6.0835+1.3373*rn-0.1959*rn*rn-5.5274*rncur)
        fnu = 0. #neutrinos
        beta=2.0379-0.7354*rn+0.3157*rn**2+1.2490*rn**3+0.3980*rn**4-0.1682*rncur+fnu*(1.081+0.395*rn**2)
        
        if True:#abs(1-om_m) > 0.01: #omega evolution
            f1a=om_m**(-0.0732)
            f2a=om_m**(-0.1423)
            f3a=om_m**(0.0725)
            f1b=om_m**(-0.0307)
            f2b=om_m**(-0.0585)
            f3b=om_m**(0.0743)       
            frac=om_v/(1.-om_m) 
            f1=frac*f1b + (1-frac)*f1a
            f2=frac*f2b + (1-frac)*f2a
            f3=frac*f3b + (1-frac)*f3a
        else:         
            f1=1.0
            f2=1.0
            f3=1.0

        if isinstance(z,np.ndarray):
            y=np.outer(rk,1./rknl)
        else:
            y=(rk/rknl)
        #fnu from cosmosis  
        ph = a*y**(f1*3.)/(1.+b*y**(f2)+(f3*c*y)**(3.-gam))*(1.+fnu*0.977)
        ph /= (1.+xmu*y**(-1)+xnu*y**(-2))
        plinaa=(plin.T*(1+fnu*47.48*rk**2/(1+1.5*rk**2))).T #added to match camb implementation
        pq = plin*(1.+plinaa)**beta/(1.+plinaa*alpha)*np.exp(-y/4.0-y**2/8.0)
        
        pnl=pq+ph
        
        #set values below minimum k cutoff to linear to avoid difference from camb implementation
        #gaussian smooth transition to avoid spikes in derivatives wrt k 
        #pnl[rk<self.linear_cutoff] = plin[rk<self.linear_cutoff]
        gauss_kernel = 1./(self.smooth_width*np.sqrt(2.*np.pi))*np.exp(-0.5*((rk-self.linear_cutoff)/self.smooth_width)**2)
        gauss_int = cumtrapz(gauss_kernel,rk,initial=0.)
        #set last value to 1. to mitigate accumulated numerical errors
        gauss_int/=gauss_int[-1]

        pnl = (plin.T*(1.-gauss_int)+pnl.T*(gauss_int)).T
        
        if return_components:
            return pnl,pq,ph,plin
        else:
            return pnl

    #TODO use or eliminate as obsolete
    def P_NL(self,k,z,return_components = False):
        if(return_components):
            pnl,pq,ph,plin=self.D2_NL(k,z,return_components)
            return pnl/k**3*(2*np.pi**2), pq, ph, plin/k**3*(2*np.pi**2)
        else:
            return self.D2_NL(k,return_components)/k**3*(2*np.pi**2)

                    

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
                                 'Omegar'   : 0.0 # check this value too
                                 }

                h=.6774
                d1=np.loadtxt('camb_m_pow_l.dat')
                d2=np.loadtxt('Pk_Planck15.dat')
                k2=d2[:,0]; P2=d2[:,1]
                k=d1[:,0]; P1=d1[:,1]
                
                #k=np.logspace(-2,2,500)
                CP=cp.CosmoPie(cosmology=defaults.cosmology)
                do_k_fix_test=True
                if do_k_fix_test:
                    from camb_power import camb_pow
                    camb_params=defaults.camb_params.copy()
                    camb_params['maxkh']=10.
                    camb_params['maxk']=10.
                    k_lin,P_lin = camb_pow(defaults.cosmology,camb_params=camb_params)
                    params1 = defaults.halofit_params.copy()
                    params1['k_fix']=10000.
                    params2=params1.copy()
                    params2['k_fix']=500. #results look good at least to 500, probably can get away with lower k_fix

                    hf1=HalofitPk(CP,k_lin,P_lin,halofit_params=params1)
                    P_nl_1=hf1.D2_NL(k_lin,0.)*np.pi**2*2/k_lin**3
                    hf2=HalofitPk(CP,k_lin,P_lin,halofit_params=params2)
                    P_nl_2=hf2.D2_NL(k_lin,0.)*np.pi**2*2/k_lin**3
                    
                    print "avg deviation:",np.average(np.abs((P_nl_2)/(P_nl_1)-1.))
                    print "max deviation:",np.max(np.abs((P_nl_2)/(P_nl_1)-1.))

                    

                do_time_test=False
                if do_time_test:
                    for itr in xrange(0,5000):
                        HF2=HalofitPk(CP,k2,P2)

                do_plot_test=False 
                if do_plot_test:
                    HF=HalofitPk(CP,k,P1)
                    Plin=HF.D2_L(k,0.)*np.pi**2*2/k**3
                    P=HF.D2_NL(k,0.)*np.pi**2*2/k**3
                    
                    
                    import matplotlib.pyplot as plt

                    ax=plt.subplot(111)
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.set_xlim(1e-3,100)
                                
                    ax.plot(k,P, label='halofit')
                    ax.plot(k,Plin, label='linear')
                    ax.plot(k,P1, label='class ')
                    ax.plot(k2,HF2.D2_NL(k2,0.)*np.pi**2*2/k2**3, '--')
                    
                    plt.legend(loc=1)
                    plt.grid()
                  #  plt.show()
