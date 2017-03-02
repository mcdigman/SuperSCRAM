import numpy as np
from numpy import pi 
import sys
from scipy.interpolate import interp1d
import cosmopie as cp
#p_h = np.loadtxt('camb_m_pow_l.dat')#
#p_interp = interp1d(p_h[:,0],p_h[:,1]*p_h[:,0]**3/(2*np.pi**2))

class halofitPk(object):
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
    def __init__(self,C,k_in,p_lin=np.array([])):

        self.C = C
        self.k_max = max(k_in)
        self.gams=.21# what is the number ?

        if p_lin.size==0:
            p_lin=self.p_cdm(k_in)*(2.*np.pi**2)/k_in**3
        self.p_interp = interp1d(k_in,p_lin*k_in**3/(2.*np.pi**2))

        self.c_threshold = 0.001
       #TODO check necessary r_min,r_max,n_r nint in wint are good
        r_min = 0.05
        r_max = 5.
        n_r = 500
        rs = np.linspace(r_min,r_max,n_r)
        sigs = np.zeros(n_r)
        sig_d1s = np.zeros(n_r)
        sig_d2s = np.zeros(n_r)
        for i in range(0,n_r):
            sig,d1,d2 = self.wint(rs[i])
            sigs[i] = sig
            sig_d1s[i] = d1
            sig_d2s[i] = d2
        self.sigs = sigs 
        self.r_grow = interp1d(1./sigs,rs)
        self.sig_d1 = interp1d(rs,sig_d1s)
        self.sig_d2 = interp1d(rs,sig_d2s)
    #way of getting parameters without doing the interpolation, old
    def spectral_parameters(self,z):

        growth=self.C.G_norm(self.z)
        self.amp=growth
        xlogr1=-2.0
        xlogr2=3.5 
        
        ''' iterate to determine the wavenumber where
                        nonlinear effects become important (rknl), 
                        the effectice spectral index (rneff),
                        the second derivative of the power spectrum at rknl (rncur) 
        '''

        diff_last = np.inf
        while True:
                rmid = 10**( 0.5*(xlogr2+xlogr1) )

                sig,d1,d2 = self.wint(rmid)
                sig = sig*growth
                diff = sig-1.0
                diff_frac = abs((diff_last-diff)/diff)
                diff_last = diff    
                if diff > 0.0001 and diff_frac>self.c_threshold:
                    xlogr1 = np.log10(rmid)
                    continue
                elif diff < -0.0001 and diff_frac>self.c_threshold:
                    xlogr2 = np.log10(rmid)
                    continue
                else:
                    self.rknl = 1./rmid
                    self.rneff = -3-d1
                    self.rncur = -d2
                    if abs(diff) > 0.0001:
                        warn('halofit may not have converged sufficiently')
                    break
#                       self.amp = growth

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
        #nint = 1000
        #nint=np.floor((self.k_max)/2.)
        #print self.k_max/2.
        #TODO figure out what to do if k_max<2
        nint =min(np.int(self.k_max/2.),1000)
        t = ( np.arange(nint)+0.5 )/nint
        y = 1./t - 1.
        rk = y
        d2 = self.D2_L(rk,0.)
        x2 = y*y*r*r
        w1=np.exp(-x2)
        w2=2.*x2*w1
        w3=4.*x2*(1.-x2)*w1

        mult = d2/y/t/t
            
        sum1 = np.sum(w1*mult)/nint
        sum2 = np.sum(w2*mult)/nint
        sum3 = np.sum(w3*mult)/nint
                
        sig = np.sqrt(sum1)
        d1  = -sum2/sum1
        d2  = -sum2*sum2/sum1/sum1 - sum3/sum1
                
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
    
    #old way 
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
        return self.C.sigma8*self.C.sigma8*((q/q8)**(3.+p_index))*tk*tk/tk8/tk8

    def D2_NL_smith(self,rk,z,return_components = False):
        """
        halo model nonlinear fitting formula as described in 
        Appendix C of Smith et al. (2002)
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
                    
    def D2_NL(self,rk,z,return_components = False):
        """
        halo model nonlinear fitting formula as described in 
        Appendix C of Smith et al. (2002)
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
        #w = -0.758
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
        ph = a*y**(f1*3.)/(1.+b*y**(f2)+(f3*c*y)**(3.-gam))*(1+fnu*0.977)
        ph /= (1.+xmu*y**(-1)+xnu*y**(-2))
        pq = plin*(1.+plin)**beta/(1.+plin*alpha)*np.exp(-y/4.0-y**2/8.0)
        
        pnl=pq+ph

        if return_components:
            return pnl,pq,ph,plin
        else:
            return pnl
    def P_NL(self,k,z,return_components = False):
        if(return_components):
            pnl,pq,ph,plin=self.D2_NL(k,z,return_components)
            return pnl/k**3*(2*pi**2), pq, ph, plin/k**3*(2*pi**2)
        else:
            return D2_NL(k,return_components)/k**3*(2*pi**2)

                    

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
                CP=cp.CosmoPie()
                HF2=halofitPk(CP,k2,P2)
                HF=halofitPk(CP,k,P1)
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
                plt.show()
