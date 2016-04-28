''' 
    This class computes various cosmological parameters.
    The default cosmology is Planck 2015. 
    Joseph E. McEwen (c) 2016
    mcewen.24@osu.edu 
    
''' 

import numpy as np
from scipy.integrate import romberg, quad


class cosmopie :
    
    def __init__(self,cosmology=None):
        # default to Planck 2015 values 
        
        if cosmology is None:
        
            self.Omegabh2 = 0.02230
            self.Omegach2 = 0.1188
            self.Omegamh2 = 0.14170
            self.OmegaL   = .6911
            self.Omegam   = .3089
            self.H0       = 67.74 
            self.sigma8   = .8159 
            self.h        = .6774 
            self.Omegak   = 0.0 # check on this value 
            self.Omegar   = 0.0 # check this value too
        
        
        self.c        = 2.998*1e5
        
        self.DH       = self.c/self.H0
            
        
    def Ez(self,z):
        zp1=z + 1
        return np.sqrt(self.Omegam*zp1**3 + self.Omegar*zp1**4 + self.Omegak*zp1**2 + self.OmegaL) 
        
    
    def D_comov(self,z):
        # the line of sight comoving distance 
        I = lambda z : 1/self.Ez(z)
        return self.DH*quad(I,0,z)[0]
        
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
        
   
   
if __name__=="__main__": 

    C=cosmopie()
    z=.1
    print C.D_comov(z) 
    print C.D_A(z) 
    print C.D_L(z)      
        
        
    
    