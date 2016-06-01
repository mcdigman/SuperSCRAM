''' 
    class to make HOD power spectra
'''
from scipy.special import erf
from numpy import exp, log10 
from hmf import hmf 
import numpy as np

class HOD():
    def __init__(self, HOD_data=None):
        if HOD_data == None: 
            # default to Zehavi parameters 
            self.log_M_min=12.24
            self.sigma=0.15
            self.log_M_0=12.14
            self.log_M1=13.43
            self.alpha=1.0
            self.HOD_type='Zheng'
    
    def N_cen(self,M):
        # M input is halo mass
        return 0.5*(1 + erf((log10(M)-self.log_M_min)/self.sigma))
        
    def N_sat(self,M):
         # M input is halo mass
        return ((M-10**self.log_M_0)/(10**self.log_M1))**self.alpha
            
        
        
if __name__=="__main__":
    M=np.linspace(11,15, 50)
    M=10**M
    
    import matplotlib.pyplot as plt
    
    ax=plt.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-1,300)
    
    hod=HOD()
    C=hod.N_cen(M)
    S=hod.N_sat(M)
    
    ax.plot(M,C)
    ax.plot(M,S)
    
    plt.show()