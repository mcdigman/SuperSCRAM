import numpy as np
from spherical_Bessel import j_n, jn_zeros

class delta_alpha(object): 
    
    def __init__(self,r_max,n_alpha,n_zeros):
        
        k_alpha=np.zeros(n_alpha.size*n_zeros)
        

        for i in range(n_alpha.size):
            j=i*n_zeros
            k_alpha[j:j+n_zeros]=jn_zeros(n_alpha[i],n_zeros)/r_max
        
        print k_alpha 
        
            
if __name__=="__main__":

    R=delta_alpha(10,np.array([0,1]),3)
    
        
