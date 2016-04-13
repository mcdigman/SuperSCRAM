import numpy as np


class super_survey:
    ''' This class holds and returns information for all surveys
    ''' 
    def __init__(self, surveys):
        self.N_surveys=len(surveys) 
        self.N_tot_obs=0
        for i in range(self.N_surveys):
            self.N_tot_obs=self.N_tot_obs + surveys[i].N_obs
            
        
        


from FASTPTcode import FASTPT 

d=np.loadtxt('FASTPTcode/Pk_Planck15.dat') 
# 
k=d[:,0]; P=d[:,1]
# 	
# 	# set the parameters for the power spectrum window and
# 	# Fourier coefficient window 
P_window=np.array([.2,.2])  
C_window=.75	
# 	
# 	# bias parameter and padding length 
nu=-2; n_pad=800
# 
from time import time
# 		
# 	# initialize the FASTPT class		
fastpt=FASTPT.FASTPT(k,nu,n_pad=n_pad) 
# 	
# 	
t1=time()	
# 	
# 	# with out P_windowing (better if you are using zero padding) 
P_spt=fastpt.one_loop(P,C_window=C_window) 
t2=time()
print('time'), t2-t1 
 