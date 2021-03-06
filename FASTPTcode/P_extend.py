''' 
    This module extends the power spectrum down to lower and higher k, 
    by calculating the power law index at both ends of the array. The extension is to 
    help with edge effects and should be removed when returned from FASTPT.  
    
    J.E. McEwen (c) 2016
    mcewen.24@osu.edu 
'''
from __future__ import division,absolute_import,print_function
from builtins import range

import numpy as np
from numpy import log, exp, log10 
import sys


class k_extend: 

    def __init__(self,k,low=None,high=None):
        
        # if (log10(k[0]) > -2):
        #     print('you should consider exteneding the power spectrum down to lower k values,so that it \
        #         is assured you are in the linear regime') 
        # if (log_end >= log10(k[0])):
        #     print('your k-vector already extends to the desired k_min')
            
        
        self.DL=log(k[1])-log(k[0]) 

       
        if low is not None:
            N=np.absolute(int((log10(k[0])-low)/self.DL))
            # if N is even add one more point, because when 
            # np.arange(1,N) is added, it will result in an 
            # even number of grid points
            if (N % 2 == 0 ):
                N=N+1 
            s=log(k[0]) - np.arange(1,N)*self.DL 
            s=s[::-1]
            self.k_min=k[0]
            self.k_low=exp(s) 
            self.k=np.append(self.k_low,k)
            self.id_extrap=np.where(self.k >=self.k_min)[0] 
            k=self.k
            

        if high is not None:
            N=np.absolute(int((log10(k[-1])-high)/self.DL))
            # if N is even add one more point, because when 
            # np.arange(1,N) is added, it will result in an 
            # even number of grid points
            if (N % 2 == 0 ):
                N=N+1 
            s=log(k[-1]) + np.arange(1,N)*self.DL 
            self.k_max=k[-1]
            self.k_high=exp(s)
            self.k=np.append(k,self.k_high)
            self.id_extrap=np.where(self.k <= self.k_max)[0] 
            

        if (high is not None) & (low is not None):
            self.id_extrap=np.where((self.k <= self.k_max) & (self.k >=self.k_min))[0]
            
            
    def extrap_k(self):
        return self.k 
        
    def extrap_P_low(self,P):

        ns=(log(P[1])-log(P[0]))/self.DL
        Amp=P[0]/self.k_min**ns
        P_low=self.k_low**ns*Amp
        return np.append(P_low,P) 

    def extrap_P_high(self,P):

        ns=(log(P[-1])-log(P[-2]))/self.DL
        Amp=P[-1]/self.k_max**ns
        P_high=self.k_high**ns*Amp
        return np.append(P,P_high) 
    
    def PK_orginal(self,P): 
        return self.k[self.id_extrap], P[self.id_extrap]
    

    
