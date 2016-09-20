import numpy as np

class SourceDistribution:
    def __init__(self,ps):
        self.ps = ps

#gaussian source distribution
class GaussianZSource(SourceDistribution):
    def __init__(self,C,zbar=1.0,sigma=0.4):
        self.ps = np.exp(-(self.zs-zbar)**2/(2.*(sigma)**2))
        for i in range(0,self.n_z-1): #compensate for different bin sizes
                       self.ps[i] = self.ps[i]/(self.chis[i+1]-self.chis[i])
        self.ps[-1] = self.ps[-1]/(C.D_comov(2*self.zs[-1]-self.zs[-2])-self.chis[-1]) #patch for last value
        self.ps = self.ps/np.trapz(self.ps,self.chis) #normalize galaxy probability distribution

#constant source distribution
class ConstantZSource(SourceDistribution):
    def __init__(self,C):
        for i in range(0,self.n_z-1):
            self.ps[i] = 1./(self.chis[i+1]-self.chis[i])
        self.ps[-1] = 1./(C.D_comov(2*self.zs[-1]-self.zs[-2])-self.chis[-1])
        self.ps = self.ps/np.trapz(self.ps,self.chis) #normalize galaxy probability distribution

#source distribution from cosmolike paper
#cosmolike uses alpha=1.3, beta=1.5, z0=0.56
class CosmolikeZSource(SourceDistribution):
    def __init__(self,C,alpha=1.24,beta=1.01,z0=0.51):
        self.ps = self.zs**alpha*np.exp(-(self.zs/z0)**beta)
        for i in range(0,self.n_z-1):
            self.ps[i] = self.ps[i]/(self.chis[i+1]-self.chis[i])
        self.ps[-1] = self.ps[i]/(C.D_comov(2*self.zs[-1]-self.zs[-2])-self.chis[-1])
        self.ps = self.ps/np.trapz(self.ps,self.chis) #normalize galaxy probability distribution
