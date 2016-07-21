import numpy as np
import cosmopie as cp
from scipy.integrate import dblquad

# the smallest value 

class geo:
    def __init__(self,zbins,volumes,v_total,rs,C=cp.CosmoPie()):
        self.zbins = zbins
        self.C = C
        self.volumes = volumes
        self.v_total = v_total
        self.rs = rs
        self.eps=np.finfo(float).eps
    def surface_integral(self,function):
        raise NotImplementedError, "Subclasses of geo should implement surface_integral"


    #volume ddeltabar_dalpha
class rect_geo(geo): 
    def __init__(self,zbins,Theta,Phi,C=cp.CosmoPie()):
            self.Theta = Theta
            self.Phi = Phi
            
            phi1,phi2=self.Phi
            theta1,theta2=self.Theta

            rs = np.zeros(zbins.size)
            for i in range(0,zbins.size):
                rs[i] = C.D_comov(zbins[i])
            
            
            volumes = np.zeros(zbins.size-1)
            for i in range(0,volumes.size):
                volumes[i] = (phi2-phi1)*(np.cos(theta1)- np.cos(theta2))*(rs[i+1]**3-rs[i]**3)/3.

            v_total=(phi2-phi1)*(np.cos(theta1)- np.cos(theta2))*(rs[-1]**3-rs[0]**3)/3.


            geo.__init__(self,zbins,volumes,v_total,rs,C)

    #function(phi,theta)
    def surface_integral(self,function):
        I=dblquad(function,self.Theta[0],self.Theta[1], lambda phi: self.Phi[0], lambda phi: self.Phi[1])[0]
        if (np.absolute(I) <= self.eps):
            return 0.0
        else:
            return I

