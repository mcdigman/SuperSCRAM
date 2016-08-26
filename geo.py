import numpy as np
import cosmopie as cp
from scipy.integrate import dblquad

# the smallest value 

class geo:
    def __init__(self,zs,volumes,v_total,rs,C):
        self.zs = zs
        self.C = C
        self.volumes = volumes
        self.v_total = v_total
        self.rs = rs
        #list r rbins as [rmin,rmax] pairs for convenience
        self.rbins = np.zeros((rs.size-1,2))
        self.zbins = np.zeros((zs.size-1,2))
        for i in range(0,rs.size-1):
            self.rbins[i,0] = self.rs[i]
            self.rbins[i,1] = self.rs[i+1]
            self.zbins[i,0] = self.zs[i]
            self.zbins[i,1] = self.zs[i+1]

        self.eps=np.finfo(float).eps
    def surface_integral(self,function):
        raise NotImplementedError, "Subclasses of geo should implement surface_integral"

    def angular_area(self):
        return self.surface_integral(lambda theta,phi: 1.0)

    #volume ddeltabar_dalpha
class rect_geo(geo): 
    def __init__(self,zs,Theta,Phi,C):
            self.Theta = Theta
            self.Phi = Phi
            
            phi1,phi2=self.Phi
            theta1,theta2=self.Theta

            rs = np.zeros(zs.size)
            for i in range(0,zs.size):
                rs[i] = C.D_comov(zs[i])
            
            
            volumes = np.zeros(zs.size-1)
            for i in range(0,volumes.size):
                volumes[i] = (phi2-phi1)*(np.cos(theta1)- np.cos(theta2))*(rs[i+1]**3-rs[i]**3)/3.

            v_total=(phi2-phi1)*(np.cos(theta1)- np.cos(theta2))*(rs[-1]**3-rs[0]**3)/3.


            geo.__init__(self,zs,volumes,v_total,rs,C)

    #function(phi,theta)
    def surface_integral(self,function):
        I=dblquad(function,self.Theta[0],self.Theta[1], lambda phi: self.Phi[0], lambda phi: self.Phi[1])[0]
        if (np.absolute(I) <= self.eps):
            return 0.0
        else:
            return I

