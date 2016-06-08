import numpy as np
from numpy import sqrt, pi, sin, cos, tan, tanh, sinh, cosh 
from scipy.interpolate import interp1d, interp2d, spline, splrep, splev
from scipy.integrate import quad , trapz
import sys


class projected_power:
    
    def __init__(self,k,P_in,CosmoPie,z_max, P_type='linear'):
    
    
        if P_type=='linear':
            z=np.linspace(0,z_max,500)
            G=CosmoPie.G_array(z)
            G=G**2
            P_in=P_in.reshape(k.size,1)
            P_grid=G*P_in
            
        self.k=k          
        self.P=interp2d(k,z,np.ravel(P_grid))
        
        self.z=np.linspace(1e-3,z_max,500)
        DA=CosmoPie.D_A_array(self.z)
        DC=CosmoPie.D_c_array(self.z)
        
        self.DA=interp1d(z,DA)
        self.DC=interp1d(z,DC)

        self.c=CosmoPie.c
        self.H0=CosmoPie.H0
        self.CosmoPie=CosmoPie
        self.c=CosmoPie.c
        self.Omegam=CosmoPie.Omegam
    
        
        cotK=np.zeros_like(self.z)
        for i in range(self.z.size):
            cotK[i]=self.cot_K(DC[i],self.z[i])
       
        self.cotK=interp1d(z,cotK)
 
    def Power(self,k,z):
        return self.P(k,z)
    
    def P_lz(self,l,DA,z):
        P=np.zeros_like(z)
        for i in range(z.size):
            k=l/DA[i]
            P[i]=self.P(k,z[i])
        return P 
    
    def Heaviside(self,x1,x2):
        if x2 >= x1 : 
            return 1.0
        else :
            print 'crap'
            return 0.0
        
    def K(self,z):
        return -self.Omegak(z)*(self.c/self.H0)**(-2)
    def Omegak(self,z):
        return self.CosmoPie.Omegak_z(z)        
        
    def cot_K(self,D_c,z):
        K=self.K(z)  
        if K < 1e-7:
            K=0.0 
      
        sqrtK=sqrt(np.absolute(K))
        if K==0:
            return D_c**(-1)
        if K < 0 : 
            return 1/tan(D_c/sqrtK)/sqrtK
        if K > 0:
            return 1/np.absolute(sqrtK)/tanh(D_c/sqrtK)/sqrtK
    
    def W_lens(self,cotKDC1, cotKDC,DA,z):
        amp=3/2.*self.CosmoPie.Omegam_z(z)*self.H0**2*(1+z)/self.c/self.c
        return amp*(cotKDC1-cotKDC)*DA**2
        
     
    def C_EE(self,z_source,l):
        I=np.zeros_like(l)
        dz=.005
        nz=int(z_source/dz+1)
        z=np.linspace(1e-4,z_source,nz)
        cotK_DC=self.cotK(z_source)
        cotK_DC1=self.cotK(z)
        DA=self.CosmoPie.D_A_array(z)
        DC1=self.CosmoPie.D_c_array(z)
        
        W=self.W_lens(cotK_DC1, cotK_DC, DA,z)
        
        p_array=np.zeros(nz)
        for i in range(l.size):
        
            for j in range(z.size):
            
                k=l[i]/DA[j]
                p_array[j]=self.P(k,z[j])[0]
                
            integrand=(W)**2*p_array/DA**2
            I[i]=trapz(integrand,DC1)
        return I 
        
        
     
if __name__=="__main__":
    
    from cosmopie import CosmoPie
    cp=CosmoPie()
    z_max=3
    d=np.loadtxt('Pk_Planck15.dat')
    k=d[:,0]; P=d[:,1]
    
    h=.6774
    pp=projected_power(k,P,cp,z_max)
    l=np.arange(2,3000)
    l=np.logspace(np.log10(2),3,500)
    C_EE1=pp.C_EE(1,l)
    C_EE5=pp.C_EE(.5,l)
    C_EE2=pp.C_EE(2,l)
    
    
    import matplotlib.pyplot as plt
    k=np.logspace(-4,1,500)
    
    P1=pp.Power(k,0)
    P2=pp.Power(k,1)
    P3=pp.Power(k,2)
    

    ax=plt.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.plot(l, l*(l+1)*C_EE5/(2*pi), label='z=.5')
    ax.plot(l, l*(l+1)*C_EE1/(2*pi), label='z=1.0')
    ax.plot(l, l*(l+1)*C_EE2/(2*pi), label='z=2.0')
   
    plt.legend(loc=4)
    plt.grid()
    plt.show()
        
        
    
    