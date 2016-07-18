import numpy as np
from numpy import pi 
from hmf import ST_hmf
import sys

def volume(r1,r2,Theta,Phi):
    phi1,phi2=Phi
    theta1,theta2=Theta
    result=(phi2-phi1)*(np.cos(theta1)- np.cos(theta2))*(r2**3-r1**3)/3. 
    return result 

class DO_n:
    def __init__(self, n_obs,zbins,mass,CosmoPie,basis,geo): 
        # the n_obs_dict should contain
        # the following information:
        # number of objects in z-bins
        # the sky coverage \Omega_s 
        # mass is the cutoff mass, will consider the number of objects below mass=mass
        
     
        self.mf=ST_hmf(CosmoPie)
        n_zbins=zbins.size
        
        n1=n_obs[0]
        n2=n_obs[1]
        Theta=geo[0]; Phi=geo[1]
        print n1, n2, zbins, Theta, Phi
          
        V1=np.zeros(n_zbins-1)
        
        r_min=np.zeros(n_zbins-1)
        r_max=np.zeros(n_zbins-1)
        for i in range(1,n_zbins):
            z1=zbins[i-1]
            z2=zbins[i]
            r1=CosmoPie.D_comov(z1)
            r2=CosmoPie.D_comov(z2)
            r_min[i-1]=r1
            r_max[i-1]=r2
            V1[i-1]=volume(r1,r2,Theta,Phi) # am I doing this value correctly ? 
            
        
        V2=np.zeros_like(V1)
        V2=V1
        bias=np.zeros_like(V1)
        n_avg=np.zeros_like(V1)
        
        d_delta=np.zeros(n_zbins-1,dtype=object)
        for i in range(n_zbins):
            
            if i > 0:

                z_avg=(zbins[i]+zbins[i-1])/2.
                bias[i-1]=self.mf.bias(mass,z_avg)
                n_avg[i-1]=self.mf.n_avg(mass,z_avg)

                d_delta[i-1]=basis.D_delta_bar_D_delta_alpha(r_min[i-1],r_max[i-1],Theta,Phi)
        
        print ' dn number n1, n2, n_avg, bias', n1/V1, n2/V2, n_avg, bias 
        print ' v1,v2', V1,V2    

        self.DO=(n1/V1 - n2/V2)/n_avg**2*bias*d_delta
        
        print self.DO
        self.N_ab=1/n_avg*(1/V1 + 1/V2)  #  N_ab
        
        self.F_alpha_beta=np.zeros(zbins.size-1,dtype=object)
        for i in range(zbins.size-1):
            v=self.DO[i]
            self.F_alpha_beta[i]=np.outer(v,v)*1./self.N_ab[i]
           
        
    def dn_ddelta(self):
        return self.DO
        
    def Fisher_alpha_beta(self):
        return self.F_alpha_beta 
                
        

        
if __name__=="__main__":
    print 'hello'
    
    from defaults import cosmology 
    from cosmopie import CosmoPie
    d=np.loadtxt('Pk_Planck15.dat')
    k=d[:,0]; P=d[:,1]
    
    Omega_s1=18000
    z_bins1=np.array([.1,.2,.3])
    N1=np.array([1e6, 5*1e7]) 
   
    
    S1={'name':'LSST', 'sky_frac': Omega_s1, 'z_bins': z_bins1, 'N': N1}
    
    Omega_s2=18000/2.
    z_bins2=np.array([.1,.2,.35,.5])
    N2=np.array([1e7, 5*1e7]) 
  
    S2={'name':'WFIRST', 'sky_frac': Omega_s2, 'z_bins': z_bins2, 'N': N2}
    
    n_obs_dict=[S1, S2]
    
    cp=CosmoPie(cosmology, k=k, P_lin=P)
    r_max=cp.D_comov(4)
    l_alpha=np.array([1,2,3,4,5])
    n_zeros=3
    
    from sph import sph_basis
    geo=sph_basis(r_max,l_alpha,n_zeros,k,P)
    
    O=DO_n(n_obs_dict,1e15,cp,geo)
    
    
