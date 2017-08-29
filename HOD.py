''' 
    class to make HOD power spectra
'''
from scipy.special import erf
from numpy import exp, log10 
from hmf import ST_hmf
import numpy as np
from halopie import u_nfw
from scipy.integrate import trapz, quad
import sys


class HOD():
    def __init__(self,mass_func,CosmoPie,HOD_data=None):
        if HOD_data == None: 
            # default to Zehavi parameters 
            self.log_M_min=12.24
            self.sigma=0.15
            self.log_M_0=12.14
            self.log_M1=13.43
            self.alpha=1.0
            self.HOD_type='Zheng'
            
        
        self.mass_func=mass_func
        self.CosmoPie=CosmoPie
    
    def N_cen(self,M):
        # M input is halo mass
        return 0.5*(1 + erf((log10(M)-self.log_M_min)/self.sigma))
        
    def N_sat(self,M):
         # M input is halo mass
        return ((M-10**self.log_M_0)/(10**self.log_M1))**self.alpha
        
    def N_avg(self,M):
        return self.N_cen(M) + self.N_sat(M) 
        
    def moment_two(self,M):
        N=self.N_sat(M)
        return N*(2 + N)
        
    def b_h(self,M):
        # return one for now 
        return 1 
    
    def P_cluster(self,k,P_lin,M,z,hmf):
        G=self.CosmoPie.G_norm(z)
        print 'this is growth', G
        rho_bar=self.CosmoPie.rho_bar(z)
        mf=mass_func.dndM(M,z)
        u=u_nfw(k,M,z,hmf)
        
        norm=trapz(mf,M)
        
        I_1=trapz(self.b_h(M)*mf,M)/norm
        
        I_2=np.zeros_like(k)
        
        for i in xrange(k.size):
            
            I_2[i]=trapz(mf*M/rho_bar*u[i],M)/norm
        
        return I_1*G**2*P_lin + I_2
    
    def P_gg(self,k,P_lin,M,z,hmf):
        h=.6774
        G=self.CosmoPie.G_norm(z)
        mf=mass_func.dndM(M,z)
        u=u_nfw(k,M,z,hmf)
        norm=trapz(u,k,axis=0)
        print norm.size, M.size
        u=u/norm
       
        print trapz(u,k,axis=0)
    
        #sys.exit()
        n_avg=self.N_avg(M)
        n=self.moment_two(M)
        n_bar=trapz(mf*n_avg,M)
        b=hmf.bias(M,z)
      
        I_1h=np.zeros_like(k)
        I_2h=np.zeros_like(k)
        for i in xrange(k.size):
            
            I_1h[i]=trapz(mf*n*np.absolute(u[i,:]*u[i,:]),M)
            I_2h[i]=trapz(mf*n_avg*b*u[i,:],M)
        

        return G**2*P_lin*I_2h**2/n_bar**2 #+ I_1h/n_bar**2
        
        
        
        
        
    
            
if __name__=="__main__":
    
    
    from cosmopie import CosmoPie
    d=np.loadtxt('Pk_Planck15.dat')
    k=d[:,0]; P=d[:,1]
    
    cp=CosmoPie(k=k,P_lin=P)
    mass_func=ST_hmf(cp)
    
    M=np.linspace(11,15, 100)
    M=10**M
    
    import matplotlib.pyplot as plt
    
    ax=plt.subplot(121)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-1,300)
    
    hod=HOD(mass_func,cp)
    C=hod.N_cen(M)
    S=hod.N_sat(M)
    N=hod.N_avg(M)
    
    ax.plot(M,C)
    ax.plot(M,S)
    ax.plot(M,N,'--')
    plt.grid()
    
    ax=plt.subplot(122)
    ax.set_xlim(k[0],1)
    ax.set_xscale('log')
    ax.set_yscale('log')
   
    
    P_gg=hod.P_gg(k,P,M,0,mass_func)
    
    ax.plot(k,P)
    ax.plot(k,P_gg)
    plt.grid()
  
    plt.show()
