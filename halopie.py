''' 
    This is a class to make halo objects for a given cosmology.
    Joseph E. McEwen (c) 2016
'''

import numpy as np
from numpy import pi, exp, log, log10,sin, cos,sqrt 
from scipy.special import sici
from scipy.integrate import trapz, quad 
from cosmopie import CosmoPie
import hmf 
import sys

d=np.loadtxt('Pk_Planck15.dat')
k=d[:,0]; P=d[:,1]
cp=CosmoPie(k=k, P_lin=P)
cosmo=cp

def N_cluster(M,zbins,Omega_s,dV,dndM):
    # return the number of cluster in redshift intervals 
    # \int_{z1}^{z2} dz dV/dz \int_{M_0}^\infty dM dn/dM
    def I(zp):
        mf=dndM(M,zp)
        n=trapz(mf*M,log(M))
        
        return Omega_s*(pi/180.)**2*dV(zp)*n
    
    N_i=np.zeros(zbins.size-1)
    z_i=np.zeros_like(N_i)
    for i in range(1,zbins.size):
        a=zbins[i-1]
        b=zbins[i]
        z_i[i-1]=(a+b)/2.
        N_i[i-1]=quad(I,a,b)[0]
         
    return z_i,N_i
       
def M_bar_lambda(C_lambda, a_lambda, b_lambda, llambda,z):
    # returns \bar{M(\lambda)}, \lambda is richness 
    return C_lambda + a_lambda*log(llambda/60.) + b_lambda*log(1+z) 
    
def pdf_Mz(params,llambda,M,z):
    C_lambda,a_lambda,b_lambda,sigma=params
    log_M_bar= M_bar_lambda(C_lambda, a_lambda, b_lambda, llambda,z)
    
    x=log(M)-log_M_bar
    return 1/M/sqrt(2*pi*sigma)*exp(-x**2/2./sigma)
    
def N_clus_rich(lambda_bin,M,zbins,Omega_s,dV,dndM,params):
    # llambda is richness 
    a,b=lambda_bin
            
    def I(zp):
    
        p=np.zeros_like(M)
        for i in range(M.size):
            def I_richness(richness):
                return pdf_Mz(params,richness,M[i],zp)
            p[i]=quad(I_richness,a,b)[0]
        
        mf=dndM(M,zp)*p
        n=trapz(mf*M,log(M))
        
        return Omega_s*(pi/180.)**2*dV(zp)*n
    
    N_i=np.zeros(zbins.size-1)
    z_i=np.zeros_like(N_i)
    for i in range(1,zbins.size):
        a=zbins[i-1]
        b=zbins[i]
        z_i[i-1]=(a+b)/2.
        N_i[i-1]=quad(I,a,b)[0]
         
    return z_i,N_i
    

    
def u_nfw(k,M,z,hmf):
    # Fourier transform of the NFW profile 
    # Taken from White 2001 
    
    # inputs are the wavevector k
    # the halo mass M 
    # the virial radius R
    # the concentration mass relation
    
    # the returned array as shape of (k.size,M.size)
    c=concentration(M,z,hmf)
    r=R_virial(M,z)
    k=k.reshape((k.size,1))
    x=k*r/c
    
    Si,Ci=sici((1+c)*x)
    si,ci=sici(x)
    
    log_part=log(1+c)-c/(1+c)
    
    S=sin(x)*(Si-si)
    C=cos(x)*(Ci-ci)
 
    return 1/log_part*(S + C - sin(c*x)/(1+c)/x)

def concentration(M,z,hmf):

    c0=9.0
    beta=-0.13
    # dummy M_star now
    M_star=hmf.M_star()
    return c0/float(1+z)*(M/M_star)**beta

def R_virial(M,z):
    
    delta_v=cosmo.delta_v(z)
    rho_bar=cosmo.rho_bar(z)
    
    r=3*M/4./(pi*delta_v*rho_bar)
    return r**(1/3.)
    

if __name__=="__main__":  
    
    degree_to_radian=0.0174533
    C_lambda=33.6
    a_lambda=1.08
    b_lambda=0.0
    sigma=0.25
    Omega_s=18000/2.

    params=[C_lambda,a_lambda,b_lambda,sigma]
    lambda_bin=[14,16]

    
    dV=cp.DV
    hmf=hmf.ST_hmf(cp)
    dndM=hmf.dndM
    
    M=np.logspace(13.5,16,40)
    dz=.3; N=20
    zbins=np.arange(0,3,.3)
    
    z,N=N_cluster(M,zbins,Omega_s,dV,dndM)
    #z,N2=N_clus_rich(lambda_bin,M,zbins,Omega_s,dV,dndM,params)
    
    
    import matplotlib.pyplot as plt
    ax=plt.subplot(121)
    ax.set_yscale('log')
    
    ax.plot(z,N)
    ax.plot(z,N,'o')
    
   # ax.plot(z,N2)
   # ax.plot(z,N2,'o')
    
    plt.grid()
    
    
    ax=plt.subplot(122)
    ax.set_yscale('log')
    ax.set_xscale('log')
    k=np.logspace(-3,3,200)
    
    M=10**np.array([11,12,13,14,15,16])
    u=u_nfw(k,M,0,hmf)
    
   
    ax.plot(k,u[:,0])
    ax.plot(k,u[:,1])  
    ax.plot(k,u[:,2])
    ax.plot(k,u[:,3])
    ax.plot(k,u[:,4])
    ax.plot(k,u[:,5])
    
    plt.grid()   
    plt.show()
    
    
    
    