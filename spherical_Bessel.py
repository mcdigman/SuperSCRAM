''' 
    Code to calculate spherical Bessel functions
    and the zeros of the spherical Bessel functions
    
    J. E. McEwen (c) 2016
'''


import numpy as np
from numpy import sin, cos, exp, pi
from scipy.misc import factorial2
from scipy.optimize import newton


def sph_Bessel_array(n,z):
    z=np.asarray(z, dtype=float)
    result=np.zeros(z.size)
    id1=np.where( z <= 1e-3)[0]
    id2=np.where( z > 1e-3)[0] 
    #print id1.size, id2.size, result[0], z[0]
    #limiting case for z near zero 
    if (id1.size !=0):
        result[id1]= z[id1]**n/factorial2(2*n+1, exact=True)
    if n==0:
        result[id2]=sin(z[id2])/z[id2]
    if n==1: 
        result[id2]=sin(z[id2])/z[id2]**2 - cos(z[id2])/z[id2]
    if n==2:
        result[id2]=(3/z[id2]**3 -1/z[id2])*sin(z[id2]) -3/z[id2]**2*cos(z[id2]) 
    if n >2:  
        j0=sin(z[id2])/z[id2]; j1=sin(z[id2])/z[id2]**2 - cos(z[id2])/z[id2] 
   
        j=np.zeros(id2.size)
        for i in range(1,n):
            j=(2*i+1)/z[id2]*j1-j0
            j0=j1
            j1=j
        
        result[id2]=j
         
    return result 
    
def sph_Bessel(n,z):
  
    # limiting case for z near zero 
    if ( z <= 1e-3):
        return z**n/factorial2(2*n+1, exact=True)
    if n==0:
        return sin(z)/z
    if n==1: 
        return sin(z)/z**2 - cos(z)/z
    if n==2:
        return (3/z**3 -1/z)*sin(z) -3/z**2*cos(z) 
    if n >2:  
        j0=sin(z)/z; j1=sin(z)/z**2 - cos(z)/z
   
        for i in range(1,n):
            j=(2*i+1)/z*j1-j0
            j0=j1
            j1=j
        
        return j
    
def j_n(n,z):
    if(z.size==1):
        return sph_Bessel(n,z)
    else:  
        return sph_Bessel_array(n,z)
        
def jn_zeros(n,n_zeros): 
    
    # fixed minimum and maximum range for zero finding 
    min=3; max=1e5
    
    # tolerance
    tol=1e-15
    zeros=np.zeros(n_zeros)
    
    def func(z):
        return j_n(n,z)
    for i in range(n_zeros):
        guess=1 + np.sqrt(2) + i*pi + n + n**(0.4)
        zeros[i]=newton(func, guess, tol=tol)
    
    return zeros

    
if __name__=="__main__": 
    
    # need to make sure calculations are complex and double precision 
    # this is not working all that well for values of z near zero 
    # only good for positive arguments 
   # print sph_Bessel(0,2.)
   # print sph_Bessel(1,2.)
   # print sph_Bessel(2,2.)
   # print sph_Bessel(3,2.)
   # print sph_Bessel(10,2.)
    
    import matplotlib.pyplot as plt
    
    z=np.linspace(0,10,200)
    
    ax=plt.subplot(111)
    ax.set_xlim(0,10)
    ax.set_ylabel(r'$j_n(z)$', size=30)
    ax.set_xlabel(r'$z$', size=30)
    
    ax.plot(z, j_n(0,z))
    ax.plot(z, j_n(1,z))
    ax.plot(z, j_n(2,z))
    ax.plot(z, j_n(3,z))
    ax.plot(z, j_n(4,z))
    
    x=jn_zeros(0,3)
    ax.plot(x,np.zeros(x.size),'o')
    x=jn_zeros(1,3)
    ax.plot(x,np.zeros(x.size),'o')
    x=jn_zeros(2,3)
    ax.plot(x,np.zeros(x.size),'o')
    x=jn_zeros(3,3)
    ax.plot(x,np.zeros(x.size),'o')
    x=jn_zeros(4,3)
    ax.plot(x,np.zeros(x.size),'o')
    
    plt.grid()
    plt.show()