'''
    Code to calculate spherical functions, such as:
    spherical Bessel functions
    the zeros of the Bessel functions
    the real spherical harmonics
    the derivatives of the Bessel function
    J. E. McEwen (c) 2016
'''

import numpy as np

from scipy.misc import factorial2
from scipy.special import sph_harm as Y_lm, jv

from the_mad_house import e_message

eps =np.finfo(float).eps
z_cut=1e-2

#data=np.loadtxt('data/spherical_bessel_zeros_360.dat')
data=np.loadtxt('data/spherical_bessel_zeros_527.dat')
data[data<0.] = np.inf

l_max_zeros = data.shape[0]-1
def sph_Bessel_down(n,z):
    n_start=n + int(np.sqrt(n*40)/2.)

    j2=0
    j1=1

    for i in xrange(n_start,-1,-1):
    # for loop until n=0, so that you can renormalize
        j0=(2*i+3)/z*j1 - j2
        j2=j1
        j1=j0
        if i==n:
            result=j0

        j0_true=np.sin(z)/z
    # renormalize and return result
    return result*j0_true/j0

#print 'this is eps', eps
#print 'check eps', 1+eps
def sph_Bessel_array(n,z):
    z=np.asarray(z, dtype=float)
    result=np.zeros(z.size)

    # id for limiting cases
    id1=np.where( z <= z_cut)[0]
    id2=np.where( z > z_cut)[0]

    if id1.size !=0:
        result[id1]= z[id1]**n/factorial2(2*n+1, exact=True)
    if n==0:
        result[id2]=np.sin(z[id2])/z[id2]
    if n==1:
        result[id2]=np.sin(z[id2])/z[id2]**2 - np.cos(z[id2])/z[id2]
    if n==2:
        result[id2]=(3/z[id2]**3 -1/z[id2])*np.sin(z[id2]) -3/z[id2]**2*np.cos(z[id2])
    if n >2:

        if n > np.real(z):
            return sph_Bessel_down(n,z)

        else:
            j0=np.sin(z[id2])/z[id2]
            j1=np.sin(z[id2])/z[id2]**2 - np.cos(z[id2])/z[id2]

            j=np.zeros(id2.size)
            for i in xrange(1,n):
                j=(2*i+1)/z[id2]*j1-j0
                j0=j1
                j1=j

            result[id2]=j

        return result

def sph_Bessel(n,z):
    # limiting case for z near zero
    if  z <= z_cut:
        return z**n/factorial2(2*n+1, exact=True)
    if n==0:
        return np.sin(z)/z
    if n==1:
        return np.sin(z)/z**2 - np.cos(z)/z
    if n==2:
        return (3/z**3 -1/z)*np.sin(z) -3/z**2*np.cos(z)
    if n >2:

        if n > np.real(z):
            return sph_Bessel_down(n,z)

        else :
        # do upward recursion
            j0=np.sin(z)/z
            j1=np.sin(z)/z**2 - np.cos(z)/z

            for i in xrange(1,n):
                j=(2*i+1)/z*j1-j0
                j0=j1
                j1=j
            return j

#TODO make sure array size logic is consistent
#TODO support vector n
def j_n(n,z):
    z = np.asarray(z)
    #j_n is well defined if z is zero, must avoid dividing by zero
    if np.any(z==0.):
        if np.isscalar(z):
            z_use=np.array([z])
        else:
            z_use = z
        result = np.zeros_like(z_use)
        result[z_use!=0.] = np.sqrt(np.pi/(2*z_use[z_use!=0.]))*jv(n+0.5,z_use[z_use!=0])
        if n==0:
            result[z_use==0.] = 1.
        if np.isscalar(z):
            return result[0]
        else:
            return result
    else:
        return np.sqrt(np.pi/(2*z))*jv(n+0.5,z)

    #if(z.size==1):
    #    return np.sqrt(np.pi/(2*z))*jv(n+0.5,z)
    #else:
        #result=np.zeros_like(z)
        #for i in xrange(z.size):
        #    result[i]=np.sqrt(np.pi/(2*z[i]))*jv(n+0.5,z[i])
    #    result=np.sqrt(np.pi/(2*z))*jv(n+0.5,z)
    #    return result

def dj_n(n,z):

    z=np.asarray(z)

    if z.size==1:
        a=j_n(n,z)
        b=j_n(n+1,z)
        c=j_n(n-1,z)
        return (c-b)/2. -a/2./z

    else:
        a=j_n(n,z)
        b=j_n(n+1,z)
        c=j_n(n-1,z)
        return (c-b)/2. -a/2./z

#return all zeros in a row smaller than a cut

def jn_zeros_cut(l,q_lim):
    if l<=l_max_zeros:
        return data[l,data[l]<q_lim]
    else:
        raise IndexError('l is greater than the number of ls in the bessel functions lookup table,choose lower ceiling or expand the lookup table')

def jn_zeros(l,n_zeros):
    if l<=l_max_zeros:
        return data[l,:n_zeros]
    else:
        raise IndexError('l is greater than the number of ls in the bessel functions lookup table,choose lower ceiling or expand the lookup table')
#     print 'this is n', n
#     # fixed minimum and maximum range for zero finding
#     min=3; max=1e5
#
#     # tolerance
#     tol=1e-15
#     zeros=np.zeros(n_zeros)
#
#     def func(z):
#         if z < np.pi:
#             return np.infty
#         return j_n(n,z)
#
#     def J_func(z):
#         return dj_n(n,z)
#
#     for i in xrange(n_zeros):
#         guess=1 + np.sqrt(2) + i*np.pi + n + n**(0.4)
#
#         zeros[i]=newton(func, guess,tol=tol)

def dJ_n(n,z):
    # check this returns correct value for derivative of Big J_n(z)
    return (n/z)*jv(n,z)-jv(n+1,z)

def Y_r(l,m,theta,phi):
    # real spherical harmonics, using scipy spherical harmonic functions
    # the scipy spherical harmonic inputs are ordered as
    # Y_lm(m,l,phi,theta)
    if np.abs(m) > l:
        print e_message()
        print('You must have |m| <=l for spherical harmonics.')
        raise ValueError('Please check the function values you sent into Y_r in module sph_functions.py')
        #sys.exit()

    if m==0.0:
        return np.real(Y_lm(m,l,phi,theta))
        #if np.real(result) < eps:
        #    result =0
        #return result
        #print 'check', result
        #if result < eps:
        #    result=0
        #return result
    if m<0:
        result =1j/np.sqrt(2)*(Y_lm(m,l,phi,theta) - (-1)**m*Y_lm(-m,l,phi,theta))
        #TODO check if eps check is done correctly
        result = np.real(result)
        #result[np.abs(result)<eps] = 0.
        #if np.absolute(np.real(result)) < eps:
        #    result=0
        #return np.sqrt(2)*(-1)**m*np.imag(Y_lm(m,l,phi,theta))
        return result
    if m>0:
        result =1/np.sqrt(2)*(Y_lm(-m,l,phi,theta)  + (-1)**m*Y_lm(m,l,phi,theta))
        result = np.real(result)
        #result[np.abs(result)<eps] = 0.
        #return np.sqrt(2)*(-1)**m*np.real(Y_lm(m,l,phi,theta))
        #if np.absolute(np.real(result)) < eps:
        #    result=0
        return result

#Daniel added - radial part of tidal force
def S_rr(l,m,theta,phi,k,r):
    return (Y_r(l,m,theta,phi)/4.) * (2*j_n(l,k*r) - j_n(l+2,k*r) - j_n(l-2,k*r))

#if __name__=="__main__":
#
#    print('check spherical Bessel against mathematica output')
#    print('function values',j_n(0,2.))
#    print('mathematica value', 0.454649)
#    print('function values',j_n(1,2.))
#    print('mathematica value', 0.435398)
#    print('function values',j_n(2,2.))
#    print('mathematica value', 0.198448)
#    print('function values',j_n(3,2.))
#    print('mathematica value', 0.0607221)
#    print('function values',j_n(10,2.))
#    print('mathematica value', 6.8253e-8)
#    print('function values',j_n(50,101.5))
#    print('mathematica value', -0.0100186)
#
#    print('check derivative of Bessel against keisan.casio.com')
#    print('function values', dJ_n(0,1))
#    print('true value', -0.4400505857449335159597)
#    print('function values', dJ_n(3,11.5))
#    print('true value', -0.0341759332779211515933)
#    print('function values', dJ_n(5,3.145))
#    print('true value', 0.0686374928139798052691)
#
#    y=lambda phi : np.sin(phi)
#    x=lambda phi : np.cos(phi)
#    z=lambda theta : np.cos(theta)
#    theta=3*np.pi/2; phi=3*np.pi/4
#
#    # check the values for the real spherical harmonics
#    # checking against values on https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#l_.3D_0.5B2.5D.5B3.5D
#    # check against wiki values
#    print 'check l=0 case'
#    print 'function value', Y_r(0,0,theta,phi)
#    print 'wiki value', .5*np.sqrt(1/np.pi)
#    print '------------------------------------------'
#    # l=1 case
#    print 'check l=1, m=-1 case'
#    print 'function value', Y_r(1,-1,theta,phi)
#    print 'wiki value', np.sqrt(3/4./np.pi)*y(phi)
#    print '------------------------------------------'
#    print 'check l=1, m=0 case'
#    print 'function value', Y_r(1,0,theta,phi)
#    print 'wiki value', np.sqrt(3/4./np.pi)*z(theta)
#    print '------------------------------------------'
#    print 'check l=1, m=1 case'
#    print 'function value', Y_r(1,1,theta,phi)
#    print 'wiki value', np.sqrt(3/4./np.pi)*x(phi)
#    print '------------------------------------------'
#    print 'check l=2, m=-2 case'
#    print 'function value', Y_r(2,-2,theta,phi)
#    print 'wiki value', .5*np.sqrt(15/np.pi)*x(phi)*y(phi)
#    print 'check l=2, m=-1 case'
#    print 'function value', Y_r(2,-1,theta,phi)
#    print 'wiki value', .5*np.sqrt(15/np.pi)*z(theta)*y(phi)
#    print 'check l=2, m=0 case'
#    print 'function value', Y_r(2,0,theta,phi)
#    print 'wiki value', .25*np.sqrt(5/np.pi)*(2*z(theta)**2 - y(phi)**2 - x(phi)**2)
#    print 'check l=2, m=1 case'
#    print 'function value', Y_r(2,1,theta,phi)
#    print 'wiki value', .5*np.sqrt(15/np.pi)*z(theta)*x(phi)
#    print 'check l=2, m=2 case'
#    print 'function value', Y_r(2,2,theta,phi)
#    print 'wiki value', .25*np.sqrt(15/np.pi)*(x(phi)**2-y(phi)**2)
#    print '------------------------------------------'
#    #print 'check an incorrect m and l value'
#    #print Y_r(2,3,theta,phi)
#    print '------------------------------------------'
#    print 'check normalization'
#    from scipy.integrate import nquad
#    def norm_check(m1,l1,m2,l2):
#        def func(theta,phi):
#            return np.sin(theta)*Y_r(l1,m1,theta,phi)*Y_r(l2,m2,theta,phi)
#
#        def funcII(theta,phi):
#            return np.sin(theta)*Y_lm(m1,l1,phi,theta)*np.conjugate(Y_lm(m2,l2,phi,theta))
#
#
#        I=nquad(func,[[0,np.pi],[0,2*np.pi]])[0]
#        #print 'check against spherical harmonics',nquad(funcII,[[0,np.pi],[0,2*np.pi]])[0]
#        if I < eps:
#            I=0
#        return I
#
#    print 'check normalization, 1,1,1,1:', norm_check(1,1,1,1)
#    print 'check normalization, 0,1,0,1:', norm_check(0,1,0,1)
#    print 'check normalization, 0,2,0,2:', norm_check(0,2,0,2)
#    print 'check normalization, 1,2,1,2:', norm_check(1,2,1,2)
#    print 'check normalization, 2,2,2,2:', norm_check(2,2,2,2)
#    print 'check normalization, -1,2,-1,2:', norm_check(-1,2,-1,2)
#    print 'check normalization, -1,2,0,2:', norm_check(-1,2,0,2)
#    print 'check normalization, -1,2,-1,3:', norm_check(-1,2,-1,3)
#
#    import matplotlib.pyplot as plt
#
#    z=np.linspace(0,10,200)
#
#
#    ax=plt.subplot(111)
#    ax.set_xlim(0,10)
#    ax.set_ylim(-1,1)
#    ax.set_ylabel(r'$j_n(z)$', size=30)
#    ax.set_xlabel(r'$z$', size=30)
#
#    ax.plot(z, j_n(0,z))
#    ax.plot(z, j_n(1,z))
#    ax.plot(z, j_n(2,z))
#    ax.plot(z, j_n(3,z))
#    ax.plot(z, j_n(4,z))
#
#    x=jn_zeros(0,3)
#    print 'this is x', x
#    ax.plot(x,np.zeros(x.size),'o')
#    x=jn_zeros(1,3)
#    ax.plot(x,np.zeros(x.size),'o')
#    x=jn_zeros(2,3)
#    ax.plot(x,np.zeros(x.size),'o')
#    x=jn_zeros(3,3)
#    ax.plot(x,np.zeros(x.size),'o')
#    x=jn_zeros(4,3)
#    ax.plot(x,np.zeros(x.size),'o')
#    x=jn_zeros(9,3)
#    ax.plot(x,np.zeros(x.size),'o', label='l=9')
#    ax.plot(z, j_n(10,z))
#
#    plt.legend(loc=2)
#    plt.grid()
#    plt.show()
