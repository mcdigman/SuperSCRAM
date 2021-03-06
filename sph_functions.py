"""
    Code to calculate spherical functions, such as:
    spherical Bessel functions
    the zeros of the Bessel functions
    the real spherical harmonics
    J. E. McEwen (c) 2016
"""
from __future__ import division,print_function,absolute_import
from builtins import range

import numpy as np

from scipy.special import sph_harm as Y_lm, jv

#eps = np.finfo(float).eps
#Z_CUT = 1e-2

#data = np.loadtxt('data/spherical_bessel_zeros_360.dat')
#data = np.loadtxt('data/spherical_bessel_zeros_527.dat')
data = np.loadtxt('data/spherical_bessel_zeros_704.dat')
data[data<0.] = np.inf

l_max_zeros = data.shape[0]-1

def j_n(n,z):
    """spherical bessel function n for array of z values"""
    z = np.asarray(z)
    #j_n is well defined if z is zero, must avoid dividing by zero
    if np.any(z==0.):
        if np.isscalar(z):
            z_use = np.array([z])
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



def jn_zeros_cut(ll,q_lim):
    """return all spherical bessel function zeros for an l value smaller than a cut value q_lim"""
    if ll<=l_max_zeros:
        return data[ll,data[ll]<q_lim]
    else:
        raise IndexError('l is greater than the number of ls in the bessel functions lookup table,choose lower ceiling or expand the lookup table')

def Y_r(l,m,theta,phi):
    """ real spherical harmonics, using scipy spherical harmonic functions
        the scipy spherical harmonic inputs are ordered as
        Y_lm(m,l,phi,theta)
    """
    if np.abs(m) > l:
        print('You must have |m|<=l for spherical harmonics.')
        raise ValueError('Please check the function values you sent into Y_r in module sph_functions.py')
        #sys.exit()

        #if np.real(result) < eps:
        #    result = 0
        #return result
        #print('check', result)
        #if result < eps:
        #    result = 0
        #return result
    if m<0:
        result = 1j/np.sqrt(2)*(Y_lm(m,l,phi,theta) - (-1)**np.abs(m)*Y_lm(-m,l,phi,theta))
        return np.real(result)
    elif m>0:
        result = 1/np.sqrt(2)*(Y_lm(-m,l,phi,theta)  + (-1)**m*Y_lm(m,l,phi,theta))
        return np.real(result)
    else:
        return np.real(Y_lm(m,l,phi,theta))
        #result[np.abs(result)<eps] = 0.
        #return np.sqrt(2)*(-1)**m*np.real(Y_lm(m,l,phi,theta))
        #if np.absolute(np.real(result)) < eps:
        #    result = 0

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
#    y = lambda phi : np.sin(phi)
#    x = lambda phi : np.cos(phi)
#    z = lambda theta : np.cos(theta)
#    theta = 3*np.pi/2; phi=3*np.pi/4
#
#    # check the values for the real spherical harmonics
#    # checking against values on https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#l_.3D_0.5B2.5D.5B3.5D
#    # check against wiki values
#    print('check l=0 case')
#    print('function value', Y_r(0,0,theta,phi))
#    print('wiki value', .5*np.sqrt(1/np.pi))
#    print('------------------------------------------')
#    # l=1 case
#    print('check l=1, m=-1 case')
#    print('function value', Y_r(1,-1,theta,phi))
#    print('wiki value', np.sqrt(3/4./np.pi)*y(phi))
#    print('------------------------------------------')
#    print('check l=1, m=0 case')
#    print('function value', Y_r(1,0,theta,phi))
#    print('wiki value', np.sqrt(3/4./np.pi)*z(theta))
#    print('------------------------------------------')
#    print('check l=1, m=1 case')
#    print('function value', Y_r(1,1,theta,phi))
#    print('wiki value', np.sqrt(3/4./np.pi)*x(phi))
#    print('------------------------------------------')
#    print('check l=2, m=-2 case')
#    print('function value', Y_r(2,-2,theta,phi))
#    print('wiki value', .5*np.sqrt(15/np.pi)*x(phi)*y(phi))
#    print('check l=2, m=-1 case')
#    print('function value', Y_r(2,-1,theta,phi))
#    print('wiki value', .5*np.sqrt(15/np.pi)*z(theta)*y(phi))
#    print('check l=2, m=0 case')
#    print('function value', Y_r(2,0,theta,phi))
#    print('wiki value', .25*np.sqrt(5/np.pi)*(2*z(theta)**2 - y(phi)**2 - x(phi)**2))
#    print('check l=2, m=1 case')
#    print('function value', Y_r(2,1,theta,phi))
#    print('wiki value', .5*np.sqrt(15/np.pi)*z(theta)*x(phi))
#    print('check l=2, m=2 case')
#    print('function value', Y_r(2,2,theta,phi))
#    print('wiki value', .25*np.sqrt(15/np.pi)*(x(phi)**2-y(phi)**2))
#    print('------------------------------------------')
#    #print('check an incorrect m and l value')
#    #print(Y_r(2,3,theta,phi))
#    print('------------------------------------------')
#    print('check normalization')
#    from scipy.integrate import nquad
#    def norm_check(m1,l1,m2,l2):
#        def func(theta,phi):
#            return np.sin(theta)*Y_r(l1,m1,theta,phi)*Y_r(l2,m2,theta,phi)
#
#        def funcII(theta,phi):
#            return np.sin(theta)*Y_lm(m1,l1,phi,theta)*np.conjugate(Y_lm(m2,l2,phi,theta))
#
#
#        I = nquad(func,[[0,np.pi],[0,2*np.pi]])[0]
#        #print('check against spherical harmonics',nquad(funcII,[[0,np.pi],[0,2*np.pi]])[0])
#        if I < eps:
#            I = 0
#        return I
#
#    print('check normalization, 1,1,1,1:', norm_check(1,1,1,1))
#    print('check normalization, 0,1,0,1:', norm_check(0,1,0,1))
#    print('check normalization, 0,2,0,2:', norm_check(0,2,0,2))
#    print('check normalization, 1,2,1,2:', norm_check(1,2,1,2))
#    print('check normalization, 2,2,2,2:', norm_check(2,2,2,2))
#    print('check normalization, -1,2,-1,2:', norm_check(-1,2,-1,2))
#    print('check normalization, -1,2,0,2:', norm_check(-1,2,0,2))
#    print('check normalization, -1,2,-1,3:', norm_check(-1,2,-1,3))
#
#    import matplotlib.pyplot as plt
#
#    z = np.linspace(0,10,200)
#
#
#    ax = plt.subplot(111)
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
#    x = jn_zeros(0,3)
#    print('this is x', x)
#    ax.plot(x,np.zeros(x.size),'o')
#    x = jn_zeros(1,3)
#    ax.plot(x,np.zeros(x.size),'o')
#    x = jn_zeros(2,3)
#    ax.plot(x,np.zeros(x.size),'o')
#    x = jn_zeros(3,3)
#    ax.plot(x,np.zeros(x.size),'o')
#    x = jn_zeros(4,3)
#    ax.plot(x,np.zeros(x.size),'o')
#    x = jn_zeros(9,3)
#    ax.plot(x,np.zeros(x.size),'o', label='l=9')
#    ax.plot(z, j_n(10,z))
#
#    plt.legend(loc=2)
#    plt.grid()
#    plt.show()
