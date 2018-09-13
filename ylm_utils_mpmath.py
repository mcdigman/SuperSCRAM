"""some utility functions for real spherical harmonic a_lm computations, used by PolygonGeo, using arbitrary precision mpmath"""
from __future__ import division,print_function,absolute_import
from builtins import range
import numpy as np
from mpmath import mp
from ylm_utils import get_lm_dict
#note should run faster if gmpy2 installed for mpmath

#set precision to 20 because really need arbitrary length more than precision
mp.dps = 20

#TODO can be consistent with table vs dict logic
#TODO could clean up and write iterator logic like in ylm_utils
def reconstruct_from_alm(l_max,thetas,phis,alms):
    """reconstruct a survey window from its spherical harmonic decomposition"""
    thetas = mp.matrix(thetas)

    reconstructed = np.zeros(thetas.rows)

    sin_theta = mp.matrix([mp.sin(val) for val in thetas])
    cos_theta = mp.matrix([mp.cos(val) for val in thetas])
    abs_sin_theta = mp.matrix([mp.fabs(val) for val in sin_theta])

    #this part doesn't need multiple precision
    sin_phi_m = np.zeros((l_max+1,phis.size))
    cos_phi_m = np.zeros((l_max+1,phis.size))
    for mm in range(0,l_max+1):
        sin_phi_m[mm,:] = np.sin(mm*phis)
        cos_phi_m[mm,:] = np.cos(mm*phis)

    factorials = mp.matrix([mp.factorial(val) for val in np.arange(0,2*l_max+1)])
    known_legendre = {(0,0):(mp.zeros(thetas.rows,1)+1.),(1,0):cos_theta,(1,1):-abs_sin_theta}
    for ll in np.arange(0,l_max+1):
        if ll>=2:
            kl11 = known_legendre[(ll-1,ll-1)]
            known_legendre[(ll,ll-1)] = mp.matrix([(2.*ll-1.)*cos_theta[i,0]*kl11[i,0] for i in range(0,thetas.rows)])
            known_legendre[(ll,ll)] = mp.matrix([-(2.*ll-1.)*abs_sin_theta[i,0]*kl11[i,0] for i in range(0,thetas.rows)])
        for mm in np.arange(0,ll+1):
            if mm<=ll-2:
                kl1 = known_legendre[(ll-1,mm)]
                kl2 = known_legendre[(ll-2,mm)]
                known_legendre[(ll,mm)] = mp.matrix([((2.*ll-1.)/(ll-mm)*cos_theta[i,0]*kl1[i,0]-(ll+mm-1.)/(ll-mm)*kl2[i,0]) for i in range(0,thetas.rows)])
                known_legendre.pop((ll-2,mm),None)
            prefactor = mp.sqrt((2.*ll+1.)/(4.*mp.pi)*factorials[ll-mm]/factorials[ll+mm])

            base = known_legendre[(ll,mm)]
            if mm==0:
                reconstructed+=alms[(ll,mm)]*np.double(prefactor*base)
            else:
                multiplier = np.double(prefactor*base)
                reconstructed+=(-1)**mm*np.sqrt(2)*alms[(ll, mm)]*multiplier*cos_phi_m[mm]
                reconstructed+=(-1)**mm*np.sqrt(2)*alms[(ll,-mm)]*multiplier*sin_phi_m[mm]
    return reconstructed

def get_Y_r_dict(l_max,thetas,phis):
    """get Y_r for thetas and phis out to l_max as a dictionary"""
    ytable,ls,ms = get_Y_r_table(l_max,thetas,phis)
    ydict = {}
    for itr in range(0,ls.size):
        ydict[(ls[itr],ms[itr])] = ytable[itr]
    return ydict

def get_Y_r_dict_central(l_max):
    """get Y_r up to l_max at the theta=np.pi/2., phi=0., which can be found analytically"""
    Y_lms = {(0,0):np.double(1./mp.sqrt(4.*mp.pi))}
    factorials = mp.matrix([mp.factorial(val) for val in np.arange(0,2*l_max+1)])
    for ll in range(1,l_max+1):
        for mm in range(-ll,0):
            Y_lms[(ll,mm)] = 0.
        for nn in range(0,np.int(ll/2.)+1):
            Y_lms[(ll,ll-2*nn-1)] = 0.
            base = factorials[(2*ll-2*nn)]/factorials[ll-nn]*1./factorials[nn]
            ratio = factorials[2*nn]/factorials[2*ll-2*nn]
            #prefactor = (-1)**(nn)*mp.sqrt((2.*ll+1.)/(2.*mp.pi)
            if 2*nn==ll:
                Y_lms[(ll,ll-2*nn)] = np.double((-1)**(nn)*mp.sqrt((2.*ll+1.)/(4.*mp.pi)*ratio)*2**-ll*base)
            else:
                Y_lms[(ll,ll-2*nn)] = np.double((-1)**(nn)*mp.sqrt((2.*ll+1.)/(2.*mp.pi)*ratio)*2**-ll*base)
            if not np.isfinite(Y_lms[(ll,ll-2*nn)]):
                raise ValueError('result not finite at l='+str(ll)+' m='+str(ll-2*nn)+' try decreasing l_max')
    return Y_lms


def get_Y_r_table(l_max,thetas,phis):
    """get Y_r as a table up to l_max for thetas and phis given"""
    n_tot = (l_max+1)**2
    #thetas = mp.matrix(thetas)

    Y_lms = np.zeros((n_tot,thetas.size))

    lm_dict,ls,ms = get_lm_dict(l_max)

    #sin_theta = mp.matrix([mp.sin(val) for val in thetas])
    #cos_theta = mp.matrix([mp.cos(val) for val in thetas])
    #abs_sin_theta = mp.matrix([mp.fabs(val) for val in sin_theta])
    sin_theta = np.sin(thetas)#mp.matrix([mp.sin(val) for val in thetas])
    cos_theta = np.cos(thetas)#mp.matrix([mp.cos(val) for val in thetas])
    abs_sin_theta = np.abs(sin_theta)#mp.matrix([mp.fabs(val) for val in sin_theta])

    sin_phi_m = np.zeros((l_max+1,phis.size))
    cos_phi_m = np.zeros((l_max+1,phis.size))
    for mm in range(0,l_max+1):
        sin_phi_m[mm,:] = np.sin(mm*phis)
        cos_phi_m[mm,:] = np.cos(mm*phis)

    factorials = np.array([mp.factorial(val) for val in np.arange(0,2*l_max+1)])
    known_legendre = {(0,0):np.full(thetas.size,mp.mpf(1.)),(1,0):np.asarray(mp.matrix(cos_theta)),(1,1):np.asarray(mp.matrix(-abs_sin_theta))}

    for ll in np.arange(0,l_max+1):
        if ll>=2:
            kl11 = known_legendre[(ll-1,ll-1)]
            known_legendre[(ll,ll-1)] = (2.*ll-1.)*cos_theta*kl11
            known_legendre[(ll,ll)] = -(2.*ll-1.)*abs_sin_theta*kl11
            kl11=None
        for mm in np.arange(0,ll+1):
            if mm<=ll-2:
                kl1 = known_legendre[(ll-1,mm)]
                kl2 = known_legendre[(ll-2,mm)]
                known_legendre[(ll,mm)] = (2.*ll-1.)/(ll-mm)*cos_theta*kl1-(ll+mm-1.)/(ll-mm)*kl2
                known_legendre.pop((ll-2,mm),None)
                kl1=None
                kl2=None

            prefactor = mp.sqrt((2.*ll+1.)/(4.*mp.pi)*factorials[ll-mm]/factorials[ll+mm])
            base = known_legendre[(ll,mm)]
            #only do arbitrary precision for part where it is needed
            if mm==0:
                Y_lms[lm_dict[(ll,mm)]] = np.double(prefactor*base)
            else:
                multiplier = (-1)**(mm)*np.sqrt(2.)*np.double(prefactor*base)
                Y_lms[lm_dict[(ll, mm)]] = multiplier*cos_phi_m[mm]
                Y_lms[lm_dict[(ll,-mm)]] = multiplier*sin_phi_m[mm]
                multiplier = None
            prefactor=None
            base=None
    return Y_lms,ls,ms
