"""some utility functions for real spherical harmonic a_lm/Y_lm computations, used by PolygonGeo"""
import numpy as np
import scipy as sp


#may be some loss of precision; fails to identify possible exact 0s
def reconstruct_from_alm(l_max,thetas,phis,alms):
    if 2*l_max>170:
        raise ValueError('Scipy factorial will fail for n>170 because 171!>2^1024, need to use arbitrary precision or implement asymptotic form')
    n_tot = (l_max+1)**2

    ls = np.zeros(n_tot,dtype=np.int)
    ms = np.zeros(n_tot,dtype=np.int)
    reconstructed = np.zeros(thetas.size)

    lm_dict = {}
    itr = 0
    for ll in xrange(0,l_max+1):
        for mm in xrange(-ll,ll+1):
            ms[itr] = mm
            ls[itr] = ll
            lm_dict[(ll,mm)] = itr
            itr+=1

    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    abs_sin_theta = np.abs(sin_theta)


    sin_phi_m = np.zeros((l_max+1,thetas.size))
    cos_phi_m = np.zeros((l_max+1,thetas.size))
    for mm in xrange(0,l_max+1):
        sin_phi_m[mm] = np.sin(mm*phis)
        cos_phi_m[mm] = np.cos(mm*phis)

    factorials = sp.misc.factorial(np.arange(0,2*l_max+1))

    known_legendre = {(0,0):(np.zeros(thetas.size)+1.),(1,0):cos_theta,(1,1):-abs_sin_theta}

    for ll in xrange(0,l_max+1):
        if ll>=2:
            known_legendre[(ll,ll-1)] = (2.*ll-1.)*cos_theta*known_legendre[(ll-1,ll-1)]
            known_legendre[(ll,ll)] = -(2.*ll-1.)*abs_sin_theta*known_legendre[(ll-1,ll-1)]
        for mm in xrange(0,ll+1):
            if mm<=ll-2:
                known_legendre[(ll,mm)] = ((2.*ll-1.)/(ll-mm)*cos_theta*known_legendre[(ll-1,mm)]-(ll+mm-1.)/(ll-mm)*known_legendre[(ll-2,mm)])

            prefactor = np.sqrt((2.*ll+1.)/(4.*np.pi)*factorials[ll-mm]/factorials[ll+mm])
            base = known_legendre[(ll,mm)]
            if not np.all(np.isfinite(prefactor)):
                raise ValueError('Value evaluates to nan, l_max='+str(l_max)+' is likely too large')
            if not np.all(np.isfinite(base)):
                raise ValueError('Value evaluates to nan, l_max='+str(l_max)+' is likely too large')
            if mm==0:
                reconstructed += prefactor*alms[(ll,mm)]*base
            else:
                #Note: check condon shortley phase convention
                reconstructed+= (-1)**(mm)*np.sqrt(2.)*alms[(ll,mm)]*prefactor*base*cos_phi_m[mm]
                reconstructed+= (-1)**(mm)*np.sqrt(2.)*alms[(ll,-mm)]*prefactor*base*sin_phi_m[mm]
        if mm<=ll-2:
            known_legendre.pop((ll-2,mm),None)

    return reconstructed

#TODO split this stuff into another file
#alternate way of computing Y_r from the way in sph_functions
def Y_r_2(ll,mm,theta,phi,known_legendre):
    if (ll+np.abs(mm))>170:
        raise ValueError('Scipy factorial will fail for n>170 because 171!>2^1024, need to use arbitrary precision or implement asymptotic form')
    prefactor = np.sqrt((2.*ll+1.)/(4.*np.pi)*sp.misc.factorial(ll-np.abs(mm))/sp.misc.factorial(ll+np.abs(mm)))
    base = (prefactor*(-1)**mm)*known_legendre[(ll,np.abs(mm))]
    if mm==0:
        return base
    elif mm>0:
        return base*np.sqrt(2.)*np.cos(mm*phi)
    else:
        return base*np.sqrt(2.)*np.sin(np.abs(mm)*phi)

def get_Y_r_dict(l_max,thetas,phis):
    ytable,ls,ms = get_Y_r_table(l_max,thetas,phis)
    ydict = {}
    for itr in xrange(0,ls.size):
        ydict[(ls[itr],ms[itr])] = ytable[itr]
    return ydict

#use analytic formula for Y_r(l,m,pi/2,0)
def get_Y_r_dict_central(l_max):
    if 2*l_max>170:
        raise ValueError('Scipy factorial will fail for n!>170 because 171!>2^1024, need to use arbitrary precision or implement asymptotic form')
#    n_tot = (l_max+1)**2
#    ls = np.zeros(n_tot)
#    ms = np.zeros(n_tot)
    Y_lms = {(0,0):1./np.sqrt(4.*np.pi)}
    factorials = sp.misc.factorial(np.arange(0,2*l_max+1))
    for ll in xrange(1,l_max+1):
        for mm in xrange(-l_max,0):
            Y_lms[(ll,mm)] = 0.
        for nn in xrange(0,np.int(ll/2.)+1):
            Y_lms[(ll,ll-2*nn-1)] = 0.
            if 2*nn==ll:
                Y_lms[(ll,ll-2*nn)] = (-1)**(nn+ll)*np.sqrt((2.*ll+1.)/(4.*np.pi)*(factorials[2*nn]/factorials[2*ll-2*nn]))*2**-ll*(factorials[(2*ll-2*nn)]/factorials[ll-nn]*1./factorials[nn])
            else:
                Y_lms[(ll,ll-2*nn)] = (-1)**(nn+ll)*np.sqrt((2.*ll+1.)/(2.*np.pi)*(factorials[2*nn]/factorials[2*ll-2*nn]))*2**-ll*(factorials[(2*ll-2*nn)]/factorials[ll-nn]*1./factorials[nn])

            if not np.isfinite(Y_lms[(ll,ll-2*nn)]):
                raise ValueError('result not finite at l='+str(ll)+' m='+str(ll-2*nn)+' try decreasing l_max')

    return Y_lms

def get_Y_r_table(l_max,thetas,phis):
    if 2*l_max>170:
        raise ValueError('Scipy factorial will fail for n!>170 because 171!>2^1024, need to use arbitrary precision or implement asymptotic form')
    n_tot = (l_max+1)**2

    ls = np.zeros(n_tot)
    ms = np.zeros(n_tot)
    Y_lms = np.zeros((n_tot,thetas.size))

    lm_dict = {}
    itr = 0
    for ll in xrange(0,l_max+1):
        for mm in xrange(-ll,ll+1):
            ms[itr] = mm
            ls[itr] = ll
            lm_dict[(ll,mm)] = itr
            itr+=1

    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    abs_sin_theta = np.abs(sin_theta)


    sin_phi_m = np.zeros((l_max+1,thetas.size))
    cos_phi_m = np.zeros((l_max+1,thetas.size))
    for mm in xrange(0,l_max+1):
        sin_phi_m[mm] = np.sin(mm*phis)
        cos_phi_m[mm] = np.cos(mm*phis)

        factorials = sp.misc.factorial(np.arange(0,2*l_max+1))

    known_legendre = {(0,0):(np.zeros(thetas.size)+1.),(1,0):cos_theta,(1,1):-abs_sin_theta}

    for ll in xrange(0,l_max+1):
        if ll>=2:
            known_legendre[(ll,ll-1)] = (2.*ll-1.)*cos_theta*known_legendre[(ll-1,ll-1)]
            known_legendre[(ll,ll)] = -(2.*ll-1.)*abs_sin_theta*known_legendre[(ll-1,ll-1)]
            #print ll,abs_sin_theta,known_legendre[(ll,ll)],known_legendre[(ll-1,ll-1)],known_legendre[(ll-1,ll-2)]
        for mm in xrange(0,ll+1):
            if mm<=ll-2:
                known_legendre[(ll,mm)] = ((2.*ll-1.)/(ll-mm)*cos_theta*known_legendre[(ll-1,mm)]-(ll+mm-1.)/(ll-mm)*known_legendre[(ll-2,mm)])
            prefactor = np.sqrt((2.*ll+1.)/(4.*np.pi)*factorials[ll-mm]/factorials[ll+mm])
            #if mm==ll:
            #    print prefactor
            base = known_legendre[(ll,mm)]

            if not np.all(np.isfinite(prefactor)):
                raise ValueError('Value not finite, l_max='+str(l_max)+' is likely too large for numerical precision')
            if not np.all(np.isfinite(base)):
                raise ValueError('Value not finite, l_max='+str(l_max)+' is likely too large for numerical precision')

            if mm==0:
                Y_lms[lm_dict[(ll,mm)]] = prefactor*base
            else:
                #Note: check condon shortley phase convention

                Y_lms[lm_dict[(ll,mm)]] = (-1)**(mm)*np.sqrt(2.)*prefactor*base*cos_phi_m[mm]
                Y_lms[lm_dict[(ll,-mm)]] = (-1)**(mm)*np.sqrt(2.)*prefactor*base*sin_phi_m[mm]
            if mm<=ll-2:
                known_legendre.pop((ll-2,mm),None)


    return Y_lms,ls,ms
