"""some utility functions for real spherical harmonic a_lm computations, used by PolygonGeo"""
import numpy as np
from mpmath import mp

#note should run faster if gmpy2 installed for mpmath
mp.dps = 20
def get_lm_dict(l_max):
    lm_dict = {}
    itr = 0
    n_tot = (l_max+1)**2
    ls = np.zeros(n_tot,dtype=np.int)
    ms = np.zeros(n_tot,dtype=np.int)
    for ll in xrange(0,l_max+1):
        for mm in xrange(-ll,ll+1):
            ms[itr] = mm
            ls[itr] = ll
            lm_dict[(ll,mm)] = itr
            itr+=1
    return lm_dict,ls,ms

#def reconstruct_from_alm(l_max,thetas,phis,alms):
#    n_tot = (l_max+1)**2
#    phis = mp.matrix(phis)
#    thetas = mp.matrix(thetas)
#
#    reconstructed = mp.zeros(thetas.rows,1)
#
#    lm_dict,ls,ms = get_lm_dict(l_max)
#
#    sin_theta = mp.matrix([mp.sin(val) for val in thetas])
#    cos_theta = mp.matrix([mp.cos(val) for val in thetas])
#    abs_sin_theta = mp.matrix([mp.fabs(val) for val in sin_theta])
#
#
#    sin_phi_m = mp.zeros(l_max+1,phis.rows)
#    cos_phi_m = mp.zeros(l_max+1,phis.rows)
#    for mm in xrange(0,l_max+1):
#        sin_phi_m[mm,:] = mp.matrix([mp.sin(mm*val) for val in phis])[:,0].T
#        cos_phi_m[mm,:] = mp.matrix([mp.cos(mm*val) for val in phis])[:,0].T
#
#    factorials = mp.matrix([mp.factorial(val) for val in np.arange(0,2*l_max+1)])
#    known_legendre = {(0,0):(mp.zeros(thetas.rows,1)+1.),(1,0):cos_theta,(1,1):-abs_sin_theta}
#
#    for ll in np.arange(0,l_max+1):
#        if ll>=2:
#            known_legendre[(ll,ll-1)] = mp.matrix([(2.*ll-1.)*cos_theta[i,0]*known_legendre[(ll-1,ll-1)][i,0] for i in xrange(0,thetas.rows)])
#            known_legendre[(ll,ll)] = mp.matrix([-(2.*ll-1.)*abs_sin_theta[i,0]*known_legendre[(ll-1,ll-1)][i,0] for i in xrange(0,thetas.rows)])
#        for mm in np.arange(0,ll+1):
#            if mm<=ll-2:
#                known_legendre[(ll,mm)] = mp.matrix([((2.*ll-1.)/(ll-mm)*cos_theta[i,0]*known_legendre[(ll-1,mm)][i,0]-(ll+mm-1.)/(ll-mm)*known_legendre[(ll-2,mm)][i,0]) for i in xrange(0,thetas.rows)])
#            prefactor = mp.sqrt((2.*ll+1.)/(4.*mp.pi)*factorials[ll-mm]/factorials[ll+mm])
#
#            base = known_legendre[(ll,mm)]
#            if mm==0:
#                for i in xrange(0,thetas.rows):
#                    reconstructed[i,:]+= prefactor*alms[(ll,mm)]*base[i,:]
#            else:
#                #Note: check condon shortley phase convention
#                for i in xrange(0,thetas.rows):
#                    reconstructed[i,:] =reconstructed[i,:]+((-1)**(mm)*mp.sqrt(2.)*alms[(ll,mm)]*prefactor*base[i,:]*cos_phi_m[mm,i])
#                    reconstructed[i,:] =reconstructed[i,:]+((-1)**(mm)*mp.sqrt(2.)*alms[(ll,-mm)]*prefactor*base[i,:]*sin_phi_m[mm,i])
#            if mm<=ll-2:
#                known_legendre.pop((ll-2,mm),None)
#    return np.array(reconstructed.tolist(),dtype=np.double)[:,0]

#TODO could clean up and write iterator logic like in ylm_utils
def reconstruct_from_alm(l_max,thetas,phis,alms):
    thetas = mp.matrix(thetas)

    #reconstructed = mp.zeros(thetas.rows,1)
    reconstructed = np.zeros(thetas.rows)

    sin_theta = mp.matrix([mp.sin(val) for val in thetas])
    cos_theta = mp.matrix([mp.cos(val) for val in thetas])
    abs_sin_theta = mp.matrix([mp.fabs(val) for val in sin_theta])

    #this part doesn't need multiple precision
    sin_phi_m = np.zeros((l_max+1,phis.size))
    cos_phi_m = np.zeros((l_max+1,phis.size))
    for mm in xrange(0,l_max+1):
        sin_phi_m[mm,:] = np.sin(mm*phis)
        cos_phi_m[mm,:] = np.cos(mm*phis)

    #multipliers = np.zeros_like(sin_cos_phi_m)

    factorials = mp.matrix([mp.factorial(val) for val in np.arange(0,2*l_max+1)])
    known_legendre = {(0,0):(mp.zeros(thetas.rows,1)+1.),(1,0):cos_theta,(1,1):-abs_sin_theta}
    for ll in np.arange(0,l_max+1):
        if ll>=2:
            known_legendre[(ll,ll-1)] = mp.matrix([(2.*ll-1.)*cos_theta[i,0]*known_legendre[(ll-1,ll-1)][i,0] for i in xrange(0,thetas.rows)])
            known_legendre[(ll,ll)] = mp.matrix([-(2.*ll-1.)*abs_sin_theta[i,0]*known_legendre[(ll-1,ll-1)][i,0] for i in xrange(0,thetas.rows)])
        for mm in np.arange(0,ll+1):
            if mm<=ll-2:
                known_legendre[(ll,mm)] = mp.matrix([((2.*ll-1.)/(ll-mm)*cos_theta[i,0]*known_legendre[(ll-1,mm)][i,0]-(ll+mm-1.)/(ll-mm)*known_legendre[(ll-2,mm)][i,0]) for i in xrange(0,thetas.rows)])
                known_legendre.pop((ll-2,mm),None)
            prefactor = mp.sqrt((2.*ll+1.)/(4.*mp.pi)*factorials[ll-mm]/factorials[ll+mm])

            base = known_legendre[(ll,mm)]
            if mm==0:
                reconstructed+=alms[(ll,mm)]*np.double(prefactor*base)
            else:
                #Note: check condon shortley phase convention
                multiplier = np.double(prefactor*base)
                reconstructed+=(-1)**mm*np.sqrt(2)*alms[(ll, mm)]*multiplier*cos_phi_m[mm]
                reconstructed+=(-1)**mm*np.sqrt(2)*alms[(ll,-mm)]*multiplier*sin_phi_m[mm]
    return reconstructed

def get_Y_r_dict(l_max,thetas,phis):
    ytable,ls,ms = get_Y_r_table(l_max,thetas,phis)
    ydict = {}
    for itr in xrange(0,ls.size):
        ydict[(ls[itr],ms[itr])] = ytable[itr]
    return ydict

def get_Y_r_dict_central(l_max):
    Y_lms = {(0,0):np.double(1./mp.sqrt(4.*mp.pi))}
    factorials = mp.matrix([mp.factorial(val) for val in np.arange(0,2*l_max+1)])
    for ll in xrange(1,l_max+1):
        for mm in xrange(-ll,0):
            Y_lms[(ll,mm)] = 0.
        for nn in xrange(0,np.int(ll/2.)+1):
            Y_lms[(ll,ll-2*nn-1)] = 0.
            if 2*nn ==ll:
                Y_lms[(ll,ll-2*nn)] = np.double((-1)**(nn)*mp.sqrt((2.*ll+1.)/(4.*mp.pi)*(factorials[2*nn]/factorials[2*ll-2*nn]))*2**-ll*(factorials[(2*ll-2*nn)]/factorials[ll-nn]*1./factorials[nn]))
            else:
                Y_lms[(ll,ll-2*nn)] = np.double((-1)**(nn)*mp.sqrt((2.*ll+1.)/(2.*mp.pi)*(factorials[2*nn]/factorials[2*ll-2*nn]))*2**-ll*(factorials[(2*ll-2*nn)]/factorials[ll-nn]*1./factorials[nn]))
            if not np.isfinite(Y_lms[(ll,ll-2*nn)]):
                raise ValueError('result not finite at l='+str(ll)+' m='+str(ll-2*nn)+' try decreasing l_max')
    return Y_lms


def get_Y_r_table(l_max,thetas,phis):
    n_tot = (l_max+1)**2
    #phis = mp.matrix(phis)
    thetas = mp.matrix(thetas)

    Y_lms = np.zeros((n_tot,thetas.rows))

    lm_dict,ls,ms = get_lm_dict(l_max)

    sin_theta = mp.matrix([mp.sin(val) for val in thetas])
    cos_theta = mp.matrix([mp.cos(val) for val in thetas])
    abs_sin_theta = mp.matrix([mp.fabs(val) for val in sin_theta])


    #sin_phi_m = mp.zeros(l_max+1,phis.rows)
    #cos_phi_m = mp.zeros(l_max+1,phis.rows)
    #for mm in xrange(0,l_max+1):
    #    sin_phi_m[mm,:] = mp.matrix([mp.sin(mm*val) for val in phis])[:,0].T
    #    cos_phi_m[mm,:] = mp.matrix([mp.cos(mm*val) for val in phis])[:,0].T
    #this part doesn't need multiple precision
    sin_phi_m = np.zeros((l_max+1,phis.size))
    cos_phi_m = np.zeros((l_max+1,phis.size))
    for mm in xrange(0,l_max+1):
        sin_phi_m[mm,:] = np.sin(mm*phis)
        cos_phi_m[mm,:] = np.cos(mm*phis)

    factorials = mp.matrix([mp.factorial(val) for val in np.arange(0,2*l_max+1)])
    known_legendre = {(0,0):(mp.zeros(thetas.rows,1)+1.),(1,0):cos_theta,(1,1):-abs_sin_theta}

    for ll in np.arange(0,l_max+1):
        if ll>=2:
            known_legendre[(ll,ll-1)] = mp.matrix([(2.*ll-1.)*cos_theta[i,0]*known_legendre[(ll-1,ll-1)][i,0] for i in xrange(0,thetas.rows)])
            known_legendre[(ll,ll)] = mp.matrix([-(2.*ll-1.)*abs_sin_theta[i,0]*known_legendre[(ll-1,ll-1)][i,0] for i in xrange(0,thetas.rows)])
        for mm in np.arange(0,ll+1):
            if mm<=ll-2:
                known_legendre[(ll,mm)] = mp.matrix([((2.*ll-1.)/(ll-mm)*cos_theta[i,0]*known_legendre[(ll-1,mm)][i,0]-(ll+mm-1.)/(ll-mm)*known_legendre[(ll-2,mm)][i,0]) for i in xrange(0,thetas.rows)])
                known_legendre.pop((ll-2,mm),None)

            prefactor = mp.sqrt((2.*ll+1.)/(4.*mp.pi)*factorials[ll-mm]/factorials[ll+mm])
            base = known_legendre[(ll,mm)]
            #only do arbitrary precision for part where it is needed
            if mm==0:
                #for i in xrange(0,thetas.rows):
                #    Y_lms[lm_dict[(ll,mm)]][i] = np.double(prefactor*base[i])
                Y_lms[lm_dict[(ll,mm)]] = np.double(prefactor*base)
                #Y_lms[lm_dict[(ll, mm)]] = multiplier
            else:
                multiplier = (-1)**(mm)*np.sqrt(2.)*np.double(prefactor*base)
                Y_lms[lm_dict[(ll, mm)]] = multiplier*cos_phi_m[mm]
                Y_lms[lm_dict[(ll,-mm)]] = multiplier*sin_phi_m[mm]
                multiplier=None
            #if mm<=ll-2:

                #Y_lms[lm_dict[(ll, mm)]] = np.array([np.double((multiplier[i]*cos_phi_m[mm,i])) for i in xrange(0,thetas.rows)])
                #Y_lms[lm_dict[(ll,-mm)]] = np.array([np.double((multiplier[i]*sin_phi_m[mm,i])) for i in xrange(0,thetas.rows)])
                #for i in xrange(0,thetas.rows):
                #    Y_lms[lm_dict[(ll,mm)]][i] = np.double(((-1)**(mm)*mp.sqrt(2.)*prefactor*base[i]*cos_phi_m[mm,i]))
                #    Y_lms[lm_dict[(ll,-mm)]][i] = np.double(((-1)**(mm)*mp.sqrt(2.)*prefactor*base[i]*sin_phi_m[mm,i]))
                #Y_lms[lm_dict[(ll, mm)]] = np.array([np.double(((-1)**(mm)*mp.sqrt(2.)*prefactor*base[i]*cos_phi_m[mm,i])) for i in xrange(0,thetas.rows)])
                #Y_lms[lm_dict[(ll,-mm)]] = np.array([np.double(((-1)**(mm)*mp.sqrt(2.)*prefactor*base[i]*sin_phi_m[mm,i])) for i in xrange(0,thetas.rows)])
    return Y_lms,ls,ms
