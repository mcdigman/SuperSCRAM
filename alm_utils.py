"""some utility functions for real spherical harmonic a_lm computations, used by PolygonGeo"""
import numpy as np
import scipy as sp

import defaults


def rot_alm_z(d_alm_table_in,angles,ls):
    """rotate alms around z axis by angle gamma_alpha"""
    print "alm_utils: rot z"
    d_alm_table_out = np.zeros_like(d_alm_table_in)
    n_v = angles.size
    for l_itr in xrange(0,ls.size):
        ll = ls[l_itr]
        d_alm_table_out[l_itr] = np.zeros((2*ll+1,n_v))
        ms = np.arange(1,ll+1)
        m_angles = np.outer(ms,angles)
        d_alm_table_out[l_itr][ll+ms] = np.cos(m_angles)*d_alm_table_in[l_itr][ll+ms]+np.sin(m_angles)*d_alm_table_in[l_itr][ll-ms]
        d_alm_table_out[l_itr][ll-ms] = -np.sin(m_angles)*d_alm_table_in[l_itr][ll+ms]+np.cos(m_angles)*d_alm_table_in[l_itr][ll-ms]
        d_alm_table_out[l_itr][ll] = d_alm_table_in[l_itr][ll]
    return d_alm_table_out

def rot_alm_x(d_alm_table_in,angles,ls,n_double=defaults.polygon_params['n_double'],debug=True):
    """rotate alms around x axis by angle theta_alpha"""
    print "alm_utils: rot x"
    d_alm_table_out=np.zeros_like(d_alm_table_in)
    n_v = angles.size
    for l_itr in xrange(0,ls.size):
        ll = ls[l_itr]
        d_alm_table_out[l_itr] = np.zeros((2*ll+1,n_v))
        ms = np.arange(-ll,ll)

        el_mat_complex = np.zeros((2*ll+1,2*ll+1),dtype=complex)
        #add -1j/2*lplus
        el_mat_complex[ms+ll,ms+ll+1] = -1j/2.*np.sqrt(ll*(ll+1.)-ms*(ms+1.))
        #add -1j/2*lminus
        el_mat_complex[ms+ll+1,ms+ll] = -1j/2.*np.sqrt(ll*(ll+1.)-(ms+1.)*ms)


        m_mat = np.zeros_like(el_mat_complex)
        m_mat[ll,ll] = 1.
        ms = np.arange(1,ll+1)
        m_mat[ms+ll,ms+ll]=1./np.sqrt(2.)
        m_mat[ms+ll,-ms+ll] = -1j/np.sqrt(2.)
        m_mat[-ms+ll,ms+ll]=(-1)**ms/np.sqrt(2.)
        m_mat[-ms+ll,-ms+ll] = 1j*(-1)**ms/np.sqrt(2.)

        #m_mat_i = np.conjugate(m_mat.T)
        #infinitesimal form of El(epsilon) (must be multiplied by epsilon)
        el_mat_real = np.real(np.dot(np.dot(np.conjugate(m_mat.T),el_mat_complex),m_mat))
        #print ll
        if debug:
            #E_l matrices should be antisymmetric
            assert(np.all(el_mat_complex==-el_mat_complex.conjugate().T))
            assert(np.all(el_mat_real==-el_mat_real.T))
            #check m_mat is actually unitary
            assert(np.allclose(np.identity(m_mat.shape[0]),np.dot(np.conjugate(m_mat.T),m_mat)))
            #TODO add assertion  for correct sparseness structure
        #print ll

        for itr in xrange(0,n_v):
            epsilon = angles[itr]/2.**n_double
            el_mat = epsilon*el_mat_real.copy()
            #use angle doubling fomula to get to correct angle
            for itr2 in xrange(0,n_double):
                el_mat = 2.*el_mat+np.dot(el_mat,el_mat)
            ######Walled
            for itr3 in xrange(1,3):
                epsilon2 = angles[itr]/2.**(n_double+itr3)
                el_mat2 = epsilon2*el_mat_real.copy()
                #use angle doubling fomula to get to correct angle
                for itr2 in xrange(0,n_double+itr3):
                    el_mat2 = 2.*el_mat2+np.dot(el_mat2,el_mat2)
                if itr3>1:
                    #print itr3,np.average(np.abs(el_mat-el_mat2))-last
                    print itr3,ll,np.average(np.abs(el_mat2-el_last))-last,np.max(np.abs(el_mat2-el_last)),(np.average(np.abs(el_mat2-el_last))-last)/last
                    last = np.average(np.abs(el_mat2-el_last))
                    if not last==0:
                        raise RuntimeError('did not converge')
                    #if np.average(np.abs(el_mat2-el_last))-last<0.:
                    #    print "stop itr,ll",itr3,ll
                      # break
                    #if itr3==9:
                    #    print "fail itr,ll",itr3,ll
                else:
                    last = np.average(np.abs(el_mat2-el_mat))
                #last=np.average(np.abs(el_mat2-el_last))

                el_last = el_mat2
            ######
            d_mat = el_mat+np.identity(el_mat.shape[0])
            d_alm_table_out[l_itr][:,itr] = np.dot(d_mat,d_alm_table_in[l_itr][:,itr])
    return d_alm_table_out

#may be some loss of precision; fails to identify possible exact 0s
def reconstruct_from_alm(l_max,thetas,phis,alms):
    if 2*l_max>170:
        raise ValueError('Scipy factorial will fail for n!>170 because 171!>2^1024, need to use arbitrary precision or implement asymptotic form')
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
