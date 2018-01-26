"""some utility functions for real spherical harmonic a_lm/Y_lm computations, used by PolygonGeo"""
import numpy as np
import scipy.misc as spm

def get_lm_dict(l_max):
    """get a dictionary which maps an (l,m) key to the position in a list of ls and ms"""
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


def _legendre_iterate(l_max,thetas,phis,function,args):
    """ iterate over l,m getting legendre polynomials
        and applying a function with arguments args at each value"""
    if 2*l_max>170:
        raise ValueError('Scipy factorial will fail for n>170 because 171!>2^1024, need to use arbitrary precision or implement asymptotic form')

    n_t = thetas.size

    #lm_dict = get_lm_dict(l_max)

    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    abs_sin_theta = np.abs(sin_theta)


    sin_phi_m = np.zeros((l_max+1,n_t))
    cos_phi_m = np.zeros((l_max+1,n_t))
    for mm in xrange(0,l_max+1):
        sin_phi_m[mm] = np.sin(mm*phis)
        cos_phi_m[mm] = np.cos(mm*phis)

    factorials = spm.factorial(np.arange(0,2*l_max+1))

    known_legendre = {(0,0):(np.zeros(n_t)+1.),(1,0):cos_theta,(1,1):-abs_sin_theta}
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

            function(ll,mm,base,prefactor,cos_phi_m,sin_phi_m,args)

        if mm<=ll-2:
            known_legendre.pop((ll-2,mm),None)

    return args

#may be some loss of precision, fails to identify exact 0s
#TODO test these actually work
def reconstruct_from_alm(l_max,thetas,phis,alms):
    """reconstruct a survey mask from the calculated decomposition"""
    args = {'reconstructed':np.zeros(thetas.size),'alms':alms}
    args = _legendre_iterate(l_max,thetas,phis,_reconstruct_function,args)
    return args['reconstructed']

def _reconstruct_function(ll,mm,base,prefactor,cos_phi_m,sin_phi_m,args):
    """reconstruct_from_alm helper function"""
    if mm==0:
        args['reconstructed'] += prefactor*args['alms'][(ll,mm)]*base
    else:
        args['reconstructed']+= (-1)**(mm)*np.sqrt(2.)*args['alms'][(ll,mm)]*prefactor*base*cos_phi_m[mm]
        args['reconstructed']+= (-1)**(mm)*np.sqrt(2.)*args['alms'][(ll,-mm)]*prefactor*base*sin_phi_m[mm]

def get_Y_r_table(l_max,thetas,phis):
    """get Y_r as a table out to l_max"""
    lm_dict,ls,ms = get_lm_dict(l_max)
    args = {'Y_lms':np.zeros(((l_max+1)**2,thetas.size)),'lm_dict':lm_dict}
    args = _legendre_iterate(l_max,thetas,phis,_Y_r_function,args)
    return args['Y_lms'],ls,ms

def _Y_r_function(ll,mm,base,prefactor,cos_phi_m,sin_phi_m,args):
    """get_Y_r_table helper function"""
    if mm==0:
        args['Y_lms'][args['lm_dict'][(ll,mm)]] = prefactor*base
    else:
        args['Y_lms'][args['lm_dict'][(ll,mm)]] = (-1)**(mm)*np.sqrt(2.)*prefactor*base*cos_phi_m[mm]
        args['Y_lms'][args['lm_dict'][(ll,-mm)]] = (-1)**(mm)*np.sqrt(2.)*prefactor*base*sin_phi_m[mm]

def get_a_lm_table(l_max,thetas,phis,pixel_area):
    """ get the spherical harmonic coefficients out to l_max
        for a geometry containing the area pixel_area around each pixel centroid present
        with centroids defined by thetas and phis"""
    lm_dict,ls,ms = get_lm_dict(l_max)
    args = {'a_lms':{},'pixel_area':pixel_area}
    args = _legendre_iterate(l_max,thetas,phis,_alm_function,args)
    return args['a_lms'],ls,ms,lm_dict

def _alm_function(ll,mm,base,prefactor,cos_phi_m,sin_phi_m,args):
    """ get_a_lm_table helper function"""
    if mm==0:
        args['a_lms'][(ll,mm)] = args['pixel_area']*prefactor*np.sum(base)
    else:
        args['a_lms'][(ll,mm)] = (-1)**(mm)*np.sqrt(2.)*args['pixel_area']*prefactor*np.sum(base*cos_phi_m[mm])
        args['a_lms'][(ll,-mm)] = (-1)**(mm)*np.sqrt(2.)*args['pixel_area']*prefactor*np.sum(base*sin_phi_m[mm])

#TODO be consistent with dict vs table
def get_Y_r_dict(l_max,thetas,phis):
    """get a set of Y_r as a table with keys (l,m) at the points thetas, phis"""
    ytable,ls,ms = get_Y_r_table(l_max,thetas,phis)
    ydict = {}
    for itr in xrange(0,ls.size):
        ydict[(ls[itr],ms[itr])] = ytable[itr]
    return ydict

def get_Y_r_dict_central(l_max):
    """use analytic formula for Y_r(l,m,pi/2,0)"""
    assert isinstance(l_max,int)
    if 2*l_max>170:
        raise ValueError('Scipy factorial will fail for n!>170 because 171!>2^1024, need to use arbitrary precision or implement asymptotic form')
    Y_lms = {(0,0):1./np.sqrt(4.*np.pi)}
    factorials = spm.factorial(np.arange(0,2*l_max+1))
    for ll in xrange(1,l_max+1):
        for mm in xrange(-ll,0):
            Y_lms[(ll,mm)] = 0.
        for nn in xrange(0,np.int(ll/2.)+1):
            Y_lms[(ll,ll-2*nn-1)] = 0.
            if 2*nn==ll:
                Y_lms[(ll,ll-2*nn)] = (-1)**(nn)*np.sqrt((2.*ll+1.)/(4.*np.pi)*(factorials[2*nn]/factorials[2*ll-2*nn]))*2**-ll*(factorials[(2*ll-2*nn)]/factorials[ll-nn]*1./factorials[nn])
            else:
                Y_lms[(ll,ll-2*nn)] = (-1)**(nn)*np.sqrt((2.*ll+1.)/(2.*np.pi)*(factorials[2*nn]/factorials[2*ll-2*nn]))*2**-ll*(factorials[(2*ll-2*nn)]/factorials[ll-nn]*1./factorials[nn])

            if not np.isfinite(Y_lms[(ll,ll-2*nn)]):
                raise ValueError('result not finite at l='+str(ll)+' m='+str(ll-2*nn)+' try decreasing l_max')

    return Y_lms
