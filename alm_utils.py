"""some utility functions for real spherical harmonic a_lm computations, used by PolygonGeo"""
from __future__ import division,print_function,absolute_import
from builtins import range
import numpy as np
import scipy.linalg as spl

def rot_alm_z(d_alm_table_in,angles,ls):
    """rotate alms around z axis by angle gamma_alpha"""
    print("alm_utils: rot z")
    d_alm_table_out = np.zeros_like(d_alm_table_in)
    n_v = angles.size
    for l_itr in range(0,ls.size):
        ll = ls[l_itr]
        d_alm_table_out[l_itr] = np.zeros((2*ll+1,n_v))
        ms = np.arange(1,ll+1)
        m_angles = np.outer(ms,angles)
        d_alm_table_out[l_itr][ll+ms] = np.cos(m_angles)*d_alm_table_in[l_itr][ll+ms]+np.sin(m_angles)*d_alm_table_in[l_itr][ll-ms]
        d_alm_table_out[l_itr][ll-ms] = -np.sin(m_angles)*d_alm_table_in[l_itr][ll+ms]+np.cos(m_angles)*d_alm_table_in[l_itr][ll-ms]
        d_alm_table_out[l_itr][ll] = d_alm_table_in[l_itr][ll]
    return d_alm_table_out

def rot_alm_x(d_alm_table_in,angles,ls,n_double=30,debug=True):
    """rotate alms around x axis by angle theta_alpha"""
    print("alm_utils: rot x")
    d_alm_table_out = np.zeros_like(d_alm_table_in)
    n_v = angles.size
    for l_itr in range(0,ls.size):
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
        m_mat[ms+ll,ms+ll] = 1./np.sqrt(2.)
        m_mat[ms+ll,-ms+ll] = -1j/np.sqrt(2.)
        m_mat[-ms+ll,ms+ll] = (-1)**ms/np.sqrt(2.)
        m_mat[-ms+ll,-ms+ll] = 1j*(-1)**ms/np.sqrt(2.)

        #m_mat_i = np.conjugate(m_mat.T)
        #infinitesimal form of El(epsilon) (must be multiplied by epsilon)
        el_mat_real = np.real(np.dot(np.dot(np.conjugate(m_mat.T),el_mat_complex),m_mat))
        #print(ll)
        if debug:
            #E_l matrices should be antisymmetric
            assert np.all(el_mat_complex==-el_mat_complex.conjugate().T)
            assert np.all(el_mat_real==-el_mat_real.T)
            #check m_mat is actually unitary
            assert np.allclose(np.identity(m_mat.shape[0]),np.dot(np.conjugate(m_mat.T),m_mat))
        #print(ll)
        el_mat = np.empty_like(el_mat_real,order='F')
        el_mat2 = np.empty_like(el_mat_real,order='F')
        for itr in range(0,n_v):
            epsilon = angles[itr]/2.**n_double
            #el_mat = epsilon*el_mat_real.copy()
            el_mat[:] = epsilon*el_mat_real
            el_mat2[:] = el_mat
            #use angle doubling fomula to get to correct angle
            for _itr2 in range(0,n_double):
                #el_mat = 2.*el_mat+np.dot(el_mat,el_mat)
                #2 arrays necessary to prevent overwrite from clobbering data, faster to maintain 2 specific arrays with no copies
                el_mat2 = spl.blas.dgemm(1.,el_mat,el_mat,2.,el_mat2,overwrite_c=True)
                el_mat[:] = el_mat2
                #assert np.allclose(el_mat,el_mat2,atol=1.e-15)
            ######Walled
#            for itr3 in range(1,3):
#                epsilon2 = angles[itr]/2.**(n_double+itr3)
#                el_mat2 = epsilon2*el_mat_real.copy()
#                #use angle doubling fomula to get to correct angle
#                for itr2 in range(0,n_double+itr3):
#                    el_mat2 = 2.*el_mat2+np.dot(el_mat2,el_mat2)
#                if itr3>1:
#                    #print(itr3,np.average(np.abs(el_mat-el_mat2))-last)
#                    print(itr3,ll,np.average(np.abs(el_mat2-el_last))-last,np.max(np.abs(el_mat2-el_last)),(np.average(np.abs(el_mat2-el_last))-last)/last)
#                    last = np.average(np.abs(el_mat2-el_last))
#                    if not last==0:
#                        raise RuntimeError('did not converge')
#                    #if np.average(np.abs(el_mat2-el_last))-last<0.:
#                    #    print("stop itr,ll",itr3,ll)
#                      # break
#                    #if itr3==9:
#                    #    print("fail itr,ll",itr3,ll)
#                else:
#                    last = np.average(np.abs(el_mat2-el_mat))
#                #last=np.average(np.abs(el_mat2-el_last))
#
#                el_last = el_mat2
            ######
            d_mat = el_mat+np.identity(el_mat.shape[0])
            d_alm_table_out[l_itr][:,itr] = np.dot(d_mat,d_alm_table_in[l_itr][:,itr])
    return d_alm_table_out
