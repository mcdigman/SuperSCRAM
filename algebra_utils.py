"""provides implementations/wrappers of some fast algebra utilities"""
from __future__ import absolute_import
from builtins import range
from warnings import warn
import numpy as np
import scipy.linalg as spl

DEBUG = False

#NOTE inplace not tested for get_inv_cholesky, get_cholesky_inv,invert_triangular
def get_inv_cholesky(A,lower=True,inplace=False,clean=True):
    """Get the inverse cholesky decomposition of a matrix A
        inplace: allow overwriting A, does not guarantee overwriting A
        clean: whether to set unneeded half of result to 0
        completely ignores opposite triangle of A"""
    if DEBUG:
        A_copy = A.copy()
    result = cholesky_inplace(A,inplace=inplace,lower=lower,clean=False)
    result = invert_triangular(result,lower=lower,inplace=inplace,clean=clean)

    if DEBUG:
        assert check_is_cholesky_inv(result,A_copy,is_clean=clean,lower=lower,B_symmetrized=False)
        if not inplace:
            assert np.all(A_copy==A)
    return result

def get_cholesky_inv(A,lower=True,inplace=False,clean=True):
    """Get the cholesky decomposition of the inverse of a matrix A"""
    if DEBUG:
        A_copy = A.copy()
    result = cholesky_inplace(np.asfortranarray(np.rot90(A,2)),lower=(not lower),inplace=inplace,clean=clean)
    result,info = spl.lapack.dtrtri(result,lower=(not lower),overwrite_c=inplace)
    if info!=0:
        raise RuntimeError('dtrtri failed with exit code '+str(info))
    result = np.asfortranarray(np.rot90(result,2))
    if DEBUG:
        if clean:
            assert check_is_triangular(result,lower)
        if not inplace:
            assert np.all(A_copy==A)
    return result

def invert_triangular(A,lower=True,inplace=False,clean=True):
    """invert a triangular matrix,
        completely ignores opposite triangle"""
    if DEBUG:
        A_copy = A.copy()
    result,info = spl.lapack.dtrtri(A,lower=lower,overwrite_c=inplace)
    if info!=0:
        raise RuntimeError('dtrtri failed with exit code '+str(info))
    if clean:
        result = clean_triangle(result,lower=lower,inplace=True)

    if DEBUG:
        if not inplace:
            assert np.all(A_copy==A)
        if clean:
            assert check_is_triangular(result,lower)
    #return spl.solve_triangular(A,np.identity(A.shape[0]),lower=lower,overwrite_b=True)
    return result

def get_mat_from_inv_cholesky(A,lower=True,inplace=False,clean=True):
    """get a matrix from its inverse cholesky decomposition
        completely ignores opposite triangle"""
    if DEBUG:
        A_copy = A.copy()
    chol_mat = invert_triangular(A,lower,inplace=inplace,clean=False)
    result,info = spl.lapack.dlauum(np.rot90(chol_mat,2),lower=not lower,overwrite_c=inplace)
    if info!=0:
        raise RuntimeError('dlauum failed with error code '+str(info))
    result = np.rot90(result,2)
    if clean:
        result = mirror_symmetrize(result,lower,inplace=True)

    if DEBUG:
        if not inplace:
            assert np.all(A_copy==A)
        if clean:
            assert np.all(result.T==result)
    return result

#cholesky_given = True if A already is the cholesky decomposition of the covariance
#NOTE add test cases for ignoring opposite triangle
def ch_inv(A,cholesky_given=False,lower=True,inplace=False,clean=True):
    """ compute inverse of positive definite matrix using cholesky decomposition
        clean: whether to symmetrize output
        completely ignores opposite triangle"""

    if DEBUG:
        A_copy = A.copy()

    if cholesky_given:
        chol_inv = A
    else:
        chol_inv = get_inv_cholesky(A,lower=lower,inplace=inplace,clean=False)

    result,info = spl.lapack.dlauum(chol_inv,lower=lower,overwrite_c=inplace)
    if info!=0:
        raise RuntimeError('dlauum failed with error code '+str(info))

    if clean:
        result = mirror_symmetrize(result,lower=lower,inplace=True)

    if DEBUG:
        if clean:
            assert np.all(result==result.T)
        if not inplace:
            assert np.all(A==A_copy)
    return result


#NOTE could add option to permit inplace if helpful
def cholesky_inv_contract(A,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=True):
    """compute vec.(A)^-1.vec2 using inverse cholesky decomposition,
        opposite triangle of A completely ignored"""
    if cholesky_given:
        chol_inv = A
    else:
        chol_inv = get_inv_cholesky(A,lower,inplace=False,clean=False)

    #potentially Save some time if inputs are identical
    if identical_inputs:
        if lower:
            right_side = spl.blas.dtrmm(1.,chol_inv,vec1,lower=True)
        else:
            right_side = spl.blas.dtrmm(1.,chol_inv,vec1,lower=False,trans_a=True)
        result = spl.blas.dsyrk(1.,right_side,lower=False,trans=True,overwrite_c=True)
        result = mirror_symmetrize(result,lower=False,inplace=True)
    else:
        if lower:
            result = np.dot(spl.blas.dtrmm(1.,chol_inv,vec1,lower=True).T,spl.blas.dtrmm(1.,chol_inv,vec2,lower=True))
        else:
            result = np.dot(spl.blas.dtrmm(1.,chol_inv,vec1,lower=False,trans_a=True).T,spl.blas.dtrmm(1.,chol_inv,vec2,lower=False,trans_a=True))

    return result

def cholesky_contract(A,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=True):
    """compute vec1.A.vec2 using cholesky decomposition
        opposite triangle of A completely ignored"""
    if cholesky_given:
        chol = A
    else:
        chol = cholesky_inplace(A,lower=lower,inplace=False,clean=False)

    #potentially Save some time if inputs are identical
    if identical_inputs:
        if lower:
            right_side = spl.blas.dtrmm(1.,chol,vec1,lower=True,trans_a=True)
        else:
            right_side = spl.blas.dtrmm(1.,chol,vec1,lower=False)
        result = spl.blas.dsyrk(1.,right_side,lower=True,trans=True,overwrite_c=True)
        result = mirror_symmetrize(result,lower=True,inplace=True)
    else:
        if lower:
            result = np.dot(spl.blas.dtrmm(1.,chol,vec1,lower=True,trans_a=True).T,spl.blas.dtrmm(1.,chol,vec2,lower=True,trans_a=True))
        else:
            result = np.dot(spl.blas.dtrmm(1.,chol,vec1,lower=False).T,spl.blas.dtrmm(1.,chol,vec2,lower=False))

    return result

def cholesky_inplace(A,inplace=True,fatal_errors=False,lower=True,clean=True):
    """ Do a cholesky decomposition, in place if inplace=True.
        For safety, the return value should still be assigned, i.e. A=cholesky_inplace(A,inplace=True).
        Cannot currently be done in place if the array is not F contiguous, but will compute decomposition anyway.
        in place will require less memory, and regardless this function should have less overhead than scipy/numpy (both in time and memory)
        If absolutely must be done in place, set fatal_errors=True.
        if lower=True return lower triangular decomposition,
        otherwise upper triangular (note lower=True is numpy default,lower=False is scipy default).
        completely ignores opposite triangle of A
    """
    if DEBUG:
        A_copy = A.copy()

    try_inplace = inplace
    #assert np.all(A==A.T)
    #dpotrf will still work on C contiguous arrays but will silently fail to do them in place
    #regardless of overwrite_a, so raise a warning or error here
    #using order='F' when creating the array or A.copy('F') when copying ensures fortran contiguous arrays.
    if (not A.flags['F_CONTIGUOUS']) and try_inplace:
        if fatal_errors:
            raise RuntimeError('algebra_utils: Cannot do cholesky decomposition in place on C continguous numpy array.')
        else:
            warn('algebra_utils: Cannot do cholesky in place on C contiguous numpy array. will output to return value',RuntimeWarning)
            try_inplace = False

    #spl.cholesky won't do them in place
    if not A.dtype==np.float_:
        raise ValueError('algebra_utils: cholesky_inplace currently only supports arrays with dtype=np.float_')

    #could write workaround, determine actual threshold (46253 is just largest successful run)
    if A.shape[0] > 46253:
        warn('algebra_utils: dpotrf may segfault for matrices this large, due to a bug in certain lapack/blas implementations')
    #result = spl.cholesky(A,lower=lower,overwrite_a=try_inplace)
    result,info = spl.lapack.dpotrf(A,lower=lower,clean=clean, overwrite_a=try_inplace)

    #Something went wrong. (spl.cholesky and np.linalg.cholesky should fail too)
    if not info==0:
        raise RuntimeError('algebra_utils: dpotrf failed with nonzero exit status '+str(info))

    if DEBUG:
        assert check_is_cholesky(result,A_copy,lower=lower,is_clean=clean,B_symmetrized=False)
        if not try_inplace:
            assert np.all(A==A_copy)

    return result

def mirror_symmetrize(A,lower=True,inplace=False):
    """copy lower triangle of a matrix into upper triangle if lower=True,vice versa if False,
    if inplace actually modify A"""
    if not inplace:
        A = A.copy()
    n = A.shape[0]
    for itr in range(0,n-1):
        if lower:
            A[itr,itr+1:n] = A[itr+1:n,itr]
        else:
            A[itr+1:n,itr] = A[itr,itr+1:n]
    return np.asfortranarray(A)

def clean_triangle(A,lower=True,inplace=False):
    """set everything but lower/upper triangle in matrix to 0,in place if inplace
        note if inplace=False, this is equivalent to tril/triu, although it is marginally faster for some reason.
        if inplace=True it is quite a bit faster because it does not create a copy of the matrix"""
    n = A.shape[0]
    if not inplace:
        A = A.copy()
    for itr in range(0,n-1):
        if lower:
            A[itr,itr+1:n] = 0.
        else:
            A[itr+1:n,itr] = 0.
    return np.asfortranarray(A)

#ie similar to np.trapz(A,xs,axis=0)
def trapz2(A,xs=None,dx=None):
    """faster trapz than numpy built in for 2d matrices along 1st dimension"""
    if xs is None:
        if dx is None:
            dx_use = 1.
        else:
            dx_use = dx
    else:
        dx_use = np.diff(xs,axis=0)
    if isinstance(dx_use,np.ndarray):
        if not dx_use.shape[0]==A.shape[0]-1:
            raise ValueError('input xs or dx has incompatible shape')
        elif dx_use.ndim>1 or A.ndim>2:
            raise ValueError('currently only support 1 dimensional dx')
        else:
            #alternate way 2x as fast if A has many elements, but more overhead if not
            #dx_tot = (dx_use[:-1:]+dx_use[1::])
            #result = (np.dot(dx_tot.T,A[1:-1:])+A[0]*dx_use[0]+A[-1]*dx_use[-1])/2.
            result = (np.dot(dx_use.T,A[:-1:])+np.dot(dx_use.T,A[1::]))/2.
    else:
        result = dx_use*np.sum(A,axis=0)-0.5*dx_use*(A[0]+A[-1])

    if DEBUG:
        if xs is None and isinstance(dx_use,np.ndarray):
            xs_use = np.hstack([0.,np.cumsum(dx_use,axis=0)])
            dx_use2 = 1.
        else:
            xs_use = xs
            dx_use2 = dx_use
        assert np.allclose(result,np.trapz(A,xs_use,dx_use2,axis=0))

    return result

def check_is_triangular(A,lower,atol_rel=1e-08,rtol=1e-05):
    """Check is A is lower/upper triangular"""
    atol_loc3 = np.max(np.abs(A))*atol_rel
    if lower:
        return np.allclose(np.tril(A),A,atol=atol_loc3,rtol=rtol)
    else:
        return np.allclose(np.triu(A),A,atol=atol_loc3,rtol=rtol)

def check_is_cholesky(A,B,atol_rel=1e-08,rtol=1e-05,lower=True,is_clean=True,B_symmetrized=True):
    """Check if A is the cholesky decomposition of B
        lower = True if lower triangular, if is_clean assume opposite triangle should be 0.
        B_symmetrized for whether to assume both triangles of B are filled"""
    if not is_clean:
        if lower:
            A = np.tril(A)
        else:
            A = np.triu(A)
    if not B_symmetrized:
        B = mirror_symmetrize(B,lower,False)
    atol_loc1 = np.max(np.abs(B))*atol_rel
    test1 = check_is_triangular(A,lower,atol_rel,rtol)
    if lower:
        return test1 and np.allclose(np.dot(A,A.T),B,atol=atol_loc1,rtol=rtol)
    else:
        return test1 and np.allclose(np.dot(A.T,A),B,atol=atol_loc1,rtol=rtol)

def check_is_cholesky_inv(A,B,atol_rel=1e-08,rtol=1e-05,lower=True,is_clean=True,B_symmetrized=True):
    """check if A is the inverse cholesky decomposition of B
        lower or upper triangular, is_clean for whether to assume other triangle should be 0.,
        B_symmetrized for whether to assume both triangles of B are filled"""
    if not is_clean:
        if lower:
            A = np.tril(A)
        else:
            A = np.triu(A)

    if not B_symmetrized:
        B = mirror_symmetrize(B,lower,False)

    chol = np.linalg.pinv(A)
    B_inv = np.linalg.pinv(B)

    atol_loc1 = atol_rel*np.max(np.abs(B))
    atol_loc2 = atol_rel*np.max(np.abs(B_inv))

    test3 = check_is_triangular(A,lower,atol_rel,rtol)
    if lower:
        test1 = np.allclose(np.dot(A.T,A),B_inv,atol=atol_loc2,rtol=rtol)
        test2 = np.allclose(np.dot(chol,chol.T),B,atol=atol_loc1,rtol=rtol)
    else:
        test1 = np.allclose(np.dot(A,A.T),B_inv,atol=atol_loc2,rtol=rtol)
        test2 = np.allclose(np.dot(chol.T,chol),B,atol=atol_loc1,rtol=rtol)
    return test1 and test2 and test3
