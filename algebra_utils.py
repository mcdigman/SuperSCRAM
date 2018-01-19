"""provides implementations/wrappers of some fast algebra utilities"""
from warnings import warn
import numpy as np
import scipy.linalg as spl

DEBUG = True
#TODO: In the long run, some of these functions would benefit from some low level optimizations, like doing cholesky decompositions
#and inverses in place (or even lower level, like storing two cholesky decompositions in the same matrix and just masking the one we don't need when doing any given operation),
#because the memory consumption of these matrices is the code's primary performance bottleneck.
#scipy.linalg.cholesky's overwrite_a option is not reliable for doing things in place, so I would probably need
#some lower level (lapack?) calls, and I would prefer to wait until the code specification is more stable/good unit tests have been writen.
#potentially use ztrmm for triangular dot products

#TODO inplace not tested for get_inv_cholesky, get_cholesky_inv,invert_triangular
def get_inv_cholesky(A,lower=True,inplace=False,clean=True):
    """Get the inverse cholesky decomposition of a matrix A
        inplace: allow overwriting A, does not guarantee overwriting A
        clean: whether to set unneeded half of result to 0
        completely ignores opposite triangle of A"""
    if DEBUG:
        A_copy = A.copy()

    result = invert_triangular(cholesky_inplace(A,inplace=inplace,lower=lower,clean=clean),lower=lower,inplace=inplace)

    if DEBUG:
        assert check_is_cholesky_inv(result,A_copy,is_clean=clean,lower=lower,B_symmetrized=False)
        if not inplace:
            assert np.all(A_copy==A)
    return result

#TODO add safety checks,tests
def get_cholesky_inv(A,lower=True,inplace=False,clean=True):
    """Get the cholesky decomposition of the inverse of a matrix A"""
    if DEBUG:
        A_copy = A.copy()
    result = np.asfortranarray(np.rot90(spl.lapack.dtrtri(cholesky_inplace(np.asfortranarray(np.rot90(A,2)),lower=(not lower),inplace=inplace,clean=clean),lower=(not lower),overwrite_c=inplace)[0],2))
    if DEBUG:
        if clean:
            assert check_is_triangular(result,lower)
        if not inplace:
            assert np.all(A_copy==A)
    return result

#TODO add safety checks
#TODO handle clean
def invert_triangular(A,lower=True,inplace=False,clean=True):
    """invert a triangular matrix,
        completely ignores opposite triangle"""
    if DEBUG:
        A_copy = A.copy()
    result = spl.lapack.dtrtri(A,lower=lower,overwrite_c=inplace)[0]
    if clean:
        if lower:
            result = np.tril(result)
        else:
            result = np.triu(result)
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
        A_copy=A.copy()
    chol_mat = invert_triangular(A,lower,inplace=inplace,clean=False)
    result = np.rot90(spl.lapack.dlauum(np.rot90(chol_mat,2),lower=not lower,overwrite_c=inplace)[0],2)
    if clean:
        if lower:
            result = np.tril(result)
        else:
            result = np.triu(result)
        result = result+result.T-np.diagflat(np.diag(result))
    if DEBUG:
        if not inplace:
            assert np.all(A_copy==A)
        if clean:
            assert np.all(result.T==result)
    return result

#cholesky_given = True if A already is the cholesky decomposition of the covariance
#TODO add test cases for ignoring opposite triangle
def ch_inv(A,cholesky_given=False,lower=True,inplace=False,clean=True):
    """ compute inverse of positive definite matrix using cholesky decomposition
        clean: whether to symmetrize output
        completely ignores opposite triangle"""

    #chol = np.linalg.cholesky(A)
    #chol_inv = np.linalg.solve(np.linalg.cholesky(A),np.identity(A.shape[0]))
    if DEBUG:
        A_copy = A.copy()

    if cholesky_given:
        chol_inv = A
    else:
        chol_inv = get_inv_cholesky(A,lower=lower,inplace=inplace,clean=False)

    result = spl.lapack.dlauum(chol_inv,lower=lower,overwrite_c=inplace)[0]
    if clean:
        if lower:
            result = np.tril(result)
        else:
            result = np.triu(result)
        result = result+result.T-np.diagflat(np.diag(result))
    if DEBUG:
        if clean:
            assert np.all(result==result.T)
        if not inplace:
            assert np.all(A==A_copy)
    return result


#TODO could add option to permit inplace if helpful
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
        result = np.dot(right_side.T,right_side)
        #result = spl.blas.dsyrk(1.,right_side,lower=True,trans=True)
        #result = result+result.T-np.diagflat(np.diag(result))
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
        result = spl.blas.dsyrk(1.,right_side,lower=True,trans=True)
        result = result+result.T-np.diagflat(np.diag(result))
    else:
        if lower:
            result = np.dot(spl.blas.dtrmm(1.,chol,vec1,lower=True,trans_a=True).T,spl.blas.dtrmm(1.,chol,vec2,lower=True,trans_a=True))
        else:
            result = np.dot(spl.blas.dtrmm(1.,chol,vec1,lower=False).T,spl.blas.dtrmm(1.,chol,vec2,lower=False))

    return result

#TODO: add more sanity check assertions, such as if lapack is actually installed
def cholesky_inplace(A,inplace=True,fatal_errors=False,lower=True,clean=True):
    """ Do a cholesky decomposition, in place if inplace=True.
        For safety, the return value should still be assigned, i.e. A=cholesky_inplace(A,inplace=True).
        Cannot currently be done in place if the array is not F contiguous, but will compute decomposition anyway.
        in place will require less memory, and regardless this function should have less overhead than scipy/numpy (both in time and memory)
        If absolutely must be done in place, set fatal_errors=True.
        if lower=True return lower triangular decomposition, otherwise upper triangular (note lower=True is numpy default,lower=False is scipy default).
        completely ignores opposite triangle of A
    """
    if DEBUG:
        A_copy = A.copy()

    try_inplace = inplace
    #assert np.all(A==A.T)
    #dpotrf will still work on C contiguous arrays but will silently fail to do them in place regardless of overwrite_a, so raise a warning or error here
    #using order='F' when creating the array or A.copy('F') when copying ensures fortran contiguous arrays.
    if (not A.flags['F_CONTIGUOUS']) and try_inplace:
        if fatal_errors:
            raise RuntimeError('algebra_utils: Cannot do cholesky decomposition in place on C continguous numpy array.')
        else:
            warn('algebra_utils: Cannot do cholesky decomposition in place on C contiguous numpy array. will output to return value',RuntimeWarning)
            try_inplace = False

    #spl.cholesky won't do them in place
    if not A.dtype == np.float_:
        raise ValueError('algebra_utils: cholesky_inplace currently only supports arrays with dtype=np.float_')

    #TODO maybe workaround, determine actual threshold (46253 is just largest successful run)
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

#ie similar to np.trapz(A,xs,axis=0)
#TODO test
#TODO just take 1 input xs and dx
#TODO what if xs is a matrix?
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
        if not dx_use.shape[0] == A.shape[0]-1:
            raise ValueError('input xs or dx has incompatible shape')
        elif dx_use.ndim>1 or A.ndim>2:
            raise ValueError('currently only support 1 dimensional dx')
        else:
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
        lower=True if lower triangular, if is_clean assume opposite triangle should be 0.
        B_symmetrized for whether to assume both triangles of B are filled"""
    if not is_clean:
        if lower:
            A = np.tril(A)
        else:
            A = np.triu(A)
    if not B_symmetrized:
        if lower:
            B = np.tril(B)
        else:
            B= np.triu(B)
        B = B+B.T-np.diagflat(np.diag(B))

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
        if lower:
            B = np.tril(B)
        else:
            B= np.triu(B)
        B = B+B.T-np.diagflat(np.diag(B))

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
