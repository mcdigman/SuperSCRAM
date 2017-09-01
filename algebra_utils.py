import numpy as np
import scipy.linalg as spl

from warnings import warn

#TODO: In the long run, some of these functions would benefit from some low level optimizations, like doing cholesky decompositions
#and inverses in place (or even lower level, like storing two cholesky decompositions in the same matrix and just masking the one we don't need when doing any given operation), 
#because the memory consumption of these matrices is the code's primary performance bottleneck.
#scipy.linalg.cholesky's overwrite_a option is not reliable for doing things in place, so I would probably need
#some lower level (lapack?) calls, and I would prefer to wait until the code specification is more stable/good unit tests have been writen.
#potentially use ztrmm for triangular dot products


#Get the inverse cholesky decomposition of a matrix A
def get_inv_cholesky(A,lower=True):
    return invert_triangular(cholesky_inplace(A,inplace=False,lower=lower),lower=lower)

#Get the cholesky decomposition of the inverse of a matrix a
#TODO add safety checks,tests
def get_cholesky_inv(A,lower=True):
    return np.rot90(spl.lapack.dtrtri(cholesky_inplace(np.rot90(A,2),lower=(not lower),inplace=False),lower=(not lower))[0],2)

#TODO: replace with spl.lapack.dtrtri
def invert_triangular(A,lower=True):
    return spl.solve_triangular(A,np.identity(A.shape[0]),lower=lower,overwrite_b=True)

def get_mat_from_inv_cholesky(A,lower=True):
    chol_mat = invert_triangular(A,lower)
    return np.dot(chol_mat,chol_mat.T)

#compute inverse of positive definite matrix using cholesky decomposition
#cholesky_given = True if A already is the cholesky decomposition of the covariance
#TODO: replace part with spl.lapack.dpotri
def ch_inv(A,cholesky_given=False,lower=True):
    #chol = np.linalg.cholesky(A)
    #chol_inv = np.linalg.solve(np.linalg.cholesky(A),np.identity(A.shape[0]))
    if cholesky_given:
        chol_inv = A
    else:
        #chol_inv = spl.solve_triangular(np.linalg.cholesky(A),np.identity(A.shape[0]),lower=lower,overwrite_b=True)
        chol_inv = get_inv_cholesky(A,lower=lower)
    if lower:
        return np.dot(chol_inv.T,chol_inv)
    else:
        return np.dot(chol_inv,chol_inv.T)


def cholesky_inv_contract(A,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=True):
    if cholesky_given:
        chol_inv = A
    else:
        chol_inv = get_inv_cholesky(A,lower)

    #potentially Save some time if inputs are identical
    #TODO: check if memory profile worse
    if identical_inputs:
        if lower:
            right_side = np.dot(chol_inv,vec1)
        else:

            right_side = np.dot(chol_inv.T,vec1)
        result = np.dot(right_side.T,right_side)
    else:
        if lower:
            result = np.dot(np.dot(vec1.T,chol_inv.T),np.dot(chol_inv,vec2))
        else:
            result = np.dot(np.dot(vec1.T,chol_inv),np.dot(chol_inv.T,vec2))

    return result

def cholesky_contract(A,vec1,vec2,cholesky_given=False,identical_inputs=False,lower=True):
    if cholesky_given:
        chol = A
    else:
        chol = cholesky_inplace(A,lower=lower,inplace=False)

    #potentially Save some time if inputs are identical
    #TODO: check if memory profile worse
    if identical_inputs:
        if lower:
            right_side = np.dot(chol.T,vec1)
        else:

            right_side = np.dot(chol,vec1)
        result = np.dot(right_side.T,right_side)
    else:
        if lower:
            result = np.dot(np.dot(vec1.T,chol),np.dot(chol.T,vec2))
        else:
            result = np.dot(np.dot(vec1.T,chol.T),np.dot(chol,vec2))

    return result

#Do a cholesky decomposition, in place if inplace=True. For safety, the return value should still be assigned, i.e. A=cholesky_inplace(A,inplace=True). 
#Cannot currently be done in place if the array is not F contiguous, but will compute decomposition anyway.
#in place will require less memory, and regardless this function should have less overhead than scipy/numpy (both in time and memory)
#If absolutely must be done in place, set fatal_errors=True.
#if lower=True return lower triangular decomposition, otherwise upper triangular (note lower=True is numpy default,lower=False is scipy default).
#TODO: add more sanity check assertions, such as if lapack is actually installed
def cholesky_inplace(A,inplace=True,fatal_errors=False,lower=True):

    try_inplace = inplace

    #dpotrf will still work on C contiguous arrays but will silently fail to do them in place regardless of overwrite_a, so raise a warning or error here 
    #using order='F' when creating the array or A.copy('F') when copying ensures fortran contiguous arrays.
    if (not A.flags['F_CONTIGUOUS']) and try_inplace:
        if fatal_errors:
            raise RuntimeError('algebra_utils: Cannot do cholesky decomposition in place on C continguous numpy array.') 
        else:
            warn('algebra_utils: Cannot do cholesky decomposition in place on C continguous numpy array. will output to return value',RuntimeWarning)
            try_inplace = False
    
    #spl.cholesky won't do them in place TODO actually handle it
    if not A.dtype == np.float_:
        raise ValueError('algebra_utils: cholesky_inplace currently only supports arrays with dtype=np.float_')

    #TODO maybe workaround, determine actual threshold (46253 is just largest successful run)
    if A.shape[0] > 46253:
        warn('algebra_utils: dpotrf may segfault for matrices this large, due to a bug in certain lapack/blas implementations')
    #result = spl.cholesky(A,lower=lower,overwrite_a=try_inplace)
    result,info = spl.lapack.dpotrf(A,lower=lower,clean=1, overwrite_a=try_inplace)
    
    #Something went wrong. (spl.cholesky and np.linalg.cholesky should fail too)
    #if not info==0:
    #    raise RuntimeError('algebra_utils: dpotrf failed with nonzero exit status')

    # check if L is the desired L cholesky factor
    #assert np.allclose(np.dot(L,L.T), A)
    return result

if __name__=='__main__':
    from time import time

    n_A = 1000
    n_iter = 10
    times_chol1 = np.zeros(n_iter)
    times_chol2 = np.zeros(n_iter)
    times_inv = np.zeros(n_iter)
    for i in xrange(n_iter):
        A = np.random.random((n_A,n_A))
        A = np.dot(A.T,A)
        V1 = np.random.random(n_A)
        V2 = np.random.random(n_A)

        t1 = time()
        AI1 = np.dot(np.dot(V1,np.linalg.inv(A)),V2)
        t2 = time()
        AI2 = cholesky_inv_contract(A,V1,V2)
        #AI2 = np.linalg.pinv(A)
        t3 = time()
        AI3 = np.dot(np.dot(V1,ch_inv(A)),V2)
        t4 = time()
        times_chol2[i] = t4-t3
        times_chol1[i] = t3-t2
        times_inv[i] = t2-t1
    print "inv avg time,std: ",np.average(times_inv),np.std(times_inv)
   # print "pinv time: ",t3-t2
    print "cholesky_inv_contract avg time,std: ",np.average(times_chol1),np.std(times_chol1)
    print "cholesky_inv avg time,std: ",np.average(times_chol2),np.std(times_chol2)
    
   # print "inv mean error ",np.average(abs(np.linalg.eigvals(np.dot(AI1,A))-np.diag(np.identity(A.shape[0]))))
    #print "pinv mean error ",np.average(abs(np.linalg.eigvals(np.dot(AI2,A))-np.diag(np.identity(A.shape[0]))))
    #print "cholesky_inv mean error ",np.average(abs(np.linalg.eigvals(np.dot(AI3,A))-np.diag(np.identity(A.shape[0]))))

