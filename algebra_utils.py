import numpy as np
from scipy.linalg import solve_triangular,solve

#Get the inverse of the cholesky decomposition
def get_inv_cholesky(A):
    return solve_triangular(np.linalg.cholesky(A),np.identity(A.shape[0]),lower=True,overwrite_b=True)

def invert_triangular(A):
    return solve_triangular(A,np.identity(A.shape[0]),lower=True,overwrite_b=True)

def get_mat_from_inv_cholesky(A):
    chol_mat = invert_triangular(A)
    return np.dot(chol_mat,chol_mat.T)

#compute inverse of positive definite matrix using cholesky decomposition
#cholesky_given = True if A already is the cholesky decomposition of the covariance
def ch_inv(A,cholesky_given=False):
    #chol = np.linalg.cholesky(A)
    #chol_inv = np.linalg.solve(np.linalg.cholesky(A),np.identity(A.shape[0]))
    if cholesky_given:
        chol_inv = A
    else:
        chol_inv = solve_triangular(np.linalg.cholesky(A),np.identity(A.shape[0]),lower=True,overwrite_b=True)
    return np.dot(chol_inv.T,chol_inv)

def cholesky_inv_contract(A,vec1,vec2,cholesky_given=False,identical_inputs=False):
    if cholesky_given:
        chol_inv = A
    else:
        chol_inv = solve_triangular(np.linalg.cholesky(A),np.identity(A.shape[0]),lower=True,overwrite_b=True)
    #chol_inv = np.linalg.solve(np.linalg.cholesky(A),np.identity(A.shape[0]))

    #Save some time if inputs are identical
    if identical_inputs:
        right_side = np.dot(chol_inv,vec2)
        result = np.dot(right_side.T,right_side)
    else:
        result = np.dot(np.dot(vec1,chol_inv.T),np.dot(chol_inv,vec2))
    return result

def inverse_cholesky(A):
    return solve_triangular(np.linalg.cholesky(A),np.identity(A.shape[0]),lower=True,overwrite_b=True)

if __name__=='__main__':
    from time import time

    n_A = 1000
    n_iter = 10
    times_chol1 = np.zeros(n_iter)
    times_chol2 = np.zeros(n_iter)
    times_inv = np.zeros(n_iter)
    for i in range(n_iter):
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
        AI3 = np.dot(np.dot(V1,cholesky_inv(A)),V2)
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

