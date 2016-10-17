import numpy as np
import numpy.linalg as npl
import scipy as sp
import scipy.linalg as spl
import warnings
from time import time
from algebra_utils import cholesky_inplace,get_inv_cholesky,ch_inv
import sys

#Check if A is the cholesky decomposition of B
def check_is_cholesky(A,B,atol=1e-08,rtol=1e-05):
   return np.allclose(np.dot(A,A.T),B,atol=atol,rtol=rtol)

def check_is_inv_cholesky(A,B,atol=1e-08,rtol=1e-05):
    chol = npl.pinv(A)
    return np.allclose(np.dot(chol,chol.T),B,atol=atol,rtol=rtol)


