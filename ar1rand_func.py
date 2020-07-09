"""def ar1rand(X=None, nsim=None):

         Red-noise random simulation of 'nsim' time series of the
         same length as the input vector 'X' and having the same
         one-order auto-correlation and mean and variance as 'X'

         Vincent Moron
         June 2006
         Ported to Python by
         David Coe
         UMass Lowell
         February 2020

"""

import numpy as np

def copy(y = None, N = None):
""" [X]=copy(Y,N);

    This function copies 'N' times a matrix 'Y'.

    Input
    'X' : matrix of real number
    'N' : scalar integer giving the number of copies

    Output
    'Y' : matrix 'X' copied 'N' times (the copy is done in row)

    Vincent Moron
    July 1997
    Ported to Python by David Coe
    UMass Lowell
    February 2020
"""
    NR = 1
    NC = len(y)

    x = y.conj().T
    x = x.flatten('F')
    x = (x[:, None]) @ (np.ones((1, N)))
    x = x.reshape(NC, NR * N, order = 'F')
    x = x.conj().T

    return x

def scale_mean_var(X, Y):

""" [Z]=scale_mean_var(X,Y);

    This function scales the columns of the input matrix 'X', so that its
    mean and variance match the ones of the input matrix 'Y'.

    Input
    'X' : matrix of real number to be scaled
    'Y' : matrix of real number used in the scaling

    Output
    'Z' : matrix of real number (the mean and variance of the columns of 'Z'
    are equal to the ones of 'Y'

    Vincent Moron
    Nov 2005
    Ported to Python by David Coe
    UMass Lowell
    February 2020
"""

    nr, nc = X.shape

    my = np.mean(Y, axis=0)
    sy = np.std(Y, axis=0, ddof=1)
    mx = np.mean(X, axis=0)
    sx = np.std(X, axis=0, ddof=1)

    Z = (X @ np.diag(sy/sx))
    dm = np.mean(Z, axis=0) - my
    dm = np.ones((nr, 1)) * dm
    Z = Z - dm

    return Z


def ar1rand(X = None, nsim = None):

    X = X.flatten('F')
    n = len(X)
    c = np.corrcoef(X[0:n-1], X[1:n])[0, 1]
    d = 1 - c
    Y = np.zeros((n, nsim))

    Y[0,:] = np.random.randn(1, nsim)
    Z = np.random.randn(n, nsim)
    Z = scale_mean_var(Z, copy(X[1:n].conj().T, nsim).conj().T)

    for j in range(1, n):

        Y[j, :] = (Y[j-1, :] * c) + (Z[j, :] * d)

    Y = scale_mean_var(Y, copy(X.conj().T, nsim).conj().T)
    
    return Y

