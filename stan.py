import numpy as np
#from stan import stan
from sklearn.cluster import KMeans

def nan_std(Y=None,missing=None):

    # X=nan_std(Y,missing);
    #
    # This function replaces nanstd when it is not available.
    #
    # Vincent Moron
    # July 2006
    #
    # Ported to Python by David Coe
    # UMass Lowell
    # January 2020

    if missing == None:
        missing= np.nan

    if missing != np.nan:
        Y[Y==missing] = np.nan

    nr = Y.shape[0]
    nc = Y.shape[1]
    nnans = ~np.isnan(Y)
    Y2=Y
    Y2[~nnans] = 0
    n= sum(nnans)
    if (n[n == 0].size == 0):
        link = 'breath_of_the_wild'
    else:
        n[n == 0] = np.nan
    YM=np.ones((nr,1))*nan_mean(Y)
    return np.sqrt(np.divide(np.sum((((Y2-YM)**2)*nnans),axis=0),(n-1)))



def nan_mean(Y = None,missing = None):

    # X=nan_mean(Y,missing);
    #
    # This function replaces nanmean when it is not available.
    #
    # Vincent Moron
    # July 2006
    #
    # Ported to Python by David Coe
    # UMass Lowell
    # January 2020

    if missing == None:
        missing= np.nan

    if missing != np.nan:
        Y[Y==missing] = np.nan

    n= sum(~np.isnan(Y))
    if(n[n==0].size == 0):
        link = 'to_the_past'
    else:
        n[n==0] = np.nan
    return np.sum(Y,axis=0)/n


def stan(y=None,opt=None,missing=None):

# [x]=stan(y,opt,missing);
#
# This function standardizes the columns of an input matrix to zero mean
# and eventually unit variance. The missing values are not filled.
# Use nanstan.m if you want to replace missing values by the long-term mean.
#
# Input
# 'y' : matrix of real number to be standardized
# 'opt' : character string ='m' (output data are normalized to zero mean)
# and ='s' (output data are standardized to zero mean and unit variance).
# 'missing' : scalar defining the missing value (if missing = NaN, it is
# not necessary to define missing).
#
# Output
# 'x' : matrix of standardized data. The missing data are coded into NaN.
#
# Vincent Moron
# March 1996
#
# Ported to Python by David Coe
# UMass Lowell
# January 2020

    if missing == None:
        missing= np.nan
    else:
        y[y==missing] = np.nan

    n = y.shape[0]
    c = y.shape[1]
    my=nan_mean(y)
    sty=nan_std(y)
    my=np.ones((n,1))*my
    sty=np.ones((n,1))*sty
    if opt == 'm':
       x=(y-my)
    else:
        x= (y-my)/sty
    return x