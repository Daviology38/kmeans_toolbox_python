"""def kmeans_ci(X=None, stand=None, weighting=None, prop=None, nclus=None, nsim=None):
    [CI,K]=kmeans_ci(X,stand,weighting,prop,nclus,nsim);

    This function makes a k-means clustering of the rows of the input matrix
    'X' and computes the classifiability index for a partition of the
    rows of the input matrix 'X' using a dynamical clustering algorithm
    (k-means) into 'nclus' clusters. There are 'nsim' different partitions
    that are compared. The clustering is performed in EOF-space taking the
    leading EOF accouting for 'prop' proprtion of the total variance. The
    algorithm computes for each cluster the mean anomaly correlation between
    its centroid and the other clusters.

    Input
    'X': input matrix to be classified (rows are classified)
    'stand': option for standardising the matrix (by columns) = 'm' for just
    removing the long-term mean and ='s' for standardising to zero mean and
    unit variance. if stand is empty, the matrix is not standardized.
    'weighting': vector for weighting the columns of X.
    'prop': scalar giving the proportion of variance to be retained in the
    EOF pre-filtering step. if prop is empty, no EOF-prefiltering is done
    'nclus': number of cluster
    'nsim' : number of different partition simulated (typically nsim= 50-100)

    Outpurs
    'CI': scalar giving the classifiability index
    'K': row vector giving the cluster for each row of input matrix 'X'.

    ref. Michelangeli et al., JAS, 1995 (1230-1246)

    Vincent Moron
    June 2006

    modification
    March 2010: the random seed is changed into "stream=RandStream('mrg32k3a')"
    September 2010: "nclus" could be a vector instead of a scalar to test in
    a loop different number of clusters
    
    modification:
    February 2020: Ported program from Matlab into Python (David Coe UMass Lowell)

"""



#Imports
import os
import numpy as np
from netCDF4 import Dataset
import xarray as xr
from sklearn.cluster import KMeans


def kmeans_ci(X=None, stand=None, weighting=None, prop=None, nclus=None, nsim=None):

    #Check if the array has been standardized prior to running kmeans_ci
    if stand != None:
        
        #Standardize the grid point values by using the mean and standard deviation
        nan_mean = np.nanmean(X, axis=0)
        nan_std = np.nanstd(X, axis=0, ddof=1)
        X = (X - nan_mean) / nan_std
    
    r = X.shape[0]
    c = X.shape[1]
    nclus = nclus
    nsim = nsim

    #Check if EOF pre-filtering has been performed on the array
    if prop != None:
        
        #Find the Singular Value Decomposition for the array X
        U, S, V = np.linalg.svd(X)
        # NOTE in Python S is returned as diag(S) and V needs to be transposed (V.T)
        V = V.T
        s = S ** 2
        sc = s / np.sum(s, axis=1)
        a = np.where(np.cumsum(sc, axis=1) > prop)
        a = a[0] + 1
        PC = U[:, 0:a] @ diag(S)[0:a, 0:a]
        print('Pre-filtering using EOF retaining the first ' + str(a) + ' components done ...')
        
    else:
        
        PC = X
        print('No EOF prefiltering ...')
    
    #Declare arrays to hold the cluster assignments and CI value
    Kens = np.ones((PC.shape[0], nclus, nsim))
    K = np.zeros(PC.shape[0])
    CI = np.zeros(1)
    
    #Loop through the data array once (redundant here, but left in for consistency to MATLAB script)
    for NC in range(0, 1, 1):
        
        #Set arraus to hold the data
        mean_cluster = np.zeros((nclus, c))
        ACCmax = np.zeros((nclus, nsim))
        k = np.zeros((PC.shape[0], nsim))
        MC = np.zeros(((nclus) * c, nsim))

        print(['K means clustering with ' + str(nclus) + ' clusters begins ...'])
        
        #Loop through the number of simulations and perform kmeans
        for i in range(nsim):
            
            k[:, i] = KMeans(n_clusters=nclus, init='k-means++', 
                             n_init=1, max_iter=1000
                             ).fit_predict(X)  
            
            #Calculate the mean of each cluster using the assigned cluster values for 
            #each row in array X
            for j in range(nclus):
                
                lj = len(np.nonzero(k[:, i] == j)[0])
                
                if lj > 1:
                    
                    mean_cluster[j, :] = np.mean(PC[np.nonzero(k[:, i] == j)[0], :],
                                                 axis=0)
                else:
                    
                    mean_cluster[j, :] = X[k[:, i] == j]
            
            #Standardize the new array of means for the clusters        
            nan_mean = np.nanmean(mean_cluster.T, axis=0)
            nan_std = np.nanstd(mean_cluster.T, axis=0, ddof=1)
            mean_cluster2 = (mean_cluster.T - nan_mean) / nan_std
            
            #Reset the mean_cluster array to zeros and store the standardized mean_cluster2 array. 
            #This array contains the centroids of each cluster.
            #Note: We need the 'F' option to store it in Fortran style ordering
            mean_cluster = np.zeros((nclus, c))
            MC[:, i] = mean_cluster2.flatten('F')  
        
        #Store all the cluster values over all simulations.    
        Kens[:, NC - 1, :] = k
        
        #Calculate the Anomaly Correlation Coefficient (ACC) between all columns (centroids) 
        #of the MC array. This allows the user to determine the optimal arrangement
        #of centroids that maximizes the ACC.
        for i in range(nclus):
            
            for j in range(nsim):
                
                sample1 = MC[(i*c):(i+1) * c, j]
                a = np.nonzero(j != np.arange(0, nsim))[0]
                sample2 = MC[:, a].reshape(c, (nsim-1) * nclus,
                                           order='F').copy()
                ind = np.isnan(sample1)
                sample1[ind] = 0
                ind = np.isnan(sample2)
                sample2[ind] = 0
                ACC = (1 / (c-1)) * sample1.conj().T @ sample2
                ACC = ACC.reshape(nclus, nsim-1, order='F').copy()
                ACCmax[i, j] = np.mean(ACC.max(0))
        
        #Find the optimal partition from the generated list of ACC values.
        part = np.nonzero(np.mean(ACCmax, axis=0) 
                          == np.max(np.mean(ACCmax, axis=0)))[0]

        # Classification index for the partition ton 'nclus' clusters
        CI[NC] = np.mean(ACCmax)  
        if len(part) > 1: part = part[0]

        # best partition that maximize the ACC with the nsim-1 other partitions
        K[:] = np.squeeze(k[:, part])  
        
    return K, CI