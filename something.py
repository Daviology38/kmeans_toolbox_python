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


def kmeans_ci2(X=None, stand=None, weighting=None, prop=None, nclus=None, nsim=None):
    # [CI,K]=kmeans_ci(X,stand,weighting,prop,nclus,nsim);
    #
    # This function makes a k-means clustering of the rows of the input matrix
    # 'X' and computes the classifiability index for a partition of the
    # rows of the input matrix 'X' using a dynamical clustering algorithm
    # (k-means) into 'nclus' clusters. There are 'nsim' different partitions
    # that are compared. The clustering is performed in EOF-space taking the
    # leading EOF accouting for 'prop' proprtion of the total variance. The
    # algorithm computes for each cluster the mean anomaly correlation between
    # its centroid and the other clusters.
    #
    # Input
    # 'X': input matrix to be classified (rows are classified)
    # 'stand': option for standardising the matrix (by columns) = 'm' for just
    # removing the long-term mean and ='s' for standardising to zero mean and
    # unit variance. if stand is empty, the matrix is not standardized.
    # 'weighting': vector for weighting the columns of X.
    # 'prop': scalar giving the proportion of variance to be retained in the
    # EOF pre-filtering step. if prop is empty, no EOF-prefiltering is done
    # 'nclus': number of cluster
    # 'nsim' : number of different partition simulated (typically nsim= 50-100)
    #
    # Outpurs
    # 'CI': scalar giving the classifiability index
    # 'K': row vector giving the cluster for each row of input matrix 'X'.
    #
    # ref. Michelangeli et al., JAS, 1995 (1230-1246)
    #
    # Vincent Moron
    # June 2006
    #
    # modification
    # March 2010: the random seed is changed into "stream=RandStream('mrg32k3a')"
    # September 2010: "nclus" could be a vector instead of a scalar to test in
    # a loop different number of clusters

    # rand('state',sum(100*clock));
    random_generator = np.random.RandomState()  # stream=RandStream('mrg32k3a');

    if stand == None:
        X = stan(X, stand)

    R = X.shape[0]
    C = X.shape[1]

    if prop == None:
        U, S, V = np.linalg.svd(X, full_matrices=True)
        s = np.diag(S) ** 2
        sc = s / np.sum(s, axis=1)
        a = np.where(np.cumsum(sc, axis=1) > prop)
        a = a(1)
        PC = U[:, 1:a] @ S[1:a, 1:a]
        print('Pre-filtering using EOF retaining the first ' + str(a) + ' components done ...')
    else:
        PC = X
        print('No EOF prefiltering ...')
    r = X.shape[0]
    c = X.shape[1]
    print(c)
    Kens = np.nan * np.ones((PC.shape[1], nclus, nsim))
    K = np.zeros((PC.shape[0], nclus))
    CI = np.zeros(nclus)
    for NC in range(nclus):  # NC=1:length(nclus)
        # clear MC mean_cluster mean_cluster2 k ACC ACCmax part
        mean_cluster = np.zeros((nclus, c))
        ACCmax = np.zeros((nclus, nsim))
        k = np.zeros((PC.shape[0], nsim))
        MC = np.zeros((nclus * R, nsim))
        print(['K means clustering with ' + str(nclus) + ' clusters begins ...'])
        for i in range(nsim):  # i=1:nsim;
            k[:, i] = KMeans(n_clusters=nclus, max_iter=1000).fit_predict(PC)  # k(:,i)=kmeans(PC,nclus(NC),'Maxiter',1000,'EmptyAction','singleton');
            for j in range(nclus):  # j=1:nclus(NC);
                lj = len(np.nonzero(k[:, i] == j)[0])  # length(find(k(:,i)==j)); Since it is a tuple, use [0] to grab data
                if lj > 1:
                    mean_cluster[j, :] = np.mean(X[k[:,i]==j], axis=0)  # mean(X(find(k(:,i)==j),:))
                else:
                    mean_cluster[j, :] = X[k[:,i]==j]  # X(find(k(:,i)==j),:)
            mean_cluster2 = stan(mean_cluster.conj().T, opt = 's')
            mean_cluster = np.zeros((nclus, c))
            MC[:, i]= mean_cluster2.flatten()  # centroids stored in MC matrix
        Kens[:, NC,:] = k
        for i in range(nclus): #i=1:nclus(NC);
            for j in range(nsim): #j = 1:nsim;
                sample1 = MC[(i*c) :(i+1)*c,j] #MC(((i - 1) * c) + 1:i * c, j);
                a = np.argwhere(j != np.arange(0,nsim)) #find(j~ = [1:nsim]);
                sample2 = MC[:,a].reshape(c,(nsim-1)*nclus,order='F') #reshape(MC(:, a), c, (nsim - 1) * nclus(NC))
                ind = np.isnan(sample1)
                sample1[ind] = 0
                ind = np.isnan(sample2)
                sample2[ind] = 0
                ACC = (1 / (c - 1))*(sample1.conj() @ sample2)
                ACC = ACC.reshape(nclus,nsim-1,order='F') #(ACC, nclus(NC), nsim - 1);
                ACCmax[i, j] = np.mean(np.max(ACC)) #mean(max(ACC));  # considering the mean instead the min
        part = np.nonzero(np.mean(ACCmax,axis=0) == np.max(np.mean(ACCmax,axis=0)))[0] #find(mean(ACCmax) == max(mean(ACCmax)))
        CI[NC] = np.mean(np.mean(ACCmax,axis=0))  # Classification index for the partition ton 'nclus' clusters
        if len(part) > 1: part=part[0]
        K[:,NC]=k[:, part]  # best partition that maximize the ACC with the nsim-1 other partitions
    return K, CI