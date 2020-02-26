#Do k-means clustering analysis on multiple varuiables
#Adapted from Agel and Moron kmeanstoolbox in MATLAB
#
#Ported to Python by David Coe
#UMass Lowell
#January 2020

#Import stuff
import os
import numpy as np
from netCDF4 import Dataset
import xarray as xr
from kmeans_ci2 import kmeans_ci, stan, nan_mean, nan_std
from sklearn.cluster import KMeans

#Define the map2mat function we will use to arrange the data
def map2mat(F,C):
    #This function takes in an array (C) and a boolean array (F)
    #C must be 3D and F must be 2D
    #We are combining the second and third dimensions of C into a new array (D)
    #The points from C that will be taken will be any locations in F that are True (1)

    #get the dimensions
    tps, nolon, nolat = C.shape

    #output array will be
    D = np.zeros((tps,nolon*nolat))

    #Now map the data
    ipt = 0

    for iy in range(nolat):
        for ix in range(nolon):
            if F[ix,iy] > 0:
                #If the data is to be kept, put into array
                D[:,ipt] = C[:,ix,iy].squeeze()
                ipt = ipt + 1

    return D

#Define the directory to write the data to and see if it exists
#If it does not exist, create it
outdir = 'era5_son'
if not os.path.exists(outdir):
    os.makedirs(outdir)

#Define our variables to start the run with (varianve retained, max clusters, simulations)
retain = 0.85
maxclust = 10
nsim = 100 #for Monte Carlo approximation
#random_generator = np.random.RandomState()  # stream=RandStream('mrg32k3a')

#Get our LAT/LON grid
lats = Dataset("F:/ERA5/h500_son/h5001979.nc").variables['latitude'][160:241]
lons = Dataset("F:/ERA5/h500_son/h5001979.nc").variables['longitude'][1080:1201]
LON, LAT = np.meshgrid(lons,lats)

#Load in the data using xarray
h500 = xr.open_dataset('F:/ERA5/h500_son_dailyavg.nc')
mslp = xr.open_dataset('F:/ERA5/mslp_son_daily_avg.nc')
u850 = xr.open_dataset('F:/ERA5/u850_son_daily_avg.nc')
v850 = xr.open_dataset('F:/ERA5/v850_son_daily_avg.nc')

#Find the seasonal (overall) mean
h500m = np.around(h500.z.mean('time'),4)
mslpm = np.around(mslp.msl.mean('time'),4)
u850m = np.around(u850.u.mean('time'),4)
v850m = np.around(v850.v.mean('time'),4)

#Subtract the seasonal mean from the data
h500n = np.around(h500.z,4) - h500m
mslpn = np.around(mslp.msl,4) - mslpm
u850n = np.around(np.around(u850.u,4) - u850m,3)
v850n = np.around(v850.v,4) - v850m

#Take the area weight of the data set to make sure all lats and lons are weighted the same
h500aw = (h500n * np.sqrt(np.cos(np.pi * LAT / 180))).squeeze()
mslpaw = (mslpn * np.sqrt(np.cos(np.pi * LAT / 180))).squeeze()
u850aw= (u850n * np.sqrt(np.cos(np.pi * LAT / 180))).squeeze()
v850aw = (v850n * np.sqrt(np.cos(np.pi * LAT / 180))).squeeze()

#Arrange as time x space
#v850u = v850aw.stack(latlon=('latitude','longitude'))
#u850u = u850aw.stack(latlon=('latitude','longitude'))
#h500u = h500aw.stack(latlon=('latitude','longitude'))
#mslpu = mslpaw.stack(latlon=('latitude','longitude'))
F = np.ones((h500aw.shape[0],h500aw.shape[1]*h500aw.shape[2]))
v850u = map2mat(F,v850aw)
u850u = map2mat(F,u850aw)
h500u = map2mat(F,h500aw)
mslpu = map2mat(F,mslpaw)

#Put everything together into one array by combining the columns
#multi = xr.concat((h500u,mslpu,u850u,v850u),dim='latlon')
multi = np.concatenate((h500u,mslpu,u850u,v850u),1)

#Standardize the value at each gridpoint to make sure our clustering isn't influenced by higher values of mslp and h500
nan_mean = np.nanmean(multi,axis=0)
nan_std = np.nanstd(multi,axis=0)
multi_s = (multi - nan_mean) / nan_std

#Replace any nan values with 0
#multi_s = multi_s.fillna(0)
multi_s = np.nan_to_num(multi_s)

#Perform eof to reduce the data size
U,S,V = np.linalg.svd(np.around(multi_s,4))
#Note that S returns as diag(S) and V needs to be transposed (V.T)
s = S ** 2
sc = s/np.sum(s,axis=0)
a= np.nonzero(np.cumsum(sc,axis=0)>retain)[0]
a=a[0]+1
Snew = np.zeros(multi_s.shape)
np.fill_diagonal(Snew,np.around(S,4))
tmpu = np.around(U[:,0:a],4) @ np.around(Snew[0:a,0:a],4)
nr = tmpu.shape[0]
nc = tmpu.shape[1]




#Now do a sanity check to make sure it looks right
g = input('Perform k-means on ' + str(nr) + ' dates, ' + str(nc) + ' variables? (Y/N): ')
if(g == 'N'):
    print('Ending Clustering')
else:
    if os.path.exists(outdir + '/CI_results.mat'):
        #tmp = np.load(outdir + '/CI_results.mat')
        #CI = tmp.CI;
        #K = tmp.K;
        #D = tmp.D;
        pass

    else:
        k_over = np.zeros((3640,10))
        ci_over = np.zeros(10)
        for n in range(1, maxclust+1,1):
            print('k=%d' % n)
            stand = None
            prop = None
            if stand != None:
                X = stan(X, stand)
            X = tmpu
            R = X.shape[0]
            C = X.shape[1]
            nclus = n
            nsim = 100

            if prop != None:
                U, S, V = np.linalg.svd(X)
                # NOTE in Python S is returned as diag(S) and V needs to be transposed (V.T)
                V = V.T
                s = S ** 2
                sc = s / np.sum(s, axis=1)
                a = np.where(np.cumsum(sc, axis=1) > prop)
                a = a[0]+1
                PC = U[:, 0:a] @ diag(S)[0:a, 0:a]
                print('Pre-filtering using EOF retaining the first ' + str(a) + ' components done ...')
            else:
                PC = X
                print('No EOF prefiltering ...')
            #PC[:,0] = PC[:,0] * -1
            #PC[:,-1] = PC[:,-1] * -1
            r = X.shape[0]
            c = X.shape[1]
            Kens = np.ones((PC.shape[0], nclus, nsim))
            K = np.zeros((PC.shape[0], nclus))
            CI = np.zeros(nclus)
            for NC in range(0,1,1):  # NC=1:length(nclus)
                # clear MC mean_cluster mean_cluster2 k ACC ACCmax part
                mean_cluster = np.zeros((nclus, c))
                ACCmax = np.zeros((nclus, nsim))
                k = np.zeros((PC.shape[0], nsim))
                MC = np.zeros(((nclus) * c, nsim))
                print(['K means clustering with ' + str(nclus) + ' clusters begins ...'])
                for i in range(nsim):  # i=1:nsim;
                    k[:, i] = KMeans(n_clusters=nclus, init='k-means++', n_init=1, max_iter=1000).fit_predict(X)  # k(:,i)=kmeans(PC,nclus(NC),'Maxiter',1000,'EmptyAction','singleton');
                    for j in range(nclus):  # j=1:nclus(NC);
                        lj = len(np.nonzero(k[:, i] == j)[0])  # length(find(k(:,i)==j)); Since it is a tuple, use [0] to grab data
                        if(lj<1900 and lj >1000):
                            print(lj)
                        if lj > 1:
                            mean_cluster[j, :] = np.mean(PC[np.nonzero(k[:, i] == j)[0],:], axis=0)  # mean(X(find(k(:,i)==j),:))
                        else:
                            mean_cluster[j, :] = X[k[:, i] == j]  # X(find(k(:,i)==j),:)
                    #mean_cluster2 = stan(mean_cluster.conj().T, opt='s')
                    nan_mean = np.nanmean(mean_cluster.T,axis=0)
                    nan_std = np.nanstd(mean_cluster.T,axis=0,ddof=1)
                    #nan_mean = np.ones((mean_cluster.shape[0],1)) * nan_mean
                   # nan_std = np.ones((mean_cluster.shape[0],1)) * nan_std
                    mean_cluster2 = (mean_cluster.T - nan_mean)/ nan_std
                    #if(mean_cluster2.shape[1] > 1):
                    #    temp = np.copy(mean_cluster2[:,0])
                    #    mean_cluster2[:,0] = mean_cluster2[:,1]
                    #   mean_cluster2[:,1] = temp
                    mean_cluster = np.zeros((nclus, c))
                    MC[:, i] = mean_cluster2.flatten('F')  # centroids stored in MC matrix
                Kens[:, NC-1, :] = k
                for i in range(nclus):  # i=1:nclus(NC);
                    for j in range(nsim):  # j = 1:nsim;
                        sample1 = MC[(i * c):(i + 1) * c, j]  # MC(((i - 1) * c) + 1:i * c, j);
                        a = np.nonzero(j != np.arange(0, nsim))[0]  # find(j~ = [1:nsim]);
                        sample2 = MC[:, a].reshape( c,(nsim - 1) * nclus, order='F').copy()  # reshape(MC(:, a), c, (nsim - 1) * nclus(NC))
                        ind = np.isnan(sample1)
                        sample1[ind] = 0
                        ind = np.isnan(sample2)
                        sample2[ind] = 0
                        ACC = (1 / (c-1)) * sample1.conj().T @ sample2
                        ACC = ACC.reshape(nclus, nsim - 1, order='F').copy()  # (ACC, nclus(NC), nsim - 1);
                        ACCmax[i, j] = np.mean(ACC.max(0))  # mean(max(ACC));  # considering the mean instead the min
                part = np.nonzero(np.mean(ACCmax, axis=0) == np.max(np.mean(ACCmax, axis=0)))[0]  # find(mean(ACCmax) == max(mean(ACCmax)))
                CI[NC] = np.mean(ACCmax)  # Classification index for the partition ton 'nclus' clusters
                if len(part) > 1: part = part[0]
                K[:, NC] = np.squeeze(k[:, part])  # best partition that maximize the ACC with the nsim-1 other partitions
            k_over[:,n-1] = K[:,0]
            ci_over[n-1] = CI[0]



