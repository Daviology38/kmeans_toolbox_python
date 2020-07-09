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
from sklearn.cluster import KMeans
from kmeans_ci2 import kmeans_ci
import pandas as pd
import matplotlib.pyplot as plt
from ar1rand_func import ar1rand

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
nan_std = np.nanstd(multi,axis=0,ddof=1)
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
    if os.path.exists(outdir + '/CI_results.nc'):
        tmp = xr.open_dataset(outdir + '/CI_results.nc')
        CI = tmp.CI
        #K = tmp.K;
        #D = tmp.D;
    else:
        k_over = np.zeros((3640,10))
        ci_over = np.zeros(10)
        for n in range(1, maxclust+1,1):
            print('k=%d' % n)
            k, ci = kmeans_ci(tmpu,None,None,None,n,100)
            k_over[:,n-1] = np.squeeze(k)
            ci_over[n-1] = ci
        #Now save to a netcdf file
        k_df = pd.DataFrame(k_over,columns=['k=1','k=2','k=3','k=4','k=5','k=6','k=7','k=8','k=9','k=10'])
        ci_df = pd.DataFrame(ci_over,columns=['CI'])
        k_df['CI'] =ci_df
        k_xr = k_df.to_xarray()
        k_xr.to_netcdf(outdir + '/CI_results.nc')
        CI = ci_df
    CI = CI.to_dataframe()
    CI.index = np.arange(1,len(CI)+1)
    ax = CI.plot()
    ax.set_ylabel('CI')
    ax.set_xlabel('Cluster')
    ax.set_title('CI Values')
    ax.set_xlim(1,maxclust)
    plt.savefig(outdir + '/plot_ci.png')

    '''
        Next we will create nsim rancom noise samples. We only perform this process once for the given set of clusters
            and use the results for all cluster sizes
        This is made from the standardized anomalies
    '''
    if os.path.exists(outdir + '/random.nc'):
        tmp2 = xr.open_dataset(outdir + '/random.nc')
        rsims = tmp2.rsims.data
    else:
        print('Creating random red noise series')
        rsims = np.zeros((nr,nc,nsim))
        #We will reduce nc through EOF
        for i in range(nc):
            p = ar1rand(tmpu[:,i],nsim)
            rsims[:,i,:] = p
        #Now save the data to a file called random.nc for future use if necessary to rerun
        rsims_xr = xr.DataArray(rsims, dims=('days','variables','nsims'), name='rsims')
        rsims_xr.to_netcdf(outdir + '/random.nc')
        #
    # #Now we will run kmeans with the red noise data for each cluster nsim times
    CIcis = np.zeros((maxclust,nsim))
    CIci = np.zeros((1,nsim))
    citop = np.zeros((maxclust,1))
    cibot = np.zeros((maxclust,1))
    for i in range(1,maxclust+1,1):
        #Check if our individual file exists
        if os.path.exists(outdir + '/CIci_'+str(i)+'.nc'):
            pass
        else:
            for j in range(nsim):
                print('Cluster '+str(i) + ' Simulation ' + str(j))
                sim = np.squeeze(rsims[:,:, j])
                k, CIci[0, j] = kmeans_ci(sim, None, None, None, i, 100)
            cici_xr = xr.DataArray(CIci, dims=('CI','nsim'), name='CIci')
            cici_xr.to_netcdf(outdir + '/CIci_' + str(i) + '.nc')
        tmp = xr.open_dataset(outdir + '/CIci_' + str(i) + '.nc')
        CIcis[i-1,:]=tmp.CIci
        cisort = np.sort(CIcis[i-1,:])
        citop[i-1,0] = cisort[int(.90 * nsim)] # one - sided 90 % confidence interval
        cibot[i-1,0] = cisort[0]
    ax = CI.plot()
    ax.set_ylabel('CI')
    ax.set_xlabel('Cluster')
    ax.set_title('Classifiability Index')
    ax.set_xlim(1, maxclust)
    for a in range(maxclust):
        ax.plot((a+1,a+1),(citop[a,0],cibot[a,0]), 'red')
    plt.savefig(outdir + '/plot_CIci.png')
    #Now fill in the area
    ax = CI.plot()
    ax.set_ylabel('CI')
    ax.set_xlabel('Cluster')
    ax.set_title('Classifiability Index')
    ax.set_xlim(1, maxclust)
    for a in range(maxclust):
        ax.plot((a+1,a+1),(citop[a,0],cibot[a,0]), 'red')
    x = np.arange(1,11,1)
    ax.fill_between(x,np.squeeze(citop),np.squeeze(cibot),color='silver')
    plt.savefig(outdir + '/plot_CIci_shaded.png')










