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
import dask.array as da

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
retain = 0.90
maxclust = 10
nsim = 100 #for Monte Carlo approximation
random_generator = np.random.RandomState()  # stream=RandStream('mrg32k3a')

#Get our LAT/LON grid
lats = Dataset("H:/ERA5/h500_son/h5001979.nc").variables['latitude'][160:241]
lons = Dataset("H:/ERA5/h500_son/h5001979.nc").variables['longitude'][1080:1201]
LON, LAT = np.meshgrid(lons,lats)

#Load in the data using xarray chunks
h500 = xr.open_dataset('H:/ERA5/h500_son_dailyavg.nc')
mslp = xr.open_dataset('H:/ERA5/mslp_son_daily_avg.nc')
u850 = xr.open_dataset('H:/ERA5/u850_son_daily_avg.nc')
v850 = xr.open_dataset('H:/ERA5/v850_son_daily_avg.nc')


#Take the area weight of the data set to make sure all lats and lons are weighted the same
h500aw = (h500.z * np.sqrt(np.cos(np.pi * LAT / 180))).squeeze()
mslpaw = (mslp.msl * np.sqrt(np.cos(np.pi * LAT / 180))).squeeze()
u850aw= (u850.u * np.sqrt(np.cos(np.pi * LAT / 180))).squeeze()
v850aw = (v850.v * np.sqrt(np.cos(np.pi * LAT / 180))).squeeze()

#Arrange as time x space
v850u = v850aw.stack(latlon=('latitude','longitude'))
u850u = u850aw.stack(latlon=('latitude','longitude'))
h500u = h500aw.stack(latlon=('latitude','longitude'))
mslpu = mslpaw.stack(latlon=('latitude','longitude'))

#Put everything together into one array by combining the columns
multi = xr.concat((v850u,u850u,h500u,mslpu),dim='latlon')

#Standardize the value at each gridpoint to make sure our clustering isn't influenced by higher values of mslp and h500
nan_mean = multi.mean(dim='time',skipna=1)
nan_std = multi.std(dim='time',skipna=1)
multi_s = (multi - nan_mean) / nan_std

#Replace any nan values with 0
multi_s = multi_s.fillna(0)

#Turn into xarray of dask arrays
multi_s = multi_s.chunk((56,54))

#Perform eof to reduce the data size
U,S,V = np.linalg.svd(multi_s)
s = np.diag(S.compute()) ** 2
sc = s/np.sum(s,axis=0)
a= np.nonzero(np.cumsum(sc,axis=0)>retain)[0]
a=a[-1]
tmpu = U.compute()[:,0:a] @ S.compute()[0:a,0:a]
nr = tmpu.shape[0]
nc = tmpu.shape[1]

#Now do a sanity check to make sure it looks right
#g = input('Perform k-means on ' + str(nr) + ' dates, ' + str(nc) + ' variables? (Y/N): ')
#if(g == 'N'):
#    print('Ending Clustering')
#else:
#    pass
