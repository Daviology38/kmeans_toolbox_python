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
from stan import stan,nan_std,nan_mean

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

#Initialize our arrays to hold the variables
h500 = np.zeros((3640,81,121))
mslp = np.zeros((3640,81,121))
u850 = np.zeros((3640,81,121))
v850 = np.zeros((3640,81,121))
h500m = np.zeros((81,121))
mslpm = np.zeros((81,121))
u850m = np.zeros((81,121))
v850m = np.zeros((81,121))

#Initialize our counting variable
count = 0

#Fill our arrays with data
for i in range(1979,2019):
    #put together strings for the filenames based on the year
    start = 'H:/ERA5/h500_son/h500'
    start2 = 'H:/ERA5/u850_son/u850'
    start3 = 'H:/ERA5/v850_son/v850'
    start4 = 'H:/ERA5/mslp_son/mslp'
    next = '.nc'
    filename = start + str(i) + next
    filename2 = start2 + str(i) + next
    filename3 = start3 + str(i) + next
    filename4 = start4 + str(i) + next

    #now we fill them. Take 12z data for each day for each year
    j = 12
    while j <= 2184:
        #Put each data array into a temp array
        tmp = Dataset(filename).variables['z'][j,160:241,1080:1201]
        tmph = tmp
        tmp = Dataset(filename2).variables['u'][j,160:241,1080:1201]
        tmpu = tmp
        tmp = Dataset(filename3).variables['v'][j,160:241,1080:1201]
        tmpv = tmp
        tmp = Dataset(filename4).variables['msl'][j,160:241,1080:1201]
        tmpm = tmp
        for a in range(121):
            for b in range(81):
                #Fill the arrays with the data from the temp array
                #Fill the mean arrays as well with the sum of the data
                h500[count,b,a] = tmph[b,a] / 9.81
                mslp[count,b,a] = tmpm[b,a]
                u850[count,b,a] = tmpu[b,a]
                v850[count,b,a] = tmpv[b,a]
                h500m[b,a] = h500m[b,a] + tmph[b,a] / 9.81
                mslpm[b,a] = mslpm[b,a] + tmpm[b,a]
                u850m[b,a] = u850m[b,a] + tmpu[b,a]
                v850m[b,a] = v850m[b,a] + tmpv[b,a]
        count = count + 1
        j = j + 24

#Now take the mean arrays and divide by the count to get the seasonal mean

h500m = h500m / count
mslpm = mslpm / count
u850m = u850m / count
v850m = v850m / count

#Now subtract the seasonal mean from our data
for i in range(count):
    for a in range(81):
        for b in range(121):
            h500[i,a,b] = h500[i,a,b] - h500m[a,b]
            mslp[i,a,b] = mslp[i,a,b] - mslpm[a,b]
            u850[i,a,b] = u850[i,a,b] - u850m[a,b]
            v850[i,a,b] = v850[i,a,b] - v850m[a,b]

#Now we need to take the area weight of the data

#Initialize new arrays for the area weight
h500aw = np.zeros((count,81,121))
mslpaw = np.zeros((count,81,121))
u850aw = np.zeros((count,81,121))
v850aw = np.zeros((count,81,121))

#Now take the area weight by multiplying each day by a weight of the lat/lon frid
#This enables us to ensure the data is not influence based on latitude/longitude factors
for l in range(count):
    h500aw[l,:,:] = (h500[l,:,:] * np.sqrt(np.cos(np.pi * LAT/180))).squeeze()
    mslpaw[l, :, :] = (mslp[l, :, :] * np.sqrt(np.cos(np.pi * LAT / 180))).squeeze()
    u850aw[l, :, :] = (u850[l, :, :] * np.sqrt(np.cos(np.pi * LAT / 180))).squeeze()
    v850aw[l, :, :] = (v850[l, :, :] * np.sqrt(np.cos(np.pi * LAT / 180))).squeeze()

#Delete the unnecessary arrays
del h500, mslp, u850, v850, h500m, mslpm, u850m, v850m


#Arrange as time x space
F = np.ones(np.shape(LON))
v850u = map2mat(F,v850aw)
u850u = map2mat(F,u850aw)
h500u = map2mat(F,h500aw)
mslpu = map2mat(F,mslpaw)


#Put everything together into one array by combining the columns
multi = np.concatenate((v850u,u850u,h500u,mslpu),1)

#Delete the unnecessary arrays
del h500u, mslpu, u850u, v850u, h500aw, mslpaw, u850aw, v850aw

#Standardize the value at each gridpoint to make sure our clustering isn't influenced by higher values of mslp and h500
multi_s = stan(multi, 's')

#Replace any nan values with 0
multi_s[np.isnan(multi_s)] = 0

#Also replace this value (Not sure the reason why, but just do it)
multi_s[multi_s==-32767] = 0

#Perform eof to reduce the data size
U,S,V = np.linalg.svd(multi_s)
s = np.diag(S) ** 2
sc = s/sum(s,axis=0)
a= np.nonzero(np.cumsum(sc,axis=0)>retain)[0]
a=a[0]
tmpu = U[:,0:a] @ S[0:a,0:a]
nr = tmpu.shape[0]
nc = tmpu.shape[1]

#Now do a sanity check to make sure it looks right
g = input('Perform k-means on ' + str(nr) + ' dates, ' + str(nc) + ' variables? (Y/N): ')
if(g == 'N'):
    print('Ending Clustering')
else:
    pass


