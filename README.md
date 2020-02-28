# kmeans_toolbox_python

Adapted from the MATLAB equivalent kmeans toolbox used by Moron et al. Adapted from Michaelangeli et al. 1995

Code ported by David Coe
UMass Lowell - 2020

era5_kmeans_xarray contains an example on how to run the functions. Sample data is provided to load into the functions.

kmeans_ci2 contains the kmeans_ci function which is used to calculate kmeans (using scikit-learn). From there, it then uses the output to calculate the Classifiability Index (CI, Michaelangeli et al. 1995) to test how well our data is correlated to the clustering (using the Anomaly Correlation Coefficient (ACC))

ar1rand_func contains the ar1rand function. This takes our data we ran through kmeans and creates a red noise dataset from it. This dataset is then run through kmeans_ci itself using the same clustering approach as our dataset to determine CI values for the 90% confidence interval of the dataset.

The data are then plotted in three graphs. The first shows just the CI value for each cluster value. The second shows the CI value and 90% confidence interval for each cluster value (red line). The third shows the CI value, 90% confidence interval (red line), and 90% confidence interval (grey shading).

Any values that fall above the red line/shaded region are considered significant.
