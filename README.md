# kmeans_toolbox_python

Adapted from the MATLAB equivalent kmeans toolbox used by Moron et al. Adapted from Michaelangeli et al. 1995

* David * * Coe *
UMass Lowell - 2020 *

# kmeans_ci2
 Contains the kmeans_ci function which uses the calculated kmeans (using scikit-learn) to calculate the Classifiability Index (CI, Michaelangeli et al. 1995) to test how well data is correlated to the clustering (using the Anomaly Correlation Coefficient (ACC)). This function returns two lists: the cluster values (K) for the best clustering and the CI value (one item in the list) for the best clustering. The CI value can be plotted as shown below for each cluster to better understand the data.

![Clustering](/images/only_ci.png)

era5_kmeans_xarray contains an example on how to run the functions. Sample data is provided to load into the functions.

 

ar1rand_func contains the ar1rand function. This takes our data we ran through kmeans and creates a red noise dataset from it. This dataset is then run through kmeans_ci itself using the same clustering approach as our dataset to determine CI values for the 90% confidence interval of the dataset.

The data are then plotted in three graphs. The first shows just the CI value for each cluster value. The second shows the CI value and 90% confidence interval for each cluster value (red line). The third shows the CI value, 90% confidence interval (red line), and 90% confidence interval (grey shading).

Any values that fall above the red line/shaded region are considered significant.
