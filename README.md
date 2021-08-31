# kmeans_toolbox_python

Adapted from the MATLAB equivalent kmeans toolbox used by Moron et al 2010. Adapted from Michaelangeli et al. 1995

*David* *Coe*
*UMass* *Lowell* *-* *2020*

# kmeans_ci2
 Contains the **kmeans_ci** function which uses the calculated kmeans (using scikit-learn) to calculate the **Classifiability Index** (**CI**, Michaelangeli et al. 1995) to test how well data is correlated to the clustering (using the **Anomaly Correlation Coefficient** (**ACC**)). This function returns two lists: the cluster values (**K**) for the best clustering and the **CI** value (one item in the list) for the best clustering. The **CI** value can be plotted as shown below for each cluster to better understand the data.

![Clustering](/images/only_ci.png)


# Red Noise Test

**ar1rand_func** contains the ar1rand function. This takes the data that was run through kmeans and creates a red noise dataset from it. This dataset is then run through **kmeans_ci2** using the same clustering approach as before to determine **CI** values for the **90% confidence interval** of the dataset. The confidence interval is determined by running the red noise data through **kmeans_ci2** 1000 times, sorting the resulting **CI** values, and taking the 50th and 950th **CI** values as our **90% confidence interval**.

![Red Noise Test and Clustering](/images/rednoise_ci.png)

The graph above shows the **CI** value, **90% confidence interval** (red line), and **90% confidence interval** (grey shading). Any values that fall above the red line/shaded region are considered significant. In this case, we see that the **CI** value for the 7 cluster solution lies above the **90% confidence interval**, so we would choose a 7 cluster solution in this case.

**Note:** Multiple values can fall above the **90% confidence interval**. In these cases, it is pertinent to compare the clusters of the possible solutions to each other. 
* Do the composites look similar?
* Compare the values and their assignments between the different solutions. Does one value fall into different clusters depending on the solution?
* If one value falls into a different cluster depending on the solution, does the newly formed cluster it is in make sense, or is it just a derivation of another cluster? If so, the lower cluster solution may provide better answers.

# Examples
* **era5_kmeans_xarray** contains an example on how to run the functions. Sample data is provided to load into the functions. 

* **kmeans_example** provides an example using randomly generated data. This is in jupyter notebook form for better accessibility.

* **kmeans_era5** provides an example using the **testdata.xlsx** file to generate a kmeans study using sample climate data. It is similar to **kmeans_example**, but diverges when real data is used in the CI method (whereas **kmeans_example** performs the CI method with the randomly generated data).

# References

* Coe, D., Barlow, M., Agel, L., Colby, F., Skinner, C., & Qian, J., (2021): Clustering Analysis of Autumn Weather Regimes in the Northeast U.S., Journal of Climate, 34,18, 7587-7605.

* Michelangeli, P.A., R. Vautard, and B. Legras, 1995: Weather regimes: recurrence and quasi-stationarity. J. Atmos. Sci., 52, 1237-1256. 

* Moron, V., A. W. Robertson, M. N. Ward, and O. Ndiaye, 2008: Weather types and rainautumn rainfall over Senegal. Part I: Observational analysis. J. Climate, 21, 266–287, doi:10.1175/2007JCLI1601.1 

* Roller, C.D., J.-H. Qian, L. Agel, M. Barlow, and V. Moron, 2016: Winter weather regimes in the Northeast United States. J. Climate, 29, 2963-2980.  

