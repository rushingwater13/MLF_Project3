This folder holds the data for each dataset divided and labeled according the their assigned clusters. 
Each dataset has its own folder, inside of which are several .csv files. 
The files named "k=#_cluster#.csv" correspond to the k-means clusters, where the first # is the total number 
of clusters and the second # is which of those clusters it contains.
The files names "dbscan_cluster_#.csv" correspond to the DBSCAN clusters. 
For the DBSCAN clusters, the "-1" cluster holds the data points classified as noise.
