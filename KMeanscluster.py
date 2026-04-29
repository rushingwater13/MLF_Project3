import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator


def cluster(data, data_name):

     # Load dataset, dropping NaNs to handle gaps
    df = pd.read_csv(f"Data/{data}", header = None).dropna()
    [n, p] = np.shape(df)

    # Hyperparameters
    k_max = 10
    q = 10
    max_iter = 500


    # Record the SSE for several values of k
    sse = []
    for k in range(1, k_max + 1):

        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        kmeans = KMeans(
            init="random", n_clusters=k, 
            n_init=q, max_iter=max_iter, 
            random_state=42
        )
        kmeans.fit(df) 

        sse.append(kmeans.inertia_)



    # Find the elbow of the SSE curve

    plt.figure()
    plt.plot(range(1, k_max + 1), sse)
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.title(data_name)
    plt.savefig(f"SSE/{data_name}_SSE.png")

    # https://pypi.org/project/kneed/
    # Satopa, V., Albrecht, J., Irwin, D., and Raghavan, B. (2011). "Finding a 'Kneedle' 
    # in a Haystack: Detecting Knee Points in System Behavior." 
    # 31st International Conference on Distributed Computing Systems Workshops, pp. 166-171.
    
    kl = KneeLocator(range(1, k_max + 1), sse, curve="convex", direction="decreasing")
    elbow = kl.elbow
    kmeans_txt = open("SSE/"+data_name + "_k-means.txt","w")
    kmeans_txt.write(f"{'SSE':<12}{'K'}\n")
    kmeans_txt.write(f"{round(sse[elbow-2],3):<12}{elbow-1}\n")
    kmeans_txt.write(f"{round(sse[elbow-1],3):<12}{elbow} (optimal k)\n")
    kmeans_txt.write(f"{round(sse[elbow],3):<12}{elbow+1}\n")

    return elbow



def plotter2d(data, data_name, k_best):

    # Load dataset, dropping NaNs to handle gaps
    df = pd.read_csv(f"Data/{data}", header = None).dropna()

    df_dupe = df.iloc[:, :2]
    X = df.iloc[:,0]
    Y = df.iloc[:,1]

    # Hyperparameters
    q = 10
    max_iter = 500

    # Make a plot for each of k-1, k, and k+1
    for i in range(k_best-1, k_best+2):
    
        # Get the labels for the data
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        kmeans = KMeans(
                init="random", n_clusters=i, 
                n_init=q, max_iter=max_iter, 
                random_state=42
            )
        kmeans.fit(df_dupe) 
        df["cluster"] = kmeans.labels_
        for k in range(i):
            df[df["cluster"] == k].to_csv("Labeled_Data/" +str(data_name)+"_clusters/k=" + str(i)+ "_cluster" +str(k)+".csv") 


        labels = kmeans.labels_
    

        plt.figure()

        # Divide and add the data according to the labels
        for j in range(i):
            x = X[labels == j]
            y = Y[labels == j]
            plt.scatter(x, y)

        plt.title(data_name)
        plt.savefig(f"Plots/{data_name}/{data_name}_{i}_Clusters.png")
        plt.close()



def plotter3d(data, data_name, k_best):

    # Load dataset, dropping NaNs to handle gaps
    df = pd.read_csv(f"Data/{data}", header = None).dropna()

    df_dupe = df.iloc[:, :3]
    X = df.iloc[:,0]
    Y = df.iloc[:,1]
    Z = df.iloc[:,2]

    # Hyperparameters
    q = 10
    max_iter = 500


    # Make a plot for each of k-1, k, and k+1
    for i in range(k_best-1, k_best+2):
    
        # Get the labels for the data
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        kmeans = KMeans(
                init="random", n_clusters=i, 
                n_init=q, max_iter=max_iter, 
                random_state=42
            )
        kmeans.fit(df_dupe) 
        df["cluster"] = kmeans.labels_

        for k in range(i):
            df[df["cluster"] == k].to_csv("Labeled_Data/" +str(data_name)+"_clusters/k=" + str(i)+ "_cluster" +str(k)+".csv") 
        labels = kmeans.labels_

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Divide and add the data according to the labels
        for j in range(i):
            x = X[labels == j]
            y = Y[labels == j]
            z = Z[labels == j]
            ax.scatter(x, y, z, alpha=0.6)

        plt.title(f"{data_name}_{i}")
        plt.savefig(f"Plots/{data_name}/{data_name}_{i}_Clusters.png")
        plt.clf()



data2d = [
    ("Dataset 1.csv", "Data1"), ("Dataset 2.csv", "Data2"), ("Dataset 3.csv", "Data3"), 
    ("Dataset 4.csv", "Data4"), ("Dataset 5.csv", "Data5"),
]

data3d = [
    ("Dataset 6.csv", "Data6"), ("Dataset 7.csv", "Data7"),
]


for data, data_name in data2d:
    k_best = cluster(data, data_name)
    plotter2d(data, data_name, k_best)

for data, data_name in data3d:
    k_best = cluster(data, data_name)
    plotter3d(data, data_name, k_best)

