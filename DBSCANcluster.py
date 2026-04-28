import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import os

def plotter2d(data, data_name, eps_val, min_samples_val):

    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

    # Load dataset, dropping NaNs to handle gaps
    df = pd.read_csv(f"Data/{data}", header = None).dropna()

    X = df.iloc[:,0]
    Y = df.iloc[:,1]

    # Get the labels for the data
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    dbscan.fit(df) 
    
    df["cluster"] = dbscan.labels_
    labels = dbscan.labels_
    
    unique_labels = set(labels)
    
    # Create the subfolder
    os.makedirs("Labeled_Data/" + str(data_name) + "_clusters/", exist_ok=True)
    
    for k in unique_labels:
        df[df["cluster"] == k].to_csv("Labeled_Data/" + str(data_name) + "_clusters/dbscan_cluster_" + str(k) + ".csv") 

    plt.figure()

    # Divide and add the data according to the labels
    for j in unique_labels:
        x = X[labels == j]
        y = Y[labels == j]
        plt.scatter(x, y)

    plt.title(data_name + " - DBSCAN")
    plt.savefig(f"Plots/{data_name}_dbscan_Clusters.png")
    plt.close()

def plotter3d(data, data_name, eps_val, min_samples_val):

    # Load dataset, dropping NaNs to handle gaps
    df = pd.read_csv(f"Data/{data}", header = None).dropna()

    X = df.iloc[:,0]
    Y = df.iloc[:,1]
    Z = df.iloc[:,2]

    # Get the labels for the data
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    dbscan.fit(df) 
    
    df["cluster"] = dbscan.labels_
    labels = dbscan.labels_

    unique_labels = set(labels)
    
    # Create the subfolder
    os.makedirs("Labeled_Data/" + str(data_name) + "_clusters/", exist_ok=True)

    for k in unique_labels:
        df[df["cluster"] == k].to_csv("Labeled_Data/" + str(data_name) + "_clusters/dbscan_cluster_" + str(k) + ".csv") 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Divide and add the data according to the labels
    for j in unique_labels:
        x = X[labels == j]
        y = Y[labels == j]
        z = Z[labels == j]
        ax.scatter(x, y, z, alpha=0.6)

    plt.title(data_name + " - DBSCAN")
    plt.savefig(f"Plots/{data_name}_dbscan_Clusters.png")
    plt.clf()

dataset_params = {
    "Dataset 1.csv": {"name": "Data1", "eps": 1.0, "min_samples": 5, "type": "2D"},
    "Dataset 2.csv": {"name": "Data2", "eps": 0.8, "min_samples": 5, "type": "2D"},
    "Dataset 3.csv": {"name": "Data3", "eps": 0.15, "min_samples": 5, "type": "2D"}, # Lowered!
    "Dataset 4.csv": {"name": "Data4", "eps": 0.11, "min_samples": 5, "type": "2D"}, # Lowered!
    "Dataset 5.csv": {"name": "Data5", "eps": 1.0, "min_samples": 7, "type": "2D"},
    
    "Dataset 6.csv": {"name": "Data6", "eps": 1.5, "min_samples": 10, "type": "3D"}, # Lowered!
    "Dataset 7.csv": {"name": "Data7", "eps": 1.5, "min_samples": 10, "type": "3D"}  # Lowered!
}

# Run the master loop
for filename, params in dataset_params.items():
    data_name = params["name"]
    eps_val = params["eps"]
    min_pts = params["min_samples"]
    
    print(f"Running {data_name} with eps={eps_val}...")
    
    if params["type"] == "2D":
        plotter2d(filename, data_name, eps_val=eps_val, min_samples_val=min_pts)
    elif params["type"] == "3D":
        plotter3d(filename, data_name, eps_val=eps_val, min_samples_val=min_pts)