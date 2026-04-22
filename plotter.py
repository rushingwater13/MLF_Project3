import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plotter2d(data, data_name):

    # Load dataset, dropping NaNs to handle gaps
    df = pd.read_csv(f"Data/{data}", header = None).dropna()

    X = df.iloc[:,0]
    Y = df.iloc[:,1]

    plt.figure()
    plt.scatter(X, Y)
    plt.title(f"{data_name}")
    plt.savefig(f"Plots/{data_name}.png")



def plotter3d(data, data_name):

    # Load dataset, dropping NaNs to handle gaps
    df = pd.read_csv(f"Data/{data}", header = None).dropna()

    X = df.iloc[:,0]
    Y = df.iloc[:,1]
    Z = df.iloc[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z, alpha=0.6)

    plt.title(f"{data_name}")
    plt.savefig(f"Plots/{data_name}.png")
    plt.clf()



data2d = [
    ("Dataset 1.csv", "Data1"), ("Dataset 2.csv", "Data2"), ("Dataset 3.csv", "Data3"), 
    ("Dataset 4.csv", "Data4"), ("Dataset 5.csv", "Data5"),
]

data3d = [
    ("Dataset 6.csv", "Data6"), ("Dataset 7.csv", "Data7"),
]


for data, data_name in data2d:
    plotter2d(data, data_name)

for data, data_name in data3d:
    plotter3d(data, data_name)