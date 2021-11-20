'''
This is a test file to check dbsan on the data.
'''

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from tqdm import tqdm


# Importing the dataset
dataset = pd.read_csv('catalogue.csv')
X = dataset.iloc[:, [4, 8, 9]].values

print(X)

# Using the elbow method to find the optimal number of clusters
dbscan=DBSCAN(eps=60,min_samples=3)

# Fitting the model

model=dbscan.fit(X)
labels=model.labels_

#identifying the points which makes up our core points
sample_cores=np.zeros_like(labels,dtype=bool)

sample_cores[dbscan.core_sample_indices_]=True

#Calculating the number of clusters

n_clusters=len(set(labels))- (1 if -1 in labels else 0)
print("number of clusters"  + str(n_clusters))


labels = labels.tolist()
#print(type(labels))
#print(labels)
dataset.insert(2, "LABEL NO" , labels, True) 
#print(dataset)

print(metrics.silhouette_score(X,labels))
dataset.to_csv('dbscan_data.csv')
