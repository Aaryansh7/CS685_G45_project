'''
This program performs dbscan on inclination and RAAN values and clusters the objects based on it.
The dbscan clusters are perfomed for each group based on sma.
'''

# Python libraries to import
import numpy as np
import math
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from tqdm import tqdm


# Importing the dataset
dataset = pd.read_csv('../data_files/kmeans+dbscan3.csv')
'''
The column 'Label' contains the group number based on clustering on sma(sem-major axis).
The numbers can be interpreted as:-
1)number before decimal corresponds to the main-group number, i-e(1,2 or 3):
	1-> group with sma within 2kms.
	2-> group with sma within 10kms
	3-> group with sma within 25 kms
2)number after decimal is the subgroup number within the main-group
'''

dataset = dataset.sort_values(by=['Label'])                 
X = dataset.iloc[:, [4, 8, 9]].values

i = 0
sma_grouped_last_index = 0
dbscan_labels = []
while i < X.shape[0]:
	label = X[i,0]
	if math.isnan(label)==True:
		dbscan_labels.append(None)
		i+=1
		continue

	if math.isnan(label) == False:
		j = i
		while X[j,0] == label:
			j += 1
		sma_grouped_last_index = j - 1
		group_index = (i,sma_grouped_last_index)

		Z = dataset.iloc[group_index[0]:group_index[1]+1, [8, 9]].values
		dbscan=DBSCAN(eps=2,min_samples=2)
		model=dbscan.fit(Z)
		labels_ =model.labels_
		labels__ = labels_.tolist()
		for k in range(len(labels__)):
			dbscan_labels.append(labels__[k])


	i = sma_grouped_last_index + 1

dataset.insert(5, "dbscan_labels" , dbscan_labels, True)
dataset.drop(dataset.columns[0], axis=1, inplace=True)
dataset.reset_index(inplace = True)
dataset.drop(dataset.columns[0], axis=1, inplace=True)
dataset.to_csv('../data_files/dbscan_raan_inc_entire_data1.csv')
