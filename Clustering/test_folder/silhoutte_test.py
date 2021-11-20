import pandas as pd 
import numpy as np
#import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
plt.style.use('seaborn')


X = np.random.rand(50,2)
Y = 2 + np.random.rand(50,2)
Z = np.concatenate((X,Y))
Z = pd.DataFrame(Z)

print(Z)
plt.scatter(Z[0],Z[1])
plt.show()
"""
KMean = KMeans(n_clusters = 2)
KMean.fit(Z)
label = KMean.predict(Z)

print(label)
print("Kmeans centers: " + str(KMean.cluster_centers_))
"""

#print(f'Silhouette Score(n=2): {silhouette_score(Z, label)}')

for i,k in enumerate([2,3,4,5]):
	fig, ax = plt.subplots(1,2,figsize=(15,5))

	km = KMeans(n_clusters = k)
	y_predict = km.fit_predict(Z)
	centroids = km.cluster_centers_

	silhouette_vals = silhouette_score(Z, y_predict)
	silhouette_vals_ = silhouette_samples(Z, y_predict)
	silhouette_avg = np.mean(silhouette_vals_)
	print(silhouette_vals_)

	print("mean: " + str(silhouette_avg))
	print("score: " + str(silhouette_vals))