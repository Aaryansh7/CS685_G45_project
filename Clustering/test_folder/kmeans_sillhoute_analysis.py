
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

np.set_printoptions(precision=2)
plt.style.use('seaborn')

dataset = pd.read_csv('catalogue.csv')
start_point = 19106
final_point = 19165
X = dataset.iloc[start_point:final_point, [4]].values
X_ = dataset.iloc[start_point:final_point, [1, 4, 8, 9]].values
#print(X)

##################################################################################################

#CLUSTERING ANALYSIS
fig2, (ax2) = plt.subplots(nrows=1, ncols=1)
colors=['red', 'blue', 'green', 'yellow', 'pink', 'orange', 'black', 'purple', 'brown', 'grey', 'skyblue', 'lightgreen']

n= len(X)
Y = np.ones(n)
Z = list(zip(X,Y))
Z = pd.DataFrame(Z)

#plt.scatter(Y,X)
#plt.show()

min_sillhoute_coeff = []
silhouette_vals_ = []
sum_squared = []
#range_ = list(range(1,401))
#print(range_)
for i,k in tqdm(enumerate(list(range(2,300)))):
	#fig, ax = plt.subplots(1,2,figsize=(15,5))

	km = KMeans(n_clusters = k)
	y_predict = km.fit_predict(Z)
	centroids = km.cluster_centers_

	silhouette_vals = silhouette_samples(Z, y_predict)
	#print(silhouette_vals)
	silhouette_mean = silhouette_score(Z, y_predict)
	minimum_val = np.amin(silhouette_vals)

	min_sillhoute_coeff.append(minimum_val)
	silhouette_vals_.append(silhouette_mean)
	sum_squared.append(km.inertia_ /len(silhouette_vals))
	if km.inertia_/len(silhouette_vals) < 1.6E6:
		break

weights_min_val = 0.2
weights_mean_val = 0.5
weights_ssumd = 7.5e+5

max_val = float("-inf")
optimal_n_cluster = 0
for i in range(len(silhouette_vals_)):
	value  = silhouette_vals_[i]*weights_mean_val + min_sillhoute_coeff[i]*weights_min_val + weights_ssumd/sum_squared[i]
	#print("value: " + str(value))
	if value > max_val:
		max_val = value
		optimal_n_cluster = i +2 

print("max value is: " + str(max_val))
print("optimal_n_cluster: " + str(optimal_n_cluster))
print("silhouette_mean: " + str(silhouette_vals_[optimal_n_cluster - 2]))
print("silhouette_min: " + str(min_sillhoute_coeff[optimal_n_cluster - 2]))

print(" ")
print(" ")

#print("silhoutte_min values: " + str(min_sillhoute_coeff))
#print("silhoutte_mean values: " + str(silhouette_vals_))

km = KMeans(n_clusters = optimal_n_cluster)
y_predict = km.fit_predict(Z)
plt.scatter(Z.iloc[:, 0], Z.iloc[:, 1], c=y_predict, s=50, cmap='viridis')
#centers = km.cluster_centers_
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()

metric = np.zeros((optimal_n_cluster, 3))

metric[:,0] = float("inf")
metric[:,1] = float("-inf")
for i in range(final_point-start_point):
	index = y_predict[i]
	a = Z.iloc[i,0]
	diff = a - km.cluster_centers_[index][0]
	if diff[0] < metric[index][0]:
		metric[index][0] = diff[0]
	if diff[0] > metric[index][1]:
		metric[index][1] = diff[0]
	metric[index][2] = metric[index][1] - metric[index][0] 

list_tuples = list(zip(metric[:,2]))

X_ = pd.DataFrame(X_)
X_.insert(1, "LABEL NO" , km.labels_, True)
X_.columns =['Name', 'Label', 'Semi-major axis(m)', 'Inclination', 'Raan'] 
print(X_)
X_.to_csv('kmeans_data.csv')

##########################################################################################
def function(data, optimal_n_cluster):

	km = KMeans(n_clusters = optimal_n_cluster)
	y_predict = km.fit_predict(data)
	centroids = km.cluster_centers_
	#print(" Centroids: " + str(centroids))

	silhouette_vals = silhouette_samples(data, y_predict)

	#values_wrt_centroid = silhouette_vals - centroids

	metric = np.zeros((optimal_n_cluster, 2))
	metric[:,0] = float("inf")
	metric[:,1] = float("-inf")

	for i in range(len(silhouette_vals)):
		index = y_predict[i]
		print(data.iloc[i, 4])
		value = data.iloc[i, 2] - centroids[index][0]
		print(value)

		if value < metric[index][0]:
			metric[index][0] = value
		if value > metric[index][1]:
			metric[index][1] = value

	#print("metric: " + str(metric))

#function(X_, 2)



