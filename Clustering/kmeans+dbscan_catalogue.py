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

dataset = pd.read_csv('../data_files/filtered_catalogue.csv')
start_point = 1
final_point = 7771
X = dataset.iloc[start_point:final_point, [5]].values
X_ = dataset.iloc[start_point:final_point, [2, 5, 9, 10]].values
#print(X)

##################################################################################################
#CLUSTERING ANALYSIS(kmeans)

def squared_sum_group(data, y_predict, centroids):
	#print(data)
	#print(centroids)
	#print(y_predict)
	d = dict()
	for i in range(len(centroids)):
		d[i] = []

	ssd = np.zeros(len(centroids))
	cnt_group = np.zeros(len(centroids))
	ssd_avg = np.zeros(len(centroids))
	sorted_per_group = np.zeros(())

	for i in range(len(y_predict)):
		label = y_predict[i]
		distance_square = (data[i] - centroids[label])**2
		#print(distance_square)
		ssd[label] += distance_square
		cnt_group[label] += 1
		d[label].append(i)

	column_sums = ssd.sum()
	#print("column_sums: " + str(column_sums))

	for i in range(len(ssd_avg)):
		ssd_avg[i] = ssd[i]/ cnt_group[i]

	#print(d)
	return ssd_avg, cnt_group, d


n= len(X)
Y = np.ones(n)
Z = list(zip(X,Y))
Z = pd.DataFrame(Z)

#plt.scatter(Y,X)
#plt.show()

min_sillhoute_coeff = []
silhouette_vals_ = []
sum_squared = []
avg_range = []


threshold_array = [4E6, 1E8, 6.25E8]
max_threshold = threshold_array[-1]
min_threshold = threshold_array[0]
num_cluster = 0 
iterations = int(0.66*(final_point - start_point))
iter_count = 0
main_data = pd.DataFrame()
lable_cnt = 0
while min_threshold <= max_threshold:
	print("min_threshold: " + str(min_threshold))
	print("iteration number: " + str(iter_count))
	outlier_data = pd.DataFrame()

	grp_numbers_tb_added = []
	grp_numbers_tb_not_added = []
	map_grps_ = dict()
	y_predict_ = []
	labels_ = []

	for i,k in tqdm(enumerate(list(range(2,iterations)))):
		#print("cluster-num: " + str(k))
		km = KMeans(n_clusters = k)
		y_predict = km.fit_predict(Z)
		centroids = km.cluster_centers_

		#print("labels: " + str(km.labels_))
		#silhouette_vals = silhouette_samples(Z, y_predict)
		#silhouette_mean = silhouette_score(Z, y_predict)
		#minimum_val = np.amin(silhouette_vals)

		#min_sillhoute_coeff.append(minimum_val)
		#silhouette_vals_.append(silhouette_mean)
		#sum_squared.append(km.inertia_ /len(silhouette_vals_))

		avg_ssumd, count_per_grp, map_grps = squared_sum_group(Z.iloc[:, 0], y_predict, centroids[:, 0])
	
		#print(avg_ssumd)
		#print(count_per_grp)
		
		'''
		metric = np.zeros((k, 3))
		metric[:,0] = float("inf")
		metric[:,1] = float("-inf")
		for i in range(Z.shape[0]):
			index = y_predict[i]
			diff = Z.iloc[i, 0] - km.cluster_centers_[index]
			if diff[0] < metric[index][0]:
				metric[index][0] = diff[0]
			if diff[0] > metric[index][1]:
				metric[index][1] = diff[0]
			metric[index][2] = metric[index][1] - metric[index][0] 

		column_sums = metric.sum(axis=0)
		sum_range = column_sums[2]
		avg_range.append(sum_range/k)
		'''
		#if km.inertia_/len(silhouette_vals) < min_threshold :
			#break
		
		map_grps_ = map_grps
		flag1 = 0
		for j in range(len(avg_ssumd)):
			if avg_ssumd[j]> min_threshold:
				flag1 = 1
				continue

		if flag1 == 0:
			for j in range(len(count_per_grp)):
				if count_per_grp[j] >= 2:
					grp_numbers_tb_added.append(j)
				if count_per_grp[j]<2:
					grp_numbers_tb_not_added.append(j)
			map_grps_ = map_grps
			y_predict_ = y_predict
			labels_ = km.labels_
			break

	weights_min_val = 4.0
	weights_mean_val = 0.6
	weights_ssumd = 1.7e+6
	weights_avg_range = 0
	'''
	max_val = float("-inf")
	optimal_n_cluster = 0
	for i in range(len(silhouette_vals_)):
		#print("avg_range for cluster: " + str(avg_range[i]))
		value  = silhouette_vals_[i]*weights_mean_val + min_sillhoute_coeff[i]*weights_min_val + weights_ssumd/sum_squared[i] + weights_avg_range/avg_range[i]
		#print("value: " + str(value))
		if value > max_val:
			max_val = value
			optimal_n_cluster = i +2 

	#print("ssumd: " + str(sum_squared[optimal_n_cluster-2]))
	km = KMeans(n_clusters = optimal_n_cluster)
	#print("cluster number: " + str(optimal_n_cluster))
	y_predict = km.fit_predict(Z)
	'''
	Y = pd.DataFrame(X_)
	Y.insert(1, "LABEL NO" , labels_, True)
	Y.columns =['Name', 'Label', 'Semi-major axis(m)', 'Inclination', 'Raan']
	Y['Label'] = Y['Label'].astype(float) 

	labels_to_be_assigned = list(range(len(grp_numbers_tb_added)))
	n = len(grp_numbers_tb_added)
	#print(" n : " + str(n))
	power_ten = 10
	while int(n/power_ten) != 0:
		power_ten *= 10

	labels_to_be_assigned = [iter_count + x/power_ten for x in labels_to_be_assigned]
	num_cluster += len(labels_to_be_assigned)

	removed_objects = 0
	for ind in Y.index: 
		flag = 0

		Y_label = Y['Label'][ind]
		for j in range(len(grp_numbers_tb_not_added)):
			if Y_label == grp_numbers_tb_not_added[j]:
				df_copy = pd.DataFrame(Y.iloc[ind - removed_objects])
				outlier_data = outlier_data.append(df_copy.T)
				outlier_data['Label'] = None
				Y.drop([ind],inplace=True) 
				removed_objects += 1
				flag = 1
				break

		if flag == 0:
			for j in range(len(grp_numbers_tb_added)):
				if Y_label == grp_numbers_tb_added[j]:
					Y['Label'][ind] = labels_to_be_assigned[j]
					break

	Y.reset_index(inplace = True) 
	Y.drop(Y.columns[0], axis=1, inplace=True)
	main_data = main_data.append(Y)
	main_data.reset_index(inplace = True) 
	main_data.drop(main_data.columns[0], axis=1, inplace=True)
	#print(outlier_data)
	if outlier_data.empty:
		break
	outlier_data.drop(outlier_data.columns[1], axis=1, inplace=True)
	outlier_data.reset_index(inplace = True) 
	outlier_data.drop(outlier_data.columns[0], axis=1, inplace=True)

	X = outlier_data.iloc[:, [1]].values
	X_ = outlier_data.iloc[:, [ 0, 1, 2, 3]].values
	#print(X)
	n= len(X)
	#print(n)
	W = np.ones(n)
	Z = list(zip(X,W))
	Z = pd.DataFrame(Z)


	print("main_data ;")
	print(main_data)
	print("outlier data: ")
	print(outlier_data)
	print("---------------------------------------------------------------------------")
	outlier_num = outlier_data.shape[0]
	iterations = outlier_num
	if outlier_num <= 7:
		label_ = list(range(outlier_num))
		#label_  = [x+num_cluster for x in label_]
		label_  = [None for x in label_]
		outlier_data.insert(1, 'Label' , label_, True)
		outlier_data.reset_index(inplace = True) 
		outlier_data.drop(outlier_data.columns[0], axis=1, inplace=True)
		#main_data = main_data.append(outlier_data)
		break
	
	iter_count += 1
	if iter_count> len(threshold_array)-1:
		break
	min_threshold = threshold_array[iter_count]

main_data = main_data.append(outlier_data)
main_data.reset_index(inplace = True) 
main_data.drop(main_data.columns[0], axis=1, inplace=True)
print(main_data)
main_data.to_csv('../data_files/kmeans+dbscan2.csv')


