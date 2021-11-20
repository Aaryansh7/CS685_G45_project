'''
This program adds specific columns like norad-ids, epochs, mean-longitude etc.
to the kmeans clustered data on semi-major axis .  
'''

# Python libraries to import 
import numpy as np 
from tqdm import tqdm
import pandas as pd
import math

first_dataset = pd.read_csv('../data_files/kmeans+dbscan2.csv')  # first-dataset->contains the kmeans clustered data on semi-major axis 
X = first_dataset.iloc[:, [1, 2]].values  
#print(X)

second_dataset = pd.read_csv('../data_files/filtered_catalogue.csv') #second-dataset-> conatins filtered catalogue
X_ = second_dataset.iloc[:, [1, 2,3, 22,23]].values
#print(X_)

norad_ids = []
mean_longitutes = []
epochs = []
sublongitudes = []
for i in range(len(X)):
	satname = X[i][0]
	#print(satname)
	for j in range((len(X_))):
		satname_ = X_[j][1]
		if satname_ == satname:
			norad_ids.append(X_[j][0])
			mean_longitutes.append(X_[j][3])
			epochs.append(X_[j][2])
			sublongitudes.append(X_[j][4])
			break

#### Separate the 'Label' into 2 columns: 'SMA_Class' and 'SMA_Cluster'
def split_float(x):
	'''split float into parts before and after the decimal'''
	before, after = str(x).split('.')
	return int(before), (int(after)*10 if len(after)==0 else int(after))
   
ascended_label_main_data = first_dataset.sort_values(by=['Label'])
ascended_labels = ascended_label_main_data.iloc[:, [2]].values
labels = first_dataset.iloc[:, [2]].values

sma_class_labels = []
sma_cluster_labels = []
for i in range(len(labels)) :
	if math.isnan(labels[i][0])==True:
		sma_class_labels.append(None)
	else:
		before, after = str(labels[i][0]).split('.')
		whole = int(before)
		sma_class_labels.append(whole)

#print(sma_class_labels)

prev_label = -1
unique_labels = []
for i in range(len(ascended_labels)):
	if ascended_labels[i][0] != prev_label:
		unique_labels.append(ascended_labels[i][0])
		prev_label = ascended_labels[i][0]

unique_labels_ = []
for i in range(len(unique_labels)) :
	if math.isnan(unique_labels[i])==True:
		continue
	whole, frac = split_float(unique_labels[i])
	unique_labels_.append(whole)

for i in range(len(labels)) :
	label = sma_class_labels[i]
	if math.isnan(labels[i])==True:
		sma_cluster_labels.append(None)
		continue
	type_count = unique_labels_.count(label)
	exponent_ten = 1
	power_ten = 0
	quotient = 1
	while quotient>=1:
		exponent_ten*=10
		power_ten+=1
		quotient = type_count/exponent_ten

	frac = labels[i][0] - sma_class_labels[i]
	frac*=exponent_ten

	sma_cluster_labels.append(round(frac))

#first_dataset.drop(first_dataset.columns[2], axis=1, inplace=True)
first_dataset.insert(1, 'Norad_ID' , norad_ids, True)
first_dataset.insert(3, 'EPOCH' , epochs, True)
first_dataset.insert(5, 'SMA_Class' , sma_class_labels, True)
first_dataset.insert(6, 'SMA_Cluster' , sma_cluster_labels, True)
first_dataset.insert(10, 'Mean_Longitude' , mean_longitutes, True)
first_dataset.insert(11, 'Sub_Longitude' , sublongitudes, True)
first_dataset.drop(first_dataset.columns[0], axis=1, inplace=True)
first_dataset.to_csv('../data_files/kmeans+dbscan3.csv')

