# This program finds types of arrangements for objects within similar orbital plane.

# libraries to import 
import math
import pandas as pd
from datetime import datetime
import math
import numpy as np
from decimal import *
from space_track_api import getTLESatellites
import os
import time
getcontext().prec = 6

#SCIENTIFIC CONSTANTS
mue = 3.986004418e+14

# load clustered data(kmeans+dbscan)
dataset = pd.read_csv('../data_files/dbscan_raan_inc_entire_data1.csv')
start_index = 0 
final_index = dataset.shape[0]
dataset = dataset.loc[250:900,:]
dataset.drop(dataset.columns[0], axis=1, inplace=True)
dataset.reset_index(inplace = True) 
dataset.drop(dataset.columns[0], axis=1, inplace=True)


# this function finds type1 arrangements( refer to readme files for clear meaning of type1 arrangements)
def equal_angular_spacing(data):
	#print('Entered the function- equal_angular_spacing')
	flag = 0
	epoch_ascended_data = pd.DataFrame()
	epoch_ascended_data = data
	epoch_ascended_data = epoch_ascended_data.sort_values(by=['EPOCH'])
	epoch_ascended_data.reset_index(inplace = True)
	epoch_ascended_data.drop(epoch_ascended_data.columns[0], axis=1, inplace=True)
	rows = epoch_ascended_data.shape[0]
	last_epoch = epoch_ascended_data.loc[rows-1,'EPOCH']

	mean_long_data = pd.DataFrame()
	mean_long_data = data
	mean_long_data = mean_long_data.sort_values(by=['Mean_Longitude'])
	mean_long_data.reset_index(inplace = True)
	mean_long_data.drop(mean_long_data.columns[0], axis=1, inplace=True)

	longitudes = []
	verify_check_no = 1
	verify_count = 0
	while verify_count < verify_check_no :
		long_per_count = []
		for i in range(mean_long_data.shape[0]):
			epoch_date = mean_long_data.loc[i,'EPOCH']
			sma = mean_long_data.loc[i, 'Semi-major axis(m)']
			epoch = datetime.strptime(epoch_date, '%Y-%m-%d %H:%M:%S')

			epoch_date1 = last_epoch
			epoch1 = datetime.strptime(epoch_date1, '%Y-%m-%d %H:%M:%S')

			d0= datetime(epoch.year, epoch.month, epoch.day,  epoch.hour, epoch.minute,  epoch.second)        
			d1= datetime(epoch1.year, epoch1.month, epoch1.day,  epoch1.hour, epoch1.minute, epoch1.second)
			duration = d1 - d0                         
			duration_in_s = duration.total_seconds() - 86400*(verify_count)

			omega = ((mue)**0.5)/(sma**1.5)
			delta_long = math.degrees(omega*duration_in_s)
			mean_long_final = (mean_long_data.loc[i,'Mean_Longitude'] + delta_long)%360
			long_per_count.append(mean_long_final)

		longitudes.append(long_per_count)
		verify_count+=1

	for i in range(verify_check_no):
		mean_long_data.insert(len(mean_long_data.columns), 'mean-longitude at last epoch ' , longitudes[i], True)

	for i in range(len(longitudes)):
		long_per_count = longitudes[i]
		long_per_count_sorted = sorted(long_per_count)

		diff = []
		for j in range(len(long_per_count_sorted)):
			if j< len(long_per_count_sorted) -1 :
				difference = long_per_count_sorted[j+1] - long_per_count_sorted[j]
				if difference< 0:
					return False, mean_long_data
				diff.append(difference)
			if j ==  len(long_per_count_sorted) - 1:
				difference = 360 - long_per_count_sorted[j] + long_per_count_sorted[0]
				if difference < 0:
					return False, mean_long_data
				diff.append(difference)

		num_outliers = 0
		allowed_outliers = 0.20 * len(long_per_count)
		angle_spacing = 360/len(long_per_count) # this is angle that must be between two consecutive objects for equal spacing 
		count = 0
		for k in range(len(diff)):
			if abs(diff[(k+1)%len(diff)] - diff[k]) <= 0.50* angle_spacing:
				count+=1
			else:
				num_outliers+=1
		if (num_outliers<= allowed_outliers) and (count+num_outliers) == len(diff) and np.sum(diff)%360 <= 0.50* angle_spacing:
			flag = 1
		elif count < len(diff) - 1:
			flag = 0
			break


	if flag == 1:
		return True, mean_long_data
	if flag == 0:
		return False, mean_long_data

# this function finds type2 arrangements( refer to readme files for clear meaning of type1 arrangements)
def equal_spacing(data):
	#print('Entered the function- equal_spacing')
	flag = 0
	epoch_ascended_data = pd.DataFrame()
	epoch_ascended_data = data
	epoch_ascended_data = epoch_ascended_data.sort_values(by=['EPOCH'])
	epoch_ascended_data.reset_index(inplace = True)
	epoch_ascended_data.drop(epoch_ascended_data.columns[0], axis=1, inplace=True)
	rows = epoch_ascended_data.shape[0]
	last_epoch = epoch_ascended_data.loc[rows-1,'EPOCH']

	mean_long_data = pd.DataFrame()
	mean_long_data = data
	mean_long_data = mean_long_data.sort_values(by=['Mean_Longitude'])
	mean_long_data.reset_index(inplace = True)
	mean_long_data.drop(mean_long_data.columns[0], axis=1, inplace=True)

	longitudes = []
	verify_count = 0
	verify_check_no = 1
	while verify_count < verify_check_no :
		long_per_count = []
		for i in range(mean_long_data.shape[0]):
			epoch_date = mean_long_data.loc[i,'EPOCH']
			sma = mean_long_data.loc[i, 'Semi-major axis(m)']
			epoch = datetime.strptime(epoch_date, '%Y-%m-%d %H:%M:%S')

			epoch_date1 = last_epoch
			epoch1 = datetime.strptime(epoch_date1, '%Y-%m-%d %H:%M:%S')

			d0= datetime(epoch.year, epoch.month, epoch.day,  epoch.hour, epoch.minute,  epoch.second)        
			d1= datetime(epoch1.year, epoch1.month, epoch1.day,  epoch1.hour, epoch1.minute, epoch1.second)
			duration = d1 - d0                         
			duration_in_s = duration.total_seconds() - 86400*(verify_count)

			omega = ((mue)**0.5)/(sma**1.5)
			delta_long = math.degrees(omega*duration_in_s)
			mean_long_final = (mean_long_data.loc[i,'Mean_Longitude'] + delta_long)%360
			long_per_count.append(mean_long_final)

		longitudes.append(long_per_count)
		verify_count+=1

	for i in range(verify_check_no):
		mean_long_data.insert(len(mean_long_data.columns), 'mean-longitude at last epoch ' , longitudes[i], True)

	for i in range(len(longitudes)):
		long_per_count = longitudes[i]
		long_per_count_sorted = sorted(long_per_count)

		diff = []
		for j in range(len(long_per_count_sorted)):
			if j< len(long_per_count_sorted) -1 :
				difference = long_per_count_sorted[j+1] - long_per_count_sorted[j]
				if difference< 0:
					return False, mean_long_data
				diff.append(difference)
			if j ==  len(long_per_count_sorted) - 1:
				difference = 360 - long_per_count_sorted[j] + long_per_count_sorted[0]
				if difference < 0:
					return False, mean_long_data
				diff.append(difference)

		count = 0
		for k in range(len(diff)):
			if abs(diff[(k+1)%len(diff)] - diff[k]) <= 5:
				count+=1
		if count == len(diff) - 1 and len(diff)>2:
			flag = 1
		elif count < len(diff) - 1:
			flag = 0
			break


	if flag == 1:
		return True, mean_long_data
	if flag == 0:
		return False, mean_long_data


# this function finds type3 arrangements( refer to readme files for clear meaning of type1 arrangements)
def equal_spacing_with_defects(data):
	#print('Entered the function- equal_spacing_with_defects')
	flag = 0
	epoch_ascended_data = pd.DataFrame()
	epoch_ascended_data = data
	epoch_ascended_data = epoch_ascended_data.sort_values(by=['EPOCH'])
	epoch_ascended_data.reset_index(inplace = True)
	epoch_ascended_data.drop(epoch_ascended_data.columns[0], axis=1, inplace=True)
	rows = epoch_ascended_data.shape[0]
	last_epoch = epoch_ascended_data.loc[rows-1,'EPOCH']

	mean_long_data = pd.DataFrame()
	mean_long_data = data
	mean_long_data = mean_long_data.sort_values(by=['Mean_Longitude'])
	mean_long_data.reset_index(inplace = True)
	mean_long_data.drop(mean_long_data.columns[0], axis=1, inplace=True)

	longitudes = []
	verify_count = 0
	verify_check_no = 1
	while verify_count < verify_check_no :
		long_per_count = []
		for i in range(mean_long_data.shape[0]):
			epoch_date = mean_long_data.loc[i,'EPOCH']
			sma = mean_long_data.loc[i, 'Semi-major axis(m)']
			##print("sma : " + str(sma))
			epoch = datetime.strptime(epoch_date, '%Y-%m-%d %H:%M:%S')

			epoch_date1 = last_epoch
			epoch1 = datetime.strptime(epoch_date1, '%Y-%m-%d %H:%M:%S')

			d0= datetime(epoch.year, epoch.month, epoch.day,  epoch.hour, epoch.minute,  epoch.second)        
			d1= datetime(epoch1.year, epoch1.month, epoch1.day,  epoch1.hour, epoch1.minute, epoch1.second)
			duration = d1 - d0                         
			duration_in_s = duration.total_seconds() - 86400*(verify_count)

			omega = ((mue)**0.5)/(sma**1.5)
			delta_long = math.degrees(omega*duration_in_s)
			mean_long_final = (mean_long_data.loc[i,'Mean_Longitude'] + delta_long)%360
			long_per_count.append(mean_long_final)

		longitudes.append(long_per_count)
		verify_count+=1

	for i in range(verify_check_no):
		mean_long_data.insert(len(mean_long_data.columns), 'mean-longitude at last epoch ' , longitudes[i], True)

	for i in range(len(longitudes)):
		long_per_count = longitudes[i]
		long_per_count_sorted = sorted(long_per_count)

		diff = []
		for j in range(len(long_per_count_sorted)):
			if j< len(long_per_count_sorted) -1 :
				difference = long_per_count_sorted[j+1] - long_per_count_sorted[j]
				if difference< 0:
					return False, mean_long_data
				diff.append(difference)
			if j ==  len(long_per_count_sorted) - 1:
				difference = 360 - long_per_count_sorted[j] + long_per_count_sorted[0]
				if difference < 0:
					return False, mean_long_data
				diff.append(difference)

		count = 0
		for k in range(len(diff)):
			if abs(diff[(k+1)%len(diff)] - diff[k]) <= 5:
				count+=1
		if count >= len(diff) - 1 or len(diff)<=2:
			flag = 0
			break
		elif count < len(diff) - 1 and count > len(diff) -3:
			flag = 1

	if flag == 1:
		return True, mean_long_data
	if flag == 0:
		return False, mean_long_data

#######################################################################

# This function does the verification step from online space track data.
def verification_equal_angular_spacing(data):
	#print('ENTERED verification_equal_angular_spacing..')
	time.sleep(1)
	latest_data = pd.DataFrame()
	file_object = open('../data_files/temp.tle', 'w')
	file_object.close()

	labels = []
	dbscan_labels = []
	sma_classes = []
	sma_clusters = []

	for i in range(len(data)):
		file_object = open('../data_files/temp.tle', 'a')

		if math.isnan(data.loc[i,'Norad_ID'])==True:
			return True, data

		NORAD_id = str(int(data.loc[i,'Norad_ID']))
		epoch = str(data.loc[i, 'EPOCH'])
		labels.append(data.loc[i,'Label'])
		dbscan_labels.append(data.loc[i,'dbscan_labels'])
		sma_classes.append(data.loc[i,'SMA_Class'])
		sma_clusters.append(data.loc[i,'SMA_Cluster'])

		time.sleep(1)
		tle_data = getTLESatellites('tle', 'EPOCH', 'desc', '1', '3le',
					 'NORAD_CAT_ID', '', NORAD_id, 'EPOCH', '<', epoch)

		if tle_data == 'NO RESULTS RETURNED':
			print("The verification has been stopped due to below objects: ")
			print("NORAD_CAT_ID: " + str(NORAD_id))
			print('NAME: ' + str(data.loc[i, 'Name']))
			print("EPOCH: " + str(epoch))
			print(" ")
			return True, data

		#print("TLE DATA" )
		#print(tle_data)

		file_object.write(tle_data)
		file_object.close()

	'''
	f = open('../data_files/temp.tle', 'r')
	file_contents = f.read()
	print("DATA IN FILE")
	print (file_contents)
	f.close()
	'''

	os.system('python3 data_from_tle.py')
	latest_data = pd.read_csv('../data_files/data_from_recent_3les.csv')
	latest_data = latest_data.loc[:,:]
	latest_data.drop(latest_data.columns[0], axis=1, inplace=True)
	latest_data.reset_index(inplace = True) 
	latest_data.drop(latest_data.columns[0], axis=1, inplace=True)

	latest_data.insert(3, 'Label', labels, True)
	latest_data.insert(4, 'dbscan_labels', dbscan_labels, True)
	latest_data.insert(5, 'SMA_Class', sma_classes, True)
	latest_data.insert(6, 'SMA_Cluster', sma_clusters, True)

	#print("latest_data")
	#print(latest_data)

	type1, data= equal_angular_spacing(latest_data)

	return type1, data

########################################################################
# MAIN_CODE
i = 0
main_data = pd.DataFrame()
while i < dataset.shape[0]:   
	sma_label = dataset.loc[i,'Label']
	if math.isnan(sma_label)==True:
		i+=1
		continue

	# (i,j) is the index for which the subgroups are found with same labels(similar values of sma)
	j = i
	while j < dataset.shape[0] and  dataset.loc[j,'Label'] == sma_label:
		j+=1

	j-=1
	if i >= j:
		break

	sma_group = (i,j)  # index tuple for objects with same labels

	# sma_subgrouped_data contains objects with similar values of sma(same labels) , but arranged in ascending values 
	# of dbscan labels(groupings done for RAAN,inc values)

	sma_subgrouped_data = pd.DataFrame()
	sma_subgrouped_data = dataset[start_index + i: start_index + j+1]
	sma_subgrouped_data = sma_subgrouped_data.sort_values(by=['dbscan_labels'])
	sma_subgrouped_data.reset_index(inplace = True)
	sma_subgrouped_data.drop(sma_subgrouped_data.columns[0], axis=1, inplace=True)

	k = 0
	while k < sma_subgrouped_data.shape[0]:
		dbscan_label = sma_subgrouped_data.loc[k,'dbscan_labels']
		label_ = dbscan_label
		if math.isnan(label_)==True or label_ == -1:
			k+=1
			continue

		# (k,p) is the index for which the subgroups(within similar sma) are found with same dbscan-labels
		p = k
		while p< sma_subgrouped_data.shape[0] and sma_subgrouped_data.loc[p,'dbscan_labels'] == label_ :
			p+=1

		p-=1
		if k >= p:
			break

		dbscan_group = (k,p) # index tuple for objects with same  dbscan-labels	
		dbscan_grouped_data = pd.DataFrame()
		dbscan_grouped_data = sma_subgrouped_data[k:p+1]

		list_= [-1 for i in range(dbscan_grouped_data.shape[0])]
		type1, dbscan_mean_long_ascend_grouped_data= equal_angular_spacing(dbscan_grouped_data)
		if type1 == True:
			type1, dbscan_mean_long_ascend_grouped_data = verification_equal_angular_spacing(dbscan_mean_long_ascend_grouped_data)
			if type1 == True:
				list_ = [1 for i in range(dbscan_grouped_data.shape[0])]

		type2, dbscan_mean_long_ascend_grouped_data = equal_spacing(dbscan_grouped_data)
		if type2 == True:
			list_ = [2 for i in range(dbscan_grouped_data.shape[0])]

		type3, dbscan_mean_long_ascend_grouped_data = equal_spacing_with_defects(dbscan_grouped_data)
		if type3 == True:
			list_ = [3 for i in range(dbscan_grouped_data.shape[0])]
		

		dbscan_mean_long_ascend_grouped_data.insert(len(dbscan_mean_long_ascend_grouped_data.columns),
														"Type of Arrangement" , list_, True)
		main_data = main_data.append(dbscan_mean_long_ascend_grouped_data)
		main_data.reset_index(inplace = True) 
		main_data.drop(main_data.columns[0], axis=1, inplace=True)

		k = p+1

	i=j+1


# final data created and saved to system.
main_data.to_csv('../data_files/arrangement_in_orbits_catalogue.csv')
