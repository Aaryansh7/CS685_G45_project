'''
This program detects Inclination Maneuvers.
'''

#Python libraries to import
import numpy as np 
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
import xlrd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


# This function filters points from local_minimum.
# It removes those points as local minima which are near to a local maxima.
# This is done to remove sharp abrupt changes, which are mostly not maneuevers
def filter_local_min(local_max, local_min):
	points_to_be_removed = []
	for i in range(len(local_max)):
		point = local_max[i]
		for j in range(len(local_min)):
			if local_min[j]>= point - 7 and local_min[j]<= point + 7:  #7 is used as a threshold
				points_to_be_removed.append(local_min[j])

	final_points = []
	for i in range(len(local_min)):
		flag=0
		for j in range(len(points_to_be_removed)):
			if local_min[i] == points_to_be_removed[j]:
				flag=1
				break
		if flag==0:
			final_points.append(local_min[i])

	return final_points


# This function takes all the possible detected maneuver dates and selects unique date
# for dates that are consecutive.
def unique_points(maneuver_dates):
	unique_maneuver_dates = []
	i = 0
	while i < len(maneuver_dates):
		epoch_date = maneuver_dates[i]
		epoch = datetime.strptime(epoch_date, '%Y-%m-%d %H:%M:%S')
		d0= datetime(epoch.year, epoch.month, epoch.day,  epoch.hour, epoch.minute,  epoch.second)  

		for j in range(i, len(maneuver_dates)):
			epoch_date1 = maneuver_dates[j]
			epoch1 = datetime.strptime(epoch_date1, '%Y-%m-%d %H:%M:%S')
			d1= datetime(epoch1.year, epoch1.month, epoch1.day,  epoch1.hour, epoch1.minute,  epoch1.second) 

			duration = d1 - d0                        
			duration_in_s = duration.total_seconds() 
			duration_in_hours = abs(duration_in_s/3600)
			duration_in_days = duration_in_hours/24

			if duration_in_days<15:
				continue

			if duration_in_days > 15:
				index = i + (j-i)/2
				unique_date = maneuver_dates[int(index)]
				unique_maneuver_dates.append(unique_date)
				i = j+1
				break

	#print(unique_maneuver_dates)
	return unique_maneuver_dates


dataset = pd.read_csv('../data_files/envisat.csv')
start_point = 526
final_point = 1026
X_inc = dataset.iloc[:, [8]].values  # contains inclination values
X_ = np.ravel(X_inc)
X_inc_smoothed_vals = savgol_filter(X_, 101, 13)   # window size 101, polynomial order 13 
X_epoch = dataset.iloc[:, [2]].values


maneuver_dates = []
moving_avg = []
points = []
inclination_values = X_inc

for i in tqdm(range(len(X_inc)-3)):
	last_inc_vals = inclination_values[i:i+20]
	avg = np.mean(last_inc_vals)

	if X_inc[i]<avg - 0.002  and X_inc[i]<X_inc[i+1] and X_inc[i+1]<X_inc[i+2]:
		maneuver_dates.append(X_epoch[i][0])
		points.append(X_inc[i][0])

final_dates = unique_points(maneuver_dates)

#########################################################################################
'''
This portion compares the detected dates with the truth data
'''
truth_dataset = pd.read_excel('../data_files/Envisat_Events.xlsx', index_col = None, engine='openpyxl')

# This block of code converts the date-time in xlsx file to python format and adds them to 'datetime_cnvrtd'
book = xlrd.open_workbook("../data_files/Envisat_Events.xlsx")
datemode = book.datemode
first_sheet = book.sheet_by_index(0)
datetime_cnvrted = []
for i in range(1,truth_dataset.shape[0]+1):
	date = first_sheet.cell(i, 0)
	py_date = datetime(*xlrd.xldate_as_tuple(date.value,
												  book.datemode))
	
	d = py_date.strftime("%Y-%m-%d %H:%M:%S")
	datetime_cnvrted.append(d)


# Select dates only for Inclination maneuvers
truth_dataset.insert(7, "datetime_cnvrted" , datetime_cnvrted, True)
df_maintenance = truth_dataset[truth_dataset["Column4"] == 'Inclination']
df_ = df_maintenance.iloc[:,[7]].values                             # df_ -> contains actual dates for inclination maneuvers
#print("dates in truth data: " + str(df_))



# The detected dates which lie within 120hrs/5 days within the actual date are updated to actual dates
for i in range(len(final_dates)):
	epoch_date = final_dates[i]
	epoch = datetime.strptime(epoch_date, '%Y-%m-%d %H:%M:%S')
	d0= datetime(epoch.year, epoch.month, epoch.day,  epoch.hour, epoch.minute,  epoch.second)  
	for j in range(len(df_)):
		epoch_date1 = df_[j][0]
		epoch1 = datetime.strptime(epoch_date1, '%Y-%m-%d %H:%M:%S')
		d1= datetime(epoch1.year, epoch1.month, epoch1.day,  epoch1.hour, epoch1.minute, epoch1.second)
		duration =  d1 - d0
		duration_in_s = duration.total_seconds() 
		duration_in_hours = abs(duration_in_s/3600)

		if duration_in_hours < 120:   # 120 hrs is the neighbourhood for detection
			final_dates[i] = epoch_date1
			break

print("dates in generated data: " )
for i in range(len(final_dates)):	
	print(final_dates[i])

# labels for truth data are created.
# The epoch which is nearest to the actual maneuver date is labelled as 1 otherwise 0.
y_truth = np.zeros(len(X_epoch))
for i in range(len(df_)):
	flag = 0
	epoch_date = df_[i][0]
	epoch = datetime.strptime(epoch_date, '%Y-%m-%d %H:%M:%S')
	d0= datetime(epoch.year, epoch.month, epoch.day,  epoch.hour, epoch.minute,  epoch.second)      
	min_duration = float("inf")  

	for j in range(len(X_epoch)):
		epoch_date1 = X_epoch[j][0]
		epoch1 = datetime.strptime(epoch_date1, '%Y-%m-%d %H:%M:%S')
		d1= datetime(epoch1.year, epoch1.month, epoch1.day,  epoch1.hour, epoch1.minute, epoch1.second)
		duration = d1 - d0                        
		duration_in_s = duration.total_seconds() 
		duration_in_hours = abs(duration_in_s/3600)

		if duration_in_hours < min_duration:
			min_duration = duration_in_hours
		if duration_in_hours > min_duration:
			flag = 1
			y_truth[j] = 1
			break
	
#print("len X_epoch " + str(len(X_epoch)))		
#print("len y_truth: " + str(len(y_truth)))
sum_y_truth = np.sum(y_truth)
print("No. of labels marked as '1'(maneuver date) when mapping to Tle data from truth data: " + str(sum_y_truth))


# labels for predicted data are created.
# The epoch which is nearest to the predicted maneuver date is labelled as 1 otherwise 0.
y_pred = np.zeros(len(X_epoch))
for i in range(len(final_dates)):
	flag = 0
	epoch_date = final_dates[i]
	epoch = datetime.strptime(epoch_date, '%Y-%m-%d %H:%M:%S')
	d0= datetime(epoch.year, epoch.month, epoch.day,  epoch.hour, epoch.minute,  epoch.second)      
	min_duration = float("inf")  

	for j in range(len(X_epoch)):
		epoch_date1 = X_epoch[j][0]
		epoch1 = datetime.strptime(epoch_date1, '%Y-%m-%d %H:%M:%S')
		d1= datetime(epoch1.year, epoch1.month, epoch1.day,  epoch1.hour, epoch1.minute, epoch1.second)
		duration = d1 - d0                        
		duration_in_s = duration.total_seconds() 
		duration_in_hours = abs(duration_in_s/3600)

		if duration_in_hours < min_duration:
			min_duration = duration_in_hours
		if duration_in_hours > min_duration:
			flag = 1
			y_pred[j] = 1
			break
#print("..............")	
#print("len y_pred: " + str(len(y_pred)))
sum_y_pred = np.sum(y_pred)
print("No. of labels marked as '1'(maneuver date) when mapping to Tle data from predicted data: " + str(sum_y_pred))


no_of_matched = 0
no_of_unmatched = 0
for i in range(len(y_truth)):
	if y_truth[i] == y_pred [i]:
		no_of_matched+=1
	else:
		no_of_unmatched+=1
		
print("no.of matched points: " +str(no_of_matched))
print("no. of unmatched points: " + str(no_of_unmatched))
print("confusion matrix : " )
print(confusion_matrix(y_truth, y_pred))
tn, fp, fn, tp = confusion_matrix(y_truth, y_pred).ravel()
#print(tn)
#print(fp)
#print(fn)
#print(tp)
score = f1_score(y_truth, y_pred)
print("F1 score: " + str(score))

#################################################
#PLOT
'''
actual_dates = []
actual_dates_inc = []
for i in range(len(X_epoch)):
	if y_truth[i] == 1:
		actual_dates.append(X_epoch[i][0])
		actual_dates_sma.append(X_inc[i])

print(actual_dates)

predicted_dates = []
predicted_dates_inc = []
for i in range(len(X_epoch)):
	if y_pred[i] == 1:
		predicted_dates.append(X_epoch[i][0])
		predicted_dates_sma.append(X_inc[i])
'''