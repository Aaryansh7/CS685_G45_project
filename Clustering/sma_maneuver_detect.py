import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter
import numpy as np
from scipy.signal import argrelextrema
import xlrd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

np.set_printoptions(precision=2)
plt.style.use('seaborn')

def filter_local_min(local_max, local_min):
	points_to_be_removed = []
	for i in range(len(local_max)):
		point = local_max[i]
		for j in range(len(local_min)):
			if local_min[j]>= point - 7 and local_min[j]<= point + 7:
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

dataset = pd.read_csv('../data_files/envisat.csv')
start_point = 9480
final_point = 9780
X_sma = dataset.iloc[:, [4]].values
X_ = np.ravel(X_sma)
X_sma_smoothed_vals = savgol_filter(X_, 25, 3) # window size 51, polynomial order 3 (31,10)
X_epoch = dataset.iloc[:, [2]].values

derivative = []
maneuver_dates = []
for i in range(len(X_sma)):
	
	if i == len(X_sma) - 1:
		epoch_date = X_epoch[i][0]
		epoch = datetime.strptime(epoch_date, '%Y-%m-%d %H:%M:%S')

		epoch_date1 = X_epoch[i-1][0]
		epoch1 = datetime.strptime(epoch_date1, '%Y-%m-%d %H:%M:%S')

		d0= datetime(epoch.year, epoch.month, epoch.day,  epoch.hour, epoch.minute,  epoch.second)        # Random date in the past
		d1= datetime(epoch1.year, epoch1.month, epoch1.day,  epoch1.hour, epoch1.minute, epoch1.second)
		duration = d1 - d0                         # For build-in functions
		duration_in_s = duration.total_seconds() 
		duration_hours = duration_in_s/3600
		#print(duration_hours)

		num0 = X_sma[i][0]
		num1 = X_sma[i-1][0]

		if num1-num0 == 0:
			derivative.append(0)
		else:
			derivative.append((num1-num0)/(duration_hours))

		break

	epoch_date = X_epoch[i][0]
	epoch = datetime.strptime(epoch_date, '%Y-%m-%d %H:%M:%S')

	epoch_date1 = X_epoch[i+1][0]
	epoch1 = datetime.strptime(epoch_date1, '%Y-%m-%d %H:%M:%S')

	d0= datetime(epoch.year, epoch.month, epoch.day,  epoch.hour, epoch.minute,  epoch.second)        # Random date in the past
	d1= datetime(epoch1.year, epoch1.month, epoch1.day,  epoch1.hour, epoch1.minute, epoch1.second)
	duration = d1 - d0                         # For build-in functions
	duration_in_s = duration.total_seconds() 
	duration_hours = duration_in_s/3600

	num0 = X_sma_smoothed_vals[i]
	num1 = X_sma_smoothed_vals[i+1]

	if num1-num0 == 0:
		derivative.append(0)
	else:
		#print(duration_hours)
		if duration_hours == 0:
			duration_hours = 0.02
		derivative.append((num1-num0)/(duration_hours))
	
	if abs(derivative[i])>2:
		derivative[i] = derivative[i-1]

a = argrelextrema(X_sma_smoothed_vals, np.greater)

# for local minima
b = argrelextrema(X_sma_smoothed_vals, np.less)

filtered_points = filter_local_min(a[0],b[0])
maneuver_dates = []
for i in range(len(filtered_points)):
	if filtered_points[i]+2<len(X_epoch):
		maneuver_dates.append(X_epoch[filtered_points[i]+2][0])
print("Number of maneuver_dates predicted : " + str(len(maneuver_dates)))

#########################################################################################
truth_dataset = pd.read_excel('../data_files/Envisat_Events.xlsx', index_col = None, engine='openpyxl')

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


truth_dataset.insert(7, "datetime_cnvrted" , datetime_cnvrted, True)
df_maintenance = truth_dataset[truth_dataset["Column4"] == 'Maintenance']
df_ = df_maintenance.iloc[:,[7]].values
print("No. of maneuver dates in truth data: " + str(df_maintenance.shape[0]))

for i in range(len(maneuver_dates)):
	epoch_date = maneuver_dates[i]
	epoch = datetime.strptime(epoch_date, '%Y-%m-%d %H:%M:%S')
	d0= datetime(epoch.year, epoch.month, epoch.day,  epoch.hour, epoch.minute,  epoch.second)  
	for j in range(len(df_)):
		epoch_date1 = df_[j][0]
		epoch1 = datetime.strptime(epoch_date1, '%Y-%m-%d %H:%M:%S')
		d1= datetime(epoch1.year, epoch1.month, epoch1.day,  epoch1.hour, epoch1.minute, epoch1.second)
		duration =  d1 - d0
		duration_in_s = duration.total_seconds() 
		duration_in_hours = abs(duration_in_s/3600)
		#print("duration_hours: " + str(duration_in_hours))
		if duration_in_hours < 120:
			maneuver_dates[i] = epoch_date1
			break


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
		#print(duration_in_hours)

		if duration_in_hours < min_duration:
			min_duration = duration_in_hours
		if duration_in_hours > min_duration:
			#print("no")
			flag = 1
			y_truth[j] = 1
			break
	
#print("len X_epoch " + str(len(X_epoch)))		
#print("len y_truth: " + str(len(y_truth)))

sum_y_truth = np.sum(y_truth)
print("No. of labels marked as '1'(maneuver date) when mapping to Tle data from truth data: " + str(sum_y_truth))
#print(y_truth)

y_pred = np.zeros(len(X_epoch))
for i in range(len(maneuver_dates)):
	flag = 0
	epoch_date = maneuver_dates[i]
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
		#print(duration_in_hours)

		if duration_in_hours < min_duration:
			min_duration = duration_in_hours
		if duration_in_hours > min_duration:
			#print("no")
			flag = 1
			y_pred[j] = 1
			break

#print("len X_epoch " + str(len(X_epoch)))		
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
print("No.of matched labels(both 0 and 1): " +str(no_of_matched))
print("No. of unmatched labels(noth 0 and 1): " + str(no_of_unmatched))
print("confusion matrix : " )
print(confusion_matrix(y_truth, y_pred))
tn, fp, fn, tp = confusion_matrix(y_truth, y_pred).ravel()
#print(tn)
#print(fp)
#print(fn)
#print(tp)
score = f1_score(y_truth, y_pred)
print("F1 score: " + str(score))


print("---------------")
print("Tuned maneuver_dates are as follows: ")
for i in range(len(maneuver_dates)):
	print(maneuver_dates[i])
#################################################
#PLOT
actual_dates = []
actual_dates_sma = []
for i in range(len(X_epoch)):
	if y_truth[i] == 1:
		actual_dates.append(X_epoch[i][0])
		actual_dates_sma.append(X_sma[i])

#print(actual_dates)

predicted_dates = []
predicted_dates_sma = []
for i in range(len(X_epoch)):
	if y_pred[i] == 1:
		predicted_dates.append(X_epoch[i][0])
		predicted_dates_sma.append(X_sma[i])

'''
#print(X_epoch)
#print(X_sma)
X_epoch_ = []
X_sma_ = []
for k in range(len(X_epoch)):
	X_epoch_.append(X_epoch[k][0])

for k in range(len(X_sma)):
	X_sma_.append(X_sma[k][0])

#print(X_epoch_)
#print(X_sma_)


colors=['red', 'blue', 'green']
fig1, (ax1) = plt.subplots(nrows=1, ncols=1)

ax1.plot(X_sma_, X_epoch_, label='TEME' , linestyle='--', marker='o')

ax1.legend()
ax1.set_xlabel('epoch_day')
ax1.set_ylabel('position')
plt.tight_layout()
plt.show()
'''