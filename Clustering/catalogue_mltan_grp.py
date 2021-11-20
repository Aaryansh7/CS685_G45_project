'''
This program classifies satellites on the basis of mltan values.
The classification is done for RADAR Satellites and Optical Imaging satellites 
'''

# Python libraries to import
import numpy as np 
from tqdm import tqdm
import pandas as pd

dataset = pd.read_csv('../data_files/filtered_catalogue.csv')
start_index_dataset = 1     
final_index_dataset = 76
X = dataset.iloc[:, [19]].values   # X-> contains mltan values

category = []
for i in tqdm(range(len(X))):
	if X[i] == None:
		category.append(X[i][0])
	else:
		if (X[i][0]< 0.5 and X[i][0] > 23.5) or (X[i][0]<12.5 and X[i][0]>11.5) or(X[i][0]< 18.5 and X[i][0]>17.5) or (X[i][0]< 6.5 and X[i][0]>5.5): 
			category.append('RADAR')

		elif (X[i][0]< 11.0 and X[i][0] > 8.5) or (X[i][0]< 23.0 and X[i][0]>20.5) or (X[i][0]< 3.5 and X[i][0]>1.0) or (X[i][0]< 15.5 and X[i][0]>13.0): 
			category.append('OPTICAL IMAGING')

		else:
			category.append(None)

dataset.insert(20, "SATELLITE TYPE" , category, True)
dataset.to_csv('../data_files/catalogue_mltan.csv')