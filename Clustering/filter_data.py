'''
This program filters the catalogue.
It removes the debris and other unwanted objects.
'''

# Python libraries to import
import numpy as np 
from tqdm import tqdm
import pandas as pd

import_dataset = pd.read_csv('../data_files/catalogue.csv')
dataset = import_dataset.loc[:, :]

for ind in tqdm(dataset.index):
	flag = 0
	satellite_name = dataset['SATNAME'][ind]
	
	split_name =satellite_name.strip('\n')
	split_name_array = split_name.split(" ")

	for i in range(len(split_name_array)):
		if split_name_array[i] == 'DEB' or split_name_array[i] == 'R/B' or split_name_array[i] == 'COOLANT' or split_name_array[i] == 'FUEL CORE':
			flag = 1
			break

	if flag ==1:
		dataset.drop([ind],inplace=True)

dataset.reset_index(inplace = True) 
dataset.drop(dataset.columns[0], axis=1, inplace=True) 
dataset.drop(dataset.columns[0], axis=1, inplace=True)
dataset.to_csv('../data_files/filtered_catalogue.csv')




