
from .Repeat_Cycle_main import *

# Search Condition
maxlat   = 1.0 # max error of latitude[deg]
maxlong  = 1.0 # max error of longitude[deg]
maxdays  =  60 # max days of search

def Repeat_Cycle(line1, line2, line3, epoch):
	Cyclicity = {
		"Initial_Latitude_" : None,
		"Initial_Longitude_" : None,
		"Latitude_" : None,
		"Longitude_" : None,
		"Diff_Latitude" : None,
		"Diff_Longitude" : None,
		"Days_" : None,
		"Min_Distance" : None,
	}
	
	TLE = (line1, line2, line3)
	epoch = str(epoch)
	datestrs = epoch
	Cyclicity['Initial_Latitude_'], Cyclicity['Initial_Longitude_'], Cyclicity['Latitude_'], Cyclicity['Longitude_'], Cyclicity['Diff_Latitude'], Cyclicity['Diff_Longitude'], Cyclicity['Days_'], Cyclicity['Min_Distance'] = searchRepeatCycles(TLE, datestrs, maxlat, maxlong, maxdays)
	return Cyclicity
