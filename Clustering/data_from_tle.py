'''
1.Satname 
2.Epoch of TLE (python date-time format) 
3.Mean motion, xno (rev/day) 
4.Orbit period, tau (s) 
5.Semi-major axis, a (m) 
6.Semi-latus rectum, p (m) 
7.Inclination, i (deg) 
8.Right Ascension of Ascending Node, raan (Deg) 
9.Argument of Perigee, w (Deg) 
10.Eccentricity, e (unitless) 
1.Perigee altitude, alt_p (km) 
2.Apogee altitude, alt_a (km) 
11.Average altitude, alt (km) 
12.Circular Orbit flag: based on eccentricity (use +/- 2% as threshold) 
13.Orbit Class: based on altitude (LEO <= 2000km; 2000 < MEO <= 35,636; 35,636 < 
GEO <= 35,836; DSO > 35,836) 
14.Annual precession of orbit per year, omega_yr (rad/yr) => compute only where 
#12 is TRUE & Orbit Class = LEO 
15.Sun-synchronous orbit flag: based on omega_yr (use +/- 5% as threshold) 
16.Mean Local Time of Ascending Node, mltan (hour) => compute only where #15 is 
TRUE, leave as NaN if FALSE 
17.Ground-track repeat cycle, gt_per (days) => compute only where #12 = TRUE & 
#13 = LEO 
18.Mean anomaly, m_a (deg) 
19. Mean longitude
'''
import numpy as np 
from sgp4.api import Satrec
from sgp4.api import jday
import matplotlib.pyplot as plt 
from tqdm import tqdm
import pandas as pd
import math
from sgp4 import exporter
import kepler
from skyfield.api import load, wgs84
from skyfield.api import EarthSatellite
from datetime import date, datetime, timedelta
from pprint import pprint
#from Satellite_Repeat_Cycle import Search_Satellite_Repeat_Cycle_test
from skyfield.elementslib import osculating_elements_of

np.set_printoptions(precision=2)
plt.style.use('seaborn')

#SCIENTIFIC CONSTANTS AND PARAMETERS
gravitation_param = 3.986004418e14
Radius_Earth = 6371 #km
rad_min2rev_day = 229.1831180523 
rad2deg = 57.2958
J2 = 1.08263E-3
Re = 6378.1E3 
A = -360/365.25
ECCENTRICITY_THRESHOLD = 0

file = open('../data_files/temp.tle', 'r') 

total_tle = 120
total_count = 0
s_count = 0
t_count = 0
list_s = []
list_t = []
list_satname = []

  
while True: 
	# Get next line from file 
	line_file = file.readline()
	if not line_file: 
		break

	b =line_file.strip('\n')
	a = b.split(" ")
	#print(a)
	list_satname.append(b)

	line_file = file.readline()
	list_s.append(line_file.strip())
	s_count +=1

	line_file = file.readline()
	list_t.append(line_file.strip())
	t_count +=1

	total_count += 3
  
file.close()

norad_id =[]
satname = []
epoch = []
mean_motion = []
orbital_period = []
semi_major = []
semi_latus_rectum = []
inclination = []
right_ascension_ascending_node = []
argument_perigee = []
eccentricity = []
altitude_perigee = []
altitude_apogee = []
avg_altitude = []
circular_orbit = []
orbit_classification = []
annual_precession = []
sun_synchronous = []
mean_local_time_ascending_node = []
mean_anamoly = []
mean_longitude = []
sublongitude = []

Initial_Latitude_ = []
Initial_Longitude_ = []
Latitude_ = [] 
Longitude_ = []
Diff_Latitude = []
Diff_Longitude = [] 
Days_ = [] 
Min_Distance = []


def semi_major_axis(n):
	#print(n)
	n_ = n*(2*np.pi)/86400
	d = n_**(2/3)
	b = gravitation_param**(1/3)
	a = b/d

	return a/1000

def orbital_period__(sem_maj):
	period = ((2*np.pi)*(sem_maj**(1.5)))/(gravitation_param**(0.5))
	return period

def perigee_altitude(a, e):
	# a must be in Km
	a = a/1000
	Rp = a*(1-e)
	altitude = Rp - Radius_Earth
	return altitude

def apogee_altitude(a, e):
	# a must be in Km
	a = a/1000
	Rp = a*(1+e)
	altitude = Rp - Radius_Earth
	return altitude

def orbital_speed_perigee(alt_per, alt_apog, e):
	Rp = (Radius_Earth + alt_per)*1000
	Ra = (Radius_Earth + alt_apog)*1000
	velocity = np.sqrt((2*gravitation_param*Ra)/(Rp*(Ra + Rp)))
	return velocity

def annual_precession_(inc, tau, p):
	inc= math.radians(inc)
	del_raan = (math.cos(inc)/tau)*(-3*np.pi*J2*((Re**2)/(p**2)))
	omega_yr = del_raan * 365.25 * 24 * 60 * 60
	return omega_yr

def mean_time_ascending_node(inc,raan, epoch):
	equinox_date = None
	frac_day = None
	lat = 0
	inc = math.radians(inc)
	B = math.asin(math.tan(lat)/math.tan(inc))

	file1 = open('data_files/equinoxes.csv', 'r')
	line = file1.readline()

	while True: 
		# Get next line from file 
		line = file1.readline()
		equinox_date = line.strip()
		equinox = datetime.strptime(equinox_date, '%d-%b-%Y %H:%M:%S')
		#epoch = datetime.strptime(epoch_, '%Y-%m-%d %H:%M:%S')

		#print(type(equinox.year))
		d0 = date(equinox.year, equinox.month, equinox.day)
		d1 = date(epoch.year, epoch.month, epoch.day)
		delta = d1 - d0
		
		if delta.days < 366:
			# datetime(year, month, day, hour, minute, second) 
			a = datetime(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second) 
			b = datetime(equinox.year, equinox.month, equinox.day, equinox.hour, equinox.minute, equinox.second) 
			c = a - b
			frac_day = c/timedelta(1)

			break
  
		# if line is empty 
		# end of file is reached 
		if not line: 
			break
			print("Line no line") 
  
	file1.close()

	r_mltan = (1/15)*((A*frac_day) + raan + B) + 12
	return r_mltan % 24

def orbit_class(alt):
	# alt in km
	if alt <= 2000:
		return 'LEO'
	if alt > 2000 and alt<= 35636:
		return 'ME0'
	if alt > 35636 and alt<=35836:
		return 'GEO'
	if alt > 35836:
		return 'DSO'

	return



for i in tqdm(range(s_count)):
	satname_ = list_satname[i]
	s = list_s[i]
	t = list_t[i]
	#print(t)
	t_ =t.strip('\n')
	p = t_.split(" ")
	#print(p[1])
	satellite_spg4 = Satrec.twoline2rv(s, t)
	ts = load.timescale()
	satellite_skyfield = EarthSatellite(s, t, satname_, ts)
	geocentric = satellite_skyfield.at(satellite_skyfield.epoch)
	subpoint = wgs84.subpoint(geocentric)
	#fields = exporter.export_omm(satellite_spg4, satname_)
	#n_ = fields['MEAN_MOTION']
	n = satellite_spg4.no_kozai
	
	dte = datetime.strptime('2000-01-01 01:01:01',"%Y-%m-%d %H:%M:%S")
	epoch_ = dte.replace(minute=satellite_skyfield.epoch.utc.minute, hour=satellite_skyfield.epoch.utc.hour,
						 second=int(satellite_skyfield.epoch.utc.second), year=satellite_skyfield.epoch.utc.year,
						  month=satellite_skyfield.epoch.utc.month, day=satellite_skyfield.epoch.utc.day)

	mean_motion_ = satellite_spg4.no_kozai * rad_min2rev_day
	semi_major_ = semi_major_axis(mean_motion_)*1000 #in m
	orbital_period_ = orbital_period__(semi_major_) # in sec
	ecc = satellite_spg4.ecco
	semi_latus_rectum_ = semi_major_*(1- ecc**2) # in m
	inclination_ = satellite_spg4.inclo * rad2deg # in deg
	right_ascension_ascending_node_ = satellite_spg4.nodeo * rad2deg
	argument_perigee_ = satellite_spg4.argpo * rad2deg
	altitude_perigee_ = perigee_altitude(semi_major_, ecc)
	altitude_apogee_ = apogee_altitude(semi_major_, ecc)
	avg_altitude_ = (altitude_apogee_ + altitude_perigee_)/2
	circular_orbit_ = True if ecc < 0.01  else False
	#circular_orbit_ = False
	orbit_class_ = orbit_class(avg_altitude_)
	omega_yr = annual_precession_(inclination_, orbital_period_, semi_latus_rectum_) if circular_orbit_ == True and orbit_class_=='LEO' else None
	sun_synchronous_ = True if omega_yr!=None and omega_yr>0.95*2*np.pi and omega_yr<1.05*2*np.pi else False
	mean_local_time_ascending_node_ = mean_time_ascending_node(inclination_, right_ascension_ascending_node_, epoch_) if sun_synchronous_ == True else None
	#cycle = Search_Satellite_Repeat_Cycle_test. Repeat_Cycle(satname_,s,t,epoch_)['Days_'] if circular_orbit_ == True and orbit_class_ =='LEO' else None
	mean_anamoly_ = math.degrees(satellite_spg4.mo)
	#mean_longitude_ = mean_anamoly_ + argument_perigee_ + right_ascension_ascending_node_
	mean_longitude_ = satellite_spg4.mo + satellite_spg4.argpo + satellite_spg4.nodeo
	mean_longitude_ = math.degrees(mean_longitude_)
	#mean_longitude_ = mean_longitude_ % 360
	sublongitude_ = subpoint.longitude.degrees
	elements = osculating_elements_of(geocentric)

	norad_id.append(p[1])
	satname.append(satname_)
	epoch.append(epoch_)
	mean_motion.append(mean_motion_)
	semi_major.append(semi_major_)
	orbital_period.append(orbital_period_)
	eccentricity.append(ecc)
	semi_latus_rectum.append(semi_latus_rectum_)
	inclination.append(inclination_)
	right_ascension_ascending_node.append(right_ascension_ascending_node_)
	argument_perigee.append(argument_perigee_)
	altitude_perigee.append(altitude_perigee_)
	altitude_apogee.append(altitude_apogee_)
	avg_altitude.append(avg_altitude_)
	circular_orbit.append(circular_orbit_)
	orbit_classification.append(orbit_class_)
	annual_precession.append(omega_yr)
	sun_synchronous.append(sun_synchronous_)
	mean_local_time_ascending_node.append(mean_local_time_ascending_node_)
	mean_anamoly.append(mean_anamoly_)
	#Days_.append(cycle) 
	mean_longitude.append(elements.mean_longitude.degrees)
	sublongitude.append(sublongitude_)
	'''

	Initial_Latitude_.append(cycle['Initial_Latitude_'])
	Initial_Longitude_.append(cycle['Initial_Longitude_'])
	Latitude_.append(cycle['Latitude_']) 
	Longitude_.append(cycle['Longitude_'])
	Diff_Latitude.append(cycle['Diff_Latitude'])
	Diff_Longitude.append(cycle['Diff_Longitude']) 
	'''
		
	#Min_Distance.append(cycle['Min_Distance'])
	

list_tuples = list(zip(norad_id, satname, epoch, semi_major, inclination, right_ascension_ascending_node,
					   mean_longitude, sublongitude))
					  #Initial_Latitude_, Initial_Longitude_, Latitude_, Longitude_, Diff_Latitude, Diff_Longitude, Days_, Min_Distance))

dframe = pd.DataFrame(list_tuples, columns=['NORAD ID', 'Name', 'EPOCH', 'Semi-major axis(m)', 'Inclination',
											 'Raan', 'Mean_Longitude', 'Sub_Longitude'])
											 #'LATITUDE AT EPOCH(deg)', 'LONGITUDE AT EPOCH(deg)', 'LATITUDE FOR MIN DISTANCE(deg)', 'LONGITUDE FOR MIN DISTANCE(deg)',
											 #'DIFFERENCE IN LATITUDE(deg)', 'DIFFERENCE IN LONGITUDE(deg)', 'DAYS', 'MINIMIUM DISTANCE(km)'])   

#print(dframe) 
dframe.to_csv('../data_files/data_from_recent_3les.csv')
