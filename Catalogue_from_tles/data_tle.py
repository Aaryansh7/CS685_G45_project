# program to maake datasheet from tles and plot them

#semi-major axis,-->
#inclination, -->
#right ascension of ascending node, -->
#eccentricity ,--> 
#true anomaly, 
#orbital period, -->
#argument of perigee -->
# altitude of perigee -->
#altitude of apogee -->
# orbital speed at perigee ->

import numpy as np 
from sgp4.api import Satrec
from sgp4.api import jday
import matplotlib.pyplot as plt 
from tqdm import tqdm
import pandas as pd
import math
from sgp4 import exporter
import kepler


np.set_printoptions(precision=2)

plt.style.use('seaborn')

file1 = open('../data_files/zarya.txt', 'r') 
total_tle = 20426
gravitation_param = 3.986004418e14
Radius_Earth = 6371 #km
total_count = 0
s_count = 0
t_count = 0
list_s = []
list_t = []
  
while total_count<total_tle: 
	# Get next line from file 
	line = file1.readline() 
	list_s.append(line.strip())
	s_count +=1
	line = file1.readline()
	list_t.append(line.strip())
	t_count +=1
	total_count += 2
  
	# if line is empty 
	# end of file is reached 
	if not line: 
		break
		print("Line no line") 
  
file1.close() 

epoch =[]
inclination = []
right_ascension_ascending_node = []
eccentricity = []
argument_perigee = []
semi_major = []
orbital_period = []
true_anomaly = []
altitude_perigee_ = []
altitude_apogee_ = []
velocity_perigee = []

def semi_major_axis(n):
	#print(n)
	n_ = n*(2*np.pi)/86400
	d = n_**(2/3)
	#print(d)
	b = gravitation_param**(1/3)
	a = b/d

	return a/1000

def orbital_period_(n):
	n_ = n*(2*np.pi)/86400
	return n

def perigee_altitude(a, e):
	# a must be in Km
	Rp = a*(1-e)
	altitude = Rp - Radius_Earth
	return altitude

def apogee_altitude(a, e):
	# a must be in Km
	Rp = a*(1+e)
	altitude = Rp - Radius_Earth
	return altitude

def orbital_speed_perigee(alt_per, alt_apog, e):
	Rp = (Radius_Earth + alt_per)*1000
	Ra = (Radius_Earth + alt_apog)*1000
	velocity = np.sqrt((2*gravitation_param*Ra)/(Rp*(Ra + Rp)))
	return velocity

for i in range(s_count):
	s = list_s[i]
	t = list_t[i]
	satellite = Satrec.twoline2rv(s, t)
	fields = exporter.export_omm(satellite, 'ISS (ZARYA)')

	inc = satellite.inclo
	asc = satellite.nodeo
	ecc = satellite.ecco
	arg_per = satellite.argpo
	mean_anomaly = satellite.mo
	n = fields['MEAN_MOTION']
	sem_maj = semi_major_axis(n)
	orb_per = orbital_period_(n)
	altitude_perigee = perigee_altitude(sem_maj, ecc)
	altitude_apogee = apogee_altitude(sem_maj, ecc)
	velocity_per = orbital_speed_perigee(altitude_perigee, altitude_apogee, ecc)

	#calculation_for_true_anamoly
	eccentric_anomaly, cos_true_anomaly, sin_true_anomaly = kepler.kepler(mean_anomaly, ecc)
	ecc_an = math.degrees(eccentric_anomaly)
	tr_an = (math.cos(eccentric_anomaly) - ecc)/(1 - ecc*math.cos(eccentric_anomaly))
	tr_an = math.degrees(tr_an)

	epoch.append(satellite.jdsatepoch)
	inclination.append(inc)
	right_ascension_ascending_node.append(asc)
	eccentricity.append(ecc)
	argument_perigee.append(arg_per)
	semi_major.append(sem_maj)
	orbital_period.append(orb_per)
	true_anomaly.append(tr_an)
	altitude_perigee_.append(altitude_perigee)
	altitude_apogee_.append(altitude_apogee)
	velocity_perigee.append(velocity_per)

list_tuples = list(zip(epoch, semi_major, inclination, right_ascension_ascending_node, eccentricity, argument_perigee, orbital_period, true_anomaly, 
						altitude_perigee_, altitude_apogee_, velocity_perigee))
dframe = pd.DataFrame(list_tuples, columns=['Epoch', 'Semi Major Axis(km)', 'Inclination', 'Right Ascension ascending node', 'Eccentricity', 'Argument perigee', 'Orbital period', 'True anomaly',
											'Altitude Of Perigee(km)', 'Altitude of Apogee(km)', 'Orbital Speed at perigee(m/s)'])   

print(dframe) 
dframe.to_csv('../data_files/data.csv')

###################################################################################

fig1, (ax1) = plt.subplots(nrows=1, ncols=1)
fig2, (ax2) = plt.subplots(nrows=1, ncols=1)
fig3, (ax3) = plt.subplots(nrows=1, ncols=1)
fig4, (ax4) = plt.subplots(nrows=1, ncols=1)
fig5, (ax5) = plt.subplots(nrows=1, ncols=1)

colors=['red', 'blue', 'green', 'yellow', 'pink', 'orange', 'black', 'purple', 'brown', 'grey', 'skyblue', 'lightgreen']

ax1.plot(epoch, semi_major, label='semi-major' ,color=colors[1], linestyle='--', marker='o')
ax2.plot(epoch, eccentricity, label='eccentricity'  ,color=colors[1], linestyle='--', marker='o')
ax3.plot(epoch, argument_perigee, label='argument_perigee'  ,color=colors[1], linestyle='--', marker='o')
ax4.plot(epoch, orbital_period, label='orbital_period'  ,color=colors[1], linestyle='--', marker='o')
ax5.plot(epoch, velocity_perigee, label='velocity_perigee'  ,color=colors[1], linestyle='--', marker='o')

ax1.set_title('Semi Major Axis')
ax1.legend()
ax1.set_xlabel('epoch(julian date)')
ax1.set_ylabel('Distance(km)')

ax2.set_title('Eccentricity')
ax2.legend()
ax2.set_xlabel('epoch(julian date)')
ax2.set_ylabel('e')

ax3.set_title('Argument Perigee')
ax3.legend()
ax3.set_xlabel('epoch(julian date)')
ax3.set_ylabel('argument_perigee')

ax4.set_title('Orbital Period')
ax4.legend()
ax4.set_xlabel('epoch(julian date)')
ax4.set_ylabel('orbital_period(rev/day)')

ax5.set_title('Velocity Perigee')
ax5.legend()
ax5.set_xlabel('epoch(julian date)')
ax5.set_ylabel('m/s')

plt.tight_layout()

plt.show()