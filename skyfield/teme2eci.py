#program to transform from teme frame to ECI frame using skyfield library

from skyfield.api import EarthSatellite
from skyfield.api import load, wgs84
from sgp4.api import Satrec
from sgp4.api import days2mdhms
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt 
np.set_printoptions(precision=5)
plt.style.use('seaborn')

file1 = open('zarya.txt', 'r') 
total_count = 0
s_count = 0
t_count = 0
list_s = []
list_t = []
au = 1.496e+8
day = 86400
km_sc = au/day
  
while total_count<24: 
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


def norm_v(v):
	return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def cross(a, b):
	cross_vector = []
	v1 = a[1]*b[2] - a[2]*b[1]
	v2 = -(a[0]*b[2] - a[2]*b[0])
	v3 = a[0]*b[1] - a[1]*b[0]

	cross_vector.append(v1)
	cross_vector.append(v2)
	cross_vector.append(v3)

	return cross_vector

position_teme= []
velocity_teme = []

position_eci = []
velocity_eci = []

position_ric= []
velocity_ric = []

error_pos = []
error_vel =[]

time = []

for i in range(s_count):
	s = list_s[i]
	t = list_t[i]

	ts = load.timescale()
	satellite = EarthSatellite(s, t, 'ISS (ZARYA)')
	satellite_ =  Satrec.twoline2rv(s, t)
	
	print ("..................................")
	print(satellite)
	#print(satellite.epoch.tt)
	#print(satellite_.epochyr)

	jd_ = satellite_.jdsatepoch
	a = satellite_.epochdays
	d = int(a)
	fr_ = a - d
	#print(jd_)
	#print(fr_)
	e, r, v = satellite_.sgp4(jd_, fr_)
	time.append(a)


	#print(..)
	#t = satellite.epoch.tt
	#geocentric = satellite.at(t)
	#print(geocentric.position.km)

	month, day, hour, minute, second = days2mdhms(satellite_.epochyr, satellite_.epochdays)
	year = 0
	if satellite_.epochyr>1999:
		year = 2000 + satellite_.epochyr
	else:
		year = 1900 + satellite_.epochyr
	t = ts.utc(year, month, day, hour, minute, int(second))
	#print(year)
	#print(month)
	#print(day)
	#print(hour)
	#print(minute)
	#print(second)
	geocentric = satellite.at(t)
	#pprint(vars(geocentric))
	#print(r)
	r_ = geocentric.position.km
	v_ = geocentric.velocity
	# v_ *= k
	#print(r_)
	#pprint(r_[0])
	#print(v)
	#pprint(vars(v_))
	#print(v_)
	#print(v_.au_per_d[0]*km_sc)

	#print(v_.info)

	reci = np.array([r_[0], r_[1], r_[2]])
	veci = np.array([v_.au_per_d[0]*km_sc, v_.au_per_d[1]*km_sc, v_.au_per_d[2]*km_sc])


	print("position in TEME: " + str(r))
	print("velocity in TEME: " + str(v))
	pos_t = norm_v(r)
	vel_t = norm_v(v)
	position_teme.append(pos_t)
	velocity_teme.append(vel_t)

	print("position in ECI: " + str(reci))
	print("veclocity in ECI: " + str(veci))
	pos_e = norm_v(reci)
	vel_e = norm_v(veci)
	position_eci.append(pos_e)
	velocity_eci.append(vel_e)

	r_u = reci/norm_v(reci)
	r_w = cross(reci, veci)/norm_v(cross(reci, veci))
	r_v = cross(r_w, r_u)/norm_v(cross(r_w, r_u))

	r_u = r_u[..., None] 
	r_w = r_w[..., None] 
	r_v = r_v[..., None] 

	T = np.zeros((3,3))
	T[0][0] = r_u[0]
	T[1][0] = r_u[1]
	T[2][0] = r_u[2]

	T[0][1] = r_v[0]
	T[1][1] = r_v[1]
	T[2][1] = r_v[2]

	T[0][2] = r_w[0]
	T[1][2] = r_w[1]
	T[2][2] = r_w[2]

	r_ric = np.dot(T, reci)
	v_ric = np.dot(T, veci)

	print("positon in RIC: " + str(r_ric))
	print("velocity in RIC: " + str(v_ric))
	pos_r = norm_v(r_ric)
	vel_r = norm_v(v_ric)
	position_ric.append(pos_r)
	velocity_ric.append(vel_r)

	err_p = r_ric - reci
	err_v = v_ric - veci
	error_pos.append(err_p)
	error_vel.append(err_v)
	#print(T)

colors=['red', 'blue', 'green']

fig1, (ax1) = plt.subplots(nrows=1, ncols=1)
fig2, (ax2) = plt.subplots(nrows=1, ncols=1)
fig3, (ax3) = plt.subplots(nrows=1, ncols=1)
fig4, (ax4) = plt.subplots(nrows=1, ncols=1)

ax1.plot(time, position_teme, label='TEME' ,color=colors[0], linestyle='--', marker='o')
ax1.plot(time, position_eci, label='ECI' ,color=colors[1], linestyle='--', marker='o')
ax1.plot(time, position_ric, label='RIC' ,color=colors[2], linestyle='--', marker='o')
ax2.plot(time,  velocity_teme, label='TEME' ,color=colors[0], linestyle='--', marker='o')
ax2.plot(time,  velocity_eci, label='ECI' ,color=colors[1], linestyle='--', marker='o')
ax2.plot(time,  velocity_ric, label='RIC' ,color=colors[2], linestyle='--', marker='o')

ax3.plot(time, error_pos, linestyle='-', marker='o')
ax4.plot(time, error_vel, linestyle='-', marker='o')

ax1.legend()
ax1.set_xlabel('epoch_day')
ax1.set_ylabel('position')

ax2.legend()
ax2.set_xlabel('epoch_day')
ax2.set_ylabel('velocity')

ax3.set_xlabel('epoch_day')
ax3.set_ylabel('err_pos')

ax4.set_xlabel('epoch_day')
ax4.set_xlabel('err_vel')
plt.tight_layout()

plt.show()





