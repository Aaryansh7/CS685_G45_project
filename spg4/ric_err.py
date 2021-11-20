#This program computes the errors in RIC frame. It also plots mean, standard devitaion in this frame. 

# libraries to import 
from sgp4.api import Satrec
from sgp4.api import jday
from skyfield.api import EarthSatellite
from skyfield.api import load, wgs84
from sgp4.api import days2mdhms
from sgp4 import omm
import numpy as np 
import statistics
from tqdm import tqdm
import matplotlib.pyplot as plt 
np.set_printoptions(precision=5)
plt.style.use('seaborn')

# Computes cross product of vectors
def cross(a, b):
	cross_vector = []
	v1 = a[1]*b[2] - a[2]*b[1]
	v2 = -(a[0]*b[2] - a[2]*b[0])
	v3 = a[0]*b[1] - a[1]*b[0]

	cross_vector.append(v1)
	cross_vector.append(v2)
	cross_vector.append(v3)

	return cross_vector

# computes norm of vector
def norm(x1, x2, x3):
	norm = np.sqrt(x1**2 + x2**2 + x3**2)
	return norm

def norm_v(v):
	return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

# converts from ECI-> RIC frame
def eci2ric(reci, veci):
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

	return r_ric, v_ric


file1 = open('zarya.txt', 'r') 

# CONSTANTS
total_count = 0
s_count = 0
t_count = 0
iter_count = 50
num_tles = 100
au = 1.496e+8
day = 86400
km_sc = au/day

time_interval = np.linspace(1, iter_count, iter_count)
tle_num = []
list_s = []
list_t = []
  
# loop to get tle data
while True: 
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

#Lists created for diff types of values to be computed

error_pos_teme = []
error_pos_eci = []
error_pos_ric = []
error_vel_teme= []
error_vel_eci = []
error_vel_ric = []

error_pos_eci_x = []
error_pos_eci_y = []
error_pos_eci_z = []
error_pos_ric_x = []
error_pos_ric_y = []
error_pos_ric_z = []
error_vel_ric_x = []
error_vel_ric_y = []
error_vel_ric_z = []
error_vel_eci_x = []
error_vel_eci_y = []
error_vel_eci_z = []

mean_err_pos_ric = []
std_err_pos_ric = []
mean_err_pos_teme = []
std_err_pos_teme = []

mean_err_pos_ric_x = []
std_err_pos_ric_x = []
mean_err_pos_ric_y = []
std_err_pos_ric_y= []
mean_err_pos_ric_z = []
std_err_pos_ric_z = []

for i in tqdm(range(num_tles)):
	s = list_s[i]
	t = list_t[i]
	tle_num.append(i)

	satellite_ref_sgp4  = Satrec.twoline2rv(s, t) # satellite object from sgp4 lib
	ts = load.timescale()
	satellite_ref_skyfield = EarthSatellite(s, t, 'ISS (ZARYA)') # satellite objet from skyfield lib


	#error lists created for diff types of values to be computed
	err_pos_teme = []
	err_pos_eci = []
	err_pos_ric = []
	err_vel_teme= []
	err_vel_eci = []
	err_vel_ric = []

	err_pos_eci_x = []
	err_pos_eci_y = []
	err_pos_eci_z = []
	err_pos_ric_x = []
	err_pos_ric_y = []
	err_pos_ric_z = []
	err_vel_ric_x = []	
	err_vel_ric_y = []
	err_vel_ric_z = []
	err_vel_eci_x = []
	err_vel_eci_y = []
	err_vel_eci_z = []

	for j in range(i, i + iter_count):
		satellite_sgp4 = Satrec.twoline2rv(list_s[j], list_t[j])
		satellite_skyfield = EarthSatellite(list_s[j], list_t[j], 'ISS (ZARA)')

		jd_ = satellite_sgp4.jdsatepoch
		a = satellite_sgp4.epochdays
		d = int(a)
		fr_ = a - d
		e, r, v = satellite_sgp4.sgp4(jd_, fr_)
		e_, r_, v_ = satellite_ref_sgp4.sgp4(jd_, fr_)
		pos_teme = norm_v(r)
		pos_teme_ = norm_v(r_)
		vel_teme = norm_v(v)
		vel_teme_ = norm_v(v_)

		month, day, hour, minute, second = days2mdhms(satellite_sgp4.epochyr, satellite_sgp4.epochdays)
		year = 0
		if satellite_sgp4.epochyr>1999:
			year = 2000 + satellite_sgp4.epochyr
		else:
			year = 1900 + satellite_sgp4.epochyr
		t_ = ts.utc(year, month, day, hour, minute, int(second))

		geocentric = satellite_skyfield.at(t_)
		geocentric_ = satellite_ref_skyfield.at(t_)
		r_eci = geocentric.position.km
		v_eci = geocentric.velocity
		r_eci_ = geocentric_.position.km
		v_eci_ = geocentric_.velocity

		reci = np.array([r_eci[0], r_eci[1], r_eci[2]])
		reci_ = np.array([r_eci_[0], r_eci_[1], r_eci_[2]])
		veci = np.array([v_eci.au_per_d[0]*km_sc, v_eci.au_per_d[1]*km_sc, v_eci.au_per_d[2]*km_sc])
		veci_ = np.array([v_eci_.au_per_d[0]*km_sc, v_eci_.au_per_d[1]*km_sc, v_eci_.au_per_d[2]*km_sc])
        
		pos_eci = norm_v(reci)
		pos_eci_ = norm_v(reci_)
		vel_eci = norm_v(veci)
		vel_eci_ = norm_v(veci_)

		r_ric, v_ric = eci2ric(reci, veci)
		r_ric_, v_ric_ = eci2ric(reci_, veci_)
		pos_ric = norm_v(r_ric)
		pos_ric_ = norm_v(r_ric_)
		vel_ric = norm_v(v_ric)
		vel_ric_ = norm_v(v_ric_)


		# error values calculated
		err_position_teme = pos_teme - pos_teme_
		err_position_eci = pos_eci - pos_eci_
		err_position_ric = abs(pos_ric - pos_ric_)
		err_velocity_teme = vel_teme - vel_teme_
		err_velocity_eci = vel_eci - vel_eci_
		err_velocity_ric = vel_ric - vel_ric_
		
		err_position_eci_x = reci[0] - reci_[0]
		err_position_eci_y = reci[1] - reci_[1]
		err_position_eci_z = reci[2] - reci_[2]
		err_velocity_eci_x = veci[0] - veci_[0]
		err_velocity_eci_y = veci[1] - veci_[1]
		err_velocity_eci_z = veci[2] - veci_[2]
		err_position_ric_x = abs(r_ric[0] - r_ric_[0])
		err_position_ric_y = abs(r_ric[1] - r_ric_[1])
		err_position_ric_z = abs(r_ric[2] - r_ric_[2])
		err_velocity_ric_x = v_ric[0] - v_ric_[0]
		err_velocity_ric_y = v_ric[1] - v_ric_[1]
		err_velocity_ric_z = v_ric[2] - v_ric_[2]
		

		# calculated values added into corresponding lists
		err_pos_teme.append(err_position_teme)
		err_pos_eci.append(err_position_eci)
		err_pos_ric.append(abs(err_position_ric))
		err_vel_teme.append(err_velocity_teme)
		err_vel_eci.append(err_velocity_eci)
		err_vel_ric.append(err_velocity_ric)

		err_pos_eci_x.append(err_position_eci_x)
		err_pos_eci_y.append(err_position_eci_y)
		err_pos_eci_z.append(err_position_eci_z)
		err_vel_eci_x.append(err_velocity_eci_x)
		err_vel_eci_y.append(err_velocity_eci_y)
		err_vel_eci_z.append(err_velocity_eci_z)
		err_pos_ric_x.append(err_position_ric_x)
		err_pos_ric_y.append(err_position_ric_y)
		err_pos_ric_z.append(err_position_ric_z)
		err_vel_ric_x.append(err_velocity_ric_x)
		err_vel_ric_y.append(err_velocity_ric_y)
		err_vel_ric_z.append(err_velocity_ric_z)
			
		


	error_pos_teme.append(err_pos_teme)
	error_pos_eci.append(err_pos_eci)
	error_pos_ric.append(err_pos_ric)
	error_vel_teme.append(err_vel_teme)
	error_vel_eci.append(err_vel_eci)
	error_vel_ric.append(err_vel_ric)

	error_pos_eci_x.append(err_pos_eci_x)
	error_pos_eci_y.append(err_pos_eci_y)
	error_pos_eci_z.append(err_pos_eci_z)
	error_vel_eci_x.append(err_vel_eci_x)
	error_vel_eci_y.append(err_vel_eci_y)
	error_vel_eci_z.append(err_vel_eci_z)
	error_pos_ric_x.append(err_pos_ric_x)
	error_pos_ric_y.append(err_pos_ric_y)
	error_pos_ric_z.append(err_pos_ric_z)
	error_vel_ric_x.append(err_vel_ric_x)
	error_vel_ric_y.append(err_vel_ric_y)
	error_vel_ric_z.append(err_vel_ric_z)

	mean_ric = statistics.mean(error_pos_ric[i])
	std_ric = statistics.stdev(error_pos_ric[i])
	mean_teme = statistics.mean(error_pos_teme[i])
	std_teme = statistics.stdev(error_pos_teme[i])

	mean_ric_x = statistics.mean(error_pos_ric_x[i])
	std_ric_x = statistics.stdev(error_pos_ric_x[i])
	mean_ric_y = statistics.mean(error_pos_ric_y[i])
	std_ric_y = statistics.stdev(error_pos_ric_y[i])
	mean_ric_z = statistics.mean(error_pos_ric_z[i])
	std_ric_z = statistics.stdev(error_pos_ric_z[i])

	mean_err_pos_ric.append(mean_ric)
	std_err_pos_ric.append(std_ric)
	mean_err_pos_teme.append(mean_teme)
	std_err_pos_teme.append(std_teme)

	mean_err_pos_ric_x.append(mean_ric_x)
	mean_err_pos_ric_y.append(mean_ric_y)
	mean_err_pos_ric_z.append(mean_ric_z)
	std_err_pos_ric_x.append(std_ric_x)
	std_err_pos_ric_y.append(std_ric_y)
	std_err_pos_ric_z.append(std_ric_z)


'''
fig1, (ax1) = plt.subplots(nrows=1, ncols=1)
fig2, (ax2) = plt.subplots(nrows=1, ncols=1)
fig3, (ax3) = plt.subplots(nrows=1, ncols=1)
fig4, (ax4) = plt.subplots(nrows=1, ncols=1)
fig5, (ax5) = plt.subplots(nrows=1, ncols=1)
fig6, (ax6) = plt.subplots(nrows=1, ncols=1)
fig7, (ax7) = plt.subplots(nrows=1, ncols=1)
fig8, (ax8) = plt.subplots(nrows=1, ncols=1)
'''

# PLotting results
fig9, (ax9) = plt.subplots(nrows=1, ncols=1)
fig10, (ax10) = plt.subplots(nrows=1, ncols=1)	
fig11, (ax11) = plt.subplots(nrows=1, ncols=1)
fig12, (ax12) = plt.subplots(nrows=1, ncols=1)		
fig13, (ax13) = plt.subplots(nrows=1, ncols=1)	
fig14, (ax14) = plt.subplots(nrows=1, ncols=1)	
fig15, (ax15) = plt.subplots(nrows=1, ncols=1)	
fig16, (ax16) = plt.subplots(nrows=1, ncols=1)	
fig17, (ax17) = plt.subplots(nrows=1, ncols=1)	
fig18, (ax18) = plt.subplots(nrows=1, ncols=1)	
fig19, (ax19) = plt.subplots(nrows=1, ncols=1)	
fig20, (ax20) = plt.subplots(nrows=1, ncols=1)	

colors=['red', 'blue', 'green', 'yellow', 'pink', 'orange', 'black', 'purple', 'brown', 'grey', 'skyblue', 'lightgreen']

ax9.plot(tle_num, mean_err_pos_ric, label='mean' ,color=colors[1], linestyle='--', marker='o')
ax10.plot(tle_num, std_err_pos_ric, label='std' ,color=colors[2], linestyle='--', marker='o')
ax11.plot(tle_num, mean_err_pos_teme, label='mean' ,color=colors[1], linestyle='--', marker='o')
ax12.plot(tle_num, std_err_pos_teme, label='std' ,color=colors[2], linestyle='--', marker='o')
ax13.plot(time_interval, error_pos_ric[60], label='error' ,color=colors[2], linestyle='--', marker='o')

ax14.plot(tle_num, mean_err_pos_ric_x, label='mean_x' ,color=colors[2], linestyle='--', marker='o')
ax15.plot(tle_num, std_err_pos_ric_x, label='std_x' ,color=colors[2], linestyle='--', marker='o')
ax16.plot(tle_num, mean_err_pos_ric_y, label='mean_y' ,color=colors[2], linestyle='--', marker='o')
ax17.plot(tle_num, std_err_pos_ric_y, label='std_y' ,color=colors[2], linestyle='--', marker='o')
ax18.plot(tle_num, mean_err_pos_ric_z, label='mean_z' ,color=colors[2], linestyle='--', marker='o')
ax19.plot(tle_num, std_err_pos_ric_z, label='std_z' ,color=colors[2], linestyle='--', marker='o')
ax20.plot(tle_num, mean_err_pos_ric_x, label='x' ,color=colors[0], linestyle='--', marker='o')
ax20.plot(tle_num, mean_err_pos_ric_y, label='y' ,color=colors[1], linestyle='--', marker='o')
ax20.plot(tle_num, mean_err_pos_ric_z, label='z' ,color=colors[2], linestyle='--', marker='o')

ax9.set_title('MEAN ERROR OF POSITION IN RIC FRAME')
ax9.legend()
ax9.set_xlabel('TLE')
ax9.set_ylabel('mean')

ax10.set_title('STANDARD DEVIATION OF POSITION IN RIC FRAME')
ax10.legend()
ax10.set_xlabel('TLE')
ax10.set_ylabel('std')

ax11.set_title('MEAN ERROR OF POSITION IN TEME FRAME')
ax11.legend()
ax11.set_xlabel('TLE')
ax11.set_ylabel('mean')

ax12.set_title('STANDARD DEVIATION OF POSITION IN TEME FRAME')
ax12.legend()
ax12.set_xlabel('TLE')
ax12.set_ylabel('std')

ax14.legend()
ax15.legend()
ax16.legend()
ax17.legend()
ax18.legend()
ax19.legend()

ax20.set_title('MEAN ERROR IN POS OF COORDINATES IN RIC FRAME')
ax20.legend()
ax20.set_xlabel('TLE')
ax20.set_ylabel('mean')
'''

for i in range(1):
	if i%1 ==0:

		ax1.plot(time_interval, error_pos_eci[i], label='%s TLE' % i ,color=colors[i], linestyle='--', marker='o')
		ax2.plot(time_interval, error_pos_ric[i], label='%s TLE' % i ,color=colors[i], linestyle='--', marker='o')
		ax3.plot(time_interval, error_vel_eci[i], label='%s TLE' % i ,color=colors[i], linestyle='--', marker='o')
		ax4.plot(time_interval, error_vel_ric[i], label='%s TLE' % i ,color=colors[i], linestyle='--', marker='o')
		ax5.plot(time_interval, error_vel_eci_x[i], label='x' ,color=colors[0], linestyle='--', marker='o')
		ax5.plot(time_interval, error_vel_eci_y[i], label='y' ,color=colors[1], linestyle='--', marker='o')
		ax5.plot(time_interval, error_vel_eci_z[i], label='z' ,color=colors[2], linestyle='--', marker='o')
		ax6.plot(time_interval, error_vel_ric_x[i], label='x' ,color=colors[0], linestyle='--', marker='o')
		ax6.plot(time_interval, error_vel_ric_y[i], label='y' ,color=colors[1], linestyle='--', marker='o')
		ax6.plot(time_interval, error_vel_ric_z[i], label='z' ,color=colors[2], linestyle='--', marker='o')
		ax7.plot(time_interval, error_pos_ric_x[i], label='x' ,color=colors[0], linestyle='--', marker='o')
		ax7.plot(time_interval, error_pos_ric_y[i], label='y' ,color=colors[1], linestyle='--', marker='o')
		ax7.plot(time_interval, error_pos_ric_z[i], label='z' ,color=colors[2], linestyle='--', marker='o')
		ax8.plot(time_interval, error_pos_eci_x[i], label='x' ,color=colors[0], linestyle='--', marker='o')
		ax8.plot(time_interval, error_pos_eci_y[i], label='y' ,color=colors[1], linestyle='--', marker='o')
		ax8.plot(time_interval, error_pos_eci_z[i], label='z' ,color=colors[2], linestyle='--', marker='o')
		#ax3.plot(time_interval, error_pos_ric_n_eci_x[i], label='x',color=colors[0], linestyle='--', marker='o')
		#ax3.plot(time_interval, error_pos_ric_n_eci_y[i], label='y',color=colors[1], linestyle='--', marker='o')
		#ax3.plot(time_interval, error_pos_ric_n_eci_z[i], label='z',color=colors[2], linestyle='--', marker='o')


	if i == 0:
		ax1.set_title('Error in position(ECI frame)')
		ax1.legend()
		ax1.set_xlabel('next TLE number')
		ax1.set_ylabel('error_pos')

		ax2.set_title('Error in position(RIC frame)')
		ax2.legend()
		ax2.set_xlabel('next TLE number')
		ax2.set_ylabel('error_vel')

		ax3.set_title('Error in velocity(ECI frame)')
		ax3.legend()
		ax3.set_xlabel('next TLE number')
		ax3.set_ylabel('error_pos')

		ax4.set_title('Error in velocity(RIC frame)')
		ax4.legend()
		ax4.set_xlabel('next TLE number')
		ax4.set_ylabel('error_vel')

		ax5.set_title('Error in velocity in axes(ECI frame)')
		ax5.legend()
		ax5.set_xlabel('next TLE number')
		ax5.set_ylabel('error_vel')

		ax6.set_title('Error in velocity in axes(RIC frame)')
		ax6.legend()
		ax6.set_xlabel('next TLE number')
		ax6.set_ylabel('error_vel')

		ax7.set_title('Error in Position in axes(RIC frame)')
		ax7.legend()
		ax7.set_xlabel('next TLE number')
		ax7.set_ylabel('error_pos')

		ax8.set_title('Error in Position in axes(ECI frame)')
		ax8.legend()
		ax8.set_xlabel('next TLE number')
		ax8.set_ylabel('error_pos')


		plt.tight_layout()

		plt.show()

		break
'''	
plt.tight_layout()

plt.show()