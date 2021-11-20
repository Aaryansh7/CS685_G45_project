# This program computes the average error in tles using spg4

#libraries to import 
from sgp4.api import Satrec
from sgp4.api import jday
from sgp4 import omm
import numpy as np 
import matplotlib.pyplot as plt 
np.set_printoptions(precision=5)
plt.style.use('seaborn')


# calculates norm of vector
def norm(x1, x2, x3):
	norm = np.sqrt(x1**2 + x2**2 + x3**2)
	return norm


file1 = open('zarya.txt', 'r') 
total_count = 0
s_count = 0
t_count = 0
list_s = []
list_t = []
  
# loop to get tle data
while total_count<500: 
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

jd = np.zeros(250)
fr = np.zeros(250)
time_interval = np.linspace(1, 100, 100)

for i in range(s_count):
	s = list_s[i]
	t = list_t[i]

	satellite = Satrec.twoline2rv(s, t)
	jd_ = satellite.jdsatepoch
	a = satellite.epochdays
	d = int(a)
	fr_ = a - d

	jd[i] = jd_
	fr[i] = fr_

err_position = []
err_velocity= []

# taking 12 tles to observe the propagation
for i in range(12):
	err_pos = []
	err_vel = []
	
	satellite_ref = Satrec.twoline2rv(list_s[i], list_t[i])
	a = satellite_ref.epochdays

	#propagates TLE to next 100 epochs/julian dates of next tles
	e, r, v = satellite_ref.sgp4_array(jd[i:i+100], fr[i:i+100])
	e_, r_, v_ = satellite_ref.sgp4(jd[i], fr[i])

	# will store pos, vel for propagated values
	pos_ref = []
	vel_ref = []
	for j in range(100):
		pos_r = norm(r[j][0], r[j][1], r[j][2])
		pos_ref.append(pos_r)

		vel_v = norm(v[j][0], v[j][1], v[j][2])
		vel_ref.append(vel_v)


	position_all = []
	velocity_all = []
	for j in range(i,i+100):
		s = list_s[j]
		t = list_t[j]

		satellite = Satrec.twoline2rv(s, t)
		a = satellite.epochdays
		jd_ = jd[j]
		fr_ = fr[j]

		e, r, v = satellite.sgp4(jd_, fr_)
		pos = norm(r[0], r[1], r[2])
		vel = norm(v[0], v[1], v[2])

		position_all.append(pos)
		velocity_all.append(vel)

	for j in range(100):
		pos_err = abs(position_all[j] - pos_ref[j])
		vel_err = abs(velocity_all[j] - vel_ref[j])

		err_pos.append(pos_err)
		err_vel.append(vel_err)

	err_position.append(err_pos)
	err_velocity.append(err_vel)

	'''
	plt.plot(time_interval, pos_ref, label='pos_ref')
	plt.plot(time_interval, position_all, label='position_all')
	plt.legend()
	plt.show()
	'''
	i+=5


# Plotting results
fig1, (ax1) = plt.subplots(nrows=1, ncols=1)
fig3, (ax2) = plt.subplots(nrows=1, ncols=1)


colors=['red', 'blue', 'green', 'yellow', 'pink', 'orange', 'black', 'purple', 'brown', 'grey', 'skyblue', 'lightgreen']

for i in range(12):
	if i ==5:

		ax1.plot(time_interval, err_position[i], label='error in position' ,color=colors[i], linestyle='--', marker='o')

		ax2.plot(time_interval, err_velocity[i], label='error in velocity' ,color=colors[i], linestyle='--', marker='o')


	if i == 5:
		ax1.legend()
		ax1.set_xlabel('next TLE number')
		ax1.set_ylabel('Error in Position(m)')

		ax2.legend()
		ax2.set_xlabel('next TLE number')
		ax2.set_ylabel('Error in velocity(m/s)')


		plt.tight_layout()

		plt.show()

		break