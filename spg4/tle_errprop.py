# program for error propagation in tles

# libraries to import 
from sgp4.api import Satrec
from sgp4.api import jday
from sgp4 import omm
import numpy as np 
import matplotlib.pyplot as plt 
np.set_printoptions(precision=2)
plt.style.use('seaborn')


# calculates norm of a vector
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
while total_count<100: 
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

jd = np.zeros(50)
fr = np.zeros(50)
time_interval = []

for i in range(s_count):
	s = list_s[i]
	t = list_t[i]

	satellite = Satrec.twoline2rv(s, t)

	# julian date from satellite epoch
	jd_ = satellite.jdsatepoch
	a = satellite.epochdays
	d = int(a)
	fr_ = a - d

	jd[i] = jd_
	fr[i] = fr_
	time_interval.append(a)

#satellite object for first tle
satellite = Satrec.twoline2rv(list_s[0], list_t[0])
e, r, v = satellite.sgp4_array(jd, fr)

position_tle_one = []
velocity_tle_one= []
satellite = Satrec.twoline2rv(list_s[0], list_t[0])
for i in range(s_count):
	'''
	pos_r = norm(r[j][0], r[j][1], r[j][2])
	position_tle_one.append(pos_r)

	vel_v = norm(v[j][0], v[j][1], v[j][2])
	velocity_tle_one.append(vel_v)
	'''
	jd_ = jd[i]
	fr_ = fr[i]
	e, r, v = satellite.sgp4(jd_, fr_)
	pos = norm(r[0], r[1], r[2])
	vel = norm(v[0], v[1], v[2])

	position_tle_one.append(pos)
	velocity_tle_one.append(vel)

position_all = []
velocity_all = []
for i in range(s_count):
	s = list_s[i]
	t = list_t[i]

	satellite_ = Satrec.twoline2rv(s, t)
	jd_ = jd[i]
	fr_ = fr[i]
	e, r, v = satellite_.sgp4(jd_, fr_)
	pos = norm(r[0], r[1], r[2])
	vel = norm(v[0], v[1], v[2])

	position_all.append(pos)
	velocity_all.append(vel)

position = []
velocity= []

position.append(position_tle_one)
position.append(position_all)
velocity.append(velocity_tle_one)
velocity.append(velocity_all)


# Plotting results
fig1, (ax1) = plt.subplots(nrows=1, ncols=1)
fig3, (ax2) = plt.subplots(nrows=1, ncols=1)


colors=['red', 'blue', 'green', 'yellow', 'pink', 'orange', 'black', 'purple', 'brown', 'grey', 'skyblue', 'lightgreen']



ax1.plot(time_interval, position[0], label='Value obtained by propagated TLE' ,color='red', linestyle='--', marker='o')
ax1.plot(time_interval, position[1], label='Actual/True Value',color='blue', linestyle='--', marker='o')
ax2.plot(time_interval, velocity[0], label='Value obtained by propagated TLE',color='red', linestyle='--', marker='o')
ax2.plot(time_interval, velocity[1], label='Actual/True Value',color='blue', linestyle='--', marker='o')

ax1.legend()
ax1.set_xlabel('epochtime')
ax1.set_ylabel('Position(m')

ax2.legend()
ax2.set_xlabel('epochtime')
ax2.set_ylabel('Velocity(m/s)')


plt.tight_layout()

plt.show()

