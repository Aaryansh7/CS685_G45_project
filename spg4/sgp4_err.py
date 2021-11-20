# This program analysis error in sgp4 frame.

#libraries to import
import numpy as np 
from sgp4.api import Satrec
from sgp4.api import jday
import matplotlib.pyplot as plt 
from tqdm import tqdm
np.set_printoptions(precision=2)
plt.style.use('seaborn')

file1 = open('zarya.txt', 'r') 
total_count = 0
s_count = 0
t_count = 0
list_s = []
list_t = []
  
# loop to get tle data
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

#########################################################

num_intervals = 29
time_interval = np.linspace(0, 336, num_intervals)

position = []
velocity = []
jd= np.zeros(num_intervals)
fr= np.zeros(num_intervals)

def norm(x1, x2, x3):
	norm = np.sqrt(x1**2 + x2**2 + x3**2)
	return norm


for i in range(num_intervals):
	hour = time_interval[i]
	jd_, fr_ = jday(1998, 11, 21, hour, 0, 0)
	jd[i] = jd_
	fr[i] = fr_

for i in range(s_count):
	s = list_s[i]
	t = list_t[i]

	satellite = Satrec.twoline2rv(s, t)

	e, r, v = satellite.sgp4_array(jd, fr)
	pos = []
	vel = [] 

	for j in range(num_intervals):
		pos_r = norm(r[j][0], r[j][1], r[j][2])
		pos.append(pos_r)

		vel_v = norm(v[j][0], v[j][1], v[j][2])
		vel.append(vel_v)

	position.append(pos)
	velocity.append(vel)


# PLotting results 
fig1, (ax1) = plt.subplots(nrows=1, ncols=1)
fig3, (ax2) = plt.subplots(nrows=1, ncols=1)


colors=['red', 'blue', 'green', 'yellow', 'pink', 'orange', 'black', 'purple', 'brown', 'grey', 'skyblue', 'lightgreen']

for i in range(s_count):

	ax1.plot(time_interval, position[i], label='%s TLE' % i ,color=colors[i])

	ax2.plot(time_interval, velocity[i], label='%s TLE' % i ,color=colors[i])


	if i == s_count-1:
		ax1.legend()
		ax1.set_xlabel('Time(epochs)')
		ax1.set_ylabel('Position(m)')

		ax2.legend()
		ax2.set_xlabel('Time(epochs)')
		ax2.set_ylabel('Velocity(m/s)')


		plt.tight_layout()

		plt.show()

		break



