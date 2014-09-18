# Paper Airplane model code
# Ian Carr Sep 14 2014

"""
This code borrows largely from M1L3 with 
modifications for modeling paper airplane 
flight.
"""

from math import sin, cos, log, ceil
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import itertools
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# model parameters
g = 9.81 	# gravity (m/s)
v_t = 4.9 	# trim velocity (m/s)
C_D = 1/5. # drag coeff
C_L = 1.0  	# lift coeff = 1 for conveinence

# ------- set initial conditions -------
#v0 = 4.7
#theta0 = 0
x0 = 0.0 		# horisontal position (arbitrary)
y0 = 2.0		# initial altitude - about 6 ft
				# assuming someone throwing form head height

# ------- test all possiblities for angle and speed -------
v01 = np.linspace(1,9.8,num=10)	# start at trim velocity
theta01 = np.linspace((-np.pi/2),(np.pi/3),num=10) 	# inital angle of trajectory

def get_flight_path(v0, theta0, x0, y0):
	# define function for sys of eqn
	def f(u):
		v = u[0]
		theta = u[1]
		x = u[2]
		y = u[3]

		return np.array([-g*sin(theta) - C_D/C_L*g/v_t**2*v**2,\
						-g*cos(theta)/v + g/v_t**2*v,\
						v*cos(theta),\
						v*sin(theta)])

	# solve system using eulers method

	def euler_step(u,f,dt):
		return u + dt*f(u) 
	    
	# ------- solve the system to get trajectory -------
	 
	T = 10.0			# final time
	dt = 0.01 			# time-step
	N = int(T/dt) + 1 	# number of time-steps
	t = np.linspace(0.0,T,N) # discretized time

	# initialize the array containing soln for each time-step
	u = np.empty((N,4))
	u[0] = np.array([v0,theta0,x0,y0]) # fill the 1st element 

	# time loop - Euler Method
	for n in range(N-1):
		u[n+1] = euler_step(u[n],f,dt)

	# get position from soln array
	x = u[:,2]
	y = u[:,3]

	# -------- find ground impact -------

	for i in range(len(y)):
		if y[i]>0:
			x_pa = x[0:i] # coordinates of plane flight (only pos)
			y_pa = y[0:i]

	impact = x_pa[-1]

	return impact

# ------- looping over all possiblities ------- 
impact_range = np.zeros((len(v01),len(theta01)))

for i in range(len(v01)):
	for j in range(len(theta01)):
		impact_range[i,j] = get_flight_path(v01[i],theta01[j],x0,y0)


impact_max = np.amax(impact_range)
impact_index = np.argmax(impact_range)

print impact_max
print "Largest index: ", impact_index
print "Velocity: ",v01[9]
print "Angle: ",theta01[4]*np.pi/180

# -------- plot the trajectory ------
"""
# plot the position information
plt.figure(figsize=(8,6))
plt.grid(True)
plt.xlabel(r'x',fontsize=18)
plt.ylabel(r'y',fontsize=18)
plt.title('Glider trajectory',fontsize=18)
plt.plot(x_pa,y_pa,'k-',lw=2)
"""
