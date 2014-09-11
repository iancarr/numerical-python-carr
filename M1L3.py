# Module 1 Lesson 3 - Full Phugoid Model
# Ian Carr Sep 11 2014

from math import sin, cos, log, ceil
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParamsp['font.size'] = 16

# model parameters
g = 9.81 	# gravity (m/s)
v_t = 30.0 	# trim velocity (m/s)
C_D = 1/40. # drag coeff
C_L = 1.0  	# lift coeff - 1 for conveinence

# ------- set initial conditions -------
v0 = v_t		# start at trim velocity
theta0 = 0.0 	# inital angle of trajectory
x0 = 0.0 		# horisontal position (arbitrary)
y0 = 1000.0		# initial altitude

# --------- system of equations --------

# define function for sys of eqn
def f(u):
	"""returns the right-hand side of the phugoid sys of eqn

	Parameters - u: array of floats
					array containing the solution at n

	Returns - dudt: array of float
					array containing the RHS for u
	"""

	v = u[0]
	theta = u[1]
	x = u[2]
	y = u[3]

	return np.array([-g*sin(theta) - C_D/C_L*g/v_t**2*v**2,
					-g*cos(theta)/v + g/v_t**2*v,
					v*cos(theta),
					v*sin(theta)])


# solve system using eulers method

def euler_step(u,f,dt):
	"""returns the solution at the next time-step

	Parameters - u: array of float
					solution to the prev time-step
				f: function
					funciton to compute the RHS of the sys
				dt: float
					time-step

	Returns: u_n_plus_1 : array of float
						approx soln at the next time-step
    """

    return u + dt * f(u)


# ------- solve the system to get trajectory -------
 
T = 100.0			# final time
dt = 0.1 			# time-step
N = int(T/dt) + 1 	# number of time-steps
t = np.linspace(0.0,T,N) # discretized time

# initialize the array containing soln for each time-step
u = np.empty((N,4))
u[0] = np.array([v0,theta0,x0,y0]) # fill the 1st element 

# time loop - Euler Method
for n in range(N-1):
	u[n+1] = euler_step(u[n],f,dt)

# -------- plot the trajectory ------

# get position from soln array
x = u[:,2]
y = u[:,3]
