# Coding assignment: Rocket Flight
# Ian Carr Sep 17 2014

import numpy as np 
import matplotlib.pyplot as plt 
from math import *

# rocket parameters
m_s = 50. # rocket shell mass (kg)
C_D = 0.15 # coeff of drag
v_e = 325. # speed of exhaust (m/s)
r = 0.5 # radius of the rocket
A = np.pi*r**2 # area of the rocket
m_pdot = 20 # propellent burn rate (kg/s)

# environmental parameters
g = 9.81 # acceleration due to grav (m/s**2)
rho = 1.091 # air density in (kg/m**3)

# initial conditions
v0 = 0. # initial rocket velocity (m/s)
h0 = 0. # initial rocket height (m)
m_po = 100. # inital mass of propellent (kg)

# getting parameters for solve
T = 150.0 # final time
dt = 0.1 # time step
N = int(T/dt) + 1 # number of time steps
t = np.linspace(0.0,T,N)

# defining array describing propellent weight
m_p = np.zeros_like(t)

for i in range(50):
	m_p[i] = m_po - (20*t[i])

# 

def f(u):
	"""
	Parameters - u: array of float
					array containing soln at n
	Returns - dudt: array of float
					array containing RHS for u
	"""
	v = u[0]
	h = u[1]
	m_p = u[2] 

	return np.array([-g + (m_pdot*v_e)/(m_s+m_p)\
					-(0.5*rho*v*np.absolute(v)*A*C_D)/(m_s+m_p),\
					v, -m_pdot])

# stepping through time steps
def euler_step(u,f,dt):
	return u + dt * f(u)

# initializing solution array
u = np.empty((N,3))
u[0] = np.array([v0,h0,m_po]) # initializing first values of array


# time loop through soln array
for n in range(N-1):
	u[n+1] = euler_step(u[n], f, dt)

# pulling out parts of solution array
v = u[:,0]
h = u[:,1]
m_p = u[:,2]

print "Mass of fuel at t = 3.2: " , m_p[32]

# determining flight duration
for i in range(len(h)):
	if h[i]>0:
		time_impact = t[i]
		index_impact = i

print 'Time of impact: ',time_impact, 's'

# determining max velocity
v_max = np.amax(v)
v_max_index = np.argmax(v)

print 'Maximum velocity: ', v_max, 'm/s'
print 'Time of max velocity: ', t[v_max_index], 's'
print 'Height of max velocity: ', h[v_max_index], 'm'

# plotting the height vs time
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('time (s)', fontsize=16)
plt.ylabel('height (m)', fontsize=16)
plt.title('rocket height', fontsize=18)
plt.plot(t,h)

# plotting velocity vs time
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('time (s)', fontsize=16)
plt.ylabel('velocity (m/s)', fontsize=16)
plt.title('rocket velocity', fontsize=18)
plt.plot(t,v)

plt.show()
