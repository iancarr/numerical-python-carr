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

# environmental parameters
g = 9.81 # acceleration due to grav (m/s**2)
rho = 1.091 # air density in (kg/m**3)

# initial conditions
v0 = 0. # initial rocket velocity (m/s)
h0 = 0. # initial rocket height (m)
m_po = 100. # inital mass of propellent (kg)

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

	return np.array([-g + (20.0*v_e)/(m_s+m_p) - \
					(0.5*rho*v*np.abs(v)*A* C_D)/(m_s+m_p),\
					v, 20])

def euler_step(u,f,dt):
	return u + dt * f(u)

# detting parameters for solve
T = 150.0 # final time
dt = 0.1 # time step
N = int(T/dt) + 1 # number of time steps
t = np.linspace(0.0,T,N)

# initializing solution array
u = np.empty((N,3))
u[0] = np.array([v0,h0,m_po]) # initializing first values of array

# time loop through soln array
for n in range(N-1):
	u[n+1] = euler_step(u[n], f, dt)

# pulling out parts of solution array
v = u[:,0]
h = u[:,1]

# plotting the height vs time
plt.figure(figsize=(10,6))
plt.grid(True)
plt.plot(t,h)

plt.show()
