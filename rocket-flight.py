# Coding assignment: Rocket Flight
# Ian Carr Sep 17 2014

import numpy as np 
import matplotlib.pyplot as plt 
from math import *
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# rocket parameters
m_s = 50. # rocket shell mass (kg)
C_D = 0.15 # coeff of drag
v_e = 325. # speed of exhaust (m/s)
r = 0.5 # radius of the rocket
A = np.pi*r**2 # area of the rocket

m_pdot = 20 # burn rate of the propellent (kg/s)

# environmental parameters
g = 9.81 # acceleration due to grav (m/s**2)
rho = 1.091 # air density in (kg/m**3)

# initial conditions
v0 = 0.0 # initial velocity of the rocket
h0 = 0.0 # initial height of the rocket
m_po = 100. # inital mass of propellent (kg)


def f(u):
	"""
	Returns the parameters of the system
	Parameters: u - array of float
				soln at time np
	Returns: dudt - array of float
				RHS of the given u
	"""
	v = u[0]
	h = u[1]
	m_p = u[2]

	return([-g + (m_pdot*v_e)/(m_s + m_p) +\
			 (.5*rho*v*np.abs(v)*A*C_D)/(m_s+m_p), v, \
			 20*dt])

def euler_step(u,f,dt):
	"""
	Returns soln at next time step
	Parameters: u - array of float - soln at prev time step
				f - func - func to compute RHS
				dt - float - time increment
	"""
	return u + dt * f(u)


# time step
dt = 0.1 # in seconds

# parameters for plotting
T = 100.0 # final time
N = int(T/dt) + 1 # time increment
t = np.linspace(0.0, T, N) # time discretization

# the soln at each step
u = np.empty((N, 3))
u[0] = np.array([v0, h0, m_po]) # to fill initial values

print type(u)
print type(f)
print type(dt)

# time loop for euler stepping
for n in range(N-1):
	u[n+1] = euler_step(u[n], f, dt)

# get parameters from euler step
v = u[:,0]
h = u[:,1]

# plot the rocket speed and height against time
plt.figure(figsize=(8,6))
plt.grid(True)
plt.xlabel('time', fontsize=18)
plt.ylabel('velocity', fontsize=18)
plt.title('Rocket velocity', fontsize=18)
plt.plot(t,v, 'k-', lw=2)

plt.show()