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

# --------- propelled flight ---------
dt = 0.1 # time step
T = 40. # final time of propelled flight
N = int(T/dt) # number of time steps
t = np.linspace(0.0,T,N)

def euler_step(u,f,dt):
	return u + dt * f(u)

# defining rate equation for propelled flight
def f_prop(u):
	v = u[0]
	h = u[1]
	m_p = u[2] 

	return np.array([-g + (m_pdot*v_e)/(m_s+m_p)\
					-(0.5*rho*v*abs(v)*A*C_D)/(m_s+m_p),\
					v, -m_pdot])

# initializing solution array
u_prop = np.empty((N,3))
u_prop[0] = np.array([v0,h0,m_po]) # initializing first values of array

# time loop through soln array
for n in range(N-1):
	u_prop[n+1] = euler_step(u_prop[n], f_prop, dt)
	if(u_prop[n+1,2]<=0):
		u_prop[n+1,2]=0
		m_pdot=0
	if(u_prop[n+1,1]<0):
		break

# pulling out parts of solution array
v = u_prop[0:n,0]
h = u_prop[0:n,1]
m = u_prop[0:n,2]
t= t[0:n]



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

print t[-1]
print v[-1]
