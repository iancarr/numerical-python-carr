# Module 2 Homework - Traffic flow
# Ian Carr - Oct 13

import numpy as np
import matplotlib.pyplot as plt 
import sympy as sp 

# parameter declarations
v_max = 136. # max velocity in km/hr
L = 11. # length of road examined
rho_max = 250. # density of cars in cars/km
nx = 51 # number of spacial points 
dt = 0.001 # time interval in hours
sigma = .5 # CFL condition

T_min = 3 # final time in minutes
T = T_min*0.0166666666667 # final time in hours
nt = T/dt # number of time steps
nt = int(nt)
print nt

# initial conditions
x = np.linspace(0,L,nx)
dx = x[2]-x[1]
rho0 = np.ones(nx)*20
rho0[10:20] = 50
rho = rho0.copy()

# interating density
for n in range(1,nt):
	rhon = rho.copy()
	rho[1:] = rhon[1:]-dt/dx*(v_max*(rhon[1:]-rhon[0:-1])*\
			(1-(rhon[1:]-rhon[0:-1])/rho_max))
	rho[0]=20
"""
for n in range(1,nt):
	rhon = rho.copy()
	for i in range(1,nx):
		rho[i] = rhon[i]-dt/dx*v_max*(rhon[i]-rhon[i-1])*\
					(1-((rhon[i]-rhon[i-1])/rho_max))
		rho[0]=10
"""
# calculation of velocity
v_cars = v_max*(1-rho/rho_max)*(0.2777778)
print v_cars 
print "average velocity: ", np.average(v_cars)
print "minimum velocity: ", min(v_cars)

plt.plot(x,rho0)
plt.plot(x,rho)
plt.ylabel('density (cars/km)')
plt.xlabel('position (km)')
plt.ylim(0,60)
plt.show()

