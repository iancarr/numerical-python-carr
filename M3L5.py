# Module 3 Lesson 5 - Sod's Shock Tube
# Ian Carr - Oct 30 2014

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
from math import *

# defining function for initial conditions
def u_initial(nx,gamma,rho_l,rho_r,u_l,u_r,p_l,p_r):

	u = np.empty((3,nx))
	u[0,nx/2:] = rho_r
	u[0,:nx/2] = rho_l
	u[1,nx/2:] = rho_l*u_l
	u[1,:nx/2] = rho_r*u_r
	u[2,nx/2:] = rho_l*(p_r/((gamma-1)*rho_l)) 
	u[2,:nx/2] = rho_r*(p_l/((gamma-1)*rho_r))

	return u

# defining initial condition values
rho_r = 0.125 # kg/m3
rho_l = 1 # kg/m3
u_l = 0 # m/s
u_r = 0 # m/s
p_l = 100000 # kN/m2
p_r = 10000 # kN/m2

# mesh info
nx = 81
dt = 0.0002
T = 0.01
nt = int(T/dt) 
dx = 0.25

# gamma value to model air
gamma = 1.4

u = u_initial(nx,gamma,rho_l,rho_r,u_l,u_r,p_l,p_r)

def computeF(u, gamma):
	""" computes flux vector """
	u1 = u[0,:]
	u2 = u[1,:]
	u3 = u[2,:]
	
	return np.array([u2, u2**2/u1+(gamma-1)*(u3-.5*(u2**2/u1)),\
			(u3+(gamma-1)*(u3-0.5*(u2**2/u1)))*(u2/u1)]) 

def richtmyer(u,nt,dt,dx,gamma):
	""" computes soln with richtmyer scheme """
	un = u.copy()
	uhalf = np.copy(u) 
	fhalf = np.empty_like(u)
	
	for i in range(1,nt):
		f = computeF(u,gamma)
		uhalf[:,:-1] = 0.5*(u[:,1:]+u[:,:-1])-\
						(dt/(2*dx))*(f[:,1:]-f[:,:-1])
		fhalf = computeF(uhalf,gamma)
		un[:,1:] = u[:,1:] - (dt/dx) * (fhalf[:,1:] - fhalf[:,:-1])
		
		u = un.copy()
	return un


# CFL
#sigma = 1
#dt = dx*sigma

u = richtmyer(u,nt,dt,dx,gamma)

x = np.linspace(-10,10,nx)

plt.figure()
plt.plot(x,u[1,:]/u[0,:])
plt.xlabel('x')
plt.ylabel('velocity')
plt.show()


plt.figure()
plt.plot(x,u[0,:])
plt.xlabel('x')
plt.ylabel('density')
plt.show()