# Module 3 Lesson 5 - Sod's Shock Tube
# Ian Carr - Oct 30 2014

import numpy as np 
import matplotlib.pyplot as plt

# defining initial conditions
def initial(nt,nx,dx):
	"""
	makes initial condition arrays
	order of initial condition arrays - rho, u, p
	in kg/m3, m/s, kN/m2 respectively
	"""

	rho_initial = np.zeros(nx)
	rho_initial[:nx/2] = 1 # initial density left
	rho_initial[nx/2:] = 0.125 # initial density right

	u_initial = np.zeros(nx)

	p_initial = np.zeros(nx)
	p_initial[:nx/2] = 100 # initial pressure left
	p_initial[nx/2:] = 10 # initial pressure right

	return [rho_initial, u_initial, p_initial]

nx = 81
nt = 70
dx = 20.0/nx-1

x = np.linspace(-10.,10,nx)

initial = initial(nt,nx,dx)

plt.plot(x,initial[2])
plt.show()

def computeF(u1, u2, u3, gamma):
	""" computes flux vector """
	return [u2, u2**2/u1+(gamma-1)*(u3-.5*(u2**2/u1)),\
			(u3+(gamma-1)*(u3-0.5*(u2**2/u1))*(u2/u1))]

def animate(data):
	x = np.linspace(-10.,10.,nx)
	y = data
	line.set_data(x,y)
	return line,

def richtmyer(u,nt,dt,dx,gamma):
	""" computes soln with richtmyer scheme """
	f = computeF(u)

