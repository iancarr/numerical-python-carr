# Module 2 Lesson 3 - 1D Diffusion
# Ian Carr Oct 5 2014

import numpy as np 
import matplotlib.pyplot as plt

# time and space parameters
nx = 41
dx = 2./(nx-1)
nt = 20
nu = 0.3  # kinematic viscosity
sigma = .2
dt = sigma*dx**2/nu

x = np.linspace(0,2,nx)

u = np.ones(nx)
u[.5/dx : 1/dx+1] = 2

# initializing dummy space array
un = np.ones(nx)

# solving central difference for diffusion eqn
for n in range(nt):
	un = u.copy()
	u[1:-1] = un[1:-1] + nu*dt/dx**2*(un[2:]-2*un[1:-1]\
						+un[0:-2])

# plotting
plt.plot(np.linspace(0,2,nx),u)
plt.ylim(0,2.5)

# --------- Animations --------
plt.ion()

# initial conditions for animation
nt = 50
u = np.ones(nx)
u[.5/dx : 1/dx+1] = 2

un = np.ones(nx)

# defining func to solve for plots at each time
def diffusion(u):
	un=u.copy()
	u[1:-1] = un[1:-1] + nu*dt/dx**2*(un[2:]-2*un[1:-1]+\
				un[0:-2])
	return u
# looping to plot animation
for n in range(nt):
	u = diffusion(u)
	plt.plot(x,u)
	plt.pause(0.01)
	plt.clf()
	plt.xlim([0,2])
	plt.ylim([0,2])




plt.show()